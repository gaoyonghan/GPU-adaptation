"""Save native test records in the portable evaluator's directory layout.

Model scripts call this module after their existing metric updates. No models,
preprocessors, or extra forwards are created here.
"""
from __future__ import annotations

import atexit
from datetime import datetime, timezone
import fcntl
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import shutil
import signal
import sys
import time
import traceback

from model_paths import PROJECT_ROOT, MODEL_ROOT, configuration, native_arguments
from timing import NativeTimer
_WORKSPACE = PROJECT_ROOT
_contract_spec = importlib.util.spec_from_file_location(
    "_native_record_contract", Path(__file__).resolve().parents[1] / "output_format/classification.py")
_contract = importlib.util.module_from_spec(_contract_spec)
_contract_spec.loader.exec_module(_contract)
file_hash, require, write = _contract.file_hash, _contract.require, _contract.write


def stamp():
    return datetime.now(timezone.utc).isoformat()


def plain(value):
    if hasattr(value, 'detach'):
        value = value.detach().cpu()
    elif hasattr(value, 'asnumpy'):
        value = value.asnumpy()
    return value.tolist() if hasattr(value, 'tolist') else value


def sample_identity(dataset, index, spec, root):
    """Use native filenames/members or the source table's actual row index."""
    if spec['sequence'] == 152:
        image, angle = divmod(index, 4)
        return str(Path(dataset.image_paths[image]).relative_to(root)) + f'#rotation={angle * 90}'
    if hasattr(dataset, 'image_paths'):
        path = Path(dataset.image_paths[index])
        return stable_identity(path, root)
    for name in ('samples', 'records'):
        rows = getattr(dataset, name, None)
        if rows is not None:
            row = rows[index]
            if isinstance(row, dict):
                for key in ('sample_id', 'image_name', 'path', 'id'):
                    if key in row:
                        return str(row[key])
            if isinstance(row, (list, tuple)) and isinstance(row[0], (str, Path)):
                path = Path(row[0])
                return stable_identity(path, root)
    source = getattr(dataset, 'dataset', dataset)
    indices = getattr(source, '_indices', None)
    source_index = int(indices.column(0)[index].as_py()) if indices is not None else int(index)
    for column in ('image_id', 'image_name', 'filename', 'file_name', 'id', 'path'):
        if column in getattr(source, 'column_names', []):
            return str(source[column][index])
    return f"{spec['dataset']['id']}:{spec['dataset'].get('split') or 'native-selection'}:source-row:{source_index}"


class TaggedDataset:
    def __init__(self, dataset, spec, root):
        self.dataset, self.spec, self.root = dataset, spec, root

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        item = self.dataset[index]
        identity = sample_identity(self.dataset, index, self.spec, self.root)
        valid = item is not None and item[0] is not None
        label = None
        if valid:
            label = plain(item[self.spec['native_script']['label_position']])
            if self.spec['sequence'] == 101:
                label = [plain(item[1]), plain(item[2])]
        return item, dict(sample_id=identity, native_label=label, valid=valid,
                          source_index=int(index))


class TaggedCollate:
    def __init__(self, collate):
        self.collate = collate

    def __call__(self, items):
        values, records = zip(*items)
        return dict(native=self.collate(list(values)), records=list(records))


class Loader:
    def __init__(self, loader, capture):
        self.loader, self.capture = loader, capture

    def __len__(self):
        return len(self.loader)

    def __getattr__(self, name):
        return getattr(self.loader, name)

    def __iter__(self):
        iterator = iter(self.loader)
        while True:
            measuring = self.capture.timer.warmup_seconds is not None and not self.capture.timer.warming
            if measuring:
                self.capture.timer.begin_batch()
            try:
                batch = next(iterator)
            except StopIteration:
                self.capture.timer.pending = None
                self.capture.timer.e2e_pending = None
                return
            self.capture.set_batch(batch['records'])
            yield batch['native']


class ScriptCapture:
    def __init__(self, spec, output, root, max_samples=None, overrides=None):
        self.spec, self.output, self.root = spec, Path(output), Path(root)
        self.max_samples = max_samples
        self.overrides = overrides or {}
        self.namespace = {}
        self.current = []
        self.ground_truth = {}
        self.excluded = {}
        self.branches = {}
        self.native_counts = {}
        self.execution = {}
        self.dataset_size = None
        self.label_mapping = {}

    def loader(self, constructor, *args, **kwargs):
        from torch.utils.data._utils.collate import default_collate
        require(not kwargs.get('shuffle', False), 'Native script export requires the native deterministic sample order')
        require(not kwargs.get('drop_last', False), 'Dropped final batches are not a full dataset evaluation')
        require(kwargs.get('sampler') is None and kwargs.get('batch_sampler') is None,
                'Custom sampling needs an explicit identity adapter')
        args = list(args)
        dataset = args[0] if args else kwargs['dataset']
        self.dataset_size = len(dataset)
        wrapped = TaggedDataset(dataset, self.spec, self.root)
        if args:
            args[0] = wrapped
        else:
            kwargs['dataset'] = wrapped
        kwargs['collate_fn'] = TaggedCollate(kwargs.get('collate_fn') or default_collate)
        return Loader(constructor(*args, **kwargs), self)

    def set_batch(self, records):
        self.current = [r for r in records if r['valid']]
        for record in records:
            target = self.ground_truth if record['valid'] else self.excluded
            old = target.get(record['sample_id'])
            require(old is None or old == record, 'Dataset identity changed between native iterations')
            target[record['sample_id']] = record

    def generator_batch(self, context):
        kind = self.spec['native_script']['loader_kind']
        if kind == 'spider':
            # The first iterator batch is reused after warmup; use loop position,
            # not generator prefetch position, to identify the inference batch.
            labels = plain(context['labels'])
            start = (context['batch_index'] - 1) * context['BATCH_SIZE']
            selected = context['records'][start:start + len(labels)]
            records = [dict(sample_id=str(r['image_name']), native_label=context['class_to_id'][r['class']],
                            valid=True, source_index=start + i) for i, r in enumerate(selected)]
        elif kind == 'lenet':
            labels = plain(context['labels'])
            start = (context['batch_index'] - 1) * context['BATCH_SIZE']
            records = [dict(sample_id=sample_identity(context['wrapped_dataset'], start + i, self.spec, self.root),
                            native_label=int(y), valid=True, source_index=start + i) for i, y in enumerate(labels)]
        elif kind == 'pointnet':
            start = context['start']
            labels = plain(context['labels'][start:start + context['BATCH_SIZE']])
            records = [dict(sample_id=str(context['meshes'][start + i][0]), native_label=int(y),
                            valid=True, source_index=start + i) for i, y in enumerate(labels)]
        else:
            return
        self.set_batch(records)
        self.dataset_size = len(context[{'spider':'records', 'lenet':'wrapped_dataset', 'pointnet':'meshes'}[kind]])

    def before_batch(self, local):
        self.generator_batch({**self.namespace, **local})
        if self.timer.pending is None:
            self.timer.begin_batch()

    def prediction_ready(self, local):
        context = {**self.namespace, **local}
        if self.spec['sequence'] == 149:
            for head in context['TASK_ORDER']:
                plain(context['task_predictions'][head][-len(self.current):])
        else:
            profile = self.spec['native_script']
            plain(eval(profile['prediction'], context))
            if profile['ranking']:
                plain(eval(profile['ranking'], context))
        self.timer.prediction_ready()

    def append(self, name, targets, predictions, ranking, classes, context, *, gt_column=None, mask=None):
        targets, predictions = plain(targets), plain(predictions)
        ranking = plain(ranking) if ranking is not None else [[int(p)] for p in predictions]
        require(len(self.current) == len(targets) == len(predictions) == len(ranking),
                f'{name}: batch IDs/GT/predictions have different lengths')
        classes = int(classes)
        branch = self.branches.setdefault(name, dict(rows=[], samples=[], categories=classes, seen=set(),
                                                    native_images=0, native_correct=0))
        require(branch['categories'] == classes, 'Output class space changed during inference')
        mask = plain(mask) if mask is not None else [True] * len(targets)
        require(len(mask) == len(targets), 'Native branch mask length differs from batch')
        branch['native_images'] += sum(bool(keep) for keep in mask)
        branch['native_correct'] += sum(int(y) == int(p) for y,p,keep in zip(targets,predictions,mask) if keep)
        for record, y, p, top, keep in zip(self.current, targets, predictions, ranking, mask):
            if not keep:
                continue
            if self.max_samples and len(branch['rows']) >= self.max_samples:
                break
            y, p, top = int(y), int(p), [int(x) for x in top]
            source_label = record['native_label']
            if gt_column is not None:
                source_label = source_label[gt_column]
            if isinstance(source_label, str):
                mapping = context.get('label_to_id')
                require(mapping is not None and source_label in mapping, 'Native string label mapping missing')
                self.label_mapping = dict(mapping)
                source_label = mapping[source_label]
            require(int(source_label) == y, f'{name}: metric target differs from native loader GT')
            require(bool(top), f'{name}: empty native ranking')
            require(all(0 <= c < classes for c in [p, *top]), 'Prediction outside actual output space')
            require(0 <= y < classes or (self.spec['sequence'] == 101 and y == -1), 'GT outside actual output space')
            sid = record['sample_id']
            require(sid not in branch['seen'], 'Duplicate captured prediction ID: ' + sid)
            branch['seen'].add(sid)
            branch['rows'].append(dict(sample_id=sid, target=y, prediction=p, topk=top, scores=None))
            branch['samples'].append(dict(sample_id=sid, label=y, source_index=record['source_index']))

    def capture(self, local):
        context = {**self.namespace, **local}
        profile = self.spec['native_script']
        if self.spec['sequence'] == 101:
            self.append('full_21841_classes', context['labels_22k'], context['full_top5_indices'][:, 0],
                        context['full_top5_indices'], context['logits'].shape[-1], context, gt_column=0)
            self.append('restricted_999_mapped_classes', context['labels_restricted'],
                        context['restricted_top5_indices'][:, 0], context['restricted_top5_indices'],
                        context['restricted_logits'].shape[-1], context, gt_column=1, mask=context['valid_mask'])
        elif self.spec['sequence'] == 149:
            for i, head in enumerate(context['TASK_ORDER']):
                y = context['batch_model_labels'][:, i]
                p = context['task_predictions'][head][-len(self.current):]
                self.append(head, y, p, None, context['logits'][head].shape[-1], context, gt_column=i)
        else:
            get = lambda expression: eval(expression, context)
            self.append('default', get(profile['labels']), get(profile['prediction']),
                        get(profile['ranking']) if profile['ranking'] else None, get(profile['classes']), context)
        for key in ('correct_top1', 'top1_correct', 'correct_full_top1', 'correct_restricted_top1'):
            if key in context:
                self.native_counts[key] = int(plain(context[key]))
        self.execution.update({key: str(context[key]) for key in ('device', 'device_name', 'BACKEND') if key in context})
        extra = context.get('extra')
        if isinstance(extra, dict):
            for key in ('providers', 'onnx_providers'):
                if key in extra:
                    self.execution['onnx_providers'] = list(extra[key])
        if self.max_samples and self.branches and all(len(b['rows']) >= self.max_samples for b in self.branches.values()):
            raise SelectionComplete()


class _Recording:
    """Persist records from the original script; this module is not a runner."""

    def _begin(self, script, phase):
        self.script = Path(script).resolve()
        self.phase = phase
        self.started_at = stamp()
        self.started_clock = time.monotonic()
        self.owner_pid = os.getpid()
        self.finished = False
        self.batch_count = 0
        self.completed = set()
        self.output.parent.mkdir(parents=True, exist_ok=True)
        self.lock = (self.output.parent / ('.' + self.output.name + '.lock')).open('a')
        fcntl.flock(self.lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        if self.output.exists() and any(self.output.iterdir()):
            history = self.output.parent / '_history'
            history.mkdir(exist_ok=True)
            previous = history / (self.output.name + '-' + str(time.time_ns()))
            self.output.rename(previous)
        self.output.mkdir(exist_ok=True)
        self.fingerprint = hashlib.sha256(json.dumps({
            'script_sha256': file_hash(self.script),
            'recorder_sha256': file_hash(__file__),
            'model_manifest_sha256': file_hash(Path(__file__).with_name('models.json')),
            'runtime_parameters': {k: os.environ.get(k) for k in ('MX_DEVICE','MODEL_BATCH_SIZE','MODEL_WORKERS','MODEL_WARMUP','MODEL_MAX_IMAGES')},
            'sequence': self.spec['sequence'], 'phase': phase,
        }, sort_keys=True).encode()).hexdigest()
        self._status('running')
        previous_hook = sys.excepthook

        def exception_hook(kind, value, tb):
            self._fail(''.join(traceback.format_exception_only(kind, value)).strip())
            previous_hook(kind, value, tb)

        sys.excepthook = exception_hook
        atexit.register(self._shutdown)
        for signum in (signal.SIGTERM, signal.SIGINT):
            signal.signal(signum, self._signal)
        print(f'测试记录目录: {self.output}', flush=True)

    def _status(self, state, reason=None):
        write(self.output / 'status.json', dict(
            sequence=self.spec['sequence'], model_id=self.spec['model_id'],
            phase=self.phase, state=state, status=state,
            full_evaluation=state == 'complete' and self.phase == 'full',
            started_at=self.started_at, updated_at=stamp(),
            elapsed_seconds=time.monotonic() - self.started_clock,
            total_images=self.dataset_size, completed_images=len(self.completed),
            excluded_images=len(self.excluded), excluded_sample_ids=list(self.excluded), protocol_fingerprint=self.fingerprint,
            source_script=str(self.script.relative_to(self.root)),
            native_execution=self.execution, extra_forward_calls=0, reason=reason,
            timing_note='Record writes add loop/wall time outside native forward timing.'))

    def _fail(self, reason):
        if os.getpid() == self.owner_pid and not self.finished:
            self._status('failed', reason)
            self.finished = True

    def _shutdown(self):
        self._fail('Script exited before record export completed')

    def _signal(self, signum, frame):
        self._fail(f'Script interrupted by signal {signum}')
        raise SystemExit(128 + signum)

    @staticmethod
    def _jsonl(path, rows):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('a', encoding='utf-8') as stream:
            for row in rows:
                stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')
            stream.flush()

    @staticmethod
    def _native_row(row):
        return dict(id=row['sample_id'], target=row['target'], prediction=row['prediction'],
                    top5=row['topk'], scores=row['scores'])

    def capture(self, local):
        before = {name: len(branch['rows']) for name, branch in self.branches.items()}
        previous_rows = len(self.rows) if hasattr(self, 'rows') else 0
        previous_calibration = len(self.calibration_rows) if hasattr(self, 'calibration_rows') else 0
        super().capture(local)
        self.timer.end_batch()
        rows = []
        if False:
            rows = self.rows[previous_rows:]
            calibration = self.calibration_rows[previous_calibration:]
            if calibration:
                self._jsonl(self.output / 'calibration_predictions.jsonl', calibration)
        else:
            multiple = bool(self.spec['native_script'].get('branches'))
            for name, branch in self.branches.items():
                new = [self._native_row(r) for r in branch['rows'][before.get(name, 0):]]
                if multiple:
                    self._jsonl(self.output / name / 'predictions.jsonl', new)
                    rows.extend(dict(r, branch=name) for r in new)
                else:
                    rows.extend(new)
        if rows:
            self._jsonl(self.output / 'predictions.jsonl', rows)
            self.completed.update(str(r.get('sample_id', r.get('id'))) for r in rows)
        self.batch_count += 1
        batch = dict(batch_index=self.batch_count, records=rows,
                     completed_images=len(self.completed), updated_at=stamp())
        if False and calibration:
            batch['calibration_records'] = calibration
        self._jsonl(self.output / 'batches.jsonl', [batch])
        self._status('running')

    def _metadata(self, branch='default'):
        return dict(model=dict(sequence=self.spec['sequence'], model_id=self.spec['model_id']),
                    dataset=self.spec['dataset'], qualification=self.spec['qualification'],
                    limitations=self.spec.get('limitations', []), phase=self.phase,
                    source_script=str(self.script.relative_to(self.root)),
                    script_sha256=file_hash(self.script), protocol_fingerprint=self.fingerprint,
                    native_execution=self.execution, branch=branch, extra_forward_calls=0)

    def _categories(self, count, name, context):
        names = {}
        if self.spec['sequence'] == 149:
            names = dict(enumerate(context['MODEL_LABELS'][name]))
        elif self.label_mapping:
            names = {int(value): key for key, value in self.label_mapping.items()}
        else:
            for key in ('class_names', 'CLASS_NAMES', 'label_names'):
                value = context.get(key)
                if isinstance(value, (list, tuple)) and len(value) == count:
                    names = dict(enumerate(value))
                    break
        return [dict(id=i, name=str(names.get(i, f'output_class_{i}'))) for i in range(count)]

    def _finish_classification(self, context):
        require(self.branches, 'No native classification predictions were captured')
        require(self.dataset_size is not None, 'Native dataset size was not recorded')
        require(len(self.ground_truth) + len(self.excluded) == self.dataset_size,
                'Native dataset traversal is incomplete')
        required = self.spec['native_script'].get('branches', ['default'])
        require(set(self.branches) == set(required), 'Some native classification branches are missing')
        summary = {}
        for name, branch in self.branches.items():
            folder = self.output if name == 'default' else self.output / name
            folder.mkdir(exist_ok=True)
            expected = set(self.ground_truth)
            if self.spec['sequence'] == 101 and name == 'restricted_999_mapped_classes':
                expected = {sid for sid, record in self.ground_truth.items() if record['native_label'][1] >= 0}
            require(branch['seen'] == expected, name + ': missing or unexpected native predictions')
            valid_samples = [r for r in branch['samples'] if r['label'] >= 0]
            valid_ids = {r['sample_id'] for r in valid_samples}
            predictions = [{k: r[k] for k in ('sample_id', 'prediction', 'topk', 'scores')}
                           for r in branch['rows'] if r['sample_id'] in valid_ids]
            annotation = dict(categories=self._categories(branch['categories'], name, context),
                              samples=valid_samples)
            registered = self.spec['label_mapping'] if name == 'default' else self.spec['label_mapping'][name]
            require(annotation['categories'] == registered, 'Native category mapping differs from frozen e07 mapping')
            write(folder / 'classification_annotations.json', annotation)
            excluded = [r['sample_id'] for r in branch['rows'] if r['target'] < 0]
            capture = dict(state='complete', phase=self.phase, expected_images=len(expected),
                           completed_images=len(branch['rows']), capture_stage='native_metric_input',
                           branch=name, extra_forward_calls=0, unmapped_gt_samples=excluded)
            write(folder / 'classification_capture.json', capture)
            summary[name] = capture
        if len(summary) > 1:
            write(self.output / 'classification_capture.json', dict(
                state='complete', phase=self.phase, branches=summary, extra_forward_calls=0))
            write(self.output / 'classification_annotations.json', dict(
                branches={name: name + '/classification_annotations.json' for name in summary}))

    def finish(self, local):
        context = {**self.namespace, **local}
        require((self.output / 'metrics.json').is_file(), 'Native metrics file was not saved')
        if False:
            self._finish_task(context)
        else:
            self._finish_classification(context)
        measured = self.timer.summary(self.phase, time.monotonic() - self.started_clock)
        require(measured['timed_images'] == len(self.completed), 'Timed native forward samples differ from completed prediction IDs')
        require(measured['timed_e2e_batches'] == sum(measured['actual_batch_sizes'].values()), 'Prediction-ready timer does not cover every measured batch')
        require(measured['warmup_batches'] == int(os.environ.get('MODEL_WARMUP', self.spec['runtime']['warmup'])), 'Observed native warmup calls differ from requested count')
        write(self.output / 'timing.json', measured)
        self._status('complete')
        self.finished = True
        print(f'测试记录保存完成: {self.output}', flush=True)


class _ClassificationRecorder(_Recording, ScriptCapture):
    pass


def stable_identity(path, root):
    path = Path(path)
    if path.is_absolute():
        require(path.is_relative_to(root), 'Dataset ID points outside delivery project')
        return str(path.relative_to(root))
    return str(path)


def start(sequence, script, namespace):
    """Initialize this model only, before native libraries/model construction."""
    native_arguments()
    spec = configuration()
    require(sequence == spec['sequence'], 'Wrong model-local configuration')
    require(Path(script).resolve() == MODEL_ROOT / 'inference/native/run_model.py', 'Wrong native entry')
    phase = os.environ.get('MODEL_PHASE', 'full')
    require(phase in ('full','smoke'), 'Unsupported native phase')
    require(phase != 'smoke' or spec.get('supports_max_samples'), 'Native limited scope is unsupported for this model')
    base = Path(os.environ.get('MODEL_OUTPUT_ROOT', str(MODEL_ROOT / 'result/run_001'))).resolve()
    require(base.is_relative_to(MODEL_ROOT / 'result'), 'Output must belong to this model')
    output = base / phase / f'm{sequence:03d}'
    recorder = _ClassificationRecorder(spec, output, _WORKSPACE)
    recorder.namespace = namespace
    recorder._begin(script, phase)
    os.environ['HF_DATASETS_CACHE'] = str(output / '.dataset_cache')
    os.environ['HF_MODULES_CACHE'] = str(output / '.module_cache')
    os.environ['XDG_CACHE_HOME'] = str(output / '.cache')
    os.environ['TORCH_HOME'] = str(output / '.torch_cache')
    import torch
    requested = os.environ.get('MX_DEVICE', 'cuda:0')
    require('metax' in torch.__version__.lower(), 'This verified workflow requires MetaX torch, not another backend')
    require(torch.cuda.is_available(), 'MX GPU unavailable; CPU fallback forbidden')
    device = torch.device(requested)
    require(device.type == 'cuda' and device.index is not None and device.index < torch.cuda.device_count(), 'Invalid MX logical device')
    torch.cuda.set_device(device)
    name = torch.cuda.get_device_name(device)
    require('metax' in name.lower() or 'c500' in name.lower(), 'Unexpected actual accelerator: ' + name)
    recorder.execution = dict(requested_device=str(device), logical_device_index=device.index, visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'), device=str(device), device_name=name, backend='MetaX MACA', torch_version=torch.__version__)
    recorder.timer = NativeTimer(device)
    return recorder
