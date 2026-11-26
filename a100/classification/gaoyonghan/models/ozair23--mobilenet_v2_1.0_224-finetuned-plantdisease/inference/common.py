"""Model-local paths, public options and native helpers; no framework imported here."""
import argparse
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import re
import sys
from model_paths import resolve as configured_path

MODEL_ROOT=Path(__file__).resolve().parents[1]
PROJECT_ROOT=MODEL_ROOT.parents[3]

def resolve_path(name,default):
    value=Path(os.environ.get(name,str(default))).expanduser()
    return (value if value.is_absolute() else MODEL_ROOT/value).resolve()

WEIGHTS_ROOT = configured_path('MODEL_WEIGHTS_ROOT', 'weights_root')
DATASET_ROOT = configured_path('MODEL_DATASET_ROOT', 'dataset_root')
OUTPUT_ROOT=resolve_path('MODEL_OUTPUT_ROOT',MODEL_ROOT/'result/run_001')

def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))

def save(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_name(path.name+'.tmp')
    temporary.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    temporary.replace(path)

def model_spec():
    value=read(MODEL_ROOT/'inference/models.json')
    if len(value['models'])!=1:raise ValueError('A model folder must describe exactly one model')
    return value['models'][0]

def parser(native=False):
    p=argparse.ArgumentParser(description='Run this model and publish the standard classification bundles.',allow_abbrev=False)
    p.add_argument('--phase',choices=['full','smoke','preflight'],default='full')
    p.add_argument('--output-root', '--output-dir', type=Path)
    p.add_argument('--device',help='auto, cpu, cuda or cuda:N; N indexes inherited visible GPUs')
    p.add_argument('--batch-size',type=int)
    p.add_argument('--workers', '--num-workers', type=int)
    p.add_argument('--warmup', '--warmup-steps', type=int,help='Native warmup iterations; original default when omitted')
    p.add_argument('--max-images', '--max-samples', type=int,default=0)
    p.add_argument('--show-command',action='store_true')
    if not native:p.add_argument('--prepare-only',action='store_true')
    return p

def runtime_options(args,p,spec):
    defaults=spec['runtime_defaults']
    if args.batch_size is not None and args.batch_size<1:p.error('--batch-size must be positive')
    if args.workers is not None and args.workers<0:p.error('--workers cannot be negative')
    if args.warmup is not None and args.warmup<0:p.error('--warmup cannot be negative')
    if args.max_images<0:p.error('--max-images cannot be negative')
    if (args.phase=='smoke' or args.max_images) and not spec.get('supports_max_samples'):
        p.error('This original workflow has no verified limited-sample option; use --phase full')
    if args.phase=='preflight' and args.max_images:p.error('--max-images does not apply to preflight')
    if args.phase=='preflight' and getattr(args,'prepare_only',False):p.error('--prepare-only requires --phase full or smoke')
    phase='preflight' if args.phase=='preflight' else ('smoke' if args.phase=='smoke' or args.max_images else 'full')
    limit=args.max_images or (10 if phase=='smoke' else 0)
    workers=args.workers if args.workers is not None else int(os.environ.get('MODEL_NUM_WORKERS',defaults.get('workers',0)))
    if not spec['capabilities']['worker_processes'] and workers:
        p.error('This native iterator has no worker pool; use --workers 0')
    device=args.device or os.environ.get('MODEL_REQUESTED_DEVICE','auto')
    if not re.fullmatch(r'auto|cpu|cuda(?::\d+)?',device):p.error('--device must be auto, cpu, cuda or cuda:N')
    if device=='cpu' and spec['capabilities']['gpu_only']:p.error('This verified native workflow requires a CUDA GPU')
    root=args.output_root or Path(os.environ.get('MODEL_OUTPUT_ROOT','result/run_001'))
    root=(root if root.is_absolute() else MODEL_ROOT/root).resolve()
    if not root.is_relative_to(MODEL_ROOT/'result'):p.error('--output-root must stay inside this model result/ directory')
    return dict(phase=phase,max_images=limit,output_root=str(root),device=device,
      batch_size=args.batch_size if args.batch_size is not None else int(os.environ.get('MODEL_BATCH_SIZE',os.environ.get('BATCH_SIZE',defaults['batch_size']))),
      workers=workers,warmup=args.warmup if args.warmup is not None else int(os.environ.get('MODEL_WARMUP_STEPS',defaults['warmup'])))

def execution_environment(options,base=None):
    env=dict(os.environ if base is None else base);root=Path(options['output_root']);device=options['device']
    search_paths = [str(MODEL_ROOT / 'inference')]
    if env.get('PYTHONPATH'):
        search_paths.append(env['PYTHONPATH'])
    env['PYTHONPATH'] = os.pathsep.join(search_paths)
    env.update(MODEL_OUTPUT_ROOT=str(root),MODEL_PHASE=options['phase'],MAX_SAMPLES=str(options['max_images']),
      MODEL_BATCH_SIZE=str(options['batch_size']),MODEL_NUM_WORKERS=str(options['workers']),MODEL_WARMUP_STEPS=str(options['warmup']),
      MODEL_REQUESTED_DEVICE=device,HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',HF_DATASETS_OFFLINE='1',
      HF_HOME=str(root/'.runtime/huggingface'),HF_HUB_CACHE=str(root/'.runtime/huggingface/hub'),
      TRANSFORMERS_CACHE=str(root/'.runtime/huggingface'),HF_MODULES_CACHE=str(root/'.runtime/hf_modules'),
      HF_DATASETS_CACHE=str(root/'.runtime/datasets'),MPLCONFIGDIR=str(root/'.runtime/matplotlib'),PYTHONDONTWRITEBYTECODE='1')
    if device.startswith('cuda'):
        if env.get('MODEL_DEVICE_MAPPING_COMPLETE')!=device:
            index=int(device.split(':')[1]) if ':' in device else 0
            visible=env.get('CUDA_VISIBLE_DEVICES')
            if visible is None:selected=str(index)
            else:
                entries=[s.strip() for s in visible.split(',') if s.strip() and s.strip()!='-1']
                if index>=len(entries):raise ValueError('Requested GPU index is outside inherited CUDA_VISIBLE_DEVICES')
                selected=entries[index]
            env['CUDA_VISIBLE_DEVICES']=selected;env['MODEL_DEVICE_MAPPING_COMPLETE']=device
        env['MODEL_DEVICE']='cuda';env['MS_DEVICE_TARGET']='GPU'
    elif device=='cpu':
        env.update(CUDA_VISIBLE_DEVICES='',MODEL_DEVICE='cpu',MS_DEVICE_TARGET='CPU',MODEL_DEVICE_MAPPING_COMPLETE=device)
    else:
        env['MODEL_DEVICE']='auto'
    return env

def native_command(options):
    config=read(MODEL_ROOT/'inference/entrypoint.json')
    mapping={'__PYTHON__':sys.executable,'__MODEL_ROOT__':str(MODEL_ROOT),'__OUTPUT_ROOT__':options['output_root'],
             '__DATASET_ROOT__':str(DATASET_ROOT),'__PHASE__':options['phase']}
    command=[]
    for value in config['commands'][0]['arguments']:
        for key,replacement in mapping.items():value=value.replace(key,replacement)
        command.append(value)
    command+=['--device',options['device'],'--batch-size',str(options['batch_size']),'--workers',str(options['workers']),'--warmup',str(options['warmup'])]
    if options['max_images']:command+=['--max-images',str(options['max_images'])]
    return command

def command_details(options):
    env=execution_environment(options)
    return dict(commands=[native_command(options)],output_root=options['output_root'],phase=options['phase'],
      dataset_root=str(DATASET_ROOT),weights=str(WEIGHTS_ROOT),options=options,
      capabilities=model_spec()['capabilities'],cuda_visible_devices=env.get('CUDA_VISIBLE_DEVICES'),inference_executed=False)

def configure_native():
    global OUTPUT_ROOT
    p=parser(native=True);args=p.parse_args();options=runtime_options(args,p,model_spec())
    if options['phase']=='preflight':
        print(json.dumps(preflight(options),ensure_ascii=False,indent=2));raise SystemExit(0)
    if args.show_command:
        print(json.dumps(command_details(options),ensure_ascii=False,indent=2));raise SystemExit(0)
    os.environ.update(execution_environment(options))
    OUTPUT_ROOT=Path(options['output_root'])
    return options

def select_torch_device(torch,default):
    requested=os.environ.get('MODEL_DEVICE','auto')
    if requested=='auto':return default
    if requested=='cuda':
        if not torch.cuda.is_available():raise RuntimeError('Explicit CUDA request cannot fall back to CPU')
        torch.cuda.set_device(0)
    return requested

def tensorflow_gpus(tf):
    requested=os.environ.get('MODEL_DEVICE','auto')
    if requested=='cpu':tf.config.set_visible_devices([],'GPU');return []
    devices=tf.config.list_physical_devices('GPU')
    if requested=='cuda' and not devices:raise RuntimeError('Explicit CUDA request cannot fall back to CPU')
    return devices

def onnx_providers(ort,defaults):
    requested=os.environ.get('MODEL_DEVICE','auto');available=ort.get_available_providers()
    if requested=='cuda' and 'CUDAExecutionProvider' not in available:raise RuntimeError('ONNX CUDAExecutionProvider is unavailable')
    preferred=['CPUExecutionProvider'] if requested=='cpu' else defaults
    return [p for p in preferred if p in available] or available[:1]

def verify_onnx_session(session):
    actual=session.get_providers()
    if os.environ.get('MODEL_DEVICE')=='cuda' and 'CUDAExecutionProvider' not in actual:raise RuntimeError('ONNX session fell back to CPU despite explicit CUDA request')

def preflight(options):
    spec=model_spec();required=spec['required_inputs']
    missing=[str(WEIGHTS_ROOT/name) for name in required['weights'] if not (WEIGHTS_ROOT/name).is_file()]
    missing.extend(str(DATASET_ROOT/name) for name in required['dataset_paths'] if not (DATASET_ROOT/name).exists())
    if missing:raise FileNotFoundError('Required local inputs are missing: '+', '.join(missing))
    return dict(command_details(options),state='preflight_complete',weight_files_checked=len(required['weights']),
      dataset_paths_checked=required['dataset_paths'],checks='Local input presence and CLI/path resolution only',
      framework_initialization_executed=False,model_inference_executed=False)


"""Shared classification metrics (not COCO object-detection AP).

Scores are unrounded fractions in [0, 1]; callers retain presentation rounding.
Definitions follow sklearn.metrics, with None for undefined binary AUC/AP.
COCO bbox/segm evaluation requires actual instance predictions and COCOeval.
"""



def _binary_inputs(targets, scores):
    import numpy as np
    targets = np.asarray(targets)
    scores = np.asarray(scores, dtype=np.float64)
    if targets.ndim != 1 or scores.ndim != 1 or len(targets) != len(scores):
        raise ValueError("targets and scores must be aligned one-dimensional arrays")
    if not np.isin(targets, [0, 1]).all():
        raise ValueError("binary targets must be 0 or 1")
    if not np.isfinite(scores).all():
        raise ValueError("scores must be finite")
    return targets.astype(bool), scores


def binary_roc_auc(targets, scores):
    """ROC-AUC, target 1 positive; tied positive/negative pairs count 1/2.

    Return None if either class is absent, including an empty input.
    """
    import numpy as np
    targets, scores = _binary_inputs(targets, scores)
    positives = int(targets.sum())
    negatives = len(targets) - positives
    if not positives or not negatives:
        return None
    order = np.argsort(scores, kind="stable")
    sorted_scores, sorted_targets = scores[order], targets[order]
    ends = np.r_[np.flatnonzero(np.diff(sorted_scores)) + 1, len(scores)]
    starts = np.r_[0, ends[:-1]]
    positive_prefix = np.r_[0, np.cumsum(sorted_targets, dtype=np.int64)]
    positive_counts = positive_prefix[ends] - positive_prefix[starts]
    average_ranks = (starts + 1.0 + ends) / 2.0
    rank_sum = float(np.dot(positive_counts, average_ranks))
    return (rank_sum - positives * (positives + 1) / 2.0) / (positives * negatives)


def classification_average_precision(targets, scores):
    """Non-interpolated classification AP, grouping equal-score thresholds.

    AP = sum(delta_recall * precision). This is NOT COCO's 101-point
    interpolated detection AP. Return None when no positive target exists;
    callers averaging present-label AP must exclude those undefined labels.
    """
    import numpy as np
    targets, scores = _binary_inputs(targets, scores)
    positives = int(targets.sum())
    if not positives:
        return None
    order = np.argsort(-scores, kind="stable")
    sorted_scores = scores[order]
    true_positives = np.cumsum(targets[order], dtype=np.int64)
    ends = np.r_[np.flatnonzero(np.diff(sorted_scores)), len(scores) - 1]
    tp_at_threshold = true_positives[ends]
    precision = tp_at_threshold / (ends + 1.0)
    delta_recall = np.diff(np.r_[0, tp_at_threshold]) / positives
    return float(np.dot(delta_recall, precision))


def balanced_accuracy_from_confusion(confusion):
    """Mean recall over classes present in ground truth; rows=true, cols=pred.

    Classes with zero true support are excluded, as in sklearn's unadjusted
    balanced_accuracy_score. Fixed-label macro recall is a separate metric.
    """
    import numpy as np
    if hasattr(confusion, "detach"):
        confusion = confusion.detach().cpu().tolist()
    confusion = np.asarray(confusion, dtype=np.float64)
    if confusion.ndim != 2 or confusion.shape[0] != confusion.shape[1]:
        raise ValueError("confusion must be a square matrix")
    if not np.isfinite(confusion).all() or (confusion < 0).any():
        raise ValueError("confusion counts must be finite and non-negative")
    support = confusion.sum(axis=1)
    present = support > 0
    if not present.any():
        raise ValueError("balanced accuracy requires at least one labeled sample")
    return float(np.mean(confusion.diagonal()[present] / support[present]))


def rounded_or_none(value, digits=6):
    """Keep undefined metrics JSON-safe instead of turning them into 0/NaN.
    """
    return None if value is None else round(float(value), digits)

"""Save native test records in the portable evaluator's directory layout.

Model scripts call this module after their existing metric updates. No models,
preprocessors, or extra forwards are created here.
"""

import atexit
from datetime import datetime, timezone
from model_paths import runtime_path

sys.path.insert(0, str(runtime_path('timing')))
from timing import native_timing
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

_MODEL_ROOT = Path(__file__).resolve().parents[1]
_WORKSPACE = _MODEL_ROOT.parents[3]
_contract_spec = importlib.util.spec_from_file_location(
    "_native_record_contract", _MODEL_ROOT / "inference/output_format/classification.py")
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
        return str(path.relative_to(root)) if path.is_relative_to(root) else str(path)
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
                return str(path.relative_to(root)) if path.is_relative_to(root) else str(row[0])
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
        for batch in self.loader:
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
            require(0 <= y < classes, 'GT outside actual output space')
            sid = record['sample_id']
            require(sid not in branch['seen'], 'Duplicate captured prediction ID: ' + sid)
            branch['seen'].add(sid)
            branch['rows'].append(dict(sample_id=sid, target=y, prediction=p, topk=top, scores=None))
            branch['samples'].append(dict(sample_id=sid, label=y, source_index=record['source_index']))

    def capture(self, local):
        context = {**self.namespace, **local}
        profile = self.spec['native_script']
        if self.spec['sequence'] == 149:
            for i, head in enumerate(context['TASK_ORDER']):
                y = context['batch_model_labels'][:, i]
                p = context['task_predictions'][head][-len(self.current):]
                self.append(head, y, p, None, context['logits'][head].shape[-1], context, gt_column=i)
        else:
            get = lambda expression: eval(expression, context)
            self.append('default', get(profile['labels']), get(profile['prediction']),
                        get(profile['ranking']) if profile['ranking'] else None, get(profile['classes']), context)
        for key in ('correct_top1', 'top1_correct'):
            if key in context:
                self.native_counts[key] = int(plain(context[key]))
        self.execution.update({key: str(context[key]) for key in
                               ('device', 'device_name', 'device_target', 'BACKEND') if key in context})
        framework = self.spec['native_script'].get('framework')
        module = sys.modules.get(framework)
        if module is not None:
            self.execution['framework'] = framework
            self.execution['framework_version'] = str(getattr(module, '__version__', 'unknown'))
        tensor = context.get('logits')
        if tensor is not None and hasattr(tensor, 'dtype'):
            self.execution['output_dtype'] = str(tensor.dtype)
        model = context.get('model')
        if framework == 'torch' and model is not None and hasattr(model, 'parameters'):
            parameter = next(model.parameters(), None)
            if parameter is not None:
                self.execution['parameter_dtype'] = str(parameter.dtype)
                self.execution['parameter_device'] = str(parameter.device)
        extra = context.get('extra')
        if isinstance(extra, dict):
            for key in ('providers', 'onnx_providers'):
                if key in extra:
                    self.execution['onnx_requested_providers'] = list(extra[key])
                    self.execution['onnx_providers'] = list(extra[key])
        if context.get('BACKEND') == 'onnx' and model is not None and hasattr(model, 'get_providers'):
            # Observe the created session: a requested CUDA provider may have
            # fallen back to CPU. This does not execute the model.
            self.execution['onnx_providers'] = list(model.get_providers())
            self.execution['tensor_postprocessing_framework'] = self.execution.get('framework')
            self.execution['tensor_postprocessing_framework_version'] = self.execution.get('framework_version')
            ort = sys.modules.get('onnxruntime')
            self.execution['framework'] = 'onnxruntime'
            self.execution['framework_version'] = str(getattr(ort, '__version__', 'unknown'))
            self.execution['device_note'] = 'device describes output tensor postprocessing; model inference uses the actual ONNX Runtime session providers.'
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
        self.execution.update(python=sys.executable, python_version=sys.version.split()[0],
                              cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'))
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
            excluded_images=len(self.excluded), protocol_fingerprint=self.fingerprint,
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
        super().capture(local)
        rows = []
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
        self._jsonl(self.output / 'batches.jsonl', [dict(batch_index=self.batch_count, records=rows,
                     completed_images=len(self.completed), updated_at=stamp())])
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
        if self.phase == 'full':
            require(len(self.ground_truth) == self.spec['expected_images'],
                    'Native valid sample count differs from the registered full evaluation scope')
        required = self.spec['native_script'].get('branches', ['default'])
        require(set(self.branches) == set(required), 'Some native classification branches are missing')
        summary = {}
        for name, branch in self.branches.items():
            folder = self.output if name == 'default' else self.output / name
            folder.mkdir(exist_ok=True)
            expected = set(self.ground_truth)
            require(branch['seen'] == expected, name + ': missing or unexpected native predictions')
            valid_samples = [r for r in branch['samples'] if r['label'] >= 0]
            valid_ids = {r['sample_id'] for r in valid_samples}
            predictions = [{k: r[k] for k in ('sample_id', 'prediction', 'topk', 'scores')}
                           for r in branch['rows'] if r['sample_id'] in valid_ids]
            annotation = dict(categories=self._categories(branch['categories'], name, context),
                              samples=valid_samples)
            write(folder / 'classification_annotations.json', annotation)
            excluded = [r['sample_id'] for r in branch['rows'] if r['target'] < 0]
            capture = dict(state='complete', phase=self.phase, expected_images=len(expected),
                           completed_images=len(branch['rows']), capture_stage='native_metric_input',
                           branch=name, extra_forward_calls=0, unmapped_gt_samples=excluded)
            write(folder / 'classification_capture.json', capture)
            if self.phase in ('full', 'smoke'):
                metadata = self._metadata(name)
                metadata['selection'] = dict(phase=self.phase, selected_images=self.dataset_size,
                    scored_images=len(valid_samples), excluded_samples=excluded,
                    excluded_decode_samples=list(self.excluded), launcher_completed=True)
                metadata['source'] = dict(path_base='bundle_parent',
                    predictions=dict(path='predictions.jsonl', sha256=file_hash(folder / 'predictions.jsonl')),
                    ground_truth=dict(kind='native_dataset_loader', path='classification_annotations.json',
                                      sha256=file_hash(folder / 'classification_annotations.json')))
                write(folder / 'classification_capture_metadata.json', metadata)
            summary[name] = capture
        if len(summary) > 1:
            write(self.output / 'classification_capture.json', dict(
                state='complete', phase=self.phase, branches=summary, extra_forward_calls=0))
            write(self.output / 'classification_annotations.json', dict(
                branches={name: name + '/classification_annotations.json' for name in summary}))

    

    def finish(self, local):
        context = {**self.namespace, **local}
        require((self.output / 'metrics.json').is_file(), 'Native metrics file was not saved')
        native_metrics = json.loads((self.output / 'metrics.json').read_text())
        if isinstance(native_metrics.get('environment'), dict):
            self.execution['native_environment'] = native_metrics['environment']
        self._finish_classification(context)
        timing = native_timing(native_metrics, self.batch_count, time.monotonic() - self.started_clock)
        write(self.output / 'timing.json', timing)
        self._status('complete')
        self.finished = True
        print(f'测试记录保存完成: {self.output}', flush=True)


class _ClassificationRecorder(_Recording, ScriptCapture):
    pass





def start(sequence, script, namespace):
    spec = model_spec()
    require(sequence == spec['sequence'], 'Unexpected model sequence')
    script = Path(script).resolve()
    require(script == _MODEL_ROOT / spec['native_script']['script'], 'Unexpected model script path')
    requested = int(os.environ.get('MAX_SAMPLES', '0')) if spec.get('supports_max_samples') else 0
    phase = 'smoke' if requested > 0 else 'full'
    base = Path(os.environ.get('MODEL_OUTPUT_ROOT', str(_MODEL_ROOT / 'result/run_001')))
    if not base.is_absolute(): base = _MODEL_ROOT / base
    output = base.resolve() / phase / f'm{sequence:03d}'
    recorder = _ClassificationRecorder(spec, output, _WORKSPACE)
    recorder.namespace = namespace
    recorder._begin(script, phase)
    return recorder
