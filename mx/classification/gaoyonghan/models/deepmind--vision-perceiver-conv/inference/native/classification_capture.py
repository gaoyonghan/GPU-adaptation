"""Passive export at the native metric boundary; never loads or runs a model."""
from contextlib import contextmanager
import json
from pathlib import Path


def save(path, data):
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def snapshot_gt(output, rows, mapping=None):
    """Export selected loader labels before inference, without reloading images."""
    samples = []
    for row in rows:
        label = row.get('target', row.get('label'))
        if mapping is not None: label = mapping[label]
        sid = row.get('id', row.get('image_id', row.get('path')))
        if sid is None: raise ValueError('Dataset sample identity unavailable')
        samples.append(dict(sample_id=str(sid), label=label, source_label=row.get('source_label')))
    save(Path(output) / 'classification_annotations.json', {'samples': samples})


class Capture:
    def __init__(self, output, samples):
        self.output = Path(output)
        self.samples = samples
        self.rows = []
        self.output.mkdir(parents=True, exist_ok=True)
        ids = [r['sample_id'] for r in samples]
        if not samples or len(set(ids)) != len(ids):
            raise ValueError('Classification capture requires unique dataset sample identities')
        save(self.output / 'classification_annotations.json', {'samples': samples})
        save(self.output / 'classification_capture.json', {'state': 'running', 'expected_images': len(samples)})

    def append(self, ranked, targets, sample_ids=None):
        rank = ranked.detach().cpu().tolist()
        truth = targets.detach().reshape(-1).cpu().tolist()
        selected = self.samples[len(self.rows):len(self.rows) + len(rank)]
        if sample_ids is not None:
            lookup = {r['sample_id']: r for r in self.samples}
            selected = [lookup[s] for s in sample_ids]
        if len(selected) != len(rank) or len(truth) != len(rank):
            raise ValueError('Native metric batch exceeds selected dataset')
        for row, label, top in zip(selected, truth, rank):
            if label != row['label']:
                raise ValueError('Native metric target differs from dataset identity')
            self.rows.append(dict(sample_id=row['sample_id'], target=int(label), prediction=int(top[0]),
                                  topk=[int(x) for x in top], scores=None))

    def finish(self):
        ids = [r['sample_id'] for r in self.rows]
        if len(ids) != len(self.samples) or set(ids) != {r['sample_id'] for r in self.samples}:
            raise ValueError('Classification capture is incomplete or duplicated')
        path = self.output / 'classification_predictions.jsonl'
        with path.open('w') as stream:
            for row in self.rows:
                stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')
        save(self.output / 'classification_capture.json', {
            'state': 'complete', 'expected_images': len(self.samples), 'completed_images': len(ids),
            'capture_stage': 'native_metric_input', 'extra_forward_calls': 0,
            'timing_note': 'Export is outside forward calls; native loop/wall time includes capture overhead.'})


def dataset_samples(dataset, source_rows):
    if hasattr(dataset, 'indices'):
        base = dataset_samples(dataset.dataset, source_rows)
        return [base[i] for i in dataset.indices]
    reader = getattr(dataset, 'reader', dataset)
    raw = getattr(reader, 'samples', None)
    if raw is None:
        raise ValueError('Native dataset does not expose a supported sample manifest')
    lookup = {str(Path(r['path']).resolve()): r for r in source_rows if r.get('path')}
    result = []
    for filename, target in raw:
        row = lookup.get(str(Path(filename).resolve()))
        if row is None:
            raise ValueError('Native dataset filename not in selected source rows: ' + str(filename))
        if row.get('target', row.get('label')) != int(target):
            raise ValueError('Native dataset class map differs from source GT')
        sid = str(row.get('id', row['path']))
        result.append(dict(sample_id=sid, label=int(target)))
    return result


@contextmanager
def accuracy_capture(owner, output, source_rows, dataset=None):
    """Wrap timm/ENOT accuracy after the original forward and postprocessing."""
    original = owner.accuracy
    original_loader = getattr(owner, 'create_loader', None)
    state = {}
    if dataset is not None:
        state['capture'] = Capture(output, dataset_samples(dataset, source_rows))
    def create_loader(ds, *args, **kwargs):
        state['capture'] = Capture(output, dataset_samples(ds, source_rows))
        return original_loader(ds, *args, **kwargs)
    def accuracy(scores, target, *args, **kwargs):
        result = original(scores, target, *args, **kwargs)
        # Identical sorted torch.topk convention used by these native evaluators.
        topk = kwargs.get('topk', args[0] if args else (1,))
        ranked = scores.topk(min(max(topk), scores.shape[1]), dim=1, largest=True, sorted=True).indices
        for k, native_value in zip(topk, result):
            exported_value = (ranked[:, :k] == target.reshape(-1, 1)).any(1).float().mean().item() * 100
            if abs(exported_value - float(native_value)) > 1e-4:
                raise ValueError('Exported ranking disagrees with native accuracy')
        state['capture'].append(ranked, target)
        return result
    owner.accuracy = accuracy
    if original_loader is not None: owner.create_loader = create_loader
    try:
        yield
        state['capture'].finish()
    finally:
        owner.accuracy = original
        if original_loader is not None: owner.create_loader = original_loader


class IdentityDataset:
    """Delegate image transforms unchanged and carry identity through collation."""
    def __init__(self, dataset, by_name):
        self.dataset, self.by_name = dataset, by_name
    def __len__(self):
        return len(self.dataset)
    def __getitem__(self, index):
        result = self.dataset[index]
        return dict(result, classification_sample_id=self.by_name[Path(self.dataset.samples[index][0]).name])
    def __getattr__(self, name):
        if name in ('dataset', 'by_name'): raise AttributeError(name)
        return getattr(self.dataset, name)


def yolo_validator(output, source_rows, fingerprint):
    from ultralytics.models.yolo.classify.val import ClassificationValidator
    expected = [dict(sample_id=str(r['id']), label=int(r['target'])) for r in source_rows]
    by_name = {fingerprint(r['id']) + '.jpg': str(r['id']) for r in source_rows}
    capture = Capture(output, expected)
    class ExportValidator(ClassificationValidator):
        def build_dataset(self, img_path):
            return IdentityDataset(super().build_dataset(img_path), by_name)
        def update_metrics(self, preds, batch):
            super().update_metrics(preds, batch)
            ids = batch['classification_sample_id']
            capture.append(self.pred[-1], batch['cls'], ids)
        def finalize_metrics(self, *args, **kwargs):
            result = super().finalize_metrics(*args, **kwargs)
            capture.finish()
            return result
    return ExportValidator


def patch_runtime(script, sequence):
    """Only runtime copies change; frozen source archives retain their hashes."""
    script = Path(script)
    edits = []
    if sequence in (74, 87):
        edits = [('            metrics = official.validate(args)',
                  '            from classification_capture import accuracy_capture\n'
                  '            with accuracy_capture(official, output, data["rows"]):\n'
                  '                metrics = official.validate(args)')]
    elif sequence == 91:
        edits = [('            accuracy = official.evaluate(torch.nn.CrossEntropyLoss(), loader, torch.device(device), model=model)',
                  '            from classification_capture import accuracy_capture\n'
                  '            with accuracy_capture(official.utils, output, data["rows"], dataset):\n'
                  '                accuracy = official.evaluate(torch.nn.CrossEntropyLoss(), loader, torch.device(device), model=model)')]
    elif sequence in (184, 193, 199, 200):
        edits = [('        result=native.validate(native_args)',
                  '        from classification_capture import accuracy_capture\n'
                  '        with accuracy_capture(native, out, rows):\n'
                  '            result=native.validate(native_args)')]
    elif sequence == 177:
        edits = [("    result=model.val(data=str(folder),split='val',imgsz=224,batch=args.batch_size,device=args.device,",
                  "    from classification_capture import yolo_validator\n"
                  "    result=model.val(validator=yolo_validator(out, rows, fingerprint),data=str(folder),split='val',imgsz=224,batch=args.batch_size,device=args.device,")]
    text = script.read_text()
    if sequence in (71,74,84,87,89,91,95):
        edits.append(('            mapping = label_mapping(spec, data, model_root)',
                      '            mapping = label_mapping(spec, data, model_root)\n'
                      '            from classification_capture import snapshot_gt\n'
                      '            snapshot_gt(output, select_rows(data, args.max_images), mapping)'))
    elif sequence in (77,78,79,80,96,97,98):
        edits.append(('            limit = len(data["rows"]) if not args.max_images else min(args.max_images, len(data["rows"]))',
                      '            limit = len(data["rows"]) if not args.max_images else min(args.max_images, len(data["rows"]))\n'
                      '            from classification_capture import snapshot_gt\n'
                      '            snapshot_gt(output, data["rows"][:limit])'))
    elif 171 <= sequence <= 200 and sequence not in (174,182):
        edits.append(("            args.segment=str(time.time_ns())",
                      "            from classification_capture import snapshot_gt\n"
                      "            snapshot_gt(out, rows)\n"
                      "            args.segment=str(time.time_ns())"))
    elif sequence in (75,81):
        edits.append(("    out.mkdir(parents=True, exist_ok=False)",
                      "    out.mkdir(parents=True, exist_ok=False)\n"
                      "    from classification_capture import snapshot_gt\n"
                      "    snapshot_gt(out, [dict(id=f'fer{index:07d}', target=target) for index, target, pixels in samples])"))
    elif sequence in (85,86,100):
        edits.append(("  if args.max_images: selected=selected[:args.max_images]",
                      "  if args.max_images: selected=selected[:args.max_images]\n"
                      "  from classification_capture import snapshot_gt\n"
                      "  snapshot_gt(out, [dict(id=image_id,target=label) for image_id,label in selected])"))
    elif sequence == 88:
        edits.append(("    status = {'state': 'running', 'requested_images': args.max_images, 'total_images': len(selected), 'completed_images': 0}",
                      "    from classification_capture import snapshot_gt\n"
                      "    snapshot_gt(run, selected)\n"
                      "    status = {'state': 'running', 'requested_images': args.max_images, 'total_images': len(selected), 'completed_images': 0}"))
    elif sequence == 83:
        edits.append(("    protocol.update(protocol_fingerprint=fingerprint, selected_images=limit)",
                      "    protocol.update(protocol_fingerprint=fingerprint, selected_images=limit)\n"
                      "    from classification_capture import snapshot_gt\n"
                      "    from itertools import islice\n"
                      "    snapshot_gt(out, [dict(id=r['id'],target=spec['id_to_label'].index(r['choice'])) for r in islice(rows(spec), limit)])"))
    elif sequence == 174:
        edits.append(('        selected = rows[:10] if args.phase == "smoke" else rows',
                      '        selected = rows[:10] if args.phase == "smoke" else rows\n'
                      '        from classification_capture import snapshot_gt\n'
                      '        snapshot_gt(out, [dict(id=r["id"],target=r["target_id"]) for r in selected])'))
    if not edits: return
    for old, new in edits:
        if text.count(old) != 1:
            raise ValueError('Classification capture anchor changed in ' + str(script))
        text = text.replace(old, new)
    compile(text, str(script), 'exec')
    script.write_text(text)
    import shutil
    shutil.copy2(__file__, script.parent / 'classification_capture.py')
