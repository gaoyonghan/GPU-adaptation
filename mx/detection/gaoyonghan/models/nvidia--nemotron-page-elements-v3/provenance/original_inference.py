"""Task-level evaluation for the excluded batch; reuses existing data readers."""
import argparse
import fcntl
import hashlib
import importlib.util
import json
import math
import sys
import time
import adapters
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
CORE_PATH = HERE.parent / 'official_object_detection_eval/run_model.py'
sys.path.insert(0, str(CORE_PATH.parent))
module_spec = importlib.util.spec_from_file_location('detection_core', CORE_PATH)
core = importlib.util.module_from_spec(module_spec)
sys.modules[module_spec.name] = core
module_spec.loader.exec_module(core)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def metrics(annotation, predictions):
    import numpy as np
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    gt = COCO()
    gt.dataset = core.ensure_coco_fields(annotation)
    gt.createIndex()
    if predictions:
        dt = gt.loadRes(predictions)
    else:
        dt = COCO()
        dt.dataset = {**gt.dataset, 'annotations': []}
        dt.createIndex()
    evaluator = COCOeval(gt, dt, 'bbox')
    evaluator.evaluate()
    evaluator.accumulate()
    evaluator.summarize()
    names = ['AP', 'AP50', 'AP75', 'AP_small', 'AP_medium', 'AP_large',
             'AR1', 'AR10', 'AR100', 'AR_small', 'AR_medium', 'AR_large']
    result = {k: float(v) if v >= 0 else None for k, v in zip(names, evaluator.stats)}
    precision = evaluator.eval['precision']
    category_names = {c['id']: c['name'] for c in annotation['categories']}
    result['per_class_AP'] = {}
    for index, cat_id in enumerate(evaluator.params.catIds):
        values = precision[:, :, index, 0, -1]
        values = values[values > -1]
        result['per_class_AP'][category_names[cat_id]] = float(np.mean(values)) if values.size else None
    return result


def run_task(model, task, manifest, args):
    import torch
    output = args.output_root / model['model_key'] / task['key']
    output.mkdir(parents=True, exist_ok=True)
    with (output / '.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        weight = Path(model['model']['local_path']) / task['weight_file']
        dataset = manifest['datasets'][task['dataset_id']]
        protocol = {'model_id': model['model_id'], 'task': task, 'dataset': dataset,
                    'weight_sha256': file_hash(weight), 'device': args.device, 'dtype': 'float32',
                    'input_size': model['protocol']['input_size'], 'conf': 0.001, 'nms_iou': 0.7,
                    'prediction_max_det': 300, 'COCO_maxDets': [1, 10, 100],
                    'runtime_versions': core.runtime_versions(), 'runner_sha256': file_hash(Path(__file__)),
                    'core_sha256': file_hash(CORE_PATH), 'fidelity': 'reference_baseline',
                    'dataset_relation': model['dataset']['relation']}
        protocol['adapter_sha256'] = file_hash(HERE / 'adapters.py')
        protocol['local_configs_sha256'] = {str(p.relative_to(weight.parent)): file_hash(p)
                                           for p in weight.parent.glob('*.json')}
        protocol.update(task.get('protocol_overrides', {}))
        if task['runner'] == 'nemotron_native':
            protocol['native_source_sha256'] = {str(p.relative_to(weight.parent)): file_hash(p)
                                                for p in weight.parent.rglob('*.py')}
        fingerprint = digest(protocol)
        status_path = output / 'status.json'
        old = json.loads(status_path.read_text()) if status_path.exists() else None
        if old and old['protocol_fingerprint'] != fingerprint:
            raise RuntimeError(f'Protocol changed; use a new output root: {output}')
        if old and old['state'] == 'complete':
            return True
        if args.max_images == 0 and (old is None or not old.get('smoke_passed')):
            raise RuntimeError('Run --max-images 10 successfully before full evaluation')
        status = {'model_id': model['model_id'], 'sequence': model['sequence'], 'task': task['key'],
                  'state': 'running', 'protocol': protocol, 'protocol_fingerprint': fingerprint,
                  'smoke_passed': bool(old and old.get('smoke_passed')), 'started_at': time.time()}
        core.atomic_json(status_path, status)
        prepared = None
        try:
            prepared = adapters.prepare(dataset, args.dataset_root, args.output_root / '_prepared_datasets', core)
            if 'category_collapse' in dataset and dataset['kind'] in {'xml_detection', 'wholebody_face_person', 'omni_layout'}:
                prepared.annotation = core.collapse_categories(prepared.annotation, dataset['category_collapse'])
            annotation = prepared.annotation
            images = annotation['images']
            image_ids = {i['id'] for i in images}
            assert images and len(image_ids) == len(images), 'Empty or duplicate image IDs'
            assert len({a['id'] for a in annotation['annotations']}) == len(annotation['annotations'])
            assert all(a['image_id'] in image_ids for a in annotation['annotations'])
            if dataset.get('expected_images') is not None:
                assert len(images) == dataset['expected_images'], (len(images), dataset['expected_images'])
            annotation_hash = digest(annotation)
            if old and old.get('annotation_sha256') and old['annotation_sha256'] != annotation_hash:
                raise RuntimeError('Ground truth changed; cannot resume')
            status['annotation_sha256'] = annotation_hash
            spec = {**model, 'protocol': {**model['protocol'], 'runner': task['runner']}}
            if task['runner'] == 'ultralytics_detection':
                model_names, predictor = core.load_ultralytics_predictor(spec, weight, args.device, 'float32')
            elif task['runner'] == 'hf_object_detection':
                model_names, predictor = core.load_hf_predictor(spec, weight.parent, args.device, 'float32')
            elif task['runner'] == 'nemotron_native':
                model_names, predictor = adapters.load_nemotron(weight, task, args.device)
            else:
                raise RuntimeError(f"Task adapter pending: {task['runner']}")
            aliases = task.get('model_label_aliases', {})
            model_names = {i: n for i, n in model_names.items() if n not in task.get('exclude_model_labels', [])}
            mapped_names = {i: aliases.get(name, name) for i, name in model_names.items()}
            mapping = core.label_mapping(mapped_names, annotation['categories'])
            # The benchmark covers only explicitly matched classes. Keep all images,
            # including negatives, and never create pseudo ground truth for other labels.
            annotation['categories'] = [c for c in annotation['categories'] if c['id'] in mapping.values()]
            annotation['annotations'] = [a for a in annotation['annotations'] if a['category_id'] in mapping.values()]
            counts = Counter(a['category_id'] for a in annotation['annotations'])
            covered = list(dict.fromkeys(model_names[i] for i, c in mapping.items() if counts[c] > 0))
            coverage = {'model_labels': model_names, 'mapping': mapping, 'covered_labels': covered,
                        'missing_labels': [n for n in model_names.values() if n not in covered],
                        'covered_count': len(covered), 'total_model_classes': len(set(model_names.values())),
                        'gt_instances_per_class': {c['name']: counts[c['id']] for c in annotation['categories']}}
            assert covered, 'No matched class has ground-truth instances'
            status.update({'coverage': coverage, 'total_images': len(images)})
            core.atomic_json(status_path, status)
            predictions_path = output / 'predictions.jsonl'
            records = core.read_jsonl(predictions_path)
            existing = {r['image_id'] for r in records}
            assert len(existing) == len(records) and existing <= image_ids, 'Invalid checkpoint image IDs'
            selected = images[:args.max_images] if args.max_images else images
            if args.device.startswith('cuda'):
                torch.cuda.set_device(args.device)
                torch.cuda.init()
                torch.cuda.reset_peak_memory_stats(args.device)
            for index, info in enumerate(selected, 1):
                if info['id'] in existing:
                    continue
                raw = predictor(prepared.read_image(info))
                detections = []
                for box, score, label in raw:
                    assert all(math.isfinite(float(v)) for v in [*box, score]), 'Non-finite prediction'
                    if label not in mapping:
                        continue
                    x1, y1, x2, y2 = box
                    assert x2 >= x1 and y2 >= y1 and 0 <= score <= 1
                    detections.append({'image_id': info['id'], 'category_id': mapping[label],
                                       'bbox': [x1, y1, x2-x1, y2-y1], 'score': score})
                core.append_jsonl(predictions_path, {'image_id': info['id'], 'detections': detections})
                if index == 1 or index % 100 == 0 or index == len(selected):
                    print(f"[{model['sequence']}:{task['key']}] {index}/{len(selected)}", flush=True)
            records = core.read_jsonl(predictions_path)
            completed = {r['image_id'] for r in records}
            assert {i['id'] for i in selected} <= completed
            status.update({'completed_images': len(completed), 'state': 'partial_complete',
                           'smoke_passed': len(selected) >= min(10, len(images))})
            if args.max_images == 0:
                assert completed == image_ids and len(records) == len(images)
                predictions = [p for r in records for p in r['detections']]
                core.atomic_json(output / 'predictions.json', predictions)
                score = metrics(annotation, predictions)
                result = {'metrics': score, 'unit': 'fraction_0_to_1', 'coverage': coverage,
                          'evaluation_images': len(images), 'annotation_sha256': annotation_hash,
                          'protocol_fingerprint': fingerprint, 'reference_check': 'not_officially_comparable'}
                core.atomic_json(output / 'metrics.json', result)
                status.update({'state': 'complete', 'metrics': result})
            if args.device.startswith('cuda'):
                status['peak_gpu_memory_bytes'] = torch.cuda.max_memory_allocated(args.device)
            return True
        except Exception as exc:
            import traceback
            traceback.print_exc()
            status.update({'state': 'failed', 'error': repr(exc)})
            return False
        finally:
            if prepared:
                prepared.close()
            status['elapsed_seconds'] = time.time() - status['started_at']
            core.atomic_json(status_path, status)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True, type=int)
    parser.add_argument('--max-images', type=int, default=10)
    parser.add_argument('--device', default='cuda:1')
    parser.add_argument('--dataset-root', type=Path, default=core.DEFAULT_DATASET_ROOT)
    parser.add_argument('--output-root', type=Path, default=Path('/home/ubuntu/gaoyonghan/gpu/results/remaining25_v3'))
    parser.add_argument('--describe', action='store_true')
    args = parser.parse_args()
    manifest = json.loads((HERE / 'models.json').read_text())
    model = next(m for m in manifest['models'] if m['sequence'] == args.model)
    if args.describe:
        print(json.dumps(model, ensure_ascii=False, indent=2))
        return 0
    if not model['tasks']:
        raise RuntimeError(model['next_action'])
    ok = True
    for task in model['tasks']:
        ok = run_task(model, task, manifest, args) and ok
    return 0 if ok else 1


if __name__ == '__main__':
    raise SystemExit(main())
