#!/usr/bin/env python3
"""Standalone unified segmentation evaluator; all paths are based on the project root."""
from __future__ import annotations
import sys, os, json
from pathlib import Path
sys.dont_write_bytecode=True
# Standalone, explicit activation of existing packages; no venv files are edited.
def _activate_project_venv(argv, project_root):
    import argparse
    import os
    import sys
    from pathlib import Path
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    parser.add_argument('--venv')
    option, _ = parser.parse_known_args(argv)
    if not option.venv:
        return None
    if not sys.flags.no_site:
        raise ValueError('--venv requires the matching Python with -I -S -B (see environment/README.md)')
    supplied = Path(option.venv).expanduser()
    project = Path(project_root).resolve()
    candidate = supplied if supplied.is_absolute() else project / supplied
    environment = candidate.resolve()
    if not environment.is_relative_to(project) or environment == project:
        raise ValueError('Environment must belong to this project copy')
    if any(part.is_symlink() for part in (candidate, *candidate.parents)):
        raise ValueError('Environment path must not use symlinks')
    values = {}
    for line in (environment / 'pyvenv.cfg').read_text().splitlines():
        key, sep, value = line.partition('=')
        if sep:
            values[key.strip().lower()] = value.strip()
    version = tuple(int(part) for part in values['version'].split('.')[:2])
    if version != sys.version_info[:2] or values.get('include-system-site-packages', '').lower() != 'false':
        raise ValueError('Requested venv must match this Python version and exclude system site-packages')
    packages = environment / 'lib' / ('python%d.%d' % sys.version_info[:2]) / 'site-packages'
    if not packages.is_dir():
        raise ValueError('Missing site-packages: ' + str(packages))
    # Do not execute .pth launch hooks or activate scripts containing old paths.
    # Keep Python's standard library before legacy backport packages (e.g. typing).
    sys.path.append(str(packages))
    sys.prefix = sys.exec_prefix = str(environment)
    sys.dont_write_bytecode = True
    os.environ['STANDALONE_VENV_READY'] = str(environment)
    os.environ['VIRTUAL_ENV'] = str(environment)
    os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
    os.environ.setdefault('PYDANTIC_DISABLE_PLUGINS', '__all__')
    return environment

def _prepare_model_runtime(project_root, model_id, stage='full'):
    """Retain the successful model-specific package selection, using only this copy."""
    import os
    import sys
    from pathlib import Path
    if stage == 'export':
        os.environ['CUDA_VISIBLE_DEVICES']=''
        return
    if '--help' in sys.argv or '-h' in sys.argv or '--dry-run' in sys.argv or '--check' in sys.argv:
        return
    if model_id.startswith(('ibm-granite__','IBM__')) or model_id=='shi-labs__oneformer_ade20k_dinat_large':
        for optional in ('transformer_engine','transformer_engine_extensions','msamp','flash_attn','flash_attn_2_cuda'):
            sys.modules[optional]=None
    if model_id=='shi-labs__oneformer_ade20k_dinat_large':
        if 'transformers' in sys.modules or 'natten' in sys.modules:
            raise RuntimeError('DiNAT compatibility must be configured before Transformers/NATTEN imports')
        alternate=Path(project_root)/'venv_tf/lib'/('python%d.%d'%sys.version_info[:2])/'site-packages'
        if not alternate.is_dir():raise ValueError('DiNAT requires the documented alternate Transformers/NATTEN environment')
        import numpy
        import torch
        if torch.cuda.is_available():
            torch.cuda.init()
            torch.backends.cudnn.version()
        main_site=str(Path(project_root)/'venv/lib'/('python%d.%d'%sys.version_info[:2])/'site-packages')
        position=sys.path.index(main_site) if main_site in sys.path else len(sys.path)
        sys.path.insert(position,str(alternate))


def _prepare_library_loader(environment, script):
    """Set package CUDA libraries before exec; copied venv binaries need not execute."""
    import os
    import sys
    from pathlib import Path
    if environment is None or any(flag in sys.argv for flag in ('--help','-h','--dry-run','--check')):
        return
    environment=Path(environment).resolve()
    marker=str(environment)
    if os.environ.get('SEGMENTATION_LIBRARY_ENV')==marker:return
    packages=environment/'lib'/('python%d.%d'%sys.version_info[:2])/'site-packages'
    libraries=sorted((packages/'nvidia').glob('*/lib'))
    if (packages/'torch/lib').is_dir():libraries.append(packages/'torch/lib')
    updated=os.environ.copy()
    for key in ('PYTHONHOME','PYTHONPATH','LD_PRELOAD','SEGMENTATION_TF_VENV','STANDALONE_VENV_READY'):
        updated.pop(key,None)
    updated['SEGMENTATION_LIBRARY_ENV']=marker
    updated['LD_LIBRARY_PATH']=os.pathsep.join(str(p) for p in libraries)
    updated['PYTHONDONTWRITEBYTECODE']='1'
    os.execve(sys.executable,[sys.executable,'-I','-S','-B',str(Path(script).resolve()),*sys.argv[1:]],updated)

if __name__ == '__main__':
    _here=Path(__file__).resolve().parent
    _root=_here.parents[1] if _here.parent.name=='Model' else _here
    if '--help' not in sys.argv and '-h' not in sys.argv:
        _activate_project_venv(sys.argv[1:],_root)

import argparse
def evaluation_parser():
    parser=argparse.ArgumentParser(description='CPU unified IoU/mIoU and class-region COCO Mask AP evaluation; serial, resume by default.')
    parser.add_argument('--venv',help='Existing same-version environment; use Python -I -S -B')
    parser.add_argument('--environment',choices=['auto','current'],default='current',help='Compatibility argument; current interpreter is used')
    group=parser.add_mutually_exclusive_group()
    group.add_argument('--model',action='append');group.add_argument('--start-from')
    parser.add_argument('--bundle',help='Explicit bundle under selected model/result; paths relative to project root')
    parser.add_argument('--output',help='Single-model output JSON under this model/result')
    parser.add_argument('--output-root',help='Isolated evaluation output root within project; one subdirectory per model')
    parser.add_argument('--force',action='store_true',help='Recompute metrics even when complete')
    parser.add_argument('--resume',action='store_true',default=True,help='Default: reuse validated completed metrics')
    parser.add_argument('--dry-run',action='store_true',help='Validate without computing metrics or writing')
    parser.add_argument('--check-level',choices=['metadata','payload','full'],default='payload')
    parser.add_argument('--stop-on-error',action='store_true')
    parser.add_argument('--parent-lock-owner',type=int,help=argparse.SUPPRESS)
    return parser
if __name__=='__main__' and ('--help' in sys.argv or '-h' in sys.argv):
    evaluation_parser().parse_args(['--help'])
import operator
import numpy as np

def encode_rle(mask) -> dict:
    """Encode a 2D mask; zero pixels are background, nonzero foreground."""
    array = np.asarray(mask)
    if array.ndim != 2:
        raise ValueError(f'COCO masks must be 2D, received shape {array.shape}')
    flat = np.asarray(array, dtype=bool).ravel(order='F')
    if flat.size:
        changes = np.flatnonzero(flat[1:] != flat[:-1]) + 1
        runs = np.diff(np.concatenate(([0], changes, [flat.size]))).tolist()
        if flat[0]:
            runs.insert(0, 0)
    else:
        runs = [0]
    characters = []
    for index, run in enumerate(runs):
        value = int(run) - (int(runs[index - 2]) if index > 2 else 0)
        while True:
            group = value & 31
            value >>= 5
            more = value != (-1 if group & 16 else 0)
            characters.append(chr((group | (32 if more else 0)) + 48))
            if not more:
                break
    return {'size': [int(array.shape[0]), int(array.shape[1])], 'counts': ''.join(characters)}

def _integer(value, role):
    try:
        if isinstance(value, (bool, np.bool_)):
            raise TypeError
        result = operator.index(value)
    except TypeError as exc:
        raise ValueError(f'{role} must be an integer') from exc
    if result < 0:
        raise ValueError(f'{role} must be nonnegative')
    return int(result)

def _compressed_runs(text):
    if isinstance(text, bytes):
        try:
            text = text.decode('ascii')
        except UnicodeDecodeError as exc:
            raise ValueError('COCO RLE counts must contain ASCII characters') from exc
    runs = []
    position = 0
    while position < len(text):
        value, shift = (0, 0)
        while True:
            if position == len(text):
                raise ValueError('Truncated COCO RLE integer')
            group = ord(text[position]) - 48
            position += 1
            if not 0 <= group <= 63:
                raise ValueError('Invalid character in COCO RLE counts')
            value |= (group & 31) << shift
            shift += 5
            if not group & 32:
                if group & 16:
                    value |= -1 << shift
                break
        if len(runs) > 2:
            value += runs[-2]
        if value < 0:
            raise ValueError('COCO RLE contains a negative run length')
        runs.append(value)
    return runs

def decode_rle(rle) -> np.ndarray:
    """Decode compressed string/bytes or uncompressed counts to a bool mask."""
    try:
        size, counts = (rle['size'], rle['counts'])
        if len(size) != 2:
            raise ValueError('COCO RLE size must contain height and width')
    except (KeyError, TypeError) as exc:
        raise ValueError('COCO RLE requires size and counts') from exc
    height = _integer(size[0], 'height')
    width = _integer(size[1], 'width')
    if isinstance(counts, (str, bytes)):
        runs = _compressed_runs(counts)
    elif isinstance(counts, (list, tuple, np.ndarray)):
        runs = [_integer(value, 'run length') for value in counts]
    else:
        raise ValueError('COCO RLE counts must be a string, bytes, or integer sequence')
    total = height * width
    if sum(runs) != total:
        raise ValueError('COCO RLE run lengths do not match the declared image size')
    flat = np.zeros(total, dtype=bool)
    offset = 0
    for index, length in enumerate(runs):
        if index & 1:
            flat[offset:offset + length] = True
        offset += length
    return flat.reshape((height, width), order='F')
import contextlib
import hashlib
import io
import json
import math
import os
from pathlib import Path
import tempfile
import numpy as np
METRIC_KEYS = ('mIoU', 'IoU', 'Mask AP@[.50:.95]', 'AP50', 'AP75')
AP_KEYS = ('Mask AP@[.50:.95]', 'AP50', 'AP75')
REGION_AP_DEFINITION = 'class_region_ap_v1'
SCHEMA = 'segmentation-coco-bundle/v1'

def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.name + '.', dir=path.parent)
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as stream:
            json.dump(value, stream, ensure_ascii=False, allow_nan=False, indent=2)
            stream.write('\n')
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)

def sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()

def encode_mask(mask):
    return encode_rle(mask)

def decode_mask(rle):
    return decode_rle(rle)

def _annotation(mask, image_id, category_id, annotation_id=None, score=None, iscrowd=0):
    rle = encode_mask(mask)
    result = {'image_id': image_id, 'category_id': int(category_id), 'segmentation': rle}
    if annotation_id is not None:
        ys, xs = np.nonzero(np.asarray(mask, bool))
        bbox = [int(xs.min()), int(ys.min()), int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1)] if len(xs) else [0, 0, 0, 0]
        result.update(id=annotation_id, area=int(len(xs)), bbox=bbox, iscrowd=int(iscrowd))
    if score is not None:
        if not math.isfinite(float(score)):
            raise ValueError('instance score must be finite')
        result['score'] = float(score)
    return result

def validate_bundle(path):
    path = Path(path)
    manifest = json.loads((path / 'manifest.json').read_text(encoding='utf-8'))
    if manifest.get('schema_version') != SCHEMA or manifest.get('complete') is not True:
        raise ValueError('unsupported or incomplete bundle manifest')
    required = {'ground_truth.json', 'predictions.json', 'semantic_ground_truth.json', 'semantic_predictions.json'}
    if set(manifest.get('payload_sha256', {})) != required:
        raise ValueError('incomplete payload hash manifest')
    for name, expected in manifest['payload_sha256'].items():
        if sha256(path / name) != expected:
            raise ValueError(f'payload hash mismatch: {name}')
    payload = {name: json.loads((path / name).read_text(encoding='utf-8')) for name in required}
    gt = payload['ground_truth.json']
    if 'image_artifacts_sha256' in manifest:
        hashes = manifest['image_artifacts_sha256']
        references = set()
        for item in gt['images']:
            artifacts = item.get('artifacts')
            if not artifacts or item['file_name'] != artifacts.get('image'):
                raise ValueError('image artifact coverage is incomplete')
            references.update([artifacts['image'], artifacts['valid_mask']])
            references.update((x['path'] for x in artifacts['originals']))
            for role in ('ground_truth', 'prediction'):
                masks = artifacts[role]
                references.update((masks[key] for key in ('mask', 'labels', 'preview', 'layout_preview') if key in masks))
                references.update((x['path'] for x in masks.get('instances', [])))
        if set(hashes) != references:
            raise ValueError('image artifact hash manifest does not match image references')
        for name, expected in hashes.items():
            candidate = path / name
            if Path(name).is_absolute() or not candidate.resolve().is_relative_to(path.resolve()):
                raise ValueError(f'invalid image artifact path: {name}')
            if not candidate.is_file() or sha256(candidate) != expected:
                raise ValueError(f'image artifact missing or hash mismatch: {name}')
    if gt['categories'] != manifest['categories']:
        raise ValueError('manifest and ground-truth categories differ')
    images = {im['id']: im for im in gt['images']}
    sample_ids = [x['sample_id'] for x in gt['images']]
    if len(images) != len(gt['images']) or len(set(sample_ids)) != len(sample_ids):
        raise ValueError('duplicate images or samples')
    if sample_ids != manifest['coverage_sample_ids'] or len(images) != manifest['expected_samples'] or len(images) != manifest['evaluated_samples']:
        raise ValueError('coverage mismatch in bundle')
    for im in images.values():
        if decode_mask(im['valid_mask']).shape != (im['height'], im['width']):
            raise ValueError('valid mask shape mismatch')
    for name in required:
        entries = payload[name]['annotations'] if name == 'ground_truth.json' else payload[name]
        for entry in entries:
            if entry['image_id'] not in images:
                raise ValueError('annotation references an unknown image')
            im = images[entry['image_id']]
            if decode_mask(entry['segmentation']).shape != (im['height'], im['width']):
                raise ValueError('annotation mask shape mismatch')
            if name == 'predictions.json' and (not math.isfinite(entry['score'])):
                raise ValueError('nonfinite prediction score')
    return (manifest, payload)

def _class_regions(annotations):
    groups = {}
    for ann in annotations:
        groups.setdefault(ann['image_id'], {}).setdefault(int(ann['category_id']), []).append(ann['segmentation'])
    return groups

def _union(rles, shape):
    result = np.zeros(shape, bool)
    for rle in rles:
        result |= decode_mask(rle)
    return result

def _coco_mask_ap(ground_truth, predictions, *, use_categories=True):
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    with contextlib.redirect_stdout(io.StringIO()) as log:
        coco_gt = COCO()
        coco_gt.dataset = ground_truth
        coco_gt.createIndex()
        if predictions:
            coco_dt = coco_gt.loadRes([{k: v for k, v in p.items() if k != 'bbox'} for p in predictions])
        else:
            coco_dt = COCO()
            coco_dt.dataset = {'images': ground_truth['images'], 'categories': ground_truth['categories'], 'annotations': []}
            coco_dt.createIndex()
        evaluator = COCOeval(coco_gt, coco_dt, 'segm')
        evaluator.params.imgIds = [im['id'] for im in ground_truth['images']]
        evaluator.params.useCats = int(use_categories)
        evaluator.evaluate()
        evaluator.accumulate()
        evaluator.summarize()
    return ({key: float(value) if value >= 0 else None for key, value in zip(AP_KEYS, evaluator.stats[:3])}, log.getvalue())

def _region_ap(manifest, payload):
    """COCO mask matching of class unions, independent of native instances.

    Constant scores are a declared proxy protocol, not recovered confidence.
    Canonical sample IDs fix COCO's otherwise order-sensitive score ties.
    """
    categories = manifest['categories']
    if manifest['task'] == 'binary' or len(categories) == 1:
        categories = [c for c in categories if int(c['id']) == manifest['foreground_id']]
    categories = sorted(categories, key=lambda c: int(c['id']))
    category_ids = {int(c['id']) for c in categories}
    gt_regions = _class_regions(payload['semantic_ground_truth.json'])
    pred_regions = _class_regions(payload['semantic_predictions.json'])
    images, actual, predicted = ([], [], [])
    ignored = 0
    for im in sorted(payload['ground_truth.json']['images'], key=lambda x: str(x['sample_id'])):
        valid = decode_mask(im['valid_mask'])
        if not valid.any():
            ignored += 1
            continue
        image_id = len(images) + 1
        images.append({'id': image_id, 'width': im['width'], 'height': im['height'], 'sample_id': im['sample_id']})
        shape = (im['height'], im['width'])
        present = set(gt_regions.get(im['id'], {})) | set(pred_regions.get(im['id'], {}))
        for cid in sorted(present & category_ids):
            for regions, destination, is_gt in ((gt_regions, actual, True), (pred_regions, predicted, False)):
                mask = _union(regions.get(im['id'], {}).get(cid, []), shape) & valid
                if mask.any():
                    destination.append(_annotation(mask, image_id, cid, annotation_id=len(destination) + 1 if is_gt else None, score=None if is_gt else 1.0))
    protocol = {'definition': REGION_AP_DEFINITION, 'metric_kind': 'derived_class_region_mask_ap', 'region_definition': 'one union mask per image and category, intersected with valid pixels', 'score_policy': 'constant 1.0 for every model; does not evaluate confidence ranking', 'tie_order': 'lexicographic sample_id, canonical image IDs, then category ID', 'tie_limitation': 'Equal-score COCO AP depends on this fixed sample order.', 'category_policy': 'foreground only for binary; all declared categories for multiclass', 'crowd_policy': 'class unions include all valid labelled pixels; native_instance_metrics separately retains native crowd-ignore rules', 'category_ids': [int(c['id']) for c in categories], 'iou_type': 'segm', 'iou_thresholds': [round(0.5 + 0.05 * i, 2) for i in range(10)], 'max_dets': [1, 10, 100], 'use_categories': True, 'evaluated_image_count': len(images), 'ignored_image_count': ignored, 'ground_truth_region_count': len(actual), 'prediction_region_count': len(predicted)}
    if not actual:
        return (dict.fromkeys(AP_KEYS), {key: 'No eligible class-region ground truth on valid pixels; region AP is undefined.' for key in AP_KEYS}, protocol)
    dataset = {'info': {'description': REGION_AP_DEFINITION}, 'images': images, 'categories': categories, 'annotations': actual}
    metrics, log = _coco_mask_ap(dataset, predicted)
    protocol['coco_summary'] = log
    reasons = {key: 'No eligible class-region ground truth.' for key, value in metrics.items() if value is None}
    return (metrics, reasons, protocol)

def _apply_forward_timing_override(result, path, output=None):
    """Apply a benchmark bound to this export, retaining its inference metadata."""
    metadata = result['metadata']
    model_id = metadata.get('model_id')
    if (not isinstance(model_id, str) or not model_id or model_id in ('.', '..')
            or '/' in model_id or '\\' in model_id):
        return result
    candidates = []
    for parent in Path(path).resolve().parents:
        if parent.name == 'result' and parent.parent.name == model_id and parent.parent.parent.name == 'Model':
            candidates.append(parent / 'forward_timing.json')
            break
    if output is not None:
        destination = Path(output).absolute().parent
        if (destination.name == 'result' and destination.parent.name == model_id
                and destination.parent.parent.name == 'Model'):
            candidates.append(destination / 'forward_timing.json')
    for candidate in candidates:
        if any(item.is_symlink() for item in (candidate, *list(candidate.parents)[:3])):
            raise ValueError(f'Forward timing override must not use symlinks: {candidate}')
        if not candidate.is_file():
            continue
        override = json.loads(candidate.read_text(encoding='utf-8'))
        if not isinstance(override, dict):
            raise ValueError(f'Forward timing override must be an object: {candidate}')
        identity = {'model_id': model_id, 'original_run_id': metadata.get('run_id'),
                    'bundle_manifest_sha256': result['bundle_manifest_sha256'],
                    'source_manifest_sha256': metadata.get('source_manifest_sha256')}
        # A new inference run must never inherit an older run's benchmark.
        if any(key in override and override[key] != value for key, value in identity.items()):
            continue
        if (override.get('schema_version') != 'segmentation-forward-timing-override/v1'
                or any(not isinstance(value, str) or not value or override.get(key) != value
                       for key, value in identity.items())):
            raise ValueError(f'Invalid forward timing override identity or schema: {candidate}')
        fields = override.get('forward_fields')
        required = {'total_forward_seconds', 'mean_forward_ms',
                    'forward_timed_sample_count', 'forward_time_reason'}
        if not isinstance(fields, dict) or set(fields) != required:
            raise ValueError(f'Invalid forward timing override fields: {candidate}')
        benchmark = override.get('benchmark_timing')
        provenance = override.get('provenance')
        if (not isinstance(benchmark, dict) or not isinstance(provenance, dict)
                or not isinstance(override.get('benchmark_environment'), dict)):
            raise ValueError(f'Missing forward timing benchmark context: {candidate}')
        original_timing = metadata.get('timing')
        if not isinstance(original_timing, dict):
            raise ValueError(f'Missing original timing for forward override: {candidate}')
        count = original_timing.get('sample_count')
        counts = (count, fields['forward_timed_sample_count'], benchmark.get('sample_count'),
                  benchmark.get('forward_timed_sample_count'))
        if any(type(value) is not int or value < 1 or value != count for value in counts):
            raise ValueError(f'Inconsistent forward timing sample counts: {candidate}')
        total, mean = fields['total_forward_seconds'], fields['mean_forward_ms']
        if (any(type(value) not in (int, float) or not math.isfinite(value) or value <= 0
                for value in (total, mean))
                or not math.isclose(mean, total * 1000 / count, rel_tol=1e-9, abs_tol=1e-9)
                or fields['forward_time_reason'] is not None
                or any(type(benchmark.get(key)) not in (int, float) or benchmark[key] != fields[key]
                       for key in ('total_forward_seconds', 'mean_forward_ms'))):
            raise ValueError(f'Invalid or inconsistent forward timing measurements: {candidate}')
        # Only forward values come from the benchmark. Environment and pipeline
        # timing still describe the original inference; provenance identifies
        # the separate benchmark timing and environment in forward_timing.json.
        timing = dict(original_timing, **fields, forward_benchmark=dict(provenance))
        return dict(result, metadata=dict(metadata, timing=timing))
    return result


def evaluate_bundle(path, output=None):
    manifest, payload = validate_bundle(path)
    metrics = dict.fromkeys(METRIC_KEYS)
    reasons = {}
    gt = payload['ground_truth.json']
    categories = manifest['categories']
    class_ids = [int(c['id']) for c in categories]
    intersections, unions = (dict.fromkeys(class_ids, 0), dict.fromkeys(class_ids, 0))
    sg = _class_regions(payload['semantic_ground_truth.json'])
    sp = _class_regions(payload['semantic_predictions.json'])
    semantic_available = all((im['semantic_available'] for im in gt['images']))
    valid_pixel_count = 0
    if semantic_available:
        for im in gt['images']:
            shape = (im['height'], im['width'])
            valid = decode_mask(im['valid_mask'])
            valid_pixel_count += int(valid.sum())
            actual, predicted = (sg.get(im['id'], {}), sp.get(im['id'], {}))
            for cid in (set(actual) | set(predicted)) & set(class_ids):
                a, b = (_union(actual.get(cid, []), shape), _union(predicted.get(cid, []), shape))
                intersections[cid] += int(np.count_nonzero(a & b & valid))
                unions[cid] += int(np.count_nonzero((a | b) & valid))
        per_class = {str(cid): intersections[cid] / unions[cid] if unions[cid] else None for cid in class_ids}
        if manifest['task'] == 'binary' or len(class_ids) == 1:
            metrics['IoU'] = per_class.get(str(manifest['foreground_id']))
            reasons['mIoU'] = 'Binary task reports foreground IoU.'
            if metrics['IoU'] is None:
                reasons['IoU'] = 'Foreground union is empty over all valid pixels.'
        else:
            values = [v for v in per_class.values() if v is not None]
            metrics['mIoU'] = sum(values) / len(values) if values else None
            reasons['IoU'] = 'Multiclass task reports mIoU.'
            if not values:
                reasons['mIoU'] = 'No class has a nonempty union.'
    else:
        per_class = {}
        reasons.update(mIoU=manifest['metadata'].get('iou_reason') or 'No pixel segmentation prediction and ground truth.', IoU='No binary pixel segmentation prediction and ground truth.')
    native_metrics = dict.fromkeys(AP_KEYS)
    native_reasons = {}
    native_details = {'definition': 'native_instance_mask_ap', 'iou_type': 'segm', 'iou_thresholds': [round(0.5 + 0.05 * i, 2) for i in range(10)], 'max_dets': [1, 10, 100], 'use_categories': not manifest['class_agnostic']}
    if manifest['ap_supported']:
        native_metrics, log = _coco_mask_ap(gt, payload['predictions.json'], use_categories=not manifest['class_agnostic'])
        native_details['coco_summary'] = log
        native_reasons = {key: 'No eligible non-crowd instance ground truth.' for key, value in native_metrics.items() if value is None}
    else:
        native_reasons = {key: manifest.get('ap_reason') or 'No native instance annotations and scored instance predictions.' for key in AP_KEYS}
    if semantic_available and manifest['task'] != 'unsupported':
        region_metrics, region_reasons, ap_details = _region_ap(manifest, payload)
        metrics.update(region_metrics)
        reasons.update(region_reasons)
    else:
        ap_details = {'definition': REGION_AP_DEFINITION, 'metric_kind': 'unavailable_without_pixel_masks'}
        reasons.update({key: 'No pixel segmentation masks; class-region Mask AP is not applicable. See box_metrics for supported layout tasks.' for key in AP_KEYS})
    result = {'schema_version': 'segmentation-evaluation/v2', 'status': 'complete', 'metrics': metrics, 'metric_reasons': reasons, 'metric_scale': 'ratio [0,1]', 'per_class_iou': per_class, 'intersections': intersections, 'unions': unions, 'valid_pixel_count': valid_pixel_count, 'coverage': {'evaluated_samples': manifest['evaluated_samples'], 'expected_samples': manifest['expected_samples'], 'scope': manifest['metadata'].get('scope', 'full')}, 'protocol': {'task': manifest['task'], 'iou_aggregation': 'global pixel intersections / global unions', 'empty_union': 'null, excluded from macro average', 'ap': ap_details, 'native_instance_ap': native_details}, 'native_instance_metrics': native_metrics, 'native_instance_metric_reasons': native_reasons, 'bundle': str(Path(path).resolve()), 'bundle_manifest_sha256': sha256(Path(path) / 'manifest.json'), 'metadata': manifest['metadata']}
    result = _apply_forward_timing_override(result, path, output)
    if output is not None:
        atomic_json(output, result)
    return result
import argparse
import time
from datetime import datetime, timezone

PAYLOADS = ('ground_truth.json','predictions.json','semantic_ground_truth.json','semantic_predictions.json')
ENTRY = Path(__file__).resolve()
SINGLE_MODEL = ENTRY.parent if ENTRY.parent.parent.name == 'Model' else None
ROOT = SINGLE_MODEL.parents[1] if SINGLE_MODEL else ENTRY.parent

def resolve_argument(value, root=ROOT):
    path=Path(value).expanduser()
    return (path if path.is_absolute() else root/path).resolve()

def select_models(root, names=None, start=None):
    if any(p.is_symlink() for p in (root/'Model',root,*root.parents)):
        raise ValueError('Project/model paths must not use symlinks')
    entries=sorted((root/'Model').iterdir())
    if any(p.is_symlink() for p in entries):raise ValueError('Model directories must not use symlinks')
    available={p.name:p for p in entries if p.is_dir() and not p.name.startswith('.')}
    unknown=set(names or [])-available.keys()
    if unknown: raise ValueError('Unknown model(s): '+', '.join(sorted(unknown)))
    if start and start not in available: raise ValueError('Unknown start model: '+start)
    return [p for n,p in available.items() if (not names or n in names) and (not start or n>=start)]

def read_object(path):
    value=json.loads(Path(path).read_text())
    if not isinstance(value,dict):raise ValueError('Expected object: '+str(path))
    return value

def select_current_bundle(model, explicit_bundle=None, check_level='payload'):
    model=Path(model).absolute()
    if any(p.is_symlink() for p in (model,*model.parents)):
        raise ValueError('Model paths must not use symlinks')
    model=model.resolve()
    bundle=resolve_argument(explicit_bundle,model.parents[1]) if explicit_bundle else model/'result/coco_bundle'
    if not bundle.is_relative_to(model/'result'):
        raise ValueError('Bundle must belong to the selected model/result')
    if any(p.is_symlink() for p in (bundle,*bundle.parents)):
        raise ValueError('Bundle path must not use symlinks')
    manifest=read_object(bundle/'manifest.json');meta=manifest.get('metadata',{})
    if manifest.get('schema_version')!=SCHEMA or manifest.get('complete') is not True or meta.get('model_id')!=model.name:
        raise ValueError('Incomplete bundle or model identity mismatch')
    count=manifest.get('expected_samples')
    if type(count) is not int or count<1 or manifest.get('evaluated_samples')!=count or len(manifest.get('coverage_sample_ids',[]))!=count:
        raise ValueError('Incomplete bundle sample coverage')
    hashes=manifest.get('payload_sha256',{})
    if set(hashes)!=set(PAYLOADS):raise ValueError('Incomplete payload hash list')
    for name in PAYLOADS:
        p=bundle/name
        if not p.is_file() or p.is_symlink():raise ValueError('Missing payload: '+name)
        if check_level in ('payload','full') and sha256(p)!=hashes[name]:raise ValueError('Payload hash mismatch: '+name)
    if check_level=='full':validate_bundle(bundle)
    digest=sha256(bundle/'manifest.json')
    if not explicit_bundle:
        current=read_object(model/'result/results.json');metadata=current.get('metadata') or {}
        if current.get('status') not in ('complete','exported'):raise ValueError('Current result has no successful exported bundle')
        for owner in (current.get('model_id'),metadata.get('model_id')):
            if owner is not None and owner!=model.name:raise ValueError('Current result model mismatch')
        for run in (current.get('run_id'),metadata.get('run_id')):
            if run is not None and run!=meta.get('run_id'):raise ValueError('Current result run mismatch')
        if current.get('bundle_manifest_sha256') not in (None,digest):raise ValueError('Current result manifest mismatch')
    return bundle,manifest,digest

def reusable_result(output, model, bundle, manifest_digest):
    try:
        value=read_object(output)
        if (value.get('status')!='complete' or value.get('bundle_manifest_sha256')!=manifest_digest
            or value.get('schema_version')!='segmentation-evaluation/v2'
            or value.get('metadata',{}).get('model_id')!=model.name
            or value.get('protocol',{}).get('ap',{}).get('definition')!=REGION_AP_DEFINITION):return None
        manifest=read_object(bundle/'manifest.json')
        actual_meta=value.get('metadata',{});expected_meta=manifest.get('metadata',{})
        for key in ('model_id','run_id','source_manifest_sha256','dataset','splits','seed','scope'):
            if actual_meta.get(key)!=expected_meta.get(key):return None
        coverage=value.get('coverage',{})
        if coverage.get('expected_samples')!=manifest.get('expected_samples') or coverage.get('evaluated_samples')!=manifest.get('evaluated_samples'):return None
        metrics=value.get('metrics',{})
        if set(metrics)!=set(METRIC_KEYS):return None
        if any(v is not None and (type(v) not in (int,float) or not math.isfinite(v) or not 0<=v<=1) for v in metrics.values()):return None
        return value
    except (OSError,ValueError,TypeError):return None

from contextlib import contextmanager

@contextmanager
def evaluation_lock(model, parent_owner=None, read_only=False):
    if read_only:
        yield
        return
    if parent_owner is not None:
        expected=str(parent_owner)+':'+str(model.resolve())
        if parent_owner!=os.getppid() or os.environ.get('SEGMENTATION_MODEL_LOCK')!=expected:
            raise ValueError('Invalid parent model lock claim')
        yield
        return
    import fcntl
    if any(p.is_symlink() for p in (model,*model.parents)):
        raise ValueError('Model paths must not use symlinks')
    directory=model/'result'
    directory.mkdir(parents=True,exist_ok=True)
    if directory.is_symlink():raise ValueError('Result must not be symlink')
    fd=os.open(directory/'.two_stage.lock',os.O_CREAT|os.O_RDWR|os.O_NOFOLLOW,0o600)
    try:
        import stat
        info=os.fstat(fd)
        if not stat.S_ISREG(info.st_mode) or info.st_nlink!=1:raise ValueError('Unsafe model lock file')
        fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
        yield
    finally:
        os.close(fd)

def evaluation_parser():
    parser=argparse.ArgumentParser(description='CPU unified IoU/mIoU and class-region COCO Mask AP evaluation; serial, resume by default.')
    parser.add_argument('--venv',help='Existing same-version environment; use Python -I -S -B')
    parser.add_argument('--environment',choices=['auto','current'],default='current',help='Compatibility argument; current interpreter is used')
    group=parser.add_mutually_exclusive_group()
    group.add_argument('--model',action='append');group.add_argument('--start-from')
    parser.add_argument('--bundle',help='Explicit bundle under selected model/result; paths relative to project root')
    parser.add_argument('--output',help='Single-model output JSON under this model/result')
    parser.add_argument('--output-root',help='Isolated evaluation output root within project; one subdirectory per model')
    parser.add_argument('--force',action='store_true',help='Recompute metrics even when complete')
    parser.add_argument('--resume',action='store_true',default=True,help='Default: reuse validated completed metrics')
    parser.add_argument('--dry-run',action='store_true',help='Validate without computing metrics or writing')
    parser.add_argument('--check-level',choices=['metadata','payload','full'],default='payload')
    parser.add_argument('--stop-on-error',action='store_true')
    parser.add_argument('--parent-lock-owner',type=int,help=argparse.SUPPRESS)
    return parser

def main(argv=None):
    parser=evaluation_parser();args=parser.parse_args(argv)
    try:
        if not args.dry_run and args.check_level=='metadata':raise ValueError('--check-level metadata is only for --dry-run; resume/evaluation verify payload hashes')
        models=[SINGLE_MODEL] if SINGLE_MODEL else select_models(ROOT,args.model,args.start_from)
        if SINGLE_MODEL and (args.start_from or args.model and args.model!=[SINGLE_MODEL.name]):raise ValueError('Single-model entry cannot select other models')
        if (args.bundle or args.output) and len(models)!=1:raise ValueError('--bundle/--output require one model')
        if args.output and args.output_root:raise ValueError('Choose --output or --output-root')
        output_root=resolve_argument(args.output_root) if args.output_root else None
        if output_root and (not output_root.is_relative_to(ROOT) or output_root==ROOT or output_root.is_relative_to(ROOT/'Model')):raise ValueError('--output-root must be an isolated directory within project outside Model')
    except ValueError as e:parser.error(str(e))
    rows=[]
    for model in models:
        started=time.perf_counter();row={'model_id':model.name}
        try:
            with evaluation_lock(model,args.parent_lock_owner,args.dry_run):
                bundle,manifest,digest=select_current_bundle(model,args.bundle,args.check_level)
                output=resolve_argument(args.output) if args.output else (output_root/model.name/'results.json' if output_root else model/'result/results.json')
                if not output_root and (not output.is_relative_to(model/'result') or output==model/'result'):raise ValueError('Output must be a file under this model/result')
                if not output_root:
                    relative=output.relative_to(model/'result')
                    reserved={'prediction_cache','coco_bundle','native_outputs','migration_evidence'}
                    if (output.name not in {'results.json','evaluated_results.json'} or relative.parts[0] in reserved
                        or any(part.startswith('.') and not (args.parent_lock_owner and i==len(relative.parts)-2 and part.startswith('.attempt-')) for i,part in enumerate(relative.parts))):
                        raise ValueError('Custom outputs must be evaluation results in an isolated result subdirectory; cache/provenance/timing/environment files are reserved')
                if output.is_relative_to(bundle):raise ValueError('Cannot write results inside bundle')
                if any(p.is_symlink() for p in (output,*output.parents)):raise ValueError('Output path must not use symlinks')
                previous=None if args.force else reusable_result(output,model,bundle,digest)
                if previous:
                    row.update(status='reused',metrics=previous['metrics'])
                elif args.dry_run:row['status']='planned'
                else:
                    result=evaluate_bundle(bundle)
                    result=_apply_forward_timing_override(result,bundle,model/'result/results.json')
                    if sha256(bundle/'manifest.json')!=digest:raise ValueError('Bundle changed during evaluation')
                    result['bundle']=os.path.relpath(bundle,model)
                    atomic_json(output,result);row.update(status='complete',metrics=result['metrics'])
                row.update(bundle=str(bundle),output=str(output),samples=manifest['evaluated_samples'])
        except Exception as e:row.update(status='failed',error=type(e).__name__+': '+str(e))
        row['seconds']=time.perf_counter()-started;rows.append(row)
        print(json.dumps(row,ensure_ascii=False),flush=True)
        if row['status']=='failed' and args.stop_on_error:break
    failed=sum(r['status']=='failed' for r in rows)
    print(json.dumps({'selected':len(models),'processed':len(rows),'failed':failed,'results':rows},ensure_ascii=False),flush=True)
    return 1 if failed else 0

if __name__=='__main__':raise SystemExit(main())
