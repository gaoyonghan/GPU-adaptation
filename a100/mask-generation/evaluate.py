#!/usr/bin/env python3
"""Mask-only evaluation: predictions plus ground truth -> {"miou": fraction}.

Binary IoU = intersection / union; two empty masks have IoU 1. Text-conditioned
instance sets use maximum-IoU one-to-one Hungarian matching, with unmatched
predictions and labels scoring zero. Each query contributes its mean matched
IoU. Video frames are averaged per tracked object, then across objects (J).
Different datasets/prompts still do not make cross-model scores comparable.
No model weights are loaded and no inference is performed by this module.
"""
from __future__ import annotations
import sys
sys.dont_write_bytecode = True
import argparse
import csv
import io
import json
import math
import os
import tempfile
from collections import defaultdict
from pathlib import Path


def save_result(output, value):
    if not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError("mIoU must be a finite fraction in [0, 1]")
    if output is None:
        return
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=output.parent,
                                         prefix=".miou-", suffix=".tmp", delete=False) as stream:
            temporary = Path(stream.name)
            json.dump({"miou": value}, stream, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(output)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def binary_iou(prediction, truth):
    import numpy as np
    prediction, truth = np.asarray(prediction, dtype=bool), np.asarray(truth, dtype=bool)
    if prediction.shape != truth.shape or not prediction.size:
        raise ValueError(f"Mask shape mismatch/empty arrays: {prediction.shape}, {truth.shape}")
    union = np.count_nonzero(prediction | truth)
    return float(np.count_nonzero(prediction & truth) / union) if union else 1.0


def canonical_rle(rle):
    from pycocotools import mask as masks
    if isinstance(rle, str):
        rle = json.loads(rle)
    rle = dict(rle)
    h, w = rle["size"]
    if isinstance(rle["counts"], list):
        rle = masks.frPyObjects(rle, h, w)
    elif isinstance(rle["counts"], str):
        rle["counts"] = rle["counts"].encode("ascii")
    return rle


def instance_set_iou(predictions, truths):
    """One-to-one matching, with zero for every unmatched instance."""
    import numpy as np
    from pycocotools import mask as masks
    from scipy.optimize import linear_sum_assignment
    if not predictions and not truths:
        return 1.0
    if not predictions or not truths:
        return 0.0
    p, g = list(map(canonical_rle, predictions)), list(map(canonical_rle, truths))
    shapes = {tuple(x["size"]) for x in p + g}
    if len(shapes) != 1:
        raise ValueError("Instance masks must have the same image shape")
    values = masks.iou(p, g, [0] * len(g))
    rows, columns = linear_sum_assignment(-values)
    return float(np.sum(values[rows, columns]) / max(len(p), len(g)))


def rle_on_image_grid(rle, height, width):
    """Align a mask to the documented image grid using nearest-neighbor only.

Gold contains 84 released annotations at another proportional resolution.
Never compare RLEs with unequal grids (COCO otherwise returns negative IoUs).
"""
    from pycocotools import mask as masks
    import numpy as np
    from PIL import Image
    rle = canonical_rle(rle)
    if list(rle["size"]) == [height, width]:
        return rle
    old_h, old_w = rle["size"]
    if abs(old_w / old_h - width / height) > 0.02:
        raise ValueError("Annotation aspect ratio does not match the image")
    image = Image.fromarray(masks.decode(rle).astype(np.uint8))
    array = np.asarray(image.resize((width, height), Image.Resampling.NEAREST))
    return masks.encode(np.asfortranarray(array, dtype=np.uint8))


def evaluate(records, output_path=None):
    """Consume matched prediction/GT records; reject duplicate sample IDs.

Each record contains sample_id, prediction, truth. A single mask pair may be
NumPy arrays or COCO RLE dictionaries; lists denote instance sets. Optional
group denotes a video object (average frames first). Dataset readers own the
complete membership checks; no sample is silently skipped here.
"""
    from pycocotools import mask as masks
    seen, groups = set(), defaultdict(lambda: [0.0, 0])
    for record in records:
        key = str(record["sample_id"])
        if key in seen:
            raise ValueError(f"Duplicate prediction/GT pair: {key}")
        seen.add(key)
        p, g = record["prediction"], record["truth"]
        if isinstance(p, list) and isinstance(g, list):
            value = instance_set_iou(p, g)
        else:
            if isinstance(p, dict):
                p = masks.decode(canonical_rle(p))
            if isinstance(g, dict):
                g = masks.decode(canonical_rle(g))
            value = binary_iou(p, g)
        group = str(record.get("group", key))
        groups[group][0] += value
        groups[group][1] += 1
    if not groups:
        raise ValueError("No ground-truth/prediction pairs were evaluated")
    value = math.fsum(total / count for total, count in groups.values()) / len(groups)
    save_result(output_path, value)
    return {"miou": value, "units": len(groups), "mask_pairs": len(seen)}


def jsonl_records(path, expected_count=None):
    count = 0
    with Path(path).open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                count += 1
                yield json.loads(line)
    if expected_count is not None and count != expected_count:
        raise ValueError(f"Incomplete predictions: {count}/{expected_count}")


def point_records(path, dataset_root, dataset_name, expected_count):
    """Match saved predictions against the original annotations, not confidences."""
    dataset_root = Path(dataset_root)
    coco = None
    if dataset_name == "COCO-2017":
        from pycocotools.coco import COCO
        coco = COCO(str(dataset_root / "annotations/lvis_v1_val.json"))
    last_path, annotation = None, None
    for record in jsonl_records(path, expected_count):
        original = record["original"]
        if coco is not None:
            ann = coco.anns[int(original["annotation_id"])]
            truth = coco.annToRLE(ann)
            image_info = coco.imgs[ann['image_id']]
            from urllib.parse import urlparse
            image_url = Path(urlparse(image_info['coco_url']).path)
            source_image = dataset_root / image_url.parent.name / image_url.name
        else:
            filename = dataset_root / original["annotation_file"]
            if filename != last_path:
                annotation = json.loads(filename.read_text())
                last_path = filename
            truth = annotation["annotations"][int(original["annotation_index"])]["segmentation"]
            source_image = filename.with_suffix('.jpg')
        yield {"sample_id": record["sample_id"], "prediction": record["prediction"], "truth": truth,
               "image": source_image, "metadata": {"dataset": dataset_name, "original": original}}


def image_value(value):
    from PIL import Image
    if isinstance(value, dict):
        return Image.open(io.BytesIO(value["bytes"])) if value.get("bytes") is not None else Image.open(value["path"])
    return Image.open(io.BytesIO(value))


def finegrain_records(predictions, dataset_root, limit=0):
    import numpy as np
    import pyarrow.parquet as pq
    from PIL import Image
    data = Path(dataset_root) / "data/test-00000-of-00001.parquet"
    index = 0
    for batch in pq.ParquetFile(data).iter_batches(batch_size=8, columns=["image", "mask"]):
        for row in batch.to_pylist():
            value = row['mask']
            if limit and index >= limit:
                return
            with image_value(value) as gt:
                truth = np.asarray(gt.convert("L")) >= 128
            with Image.open(Path(predictions) / f"{index:05d}.png") as pred:
                prediction = np.asarray(pred.convert("L")) >= 128
            yield {"sample_id": str(index), "prediction": prediction, "truth": truth,
                   "image": row['image'], "metadata": {"dataset": "finegrain-product-masks-lite"}}
            index += 1


def medical_records(predictions, dataset_root, case_names):
    import numpy as np
    import SimpleITK as sitk
    root = Path(dataset_root) / "FLARE22Train"
    for case in case_names:
        truth_itk = sitk.ReadImage(str(root / "labels" / case))
        pred_itk = sitk.ReadImage(str(Path(predictions) / case))
        if truth_itk.GetSize() != pred_itk.GetSize() or truth_itk.GetSpacing() != pred_itk.GetSpacing():
            raise ValueError(f"Mismatched medical image geometry: {case}")
        truth, prediction = sitk.GetArrayFromImage(truth_itk), sitk.GetArrayFromImage(pred_itk)
        for label in sorted(set(np.unique(truth)) | set(np.unique(prediction))):
            if label == 0:
                continue
            yield {"sample_id": f"{case}:{label}", "prediction": prediction == label, "truth": truth == label,
                   "volume_source": root / 'images' / case.replace('.nii.gz', '_0000.nii.gz'),
                   "metadata": {"case_id": case, "organ_id": int(label), "spacing": list(truth_itk.GetSpacing()),
                                "origin": list(truth_itk.GetOrigin()), "direction": list(truth_itk.GetDirection()),
                                "aggregation": "3D-organ-case-IoU"}}


def video_records(predictions, gt_root, videos, per_object=False):
    """Full-GT VOS, skip first/last annotated frame; never score only init GT."""
    import numpy as np
    from PIL import Image
    from tqdm.auto import tqdm
    predictions, gt_root = Path(predictions), Path(gt_root)
    for video in tqdm(videos, desc="Video mIoU", unit="video"):
        folder = gt_root / video
        if per_object:
            objects = sorted(p for p in folder.iterdir() if p.is_dir())
            for obj in objects:
                frames = sorted(obj.glob("*.png"))
                if len(frames) < 3:
                    raise ValueError(f"Full temporal GT is unavailable for {video}/{obj.name}")
                for frame in frames[1:-1]:
                    with Image.open(frame) as im:
                        truth = np.asarray(im) > 0
                    with Image.open(predictions / video / obj.name / frame.name) as im:
                        prediction = np.asarray(im) > 0
                    yield {"sample_id": f"{video}/{obj.name}/{frame.name}", "group": f"{video}/{obj.name}",
                           "prediction": prediction, "truth": truth,
                           "image": gt_root.parent / 'JPEGImages' / video / (frame.stem + '.jpg'),
                           "metadata": {"video_id": video, "object_id": obj.name, "frame": frame.stem,
                                        "ignore_first_last": True}}
        else:
            frames = sorted(folder.glob("*.png"))
            if len(frames) < 3:
                raise ValueError(f"{video}: only initialization GT is public; import an official server J score or use full-GT data")
            object_ids = set()
            for frame in frames:
                with Image.open(frame) as im:
                    object_ids.update(int(x) for x in np.unique(np.asarray(im)) if x not in (0, 255))
            for frame in frames[1:-1]:
                with Image.open(frame) as im:
                    truth = np.asarray(im)
                with Image.open(predictions / video / frame.name) as im:
                    prediction = np.asarray(im)
                if prediction.shape != truth.shape:
                    raise ValueError(f"Frame shape mismatch: {video}/{frame.name}")
                valid = truth != 255
                extra = set(int(x) for x in np.unique(prediction[valid])) - object_ids - {0}
                if extra:
                    raise ValueError(f"Unexpected predicted tracking IDs: {extra}")
                for obj in sorted(object_ids):
                    yield {"sample_id": f"{video}/{obj}/{frame.name}", "group": f"{video}/{obj}",
                           "prediction": (prediction == obj) & valid, "truth": (truth == obj) & valid,
                           "image": gt_root.parent.parent / 'JPEGImages' / gt_root.name / video / (frame.stem + '.jpg'),
                           "metadata": {"video_id": video, "object_id": obj, "frame": frame.stem,
                                        "ignore_label": 255, "ignore_first_last": True}}


GOLD_SUBSETS = {
    "metaclip_nps": "metaclip", "sa1b_nps": "sa1b", "attributes": "attributes",
    "crowded": "crowded", "fg_food": "fg_food", "fg_sports": "fg_sports_equipment", "wiki_common": "wiki_common",
}


def gold_image_path(data, filename):
    # Dataset contains original metaclip/sa1b images. Names in Gold annotations
    # can repeat across queries but must resolve to exactly one physical image.
    from functools import lru_cache
    return _gold_image_index(str(Path(data).resolve()))[Path(filename).name]


from functools import lru_cache
@lru_cache(maxsize=2)
def _gold_image_index(data):
    index = {}
    for file in Path(data).rglob('*'):
        if file.is_file() and file.suffix.lower() in {'.jpg', '.jpeg', '.png', '.webp'}:
            if file.name in index: raise ValueError('Ambiguous original Gold image filename')
            index[file.name] = file
    return index


def captured_falcon_records(path, expected):
    from PIL import Image
    for record in jsonl_records(path, expected):
        with Image.open(Path(path).parent / record.pop('source_image')) as image:
            record['image'] = image.copy()
        yield record


def gold_records(prediction_root, dataset_root, score_threshold=0.3, limit=0):
    """Use all three released annotator sets; macro-average the seven subsets."""
    from tqdm.auto import tqdm
    predictions, data = Path(prediction_root), Path(dataset_root)
    for subset, gt_prefix in GOLD_SUBSETS.items():
        path = predictions / f"gold_{subset}" / "dumps" / f"gold_{subset}" / "coco_predictions_segm.json"
        if not path.exists() and subset == "fg_sports":
            path = predictions / "gold_fg_sports_equipment/dumps/gold_fg_sports_equipment/coco_predictions_segm.json"
        with path.open() as stream:
            raw = json.load(stream)
        by_image = defaultdict(list)
        for pred in raw:
            if float(pred.get("score", 1)) >= score_threshold:
                by_image[pred["image_id"]].append(pred["segmentation"])
        for annotator in ("a", "b", "c"):
            gt = json.loads((data / "gt-annotations" / f"gold_{gt_prefix}_merged_{annotator}_release_test.json").read_text())
            unknown = set(by_image) - {image["id"] for image in gt["images"]}
            if unknown:
                raise ValueError(f"Prediction query IDs not in Gold {subset}/{annotator}: {len(unknown)}")
            anns = defaultdict(list)
            for ann in gt["annotations"]:
                if not ann.get("iscrowd", False):
                    anns[ann["image_id"]].append(ann["segmentation"])
            images = gt["images"][:limit or None]
            for image in tqdm(images, desc=f"Gold {subset}/{annotator}", unit="query"):
                image_id = image["id"]
                h, w = image["height"], image["width"]
                yield {"sample_id": f"{subset}/{annotator}/{image_id}", "group": subset,
                       "prediction": [rle_on_image_grid(x, h, w) for x in by_image[image_id]],
                       "truth": [rle_on_image_grid(x, h, w) for x in anns[image_id]],
                       "image": gold_image_path(data, image['file_name']),
                       "metadata": {"query": image.get('text_input', ''), "subset": subset,
                                    "annotator": annotator, "score_threshold": score_threshold}}


# Appended to the existing task-independent metric/data-reader functions.
SCHEMA = 'model-eval/mask-coco/v1'
PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _save_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix='.' + path.name, dir=path.parent)
    try:
        with os.fdopen(fd, 'w') as stream:
            json.dump(value, stream, ensure_ascii=False, allow_nan=False, indent=2)
            stream.write('\n'); stream.flush(); os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary): os.unlink(temporary)

def _sha(path):
    import hashlib
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''): h.update(block)
    return h.hexdigest()


def _relative_inside(root, value):
    if not isinstance(value, str) or Path(value).is_absolute():
        raise ValueError('Expected a relative artifact path')
    path = (root / value).resolve()
    if not path.is_relative_to(root.resolve()): raise ValueError('Artifact escapes its run directory')
    return path


def resolve_run(value):
    path = Path(value)
    if not path.is_absolute(): path = PROJECT_ROOT / path
    if path.is_file():
        if path.name != 'protocol.json': raise ValueError('Expected protocol.json or an evaluation package directory')
        return path.parent.resolve()
    if (path / 'protocol.json').is_file(): return path.resolve()
    marker = json.loads((path / 'inference_output.json').read_text())
    if marker.get('state') != 'completed': raise ValueError('Evaluation package is not complete')
    return _relative_inside(path, marker['run_directory'])


def read_coco_bundle(run_dir):
    import time
    run = resolve_run(run_dir)
    protocol = json.loads((run / 'protocol.json').read_text())
    if protocol.get('schema') != SCHEMA or protocol.get('task') != 'mask_generation' or protocol.get('complete') is not True:
        raise ValueError('Wrong or incomplete mask COCO package')
    if protocol.get('path_base') != 'project_root': raise ValueError('Unsupported media path base')
    expected_package = protocol['evaluation']['dataset'] + '_' + protocol['model'] + '_coco'
    if run.parent.name != 'runs' or run.parent.parent.name != expected_package or run.name != protocol['run_id']:
        raise ValueError('Evaluation package/model/run directory mismatch')
    if type(protocol.get('sample_count')) is not int or protocol['sample_count'] <= 0:
        raise ValueError('Invalid sample count')
    model = _relative_inside(PROJECT_ROOT, protocol['model_directory'])
    if not model.is_relative_to(PROJECT_ROOT / 'mask-generation') or model.name != protocol['model']:
        raise ValueError('Model identity/path mismatch')
    dataset_value = os.environ.get('MASK_DATASET_ROOT', protocol['dataset_root'])
    data_root = Path(dataset_value)
    if not data_root.is_absolute(): data_root = PROJECT_ROOT / data_root
    data_root = data_root.resolve()
    if not run.is_relative_to(data_root): raise ValueError('Evaluation package is outside the configured datasets')
    if not protocol.get('shards'): raise ValueError('Empty COCO package')
    def records():
        import numpy as np
        seen, media_seen, total = set(), set(), 0
        last = time.monotonic()
        for shard in protocol['shards']:
            gt_path = _relative_inside(run, shard['annotations'])
            pred_path = _relative_inside(run, shard['predictions'])
            if _sha(gt_path) != shard['annotations_sha256'] or _sha(pred_path) != shard['predictions_sha256']:
                raise ValueError('Evaluation shard checksum mismatch')
            gt, preds = json.loads(gt_path.read_text()), json.loads(pred_path.read_text())
            images = {x['id']: x for x in gt['images']}
            units = {x['id']: x for x in gt['evaluation_units']}
            if len(images) != len(gt['images']) or len(units) != len(gt['evaluation_units']):
                raise ValueError('Duplicate COCO image/unit IDs')
            if len(units) != shard['sample_count']: raise ValueError('Shard sample count mismatch')
            targets, outputs = {}, {}
            for collection, index in ((gt['annotations'], targets), (preds, outputs)):
                for ann in collection:
                    if ann['image_id'] not in images or ann['eval_id'] not in units:
                        raise ValueError('Unknown annotation reference')
                    if ann['image_id'] not in units[ann['eval_id']]['image_ids']:
                        raise ValueError('Wrong query/instance match')
                    image = images[ann['image_id']]
                    if ann['segmentation']['size'] != [image['height'], image['width']]:
                        raise ValueError('Mask/image grid mismatch')
                    index.setdefault((ann['eval_id'], ann['image_id']), []).append(ann)
            for unit in gt['evaluation_units']:
                identity = (type(unit['sample_id']).__name__, unit['sample_id'])
                if identity in seen: raise ValueError('Duplicate evaluated sample ID')
                seen.add(identity); total += 1
                g, p = [], []
                for image_id in unit['image_ids']:
                    image = images[image_id]; media = image['file_name']
                    if not isinstance(media, str) or Path(media).is_absolute(): raise ValueError('Nonportable media path')
                    if media not in media_seen:
                        # Metrics use the masks in JSON, not image pixels. Validate
                        # the reference without reopening one image for every prompt.
                        if not (PROJECT_ROOT / media).resolve().is_relative_to(data_root):
                            raise ValueError('Media reference escapes the configured dataset root')
                        media_seen.add(media)
                    ga = targets.get((unit['id'], image_id), [])
                    pa = outputs.get((unit['id'], image_id), [])
                    if unit['kind'] in ('volume', 'mask_pair') and (len(ga) != 1 or len(pa) != 1):
                        raise ValueError('Missing paired prediction/ground truth')
                    g.extend(x['segmentation'] for x in ga); p.extend(x['segmentation'] for x in pa)
                if unit['kind'] == 'mask_pair':
                    if len(g) != 1 or len(p) != 1: raise ValueError('Expected one mask pair')
                    g, p = g[0], p[0]
                elif unit['kind'] == 'volume':
                    from pycocotools import mask as masks
                    g = np.stack([masks.decode(canonical_rle(x)) for x in g])
                    p = np.stack([masks.decode(canonical_rle(x)) for x in p])
                    if list(g.shape) != unit['volume_shape'] or p.shape != g.shape:
                        raise ValueError('Volume grid mismatch')
                elif unit['kind'] != 'instance_set': raise ValueError('Unknown mask evaluation unit')
                yield {'sample_id': unit['sample_id'], 'group': unit['group'], 'prediction': p, 'truth': g}
            if time.monotonic() - last >= 5:
                print(f'COCO scoring: {total}/{protocol["sample_count"]} units', flush=True)
                last = time.monotonic()
        if total != protocol['sample_count']: raise ValueError('Incomplete prediction bundle')
        print(f'COCO scoring: {total}/{protocol["sample_count"]} units', flush=True)
    return run, protocol, model, records()


def evaluate_coco(run_dir, output_path=None):
    run, protocol, model, records = read_coco_bundle(run_dir)
    custom = os.environ.get('MASK_RESULT_ROOT')
    base = Path(custom) if custom else model / 'result'
    if custom:
        if not base.is_absolute(): base = PROJECT_ROOT / base
        base = base / model.name
    base = base.resolve()
    limited = bool(protocol['evaluation'].get('sample_limit', 0))
    if output_path is None: destination = base / ('smoke_results.json' if limited else 'results.json')
    else:
        destination = Path(output_path)
        if not destination.is_absolute(): destination = PROJECT_ROOT / destination
        destination = destination.resolve()
    if destination.is_relative_to((PROJECT_ROOT / 'dataset').resolve()):
        raise ValueError('Result output must not overwrite dataset or evaluation-package files')
    if not destination.is_relative_to(base) or destination.suffix != '.json':
        raise ValueError('Metric JSON must be saved in this model result directory')
    if limited and destination == base / 'results.json':
        raise ValueError('Limited evaluation cannot overwrite full results')
    result = evaluate(records, destination)
    _save_json(run / 'metrics.json', result)
    model_run = base / 'runs' / protocol['run_id']
    model_run.mkdir(parents=True, exist_ok=True)
    save_result(model_run / 'results.json', result['miou'])
    reference = {'schema': SCHEMA, 'model': model.name, 'complete': True,
                 'bundle': os.path.relpath(run, base), 'sample_count': protocol['sample_count']}
    if not limited and destination == base / 'results.json':
        _save_json(base / 'latest.json', reference)
        package = run.parent.parent
        _save_json(package / 'inference_output.json', {
            'state': 'completed', 'task': 'mask_generation', 'model': model.name,
            'run_directory': os.path.relpath(run, package)})
    return result


def main():
    parser = argparse.ArgumentParser(description='Shared mask mIoU evaluation; no model weights/inference required.')
    parser.add_argument('--run-dir', '--bundle', dest='run_dir', required=True, type=Path,
                        help='Relative project path to a COCO package, its runs/<ID>/, or protocol.json')
    parser.add_argument('--output', type=Path, help='Optional result JSON path relative to the project')
    args = parser.parse_args()
    print(json.dumps(evaluate_coco(args.run_dir, args.output), ensure_ascii=False))


if __name__ == '__main__': main()
