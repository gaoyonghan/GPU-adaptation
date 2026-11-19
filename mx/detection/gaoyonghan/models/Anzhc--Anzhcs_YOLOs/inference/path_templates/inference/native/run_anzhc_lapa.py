"""LaPa test replacement benchmark; not the author's private face benchmark.

Use the existing object-detection-eval environment. No training or downloads.
GT = union of LaPa labels 1..9, one annotated target face per image; bbox
is derived from this mask. This does not establish multi-face/anime coverage.
Sources: https://github.com/jd-opensource/lapa-dataset
         https://huggingface.co/Anzhc/Anzhcs_YOLOs
"""
import argparse
import fcntl
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image
from pycocotools import mask as masks
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

DATA = Path('__DATASET_ROOT__/lapa_candidate/LaPa/test')
WEIGHT = Path('__MODEL_ROOT__/weights/Anzhc/Anzhcs_YOLOs/Anzhc Face seg 640 v4 y11n.pt')
OUTPUT = Path('__OUTPUT_ROOT__/m008')


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def save(path, value):
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False))
    tmp.replace(path)


def encode(mask):
    rle = masks.encode(np.asfortranarray(mask, dtype=np.uint8))
    rle['counts'] = rle['counts'].decode('ascii')
    return rle


def ground_truth():
    images = sorted((DATA / 'images').glob('*.jpg'))
    labels = sorted((DATA / 'labels').glob('*.png'))
    if len(images) != 2000 or {p.stem for p in images} != {p.stem for p in labels}:
        raise ValueError('Expected exactly 2000 paired LaPa test images/labels')
    gt = {'images': [], 'annotations': [], 'categories': [{'id': 1, 'name': 'face'}],
          'info': {'source': 'LaPa official test', 'relation': 'public_replacement'}, 'licenses': []}
    for index, path in enumerate(images, 1):
        label = DATA / 'labels' / (path.stem + '.png')
        with Image.open(path) as image, Image.open(label) as label_image:
            arr = np.array(label_image)
            w, h = image.size
        if arr.shape != (h, w) or not np.isin(arr, np.arange(11)).all():
            raise ValueError(f'Invalid label geometry/values: {label}')
        rle = encode((arr >= 1) & (arr <= 9))
        area = float(masks.area(rle))
        if area == 0:
            raise ValueError(f'Empty face: {label}')
        gt['images'].append({'id': index, 'file_name': str(path), 'width': w, 'height': h,
                             'source_dataset': 'LaPa', 'source_split': 'test',
                             'source_image': path.name, 'source_label': str(label),
                             'image_sha256': sha(path), 'label_sha256': sha(label)})
        gt['annotations'].append({'id': index, 'image_id': index, 'category_id': 1,
                                  'segmentation': rle, 'bbox': masks.toBbox(rle).tolist(),
                                  'area': area, 'iscrowd': 0})
    return gt


def score(gt_data, predictions, kind):
    gt = COCO()
    gt.dataset = gt_data
    gt.createIndex()
    if predictions:
        # Omitting bbox is essential for mask-area calculation in COCO.loadRes.
        rows = [{k: v for k, v in p.items() if k != ('bbox' if kind == 'segm' else 'segmentation')}
                for p in predictions]
        dt = gt.loadRes(rows)
    else:
        dt = COCO()
        dt.dataset = {**gt_data, 'annotations': []}
        dt.createIndex()
    evaluator = COCOeval(gt, dt, kind)
    evaluator.evaluate()
    evaluator.accumulate()
    evaluator.summarize()
    return {k: float(v) if v >= 0 else None for k, v in zip(
        ['AP', 'AP50', 'AP75', 'AP_small', 'AP_medium', 'AP_large',
         'AR1', 'AR10', 'AR100', 'AR_small', 'AR_medium', 'AR_large'], evaluator.stats)}


def run(args, out):
    import torch
    import ultralytics
    from ultralytics import YOLO
    import importlib.metadata

    print('Checking 2000 test image/label pairs and hashes...', flush=True)
    gt = ground_truth()
    if args.max_images:
        gt['images'] = gt['images'][:args.max_images]
        gt['annotations'] = gt['annotations'][:args.max_images]
    protocol = {'model': str(WEIGHT), 'weight_sha256': sha(WEIGHT), 'script_sha256': sha(Path(__file__)),
                'dataset': 'LaPa', 'split': 'test', 'image_count': len(gt['images']),
                'gt_sha256': hashlib.sha256(json.dumps(gt, sort_keys=True).encode()).hexdigest(),
                'face_labels': list(range(1, 10)), 'excluded_labels': [0, 10],
                'bbox_source': 'tight external bbox of merged GT mask',
                'imgsz': 640, 'conf': .001, 'iou': .7, 'max_det': 300, 'retina_masks': True,
                'half': False, 'augment': False, 'rect': True, 'device': args.device,
                'torch': torch.__version__, 'ultralytics': ultralytics.__version__,
                'numpy': np.__version__, 'pycocotools': importlib.metadata.version('pycocotools'),
                'evaluation': 'COCOeval bbox/segm; maxDets=[1,10,100]; IoU=.50:.05:.95',
                'relation': 'public_replacement', 'official_comparable': False,
                'limitations': ['Real target-face parsing images, not anime or multi-face benchmark.',
                                'Labels 1..9 union is our frozen mapping, not verified private GT boundary.',
                                'Native Ultralytics inference, custom GT/export adapter, COCO scoring.',
                                'Other unannotated faces may count as false positives; inspect samples.',
                                'No claim of private training-data independence or official metric reproduction.']}
    fingerprint = hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest()
    status_path = out / 'status.json'
    if status_path.exists():
        previous = json.loads(status_path.read_text())
        if previous['fingerprint'] != fingerprint:
            raise ValueError('Protocol/data changed: use a new output directory; old results preserved')
        if previous['state'] == 'complete' and (out / 'metrics.json').exists():
            print(f'Already complete: {out}', flush=True)
            return
    status = {'state': 'running', 'fingerprint': fingerprint, 'protocol': protocol,
              'all_model_tasks_complete': False}
    save(status_path, status)
    try:
        save(out / 'annotations.coco.json', gt)
        predictions_path = out / 'predictions.jsonl'
        records = []
        if predictions_path.exists():
            records = [json.loads(line) for line in predictions_path.read_text().splitlines()]
        done = {r['image_id'] for r in records}
        expected = {r['id'] for r in gt['images']}
        if len(done) != len(records) or not done <= expected:
            raise ValueError('Duplicate or unexpected prediction image IDs')
        if done != expected:
            model = YOLO(str(WEIGHT))
            if model.task != 'segment' or len(model.names) != 1 or model.names[0].lower() != 'face':
                raise ValueError(f'Unexpected task/classes: {model.task}, {model.names}')
            with predictions_path.open('a') as stream:
                for item in gt['images']:
                    if item['id'] in done:
                        continue
                    result = model.predict(item['file_name'], imgsz=640, device=args.device,
                                           conf=.001, iou=.7, max_det=300, half=False,
                                           augment=False, rect=True, retina_masks=True,
                                           save=False, verbose=False)[0]
                    rows = []
                    if len(result.boxes):
                        if result.masks is None or len(result.masks) != len(result.boxes):
                            raise ValueError('Missing/mismatched predicted masks')
                        for box, confidence, mask in zip(result.boxes.xyxy.cpu().numpy(),
                                                         result.boxes.conf.cpu().numpy(),
                                                         result.masks.data.cpu().numpy()):
                            if mask.shape != (item['height'], item['width']):
                                raise ValueError('Predicted mask not in original image coordinates')
                            if not np.isfinite(box).all() or not np.isfinite(confidence):
                                raise ValueError('Non-finite prediction')
                            x1, y1, x2, y2 = map(float, box)
                            rows.append({'image_id': item['id'], 'category_id': 1,
                                         'score': float(confidence), 'bbox': [x1, y1, x2-x1, y2-y1],
                                         'segmentation': encode(mask > .5)})
                    record = {'image_id': item['id'], 'detections': rows}
                    stream.write(json.dumps(record, allow_nan=False) + '\n')
                    stream.flush()
                    records.append(record)
                    print(f"LaPa face {len(records)}/{len(expected)}", flush=True)
        predictions = [p for r in records for p in r['detections']]
        result = {'bbox': score(gt, predictions, 'bbox'), 'segm': score(gt, predictions, 'segm'),
                  'unit': 'fraction_0_to_1', 'images': len(records),
                  'scope': 'smoke' if args.max_images else 'full_test',
                  'official_comparable': False, 'boundary_audit': 'pending',
                  'all_model_tasks_complete': False}
        save(out / 'metrics.json', result)
        status.update(state='complete', completed_images=len(records))
        print(f'Completed replacement evaluation (not official reproduction): {out}', flush=True)
    except BaseException as exc:
        status.update(state='failed', error=repr(exc))
        raise
    finally:
        save(status_path, status)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--max-images', type=int, default=0, help='0: full test; positive: isolated smoke result')
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    if not 0 <= args.max_images <= 2000:
        parser.error('--max-images must be between 0 and 2000')
    out = OUTPUT / (f'smoke_{args.max_images}' if args.max_images else 'full')
    out.mkdir(parents=True, exist_ok=True)
    with (out / '.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        run(args, out)


if __name__ == '__main__':
    main()
