"""Author batch8 validation, preserving batch padding and the removed mask."""
import argparse
import collections
import fcntl
import hashlib
import importlib.metadata
import io
import json
import math
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / '_deps'))
DATA = Path('__DATASET_ROOT__/supplemental_20260905/model30/test-00000-of-00001.parquet')
WEIGHTS = Path('__MODEL_ROOT__/weights/yainage90/fashion-object-detection')
OUTPUT = Path('__MODEL_ROOT__/resources/runtime/assets/workspace/results/remaining25_supplemental_20260905/m30-author-test5544-author-batch8')
LABELS = ['bag', 'bottom', 'dress', 'hat', 'outer', 'shoes', 'top']


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def write(path, value):
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False))
    tmp.replace(path)


def finite(value):
    if isinstance(value, (int, float)):
        assert math.isfinite(value)
    elif isinstance(value, dict):
        for item in value.values():
            finite(item)
    elif isinstance(value, list):
        for item in value:
            finite(item)


def run(max_images):
    import albumentations as A
    import numpy as np
    import pyarrow.parquet as pq
    import torch
    from PIL import Image
    from transformers import AutoConfig, AutoImageProcessor, AutoModelForObjectDetection
    from transformers.image_transforms import center_to_corners_format
    from torchmetrics.detection.mean_ap import MeanAveragePrecision
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    began = time.time()
    versions = {name: importlib.metadata.version(name) for name in ['torch', 'torchvision', 'transformers', 'torchmetrics', 'albumentations', 'albucore', 'pyarrow', 'numpy', 'pycocotools', 'Pillow']}
    protocol = {'model_id': 'yainage90/fashion-object-detection', 'dataset_revision': '959476ce2fde6bd870c6578be22183114b0da131',
                'dataset_path': str(DATA), 'dataset_sha256': sha(DATA), 'weight_sha256': sha(WEIGHTS / 'model.safetensors'),
                'source_commit': '76c1dcd166ebae50bd1cad5ed1bbf3417e2fcc7c', 'runner_sha256': sha(__file__),
                'labels': LABELS, 'confidence_threshold': 0.0, 'nms': None, 'num_queries': 300, 'processor_top_k': 100, 'prediction_limit': 100,
                'device': 'cuda:1', 'dtype': 'float32', 'batch_size': 8, 'versions': versions,
                'validation_transform': 'Albumentations1.4.18 Compose([NoOp()], BboxParams(format=coco,label_fields=category,clip=True))',
                'target_processing': 'author processor labels normalized cxcywh -> center_to_corners_format -> original width/height; area from restored bbox',
                'pixel_mask': 'removed as in author train_model.ipynb', 'batch_order': 'Parquet row order, consecutive groups of8, all final rows retained', 'evaluation_coordinate_frame': 'Both predicted and target normalized boxes on padded canvas restored with each original H/W exactly as author; may differ from physical original-image pixels', 'evaluator': 'Torchmetrics1.4.1 MeanAveragePrecision(xyxy,class_metrics=True)',
                'known_differences': ['Author requirements Torch2.3.0/Transformers4.44.0; reused validated Torch2.5.1/Transformers4.50.0.'],
                'configs_sha256': {p.name: sha(p) for p in WEIGHTS.glob('*.json')}}
    fingerprint = digest(protocol)
    status_path = OUTPUT / 'status.json'
    previous = json.loads(status_path.read_text()) if status_path.exists() else {}
    if previous:
        assert previous['protocol_fingerprint'] == fingerprint, 'Protocol changed; use a new output directory'
    if max_images == 0:
        assert previous.get('smoke_passed'), 'Run --max-images 16 first'
    status = {'sequence': 30, 'model_id': protocol['model_id'], 'state': 'preparing', 'protocol': protocol,
              'protocol_fingerprint': fingerprint, 'smoke_passed': bool(previous.get('smoke_passed')), 'started_at': began, 'total_images': 5544}
    write(status_path, status)
    try:
        table = pq.read_table(DATA)
        assert table.num_rows == 5544
        metadata = json.loads(table.schema.metadata[b'huggingface'])['info']['features']
        assert metadata['objects']['feature']['category']['names'] == LABELS
        rows = table.to_pylist()
        counts = collections.Counter()
        ids = collections.Counter()
        metadata_rows = []
        for index, row in enumerate(rows):
            payload = row['image']['bytes']
            with Image.open(io.BytesIO(payload)) as image:
                image.load()
                assert image.size == (row['width'], row['height'])
            obj = row['objects']
            assert len(obj['bbox']) == len(obj['category']) == len(obj['bbox_id']) == len(obj['area'])
            finite(obj)
            for box, label in zip(obj['bbox'], obj['category']):
                assert label in range(7) and len(box) == 4 and box[2] > 0 and box[3] > 0
                counts[LABELS[label]] += 1
            ids[row['image_id']] += 1
            metadata_rows.append({'image_id': index + 1, 'author_image_id': row['image_id'], 'width': row['width'], 'height': row['height'], 'image_sha256': hashlib.sha256(payload).hexdigest()})
        assert set(counts) == set(LABELS)
        audit = {'images': len(rows), 'all_images_decoded': True, 'all_dimensions_verified': True, 'row_id_unique': True,
                 'author_image_id_collisions': sum(n - 1 for n in ids.values()), 'raw_gt_count_per_class': dict(counts),
                 'dataset_sha256': protocol['dataset_sha256'], 'schema': metadata, 'images_metadata': metadata_rows}
        write(OUTPUT / 'data_audit.json', audit)
        print('DATA_VERIFIED', len(rows), dict(counts), flush=True)
        processor = AutoImageProcessor.from_pretrained(WEIGHTS, local_files_only=True, use_fast=False)
        config = AutoConfig.from_pretrained(WEIGHTS, local_files_only=True)
        config.use_pretrained_backbone = False
        model, loading = AutoModelForObjectDetection.from_pretrained(WEIGHTS, config=config, local_files_only=True, output_loading_info=True)
        assert not any(loading.get(k) for k in ['missing_keys', 'unexpected_keys', 'mismatched_keys', 'error_msgs']), loading
        assert [model.config.id2label[i] for i in range(7)] == LABELS
        model.eval().to(device='cuda:1', dtype=torch.float32)
        torch.cuda.reset_peak_memory_stats('cuda:1')
        write(OUTPUT / 'loading_info.json', loading)
        validation = A.Compose([A.NoOp()], bbox_params=A.BboxParams(format='coco', label_fields=['category'], clip=True))
        pred_path = OUTPUT / 'predictions.jsonl'
        records = [json.loads(line) for line in pred_path.read_text().splitlines()] if pred_path.exists() else []
        existing = {r['image_id'] for r in records}
        assert len(existing) == len(records) and existing <= set(range(1, 5545))
        selected = rows if max_images == 0 else rows[:max_images]
        status.update(state='running', completed_images=len(existing))
        write(status_path, status)
        with pred_path.open('a') as stream:
            for start in range(0, len(selected), 8):
                batch_rows = selected[start:start+8]
                batch_ids = list(range(start+1, start+len(batch_rows)+1))
                if all(image_id in existing for image_id in batch_ids):
                    continue
                images, batch_annotations = [], []
                for image_id, row in zip(batch_ids, batch_rows):
                    image = Image.open(io.BytesIO(row['image']['bytes'])).convert('RGB')
                    obj = row['objects']
                    transformed = validation(image=np.array(image), bboxes=obj['bbox'], category=obj['category'])
                    annotations = [{'image_id': image_id, 'category_id': int(c), 'iscrowd': 0, 'area': area, 'bbox': list(box)}
                                   for c, area, box in zip(transformed['category'], obj['area'], transformed['bboxes'])]
                    images.append(transformed['image'])
                    batch_annotations.append({'image_id': image_id, 'annotations': annotations})
                inputs = processor(images=images, annotations=batch_annotations, return_tensors='pt')
                targets = inputs.pop('labels')
                inputs.pop('pixel_mask', None)
                padded_shape = list(inputs['pixel_values'].shape)
                with torch.inference_mode():
                    outputs = model(**{k: v.to('cuda:1') for k, v in inputs.items()})
                results = processor.post_process_object_detection(outputs, threshold=0.0, target_sizes=torch.tensor([[row['height'], row['width']] for row in batch_rows], device='cuda:1'))
                for image_id, row, target, result in zip(batch_ids, batch_rows, targets, results):
                    if image_id in existing:
                        continue
                    gt_boxes = center_to_corners_format(target['boxes']) * torch.tensor([[row['width'], row['height'], row['width'], row['height']]])
                    predictions = [{'image_id': image_id, 'category_id': int(c), 'bbox': [x1, y1, x2-x1, y2-y1], 'score': float(s)}
                                   for (x1,y1,x2,y2),s,c in zip(result['boxes'].cpu().tolist(), result['scores'].cpu().tolist(), result['labels'].cpu().tolist())]
                    target_records = [{'category_id': int(c), 'bbox': [x1,y1,x2-x1,y2-y1]} for (x1,y1,x2,y2),c in zip(gt_boxes.tolist(), target['class_labels'].tolist())]
                    assert len(predictions) == 100
                    record = {'image_id': image_id, 'author_image_id': row['image_id'], 'detections': predictions, 'targets': target_records,
                              'batch_image_ids': batch_ids, 'padded_input_shape': padded_shape, 'orig_size': target['orig_size'].tolist(), 'processor_target_size': target['size'].tolist()}
                    finite(record)
                    stream.write(json.dumps(record, allow_nan=False) + '\n')
                    stream.flush()
                    records.append(record)
                if batch_ids[-1] % 80 == 0:
                    status.update(completed_images=len(records), elapsed_seconds=time.time()-began)
                    write(status_path, status)
                    print('INFERRED', len(records), '/', len(selected), flush=True)
        assert {r['image_id'] for r in records} == set(range(1, len(selected)+1))
        annotation = {'info': {}, 'images': [{'id': i+1, 'width': r['width'], 'height': r['height']} for i,r in enumerate(selected)],
                      'categories': [{'id': i, 'name': label} for i,label in enumerate(LABELS)], 'annotations': []}
        metric = MeanAveragePrecision(box_format='xyxy', class_metrics=True)
        all_predictions = []
        for record in records:
            preds, targets = record['detections'], record['targets']
            def tensor_boxes(items):
                return torch.tensor([[x,y,x+w,y+h] for x,y,w,h in [v['bbox'] for v in items]], dtype=torch.float32).reshape(-1,4)
            metric.update([{'boxes': tensor_boxes(preds), 'scores': torch.tensor([v['score'] for v in preds]), 'labels': torch.tensor([v['category_id'] for v in preds])}],
                          [{'boxes': tensor_boxes(targets), 'labels': torch.tensor([v['category_id'] for v in targets])}])
            all_predictions.extend(preds)
            for target in targets:
                box = target['bbox']
                annotation['annotations'].append({'id':len(annotation['annotations'])+1, 'image_id':record['image_id'], 'category_id':target['category_id'], 'bbox':box, 'area':box[2]*box[3], 'iscrowd':0})
        write(OUTPUT / 'annotation.json', annotation)
        status.update(state='scoring', completed_images=len(records), annotation_sha256=sha(OUTPUT/'annotation.json'))
        write(status_path, status)
        print('SCORING', len(records), flush=True)
        raw = metric.compute()
        scores = {key: value.tolist() for key,value in raw.items()}
        scores['per_class_AP'] = {LABELS[int(c)]: float(v) for c,v in zip(raw['classes'], raw['map_per_class'])}
        gt = COCO(); gt.dataset = annotation; gt.createIndex()
        evaluation = COCOeval(gt, gt.loadRes(all_predictions), 'bbox')
        evaluation.params.iouThrs = np.array(metric.iou_thresholds)
        evaluation.params.recThrs = np.array(metric.rec_thresholds)
        evaluation.evaluate(); evaluation.accumulate(); evaluation.summarize()
        assert abs(float(evaluation.stats[0]) - scores['map']) < 1e-6
        scores.update(COCO_AP=float(evaluation.stats[0]), original_card_AP=0.7542, delta_percentage_points=(scores['map']-0.7542)*100,
                      images=len(records), unit='fraction_0_to_1', protocol_fingerprint=fingerprint, annotation_sha256=status['annotation_sha256'])
        finite(scores)
        write(OUTPUT / 'metrics.json', scores)
        status.update(state='complete' if max_images == 0 else 'smoke_complete', smoke_passed=True, completed_images=len(records),
                      elapsed_seconds=time.time()-began, peak_gpu_memory_bytes=torch.cuda.max_memory_allocated('cuda:1'),
                      predictions_sha256=sha(pred_path), metrics=scores)
        write(status_path, status)
        print('FINISHED', status['state'], scores['map'], flush=True)
    except BaseException as error:
        status.update(state='failed', error=repr(error), elapsed_seconds=time.time()-began)
        write(status_path, status)
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--max-images', type=int, choices=[0,16], required=True)
    args = parser.parse_args()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    with (OUTPUT / '.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        run(args.max_images)
