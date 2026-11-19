"""Independently check a completed supplemental run's image and class coverage."""
import argparse
import hashlib
import json
import math
from collections import Counter
from pathlib import Path


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('result', type=Path)
    parser.add_argument('--annotation', required=True, type=Path)
    parser.add_argument('--images', required=True, type=int)
    parser.add_argument('--classes', required=True, type=int)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    status_path = args.result / 'status.json'
    metrics_path = args.result / 'metrics.json'
    predictions_path = args.result / 'predictions.jsonl'
    status = json.loads(status_path.read_text())
    assert status['state'] == 'complete', status.get('state')
    metrics = json.loads(metrics_path.read_text())
    annotation = json.loads(args.annotation.read_text())
    ids = [image['id'] for image in annotation['images']]
    image_ids = set(ids)
    assert len(ids) == len(image_ids) == args.images
    category_names = {c['id']: c['name'] for c in annotation['categories']}
    assert len(category_names) == len(annotation['categories']) == args.classes
    gt_counts = Counter(a['category_id'] for a in annotation['annotations'])
    assert set(gt_counts) == set(category_names)
    ann_ids = [a['id'] for a in annotation['annotations']]
    assert len(ann_ids) == len(set(ann_ids))
    assert all(a['image_id'] in image_ids for a in annotation['annotations'])
    pred_counts = Counter()
    seen = set()
    number_predictions = 0
    for line in predictions_path.open():
        row = json.loads(line)
        assert row['image_id'] not in seen
        seen.add(row['image_id'])
        for det in row['detections']:
            assert det['image_id'] == row['image_id']
            assert det['category_id'] in category_names
            box = det['bbox']
            assert len(box) == 4 and all(math.isfinite(v) for v in box)
            assert box[2] >= 0 and box[3] >= 0
            assert math.isfinite(det['score']) and 0 <= det['score'] <= 1
            pred_counts[det['category_id']] += 1
            number_predictions += 1
    assert seen == image_ids
    def finite(value):
        if isinstance(value, dict):
            for item in value.values():
                finite(item)
        elif isinstance(value, list):
            for item in value:
                finite(item)
        elif isinstance(value, float):
            assert math.isfinite(value)
    finite(metrics)
    result = {
        'verified': True, 'new_inference_performed_by_this_check': False,
        'result_directory': str(args.result), 'images': len(seen),
        'annotations': len(ann_ids), 'predictions': number_predictions,
        'ground_truth_per_class': {category_names[k]: v for k, v in gt_counts.items()},
        'predictions_per_class': {category_names[k]: v for k, v in pred_counts.items()},
        'sha256': {str(p): sha256(p) for p in
                   [args.annotation, status_path, metrics_path, predictions_path, Path(__file__)]},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({k: result[k] for k in ['verified', 'images', 'annotations', 'predictions']}))


if __name__ == '__main__':
    main()
