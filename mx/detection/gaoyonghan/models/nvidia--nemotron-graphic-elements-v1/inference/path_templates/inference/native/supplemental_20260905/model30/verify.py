"""Read-only independent checks of the completed model30 artifacts."""
import collections
import hashlib
import json
import math
from pathlib import Path

ROOT = Path('__MODEL_ROOT__/resources/runtime/assets/workspace/results/remaining25_supplemental_20260905/m30-author-test5544')


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def finite(value):
    if isinstance(value, (int, float)):
        assert math.isfinite(value)
    elif isinstance(value, dict):
        for child in value.values():
            finite(child)
    elif isinstance(value, list):
        for child in value:
            finite(child)


status = json.loads((ROOT / 'status.json').read_text())
scores = json.loads((ROOT / 'metrics.json').read_text())
audit = json.loads((ROOT / 'data_audit.json').read_text())
annotation = json.loads((ROOT / 'annotation.json').read_text())
assert status['state'] == 'complete' and status['completed_images'] == scores['images'] == 5544
assert status['protocol_fingerprint'] == scores['protocol_fingerprint']
assert status['annotation_sha256'] == scores['annotation_sha256'] == sha(ROOT / 'annotation.json')
assert status['predictions_sha256'] == sha(ROOT / 'predictions.jsonl')
assert status['protocol']['dataset_sha256'] == sha(Path(status['protocol']['dataset_path']))
finite(scores)
ids = set()
gt_count = collections.Counter()
prediction_count = 0
gt_by_image = collections.defaultdict(list)
for item in annotation['annotations']:
    gt_by_image[item['image_id']].append({key: item[key] for key in ['category_id', 'bbox']})
with (ROOT / 'predictions.jsonl').open() as stream:
    for line in stream:
        row = json.loads(line)
        finite(row)
        image_id = row['image_id']
        assert image_id not in ids
        ids.add(image_id)
        assert row['targets'] == gt_by_image[image_id]
        for detection in row['detections']:
            assert detection['image_id'] == image_id and detection['category_id'] in range(7)
            assert detection['bbox'][2] >= 0 and detection['bbox'][3] >= 0
            assert 0 <= detection['score'] <= 1
        for target in row['targets']:
            assert target['bbox'][2] > 0 and target['bbox'][3] > 0
            gt_count[target['category_id']] += 1
        prediction_count += len(row['detections'])
assert ids == set(range(1, 5545))
assert set(gt_count) == set(range(7))
assert sum(gt_count.values()) == len(annotation['annotations'])
assert abs(scores['map'] - scores['COCO_AP']) < 1e-6
image_hashes = collections.Counter(row['image_sha256'] for row in audit['images_metadata'])
result = {'passed': True, 'images': len(ids), 'predictions': prediction_count, 'ground_truth_boxes': sum(gt_count.values()),
          'gt_per_class': {status['protocol']['labels'][key]: value for key,value in sorted(gt_count.items())},
          'duplicate_image_bytes': sum(n-1 for n in image_hashes.values()), 'author_image_id_collisions_safely_reindexed': audit['author_image_id_collisions'],
          'all_prediction_coordinates_finite': True, 'all_image_ids_unique': True, 'all_seven_classes_present': True,
          'Torchmetrics_vs_COCO_AP_absolute_error': abs(scores['map']-scores['COCO_AP']),
          'artifact_sha256': {name: sha(ROOT / name) for name in ['status.json','metrics.json','annotation.json','predictions.jsonl','data_audit.json','loading_info.json']}}
(ROOT / 'verification.json').write_text(json.dumps(result, ensure_ascii=False, indent=2))
print(json.dumps(result, ensure_ascii=False, indent=2))
