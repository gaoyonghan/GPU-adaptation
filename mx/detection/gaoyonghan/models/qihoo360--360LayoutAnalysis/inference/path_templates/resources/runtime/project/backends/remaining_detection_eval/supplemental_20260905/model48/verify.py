"""Independent artifact checks for the completed val560 evaluation."""
import collections
import hashlib
import json
import math
from pathlib import Path

DATA = Path('__DATASET_ROOT__/supplemental_20260905/model48')
RESULT = Path('__MODEL_ROOT__/resources/runtime/assets/workspace/results/remaining25_supplemental_20260905/m48-cached-val560-native10')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def finite(value):
    if isinstance(value, dict):
        for v in value.values(): finite(v)
    elif isinstance(value, list):
        for v in value: finite(v)
    elif isinstance(value, (int, float)):
        assert math.isfinite(value)


def main():
    status = json.loads((RESULT / 'status.json').read_text())
    assert status['state'] == 'complete' and status['completed_images'] == 560
    manifest = json.loads((DATA / 'source_manifest.json').read_text())
    ann = json.loads((RESULT / 'annotation.json').read_text())
    original = json.loads((DATA / 'original_CACHED_val.json').read_text())
    assert sha(RESULT/'annotation.json') == sha(DATA/'annotations_native10.json') == manifest['annotation_sha256']
    assert ann['images'] == original['images'] and len(ann['images']) == 560
    original_cats = {c['id']: c['name'] for c in original['categories']}
    original_anns = {a['id']: a for a in original['annotations']}
    cats = {c['id']: c['name'] for c in ann['categories']}
    counts = collections.Counter()
    for a in ann['annotations']:
        old = original_anns[a['id']]
        assert original_cats[old['category_id']] == cats[a['category_id']]
        assert {k:v for k,v in a.items() if k!='category_id'} == {k:v for k,v in old.items() if k!='category_id'}
        counts[cats[a['category_id']]] += 1
    assert dict(counts) == manifest['native10_counts'] and sum(counts.values()) == 11520
    for info in manifest['image_checks']:
        assert sha(DATA/'images'/info['file_name']) == info['sha256']
    records = [json.loads(line) for line in (RESULT/'predictions.jsonl').open()]
    image_ids = {r['image_id'] for r in records}; images = {i['id']: i for i in ann['images']}
    assert len(records) == len(image_ids) == 560 and image_ids == set(images)
    prediction_counts = collections.Counter()
    for row in records:
        finite(row); image = images[row['image_id']]
        assert row['file_name'] == image['file_name']
        for d in row['detections']:
            assert d['image_id'] == row['image_id'] and d['native_class_id']+1 == d['category_id']
            assert d['class_name'] == cats[d['category_id']]
            x,y,w,h = d['bbox']
            assert w>0 and h>0 and 0<=d['score']<=1 and x>=-1e-3 and y>=-1e-3
            assert x+w<=image['width']+1e-3 and y+h<=image['height']+1e-3
            prediction_counts[d['class_name']] += 1
    assert json.loads((RESULT/'predictions.json').read_text()) == [d for r in records for d in r['detections']]
    metrics = json.loads((RESULT/'metrics.json').read_text()); finite(metrics)
    assert set(metrics['metrics']['per_class']) == set(counts)
    assert metrics['evaluation_images'] == 560 and metrics['GT_instances'] == 11520
    assert metrics['protocol_fingerprint'] == status['protocol_fingerprint']
    report = {'passed': True, 'images': 560, 'GT_instances': 11520, 'native_classes': 10,
              'GT_counts': dict(counts), 'prediction_counts': dict(prediction_counts),
              'detections': sum(prediction_counts.values()),
              'checks': ['All published GT fields except category ID unchanged', '560 image SHA256 checks',
                         'Exact unique image ID and filename coverage', 'Finite original-image boxes and scores',
                         'Native class IDs and unmerged class names', 'JSON and JSONL predictions identical',
                         'Finite metrics with all ten per-class entries'],
              'artifact_sha256': {name: sha(RESULT/name) for name in ['annotation.json','status.json','metrics.json','predictions.json','predictions.jsonl']}}
    (RESULT/'verification.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
