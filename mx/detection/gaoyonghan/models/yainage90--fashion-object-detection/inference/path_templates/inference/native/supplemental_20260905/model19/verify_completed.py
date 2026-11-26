"""Independent read-only input/prediction audit and pooled CPU score reproduction."""
import collections
import hashlib
import io
import json
import math
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

HERE = Path(__file__).resolve().parent
DATA = Path('__DATASET_ROOT__/supplemental_20260905/model19')
OUT = Path('__MODEL_ROOT__/resources/runtime/assets/workspace/results/remaining25_supplemental_20260905/m19-doclaynet-competition498')
MODEL = Path('__MODEL_ROOT__/weights/Aryn/deformable-detr-DocLayNet')


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def finite(value):
    if isinstance(value, dict):
        for v in value.values(): finite(v)
    elif isinstance(value, list):
        for v in value: finite(v)
    elif isinstance(value, (int, float)):
        assert math.isfinite(value)


def main():
    status = json.loads((OUT/'status.json').read_text()); protocol = status['protocol']
    metrics = json.loads((OUT/'metrics.json').read_text()); finite(metrics)
    assert status['state'] == 'complete' and status['completed_images'] == 498
    assert hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest() == status['protocol_fingerprint'] == metrics['protocol_fingerprint']
    before_hashes = {n: sha(OUT/n) for n in ['status.json','metrics.json','predictions.json','predictions.jsonl','annotation.json']}
    snapshot = OUT/'source_snapshot'; snapshot.mkdir(exist_ok=True)
    source_proof = {}
    for name, key in [('run.py','script_sha256'),('prepare.py','prepare_script_sha256')]:
        assert sha(HERE/name) == protocol[key]
        target = snapshot/name
        if target.exists(): assert sha(target) == protocol[key]
        else: target.write_bytes((HERE/name).read_bytes())
        source_proof[name] = sha(target)
    snapshot_note = {'archived_at_utc': datetime.now(timezone.utc).isoformat(),
                     'timing': 'Post-run archive: current files match hashes recorded by actual execution. Not claimed to be copied before inference.',
                     'sha256': source_proof}
    (snapshot/'archive_note.json').write_text(json.dumps(snapshot_note, indent=2))
    for name,key in [('model.safetensors','weight_sha256'),('config.json','config_sha256'),('preprocessor_config.json','preprocessor_config_sha256')]:
        assert sha(MODEL/name) == protocol[key]
    prepared = json.loads((DATA/'prepared.json').read_text())
    assert protocol['dataset'] == prepared
    ann = json.loads((OUT/'annotation.json').read_text())
    assert sha(OUT/'annotation.json') == prepared['annotation_sha256'] == metrics['annotation_sha256'] == sha(DATA/'annotation.json')
    hashes = json.loads((DATA/'image_sha256.json').read_text())
    archives = {name: zipfile.ZipFile(DATA/name) for name in prepared['sources']}
    metadata_counts = {}
    for name, archive in archives.items():
        assert sha(DATA/name) == prepared['sources'][name]['sha256']
        assert archive.testzip() is None
        coco = json.loads(archive.read('coco.json'))
        assert coco['images'] == ann['images'] and coco['categories'] == ann['categories']
        assert not any('doc_category' in image for image in coco['images'])
        metadata = [json.loads(archive.read(n))['metadata'] for n in archive.namelist() if n.startswith('JSON/') and n.endswith('.json')]
        counts = {'rows': len(metadata), 'doc_category': dict(collections.Counter(m.get('doc_category') for m in metadata)),
                  'collection': dict(collections.Counter(m.get('collection') for m in metadata))}
        assert counts == {'rows': 498, 'doc_category': {'other': 498}, 'collection': {'subset': 498}}
        metadata_counts[name] = counts
    original = json.loads(archives['competition_dataset_groundtruth.zip'].read('coco.json'))
    assert ann['annotations'] == original['annotations']
    image_ids = {i['id'] for i in ann['images']}; images = {i['id']:i for i in ann['images']}
    assert len(image_ids) == len(images) == 498 and len(ann['annotations']) == 9158
    for image in ann['images']:
        raw = [archive.read('PNG/'+image['file_name']) for archive in archives.values()]
        assert raw[0] == raw[1] and hashlib.sha256(raw[0]).hexdigest() == hashes[image['file_name']]
        with Image.open(io.BytesIO(raw[0])) as opened:
            opened.load(); assert opened.size == (image['width'], image['height']) == (1025, 1025)
    for archive in archives.values(): archive.close()
    categories = {c['id']: c['name'] for c in ann['categories']}
    model_labels = json.loads((MODEL/'config.json').read_text())['id2label']
    assert all(model_labels[str(k)] == v for k,v in categories.items()) and len(categories) == 11
    counts = collections.Counter(categories[a['category_id']] for a in ann['annotations'])
    assert dict(counts) == metrics['pooled']['gt_instances_per_class']
    rows = [json.loads(line) for line in (OUT/'predictions.jsonl').open()]
    assert len(rows) == len({r['image_id'] for r in rows}) == 498
    assert {r['image_id'] for r in rows} == image_ids
    predictions = []; out_of_bounds = 0
    for row in rows:
        finite(row); image = images[row['image_id']]
        assert row['file_name'] == image['file_name'] and row['domain'] is None
        assert len(row['detections']) <= 100
        for d in row['detections']:
            assert d['image_id'] == row['image_id'] and d['category_id'] in categories and 0 < d['score'] <= 1
            x,y,w,h = d['bbox']; assert w > 0 and h > 0
            out_of_bounds += int(x<0 or y<0 or x+w>image['width'] or y+h>image['height'])
        predictions.extend(row['detections'])
    assert len(predictions) == 49800 and predictions == json.loads((OUT/'predictions.json').read_text())
    assert metrics['competition_AP'] is None and metrics['competition_AR100'] is None and metrics['per_domain'] == {}
    gt = COCO(); gt.dataset = original; gt.dataset.setdefault('info',{}); gt.createIndex()
    dt = gt.loadRes(predictions); evaluator = COCOeval(gt, dt, 'bbox')
    evaluator.evaluate(); evaluator.accumulate(); evaluator.summarize()
    names = ['AP','AP50','AP75','AP_small','AP_medium','AP_large','AR1','AR10','AR100','AR_small','AR_medium','AR_large']
    independent = dict(zip(names,map(float,evaluator.stats)))
    for name,value in independent.items(): assert abs(value-metrics['pooled'][name]) < 1e-12
    per_class = {}
    for k,category in enumerate(evaluator.params.catIds):
        values = evaluator.eval['precision'][:,:,k,0,-1]; value = float(values[values>-1].mean())
        per_class[categories[category]] = value
        assert abs(value-metrics['pooled']['per_class_AP'][categories[category]]) < 1e-12
    assert all(sha(OUT/n) == v for n,v in before_hashes.items()), 'Frozen results changed during audit'
    report = {'passed': True, 'images': 498, 'GT_instances': 9158, 'classes': 11, 'predictions': 49800,
              'image_decode_and_SHA_checks': 498, 'GT_unchanged_from_original_ZIP': True,
              'metadata_counts': metadata_counts, 'domain_mapping_verified': False,
              'competition_AP': None, 'independent_pooled_metrics': independent, 'per_class_AP': per_class,
              'recomputed_metric_tolerance': 1e-12, 'out_of_bounds_native_predictions_retained': out_of_bounds,
              'box_note': 'Native DETR postprocessor does not clip boxes; finite positive boxes outside image remain unchanged for native COCO scoring.',
              'snapshot': snapshot_note, 'artifact_sha256': before_hashes}
    (OUT/'verification.json').write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
