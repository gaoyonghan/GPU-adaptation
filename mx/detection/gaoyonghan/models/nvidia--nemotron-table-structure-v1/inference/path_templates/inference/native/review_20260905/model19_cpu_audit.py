"""Read-only audit of the saved Aryn predictions against original COCO JSON."""
import collections
import contextlib
import hashlib
import io
import json
from pathlib import Path
import zipfile

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

HERE = Path(__file__).resolve().parent
ARCHIVE = Path('__DATASET_ROOT__/doclaynet_test/DocLayNet_core.zip')
RESULT = Path('__MODEL_ROOT__/resources/runtime/assets/workspace/results/remaining25_v3/m19-deformable-detr-doclaynet')
report = {'purpose': 'independent read-only CPU audit; threshold .7 is a usage-example diagnostic, not a claimed official evaluation setting', 'archive': str(ARCHIVE), 'splits': {}, 'overlap': {}}
index = {}
with zipfile.ZipFile(ARCHIVE) as archive:
    members = collections.Counter(Path(x).name for x in archive.namelist() if x.lower().endswith('.png'))
    report['duplicate_image_basenames'] = sum(n > 1 for n in members.values())
    for split in ['train', 'val', 'test']:
        raw = archive.read(f'COCO/{split}.json')
        annotation = json.loads(raw)
        images = annotation['images']
        annotations = annotation['annotations']
        index[split] = {key: {image[key] for image in images} for key in ['id', 'file_name', 'doc_name']}
        index[split]['doc_page'] = {(image['doc_name'], image['page_no']) for image in images}
        item = report['splits'][split] = {
            'images': len(images), 'annotations': len(annotations),
            'categories': annotation['categories'], 'raw_annotation_sha256': hashlib.sha256(raw).hexdigest(),
            'unique_files': len(index[split]['file_name']), 'unique_docs': len(index[split]['doc_name']),
            'precedence_images': dict(collections.Counter(x.get('precedence') for x in images)),
            'precedence_annotations': dict(collections.Counter(x.get('precedence') for x in annotations)),
            'crowd': dict(collections.Counter(x['iscrowd'] for x in annotations)),
            'all_png_present': all(image['file_name'] in members for image in images),
        }
        if split == 'train':
            continue
        rows = [json.loads(line) for line in (RESULT / split / 'predictions.jsonl').open()]
        predictions = [prediction for row in rows for prediction in row['detections']]
        item['prediction_rows'] = len(rows)
        item['all_image_ids_exact'] = {row['image_id'] for row in rows} == index[split]['id']
        item['prediction_count'] = len(predictions)
        item['predictions_json_sha256'] = hashlib.sha256((RESULT / split / 'predictions.json').read_bytes()).hexdigest()
        item['matches_saved_json'] = predictions == json.loads((RESULT / split / 'predictions.json').read_text())
        item['score_range'] = [min(p['score'] for p in predictions), max(p['score'] for p in predictions)]
        item['thresholds'] = {}
        # val native score is already covered by the same reviewed runner; independently rescore test.
        if split == 'test':
            for threshold in [0.0, 0.7]:
                kept = [dict(p) for p in predictions if p['score'] > threshold]
                with contextlib.redirect_stdout(io.StringIO()):
                    gt = COCO()
                    gt.dataset = annotation
                    gt.createIndex()
                    dt = gt.loadRes(kept)
                    evaluator = COCOeval(gt, dt, 'bbox')
                    evaluator.evaluate()
                    evaluator.accumulate()
                    evaluator.summarize()
                item['thresholds'][str(threshold)] = {
                    'kept_predictions': len(kept), 'AP': float(evaluator.stats[0]),
                    'AP50': float(evaluator.stats[1]), 'AR100': float(evaluator.stats[8]),
                }
                print(split, threshold, item['thresholds'][str(threshold)], flush=True)
            item['independent_AP_matches_saved'] = abs(item['thresholds']['0.0']['AP'] - json.loads((RESULT / split / 'metrics.json').read_text())['metrics']['AP']) < 1e-12
        print(split, 'split audit complete', flush=True)
for a, b in [('train', 'val'), ('train', 'test'), ('val', 'test')]:
    report['overlap'][a + '__' + b] = {key: len(index[a][key] & index[b][key]) for key in index[a]}
report['overlap_note'] = 'Numeric image IDs restart in each independent COCO split; overlapping IDs alone do not mean the same image. File names, document names and document-page pairs are disjoint.'
(HERE / 'model19_cpu_audit.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
