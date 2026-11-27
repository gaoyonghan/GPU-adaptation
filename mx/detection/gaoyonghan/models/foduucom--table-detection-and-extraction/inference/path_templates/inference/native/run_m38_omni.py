"""Official PP-DocLayoutV2 inference; project-defined Omni subset COCO scoring."""
import argparse
from collections import Counter, defaultdict
import fcntl
from importlib.metadata import distribution, version
import json
import math
from pathlib import Path
import shutil
import time

import numpy as np
from PIL import Image
from run_pp_layout import GT_MAP, PRED_MAP, order_counts
from run_model import core, digest, file_hash

HERE = Path(__file__).resolve().parent
DATA = Path('__DATASET_ROOT__/omnidocbench_eval')
MODEL = Path('__MODEL_ROOT__/weights/PaddlePaddle/PP-DocLayoutV2')
OUT = Path('__MODEL_ROOT__/resources/runtime/assets/workspace/results/m38_omni_native_v1')
DIRECT = {'equation_isolated': 'display_formula', 'footer': 'footer',
          'page_footnote': 'footnote', 'equation_caption': 'formula_number',
          'header': 'header', 'page_number': 'number', 'table': 'table'}
COARSE_PRED = {**PRED_MAP, 'display_formula': 'formula'}
COARSE_PRED.pop('formula')  # V2 has no native label named formula.
PREDICT_OPTIONS = dict(batch_size=1, threshold=0.001, layout_nms=True,
                       layout_shape_mode='rect', filter_overlap_boxes=True)


def finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def preflight():
    from pycocotools import mask  # Fail before GPU work if compiled dependency is unavailable.
    assert mask.area is not None
    assert file_hash(DATA / 'OmniDocBench.json') == 'a45cd84b04ad8b793e775089640e6b681209abea33ead54c1828ddca35fae496'
    assert file_hash(MODEL / 'config.json') == 'e1a25a0cbdc59c3668bc96467ecca6b9747eca29ff26d90282f452adc89df21f'
    config = json.loads((MODEL / 'config.json').read_text())
    labels = config['label_list']
    assert len(labels) == 25 and set(DIRECT.values()) <= set(labels)
    pages = json.loads((DATA / 'OmniDocBench.json').read_text())
    assert len(pages) == 1651
    images = []
    print('CPU预检：校验1651张图像、标注和本地权重；不执行模型推理。', flush=True)
    for index, page in enumerate(pages, 1):
        info = page['page_info']
        path = DATA / 'images' / Path(info['image_path']).name
        with Image.open(path) as image:
            assert image.size == (info['width'], info['height']), str(path)
        images.append(dict(id=index, path=str(path), sha256=file_hash(path)))
        if index % 250 == 0 or index == len(pages):
            print(f'图像校验 {index}/{len(pages)}', flush=True)
    assert len({im['path'] for im in images}) == 1651
    packages = ['paddlepaddle-gpu', 'paddleocr', 'paddlex', 'numpy', 'scipy', 'Pillow', 'pycocotools']
    software = {name: version(name) for name in packages}
    assert {k: software[k] for k in ['paddlepaddle-gpu', 'paddleocr', 'paddlex']} == {
        'paddlepaddle-gpu': '3.2.1', 'paddleocr': '3.6.0', 'paddlex': '3.6.1'}, 'Native environment changed; audit before rerunning.'
    sources = [Path(__file__), HERE / 'run_m38_omni.sh', HERE / 'run_pp_layout.py', HERE / 'run_model.py',
               HERE / 'adapters.py', HERE.parent / 'official_object_detection_eval/run_model.py',
               HERE.parent / 'official_object_detection_eval/download_official_source.py']
    for package, relative in [('paddleocr', 'paddleocr/_models'), ('paddlex', 'paddlex/inference/models/layout_analysis'),
                              ('paddlex', 'paddlex/inference/models/object_detection')]:
        sources.extend(sorted(Path(distribution(package).locate_file(relative)).glob('*.py')))
    for relative in ['pycocotools/coco.py', 'pycocotools/cocoeval.py', 'pycocotools/mask.py']:
        sources.append(Path(distribution('pycocotools').locate_file(relative)))
    protocol = dict(sequence=38, model_id='PaddlePaddle/PP-DocLayoutV2', dataset_relation='public_replacement',
                    dataset='OmniDocBench local 1651-page snapshot', official_comparable=False,
                    annotation_sha256=file_hash(DATA / 'OmniDocBench.json'), images=images,
                    model_files={name: file_hash(MODEL / name) for name in
                                 ['config.json', 'inference.yml', 'inference.json', 'inference.pdiparams']},
                    software=software, source_sha256={str(p): file_hash(p) for p in sources},
                    native_entry='paddleocr.LayoutDetection(model_dir=local, engine=paddle_static).predict',
                    device='gpu:0 under CUDA_VISIBLE_DEVICES', engine='paddle_static', fp32=True,
                    prediction_options=PREDICT_OPTIONS,
                    threshold_note='Project AP baseline uses 0.001 to retain low-score detections; not an author-published evaluation threshold. Old run used 0.3.',
                    native_postprocess='Official layout_nms=True example; keep native overlap filter and skip_order_labels defaults, no custom merges.',
                    primary_gt_map=DIRECT, primary_pred_map={v: v for v in DIRECT.values()},
                    diagnostic_gt_map=GT_MAP, diagnostic_pred_map=COARSE_PRED,
                    scoring='pycocotools COCOeval bbox IoU .50:.05:.95, maxDets=[1,10,100], plus per-class AP50; project subset definition, not Paddle official evaluation recipe.',
                    gt_geometry='Top-level layout_dets polygons -> enclosing xywh, unchanged coordinates; no clipping, no fragment duplication.',
                    ignore='*_mask and abandon regions, plus explicit ignored GT: project class-specific COCO crowd regions. Not positive boxes.',
                    inline_formula='Retain raw native predictions but DO NOT score: exhaustive span annotation scope unverified.',
                    coverage='7 explicit block classes primary; legacy 11 merged groups diagnostic only; not full 25-class evaluation.',
                    reading_order='Legacy 11-group class-aware Hungarian IoU>=0.5; matched-pair accuracy, GT coverage and end-to-end correct-pair fraction. Not official metric.',
                    reference=dict(AP50_percent=81.4, dataset='author-built 1000 images / 25 classes',
                                   url='https://www.paddleocr.ai/latest/en/version3.x/module_usage/layout_analysis.html'),
                    changes_from_old=['display_formula mapping fixed', 'new 7-class primary including formula_number',
                                      'score threshold 0.3 -> 0.001', 'explicit official layout_nms=True', 'all 25 native labels retained'])
    return pages, labels, protocol


def annotation_for(pages, mapping):
    names = list(dict.fromkeys(mapping.values()))
    ids = {name: i + 1 for i, name in enumerate(names)}
    annotation = dict(images=[], annotations=[], categories=[dict(id=i, name=n) for n, i in ids.items()], info={}, licenses=[])
    for image_id, page in enumerate(pages, 1):
        info = page['page_info']
        annotation['images'].append(dict(id=image_id, file_name=info['image_path'], width=info['width'], height=info['height']))
        for region in page['layout_dets']:
            cat = region['category_type']
            masked = cat.endswith('_mask') or cat == 'abandon'
            if cat not in mapping and not masked:
                continue
            poly = region['poly']
            assert len(poly) >= 6 and len(poly) % 2 == 0 and all(finite(v) for v in poly)
            x1, y1, x2, y2 = min(poly[::2]), min(poly[1::2]), max(poly[::2]), max(poly[1::2])
            assert x2 > x1 and y2 > y1
            for cat_id in ids.values() if masked else [ids[mapping[cat]]]:
                annotation['annotations'].append(dict(id=len(annotation['annotations']) + 1, image_id=image_id,
                    category_id=cat_id, bbox=[x1, y1, x2-x1, y2-y1], area=(x2-x1)*(y2-y1),
                    iscrowd=int(masked or region.get('ignore', False)), order=region.get('order')))
    return annotation


def predictions_for(records, annotation, mapping):
    ids = {cat['name']: cat['id'] for cat in annotation['categories']}
    predictions = []
    for record in records:
        for box in record['boxes']:
            if box['label'] not in mapping:
                continue
            x1, y1, x2, y2 = box['coordinate']
            predictions.append(dict(image_id=record['image_id'], category_id=ids[mapping[box['label']]],
                                    score=box['score'], bbox=[x1, y1, x2-x1, y2-y1], order=box.get('order')))
    return predictions


def score(annotation, predictions):
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    gt = COCO(); gt.dataset = annotation; gt.createIndex()
    if predictions:
        dt = gt.loadRes(predictions)
    else:
        dt = COCO(); dt.dataset = {**annotation, 'annotations': []}; dt.createIndex()
    ev = COCOeval(gt, dt, 'bbox'); ev.evaluate(); ev.accumulate(); ev.summarize()
    result = {name: float(v) if v >= 0 else None for name, v in zip(['AP', 'AP50', 'AP75'], ev.stats[:3])}
    result['per_class'] = {}
    for i, cat in enumerate(annotation['categories']):
        precision = ev.eval['precision'][:, :, i, 0, -1]
        ap, ap50 = precision[precision >= 0], precision[0][precision[0] >= 0]
        result['per_class'][cat['name']] = dict(AP=float(ap.mean()) if ap.size else None,
            AP50=float(ap50.mean()) if ap50.size else None,
            positive_boxes=sum(a['category_id'] == cat['id'] and not a['iscrowd'] for a in annotation['annotations']))
    return result


def evaluate(pages, records):
    result = dict(unit='fraction_0_to_1', official_comparable=False, evaluated_images=len(pages))
    for name, gt_map, pred_map in [('primary_7_explicit_block_classes', DIRECT, {v: v for v in DIRECT.values()}),
                                  ('diagnostic_11_merged_groups', GT_MAP, COARSE_PRED)]:
        annotation = annotation_for(pages, gt_map)
        predictions = predictions_for(records, annotation, pred_map)
        result[name] = score(annotation, predictions)
        if name.startswith('diagnostic'):
            totals = np.zeros(5, dtype=np.int64)
            gold_by_image, pred_by_image = defaultdict(list), defaultdict(list)
            for item in annotation['annotations']:
                gold_by_image[item['image_id']].append(item)
            for item in predictions:
                pred_by_image[item['image_id']].append(item)
            for im in annotation['images']:
                totals += order_counts(gold_by_image[im['id']], pred_by_image[im['id']])
            a, b, c, d, e = map(int, totals)
            result['reading_order_diagnostic'] = dict(correct_pairs=a, matched_pairs=b, all_gt_pairs=c,
                matched_elements=d, gt_ordered_elements=e, pair_accuracy=a/b if b else None,
                matched_element_recall=d/e if e else None, end_to_end_correct_pair_fraction=a/c if c else None)
    return result


def validate_boxes(boxes, labels):
    for box in boxes:
        assert box['label'] in labels, f'Unexpected native label: {box}'
        coords = box['coordinate']
        assert len(coords) == 4 and all(finite(v) for v in coords)
        assert coords[2] >= coords[0] and coords[3] >= coords[1]
        assert finite(box['score']) and 0 <= box['score'] <= 1
        assert box.get('order') is None or finite(box['order'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--max-images', type=int, choices=[0, 10], default=0)
    parser.add_argument('--preflight', action='store_true')
    args = parser.parse_args()
    pages, labels, protocol = preflight()
    if args.preflight:
        for mapping in [DIRECT, GT_MAP]:
            annotation = annotation_for(pages, mapping)
            print('GT classes / positive boxes:', len(annotation['categories']), sum(not a['iscrowd'] for a in annotation['annotations']))
        print('ready: 1651 images, local model files and environment checked; no model construction/forward executed')
        return
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / '.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        fingerprint = digest(protocol)
        status_path = OUT / 'status.json'
        old = json.loads(status_path.read_text()) if status_path.exists() else None
        if old and old['protocol_fingerprint'] != fingerprint:
            raise RuntimeError('Protocol/data changed; preserve this output and choose an audited new output version.')
        if old and old['state'] == 'complete':
            print('38号已全量完成，跳过。'); return
        core.atomic_json(OUT / 'protocol.json', protocol)
        snapshot = OUT / 'source_snapshot'; snapshot.mkdir(exist_ok=True)
        for i, name in enumerate(protocol['source_sha256']):
            shutil.copy2(name, snapshot / f'{i:03d}_{Path(name).name}')
        status = dict(state='running', sequence=38, protocol_fingerprint=fingerprint, started_at=time.time())
        core.atomic_json(status_path, status)
        try:
            records = core.read_jsonl(OUT / 'native_predictions.jsonl')
            existing = {r['image_id'] for r in records}
            assert len(existing) == len(records) and existing <= set(range(1, 1652))
            for record in records:
                assert record['image_sha256'] == protocol['images'][record['image_id']-1]['sha256']
                validate_boxes(record['boxes'], labels)
            count = args.max_images or len(pages)
            if len(existing.intersection(range(1, count+1))) < count:
                import paddle
                from paddleocr import LayoutDetection
                assert paddle.is_compiled_with_cuda() and paddle.device.cuda.device_count() > 0
                model = LayoutDetection(model_name='PP-DocLayoutV2', model_dir=str(MODEL), device='gpu:0',
                                        engine='paddle_static', enable_hpi=False, use_tensorrt=False, precision='fp32')
                for info in protocol['images'][:count]:
                    if info['id'] in existing:
                        continue
                    print(f'38号 {info["id"]}/{count}: {Path(info["path"]).name}', flush=True)
                    output = model.predict(info['path'], **PREDICT_OPTIONS)
                    assert len(output) == 1
                    boxes = output[0].json['res']['boxes']
                    validate_boxes(boxes, labels)
                    record = dict(image_id=info['id'], image_path=info['path'], image_sha256=info['sha256'], boxes=boxes)
                    core.append_jsonl(OUT / 'native_predictions.jsonl', record)
                    records.append(record)
                model.close()
            selected = sorted([r for r in records if r['image_id'] <= count], key=lambda r: r['image_id'])
            assert [r['image_id'] for r in selected] == list(range(1, count+1))
            status.update(state='scoring', completed_images=count); core.atomic_json(status_path, status)
            result = evaluate(pages[:count], selected)
            result['run_scope'] = 'smoke_only' if args.max_images else 'full_1651'
            result['native_prediction_counts'] = dict(Counter(b['label'] for r in selected for b in r['boxes']))
            core.atomic_json(OUT / ('smoke_metrics.json' if args.max_images else 'metrics.json'), result)
            status.update(state='smoke_complete' if args.max_images else 'complete', completed_images=count)
            print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
        except BaseException as exc:
            status.update(state='failed', error=repr(exc)); raise
        finally:
            status['elapsed_seconds'] = time.time()-status['started_at']
            core.atomic_json(status_path, status)


if __name__ == '__main__':
    main()
