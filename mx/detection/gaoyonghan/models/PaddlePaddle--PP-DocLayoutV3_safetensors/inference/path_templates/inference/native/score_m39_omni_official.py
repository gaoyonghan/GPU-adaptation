"""Official Omni bbox scoring with the same configuration as accepted model 38."""
import argparse
from collections import Counter
from datetime import datetime
import hashlib
from importlib.metadata import version
import json
import math
from pathlib import Path
import shutil
import sys

import numpy as np
import yaml
import score_m38_omni_official as shared

HERE = Path(__file__).resolve().parent
PRED = Path('__OUTPUT_ROOT__/m039/native')
OUT = Path('__OUTPUT_ROOT__/m039/official')
BASELINE = Path('__MODEL_ROOT__/resources/runtime/assets/workspace/results/m38_omni_official_v1/full/run_20260914_232048_399395')


def prepare(run, limit):
    baseline = json.loads((BASELINE/'protocol.json').read_text())
    status = json.loads((PRED/'status.json').read_text())
    inference = json.loads((PRED/'protocol.json').read_text())
    assert inference['sequence'] == status['sequence'] == 39
    assert hashlib.sha256(json.dumps(inference, sort_keys=True, ensure_ascii=False).encode()).hexdigest() == status['protocol_fingerprint']
    assert status['state'] in ['smoke_complete','complete']
    assert limit or (status['state'] == 'complete' and status['completed_images'] == 1651)
    assert shared.sha(shared.DATA) == inference['dataset_sha256'] == baseline['dataset_sha256']
    assert shared.CLASSES == baseline['classes'] and shared.PRED_MAP == baseline['native_prediction_mapping']
    assert shared.REV == baseline['source_revision']
    software = {n:version(n) for n in ['mmeval','pycocotools','numpy','PyYAML']}
    if software != baseline['software']:
        print('Scoring environment differences recorded:', software, flush=True)
    pages = json.loads(shared.DATA.read_text())
    records = [json.loads(line) for line in (PRED/'native_predictions.jsonl').read_text().splitlines() if line.strip()]
    ids = [r['image_id'] for r in records]
    assert len(ids) == len(set(ids)) and set(ids) <= set(range(1,1652))
    count = limit or 1651
    records = sorted([r for r in records if r['image_id'] <= count], key=lambda r:r['image_id'])
    assert [r['image_id'] for r in records] == list(range(1,count+1))
    selected = pages[:count]
    source = run/'official_source'
    for name in shared.FILES:
        path = source/name; path.parent.mkdir(parents=True, exist_ok=True)
        bundled=shared.REPO/name
        assert bundled.is_file(), f'Missing packaged Omni source: {bundled}'
        assert shared.sha(bundled)==baseline['source_files_sha256'][name], f'Packaged source differs: {name}'
        path.write_bytes(bundled.read_bytes())
        assert shared.sha(path) == baseline['source_files_sha256'][name]
    cfg = yaml.safe_load((source/'configs/layout_detection.yaml').read_text())
    task = cfg['detection_eval']
    task['dataset']['dataset_name'] = 'detection_dataset'
    task['categories']['eval_cat'] = {'block_level': shared.CLASSES}
    task['categories']['pred_cat_mapping'] = shared.PRED_MAP
    previous_cfg = yaml.safe_load((BASELINE/'layout_detection.yaml').read_text())
    assert task['categories'] == previous_cfg['detection_eval']['categories']
    predictions, excluded = [], Counter()
    labels = inference['canonical_labels_by_id']
    assert labels[5] == 'display_formula' and labels[15] == 'inline_formula'
    for page, record in zip(selected, records):
        assert record['image_sha256'] == inference['images'][record['image_id']-1]['sha256']
        assert Path(record['image_path']).name == Path(page['page_info']['image_path']).name
        regions = []
        for b in record['boxes']:
            cid = b['native_class_id']; assert type(cid) is int and 0 <= cid < 25
            label = labels[cid]
            assert b['label'] == label and b['config_label'] == inference['id2label'][str(cid)]
            assert np.isfinite([b['score'], *b['coordinate']]).all() and 0 < b['score'] <= 1
            x1,y1,x2,y2 = b['coordinate']; assert x2 > x1 and y2 > y1
            # Omni bbox task uses returned native boxes, not polygon enclosing boxes.
            regions.append(dict(category_type=label, poly=[x1,y1,x2,y1,x2,y2,x1,y2], score=b['score']))
            if label not in shared.PRED_MAP: excluded[label] += 1
        predictions.append(dict(page_info=page['page_info'], layout_dets=regions))
    shared.write(run/'predictions_omni.json', predictions)
    gt_path = shared.DATA
    if limit:
        gt_path = run/'smoke_gt.json'; shared.write(gt_path, selected)
    task['dataset']['ground_truth']['data_path'] = str(gt_path)
    task['dataset']['prediction']['data_path'] = str(run/'predictions_omni.json')
    (run/'layout_detection.yaml').write_text(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False), encoding='utf-8')
    protocol = dict(sequence=39, model='PaddlePaddle/PP-DocLayoutV3_safetensors', images=count,
        source_repository=baseline['source_repository'], source_revision=shared.REV,
        source_files_sha256=baseline['source_files_sha256'], software=software, reference_software=baseline['software'],
        dataset_sha256=shared.sha(shared.DATA), classes=shared.CLASSES, native_prediction_mapping=shared.PRED_MAP,
        matched_model38_baseline=str(BASELINE), input_result_directory=str(PRED),
        input_predictions_sha256=shared.sha(PRED/'native_predictions.jsonl'),
        input_protocol_sha256=shared.sha(PRED/'protocol.json'),
        converted_predictions_sha256=shared.sha(run/'predictions_omni.json'),
        scoring_config_sha256=shared.sha(run/'layout_detection.yaml'),
        known_upstream_behavior=baseline['known_upstream_behavior'],
        code_path=baseline['code_path'], adjustments=baseline['adjustments'],
        label_resolution='Native numeric IDs -> official inference.yml names; config.json aliases are preserved separately, not used to collapse formula classes.',
        geometry='Native returned bbox -> rectangular poly, no clipping or merging. Original predicted polygons are saved but not scored in this run.',
        inference_unchanged=inference['prediction_options'], inherited_inference_patch=inference['existing_patch'],
        official_comparable=False, gpu_inference_executed=False,
        scope='Omni official detection implementation and identical 7-category/GT config as model 38. Not a segmentation/reading-order or end-to-end paper reproduction.')
    snapshot = run/'project_source'; snapshot.mkdir()
    for path in [Path(__file__), HERE/'run_m39_omni.sh', Path(shared.__file__)]:
        shutil.copy2(path, snapshot/path.name)
        protocol.setdefault('project_source_sha256', {})[str(path)] = shared.sha(path)
    shared.write(run/'protocol.json', protocol)
    sys.path.insert(0, str(source))
    module = shared.load_file('omni39_detection_dataset', source/'dataset/detection_dataset.py')
    task_module = shared.load_file('omni39_detection_task', source/'task/detection_eval.py')
    dataset = module.DetectionDataset(task)
    assert len(dataset.samples['gts']) == len(dataset.samples['preds']) == count
    counts = Counter()
    for gt, pred in zip(dataset.samples['gts'], dataset.samples['preds']):
        counts.update(shared.CLASSES[int(cid)] for cid in gt['labels'])
        for item in [gt,pred]:
            boxes = item['bboxes']
            if boxes.size: assert np.isfinite(boxes).all() and (boxes[:,2:] > boxes[:,:2]).all()
    if not limit: assert dict(counts) == json.loads((BASELINE/'coverage.json').read_text())['gt_instances']
    shared.write(run/'coverage.json', dict(images=count, gt_instances=dict(counts), prediction_exclusions=dict(excluded)))
    return dataset, task_module


def main():
    global BASELINE, PRED
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--max-images', type=int, choices=[0,10], default=0)
    parser.add_argument('--output-root', type=Path, default=OUT)
    parser.add_argument('--baseline', type=Path, default=BASELINE)
    parser.add_argument('--prediction-root', type=Path, default=PRED)
    args = parser.parse_args()
    BASELINE, PRED = args.baseline, args.prediction_root
    run = args.output_root/('smoke' if args.max_images else 'full')/datetime.now().strftime('run_%Y%m%d_%H%M%S_%f')
    run.mkdir(parents=True)
    status = dict(sequence=39, state='scoring')
    shared.write(run/'status.json', status)
    try:
        dataset, task_module = prepare(run, args.max_images)
        capture = shared.CaptureMetric(dataset.coco_det_metric); dataset.coco_det_metric = capture
        task_module.DetectionEval(dataset, metrics_list=['COCODet'], save_name='m39_omni')
        assert capture.result is not None
        result = {k:float(v) if math.isfinite(float(v)) else None for k,v in capture.result.items()}
        shared.write(run/('smoke_metrics.json' if args.max_images else 'metrics.json'), dict(
            run_scope='smoke_only' if args.max_images else 'full_1651', classes=shared.CLASSES,
            unit='fraction_0_to_1', official_comparable=False, official_omni_metrics=result))
        status.update(state='smoke_complete' if args.max_images else 'complete', images=args.max_images or 1651)
        print(f'39号Omni官方评分完成：{run}', flush=True)
    except BaseException as exc:
        status.update(state='failed', error=repr(exc)); raise
    finally: shared.write(run/'status.json', status)


if __name__ == '__main__': main()
