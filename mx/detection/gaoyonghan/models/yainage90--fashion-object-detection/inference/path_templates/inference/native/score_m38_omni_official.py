"""Reuse saved V2 predictions with the pinned official Omni detection pipeline.

No GPU inference. Only prediction-format conversion/configuration and provenance
are project code; GT reading and metric computation are unmodified official code.
"""
import argparse
from collections import Counter
from datetime import datetime
import hashlib
import importlib.util
from importlib.metadata import distribution, version
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np
import yaml

HERE = Path(__file__).resolve().parent
REPO = Path('__MODEL_ROOT__/resources/runtime/assets/OmniDocBench')
REV = '85f760aa60e5f9f087e3f582109f55277937a565'
DATA = Path('__DATASET_ROOT__/omnidocbench_eval/OmniDocBench.json')
PRED = Path('__MODEL_ROOT__/resources/runtime/assets/workspace/results/m38_omni_native_v1')
OUT = Path('__MODEL_ROOT__/resources/runtime/assets/workspace/results/m38_omni_official_v1')
FILES = ['LICENSE', 'configs/layout_detection.yaml', 'dataset/detection_dataset.py',
         'task/detection_eval.py', 'registry/__init__.py', 'registry/registry.py', 'utils/ocr_utils.py']
CLASSES = ['title', 'text', 'abandon', 'figure', 'table', 'isolate_formula', 'formula_caption']
# Only native prediction names are mapped; GT mappings remain exactly as upstream.
PRED_MAP = {
    'doc_title': 'title', 'paragraph_title': 'title',
    'text': 'text', 'abstract': 'text', 'aside_text': 'text', 'content': 'text', 'vertical_text': 'text',
    'header': 'abandon', 'footer': 'abandon', 'number': 'abandon', 'footnote': 'abandon',
    'header_image': 'abandon', 'footer_image': 'abandon',
    'image': 'figure', 'chart': 'figure', 'table': 'table',
    'display_formula': 'isolate_formula', 'formula_number': 'formula_caption',
}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def git_file(name):
    return subprocess.check_output(['git', '-C', str(REPO), 'show', f'{REV}:{name}'])


def load_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def prepare(run, limit):
    assert subprocess.check_output(['git', '-C', str(REPO), 'remote', 'get-url', 'origin'], text=True).strip() == 'https://github.com/opendatalab/OmniDocBench.git'
    assert sha(DATA) == 'a45cd84b04ad8b793e775089640e6b681209abea33ead54c1828ddca35fae496'
    status = json.loads((PRED / 'status.json').read_text())
    inference = json.loads((PRED / 'protocol.json').read_text())
    assert status['state'] == 'complete' and status['completed_images'] == 1651
    assert hashlib.sha256(json.dumps(inference, sort_keys=True, ensure_ascii=False).encode()).hexdigest() == status['protocol_fingerprint']
    pages = json.loads(DATA.read_text())
    records = [json.loads(line) for line in (PRED / 'native_predictions.jsonl').read_text().splitlines() if line.strip()]
    records.sort(key=lambda r: r['image_id'])
    assert [r['image_id'] for r in records] == list(range(1, 1652))
    assert len(pages) == 1651
    source = run / 'official_source'
    hashes = {}
    for name in FILES:
        target = source / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(git_file(name))
        hashes[name] = sha(target)
    original_cfg = yaml.safe_load((source / 'configs/layout_detection.yaml').read_text())
    cfg = yaml.safe_load((source / 'configs/layout_detection.yaml').read_text())
    task = cfg['detection_eval']
    original_classes = task['categories']['eval_cat']['block_level']
    assert [c for c in original_classes if c in CLASSES] == CLASSES
    task['categories']['eval_cat'] = {'block_level': CLASSES}
    task['categories']['pred_cat_mapping'] = PRED_MAP
    # Official Omni-format loader supports exact .png names. Simple format appends .jpg.
    task['dataset']['dataset_name'] = 'detection_dataset'
    selected = pages[:limit] if limit else pages
    prediction_pages, skipped = [], Counter()
    for page, record in zip(selected, records):
        assert record['image_sha256'] == inference['images'][record['image_id']-1]['sha256']
        assert Path(record['image_path']).name == Path(page['page_info']['image_path']).name
        regions = []
        for box in record['boxes']:
            label, score = box['label'], float(box['score'])
            coords = box['coordinate']
            assert len(coords) == 4 and all(math.isfinite(v) for v in coords)
            assert math.isfinite(score) and 0 < score <= 1
            x1, y1, x2, y2 = coords
            assert x2 >= x1 and y2 >= y1
            # Preserve all predictions in the converted input. Official mapping filters them.
            regions.append(dict(category_type=label, poly=[x1,y1,x2,y1,x2,y2,x1,y2], score=score))
            if label not in PRED_MAP:
                skipped[label] += 1
        prediction_pages.append(dict(page_info=page['page_info'], layout_dets=regions))
    write(run / 'predictions_omni.json', prediction_pages)
    gt_path = DATA
    if limit:
        gt_path = run / 'smoke_gt.json'
        write(gt_path, selected)
    task['dataset']['ground_truth']['data_path'] = str(gt_path)
    task['dataset']['prediction']['data_path'] = str(run / 'predictions_omni.json')
    assert task['categories']['gt_cat_mapping'] == original_cfg['detection_eval']['categories']['gt_cat_mapping']
    (run / 'layout_detection.yaml').write_text(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False), encoding='utf-8')
    software = {n: version(n) for n in ['mmeval', 'pycocotools', 'numpy', 'PyYAML']}
    assert software == {'mmeval': '0.2.1', 'pycocotools': '2.0.7', 'numpy': '1.24.4', 'PyYAML': '6.0.2'}, 'Scoring environment changed; review versions.'
    provenance = dict(sequence=38, model='PaddlePaddle/PP-DocLayoutV2', images=len(selected),
        source_repository='https://github.com/opendatalab/OmniDocBench', source_revision=REV,
        source_files_sha256=hashes, software=software, dataset_sha256=sha(DATA),
        input_predictions_sha256=sha(PRED / 'native_predictions.jsonl'),
        input_protocol_sha256=sha(PRED / 'protocol.json'), input_result_directory=str(PRED),
        converted_predictions_sha256=sha(run / 'predictions_omni.json'),
        scoring_config_sha256=sha(run / 'layout_detection.yaml'),
        runner_sha256=sha(Path(__file__)), shell_sha256=sha(HERE / 'score_m38_omni_official.sh'),
        classes=CLASSES, native_prediction_mapping=PRED_MAP, excluded_prediction_counts=dict(skipped),
        original_official_classes=original_classes, excluded_official_classes=[c for c in original_classes if c not in CLASSES],
        scope='Official Omni layout detection, configurable 7-category subset; NOT previous native 7 classes or Paddle 25-class reproduction.',
        code_path='Unmodified DetectionDataset -> unmodified DetectionEval -> its original MMEval COCODetection. Project bootstrap imports only detection modules; no pdf_validation.py global OCR/formula imports.',
        adjustments='Change input paths, select 7 official categories, configure native prediction mapping, use official detection_dataset instead of simple-format loader. No GT mapping or scorer code edits.',
        known_upstream_behavior=[
            'Official sample config misspells refernece; actual reference GT is excluded. Keep upstream config unchanged and exclude native reference/reference_content as well.',
            'abandon is a scored class including headers/footers/page numbers/footnotes, NOT a global ignore region as in previous project scorer.',
            'Upstream loader ignores item.ignore and excludes unmapped masks; it does NOT create project crowd regions. Behavior preserved, not silently patched.',
            'Upstream poly2bbox uses selected polygon corners, not project min/max; preserved.',
            'No inline formula or reading-order score in this layout-only configuration.',
            'Omni default figure GT also includes code_txt; no reliable native code label, so those samples may be missed. Native image/chart are mapped to figure; algorithm/seal are not guessed into figure.',
            'Figure captions/table captions/table footnotes excluded: native outputs cannot reliably separate these three official categories.'],
        official_comparable=False, gpu_inference_executed=False,
        inference_unchanged=inference['prediction_options'])
    for package, rel in [('mmeval','mmeval/metrics/coco_detection.py'), ('pycocotools','pycocotools/cocoeval.py')]:
        path = Path(distribution(package).locate_file(rel))
        provenance.setdefault('metric_source_sha256', {})[str(path)] = sha(path)
    write(run / 'protocol.json', provenance)
    sys.path.insert(0, str(source))
    dataset_module = load_file('omni_native_detection_dataset', source / 'dataset/detection_dataset.py')
    task_module = load_file('omni_native_detection_task', source / 'task/detection_eval.py')
    dataset = dataset_module.DetectionDataset(task)
    assert len(dataset.samples['gts']) == len(dataset.samples['preds']) == len(selected)
    counts = Counter()
    for gt, pred in zip(dataset.samples['gts'], dataset.samples['preds']):
        for item in (gt, pred):
            b = item['bboxes']
            if b.size:
                assert b.ndim == 2 and b.shape[1] == 4 and np.isfinite(b).all()
                assert (b[:, 2] > b[:, 0]).all() and (b[:, 3] > b[:, 1]).all(), 'Official loader produced degenerate box; investigate before scoring.'
        counts.update(CLASSES[int(i)] for i in gt['labels'])
    write(run / 'coverage.json', dict(images=len(selected), gt_instances=dict(counts), prediction_exclusions=dict(skipped)))
    return dataset, task_module


class CaptureMetric:
    """Persist the metric return value, which the official task only prints."""
    def __init__(self, metric):
        self.metric = metric
        self.result = None

    def __call__(self, *args, **kwargs):
        self.result = self.metric(*args, **kwargs)
        return self.result


def evaluate(run, limit):
    dataset, task_module = prepare(run, limit)
    capture = CaptureMetric(dataset.coco_det_metric)
    dataset.coco_det_metric = capture
    task_module.DetectionEval(dataset, metrics_list=['COCODet'], save_name='m38_omni')
    assert capture.result is not None
    result = {k: float(v) if math.isfinite(float(v)) else None for k, v in capture.result.items()}
    write(run / ('smoke_metrics.json' if limit else 'metrics.json'), dict(
        run_scope='smoke_only' if limit else 'full_1651', official_comparable=False,
        unit='fraction_0_to_1', classes=CLASSES, official_omni_metrics=result))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--preflight', action='store_true')
    parser.add_argument('--max-images', type=int, choices=[0, 10], default=0)
    parser.add_argument('--output-root', type=Path, default=OUT)
    args = parser.parse_args()
    if args.preflight:
        with tempfile.TemporaryDirectory(prefix='m38-omni-preflight-') as directory:
            dataset, _ = prepare(Path(directory), 0)
            print('ready: official loader validated 1651 pages; no GPU inference or metric evaluation.')
        return
    run = args.output_root / ('smoke' if args.max_images else 'full') / datetime.now().strftime('run_%Y%m%d_%H%M%S_%f')
    run.mkdir(parents=True)
    status = dict(sequence=38, state='scoring', run_directory=str(run))
    write(run / 'status.json', status)
    try:
        evaluate(run, args.max_images)
        status.update(state='smoke_complete' if args.max_images else 'complete', images=args.max_images or 1651)
        print(f'Omni官方检测流程完成：{run}', flush=True)
    except BaseException as exc:
        status.update(state='failed', error=repr(exc)); raise
    finally:
        write(run / 'status.json', status)


if __name__ == '__main__':
    main()
