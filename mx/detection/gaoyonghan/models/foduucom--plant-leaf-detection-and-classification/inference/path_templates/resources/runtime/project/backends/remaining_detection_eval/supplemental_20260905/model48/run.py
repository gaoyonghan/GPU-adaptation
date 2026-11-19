"""Official Nemotron inference on the recovered published CACHED val560."""
import argparse
import hashlib
import importlib.metadata
import json
import math
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

HERE = Path(__file__).resolve().parent
DATA = Path('__DATASET_ROOT__/supplemental_20260905/model48')
OUTPUT = Path('__MODEL_ROOT__/resources/runtime/assets/workspace/results/remaining25_supplemental_20260905/m48-cached-val560-native10')
MODEL = Path('__MODEL_ROOT__/weights/nvidia/nemotron-graphic-elements-v1')
CARD_AP = dict(zip(['chart_title', 'x_title', 'y_title', 'xlabel', 'ylabel', 'other',
                   'legend_label', 'legend_title', 'mark_label', 'value_label'],
                  [82.38, 88.77, 89.48, 85.04, 86.22, 55.14, 84.09, 60.61, 49.31, 62.66]))
CARD_AR = dict(zip(CARD_AP, [93.16, 92.31, 92.32, 88.93, 89.40, 79.48, 88.07, 68.42, 73.61, 68.32]))


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda: f.read(8*1024*1024), b''):
            h.update(b)
    return h.hexdigest()


def write(path, data):
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False))
    temp.replace(path)


def run(args):
    started = time.perf_counter()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(MODEL))
    from nemotron_graphic_elements_v1.graphic_element_v1 import Exp
    from nemotron_graphic_elements_v1.model import YoloXWrapper
    from nemotron_graphic_elements_v1.utils import postprocess_preds_graphic_element
    config = Exp(); config.device = args.device
    assert list(CARD_AP) == config.labels
    annotation_path = DATA / 'annotations_native10.json'
    annotation = json.loads(annotation_path.read_text())
    assert len(annotation['images']) == 560 and len(annotation['annotations']) == 11520
    versions = {'python': sys.version.split()[0]}
    for name in ['torch', 'torchvision', 'numpy', 'Pillow', 'pycocotools']:
        versions[name] = importlib.metadata.version(name)
    protocol = {
        'model_id': 'nvidia/nemotron-graphic-elements-v1', 'device': args.device, 'dtype': 'float32',
        'weight_sha256': sha(MODEL / 'nemotron_graphic_elements_v1/weights.pth'),
        'annotation': str(annotation_path), 'annotation_sha256': sha(annotation_path),
        'source_manifest_sha256': sha(DATA / 'source_manifest.json'),
        'runner_sha256': sha(Path(__file__)), 'prepare_sha256': sha(HERE / 'prepare.py'),
        'native_source_sha256': {str(p.relative_to(MODEL)): sha(p) for p in sorted(MODEL.rglob('*.py'))},
        'runtime_versions': versions, 'batch_size': 1, 'input_size': list(config.size),
        'preprocess': 'Official RGB uint8, aspect-preserving bilinear resize, right/bottom pad 114; range 0..255.',
        'conf_threshold': config.conf_thresh, 'nms_iou': config.iou_thresh,
        'class_agnostic_nms': config.class_agnostic, 'postprocess_threshold': config.threshold,
        'label_mapping': {str(i): {'category_id': i+1, 'name': n} for i, n in enumerate(config.labels)},
        'metric': 'COCO bbox AP IoU .50:.05:.95 and AR; maxDets [1,10,100]; all 10 classes independently.',
        'reference': 'https://huggingface.co/nvidia/nemotron-graphic-elements-v1/blob/4a76546bb1bb4cbab3401361c91cf01706321805/README.md',
        'reference_limit': 'Published 560 images and all class counts match; original AP-time extra score threshold is not disclosed.',
        'no_threshold_sweep': True}
    fingerprint = hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest()
    status_path = OUTPUT / 'status.json'
    if status_path.exists():
        old_status = json.loads(status_path.read_text())
        assert old_status['protocol_fingerprint'] == fingerprint, 'Changed frozen protocol; use a new result root.'
        if old_status['state'] == 'complete':
            print('Already complete, no rerun.'); return
    else:
        snapshot = OUTPUT / 'source_snapshot'; snapshot.mkdir(exist_ok=True)
        for name in ['run.py', 'prepare.py']:
            (snapshot / name).write_bytes((HERE / name).read_bytes())
        (OUTPUT / 'annotation.json').write_bytes(annotation_path.read_bytes())
    status = {'sequence': 48, 'model_id': protocol['model_id'], 'state': 'running',
              'protocol': protocol, 'protocol_fingerprint': fingerprint,
              'total_images': 560, 'completed_images': 0,
              'started_at_utc': datetime.now(timezone.utc).isoformat()}
    write(status_path, status)
    try:
        model = config.get_model()
        state = torch.load(MODEL / 'nemotron_graphic_elements_v1/weights.pth', map_location='cpu', weights_only=False)
        model.load_state_dict(state['model'], strict=True)
        model = YoloXWrapper(model, config).eval().to(args.device)
        torch.cuda.reset_peak_memory_stats(args.device)
        predictions_path = OUTPUT / 'predictions.jsonl'
        records = [json.loads(line) for line in predictions_path.open()] if predictions_path.exists() else []
        done = {r['image_id'] for r in records}
        assert len(done) == len(records)
        images = annotation['images'][:args.max_images] if args.max_images else annotation['images']
        native_label_counts = {}
        with predictions_path.open('a') as out:
            for i, info in enumerate(images):
                if info['id'] in done: continue
                with Image.open(DATA / 'images' / info['file_name']) as opened:
                    image = np.asarray(opened.convert('RGB')).copy()
                torch.cuda.synchronize(args.device); tick = time.perf_counter()
                with torch.inference_mode():
                    raw = model(model.preprocess(image), image.shape)[0]
                    boxes, labels, scores = postprocess_preds_graphic_element(raw, model.threshold, model.labels)
                torch.cuda.synchronize(args.device)
                seconds = time.perf_counter() - tick
                if config.normalize_boxes:
                    boxes *= np.array([info['width'], info['height'], info['width'], info['height']])
                detections = []
                for box, label, score in zip(boxes, labels, scores):
                    x1, y1, x2, y2 = map(float, box); label = int(label); score = float(score)
                    assert all(math.isfinite(v) for v in [x1, y1, x2, y2, score])
                    assert 0 <= label < 10 and 0 <= score <= 1 and x2 > x1 and y2 > y1
                    assert -1e-3 <= x1 < x2 <= info['width']+1e-3 and -1e-3 <= y1 < y2 <= info['height']+1e-3
                    detections.append({'image_id': info['id'], 'category_id': label+1, 'native_class_id': label,
                                       'class_name': config.labels[label], 'bbox': [x1, y1, x2-x1, y2-y1], 'score': score})
                    native_label_counts[config.labels[label]] = native_label_counts.get(config.labels[label], 0)+1
                record = {'image_id': info['id'], 'file_name': info['file_name'], 'inference_seconds': seconds, 'detections': detections}
                out.write(json.dumps(record, allow_nan=False)+'\n'); out.flush()
                records.append(record); done.add(info['id'])
                if i == 0 or (i+1) % 50 == 0 or i+1 == len(images): print(f'[{i+1}/{len(images)}] images', flush=True)
        status.update(completed_images=len(records), smoke_passed=len(records) >= 10,
                      peak_gpu_memory_bytes=torch.cuda.max_memory_allocated(args.device))
        del model, state
        torch.cuda.empty_cache()
        print('GPU_INFERENCE_COMPLETE', len(records), flush=True)
        if args.max_images:
            status.update(state='partial_complete', smoke_native_label_counts=native_label_counts)
        else:
            assert done == {i['id'] for i in annotation['images']} and len(records) == 560
            predictions = [d for r in records for d in r['detections']]
            write(OUTPUT / 'predictions.json', predictions)
            gt = COCO(str(annotation_path)); dt = gt.loadRes(predictions)
            evaluator = COCOeval(gt, dt, 'bbox'); evaluator.evaluate(); evaluator.accumulate(); evaluator.summarize()
            names = ['AP', 'AP50', 'AP75', 'AP_small', 'AP_medium', 'AP_large', 'AR1', 'AR10', 'AR100', 'AR_small', 'AR_medium', 'AR_large']
            metrics = dict(zip(names, map(float, evaluator.stats)))
            per_class = {}
            for k, catid in enumerate(evaluator.params.catIds):
                name = gt.cats[catid]['name']; precision = evaluator.eval['precision'][:, :, k, 0, -1]
                recall = evaluator.eval['recall'][:, k, 0, -1]
                ap = float(precision[precision > -1].mean()); ar = float(recall[recall > -1].mean())
                per_class[name] = {'AP': ap, 'AR100': ar, 'model_card_AP_percent': CARD_AP[name], 'model_card_AR_percent': CARD_AR[name],
                                   'AP_difference_percentage_points': ap*100-CARD_AP[name], 'AR_difference_percentage_points': ar*100-CARD_AR[name]}
            metrics.update(per_class=per_class, per_class_AP={n:v['AP'] for n,v in per_class.items()})
            result = {'metrics': metrics, 'unit': 'fraction_0_to_1', 'evaluation_images': 560, 'GT_instances': 11520,
                      'annotation_sha256': protocol['annotation_sha256'], 'protocol_fingerprint': fingerprint,
                      'comparison': 'Recovered published val; all ten class image/box counts match card. Inference defaults frozen.',
                      'inference_seconds_sum': sum(r['inference_seconds'] for r in records),
                      'performance_note': 'GPU shared with other jobs; timings are diagnostic, not controlled throughput benchmark.'}
            write(OUTPUT / 'metrics.json', result)
            status.update(state='complete', metrics=result, predictions_sha256=sha(OUTPUT/'predictions.json'),
                          predictions_jsonl_sha256=sha(predictions_path))
    except Exception as error:
        status.update(state='failed', error=repr(error)); raise
    finally:
        status['elapsed_seconds_this_invocation'] = time.perf_counter()-started
        status['updated_at_utc'] = datetime.now(timezone.utc).isoformat()
        write(status_path, status)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--max-images', type=int, default=10)
    parser.add_argument('--device', default='cuda:0')
    run(parser.parse_args())
