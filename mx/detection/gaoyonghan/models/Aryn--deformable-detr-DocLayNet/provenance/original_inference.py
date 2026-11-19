"""Evaluate Aryn on the original 498-page DocLayNet competition dataset."""
import argparse
import collections
import fcntl
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import time
import zipfile

import numpy as np
import torch
from PIL import Image
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from transformers import AutoConfig, AutoImageProcessor, AutoModelForObjectDetection

HERE = Path(__file__).resolve().parent
DATA = Path('/data2/gaoyonghan/gpu/dataset/supplemental_20260905/model19')
WEIGHT = Path('/data2/gaoyonghan/gpu/model/Aryn/deformable-detr-DocLayNet')
OUTPUT = Path('/home/ubuntu/gaoyonghan/gpu/results/remaining25_supplemental_20260905/m19-doclaynet-competition498')


def sha(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def save(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def evaluate(annotation, predictions, image_ids):
    gt = COCO()
    gt.dataset = annotation
    gt.createIndex()
    selected = [dict(p) for p in predictions if p['image_id'] in image_ids]
    if selected:
        dt = gt.loadRes(selected)
    else:
        dt = COCO()
        dt.dataset = {**annotation, 'annotations': []}
        dt.createIndex()
    evaluator = COCOeval(gt, dt, 'bbox')
    evaluator.params.imgIds = sorted(image_ids)
    evaluator.evaluate()
    evaluator.accumulate()
    evaluator.summarize()
    names = ['AP', 'AP50', 'AP75', 'AP_small', 'AP_medium', 'AP_large',
             'AR1', 'AR10', 'AR100', 'AR_small', 'AR_medium', 'AR_large']
    metrics = {k: float(v) if v >= 0 else None for k, v in zip(names, evaluator.stats)}
    category_names = {c['id']: c['name'] for c in annotation['categories']}
    metrics['per_class_AP'] = {}
    for i, cat in enumerate(evaluator.params.catIds):
        values = evaluator.eval['precision'][:, :, i, 0, -1]
        values = values[values > -1]
        metrics['per_class_AP'][category_names[cat]] = float(values.mean()) if values.size else None
    metrics['images'] = len(image_ids)
    metrics['gt_instances_per_class'] = dict(collections.Counter(
        category_names[a['category_id']] for a in annotation['annotations'] if a['image_id'] in image_ids))
    metrics['predictions'] = len(selected)
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--max-images', type=int, choices=[0, 10], default=10)
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    with (OUTPUT / '.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        prepared = json.loads((DATA / 'prepared.json').read_text())
        annotation = json.loads((DATA / 'annotation.json').read_text())
        assert len(annotation['images']) == 498 and len(annotation['categories']) == 11
        annotation_hash = sha(DATA / 'annotation.json')
        assert annotation_hash == prepared['annotation_sha256']
        domain_mapping = json.loads((DATA / 'domain_mapping.json').read_text())
        protocol = {
            'sequence': 19, 'model_id': 'Aryn/deformable-detr-DocLayNet',
            'hf_revision': 'd5503a90ae08dd43565de6984a5dd7924cad2400',
            'weight_sha256': sha(WEIGHT / 'model.safetensors'),
            'config_sha256': sha(WEIGHT / 'config.json'),
            'preprocessor_config_sha256': sha(WEIGHT / 'preprocessor_config.json'),
            'script_sha256': sha(Path(__file__)), 'prepare_script_sha256': sha(HERE / 'prepare.py'),
            'annotation_sha256': annotation_hash, 'dataset': prepared,
            'device': args.device, 'dtype': 'float32', 'batch_size': 1,
            'score_threshold': 0.0, 'nms': None, 'processor_top_k': 100,
            'COCO_maxDets': [1, 10, 100], 'IoUs': '.50:.05:.95',
            'primary_metric': 'pooled COCO AP; official four-domain mean unavailable until verified domain mapping is obtained',
            'secondary_metric': 'pooled COCO AP across all 498 images',
            'label_mapping': 'identity for native 1..11 DocLayNet classes; exclude N/A=0',
            'runtime_versions': {n: importlib.metadata.version(n) for n in ['torch', 'torchvision', 'transformers', 'huggingface-hub', 'numpy', 'Pillow', 'pycocotools']},
        }
        fingerprint = hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest()
        status_path = OUTPUT / 'status.json'
        old = json.loads(status_path.read_text()) if status_path.exists() else {}
        if old and old['protocol_fingerprint'] != fingerprint:
            raise RuntimeError('Protocol changed; use a different result directory')
        if old.get('state') == 'complete':
            print('Already complete', flush=True)
            return
        if args.max_images == 0 and not old.get('smoke_passed'):
            raise RuntimeError('First complete --max-images 10')
        status = {'model_id': protocol['model_id'], 'sequence': 19, 'state': 'running',
                  'protocol': protocol, 'protocol_fingerprint': fingerprint,
                  'total_images': 498, 'annotation_path': str(OUTPUT / 'annotation.json'),
                  'smoke_passed': old.get('smoke_passed', False), 'started_at': time.time()}
        save(status_path, status)
        try:
            save(OUTPUT / 'annotation.json', annotation)
            save(OUTPUT / 'domain_mapping.json', domain_mapping)
            processor = AutoImageProcessor.from_pretrained(WEIGHT, local_files_only=True, use_fast=False)
            config = AutoConfig.from_pretrained(WEIGHT, local_files_only=True)
            config.use_pretrained_backbone = False
            model, loading = AutoModelForObjectDetection.from_pretrained(WEIGHT, config=config, local_files_only=True, output_loading_info=True)
            assert not loading['missing_keys'] and not loading['unexpected_keys'] and not loading.get('mismatched_keys'), loading
            assert all(model.config.id2label[c['id']] == c['name'] for c in annotation['categories'])
            model.to(device=args.device, dtype=torch.float32).eval()
            save(OUTPUT / 'loading_info.json', loading)
            save(OUTPUT / 'effective_processor.json', processor.to_dict())
            rows_path = OUTPUT / 'predictions.jsonl'
            rows = [json.loads(line) for line in rows_path.open()] if rows_path.exists() else []
            existing = {row['image_id'] for row in rows}
            all_ids = {image['id'] for image in annotation['images']}
            assert len(rows) == len(existing) and existing <= all_ids
            selected = annotation['images'][:args.max_images] if args.max_images else annotation['images']
            archive = zipfile.ZipFile(DATA / prepared['image_archive'])
            members = prepared['image_members']
            for index, info in enumerate(selected, 1):
                if info['id'] in existing:
                    continue
                with archive.open(members[info['file_name']]) as f, Image.open(f) as opened:
                    image = opened.convert('RGB')
                assert image.size == (info['width'], info['height'])
                inputs = processor(images=image, return_tensors='pt').to(args.device)
                with torch.inference_mode():
                    output = model(**inputs)
                target = torch.tensor([[image.height, image.width]], device=args.device)
                result = processor.post_process_object_detection(output, threshold=0.0, target_sizes=target, top_k=100)[0]
                detections = []
                for box, score, label in zip(result['boxes'].cpu().tolist(), result['scores'].cpu().tolist(), result['labels'].cpu().tolist()):
                    if label == 0:
                        continue
                    x1, y1, x2, y2 = box
                    assert label in range(1, 12) and x2 >= x1 and y2 >= y1 and 0 < score <= 1
                    assert all(math.isfinite(x) for x in [*box, score])
                    detections.append({'image_id': info['id'], 'category_id': label, 'bbox': [x1, y1, x2-x1, y2-y1], 'score': score})
                row = {'image_id': info['id'], 'file_name': info['file_name'], 'domain': domain_mapping[str(info['id'])], 'detections': detections}
                with rows_path.open('a') as f:
                    f.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')
                rows.append(row)
                if index % 25 == 0 or index == len(selected):
                    print(f'{index}/{len(selected)}', flush=True)
            archive.close()
            status.update({'completed_images': len(rows), 'smoke_passed': True, 'state': 'partial_complete'})
            if args.max_images == 0:
                assert len(rows) == 498 and {r['image_id'] for r in rows} == all_ids
                predictions = [p for row in rows for p in row['detections']]
                save(OUTPUT / 'predictions.json', predictions)
                domains = {}
                for domain in sorted(set(d for d in domain_mapping.values() if d is not None)):
                    ids = {int(i) for i, d in domain_mapping.items() if d == domain}
                    domains[domain] = evaluate(annotation, predictions, ids)
                primary = float(np.mean([v['AP'] for v in domains.values()])) if len(domains) == 4 else None
                metrics = {'unit': 'fraction_0_to_1', 'evaluation_images': 498,
                           'competition_AP': primary, 'competition_AR100': float(np.mean([v['AR100'] for v in domains.values()])) if len(domains) == 4 else None,
                           'domain_mapping_state': prepared['domain_mapping_state'], 'domain_mapping_note': prepared['domain_note'],
                           'per_domain': domains, 'pooled': evaluate(annotation, predictions, all_ids),
                           'reference': {'paper_competition_AP': .602, 'paper_competition_AR': .743, 'model_card_box_mAP': .571,
                                         'paper_delta_percentage_points': (primary-.602)*100 if primary is not None else None,
                                         'limitation': 'Author checkpoint and full postprocessing unspecified; model-card 57.1 dataset/protocol unresolved.'},
                           'protocol_fingerprint': fingerprint, 'annotation_sha256': annotation_hash,
                           'predictions_sha256': sha(OUTPUT / 'predictions.json')}
                save(OUTPUT / 'metrics.json', metrics)
                status.update({'state': 'complete', 'metrics': metrics})
        except Exception as e:
            status.update({'state': 'failed', 'error': repr(e)})
            raise
        finally:
            status['elapsed_seconds'] = time.time() - status['started_at']
            save(status_path, status)


if __name__ == '__main__':
    main()
