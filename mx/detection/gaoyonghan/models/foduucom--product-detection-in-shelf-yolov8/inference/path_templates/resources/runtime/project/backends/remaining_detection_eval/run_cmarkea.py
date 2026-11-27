"""Evaluate both outputs of the actual DetrForSegmentation checkpoint."""
import argparse
import fcntl
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from pycocotools import mask as mask_api
from transformers import AutoConfig, AutoImageProcessor, DetrForSegmentation

from run_model import core, digest, file_hash, metrics

MODEL = Path('__MODEL_ROOT__/weights/cmarkea/detr-layout-detection')
DATA = Path('__DATASET_ROOT__')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--max-images', type=int, default=10)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--output-root', type=Path, default=Path('__MODEL_ROOT__/resources/runtime/assets/workspace/results/remaining25_cmarkea_v1'))
    args = parser.parse_args()
    out = args.output_root
    out.mkdir(parents=True, exist_ok=True)
    with (out / '.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        prepared = core.prepare_coco_single_zip({'archive': 'doclaynet_test/DocLayNet_core.zip',
                                                 'annotation_member': 'COCO/test.json'}, DATA)
        annotation = prepared.annotation
        assert len(annotation['images']) == 4999
        protocol = {'model_id': 'cmarkea/detr-layout-detection', 'sequence': 42,
                    'model_class': 'DetrForSegmentation', 'weight_sha256': file_hash(MODEL / 'model.safetensors'),
                    'annotation_sha256': digest(annotation), 'script_sha256': file_hash(Path(__file__)),
                    'helpers_sha256': file_hash(Path(__file__).with_name('run_model.py')),
                    'runtime_versions': core.runtime_versions(), 'dataset': 'DocLayNet test, 4999 pages',
                    'tasks': ['bbox_detection_classification', 'semantic_segmentation'],
                    'fidelity': 'reference_baseline', 'dataset_relation': 'original_protocol_incomplete',
                    'device': args.device, 'dtype': 'float32', 'box_threshold': 0.0,
                    'mask_score_threshold': 0.4, 'mask_threshold': 0.5,
                    'pixel_protocol': 'Independent per-class unions of instance masks; background=complement of all foreground. No arbitrary layer ordering.',
                    'reference_check': 'not_officially_comparable; author 500-page subset unknown'}
        fingerprint = digest(protocol)
        status_path = out / 'status.json'
        previous = json.loads(status_path.read_text()) if status_path.exists() else None
        if previous and previous['protocol_fingerprint'] != fingerprint:
            raise RuntimeError('Protocol changed; choose a new output root')
        if previous and previous['state'] == 'complete':
            return
        if args.max_images == 0 and not (previous and previous.get('smoke_passed')):
            raise RuntimeError('First run --max-images 10')
        state = {'model_id': protocol['model_id'], 'sequence': 42, 'state': 'running',
                 'protocol': protocol, 'protocol_fingerprint': fingerprint, 'started_at': time.time()}
        core.atomic_json(status_path, state)
        try:
            config = AutoConfig.from_pretrained(MODEL, local_files_only=True)
            config.use_pretrained_backbone = False
            model, loading = DetrForSegmentation.from_pretrained(MODEL, config=config, local_files_only=True, output_loading_info=True)
            assert not loading['missing_keys'] and not loading.get('mismatched_keys'), loading
            assert all(k.endswith('num_batches_tracked') for k in loading['unexpected_keys']), loading
            state['loading_info'] = loading
            model = model.eval().to(args.device)
            processor = AutoImageProcessor.from_pretrained(MODEL, local_files_only=True)
            by_image = defaultdict(list)
            for ann in annotation['annotations']:
                by_image[ann['image_id']].append(ann)
            records = core.read_jsonl(out / 'predictions.jsonl')
            existing = {r['image_id'] for r in records}
            assert len(existing) == len(records)
            images = annotation['images'][:args.max_images] if args.max_images else annotation['images']
            torch.cuda.reset_peak_memory_stats(args.device)
            for index, info in enumerate(images, 1):
                if info['id'] in existing:
                    continue
                image = prepared.read_image(info)
                width, height = image.size
                with torch.inference_mode():
                    outputs = model(**processor(images=image, return_tensors='pt').to(args.device))
                    boxes = processor.post_process_object_detection(outputs, threshold=0.0, target_sizes=[(height, width)])[0]
                    masks = processor.post_process_segmentation(outputs, target_sizes=[(height, width)], threshold=0.4, mask_threshold=0.5)[0]
                detections = []
                for box, label, score in zip(boxes['boxes'].cpu().tolist(), boxes['labels'].cpu().tolist(), boxes['scores'].cpu().tolist()):
                    x1, y1, x2, y2 = box
                    detections.append({'image_id': info['id'], 'category_id': label+1,
                                       'bbox': [x1,y1,x2-x1,y2-y1], 'score': score})
                predicted = np.zeros((12, height, width), dtype=bool)
                encoded = []
                for mask, label, score in zip(masks['masks'].cpu().numpy(), masks['labels'].cpu().tolist(), masks['scores'].cpu().tolist()):
                    mask = mask.astype(bool)
                    predicted[label+1] |= mask
                    rle = mask_api.encode(np.asfortranarray(mask.astype(np.uint8)))
                    rle['counts'] = rle['counts'].decode('ascii')
                    encoded.append({'category_id': label+1, 'score': score, 'segmentation': rle})
                target = np.zeros_like(predicted)
                for ann in by_image[info['id']]:
                    rle = mask_api.frPyObjects(ann['segmentation'], height, width)
                    target[ann['category_id']] |= mask_api.decode(mask_api.merge(rle)).astype(bool)
                predicted[0] = ~predicted[1:].any(axis=0)
                target[0] = ~target[1:].any(axis=0)
                tp = (predicted & target).sum(axis=(1,2))
                fp = (predicted & ~target).sum(axis=(1,2))
                fn = (~predicted & target).sum(axis=(1,2))
                core.append_jsonl(out / 'predictions.jsonl', {'image_id': info['id'], 'detections': detections,
                    'masks': encoded, 'pixel_tp_fp_fn': np.stack([tp,fp,fn],axis=1).tolist()})
                if index == 1 or index % 100 == 0 or index == len(images):
                    print('cmarkea', index, '/', len(images), flush=True)
            records = core.read_jsonl(out / 'predictions.jsonl')
            completed = {r['image_id'] for r in records}
            assert len(completed) == len(records) and {i['id'] for i in images} <= completed
            state.update(state='partial_complete', smoke_passed=True, completed_images=len(records), total_images=4999)
            if args.max_images == 0:
                assert completed == {i['id'] for i in annotation['images']}
                predictions = [p for r in records for p in r['detections']]
                pixel = np.array([r['pixel_tp_fp_fn'] for r in records],dtype=np.int64).sum(axis=0)
                f1 = 2*pixel[:,0]/np.maximum(1,2*pixel[:,0]+pixel[:,1]+pixel[:,2])
                names = ['Background']+[c['name'] for c in annotation['categories']]
                result = {'bbox': metrics(annotation,predictions), 'semantic_F1': dict(zip(names,f1.tolist())),
                          'semantic_macro_foreground_F1': float(f1[1:].mean()), 'pixel_counts': pixel.tolist(),
                          'evaluation_images':4999, 'protocol_fingerprint':fingerprint,
                          'reference_check':'not_officially_comparable', 'unit':'fraction_0_to_1'}
                core.atomic_json(out/'metrics.json',result)
                state.update(state='complete',metrics=result)
            state['peak_gpu_memory_bytes'] = torch.cuda.max_memory_allocated(args.device)
        except BaseException as exc:
            state.update(state='failed',error=repr(exc))
            raise
        finally:
            prepared.close()
            state['elapsed_seconds']=time.time()-state['started_at']
            core.atomic_json(status_path,state)


if __name__=='__main__':
    main()
