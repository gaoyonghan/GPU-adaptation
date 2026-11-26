"""Official NVIDIA inference entry on the frozen 1000-image public benchmark.

Only checkpoint path resolution, batch orchestration and COCO scoring are ours.
The model card does not publish a complete official benchmark evaluator.
"""
import argparse
from collections import Counter
from datetime import datetime
from importlib.metadata import version
import json
import os
from pathlib import Path
import shutil
import sys
from unittest.mock import patch

from run_model import core, digest, file_hash, metrics

HERE = Path(__file__).resolve().parent
MODEL = Path('__MODEL_ROOT__/weights/nvidia/nemotron-table-structure-v1')
DATA = Path('__MODEL_ROOT__/resources/runtime/assets/workspace/results/remaining25_prepared/pubtables_structure_1000')
OLD = Path('__MODEL_ROOT__/resources/runtime/assets/workspace/results/remaining25_nemotron_v2/m37-nemotron-table-structure-v1/structure')
OUT = Path('__MODEL_ROOT__/resources/runtime/assets/workspace/results/m37_native_entry_v1')
WEIGHT = MODEL / 'nemotron_table_structure_v1/weights.pth'


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False))


def preflight():
    from PIL import Image
    old = json.loads((OLD / 'status.json').read_text())
    annotation = core.ensure_coco_fields(json.loads((DATA / 'annotations.json').read_text()))
    assert digest(annotation) == old['annotation_sha256'], 'Frozen ground truth changed'
    assert file_hash(WEIGHT) == old['protocol']['weight_sha256'], 'Checkpoint changed'
    sources = {str(p.relative_to(WEIGHT.parent)): file_hash(p) for p in WEIGHT.parent.rglob('*.py')}
    assert sources == old['protocol']['native_source_sha256'], 'Official source changed since v2'
    ids = [im['id'] for im in annotation['images']]
    assert len(ids) == len(set(ids)) == 1000
    assert annotation['categories'] == [dict(id=1, name='cell'), dict(id=2, name='row'), dict(id=3, name='column')]
    counts = Counter(ann['category_id'] for ann in annotation['annotations'])
    assert counts == {1: 63766, 2: 12947, 3: 5416}
    image_ids = set(ids)
    assert all(a['image_id'] in image_ids for a in annotation['annotations'])
    images = []
    for im in annotation['images']:
        path = DATA / 'images' / im['file_name']
        with Image.open(path) as image:
            assert image.size == (im['width'], im['height']), str(path)
            image.verify()
        images.append(dict(image_id=im['id'], path=str(path), sha256=file_hash(path)))
    audit = dict(sequence=37, model_id=old['model_id'], weight_sha256=file_hash(WEIGHT),
                 dataset_relation='public_replacement', official_comparable=False,
                 annotation_sha256=digest(annotation), source=json.loads((DATA / 'source.json').read_text()),
                 source_manifest_sha256=file_hash(DATA / 'source.json'), image_files=images,
                 native_source_sha256=sources, images=1000, instances_per_class=dict(counts),
                 entrypoint='define_model -> model.preprocess -> model(x, img.shape) -> postprocess_preds_table_structure',
                 entrypoint_adjustment='Resolve the exact HF checkpoint request to its SHA-verified local file; no model/config/source edits.',
                 evaluator='pycocotools COCOeval bbox, IoU .50:.05:.95, maxDets=[1,10,100]; project scoring wrapper, NOT NVIDIA official evaluation code',
                 dataset_note='Same fixed 1000 of 93834 PubTables test images; cell boxes derived from GT structures. Not original Azure/Digital Corpora evaluation.',
                 old_result_directory=str(OLD), old_metrics=json.loads((OLD / 'metrics.json').read_text())['metrics'],
                 model_card_reference=dict(cell_AP=58.365, row_AP=76.992, column_AP=85.293),
                 reference_url='https://huggingface.co/nvidia/nemotron-table-structure-v1',
                 software={n: version(n) for n in ['torch', 'torchvision', 'numpy', 'Pillow', 'pandas', 'matplotlib', 'huggingface-hub', 'pycocotools']},
                 project_source_sha256={str(p): file_hash(p) for p in
                     [Path(__file__), HERE / 'run_m37_native.sh', HERE / 'run_model.py',
                      HERE.parent / 'official_object_detection_eval/run_model.py']})
    return annotation, audit


def load_model():
    sys.path.insert(0, str(MODEL))
    from nemotron_table_structure_v1 import define_model
    from nemotron_table_structure_v1 import model as native

    def local_checkpoint(repo_id, filename, **kwargs):
        assert repo_id == 'nvidia/nemotron-table-structure-v1'
        assert filename == 'nemotron_table_structure_v1/weights.pth'
        return str(WEIGHT)

    with patch.object(native, 'hf_hub_download', local_checkpoint):
        model = define_model('table_structure_v1')
    assert model.labels == ['border', 'cell', 'row', 'column', 'header']
    assert tuple(model.img_size) == (1024, 1024) and model.normalize_boxes
    assert (model.conf_thresh, model.iou_thresh, model.threshold) == (.01, .25, .05)
    assert not model.class_agnostic and not model.training
    return model


def to_coco(image_id, boxes, labels, scores, width, height):
    import numpy as np
    assert len(boxes) == len(labels) == len(scores)
    assert np.isfinite(boxes).all() and np.isfinite(scores).all()
    pixel_boxes = boxes.copy()
    pixel_boxes *= np.array([width, height, width, height])  # Same arithmetic as v2.
    detections = []
    for box, pixel_box, label, score in zip(boxes, pixel_boxes, labels, scores):
        assert int(label) == label and int(label) in range(5)
        assert 0 <= float(score) <= 1
        if int(label) not in [1, 2, 3]:  # NVIDIA marks border/header as unused.
            continue
        x1, y1, x2, y2 = [float(v) for v in box]
        assert 0 <= x1 <= x2 <= 1.000001 and 0 <= y1 <= y2 <= 1.000001
        x1, y1, x2, y2 = [float(v) for v in pixel_box]
        detections.append(dict(image_id=image_id, category_id=int(label), score=float(score),
                               bbox=[x1, y1, x2-x1, y2-y1]))
    return detections


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--max-images', type=int, choices=[0, 10], default=0)
    parser.add_argument('--preflight', action='store_true', help='CPU checkpoint/data check only, no model forward')
    args = parser.parse_args()
    if args.preflight:
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
    annotation, audit = preflight()
    if args.preflight:
        model = load_model()
        assert next(model.parameters()).device.type == 'cpu'
        print(json.dumps({'state': 'ready', 'images': 1000, 'boxes': 82129, 'classes': 3,
                          'strict_official_checkpoint_load': True, 'model_forward_executed': False}))
        return
    import numpy as np
    import torch
    from PIL import Image
    assert torch.cuda.is_available(), 'CUDA unavailable; no CPU inference fallback'
    run = OUT / ('smoke' if args.max_images else 'full') / datetime.now().strftime('run_%Y%m%d_%H%M%S_%f')
    run.mkdir(parents=True)
    for name in ['run_m37_native.py', 'run_m37_native.sh', 'run_model.py']:
        shutil.copy2(HERE / name, run / name)
    audit.update(device='cuda:0', visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'), dtype='float32', batch=1,
                 conf=.01, nms_iou=.25, postprocess_threshold=.05, class_agnostic_nms=False,
                 input_size=[1024, 1024], prediction_limit='official code, no additional cap')
    write(run / 'protocol.json', audit)
    state = dict(state='running', full_dataset=not bool(args.max_images), protocol_fingerprint=digest(audit))
    write(run / 'status.json', state)
    try:
        model = load_model()
        from nemotron_table_structure_v1 import postprocess_preds_table_structure
        selected = annotation['images'][:args.max_images] if args.max_images else annotation['images']
        predictions, seen = [], []
        with (run / 'native_predictions.jsonl').open('w') as stream:
            for info in selected:
                with Image.open(DATA / 'images' / info['file_name']) as image:
                    array = np.array(image.convert('RGB'))
                with torch.inference_mode():
                    preds = model(model.preprocess(array), array.shape)[0]
                boxes, labels, scores = postprocess_preds_table_structure(preds, model.threshold, model.labels)
                stream.write(json.dumps(dict(image_id=info['id'], normalized_xyxy=boxes.tolist(),
                                              labels=labels.tolist(), scores=scores.tolist()), allow_nan=False) + '\n')
                stream.flush()
                predictions.extend(to_coco(info['id'], boxes, labels, scores, info['width'], info['height']))
                seen.append(info['id'])
                if len(seen) == 1 or len(seen) % 50 == 0 or len(seen) == len(selected):
                    print(f'M37: {len(seen)}/{len(selected)} images', flush=True)
        assert seen == [i['id'] for i in selected] and len(seen) == len(set(seen))
        write(run / 'coverage.json', dict(images=len(seen), image_ids=seen))
        write(run / 'predictions.json', predictions)
        if not args.max_images:
            scored = metrics(annotation, predictions)
            assert all(scored['per_class_AP'][name] is not None for name in ['cell', 'row', 'column'])
            write(run / 'metrics.json', dict(metrics=scored, unit='fraction_0_to_1', images=len(seen),
                                            official_comparable=False, dataset_relation='public_replacement',
                                            predictions_sha256=file_hash(run / 'predictions.json'),
                                            old_metrics=audit['old_metrics'],
                                            AP_difference_from_old_percentage_points=100*(scored['AP']-audit['old_metrics']['AP'])))
        state.update(state='smoke_complete' if args.max_images else 'complete', completed_images=len(seen))
        print(f'M37 completed: {run}', flush=True)
    except BaseException as exc:
        state.update(state='failed', error=repr(exc))
        raise
    finally:
        write(run / 'status.json', state)


if __name__ == '__main__':
    main()
