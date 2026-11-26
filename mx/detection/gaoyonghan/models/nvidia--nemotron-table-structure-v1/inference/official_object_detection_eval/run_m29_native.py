#!/usr/bin/env python3
"""M29: author's YOLOv10.val, native COCO export, independently verified COCO score."""
import argparse
from datetime import datetime
import hashlib
from importlib import import_module
from importlib.metadata import version
import json
from pathlib import Path
import shutil
import subprocess

from run_m07_native import OUT as COCO_OUT, sha, write
from run_native_18_20_21 import dataset_for_run

HERE = Path(__file__).resolve().parent
SOURCE = Path('/data2/gaoyonghan/gpu_model_standalone_20261110/release/model/检测/nvidia--nemotron-table-structure-v1/resources/runtime/assets/workspace/official_sources/object_detection/yolov10')
REVISION = '453c6e38a51e9d1d5a2aa5fb7f1014a711913397'
MODEL = Path('/data2/gaoyonghan/gpu_model_standalone_20261110/release/model/检测/nvidia--nemotron-table-structure-v1/resources/model_index/jameslahm/yolov10x')
OUT = Path('/data2/gaoyonghan/gpu_model_standalone_20261110/release/model/检测/nvidia--nemotron-table-structure-v1/resources/runtime/assets/workspace/results/m29_native_v1')
REFERENCE = 'https://github.com/THU-MIG/yolov10#validation'


def preflight():
    import ultralytics
    assert version('ultralytics') == '8.1.34'
    assert SOURCE in Path(ultralytics.__file__).resolve().parents, 'Not the author YOLOv10 fork'
    assert subprocess.check_output(['git', '-C', str(SOURCE), 'rev-parse', 'HEAD'], text=True).strip() == REVISION
    assert not subprocess.check_output(['git', '-C', str(SOURCE), 'diff', 'HEAD', '--', 'ultralytics'], text=True), 'Official source changed'
    for name in ['model.safetensors', 'config.json']:
        assert (MODEL / name).is_file(), name
    cfg = json.loads((MODEL / 'config.json').read_text())
    assert cfg['model'] == 'yolov10x.yaml' and cfg['task'] == 'detect' and len(cfg['names']) == 80
    assert sha(MODEL / 'model.safetensors') == 'e0bbf5e028f0ec1f0b26397b09957bccd27618ce13d5e00154709d6da5a62a30', 'Weight changed'
    assert sha(MODEL / 'config.json') == '71093308e6172c49335184252801b2720e2cee6e40fa9636771f62f6907c4ef6', 'Model config changed'
    root = COCO_OUT / 'coco'
    annotation = root / 'annotations/instances_val2017.json'
    assert sha(annotation) == 'e8c7f7908f1d7278341fae127d0da654f102f11bd7b21d8aeefa635b8c810b6f', 'Original COCO annotation changed'
    assert (root / 'prepared.json').is_file() and (root / 'coco.yaml').is_file()
    import_module('pycocotools.cocoeval')
    font = ultralytics.utils.USER_CONFIG_DIR / 'Arial.ttf'
    if not font.exists():
        cached = Path('/data2/gaoyonghan/gpu_model_standalone_20261110/release/model/检测/nvidia--nemotron-table-structure-v1/resources/runtime/assets/fonts/Arial.ttf')
        assert cached.is_file(), 'Missing cached Arial.ttf'
        font.symlink_to(cached)
    return {'source': str(SOURCE), 'revision': REVISION,
            'weight_sha256': sha(MODEL / 'model.safetensors'), 'model_config_sha256': sha(MODEL / 'config.json'),
            'annotation_sha256': sha(annotation),
            'software': {p: version(p) for p in ['ultralytics', 'torch', 'torchvision', 'huggingface-hub', 'safetensors', 'numpy', 'Pillow', 'pycocotools']}}


def evaluate(args, identity):
    from ultralytics import YOLOv10
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    run = OUT / ('smoke' if args.max_images else 'full') / datetime.now().strftime('run_%Y%m%d_%H%M%S_%f')
    run.mkdir(parents=True)
    state = {'state': 'preparing', 'sequence': 29}
    write(run / 'status.json', state)
    print(f'M29 results: {run}', flush=True)
    try:
        # Reuse the accepted M20 COCO reader only: same 5000 images/36781 annotations.
        # No M20 weight, inference routine or result is used.
        annotation, config, dataset = dataset_for_run(20, run, args.max_images)
        ids = dataset['image_ids']
        options = dict(data=str(config), split='val', imgsz=640, conf=.001, max_det=300,
                       batch=args.batch, device=args.device, half=False, rect=True, augment=False,
                       single_cls=False, workers=0, save_json=True, plots=False,
                       project=str(run.parent), name=run.name, exist_ok=True)
        paths = [Path(__file__), HERE / 'run_m29_native.sh', HERE / 'run_m07_native.py',
                 HERE / 'run_native_18_20_21.py', HERE / 'native_18_20_21.json']
        scripts = {p.name: sha(p) for p in paths}
        for path in paths:
            shutil.copy2(path, run / path.name)
        modules = ['ultralytics.models.yolov10.val', 'ultralytics.engine.validator',
                   'ultralytics.models.yolo.detect.val', 'ultralytics.data.augment',
                   'ultralytics.data.build', 'ultralytics.utils.ops', 'pycocotools.cocoeval']
        protocol = dict(identity, model_id='jameslahm/yolov10x', entrypoint='YOLOv10.from_pretrained(...).val(...)',
                        dataset=dataset, options=options, scripts_sha256=scripts,
                        official_code_sha256={m: sha(Path(import_module(m).__file__)) for m in modules},
                        reference_AP_percent=54.4, reference_url=REFERENCE,
                        fidelity='reference_baseline', dataset_relation='original_exact',
                        limitations=['Native validation workflow; default batch16 instead of author example256. Batch is recorded, never silently changed.',
                                     'FP32, conf=.001, imgsz640, max_det300 follow standalone native validation defaults. No extra NMS.',
                                     'Reuse officially converted YOLO labels for loading; final COCOeval uses untouched original annotations including crowd.',
                                     'Result proximity is not proof of identical historical environment or a single cause for previous differences.'])
        numerical = {k: v for k, v in options.items() if k not in ['data', 'project', 'name', 'device']}
        fingerprint = hashlib.sha256(json.dumps(dict(identity, scripts=scripts, options=numerical), sort_keys=True).encode()).hexdigest()
        state.update(state='running', protocol=protocol, protocol_fingerprint=fingerprint)
        write(run / 'status.json', state)
        model = YOLOv10.from_pretrained(str(MODEL), local_files_only=True, strict=True)
        write(run / 'weight_load_audit.json', {'strict_load': True, 'weight_sha256': identity['weight_sha256']})
        captured = {}

        def capture(validator):
            assert validator.is_coco and validator.seen == len(ids)
            actual = sorted(int(Path(p).stem) for p in validator.dataloader.dataset.im_files)
            assert actual == sorted(ids), 'Native dataset image coverage changed'
            captured.update(validator.metrics.results_dict)
            write(run / 'coverage.json', {'images': validator.seen, 'image_ids': actual})
            write(run / 'native_options.json', vars(validator.args))

        model.add_callback('on_val_end', capture)
        model.val(**options)
        assert captured, 'Native validator did not finish'
        # Native eval_json catches scoring failures internally. Independently call
        # the SAME COCO API on its saved output so failure cannot pass as completion.
        gt = COCO(str(annotation))
        predictions = json.loads((run / 'predictions.json').read_text())
        assert predictions and set(p['image_id'] for p in predictions) <= set(ids)
        if args.max_images:
            write(run / 'metrics.json', {'full_dataset': False, 'evaluation_images': len(ids),
                  'reference_check': 'smoke_not_reportable', 'formal_metrics': False,
                  'predictions_sha256': sha(run / 'predictions.json')})
            state.update(state='smoke_complete', completed_images=len(ids))
            print(f'M29 smoke passed: {len(ids)} images; no reportable AP; {run}', flush=True)
            return
        evaluator = COCOeval(gt, gt.loadRes(predictions), 'bbox')
        evaluator.params.imgIds = ids
        evaluator.evaluate(); evaluator.accumulate(); evaluator.summarize()
        names = ['AP', 'AP50', 'AP75', 'AP_small', 'AP_medium', 'AP_large', 'AR1', 'AR10', 'AR100', 'AR_small', 'AR_medium', 'AR_large']
        coco = dict(zip(names, map(float, evaluator.stats)))
        assert 0 <= coco['AP'] <= 1
        full = len(ids) == 5000
        write(run / 'metrics.json', {'unit': 'fraction_0_to_1', 'full_dataset': full, 'evaluation_images': len(ids),
              'native_internal_metrics': captured, 'coco_metrics': coco, 'primary_metric': 'coco_metrics.AP',
              'reference_AP_percent': 54.4, 'difference_percentage_points': coco['AP'] * 100 - 54.4 if full else None,
              'reference_check': 'numeric_comparison_only' if full else 'smoke_not_reportable',
              'predictions_sha256': sha(run / 'predictions.json'), 'official_match_verified': False})
        state.update(state='complete' if full else 'smoke_complete', completed_images=len(ids))
        print(f'M29 completed: COCO AP={coco["AP"] * 100:.4f}%; full={full}; {run}', flush=True)
    except BaseException as exc:
        state.update(state='failed', error=repr(exc))
        raise
    finally:
        write(run / 'status.json', state)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', default='0')
    parser.add_argument('--batch', type=int, default=16)
    parser.add_argument('--max-images', type=int, choices=[0, 10], default=0)
    parser.add_argument('--check', '--describe', action='store_true')
    args = parser.parse_args()
    if args.batch < 1 or not args.device.isdigit():
        parser.error('Use a positive batch and one GPU index')
    identity = preflight()
    if args.check:
        print(json.dumps(dict(identity, model_id='jameslahm/yolov10x', dataset='COCO val2017 / 5000 images / 80 classes',
                              entrypoint='YOLOv10.from_pretrained(..., strict=True).val(...)',
                              imgsz=640, batch=args.batch, dtype='float32', conf=.001, max_det=300,
                              extra_nms=False, evaluator='native validator + original COCOeval bbox',
                              reference_AP_percent=54.4, reference_url=REFERENCE,
                              limitation='Author example batch256; default local batch16 is explicitly recorded'), indent=2)); return
    evaluate(args, identity)


if __name__ == '__main__':
    main()
