#!/usr/bin/env python3
"""M33 YOLO11x: original checkpoint-version validator on author v11 test split."""
import argparse
from collections import defaultdict
from datetime import datetime
import hashlib
from importlib import import_module
from importlib.metadata import version as _package_version, PackageNotFoundError

def version(name):
    # Audit metadata is optional; actual framework imports still raise on failure.
    try:
        return _package_version(name)
    except PackageNotFoundError:
        return "not-installed-or-no-metadata"

import json
import math
from pathlib import Path
import shutil
import tempfile

from run_m07_native import sha, write

HERE = Path(__file__).resolve().parent
OUT = Path('__OUTPUT_ROOT__/m033')
DATA = Path('__DATASET_ROOT__/roboflow_licenseplates_v11_test/test')
WEIGHT = Path('__MODEL_ROOT__/weights/morsetechlab/yolov11-license-plate-detection/license-plate-finetune-v1x.pt')
WEIGHT_SHA = 'bf87430a3b2d0d0511a153a58eb73e97da6212861d258016cbad65a105068a14'
GT_SHA = '08d7d0b4b6405a17f897f71dc598176080b09ee4d9b656803b240ea88305fdda'
SPEC = {'sequence': 33, 'model_id': 'morsetechlab/yolov11-license-plate-detection', 'variant': 'x',
        'weight': str(WEIGHT), 'weight_sha256': WEIGHT_SHA, 'ultralytics_version': '8.3.123',
        'dataset': 'Roboflow License Plate Recognition v11', 'split': 'test',
        'images': 1020, 'boxes': 1085, 'dataset_relation': 'original_exact',
        'fidelity': 'reference_baseline', 'case_type': 'special',
        'special_tags': ['training_eval_overlap'],
        'entrypoint': 'ultralytics.YOLO(local_x_checkpoint).val',
        'default_parameters': {'imgsz': 640, 'batch': 16, 'dtype': 'float32', 'conf': .001,
                               'iou': .7, 'max_det': 300, 'rect': True, 'augment': False},
        'reference': {'AP': .7260, 'AP50': .9813, 'precision': .9893, 'recall': .9508},
        'reference_url': 'https://huggingface.co/morsetechlab/yolov11-license-plate-detection',
        'known_differences': ['Author acknowledges train/test contamination; do not interpret scores as clean generalization.',
                              'Checkpoint framework version is matched; reuse Torch2.5.1/CUDA12.1, not a complete historical environment.',
                              'Exact author test command is not published; use native standalone val defaults with batch16 explicitly recorded.',
                              'COCO category1 License_Plate -> YOLO class0 via official converter; category0 is an unused parent with no boxes.',
                              'Preserve original images/boxes/test membership; do not deduplicate train/test to manufacture a match.']}


def prepared_root(name):
    """Locate immutable MX-prepared inputs using this model's dataset setting."""
    model = Path(__file__).resolve().parents[2]
    config = json.loads((model / "paths.json").read_text())
    data = Path(__import__('os').environ.get("MODEL_DATASET_ROOT", config["dataset_root"]))
    if not data.is_absolute():
        data = model / data
    return data.resolve() / "prepared" / name


def check():
    # Framework version is recorded, not used as an inference gate.
    assert sha(WEIGHT) == WEIGHT_SHA, 'Author X checkpoint changed'
    assert sha(DATA / '_annotations.coco.json') == GT_SHA, 'Original test annotation changed'
    import ultralytics.utils as utils
    font = utils.USER_CONFIG_DIR / 'Arial.ttf'
    if not font.exists():
        cached = Path('__MODEL_ROOT__/resources/runtime/assets/fonts/Arial.ttf')
        assert cached.is_file(), 'Missing cached Arial.ttf; refusing implicit download'
        shutil.copy2(cached, font)


def prepare():
    root = prepared_root('prepared_m33_root')
    record = json.loads((root / "prepared.json").read_text())
    return record


def evaluate(args, prepared):
    import yaml
    from ultralytics import YOLO
    root = prepared_root('prepared_m33_root')
    data = json.loads((root / 'annotation.json').read_text())
    chosen = sorted(data['images'], key=lambda x: x['id'])
    if args.max_images: chosen = chosen[:args.max_images]
    ids = [im['id'] for im in chosen]
    run = OUT / ('smoke' if args.max_images else 'full') / datetime.now().strftime('run_%Y%m%d_%H%M%S_%f')
    run.mkdir(parents=True)
    listing = run / 'test.txt'
    listing.write_text(''.join(str(root / 'images/test' / (str(im['id']) + Path(im['file_name']).suffix)) + '\n' for im in chosen))
    config = run / 'data.yaml'
    config.write_text(yaml.safe_dump({'path': str(root), 'train': str(listing), 'val': str(listing),
                                     'test': str(listing), 'names': {0: 'License_Plate'}}, sort_keys=False))
    options = dict(data=str(config), split='test', imgsz=640, conf=.001, iou=.7, max_det=300,
                   batch=args.batch, device=args.device, half=False, rect=True, augment=False,
                   agnostic_nms=False, single_cls=False, workers=0, plots=False, save_json=True,
                   project=str(run.parent), name=run.name, exist_ok=True)
    protocol = dict(SPEC, prepared=prepared, options=options,
                    software={p: version(p) for p in ['ultralytics', 'torch', 'torchvision', 'numpy', 'Pillow',
                              'tqdm', 'pandas', 'seaborn', 'py-cpuinfo', 'pytz', 'tzdata']},
                    code_sha256={m: sha(Path(import_module(m).__file__)) for m in ['ultralytics.engine.validator',
                                  'ultralytics.models.yolo.detect.val', 'ultralytics.data.augment', 'ultralytics.utils.metrics']})
    paths = [Path(__file__), HERE / 'run_m33x_native.sh', HERE / 'run_m07_native.py']
    protocol['scripts_sha256'] = {p.name: sha(p) for p in paths}
    for path in paths: shutil.copy2(path, run / path.name)
    identity = {**protocol, 'options': {k:v for k,v in options.items() if k not in ['data', 'project', 'name', 'device']}}
    fingerprint = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    state = {'sequence': 33, 'variant': 'x', 'state': 'running', 'protocol': protocol, 'protocol_fingerprint': fingerprint}
    write(run / 'status.json', state)
    print(f'M33 X results: {run}', flush=True)
    try:
        model = YOLO(str(WEIGHT))
        assert model.ckpt['version'] == '8.3.123' and model.model.yaml['scale'] == 'x'
        assert model.names == {0: 'License_Plate'}
        write(run / 'checkpoint_metadata.json', {k:model.ckpt.get(k) for k in ['version','date','train_args','train_metrics']})
        seen = []
        captured = {}

        def capture(validator):
            actual = sorted(int(Path(p).stem) for p in validator.dataloader.dataset.im_files)
            assert actual == sorted(ids) and validator.seen == len(ids), 'Native loader coverage mismatch'
            assert not validator.is_coco, 'Use native custom-dataset metrics, not COCO protocol switching'
            seen.extend(actual); captured.update({k:float(v) for k,v in validator.metrics.results_dict.items()})
            write(run / 'native_options.json', vars(validator.args))

        model.add_callback('on_val_end', capture)
        model.val(**options)
        assert len(seen) == len(set(seen)) == len(ids) and captured
        pred_file = run / 'predictions.json'
        predictions = json.loads(pred_file.read_text())
        assert set(p['image_id'] for p in predictions) <= set(ids)
        write(run / 'coverage.json', {'image_ids': seen, 'images': len(seen)})
        full = not args.max_images
        result = {'full_dataset': full, 'evaluation_images': len(ids), 'unit': 'fraction_0_to_1',
                  'reference_check': 'numeric_comparison_only' if full else 'smoke_not_reportable',
                  'predictions_sha256': sha(pred_file), 'official_match_verified': False,
                  'training_test_overlap_warning': True}
        if full:
            native = {name:captured[key] for name,key in zip(['precision','recall','AP50','AP'],
                       ['metrics/precision(B)','metrics/recall(B)','metrics/mAP50(B)','metrics/mAP50-95(B)'])}
            assert all(math.isfinite(v) and 0 <= v <= 1 for v in native.values())
            result.update(native_metrics=native, reference=SPEC['reference'],
                          difference_percentage_points={k:100*(v-SPEC['reference'][k]) for k,v in native.items()})
        write(run / 'metrics.json', result)
        state.update(state='complete' if full else 'smoke_complete', completed_images=len(ids))
        print(f'M33 X completed, full={full}: {run}', flush=True)
    except BaseException as exc:
        state.update(state='failed', error=repr(exc)); raise
    finally:
        write(run / 'status.json', state)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--describe', action='store_true')
    parser.add_argument('--prepare', action='store_true')
    parser.add_argument('--device', default='0')
    parser.add_argument('--batch', type=int, default=16)
    parser.add_argument('--max-images', type=int, choices=[0,10], default=0)
    args = parser.parse_args()
    if args.describe: print(json.dumps(SPEC, ensure_ascii=False, indent=2)); return
    if args.batch < 1 or not args.device.isdigit(): parser.error('Use positive batch and one GPU index')
    check(); prepared = prepare()
    if args.prepare: print('PASS: all 1020 test images and 1085 native-converted boxes verified'); return
    evaluate(args, prepared)


if __name__ == '__main__': main()
