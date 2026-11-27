"""M12 native Ultralytics 8.0.43 validation; keep prior COCO results intact.

Evidence: model card/config.json and best.pt train_args/version. Validation
conf=.001/iou=.7/max_det=300 differ from the card's predict() demonstration.
The author's exact split identity, validation batch and full environment are
not established. These runs are controlled comparisons, not certified replicas.
"""
import argparse
from collections import Counter
import fcntl
import hashlib
import json
from pathlib import Path
import zipfile

MODEL = Path('/data2/gaoyonghan/gpu/model/foduucom/table-detection-and-extraction')
TEST = Path('/data2/gaoyonghan/gpu/dataset/tablebordernet_test/test')
VALID_ZIP = Path('/data2/gaoyonghan/gpu/dataset/tablebordernet_valid/data/valid.zip')
OUT = Path('/home/ubuntu/gaoyonghan/gpu/results/m12_native8043_v1')
WEIGHT_SHA = '60f9c0d863af2c53c6f8c041bf7c4b7cf80dfb3e0383f03224c3f94b6f939c75'


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write(path, data):
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False, default=str, allow_nan=False))
    tmp.replace(path)


def prepare(split):
    root = TEST
    if split == 'valid':
        root = OUT / 'source_valid' / 'valid'
        with zipfile.ZipFile(VALID_ZIP) as z:
            for name in z.namelist():
                parts = Path(name).parts
                if len(parts) != 3 or parts[0] != 'valid' or parts[1] not in ['images', 'labels']:
                    continue
                if Path(name).suffix.lower() not in ['.jpg', '.jpeg', '.png', '.txt']:
                    continue
                dest = root.parent / name
                dest.parent.mkdir(parents=True, exist_ok=True)
                content = z.read(name)
                if dest.exists() and dest.read_bytes() != content:
                    raise ValueError(f'Existing extracted data differs: {dest}')
                if not dest.exists():
                    dest.write_bytes(content)
    images = sorted(p for p in (root / 'images').iterdir() if p.suffix.lower() in ['.jpg', '.jpeg', '.png'])
    labels = list((root / 'labels').glob('*.txt'))
    expected = 70 if split == 'valid' else 34
    if len(images) != expected or {p.stem for p in images} != {p.stem for p in labels}:
        raise ValueError(f'{split}: expected {expected} paired images/labels')
    manifest, counts = [], Counter()
    for image in images:
        label = root / 'labels' / (image.stem + '.txt')
        for row in label.read_text().splitlines():
            if not row.strip():
                continue
            values = list(map(float, row.split()))
            if len(values) != 5 or values[0] not in [0, 1] or not all(0 <= v <= 1 for v in values[1:]) or min(values[3:]) <= 0:
                raise ValueError(f'Invalid YOLO label: {label}: {row}')
            counts[int(values[0])] += 1
        manifest.append({'source': 'foduucom/table-detection-yolo', 'split': split,
                         'image': str(image), 'label': str(label),
                         'image_sha256': sha(image), 'label_sha256': sha(label)})
    if set(counts) != {0, 1}:
        raise ValueError(f'{split} does not contain both classes')
    return root, manifest, dict(counts)


def evaluate(split, device):
    from importlib import import_module
    from importlib.metadata import version
    import ultralytics
    import torch
    from ultralytics.yolo.cfg import get_cfg
    from ultralytics.yolo.v8.detect.val import DetectionValidator
    from ultralytics.yolo.utils import metrics as native_metrics
    native_val = import_module('ultralytics.yolo.v8.detect.val')

    if ultralytics.__version__ != '8.0.43' or version('ultralyticsplus') != '0.0.28':
        raise RuntimeError('Run setup_m12_native.sh and use .venv-m12-ultralytics8043/bin/python')
    if sha(MODEL / 'best.pt') != WEIGHT_SHA:
        raise RuntimeError('Checkpoint changed; previously audited train_args no longer applicable')
    root, manifest, counts = prepare(split)
    out = OUT / split
    out.mkdir(parents=True, exist_ok=True)
    with (out / '.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        # Both val and test keys are explicit; do not relabel test as validation.
        yaml = out / 'data.yaml'
        content = f'path: {json.dumps(str(root))}\ntrain: images\nval: images\ntest: images\nnames: [bordered, borderless]\n'
        yaml.write_text(content)
        options = dict(data=str(yaml), split='val' if split == 'valid' else 'test',
                       task='detect', mode='val', imgsz=640, conf=.001, iou=.7,
                       max_det=300, agnostic_nms=False, single_cls=False,
                       batch=1, workers=0, device=device, half=False, rect=True,
                       augment=False, save_json=True, save_txt=False, plots=False,
                       project=str(OUT), name=split, exist_ok=True, verbose=True)
        protocol = {'sequence': 12, 'model': 'foduucom/table-detection-and-extraction',
                    'weight_sha256': WEIGHT_SHA, 'source_files': manifest,
                    'instances_per_class': counts, 'options': options,
                    'software': {n: version(n) for n in ['ultralytics', 'ultralyticsplus', 'torch', 'torchvision', 'numpy', 'Pillow']},
                    'script_sha256': sha(Path(__file__)),
                    'native_validator_sha256': sha(Path(native_val.__file__)),
                    'native_metrics_sha256': sha(Path(native_metrics.__file__)),
                    'checkpoint_evidence': {'version': '8.0.43', 'train_mode': 'train',
                                            'split': 'val', 'imgsz': 640, 'conf': .001, 'iou': .7, 'max_det': 300},
                    'model_card_predict_example_not_used_for_val': {'conf': .25, 'iou': .45, 'max_det': 1000},
                    'official_reference_AP50': .962,
                    'official_comparable': False,
                    'limitations': ['Exact author split file list is not published; checkpoint only says val.',
                                    'Native 8.0.43 validator/AP implementation, not COCOeval.',
                                    'Reuse installed PyTorch/CUDA; original full dependency lock unavailable.',
                                    'Standalone validation batch=1, FP32, rectangular batching.',
                                    '8.0.43 training-time CUDA validation forces FP16; no claim of identical numerical path.',
                                    'Changing split/version/scoring simultaneously cannot isolate the original gap.']}
        fingerprint = hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest()
        status_path = out / 'status.json'
        if status_path.exists():
            old = json.loads(status_path.read_text())
            if old['fingerprint'] != fingerprint:
                previous_protocol = dict(old['protocol'])
                current_protocol = json.loads(json.dumps(protocol, default=str))
                previous_protocol.pop('script_sha256', None)
                current_protocol.pop('script_sha256', None)
                if old['state'] != 'failed' or previous_protocol != current_protocol:
                    raise ValueError('Protocol/data changed. Preserve this output and choose a new version directory.')
                # Retain the failed attempt when retrying a script-only fix.
                write(out / f"status.failed.{old['fingerprint']}.json", old)
            if old['state'] == 'complete':
                print(f'Already complete: {out}')
                return
        status = {'state': 'running', 'fingerprint': fingerprint, 'protocol': protocol}
        write(status_path, status)
        try:
            print(f'Native validation: {split}, {len(manifest)} images, {counts}', flush=True)
            validator = DetectionValidator(args=get_cfg(overrides=options), save_dir=out)
            validator(model=str(MODEL / 'best.pt'))
            if validator.seen != len(manifest) or validator.nt_per_class.tolist() != [counts[0], counts[1]]:
                raise ValueError('Native loader count differs from source audit; inspect corrupt/duplicate labels')
            if validator.names != {0: 'bordered', 1: 'borderless'}:
                raise ValueError(f'Unexpected model class mapping: {validator.names}')
            scores = {k: float(v) for k, v in validator.metrics.results_dict.items()}
            per_class = {}
            for i, c in enumerate(validator.metrics.ap_class_index):
                per_class[validator.names[int(c)]] = dict(zip(
                    ['precision', 'recall', 'AP50', 'AP'], map(float, validator.metrics.class_result(i))))
            result = {'split': split, 'images': validator.seen, 'unit': 'fraction_0_to_1',
                      'metrics': scores, 'per_class': per_class, 'official_reference_AP50': .962,
                      'AP50_difference_percentage_points': 100 * (scores['metrics/mAP50(B)'] - .962),
                      'official_comparable': False, 'actual_validator_args': vars(validator.args),
                      'actual_half': bool(validator.args.half), 'cuda_runtime': torch.version.cuda}
            write(out / 'metrics.json', result)
            status.update(state='complete', completed_images=validator.seen)
            print(f'Completed {split}: {scores}', flush=True)
        except BaseException as exc:
            status.update(state='failed', error=repr(exc))
            raise
        finally:
            write(status_path, status)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--split', choices=['valid', 'test', 'both'], default='both')
    parser.add_argument('--device', default='0')
    args = parser.parse_args()
    for split in (['valid', 'test'] if args.split == 'both' else [args.split]):
        evaluate(split, args.device)


if __name__ == '__main__':
    main()
