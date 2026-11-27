"""M04/M07: official YOLO.val on COCO val2017; no custom inference or AP algorithm.

Keep Ultralytics 8.4.0 for the first controlled workflow comparison. The
checkpoints record older versions, so this is not a full historical reproduction.
Native labels use Ultralytics convert_coco (crowd omitted, duplicate boxes
deduplicated); the final COCO score uses the untouched official JSON, including
crowd annotations. Native AP and COCO AP are therefore reported separately.
"""
import argparse
from datetime import datetime
import fcntl
import hashlib
import json
from pathlib import Path
import tempfile
import zipfile

SOURCE = Path('__DATASET_ROOT__/COCO2017')
WEIGHT = Path('__MODEL_ROOT__/weights/Ultralytics/YOLO11/yolo11n.pt')
OUT = Path('__OUTPUT_ROOT__/m029/shared_preparation')


def prepared_root(name):
    """Locate immutable MX-prepared inputs using this model's dataset setting."""
    model = Path(__file__).resolve().parents[2]
    config = json.loads((model / "paths.json").read_text())
    data = Path(__import__('os').environ.get("MODEL_DATASET_ROOT", config["dataset_root"]))
    if not data.is_absolute():
        data = model / data
    return data.resolve() / "prepared" / name


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write(path, data):
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2, default=str, allow_nan=False))
    tmp.replace(path)


def prepare():
    root = prepared_root("prepared_coco_root")
    return root, json.loads((root / "prepared.json").read_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-number', type=int, choices=[4, 7], default=7)
    parser.add_argument('--device', default='0')
    parser.add_argument('--batch', type=int, default=16, help='Native validation batch; fixed and recorded (default 16).')
    args = parser.parse_args()
    if args.batch < 1:
        parser.error('--batch must be positive')
    from importlib import import_module
    from importlib.metadata import version
    import ultralytics
    from ultralytics import YOLO
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    assert ultralytics.__version__ == '8.4.0'
    if args.model_number == 4:
        model_id, checkpoint_version, reference = 'Ultralytics/YOLOv8', '8.0.0.dev0', 37.3
        weight = Path('__MODEL_ROOT__/weights/Ultralytics/YOLOv8/yolov8n.pt')
        output = OUT.parent / 'm04_native840_v1'
    else:
        model_id, checkpoint_version, reference = 'Ultralytics/YOLO11', '8.2.100', 39.5
        weight, output = WEIGHT, OUT
    # Both models share the official COCO preparation, but never share run outputs.
    OUT.mkdir(parents=True, exist_ok=True)
    output.mkdir(parents=True, exist_ok=True)
    with (OUT / '.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        root, dataset = prepare()
        label_hash = hashlib.sha256()
        for label in sorted((root / 'labels/val2017').glob('*.txt')):
            label_hash.update(label.name.encode() + b'\0' + label.read_bytes() + b'\0')
        dataset = {**dataset, 'derived_labels_sha256': label_hash.hexdigest(),
                   'runtime_yaml_sha256': sha(root / 'coco.yaml'),
                   'runtime_image_list_sha256': sha(root / 'val2017.txt')}
        run = output / datetime.now().strftime('run_%Y%m%d_%H%M%S_%f')
        run.mkdir()
        options = dict(data=str(root / 'coco.yaml'), split='val', imgsz=640,
                       conf=.001, iou=.7, max_det=300, batch=args.batch, device=args.device,
                       half=False, rect=True, augment=False, agnostic_nms=False,
                       single_cls=False, workers=0, save_json=True, plots=False,
                       project=str(output), name=run.name, exist_ok=True)
        protocol = {'sequence': args.model_number, 'model': model_id, 'weight': str(weight),
                    'weight_sha256': sha(weight), 'checkpoint_version': checkpoint_version,
                    'dataset': dataset, 'options': options, 'script_sha256': sha(Path(__file__)),
                    'software': {n: version(n) for n in ['ultralytics', 'torch', 'torchvision', 'numpy',
                                                       'Pillow', 'pycocotools', 'faster-coco-eval']},
                    'official_code_sha256': {n: sha(Path(import_module(n).__file__)) for n in [
                        'ultralytics.models.yolo.detect.val', 'ultralytics.data.converter',
                        'ultralytics.data.base', 'ultralytics.data.augment', 'ultralytics.utils.metrics']},
                    'reference_AP_percent': reference, 'reference_url': f'https://huggingface.co/{model_id}',
                    'limitations': [f'8.4.0 runtime retained; checkpoint metadata is {checkpoint_version}.',
                                    'Native val replaces predict plus external scoring; batch changes from 1 to recorded native batch.',
                                    'Do not infer a single causal factor or strict reproduction from score proximity.']}
        state = {'state': 'running', 'protocol': protocol}
        write(run / 'status.json', state)
        print(f'Results: {run}', flush=True)
        try:
            captured = {}
            model = YOLO(str(weight))

            def capture(validator):
                assert validator.is_coco and validator.seen == 5000
                actual_ids = sorted(int(Path(p).stem) for p in validator.dataloader.dataset.im_files)
                assert actual_ids == dataset['image_ids'], 'Native loader skipped or changed images'
                captured.update(native_metrics=dict(validator.metrics.results_dict),
                                actual_args=vars(validator.args).copy(), images=validator.seen)

            model.add_callback('on_val_end', capture)
            model.val(**options)
            assert captured, 'Native validation did not complete'
            # Audit the SAME native predictions with standard COCOeval, without altering boxes or scores.
            gt = COCO(str(root / 'annotations/instances_val2017.json'))
            dt = gt.loadRes(str(run / 'predictions.json'))
            evaluator = COCOeval(gt, dt, 'bbox')
            evaluator.params.imgIds = dataset['image_ids']
            evaluator.evaluate()
            evaluator.accumulate()
            evaluator.summarize()
            keys = ['AP', 'AP50', 'AP75', 'AP_small', 'AP_medium', 'AP_large',
                    'AR1', 'AR10', 'AR100', 'AR_small', 'AR_medium', 'AR_large']
            coco = dict(zip(keys, map(float, evaluator.stats)))
            write(run / 'metrics.json', {**captured, 'unit': 'fraction_0_to_1',
                  'coco_metrics': coco, 'reference_AP_percent': reference,
                  'coco_AP_difference_percentage_points': coco['AP'] * 100 - reference,
                  'native_AP_difference_percentage_points': captured['native_metrics']['metrics/mAP50-95(B)'] * 100 - reference})
            state.update(state='complete', completed_images=5000)
            print(f'Completed: native AP={captured["native_metrics"]["metrics/mAP50-95(B)"] * 100:.4f}%; '
                  f'COCO AP={coco["AP"] * 100:.4f}%; reference={reference}%', flush=True)
        except BaseException as exc:
            state.update(state='failed', error=repr(exc))
            raise
        finally:
            write(run / 'status.json', state)


if __name__ == '__main__':
    main()
