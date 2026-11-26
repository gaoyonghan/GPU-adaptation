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

SOURCE = Path('/home/ubuntu/文档/ChatGPT/idea 2/outputs/launch_audit_20261005/migration_inputs/datasets/COCO2017')
WEIGHT = Path('/data2/gaoyonghan/gpu_model_standalone_20261110/release/model/检测/iitolstykh--YOLO-Face-Person-Detector/resources/model_index/Ultralytics/YOLO11/yolo11n.pt')
OUT = Path('/data2/gaoyonghan/gpu_model_standalone_20261110/release/model/检测/iitolstykh--YOLO-Face-Person-Detector/resources/runtime/assets/workspace/results/m07_native840_v1')


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
    import yaml
    import ultralytics
    from ultralytics.data.converter import convert_coco, coco80_to_coco91_class

    source_hashes = {n: sha(SOURCE / n) for n in ['val2017.zip', 'annotations_trainval2017.zip']}
    root = OUT / 'coco'
    marker = root / 'prepared.json'
    if marker.exists():
        saved = json.loads(marker.read_text())
        if saved['source_archives'] != source_hashes:
            raise RuntimeError('Source archives changed; preserve this dataset and use a new output version.')
        return root, saved
    with zipfile.ZipFile(SOURCE / 'annotations_trainval2017.zip') as z:
        raw = z.read('annotations/instances_val2017.json')
    data = json.loads(raw)
    ids = sorted(x['id'] for x in data['images'])
    assert len(ids) == len(set(ids)) == 5000 and len(data['categories']) == 80
    assert sorted(x['id'] for x in data['categories']) == coco80_to_coco91_class()
    with tempfile.TemporaryDirectory(prefix='m07-coco-', dir=OUT) as tmp:
        staging = Path(tmp) / 'coco'
        annotations = Path(tmp) / 'annotations'
        annotations.mkdir()
        (annotations / 'instances_val2017.json').write_bytes(raw)
        # Official converter owns the COCO-to-YOLO mapping; never change original JSON.
        convert_coco(labels_dir=str(annotations), save_dir=str(staging), cls91to80=True)
        (staging / 'annotations').mkdir()
        (staging / 'annotations' / 'instances_val2017.json').write_bytes(raw)
        image_dir = staging / 'images' / 'val2017'
        image_dir.mkdir()
        with zipfile.ZipFile(SOURCE / 'val2017.zip') as z:
            for image in sorted(data['images'], key=lambda x: x['id']):
                name = image['file_name']
                assert Path(name).name == name and int(Path(name).stem) == image['id']
                (image_dir / name).write_bytes(z.read('val2017/' + name))
        (staging / 'val2017.txt').write_text(''.join(
            './images/val2017/' + x['file_name'] + '\n'
            for x in sorted(data['images'], key=lambda x: x['id'])))
        # Retain the official names and val2017.txt convention for native COCO detection.
        template = Path(ultralytics.__file__).parent / 'cfg/datasets/coco.yaml'
        cfg = yaml.safe_load(template.read_text())
        cfg.pop('download', None)
        cfg['path'] = str(root)
        (staging / 'coco.yaml').write_text(yaml.safe_dump(cfg, sort_keys=False))
        saved = {'source_archives': source_hashes, 'images': len(ids),
                 'annotations': len(data['annotations']), 'annotation_sha256': hashlib.sha256(raw).hexdigest(),
                 'categories': len(data['categories']), 'image_ids': ids,
                 'official_converter': 'ultralytics.data.converter.convert_coco',
                 'official_coco_yaml_sha256': sha(template)}
        write(staging / 'prepared.json', saved)
        staging.rename(root)
    return root, saved


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
        weight = Path('/data2/gaoyonghan/gpu_model_standalone_20261110/release/model/检测/iitolstykh--YOLO-Face-Person-Detector/resources/model_index/Ultralytics/YOLOv8/yolov8n.pt')
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
