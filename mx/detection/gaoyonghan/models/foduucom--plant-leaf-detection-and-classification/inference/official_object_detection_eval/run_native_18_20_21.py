"""Native validation wrappers only; no custom inference, NMS or AP algorithm."""
import argparse
from datetime import datetime
import fcntl
from importlib import import_module
from importlib.metadata import version
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import zipfile

from run_m07_native import OUT as COCO_OUT, prepare as prepare_coco, sha, write

HERE = Path(__file__).resolve().parent
MANIFEST = HERE / 'native_18_20_21.json'
OUT = Path('/data2/gaoyonghan/gpu_model_standalone_20261110/release/model/检测/foduucom--plant-leaf-detection-and-classification/resources/runtime/assets/workspace/results/native_18_20_21_v1')
PLATE_ZIP = Path('/home/ubuntu/文档/ChatGPT/idea 2/outputs/launch_audit_20261005/migration_inputs/datasets/license_plate_v1_valid/data/valid.zip')


def prepare_plate():
    from ultralytics.data.converter import convert_coco
    root = OUT / 'plate'
    marker = root / 'prepared.json'
    archive_hash = sha(PLATE_ZIP)
    if marker.exists():
        assert json.loads(marker.read_text())['archive_sha256'] == archive_hash
        return
    with tempfile.TemporaryDirectory(prefix='plate-', dir=OUT) as tmp:
        staging = Path(tmp) / 'prepared'
        source = Path(tmp) / 'annotations'
        source.mkdir()
        with zipfile.ZipFile(PLATE_ZIP) as z:
            raw = z.read('_annotations.coco.json')
            annotation = json.loads(raw)
            assert len(annotation['images']) == 1765 and len(annotation['annotations']) == 1840
            assert [x['id'] for x in annotation['categories']] == [0]
            converted = json.loads(raw)
            converted['categories'][0]['id'] = 1
            for x in converted['images']:
                x['file_name'] = str(x['id']) + Path(x['file_name']).suffix
            for x in converted['annotations']:
                assert x['category_id'] == 0 and not x.get('iscrowd', 0)
                x['category_id'] = 1
            write(source / 'instances_val.json', converted)
            convert_coco(str(source), str(staging), cls91to80=False)
            image_dir = staging / 'images/val'
            image_dir.mkdir(parents=True, exist_ok=True)
            for x in annotation['images']:
                (image_dir / (str(x['id']) + Path(x['file_name']).suffix)).write_bytes(z.read(x['file_name']))
        (staging / 'annotation.json').write_bytes(raw)
        count = 0
        for p in (staging / 'labels/val').glob('*.txt'):
            for line in p.read_text().splitlines():
                values = list(map(float, line.split()))
                assert len(values) == 5 and values[0] == 0
                assert all(0 <= v <= 1 for v in values[1:])
                count += 1
        assert count == 1840, 'Converter dropped or deduplicated annotations: audit before running'
        write(staging / 'prepared.json', {'archive_sha256': archive_hash,
              'images': 1765, 'boxes': count, 'converter_version': version('ultralytics'),
              'converter_sha256': sha(Path(import_module('ultralytics.data.converter').__file__)),
              'label_mapping': 'original COCO 0 -> conversion COCO 1 -> YOLO 0',
              'filenames': 'numeric original image_id, image bytes unchanged'})
        staging.rename(root)


def preflight(number):
    spec = json.loads(MANIFEST.read_text())[str(number)]
    assert Path(spec['weight']).is_file(), spec['weight']
    assert version(spec['package']) == spec['version'], 'Unexpected framework version'
    import_module('pycocotools.cocoeval')
    if number == 18:
        import_module('yolov5.val')
        config_dir = import_module('yolov5.utils.general').CONFIG_DIR
    else:
        import_module('ultralytics.models.yolo.detect.val')
        config_dir = import_module('ultralytics.utils').USER_CONFIG_DIR
    # Validators check this font even with plots=False; reuse it without a network download.
    font = config_dir / 'Arial.ttf'
    if not font.exists():
        existing_font = Path('/data2/gaoyonghan/gpu_model_standalone_20261110/release/model/检测/foduucom--plant-leaf-detection-and-classification/resources/runtime/assets/fonts/Arial.ttf')
        assert existing_font.is_file(), 'Missing cached Arial.ttf required by native validator'
        font.symlink_to(existing_font)
    if number == 21:
        import_module('faster_coco_eval')
    if number == 20:
        source = Path(spec['source'])
        actual = Path(import_module('ultralytics').__file__).resolve()
        assert source in actual.parents, f'Wrong YOLOv10 import: {actual}'
        revision = subprocess.check_output(['git', '-C', str(source), 'rev-parse', 'HEAD'], text=True).strip()
        assert revision == spec['source_revision']
        changed = subprocess.check_output(['git', '-C', str(source), 'diff', 'HEAD', '--', 'ultralytics'], text=True)
        assert not changed, 'Tracked YOLOv10 source modified; audit before running'
    return spec


def dataset_for_run(number, run, max_images):
    import yaml
    from PIL import Image
    root = OUT / 'plate' if number == 18 else COCO_OUT / 'coco'
    annotation = root / ('annotation.json' if number == 18 else 'annotations/instances_val2017.json')
    a = json.loads(annotation.read_text())
    spec = json.loads(MANIFEST.read_text())[str(number)]
    assert len(a['images']) == spec['expected_images'] and len(a['annotations']) == spec['expected_boxes']
    assert len({x['id'] for x in a['images']}) == len(a['images'])
    chosen = sorted(a['images'], key=lambda x: x['id'])
    if max_images:
        chosen = chosen[:max_images]
    image_dir = root / ('images/val' if number == 18 else 'images/val2017')
    images = [image_dir / (str(x['id']) + Path(x['file_name']).suffix) if number == 18
              else image_dir / x['file_name'] for x in chosen]
    # Native COCO detection depends on a path containing coco and ending /val2017.txt.
    config_dir = run / ('plate' if number == 18 else 'coco')
    config_dir.mkdir()
    listing = config_dir / ('val.txt' if number == 18 else 'val2017.txt')
    listing.write_text(''.join(str(p) + '\n' for p in images))
    if number == 18:
        cfg = {'nc': 1, 'names': ['license_plate']}
    else:
        cfg = yaml.safe_load((root / 'coco.yaml').read_text())
        cfg.pop('download', None)
    cfg.update(path=str(root), train=str(listing), val=str(listing))
    config = config_dir / 'data.yaml'
    config.write_text(yaml.safe_dump(cfg, sort_keys=False))
    files = {}
    for p, info in zip(images, chosen):
        with Image.open(p) as im:
            assert im.size == (info['width'], info['height'])
            im.verify()
        label = Path(str(p).replace('/images/', '/labels/')).with_suffix('.txt')
        files[str(p)] = sha(p)
        files[str(label)] = sha(label) if label.exists() else None
    return annotation, config, {'image_ids': [x['id'] for x in chosen],
           'images': len(chosen), 'annotation_sha256': sha(annotation),
           'yaml_sha256': sha(config), 'input_files_sha256': files,
           'preparation': json.loads((root / 'prepared.json').read_text())}


def native_plate(spec, config, run, args, expected_ids):
    from yolov5 import val
    from yolov5.utils.callbacks import Callbacks
    callbacks = Callbacks()
    seen, predictions = [], []

    def image_end(pred, predn, path, names, image):
        # Use the official exporter on already-restored boxes, not a custom postprocessor.
        val.save_one_json(predn, predictions, path, [0])

    def batch_end(batch_i, im, targets, paths, shapes, preds):
        seen.extend(int(Path(p).stem) for p in paths)

    callbacks.register_action('on_val_image_end', callback=image_end)
    callbacks.register_action('on_val_batch_end', callback=batch_end)
    options = dict(data=str(config), weights=spec['weight'], batch_size=args.batch, imgsz=640,
                   conf_thres=.001, iou_thres=.6, max_det=300, task='val', device=args.device,
                   workers=0, single_cls=False, augment=False, half=False, plots=False,
                   save_json=False, project=str(run.parent), name=run.name, exist_ok=True)
    write(run / 'native_options.json', options)
    # save_json=False avoids yolov5-pip's hard-coded COCO path overwriting custom-dataset AP.
    # yolov5-pip 7.0.14 returns metrics, per-class AP, per-class AP50, timings.
    result, _, _, _ = val.run(**options, callbacks=callbacks)
    assert sorted(seen) == sorted(expected_ids), 'Native loader skipped/duplicated images'
    write(run / 'predictions.json', predictions)
    return dict(zip(['precision', 'recall', 'AP50', 'AP'], map(float, result[:4])))


def native_coco(number, spec, config, run, args, expected_ids):
    import ultralytics
    constructor = ultralytics.YOLOv10 if number == 20 else ultralytics.YOLO
    model = constructor(spec['weight'])
    captured = {}

    def capture(validator):
        assert validator.is_coco and validator.seen == len(expected_ids)
        ids = sorted(int(Path(p).stem) for p in validator.dataloader.dataset.im_files)
        assert ids == sorted(expected_ids), 'Native loader skipped/duplicated images'
        captured.update(validator.metrics.results_dict)
        write(run / 'native_options.json', vars(validator.args))

    model.add_callback('on_val_end', capture)
    options = dict(data=str(config), split='val', imgsz=640, conf=.001, max_det=300,
                   batch=args.batch, device=args.device, half=False, rect=True, augment=False,
                   workers=0, save_json=True, plots=False, project=str(run.parent),
                   name=run.name, exist_ok=True)
    if number == 21:
        options.update(iou=.7, agnostic_nms=False)
    write(run / 'native_options.json', options)
    model.val(**options)
    assert captured, 'Native validation did not complete'
    return dict(zip(['precision', 'recall', 'AP50', 'AP'], [float(captured[k]) for k in [
        'metrics/precision(B)', 'metrics/recall(B)', 'metrics/mAP50(B)', 'metrics/mAP50-95(B)']]))


def evaluate(number, args):
    spec = preflight(number)
    parent = OUT / ('m' + str(number))
    parent.mkdir(parents=True, exist_ok=True)
    run = parent / datetime.now().strftime('run_%Y%m%d_%H%M%S_%f')
    run.mkdir()
    state = {'state': 'preparing', 'sequence': number, 'max_images': args.max_images}
    write(run / 'status.json', state)
    print('Results: ' + str(run), flush=True)
    try:
        annotation, config, dataset = dataset_for_run(number, run, args.max_images)
        modules = (['yolov5.val', 'yolov5.utils.dataloaders', 'yolov5.utils.metrics'] if number == 18 else
                   ['ultralytics.engine.validator', 'ultralytics.models.yolo.detect.val',
                    'ultralytics.data.augment', 'ultralytics.utils.metrics'])
        if number == 20:
            modules += ['ultralytics.models.yolov10.val', 'ultralytics.utils.ops']
        protocol = dict(spec, dataset_snapshot=dataset, weight_sha256=sha(Path(spec['weight'])),
                        batch=args.batch, device=args.device, dtype='float32',
                        fidelity='reference_baseline', dataset_relation='original_exact',
                        software={p: version(p) for p in [spec['package'], 'torch', 'torchvision',
                                                         'numpy', 'Pillow', 'pycocotools']},
                        official_code_sha256={m: sha(Path(import_module(m).__file__)) for m in modules})
        for path in [Path(__file__), MANIFEST, HERE / 'run_m07_native.py', HERE / 'run_native_18_20_21.sh']:
            shutil.copy2(path, run / path.name)
        import hashlib
        protocol['script_sha256'] = sha(Path(__file__))
        fingerprint = hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest()
        state.update(state='running', protocol=protocol, protocol_fingerprint=fingerprint)
        write(run / 'status.json', state)
        ids = dataset['image_ids']
        native = (native_plate(spec, config, run, args, ids) if number == 18 else
                  native_coco(number, spec, config, run, args, ids))
        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval
        gt = COCO(str(annotation))
        predictions = json.loads((run / 'predictions.json').read_text())
        assert set(p['image_id'] for p in predictions) <= set(ids)
        if predictions:
            dt = gt.loadRes(predictions)
        else:
            dt = COCO()
            dt.dataset = dict(images=gt.dataset['images'], categories=gt.dataset['categories'], annotations=[])
            dt.createIndex()
        ev = COCOeval(gt, dt, 'bbox')
        ev.params.imgIds = ids
        ev.evaluate()
        ev.accumulate()
        ev.summarize()
        coco = dict(zip(['AP', 'AP50', 'AP75', 'AP_small', 'AP_medium', 'AP_large',
                         'AR1', 'AR10', 'AR100', 'AR_small', 'AR_medium', 'AR_large'], map(float, ev.stats)))
        full = len(ids) == spec['expected_images']
        primary = native if number == 18 else coco
        write(run / 'metrics.json', {'unit': 'fraction_0_to_1', 'full_dataset': full,
              'native_metrics': native, 'coco_metrics': coco, 'evaluation_images': len(ids),
              'reference_percent': spec['reference_percent'], 'primary_metric': spec['reference_metric'],
              'difference_percentage_points': primary[spec['reference_metric']] * 100 - spec['reference_percent'] if full else None,
              'reference_check': 'numeric_comparison_only_protocol_limitations_apply' if full else 'smoke_not_reportable',
              'predictions_sha256': sha(run / 'predictions.json')})
        state.update(state='complete' if full else 'smoke_complete', completed_images=len(ids))
        print(f'M{number}: native={native}; COCO AP={coco["AP"]:.6f}; full={full}', flush=True)
    except BaseException as exc:
        state.update(state='failed', error=repr(exc))
        raise
    finally:
        write(run / 'status.json', state)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-number', type=int, choices=[18, 20, 21])
    parser.add_argument('--prepare', action='store_true')
    parser.add_argument('--check', action='store_true')
    parser.add_argument('--describe', action='store_true')
    parser.add_argument('--device', default='0')
    parser.add_argument('--batch', type=int, default=16)
    parser.add_argument('--max-images', type=int, default=0)
    args = parser.parse_args()
    if args.batch < 1 or args.max_images < 0:
        parser.error('batch must be positive; max-images must be nonnegative')
    if args.describe:
        print(MANIFEST.read_text()); return
    if args.check:
        assert args.model_number is not None
        print(json.dumps(preflight(args.model_number), indent=2)); return
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / '.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if args.prepare:
            assert version('ultralytics') == '8.4.0'
            COCO_OUT.mkdir(parents=True, exist_ok=True)
            with (COCO_OUT / '.lock').open('w') as coco_lock:
                fcntl.flock(coco_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                prepare_coco()
            prepare_plate()
            print('Native datasets ready', flush=True)
        else:
            if args.model_number is None:
                parser.error('--model-number is required for evaluation')
            evaluate(args.model_number, args)


if __name__ == '__main__':
    main()
