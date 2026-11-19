"""M34: native 8.0.43 validation on the unchanged PlantDoc common-class subset."""
import argparse
from collections import Counter, defaultdict
from datetime import datetime
import hashlib
import importlib.util
from importlib.metadata import version
import io
import json
from pathlib import Path
import shutil
import sys
import zipfile

HERE = Path(__file__).resolve().parent
OLD = Path('/project/detection/gaoyonghan/models/foduucom--plant-leaf-detection-and-classification/resources/runtime/assets/workspace/results/remaining25_v4/m34-plant-leaf-detection-and-classification/plantdoc_species')
DATA = Path('/project/dataset')
MODEL = Path('/project/detection/gaoyonghan/models/foduucom--plant-leaf-detection-and-classification/weights/foduucom/plant-leaf-detection-and-classification')
OUT = Path('/project/detection/gaoyonghan/models/foduucom--plant-leaf-detection-and-classification/result/delivery_20261118_image/m034')


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False))


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    sys.modules[name] = value
    spec.loader.exec_module(value)
    return value


def prepare(output):
    """Use the frozen PlantDoc preparation already transferred from MX."""
    model = Path(__file__).resolve().parents[2]
    path_config = json.loads((model / 'paths.json').read_text())
    data_root = Path(__import__('os').environ.get('MODEL_DATASET_ROOT', path_config['dataset_root']))
    if not data_root.is_absolute():
        data_root = model / data_root
    root = data_root.resolve() / 'prepared/m34_dataset'
    provenance = json.loads((root / 'provenance.json').read_text())
    assert provenance['sequence'] == 34 and len(provenance['source_files']) == 237
    assert provenance['boxes'] == 446 and provenance['original_annotation_sha256'] == '56a605ada7051fcea012b2202cd9bfa3334d3a69e2c2d1390deaa02c3c118b62'
    assert sha(HERE / 'source_snapshots/plantdoc/adapters.py') == provenance['reader_sha256']
    # Historical preparation core differs from the packaged MX copy. The frozen
    # prepared images, labels, reader, original annotation and counts are checked below.
    packaged_core_sha256 = sha(HERE.parent / 'official_object_detection_eval/run_model.py')
    names = {int(k): v for k, v in provenance['coverage']['model_labels'].items()}
    counts = Counter()
    manifest = []
    for record in provenance['source_files']:
        image_id = int(record['image_id'])
        image = root / 'images/test' / (str(image_id) + '.png')
        label = root / 'labels/test' / (str(image_id) + '.txt')
        assert sha(image) == record['image_sha256'], 'Prepared image changed: ' + str(image)
        assert sha(label) == record['label_sha256'], 'Prepared label changed: ' + str(label)
        rows = [line.split() for line in label.read_text().splitlines() if line.strip()]
        assert len(rows) == record['boxes']
        for row in rows:
            assert len(row) == 5
            cls = int(row[0])
            assert cls in names
            assert all(0 <= float(x) <= 1 for x in row[1:])
            counts[cls] += 1
        manifest.append({**record, 'image': str(image), 'label': str(label)})
    assert sorted(x['image_id'] for x in manifest) == list(range(1, 238))
    assert sum(counts.values()) == provenance['boxes']
    assert {names[k].lower(): v for k, v in counts.items()} == provenance['coverage']['gt_instances_per_class']
    protocol = dict(provenance, source_archive=str(data_root.resolve() / 'plantdoc_object_detection_test/PlantDoc-Object-Detection-Dataset-master.zip'), source_files=manifest, prepared_root=str(root), prepared_provenance_sha256=sha(root / 'provenance.json'), packaged_core_sha256=packaged_core_sha256, historical_core_matches_packaged_mx=(packaged_core_sha256 == provenance['core_sha256']))
    return manifest, names, counts, protocol



def evaluate(args, manifest, names, counts, protocol):
    import torch
    import ultralytics
    from ultralytics.yolo.cfg import get_cfg
    from ultralytics.yolo.v8.detect.val import DetectionValidator
    from ultralytics.yolo.utils import metrics as native_metrics
    assert ultralytics.__version__ == '8.0.43' and version('ultralyticsplus') == '0.0.28'
    weight = MODEL / 'best.pt'
    assert sha(weight) == 'eaedbb5c34b0c772c608dca209c511d15b1fa42eb05914e2346214f984a1d290'
    checkpoint = torch.load(weight, map_location='cpu', weights_only=False)
    assert checkpoint['version'] == '8.0.43' and checkpoint['model'].names == names
    train_args = {k: checkpoint['train_args'].get(k) for k in ['imgsz', 'conf', 'iou', 'max_det', 'batch', 'split']}
    del checkpoint
    selected = manifest[:args.max_images] if args.max_images else manifest
    run = args.output_root / ('smoke' if args.max_images else 'full') / datetime.now().strftime('run_%Y%m%d_%H%M%S_%f')
    run.mkdir(parents=True)
    for name in ['run_m34_native.py', 'run_m34_native.sh']:
        shutil.copy2(HERE / name, run / name)
    (run / 'test.txt').write_text('\n'.join(item['image'] for item in selected) + '\n')
    # JSON is valid YAML; aliases below are required by the native dataset reader.
    write(run / 'data.yaml', dict(path=str(run), train='test.txt', val='test.txt', test='test.txt', names=[names[i] for i in range(46)]))
    options = dict(data=str(run / 'data.yaml'), split='test', task='detect', mode='val',
                   imgsz=640, conf=.001, iou=.7, max_det=300, batch=1, workers=0,
                   device=args.device, half=False, rect=True, augment=False,
                   agnostic_nms=False, single_cls=False, save_json=True,
                   save_txt=False, plots=False, project=str(run.parent), name=run.name, exist_ok=True)
    protocol.update(weight_sha256=sha(weight), checkpoint_version='8.0.43', checkpoint_train_args=train_args,
                    model_card_reference={'AP50': .946, 'url': 'https://huggingface.co/foduucom/plant-leaf-detection-and-classification'},
                    local_config_reference=json.loads((MODEL / 'config.json').read_text()),
                    options=options, software={n: version(n) for n in ['ultralytics', 'ultralyticsplus', 'torch', 'torchvision', 'numpy']},
                    native_metrics_sha256=sha(Path(native_metrics.__file__)),
                    native_validator_sha256=sha(Path(sys.modules[DetectionValidator.__module__].__file__)),
                    script_sha256=sha(Path(__file__)),
                    limitations=['Only 12/46 species; not an official-dataset reproduction.',
                                 'Native val defaults agree with checkpoint args; card single-image predict thresholds are not an evaluation protocol.',
                                 'Batch1/FP32 standalone validation; not historical training-time validation.',
                                 'Version, native preprocessing/NMS and AP implementation change together; not a single-factor test.',
                                 'Card AP50=.946 conflicts with local config score_map50=.58305; neither is a comparable PlantDoc target.'])
    write(run / 'protocol.json', protocol)
    state = {'state': 'running', 'protocol_fingerprint': digest(protocol), 'full_dataset': not bool(args.max_images)}
    write(run / 'status.json', state)
    seen = []

    class CoverageValidator(DetectionValidator):
        def update_metrics(self, preds, batch):
            seen.extend(int(Path(p).stem) for p in batch['im_file'])
            return super().update_metrics(preds, batch)

    try:
        print(f'M34 native: {len(selected)} images, output={run}', flush=True)
        validator = CoverageValidator(args=get_cfg(overrides=options), save_dir=run)
        validator(model=str(weight))
        assert sorted(seen) == sorted(item['image_id'] for item in selected)
        assert len(set(seen)) == validator.seen == len(selected)
        assert validator.names == names
        if not args.max_images:
            assert validator.nt_per_class.tolist() == [counts[i] for i in range(46)]
            per_class = {names[int(c)]: dict(zip(['precision', 'recall', 'AP50', 'AP'], map(float, validator.metrics.class_result(i))))
                         for i, c in enumerate(validator.metrics.ap_class_index)}
            assert len(per_class) == 12, 'Native AP must average over the 12 GT classes'
            write(run / 'metrics.json', {'native_metrics': {k: float(v) for k, v in validator.metrics.results_dict.items()},
                                        'per_class': per_class, 'unit': 'fraction_0_to_1', 'evaluation_images': len(seen),
                                        'covered_classes': 12, 'total_model_classes': 46, 'official_comparable': False,
                                        'actual_half': bool(validator.args.half), 'actual_options': vars(validator.args)})
        write(run / 'coverage.json', {'image_ids': seen, 'images': len(seen)})
        state.update(state='smoke_complete' if args.max_images else 'complete', completed_images=len(seen))
        print(f'M34 completed: {run}', flush=True)
    except BaseException as exc:
        state.update(state='failed', error=repr(exc))
        raise
    finally:
        write(run / 'status.json', state)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', default='0')
    parser.add_argument('--max-images', type=int, choices=[0, 10], default=0)
    parser.add_argument('--output-root', type=Path, default=OUT)
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args()
    manifest, names, counts, protocol = prepare(args.output_root)
    print(f'Prepared: {len(manifest)} images, {sum(counts.values())} boxes, {len(counts)}/46 classes', flush=True)
    if not args.prepare_only:
        evaluate(args, manifest, names, counts, protocol)


if __name__ == '__main__':
    main()
