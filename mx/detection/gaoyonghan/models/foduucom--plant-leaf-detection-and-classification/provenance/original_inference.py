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
OLD = Path('/home/ubuntu/gaoyonghan/gpu/results/remaining25_v4/m34-plant-leaf-detection-and-classification/plantdoc_species')
DATA = Path('/data2/gaoyonghan/gpu/dataset')
MODEL = Path('/data2/gaoyonghan/gpu/model/foduucom/plant-leaf-detection-and-classification')
OUT = Path('/home/ubuntu/gaoyonghan/gpu/results/m34_native8043_v1')


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
    """Reuse frozen XML interpretation; change label encoding, not boxes/split."""
    old = json.loads((OLD / 'status.json').read_text())
    core_path = HERE.parent / 'official_object_detection_eval/run_model.py'
    sys.path.insert(0, str(core_path.parent))
    core = module('m34_core', core_path)
    reader_path = HERE / 'source_snapshots/plantdoc/adapters.py'
    assert sha(reader_path) == old['protocol']['adapter_sha256'], 'Frozen reader changed'
    reader = module('m34_reader', reader_path)
    spec = old['protocol']['dataset']
    prepared = reader.xml_detection(spec, DATA, core)
    annotation = core.collapse_categories(prepared.annotation, spec['category_collapse'])
    assert digest(annotation) == old['annotation_sha256'], 'Dataset differs from old experiment'
    names = {int(k): v for k, v in old['coverage']['model_labels'].items()}
    inverse = {int(v): int(k) for k, v in old['coverage']['mapping'].items()}
    boxes = defaultdict(list)
    for ann in annotation['annotations']:
        if ann['category_id'] in inverse:
            assert not ann['iscrowd'], 'Native YOLO labels cannot represent crowd/difficult flags'
            boxes[ann['image_id']].append((inverse[ann['category_id']], ann['bbox']))
    prepared.close()
    root = output / 'dataset'
    (root / 'images/test').mkdir(parents=True, exist_ok=True)
    (root / 'labels/test').mkdir(parents=True, exist_ok=True)
    counts, manifest = Counter(), []
    with zipfile.ZipFile(DATA / spec['archive']) as archive:
        members = {Path(n).name: n for n in archive.namelist()
                   if n.startswith(spec['prefix']) and Path(n).suffix.lower() in core.IMAGE_SUFFIXES}
        for info in annotation['images']:
            member = members[info['file_name']]
            image = root / 'images/test' / (str(info['id']) + '.png')
            source = archive.read(member)
            from PIL import Image
            # Match the old reader's decoded RGB pixels exactly. Remove EXIF
            # orientation metadata so native OpenCV cannot rotate only the image.
            with Image.open(io.BytesIO(source)) as opened:
                assert opened.size == (info['width'], info['height'])
                orientation = opened.getexif().get(274, 1)
                rgb = opened.convert('RGB')
            pixels = rgb.tobytes()
            if not image.exists():
                Image.frombytes('RGB', rgb.size, pixels).save(image, format='PNG', compress_level=1)
            with Image.open(image) as opened:
                assert opened.size == (info['width'], info['height'])
                assert opened.convert('RGB').tobytes() == pixels, 'Decoded image changed'
            w, h = info['width'], info['height']
            rows = []
            for cls, (x, y, bw, bh) in boxes[info['id']]:
                values = [(x + bw / 2) / w, (y + bh / 2) / h, bw / w, bh / h]
                assert all(0 <= v <= 1 for v in values) and min(bw, bh) > 0
                rows.append(str(cls) + ' ' + ' '.join(f'{v:.12g}' for v in values))
                cx, cy, nw, nh = map(float, rows[-1].split()[1:])
                restored = [(cx - nw / 2) * w, (cy - nh / 2) * h, nw * w, nh * h]
                assert max(abs(a - b) for a, b in zip(restored, [x, y, bw, bh])) < 1e-5
                counts[cls] += 1
            assert len(rows) == len(set(rows)), 'Native loader would remove duplicate labels'
            label = root / 'labels/test' / (str(info['id']) + '.txt')
            content = '\n'.join(rows) + ('\n' if rows else '')
            if label.exists():
                assert label.read_text() == content, f'Cached labels changed: {label}'
            else:
                label.write_text(content)
            manifest.append({'image_id': info['id'], 'archive_member': member,
                             'image': str(image), 'image_sha256': sha(image),
                             'source_sha256': hashlib.sha256(source).hexdigest(),
                             'source_exif_orientation': orientation,
                             'decoded_rgb_sha256': hashlib.sha256(pixels).hexdigest(),
                             'label_sha256': sha(label), 'boxes': len(rows)})
            if len(manifest) % 50 == 0:
                print(f'Preparing M34 dataset: {len(manifest)}/{len(annotation["images"])}', flush=True)
    expected = old['coverage']['gt_instances_per_class']
    assert len(manifest) == 237 and len(counts) == 12
    assert {names[k].lower(): v for k, v in counts.items()} == expected
    protocol = {'sequence': 34, 'model_id': old['model_id'], 'dataset': 'PlantDoc TEST',
                'dataset_relation': 'public_replacement', 'official_comparable': False,
                'images': len(manifest), 'boxes': sum(counts.values()),
                'coverage': old['coverage'], 'original_annotation_sha256': old['annotation_sha256'],
                'source_archive': str(DATA / spec['archive']), 'source_files': manifest,
                'reader_sha256': sha(reader_path), 'core_sha256': sha(core_path),
                'old_result_directory': str(OLD), 'old_metrics': json.loads((OLD / 'metrics.json').read_text())['metrics'],
                'label_conversion': 'Same species mapping and VOC coordinates as v4; encode in YOLO using original 46 checkpoint class IDs. No box merging.',
                'image_conversion': 'Lossless PNG of old PIL RGB decoding, no rotation/resizing; strip metadata to prevent OpenCV EXIF auto-rotation. Source archive remains unchanged.',
                'missing_labels': old['coverage']['missing_labels']}
    write(root / 'provenance.json', protocol)
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
