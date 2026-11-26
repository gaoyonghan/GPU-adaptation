"""M46: unchanged checkpoint + ISFire-4 two-class test + Ultralytics native val."""
import argparse
from collections import Counter
from datetime import datetime
import hashlib
from importlib.metadata import version
import inspect
import io
import json
import math
from pathlib import Path
import shutil
import tempfile
import zipfile

from PIL import Image
import yaml

HERE = Path(__file__).resolve().parent
WEIGHT = Path('__MODEL_ROOT__/weights/TommyNgx/YOLOv10-Fire-and-Smoke-Detection/best.pt')
ARCHIVE = Path('__DATASET_ROOT__/dfire_test_replacement_isfire4/Dataset.zip')
OUT = Path('__MODEL_ROOT__/resources/runtime/assets/workspace/results/m46_native_v1')
NAMES = {0: 'Fire', 1: 'Smoke'}
MAPPING = {0: 1, 1: 0, 2: 1, 3: 0}


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''): h.update(block)
    return h.hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def prepare(root):
    assert sha(ARCHIVE) == 'fb90a9bfed10f3f00529e2a8d8c512d5f92d8d2e19baac8135e6bdb751722ce9', 'Dataset archive changed'
    assert sha(WEIGHT) == '5f4bf49497480dcb78baeeefbf12119e4b45344d5644aa2bbed1a20017e14ca8', 'Model weights changed'
    data = root/'dataset'
    for folder in ['images/test', 'labels/test']: (data/folder).mkdir(parents=True, exist_ok=True)
    manifest, counts = [], Counter()
    with zipfile.ZipFile(ARCHIVE) as z:
        cfg = yaml.safe_load(z.read('Dataset/data.yaml'))
        assert cfg['names'] == ['black smoke','blue fire','white-gray smoke','yellow-orange fire']
        members = sorted(n for n in z.namelist() if n.startswith('Dataset/test/images/') and Path(n).suffix.lower() in ['.jpg','.jpeg','.png','.bmp','.webp'])
        assert len(members) == 703 and len({Path(n).stem for n in members}) == 703
        for member in members:
            image = data/'images/test'/Path(member).name
            payload = z.read(member)
            with Image.open(io.BytesIO(payload)) as im:
                assert im.getexif().get(274, 1) == 1, 'Unexpected EXIF rotation; review before using native loader'
                im.verify()
            if image.exists(): assert image.read_bytes() == payload
            else: image.write_bytes(payload)
            label_member = 'Dataset/test/labels/' + Path(member).stem + '.txt'
            source_label = z.read(label_member)
            lines = []
            for line in source_label.decode().splitlines():
                if not line.strip(): continue
                cid, *coordinates = line.split()
                values = list(map(float, coordinates))
                assert len(values) == 4 and all(math.isfinite(x) and 0 <= x <= 1 for x in values) and min(values[2:]) > 0
                target = MAPPING[int(cid)]
                # Preserve coordinate strings exactly. Only replace the class ID.
                lines.append(str(target) + ' ' + ' '.join(coordinates))
                counts[target] += 1
            assert len(lines) == len(set(tuple(map(float, x.split())) for x in lines)), 'Native loader would drop duplicate labels'
            label = data/'labels/test'/(Path(member).stem + '.txt')
            content = '\n'.join(lines) + ('\n' if lines else '')
            if label.exists(): assert label.read_text() == content
            else: label.write_text(content)
            manifest.append(dict(image=str(image), image_sha256=sha(image), archive_member=member,
                                 source_label_sha256=hashlib.sha256(source_label).hexdigest(), label_sha256=sha(label)))
    assert counts == {0:543, 1:535}
    print('Prepared: 703 test images, Fire=543 boxes, Smoke=535 boxes; original image bytes/box coordinates preserved.', flush=True)
    return manifest, counts


def options(run, device):
    return dict(data=str(run/'data.yaml'), split='test', imgsz=640, batch=1, workers=0,
                conf=.001, iou=.7, max_det=300, device=device, half=False, rect=True,
                augment=False, single_cls=False, agnostic_nms=False, classes=None,
                save_json=True, save_txt=False, plots=False, project=str(run.parent), name=run.name, exist_ok=True)


def load_model():
    from ultralytics import YOLO
    assert version('ultralytics') == '8.4.0', 'Use the checked existing object-detection-yolo-v2 environment'
    model = YOLO(WEIGHT)
    assert model.names == NAMES and model.task == 'detect'
    assert model.ckpt['version'] == '8.2.59' and model.model.end2end
    assert type(model.model.model[-1]).__name__ == 'v10Detect'
    return model


def dataset_config(run, selected):
    (run/'test.txt').write_text('\n'.join(x['image'] for x in selected) + '\n')
    write(run/'data.yaml', dict(path=str(run), train='test.txt', val='test.txt', test='test.txt', names=NAMES))


def run(args, root):
    from ultralytics.models.yolo.detect.val import DetectionValidator
    from ultralytics.utils import metrics as native_metrics
    from ultralytics.data.utils import check_det_dataset
    manifest, counts = prepare(root)
    model = load_model()
    selected = manifest[:args.max_images] if args.max_images else manifest
    run_dir = root/('preflight' if args.preflight else 'smoke' if args.max_images else 'full')/datetime.now().strftime('run_%Y%m%d_%H%M%S_%f')
    run_dir.mkdir(parents=True)
    dataset_config(run_dir, selected)
    opts = options(run_dir, 'cpu' if args.preflight else args.device)
    if args.preflight:
        validator = DetectionValidator(args=opts, save_dir=run_dir)
        validator.data = check_det_dataset(opts['data'], autodownload=False)
        validator.stride = int(model.model.stride.max())
        loader = validator.get_dataloader(validator.data['test'], 1)
        seen, gt = [], Counter()
        for batch in loader:
            seen.extend(batch['im_file'])
            gt.update(map(int, batch['cls'].flatten().tolist()))
        assert sorted(seen) == sorted(x['image'] for x in selected) and len(set(seen)) == len(selected)
        assert gt == counts
        assert next(model.model.parameters()).device.type == 'cpu'
        print('READY: CPU weight load + all 703 images through native validation loader passed. No forward/GPU inference executed.')
        return
    import torch
    assert args.device.isdigit() and torch.cuda.is_available(), 'Specify one available GPU with DEVICE=0 (no CPU fallback)'
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    source_files = [Path(__file__), HERE/'run_m46_native.sh', Path(inspect.getfile(DetectionValidator)), Path(native_metrics.__file__)]
    protocol = dict(sequence=46, model_id='TommyNgx/YOLOv10-Fire-and-Smoke-Detection',
        weight_sha256=sha(WEIGHT), archive_sha256=sha(ARCHIVE), dataset='ISFire-4 test, four classes collapsed to Fire/Smoke',
        prior_coco_annotation_sha256='a1a31bfd1ab459daba26919b20d76eb288802fd9c1f4d699af1ec331052c18e6',
        prior_annotation_check='Frozen ZIP resolves to the same old COCO annotation SHA in delivery audit; COCO is not used for this run scoring.',
        dataset_relation='public_replacement', official_comparable=False, label_mapping=MAPPING,
        manifest=selected, options=opts, checkpoint_version=model.ckpt['version'],
        author_demo_version='8.2.56', checkpoint_train_args=model.ckpt['train_args'],
        software={n:version(n) for n in ['ultralytics','torch','torchvision','numpy','Pillow','opencv-python']},
        source_sha256={str(p):sha(p) for p in source_files},
        entry='YOLO(local best.pt).val(): native loader, forward, YOLOv10 end-to-end postprocessing, matching and AP.',
        differences=['Existing Ultralytics 8.4.0 reused; not author demo 8.2.56 or checkpoint training 8.2.59.',
                     'No separate ISFire-4 official evaluator found; this is framework-native validation on a two-class remapping.',
                     'FP32/batch1/test split fixed for cross-GPU baseline. conf=.001/iou=.7/max_det=300, not demo display thresholds.',
                     'YOLOv10 end2end path retained; iou argument does not imply extra project NMS.',
                     'Native AP implementation/max_det and preprocessing differ from prior project COCO evaluation; not a single-factor test.',
                     'Author-linked GitHub provides a dataset download link; availability and identity to published 85% not verified. No dataset switch in this run.'],
        references=['https://huggingface.co/TommyNgx/YOLOv10-Fire-and-Smoke-Detection',
                    'https://github.com/X-Men01/YOLOv10-Fire-and-Smoke-Detection',
                    'https://huggingface.co/datasets/shahriar-5/ISFire-4'])
    write(run_dir/'protocol.json', protocol)
    snapshot = run_dir/'source_snapshot'; snapshot.mkdir()
    for i, p in enumerate(source_files): shutil.copy2(p, snapshot/f'{i:02d}_{p.name}')
    state = dict(sequence=46, state='running', images=len(selected))
    write(run_dir/'status.json', state)
    captured = []
    model.add_callback('on_val_end', lambda validator: captured.append(validator))
    try:
        result = model.val(**opts)
        validator = captured[0]
        actual_files = validator.dataloader.dataset.im_files
        assert validator.seen == len(selected) == len(set(actual_files))
        assert sorted(actual_files) == sorted(x['image'] for x in selected)
        assert validator.names == NAMES and validator.end2end
        actual_counts = result.nt_per_class.tolist()
        if not args.max_images: assert actual_counts == [counts[i] for i in range(2)]
        metrics = {k:float(v) for k,v in result.results_dict.items()}
        assert all(math.isfinite(v) for v in metrics.values())
        per_class = {NAMES[int(cid)]: dict(zip(['precision','recall','AP50','AP'], map(float,result.class_result(i))))
                     for i,cid in enumerate(result.ap_class_index)}
        write(run_dir/('smoke_metrics.json' if args.max_images else 'metrics.json'), dict(
            native_metrics=metrics, per_class=per_class, unit='fraction_0_to_1', images=validator.seen,
            official_comparable=False, actual_half=bool(validator.args.half), actual_options=vars(validator.args)))
        write(run_dir/'coverage.json', dict(images=actual_files, gt_instances=dict(zip(NAMES.values(),actual_counts))))
        state['state'] = 'smoke_complete' if args.max_images else 'complete'
        print(f'46号原生评测完成：{run_dir}', flush=True)
    except BaseException as exc:
        state.update(state='failed', error=repr(exc)); raise
    finally: write(run_dir/'status.json', state)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', default='0')
    parser.add_argument('--max-images', type=int, choices=[0,10], default=0)
    parser.add_argument('--preflight', action='store_true')
    args = parser.parse_args()
    if args.preflight:
        args.max_images = 0
        with tempfile.TemporaryDirectory(prefix='m46-native-preflight-') as tmp: run(args, Path(tmp))
    else: run(args, OUT)
