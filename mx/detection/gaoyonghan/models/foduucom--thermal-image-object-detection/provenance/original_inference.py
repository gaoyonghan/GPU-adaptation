"""M47: original 8.0.43 validator on eight supported FLIR thermal classes."""
import argparse
from collections import Counter, defaultdict
from datetime import datetime
import hashlib
from importlib.metadata import version
import inspect
import json
import math
import os
from pathlib import Path
import shutil
import tempfile

from PIL import Image
import numpy as np

HERE = Path(__file__).resolve().parent
DATA = Path('/data2/gaoyonghan/gpu/dataset/flir_adas_thermal_eval/images_thermal_val')
WEIGHT = Path('/data2/gaoyonghan/gpu/model/foduucom/thermal-image-object-detection/best.pt')
OUT = Path('/home/ubuntu/gaoyonghan/gpu/results/m47_native8043_flir8_v1')
NAMES = dict(enumerate(['empty','person','bicycle','car','motorcycle','airplane','bus','train',
                        'truck','boat','traffic light','fire hydrant','stop sign','parking meter']))
SOURCE_NAMES = {1:'person', 2:'bike', 3:'car', 4:'motor', 6:'bus', 8:'truck', 10:'light', 11:'hydrant'}
COUNTS = {1:4470, 2:170, 3:7133, 4:55, 6:179, 8:46, 10:2005, 11:94}
NATIVE_COUNTS = {**COUNTS, 3:7128, 10:2002}


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''): h.update(block)
    return h.hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def prepare(root):
    annotation = DATA/'coco.json'
    assert sha(annotation) == '14726d929f26229fd0b5059f8ca02613473affd66fda3fd790213a98b59ba486', 'Source annotations changed'
    assert sha(WEIGHT) == '83f4bfa1779754936bf7897563246f2b2cb5c32f76a67730da5e7b1206910953', 'Checkpoint changed'
    d = json.loads(annotation.read_text())
    cats = {c['id']:c['name'] for c in d['categories']}
    assert {i:cats[i] for i in SOURCE_NAMES} == SOURCE_NAMES
    assert len(d['images']) == len({i['id'] for i in d['images']}) == 1144
    assert len(d['annotations']) == len({a['id'] for a in d['annotations']})
    images = {i['id']:i for i in d['images']}
    by_image, excluded, counts = defaultdict(list), Counter(), Counter()
    for a in d['annotations']:
        assert a['image_id'] in images
        if a['category_id'] not in SOURCE_NAMES:
            excluded[cats[a['category_id']]] += 1
            continue
        assert not a.get('iscrowd') and not a.get('ignore'), 'YOLO labels cannot preserve crowd/ignore'
        by_image[a['image_id']].append(a)
    for folder in ['images/val','labels/val']: (root/'dataset'/folder).mkdir(parents=True, exist_ok=True)
    manifest = []
    for info in sorted(d['images'], key=lambda i:i['id']):
        source = (DATA/info['file_name']).resolve()
        assert source.is_relative_to(DATA.resolve())
        target = root/'dataset/images/val'/f'{info["id"]}{source.suffix}'
        with Image.open(source) as im:
            assert im.size == (info['width'],info['height']) and im.getexif().get(274,1) == 1
            im.verify()
        if target.exists(): assert sha(target) == sha(source), 'Prepared image changed'
        else: shutil.copyfile(source, target)
        rows = []
        for a in by_image[info['id']]:
            cid = a['category_id']  # Native checkpoint IDs match these verified FLIR IDs.
            x,y,w,h = a['bbox']; W,H = info['width'],info['height']
            assert all(math.isfinite(v) for v in [x,y,w,h]) and min(w,h)>0
            assert 0 <= x < x+w <= W and 0 <= y < y+h <= H
            coords = [(x+w/2)/W,(y+h/2)/H,w/W,h/H]
            row = str(cid)+' '+' '.join(f'{v:.12g}' for v in coords)
            cx,cy,nw,nh = map(float,row.split()[1:])
            restored = [(cx-nw/2)*W,(cy-nh/2)*H,nw*W,nh*H]
            assert max(abs(v-u) for v,u in zip(restored,a['bbox'])) < 1e-5
            rows.append(row); counts[cid] += 1
        unique_count = len(np.unique(np.array([list(map(float,r.split())) for r in rows], dtype=np.float32), axis=0)) if rows else 0
        label = root/'dataset/labels/val'/f'{info["id"]}.txt'
        content = '\n'.join(rows)+ ('\n' if rows else '')
        if label.exists(): assert label.read_text() == content
        else: label.write_text(content)
        manifest.append(dict(image_id=info['id'], source_image=str(source), image=str(target),
                             image_sha256=sha(target), label_sha256=sha(label), boxes=len(rows),
                             native_duplicate_removals=len(rows)-unique_count))
    assert counts == COUNTS
    assert {x['image_id']:x['native_duplicate_removals'] for x in manifest if x['native_duplicate_removals']} == {5808:3,5825:5}
    print('Prepared FLIR thermal validation: 1144 images, 14152 original boxes; native loader removes 8 duplicates -> 14144 scored boxes.', flush=True)
    return manifest, dict(excluded)


def execute(args, root):
    # Keep plotting caches with runtime artifacts, not in the original dataset or shared environment.
    os.environ['MPLCONFIGDIR'] = str(root/'matplotlib')
    import torch
    import ultralytics
    from ultralytics.yolo.cfg import get_cfg
    from ultralytics.yolo.v8.detect.val import DetectionValidator
    from ultralytics.yolo.data.utils import check_det_dataset
    from ultralytics.yolo.utils import metrics as native_metrics
    assert ultralytics.__version__ == '8.0.43' and version('ultralyticsplus') == '0.0.28'
    manifest, excluded = prepare(root)
    checkpoint = torch.load(WEIGHT, map_location='cpu', weights_only=False)
    model = checkpoint['model']
    assert checkpoint['version'] == '8.0.43' and model.names == NAMES and model.model[-1].nc == 14
    selected = manifest[:args.max_images] if args.max_images else manifest
    run = root/('preflight' if args.preflight else 'smoke' if args.max_images else 'full')/datetime.now().strftime('run_%Y%m%d_%H%M%S_%f')
    run.mkdir(parents=True)
    (run/'val.txt').write_text('\n'.join(x['image'] for x in selected)+'\n')
    # Keep all 14 original output IDs; only eight GT classes enter the AP average.
    write(run/'data.yaml', dict(path=str(run), train='val.txt', val='val.txt', names=NAMES))
    options = dict(data=str(run/'data.yaml'), split='val', task='detect', mode='val',
                   imgsz=640, batch=1, workers=0, conf=.001, iou=.7, max_det=300,
                   device='cpu' if args.preflight else args.device, half=False, rect=True,
                   augment=False, single_cls=False, agnostic_nms=False, save_json=True,
                   save_txt=False, plots=False, project=str(run.parent), name=run.name, exist_ok=True)
    cfg = get_cfg(overrides=options)
    if args.preflight:
        validator = DetectionValidator(args=cfg, save_dir=run)
        validator.data = check_det_dataset(options['data'], autodownload=False)
        validator.stride = model.stride
        loader = validator.get_dataloader(validator.data['val'], 1)
        seen, counts = [], Counter()
        for batch in loader:
            seen.extend(int(Path(p).stem) for p in batch['im_file'])
            counts.update(map(int,batch['cls'].flatten().tolist()))
        assert sorted(seen) == sorted(x['image_id'] for x in manifest) and len(set(seen)) == 1144
        assert counts == NATIVE_COUNTS and next(model.parameters()).device.type == 'cpu'
        print('READY: original CPU checkpoint and all 1144 images / 14144 boxes through native loader verified; NO forward executed.')
        return
    assert args.device.isdigit() and torch.cuda.is_available(), 'Use one GPU; no CPU inference fallback'
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    sources = [Path(__file__), HERE/'run_m47_native.sh', Path(inspect.getfile(DetectionValidator)), Path(native_metrics.__file__)]
    protocol = dict(sequence=47, model_id='foduucom/thermal-image-object-detection',
        weight_sha256=sha(WEIGHT), annotation_sha256=sha(DATA/'coco.json'), source_images=selected,
        dataset='FLIR ADAS thermal validation, eight-class subset', dataset_relation='public_replacement',
        official_comparable=False, source_names=SOURCE_NAMES, checkpoint_names=NAMES, selected_class_ids=list(SOURCE_NAMES),
        excluded_gt_instances=excluded, full_gt_instances_before_native_dedup=COUNTS,
        full_gt_instances_after_native_dedup=NATIVE_COUNTS, options=options,
        checkpoint_version=checkpoint['version'], checkpoint_train_args=checkpoint['train_args'],
        local_model_config=json.loads((WEIGHT.parent/'config.json').read_text()),
        software={n:version(n) for n in ['ultralytics','ultralyticsplus','torch','torchvision','numpy','Pillow']},
        source_sha256={str(p):sha(p) for p in sources},
        entry='Ultralytics 8.0.43 DetectionValidator: original loader, forward, NMS, matching and AP; subclass only logs image IDs.',
        label_conversion='COCO xywh -> normalized YOLO xywh with round-trip check; preserve original numeric model IDs, no box clipping/merging.',
        limitations=['Eight positive GT categories only; all 14 output IDs preserved. Native AP averages GT-present categories, absent categories are not scored as zero.',
                     'No manual filtering of model outputs before native NMS. Predictions for unscored classes may remain in saved JSON.',
                     'Keep all 14152 source labels; original native loader removes 8 exact duplicates (5 car, 3 traffic light) on images 5808/5825. Scored GT=14144; no custom deduplication.',
                     'FLIR sign is not assumed to mean stop sign; empty class meaning is unresolved. Neither is scored.',
                     'Standalone FP32/batch1 validation; inference thresholds follow checkpoint args, not display example.',
                     'Eight classes and native 8.0.43 scoring differ from historical six-class COCO run; overall AP is not directly comparable.',
                     'This is framework-native validation on a public replacement, not a claimed FLIR author script or model-card metric reproduction.'])
    write(run/'protocol.json', protocol)
    snapshot = run/'source_snapshot'; snapshot.mkdir()
    for i,p in enumerate(sources): shutil.copy2(p,snapshot/f'{i:02d}_{p.name}')
    del model, checkpoint
    seen = []

    class CoverageValidator(DetectionValidator):
        def update_metrics(self, preds, batch):
            seen.extend(int(Path(p).stem) for p in batch['im_file'])
            return super().update_metrics(preds, batch)

    state = dict(sequence=47, state='running', images=len(selected))
    write(run/'status.json', state)
    try:
        validator = CoverageValidator(args=cfg, save_dir=run)
        validator(model=str(WEIGHT))
        assert sorted(seen) == sorted(x['image_id'] for x in selected)
        assert len(set(seen)) == validator.seen == len(selected) and validator.names == NAMES
        if not args.max_images:
            assert validator.nt_per_class.tolist() == [NATIVE_COUNTS.get(i,0) for i in range(14)]
            assert set(map(int,validator.metrics.ap_class_index)) == set(SOURCE_NAMES)
        per_class = {NAMES[int(cid)]:dict(zip(['precision','recall','AP50','AP'],map(float,validator.metrics.class_result(i))))
                     for i,cid in enumerate(validator.metrics.ap_class_index)}
        result = {k:float(v) for k,v in validator.metrics.results_dict.items()}
        assert all(math.isfinite(v) for v in result.values())
        write(run/('smoke_metrics.json' if args.max_images else 'metrics.json'), dict(
            native_metrics=result, per_class=per_class, unit='fraction_0_to_1', images=len(seen),
            evaluated_classes=len(per_class), model_output_classes=14, official_comparable=False,
            actual_half=bool(validator.args.half), actual_options=vars(validator.args)))
        write(run/'coverage.json', dict(image_ids=seen, gt_instances={NAMES[i]:int(v) for i,v in enumerate(validator.nt_per_class)},
                                        excluded_gt_instances=excluded))
        state['state'] = 'smoke_complete' if args.max_images else 'complete'
        print(f'47号FLIR八类原生评测完成：{run}', flush=True)
    except BaseException as exc:
        state.update(state='failed', error=repr(exc)); raise
    finally: write(run/'status.json', state)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', default='0')
    parser.add_argument('--max-images', type=int, choices=[0,10], default=0)
    parser.add_argument('--preflight', action='store_true')
    args = parser.parse_args()
    if args.preflight:
        args.max_images = 0
        with tempfile.TemporaryDirectory(prefix='m47-native-preflight-') as tmp: execute(args,Path(tmp))
    else: execute(args,OUT)
