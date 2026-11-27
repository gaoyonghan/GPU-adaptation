"""M54 weights on public SKU110K test, scored by the original CVPR2020 challenge code."""
import argparse
import csv
from datetime import datetime
import hashlib
import importlib
from importlib.metadata import version
import json
import math
import os
from pathlib import Path
import shutil
import sys
import tempfile
import zipfile

import numpy as np
from PIL import Image, ImageFile

HERE = Path(__file__).resolve().parent
DATA = Path('__DATASET_ROOT__/prepared/sku110k_test')
WEIGHT = Path('__MODEL_ROOT__/weights/foduucom/product-detection-in-shelf-yolov8/best.pt')
OUT = Path('__OUTPUT_ROOT__/m054')
REV = '9303b450077244fb8c7729717b67c2da6c807819'
URL = f'https://raw.githubusercontent.com/retailvisionworkshop/retailvisionworkshop.github.io/{REV}/_data/densely_packed_eval_2020-02-06.zip'
ZIP_SHA = '2db06f173154650e8a182ad9a905a7e3b7b8e21fd9c829f2367d946fa77a603f'
FILES = {
    'demo.py': '00cc769e93c115a9347bd71af425b7e86c0da79be96fec0905cfa00b67a005aa',
    'eval.py': '9d9278267fff1d4de3018179849294e825eba65d8e272dfe03bebbc27fc3d6b8',
    'utils.py': '977dd7e3f869ae9514d0987fcb55178ac4ad1eac95469553bface32ce86c32dd',
    'readme.txt': '8c649b0d3cacaa825534e0556ea46344bdf3733f826310c28de54d0fa100be00',
}


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda: f.read(1024*1024), b''): h.update(b)
    return h.hexdigest()


def write(path, value):
    temp = path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    temp.replace(path)


def official_source(root):
    root.mkdir(parents=True, exist_ok=True)
    archive = root/'densely_packed_eval_2020-02-06.zip'
    if not archive.exists():
        bundled = HERE/archive.name
        assert bundled.is_file(), f'Missing bundled official scoring package: {bundled}'
        assert sha(bundled) == ZIP_SHA, 'Bundled official package hash mismatch'
        shutil.copyfile(bundled,archive)
        print('Using bundled, SHA256-verified official scoring package; no download.', flush=True)
    assert sha(archive) == ZIP_SHA
    with zipfile.ZipFile(archive) as z:
        for name, digest in FILES.items():
            data = z.read(name)
            assert hashlib.sha256(data).hexdigest() == digest
            path = root/name
            if path.exists(): assert sha(path) == digest, f'Changed official source: {path}'
            else: path.write_bytes(data)
    # Original demo uses top-level imports; this is a dedicated process with no prior utils/eval modules.
    assert not any(n in sys.modules for n in ['demo', 'eval', 'utils']), 'Module name collision'
    sys.path.insert(0, str(root))
    demo = importlib.import_module('demo')
    evaluator = importlib.import_module('eval')
    utils = importlib.import_module('utils')
    for module in [demo, evaluator, utils]: assert Path(module.__file__).parent.resolve() == root.resolve()
    return demo, evaluator, utils


def read_image(path):
    repair = None
    try:
        with Image.open(path) as im: image = np.asarray(im.convert('RGB')).copy()
    except OSError as exc:
        if 'truncated' not in str(exc): raise
        repair = str(exc)
        old = ImageFile.LOAD_TRUNCATED_IMAGES
        try:
            ImageFile.LOAD_TRUNCATED_IMAGES = True
            with Image.open(path) as im: image = np.asarray(im.convert('RGB')).copy()
        finally: ImageFile.LOAD_TRUNCATED_IMAGES = old
    return image, repair


def preflight(max_images=0):
    assert sha(WEIGHT) == 'aac44618866c15502f8158778c1b2ae0e8f66b57a9a7bce318b269c7b283b148'
    assert sha(DATA/'annotations_test.csv') == 'cd905bc73a3e97e5d458f48b4ac0622bf3b1a4c4f347ee5f25fda96e6a72315f'
    groups = {}
    with (DATA/'annotations_test.csv').open(newline='') as f:
        for row in csv.reader(f):
            name,x1,y1,x2,y2,label,width,height = row
            box = list(map(float,[x1,y1,x2,y2]))
            assert all(math.isfinite(v) for v in box) and box[2]>box[0] and box[3]>box[1]
            assert Path(name).name == name and label == 'object'
            info = groups.setdefault(name, dict(name=name,width=int(width),height=int(height),rows=[]))
            assert (info['width'],info['height']) == (int(width),int(height))
            info['rows'].append(row)
    assert len(groups)==2936 and sum(len(i['rows']) for i in groups.values())==431546
    selected = list(groups.values())[:max_images] if max_images else list(groups.values())
    for index,info in enumerate(selected,1):
        path = DATA/'images'/info['name']
        image, repair = read_image(path)
        assert image.shape[:2] == (info['height'],info['width'])
        info.update(sha256=sha(path), decoder_repair=repair)
        if index%500==0: print(f'CPU image verification {index}/{len(selected)}',flush=True)
    print(f'Verified original checkpoint, all 431546 CSV boxes and {len(selected)}/2936 test images.', flush=True)
    return list(groups.values())


def score(run, modules, smoke):
    demo, evaluator, utils = modules
    gt = str(run/'annotations_test.csv'); dt = str(run/'predictions.csv')
    print('Official scoring: 10 IoU thresholds; CPU scoring can take several minutes.', flush=True)
    ap = float(demo.eval_results(gt,dt))
    gi,gb,_ = utils.read_boxes_file(gt,has_confidence=False)
    di,db,dc = utils.read_boxes_file(dt,has_confidence=True)
    ap50 = float(evaluator.eval_single_threshold(gi,gb,di,db,dc,ovthresh=.5))
    ap75 = float(evaluator.eval_single_threshold(gi,gb,di,db,dc,ovthresh=.75))
    result = dict(metrics=dict(AP=ap,AP50=ap50,AP75=ap75), unit='fraction_0_to_1',
        evaluation_images=len(set(gi)), GT_instances=len(gi), detections=len(di),
        evaluator='Unmodified CVPR2020 Retail-Vision densely_packed_eval_2020-02-06',
        dataset='SKU110K_fixed public test, not hidden CVPR2020 competition test',
        coverage=dict(scored=['product'], not_scored=['empty']), model_card_comparable=False,
        model_card_AP50_reference=.91)
    write(run/('smoke_metrics.json' if smoke else 'metrics.json'),result)
    return result


def prediction_rows(name, boxes):
    rows, collapsed = [], 0
    for box in boxes:
        x1,y1,x2,y2,confidence,label = box
        if not (all(math.isfinite(v) for v in box) and label in (0,1)
                and x2>=x1 and y2>=y1 and 0<=confidence<=1):
            raise ValueError(f'Invalid native prediction: image={name}, xyxy_conf_class={box}')
        # Native 8.0.43 clips and rounds coordinates. Equal endpoints are valid
        # for the original challenge scorer's inclusive (+1) pixel geometry.
        collapsed += int(x2==x1 or y2==y1)
        if int(label)==1: rows.append([name,x1,y1,x2,y2,confidence])
    return rows, collapsed


def execute(args, root):
    os.environ['MPLCONFIGDIR'] = str(root/'matplotlib')
    modules = official_source(root/'official_source')
    images = preflight(args.max_images)
    import torch
    from ultralytics import YOLO
    assert version('ultralytics') == '8.0.43' and version('ultralyticsplus') == '0.0.28'
    checkpoint = torch.load(WEIGHT,map_location='cpu',weights_only=False)
    assert checkpoint['model'].names == {0:'empty',1:'product'}
    assert checkpoint['version'] == '8.0.43'
    if args.preflight:
        assert next(checkpoint['model'].parameters()).device.type == 'cpu'
        native = YOLO(str(WEIGHT))
        assert native.names == {0:'empty',1:'product'}
        print('READY: CPU checkpoint, all images, CSV and original evaluator checked; no forward executed.')
        return
    assert args.device.isdigit() and torch.cuda.is_available(), 'GPU required; no silent CPU fallback'
    selected = images[:args.max_images] if args.max_images else images
    run = root/('smoke' if args.max_images else 'full')/datetime.now().strftime('run_%Y%m%d_%H%M%S_%f')
    run.mkdir(parents=True)
    options = dict(imgsz=640,conf=.001,iou=.7,max_det=1000,device=args.device,half=False,
        augment=False,agnostic_nms=False,save=False,save_txt=False,verbose=False,
        project=str(run),name='inference',exist_ok=True)
    protocol = dict(sequence=54,model_id='foduucom/product-detection-in-shelf-yolov8',
        weight_sha256=sha(WEIGHT),checkpoint_version=checkpoint['version'],
        checkpoint_train_args=checkpoint.get('train_args'), inference_options=options,
        original_csv_sha256=sha(DATA/'annotations_test.csv'), dataset_relation='public_replacement',
        official_evaluator_url=URL, official_evaluator_zip_sha256=ZIP_SHA, source_sha256=FILES,
        software={n:version(n) for n in ['torch','torchvision','ultralytics','ultralyticsplus','numpy','Pillow']},
        source_images=[{k:v for k,v in i.items() if k!='rows'} for i in selected],
        limitations=['Original YOLO checkpoint, not the SKU110K RetinaNet model; no author-specific EM merging.',
            'Original challenge scorer: inclusive +1 IoU geometry, strict IoU > threshold, VOC integral AP; no COCO maxDets100 cap.',
            'Official demo targets annotations_test.csv. We use public fixed test, not the hidden competition test set.',
            'Only class1 product exported; class0 empty is unscored, not merged into product. Both classes remain through native NMS.',
            'Keep equal-endpoint boxes from native clipping/rounding unchanged: the official scorer uses inclusive +1 geometry. Record their count; reject reversed/nonfinite boxes.',
            'Inference 8.0.43; conf.001/iou.7 are validation settings, not an assertion about author 91% protocol; max_det1000 follows card display cap.',
            'Old result used 8.4.0/max_det300/COCO maxDets100; changes are not a pure scorer-only comparison.',
            'RGB decoded by PIL then supplied as BGR numpy to native YOLO; truncated JPEG tolerant retry recorded, no image omitted.'],
        model_card_comparable=False)
    with (run/'annotations_test.csv').open('w',newline='') as f:
        csv.writer(f).writerows(row for info in selected for row in info['rows'])
    snapshot = run/'source_snapshot'; snapshot.mkdir()
    for path in [Path(__file__),HERE/'run_m54_sku_official.sh']:
        shutil.copy2(path,snapshot/path.name)
        protocol.setdefault('runner_sha256',{})[path.name]=sha(path)
    shutil.copytree(root/'official_source',snapshot/'official')
    write(run/'protocol.json',protocol)
    status = dict(state='running',total_images=len(selected),completed_images=0,equal_endpoint_native_boxes=0)
    write(run/'status.json',status)
    try:
        model = YOLO(str(WEIGHT))
        with (run/'predictions.csv').open('w',newline='') as f, (run/'predictions.jsonl').open('w') as records:
            writer = csv.writer(f)
            for index,info in enumerate(selected,1):
                rgb,repair = read_image(DATA/'images'/info['name'])
                assert repair == info['decoder_repair']
                result = model.predict(source=np.ascontiguousarray(rgb[:,:,::-1]),**options)[0]
                boxes = result.boxes.data.cpu().numpy().tolist()
                rows,collapsed = prediction_rows(info['name'],boxes)
                writer.writerows(rows)
                status['equal_endpoint_native_boxes'] += collapsed
                if collapsed: print(f'{info["name"]}: retained {collapsed} native equal-endpoint boxes for official +1 scoring.',flush=True)
                records.write(json.dumps(dict(file_name=info['name'],native_xyxy_conf_class=boxes),allow_nan=False)+'\n')
                f.flush(); records.flush()
                status['completed_images']=index
                if index==1 or index%100==0 or index==len(selected):
                    write(run/'status.json',status); print(f'54号 {index}/{len(selected)}',flush=True)
        status['state']='scoring'; write(run/'status.json',status)
        metrics = score(run,modules,bool(args.max_images))
        status.update(state='smoke_complete' if args.max_images else 'complete',
            predictions_sha256=sha(run/'predictions.csv'), prediction_records_sha256=sha(run/'predictions.jsonl'))
        print(json.dumps(metrics,ensure_ascii=False),flush=True)
        print(f'54号完成：{run}',flush=True)
    except BaseException as exc:
        status.update(state='failed',error=repr(exc)); raise
    finally: write(run/'status.json',status)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--device',default='0')
    p.add_argument('--max-images',type=int,choices=[0,10],default=0)
    p.add_argument('--preflight',action='store_true')
    args=p.parse_args()
    if args.preflight:
        with tempfile.TemporaryDirectory(prefix='m54-sku-preflight-') as tmp: execute(args,Path(tmp))
    else: execute(args,OUT)
