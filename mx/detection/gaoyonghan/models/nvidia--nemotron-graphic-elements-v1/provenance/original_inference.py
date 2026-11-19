"""Repeat M48 through NVIDIA's define_model entry; COCO scoring is explicit project glue."""
import argparse
from collections import Counter
from datetime import datetime
from importlib.metadata import version
import json
import math
import os
from pathlib import Path
import shutil
import sys
import tempfile

import numpy as np
import torch
from PIL import Image
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from supplemental_20260905.model48.run import CARD_AP, CARD_AR, DATA, MODEL, sha, write

HERE = Path(__file__).resolve().parent
OUT = Path('/home/ubuntu/gaoyonghan/gpu/results/m48_native_factory_v1')
OLD = Path('/home/ubuntu/gaoyonghan/gpu/results/remaining25_supplemental_20260905/m48-cached-val560-native10')
REV = '4a76546bb1bb4cbab3401361c91cf01706321805'


def preflight():
    previous = json.loads((OLD/'status.json').read_text())['protocol']
    assert sha(MODEL/'nemotron_graphic_elements_v1/weights.pth') == previous['weight_sha256']
    assert sha(DATA/'annotations_native10.json') == previous['annotation_sha256']
    assert sha(DATA/'source_manifest.json') == previous['source_manifest_sha256']
    for name, digest in previous['native_source_sha256'].items(): assert sha(MODEL/name) == digest, name
    annotation = json.loads((DATA/'annotations_native10.json').read_text())
    manifest = json.loads((DATA/'source_manifest.json').read_text())
    assert len(annotation['images']) == 560 and len(annotation['annotations']) == 11520
    assert [c['name'] for c in sorted(annotation['categories'],key=lambda c:c['id'])] == list(CARD_AP)
    assert len({i['id'] for i in annotation['images']}) == 560
    checks = {i['file_name']:i for i in manifest['image_checks']}
    for info in annotation['images']:
        path = DATA/'images'/info['file_name']
        assert sha(path) == checks[info['file_name']]['sha256']
        with Image.open(path) as im:
            assert im.size == (info['width'],info['height'])
            im.verify()
    print('Verified: original 560 images / 11520 boxes / 10 categories, weight and NVIDIA source hashes match previous run.', flush=True)
    return annotation, previous


def factory(root):
    # Populate an offline HF cache with a symlink, not a second copy of the weights.
    # This lets the unmodified define_model() use its own hf_hub_download call.
    cache = root/'hf-cache'
    os.environ['HF_HUB_CACHE'] = os.environ['HUGGINGFACE_HUB_CACHE'] = str(cache)
    os.environ['HF_HUB_OFFLINE'] = '1'
    repo = cache/'models--nvidia--nemotron-graphic-elements-v1'
    target = repo/'snapshots'/REV/'nemotron_graphic_elements_v1/weights.pth'
    target.parent.mkdir(parents=True, exist_ok=True)
    weight = MODEL/'nemotron_graphic_elements_v1/weights.pth'
    if target.is_symlink(): assert target.resolve() == weight.resolve()
    elif target.exists(): raise RuntimeError('Unexpected cache file; preserve it and inspect before rerun')
    else: target.symlink_to(weight)
    ref = repo/'refs/main'; ref.parent.mkdir(parents=True, exist_ok=True)
    if ref.exists(): assert ref.read_text() == REV
    else: ref.write_text(REV)
    sys.path.insert(0,str(MODEL))
    from huggingface_hub import hf_hub_download
    assert Path(hf_hub_download('nvidia/nemotron-graphic-elements-v1',
        'nemotron_graphic_elements_v1/weights.pth',local_files_only=True)).resolve() == weight.resolve()
    from nemotron_graphic_elements_v1.model import define_model
    model = define_model('graphic_element_v1')
    assert model.labels == list(CARD_AP) and model.normalize_boxes
    assert model.img_size == (1024,1024) and model.conf_thresh == .01 and model.iou_thresh == .25
    assert model.threshold == .1 and model.class_agnostic
    return model


def score(run, annotation, records, smoke):
    predictions = [d for r in records for d in r['detections']]
    write(run/'predictions.json', predictions)
    write(run/'annotation.json', annotation)
    gt = COCO(str(run/'annotation.json'))
    if predictions: dt = gt.loadRes(predictions)
    else:
        dt = COCO(); dt.dataset = {**gt.dataset,'annotations':[]}; dt.createIndex()
    evaluator = COCOeval(gt,dt,'bbox')
    evaluator.evaluate(); evaluator.accumulate(); evaluator.summarize()
    per_class = {}
    for k,cid in enumerate(evaluator.params.catIds):
        name = gt.cats[cid]['name']
        p = evaluator.eval['precision'][:,:,k,0,-1]; r = evaluator.eval['recall'][:,k,0,-1]
        ap = float(p[p>-1].mean()) if (p>-1).any() else None
        ar = float(r[r>-1].mean()) if (r>-1).any() else None
        per_class[name] = dict(AP=ap, AR100=ar, model_card_AP_percent=CARD_AP[name], model_card_AR_percent=CARD_AR[name])
    fields = ['AP','AP50','AP75','AP_small','AP_medium','AP_large','AR1','AR10','AR100','AR_small','AR_medium','AR_large']
    metrics = {k:float(v) if v>=0 else None for k,v in zip(fields,evaluator.stats)}
    write(run/('smoke_metrics.json' if smoke else 'metrics.json'), dict(metrics=metrics, per_class=per_class,
        evaluation_images=len(records), GT_instances=len(annotation['annotations']), unit='fraction_0_to_1',
        official_comparable=False, metric_source='Standard pycocotools COCOeval; NVIDIA publication evaluator not obtained.'))
    write(run/'coverage.json',dict(image_ids=[r['image_id'] for r in records],
        gt_instances=dict(Counter(gt.cats[a['category_id']]['name'] for a in annotation['annotations'])),
        prediction_counts=dict(Counter(d['class_name'] for d in predictions))))


def execute(args, root):
    annotation, previous = preflight()
    model = factory(root)
    if args.preflight:
        assert next(model.parameters()).device.type == 'cpu'
        print('READY: unmodified NVIDIA define_model() strictly loaded the local checkpoint on CPU; no forward executed.')
        return
    assert next(model.parameters()).device.type == 'cuda', 'No CPU inference fallback'
    from nemotron_graphic_elements_v1.utils import postprocess_preds_graphic_element
    images = annotation['images'][:args.max_images] if args.max_images else annotation['images']
    ids = {i['id'] for i in images}
    selected = {**annotation,'images':images,'annotations':[a for a in annotation['annotations'] if a['image_id'] in ids]}
    run = root/('smoke' if args.max_images else 'full')/datetime.now().strftime('run_%Y%m%d_%H%M%S_%f')
    run.mkdir(parents=True)
    versions = {n:version(n) for n in ['torch','torchvision','numpy','Pillow','pycocotools','huggingface-hub']}
    protocol = dict(sequence=48, model_id='nvidia/nemotron-graphic-elements-v1', source_revision=REV,
        weight_sha256=previous['weight_sha256'], annotation_sha256=previous['annotation_sha256'],
        source_manifest_sha256=previous['source_manifest_sha256'], native_source_sha256=previous['native_source_sha256'],
        software=versions, images=len(images), factory='Unmodified NVIDIA define_model("graphic_element_v1")',
        offline_weights='Official HF cache resolves main to pinned revision and symlinks original checked local weights; no download or source monkeypatch.',
        input_size=list(model.img_size), conf=model.conf_thresh, iou=model.iou_thresh,
        class_agnostic_nms=model.class_agnostic, extra_postprocess_threshold=model.threshold,
        dtype=str(next(model.parameters()).dtype), device=model.device, physical_device=os.environ.get('CUDA_VISIBLE_DEVICES'), batch_size=1,
        evaluator='pycocotools COCOeval bbox, IoU .50:.05:.95, maxDets [1,10,100]',
        official_comparable=False, previous_result=str(OLD),
        limitations=['NVIDIA publishes minimal inference code, not its complete benchmark evaluator. This is not a new claim of official score reproduction.',
                     'Previous run manually constructed the same Exp/model/strict checkpoint/wrapper. Factory rerun should be comparable; no threshold sweep.',
                     'Only normalized native xyxy to pixel COCO xywh and class-ID encoding are project conversion; no extra NMS or clipping.',
                     'Original GT coordinates, including 12 out-of-image boxes, remain unchanged.'])
    snapshot = run/'source_snapshot'; snapshot.mkdir()
    for p in [Path(__file__),HERE/'run_m48_native.sh',HERE/'supplemental_20260905/model48/run.py']:
        shutil.copy2(p,snapshot/p.name)
        protocol.setdefault('project_source_sha256',{})[str(p)] = sha(p)
    for name in [*previous['native_source_sha256'], 'LICENSE', 'THIRD_PARTY_NOTICES.md']:
        dest = snapshot/'nvidia'/name; dest.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(MODEL/name,dest)
    write(run/'protocol.json',protocol)
    status = dict(sequence=48,state='running',completed_images=0,total_images=len(images))
    write(run/'status.json',status)
    records = []
    try:
        with (run/'predictions.jsonl').open('w') as f:
            for index,info in enumerate(images,1):
                with Image.open(DATA/'images'/info['file_name']) as opened: image = np.asarray(opened.convert('RGB')).copy()
                with torch.inference_mode():
                    raw = model(model.preprocess(image),image.shape)[0]
                    boxes, labels, scores = postprocess_preds_graphic_element(raw,model.threshold,model.labels)
                boxes *= np.array([info['width'],info['height'],info['width'],info['height']])
                detections = []
                for box,label,confidence in zip(boxes,labels,scores):
                    x1,y1,x2,y2 = map(float,box); label=int(label); confidence=float(confidence)
                    assert all(math.isfinite(v) for v in [x1,y1,x2,y2,confidence]) and 0<=label<10 and 0<=confidence<=1
                    assert x2>x1 and y2>y1
                    detections.append(dict(image_id=info['id'],category_id=label+1,native_class_id=label,
                        class_name=model.labels[label],bbox=[x1,y1,x2-x1,y2-y1],score=confidence))
                row = dict(image_id=info['id'],file_name=info['file_name'],detections=detections)
                f.write(json.dumps(row,allow_nan=False)+'\n'); f.flush(); records.append(row)
                status['completed_images'] = index
                if index==1 or index%50==0 or index==len(images): print(f'48号 {index}/{len(images)}',flush=True)
        assert len(records)==len(ids) and {r['image_id'] for r in records}==ids
        score(run,selected,records,bool(args.max_images))
        status.update(state='smoke_complete' if args.max_images else 'complete', predictions_sha256=sha(run/'predictions.jsonl'))
        print(f'48号原生入口复跑完成：{run}',flush=True)
    except BaseException as exc:
        status.update(state='failed',error=repr(exc)); raise
    finally: write(run/'status.json',status)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--max-images',type=int,choices=[0,10],default=0)
    parser.add_argument('--preflight',action='store_true')
    args=parser.parse_args()
    if args.preflight:
        with tempfile.TemporaryDirectory(prefix='m48-native-preflight-') as tmp: execute(args,Path(tmp))
    else: execute(args,OUT)
