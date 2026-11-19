"""M60: unmodified model-card yolov5 7.0.6 validator on original valid57."""
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
import sys
import tempfile
import zipfile

HERE = Path(__file__).resolve().parent
OUT = Path('/data2/gaoyonghan/gpu_model_standalone_20261110/release/model/检测/TahaDouaji--detr-doc-table-detection/resources/runtime/assets/workspace/results/m60_native706_v1')
ARCHIVE = Path('/home/ubuntu/文档/ChatGPT/idea 2/outputs/launch_audit_20261005/migration_inputs/datasets/construction_safety_v1_valid/data/valid.zip')
WEIGHT = Path('/data2/gaoyonghan/gpu_model_standalone_20261110/release/model/检测/TahaDouaji--detr-doc-table-detection/resources/model_index/keremberke/yolov5n-construction-safety/best.pt')
PACKAGES = {
    'yolov5-7.0.6-py36.py37.py38-none-any.whl': '2f03334b7d1c53f6dd438b9c3853f529db84e3e0a5d650be82d0d350194d0930',
    'm60_ipython_runtime.zip': '4042fb0d695a5b48579c9d26dfb17306ce204367027079464870d1dac453ffe8',
}
REFERENCE = .36535576104287554


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
    return h.hexdigest()


def write(path, value):
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')


def runtime(root):
    site = root/'runtime'
    for name,digest in PACKAGES.items():
        assert sha(HERE/name)==digest, f'Changed offline package: {name}'
        with zipfile.ZipFile(HERE/name) as z:
            for item in z.infolist():
                if item.is_dir(): continue
                target=site/item.filename
                assert target.resolve().is_relative_to(site.resolve())
                if target.exists(): assert target.read_bytes()==z.read(item), f'Changed runtime: {target}'
                else:
                    target.parent.mkdir(parents=True,exist_ok=True)
                    target.write_bytes(z.read(item))
    sys.path.insert(0,str(site))
    os.environ['YOLOV5_CONFIG_DIR']=str(root/'config')
    os.environ['MPLCONFIGDIR']=str(root/'matplotlib')
    os.environ['YOLOv5_AUTOINSTALL']='false'
    from yolov5 import val
    from yolov5.utils.general import CONFIG_DIR
    assert version('yolov5')=='7.0.6' and site.resolve() in Path(val.__file__).resolve().parents
    font=CONFIG_DIR/'Arial.ttf'
    if not font.exists():
        source=Path('/data2/gaoyonghan/gpu_model_standalone_20261110/release/model/检测/TahaDouaji--detr-doc-table-detection/resources/runtime/assets/fonts/Arial.ttf')
        assert source.is_file(), 'Missing cached font; no automatic download'
        font.symlink_to(source)
    return val


def prepare(root):
    from PIL import Image
    import numpy as np
    assert sha(ARCHIVE)=='7332feb6f5abd953e579d71f5b450b1cde4976d0aa8eb5982cc2ca0db45a7935'
    assert sha(WEIGHT)=='3ce9a9eaf22c56759786ff7a52952d0d09c38877a53ce3eb016db2b2baceb989'
    with zipfile.ZipFile(ARCHIVE) as z:
        annotation=json.loads(z.read('_annotations.coco.json'))
        assert len(annotation['images'])==57 and len(annotation['annotations'])==221
        assert [c['id'] for c in annotation['categories']]==list(range(17))
        by_image=defaultdict(list)
        for a in annotation['annotations']:
            assert not a.get('iscrowd',0) and not a.get('ignore',0)
            by_image[a['image_id']].append(a)
        manifest=[]
        for info in sorted(annotation['images'],key=lambda i:i['id']):
            target=root/'dataset/images/valid'/f'{info["id"]}.jpg'
            target.parent.mkdir(parents=True,exist_ok=True)
            data=z.read(info['file_name'])
            if target.exists(): assert target.read_bytes()==data
            else: target.write_bytes(data)
            with Image.open(target) as im:
                assert im.format=='JPEG' and im.size==(info['width'],info['height']) and im.getexif().get(274,1)==1
                im.verify()
            W,H=info['width'],info['height']; lines=[]
            for a in by_image[info['id']]:
                x,y,w,h=map(float,a['bbox'])
                assert all(math.isfinite(v) for v in [x,y,w,h]) and w>0 and h>0
                coords=[(x+w/2)/W,(y+h/2)/H,w/W,h/H]
                assert all(0<=v<=1 for v in coords), a
                line=str(a['category_id'])+' '+' '.join(f'{v:.12g}' for v in coords)
                cx,cy,nw,nh=map(float,line.split()[1:])
                assert max(abs(v-u) for v,u in zip([(cx-nw/2)*W,(cy-nh/2)*H,nw*W,nh*H],[x,y,w,h]))<1e-5
                lines.append(line)
            label=root/'dataset/labels/valid'/f'{info["id"]}.txt';label.parent.mkdir(parents=True,exist_ok=True)
            text='\n'.join(lines)+('\n' if lines else '')
            if label.exists(): assert label.read_text()==text
            else: label.write_text(text)
            array=np.array([list(map(float,line.split())) for line in lines],dtype=np.float32)
            assert len(np.unique(array,axis=0))==len(lines), 'Duplicate GT requires explicit native-loader audit'
            manifest.append(dict(image_id=info['id'],source_file=info['file_name'],image=str(target),
                image_sha256=sha(target),label_sha256=sha(label),GT_count=len(lines)))
    return annotation,manifest


def execute(args,root):
    val=runtime(root)
    import torch
    from yolov5.utils.callbacks import Callbacks
    from yolov5.utils.dataloaders import create_dataloader
    from yolov5.utils.general import check_dataset
    annotation,manifest=prepare(root)
    checkpoint=torch.load(WEIGHT,map_location='cpu',weights_only=False)
    names=checkpoint['model'].names
    assert {i:n.lower() for i,n in names.items()}=={c['id']:c['name'] for c in annotation['categories']}
    assert checkpoint['opt']['imgsz']==640 and checkpoint['opt']['batch_size']==8
    selected=manifest[:args.max_images] if args.max_images else manifest
    ids={i['image_id'] for i in selected}
    counts=Counter(a['category_id'] for a in annotation['annotations'] if a['image_id'] in ids)
    run=root/('preflight' if args.preflight else 'smoke' if args.max_images else 'full')/datetime.now().strftime('run_%Y%m%d_%H%M%S_%f')
    run.mkdir(parents=True)
    listing=run/'valid.txt';listing.write_text(''.join(i['image']+'\n' for i in selected))
    config=run/'data.yaml';write(config,dict(path=str(root/'dataset'),train=str(listing),val=str(listing),nc=17,names=list(names.values())))
    data=check_dataset(str(config),autodownload=False)
    # Use the very same native validation loader, without model forward, to check every image and GT.
    loader,dataset=create_dataloader(data['val'],640,16,32,False,pad=.5,rect=True,workers=0)
    native_counts=Counter();loaded=[]
    for _,labels,paths,_ in loader:
        loaded.extend(int(Path(p).stem) for p in paths)
        native_counts.update(map(int,labels[:,1].tolist()))
    assert len(loaded)==len(ids) and set(loaded)==ids and native_counts==counts
    if args.preflight:
        assert next(checkpoint['model'].parameters()).device.type=='cpu'
        print(f'READY: native 7.0.6 loader verified {len(ids)} images / {sum(counts.values())} boxes; original CPU checkpoint loaded; no forward executed.')
        return
    assert args.device.isdigit() and torch.cuda.is_available(), 'One GPU required; no silent CPU fallback'
    options=dict(data=str(config),weights=str(WEIGHT),batch_size=16,imgsz=640,conf_thres=.001,
        iou_thres=.6,max_det=300,task='val',device=args.device,workers=0,single_cls=False,
        augment=False,half=True,plots=False,save_json=False,project=str(run.parent),name=run.name,exist_ok=True)
    protocol=dict(sequence=60,model_id='keremberke/yolov5n-construction-safety',weight_sha256=sha(WEIGHT),
        archive_sha256=sha(ARCHIVE),annotation_source='Original data/valid.zip/_annotations.coco.json',
        dataset_relation='original validation source',source_images=selected,model_names=names,
        options=options,loader=dict(rect=True,pad=.5,stride=32),multi_label_nms=True,
        packages=PACKAGES,software={n:version(n) for n in ['yolov5','torch','torchvision','numpy','Pillow','ipython']},
        training_record={k:checkpoint['opt'][k] for k in ['epochs','batch_size','imgsz','seed']},
        reference_AP50=REFERENCE,reference_url='https://huggingface.co/keremberke/yolov5n-construction-safety',
        limitations=['Unmodified yolov5-pip 7.0.6 val.run; no project NMS, matching or AP calculation.',
            'Batch16 is twice checkpoint training batch8 as original train.py validation; FP16 follows original best-model validation. Other dependencies are reused, not a full historical environment recreation.',
            'Native validation conf.001/IoU.6/multi_label=True; card conf.25/IoU.45 is a display example, not the benchmark.',
            'COCO xywh -> YOLO normalized xywh only, round-trip checked; original numeric class IDs and image bytes retained.',
            'Six of seventeen categories have no GT in full valid. Report null for their AP, not zero or the overall AP fallback.',
            'save_json=False avoids the package hard-coded COCO scoring path; native exporter is called via observation callback.',
            'Checkpoint upload credentials, if present, are excluded from protocol; only four safe training fields are recorded.'])
    snapshot=run/'source_snapshot';snapshot.mkdir()
    sources=[Path(__file__),HERE/'run_m60_native.sh',Path(val.__file__),
        Path(inspect.getfile(create_dataloader)),Path(inspect.getfile(val.ap_per_class)),Path(inspect.getfile(val.non_max_suppression))]
    for i,path in enumerate(sources):
        shutil.copy2(path,snapshot/f'{i}_{path.name}')
        protocol.setdefault('source_sha256',{})[str(path)]=sha(path)
    write(run/'protocol.json',protocol)
    seen=[];predictions=[];callbacks=Callbacks()
    def image_end(pred,predn,path,names,image):
        val.save_one_json(predn,predictions,path,list(range(17)))
    def batch_end(batch_i,im,targets,paths,shapes,preds):
        seen.extend(int(Path(p).stem) for p in paths)
    callbacks.register_action('on_val_image_end',callback=image_end)
    callbacks.register_action('on_val_batch_end',callback=batch_end)
    status=dict(state='running',total_images=len(ids),completed_images=0)
    write(run/'status.json',status)
    try:
        result,maps,map50s,times=val.run(**options,callbacks=callbacks)
        assert len(seen)==len(ids) and set(seen)==ids, 'Native inference skipped/duplicated images'
        metrics=dict(zip(['precision','recall','AP50','AP'],map(float,result[:4])))
        assert all(math.isfinite(v) and 0<=v<=1 for v in metrics.values())
        per_class={name:dict(GT=counts[i],AP=float(maps[i]) if counts[i] else None,
            AP50=float(map50s[i]) if counts[i] else None) for i,name in names.items()}
        write(run/'predictions.json',predictions)
        write(run/('smoke_metrics.json' if args.max_images else 'metrics.json'),dict(metrics=metrics,
            per_class=per_class,unit='fraction_0_to_1',evaluation_images=len(seen),GT_instances=sum(counts.values()),
            reference_AP50=REFERENCE,AP50_difference_percentage_points=(metrics['AP50']-REFERENCE)*100 if not args.max_images else None,
            evaluator='unmodified yolov5 7.0.6 val.run / ap_per_class',timings_ms=list(map(float,times))))
        status.update(state='smoke_complete' if args.max_images else 'complete',predictions_sha256=sha(run/'predictions.json'))
        print(f'60号完成：{run}',flush=True)
    except BaseException as exc:
        status.update(state='failed',error=repr(exc));raise
    finally:
        status.update(completed_images=len(seen),processed_image_ids=seen);write(run/'status.json',status)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--device',default='0');p.add_argument('--max-images',type=int,choices=[0,10],default=0)
    p.add_argument('--preflight',action='store_true');args=p.parse_args()
    if args.preflight:
        with tempfile.TemporaryDirectory(prefix='m60-preflight-') as tmp:execute(args,Path(tmp))
    else:execute(args,OUT)
