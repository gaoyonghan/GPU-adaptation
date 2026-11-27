"""YOLO face detection with WIDER FACE easy/medium/hard evaluation."""
import argparse
import fcntl
import io
import json
import sys
import time
import zipfile
from pathlib import Path
import numpy as np
from PIL import Image
from scipy.io import loadmat
from run_model import core, digest, file_hash

HERE=Path(__file__).resolve().parent
DATA=Path('__DATASET_ROOT__/widerface_val')
EVAL=HERE/'assets/widerface_evaluator'

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--max-images',type=int,default=10);parser.add_argument('--device',default='cuda:0');args=parser.parse_args()
    out=Path('__MODEL_ROOT__/resources/runtime/assets/workspace/results/remaining25_wider_v2/YOLOv11n-face-detection');out.mkdir(parents=True,exist_ok=True)
    with (out/'.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        weight=Path('__MODEL_ROOT__/weights/AdamCodd/YOLOv11n-face-detection/model.pt')
        gt_dir=DATA/'eval_tools/ground_truth';gt=loadmat(gt_dir/'wider_face_val.mat')
        samples=[(str(e[0][0]),str(f[0][0])) for e,fs in zip(gt['event_list'],gt['file_list']) for f in fs[0]]
        assert len(samples)==3226 and len(set(samples))==3226
        protocol={'sequence':26,'model_id':'AdamCodd/YOLOv11n-face-detection','dataset':'WIDER FACE validation 3226',
          'weight_sha256':file_hash(weight),'script_sha256':file_hash(Path(__file__)),
          'gt_sha256':{p.name:file_hash(p) for p in gt_dir.glob('*.mat')},'evaluator_sha256':{p.name:file_hash(p) for p in EVAL.glob('*') if p.suffix in {'.py','.pyx','.so','.json'}},
          'evaluator_revision':json.loads((EVAL/'revision.json').read_text()),'evaluator_patches':['numpy.float -> numpy.float64','remove unused IPython debugger import','return AP list','precision=0 when no proposals'],
          'input_size':640,'conf':.001,'nms_iou':.7,'max_det':3000,'half':False,'iou_match':.5,'device':args.device,
          'fidelity':'reference_baseline; author inference settings not fully specified','runtime_versions':core.runtime_versions()}
        fingerprint=digest(protocol);path=out/'status.json';old=json.loads(path.read_text()) if path.exists() else None
        if old and old['protocol_fingerprint']!=fingerprint:raise RuntimeError('Protocol changed; preserve old result')
        if old and old['state']=='complete':return
        if args.max_images==0 and not(old and old.get('smoke_passed')):raise RuntimeError('First run smoke')
        status={'sequence':26,'task':'widerface','model_id':protocol['model_id'],'protocol':protocol,'protocol_fingerprint':fingerprint,'state':'running','started_at':time.time()}
        core.atomic_json(path,status)
        try:
            from ultralytics import YOLO
            model=YOLO(str(weight));assert list(model.names.values())==['face'],model.names
            records=[json.loads(line) for line in (out/'predictions.jsonl').read_text().splitlines()] if (out/'predictions.jsonl').exists() else [];done={(r['event'],r['filename']) for r in records};assert len(done)==len(records)
            with zipfile.ZipFile(DATA/'data/WIDER_val.zip') as z:
                for i,(event,name) in enumerate(samples[:args.max_images] if args.max_images else samples,1):
                    if (event,name) in done:continue
                    image=Image.open(io.BytesIO(z.read(f'WIDER_val/images/{event}/{name}.jpg'))).convert('RGB')
                    result=model.predict(image,imgsz=640,conf=.001,iou=.7,max_det=3000,half=False,device=args.device,verbose=False)[0]
                    boxes=[]
                    for box,score in zip(result.boxes.xyxy.cpu().tolist(),result.boxes.conf.cpu().tolist()):
                        x1,y1,x2,y2=box;boxes.append([x1,y1,x2-x1,y2-y1,score])
                    boxes.sort(key=lambda b:-b[4]);core.append_jsonl(out/'predictions.jsonl',{'event':event,'filename':name,'boxes':boxes})
                    if i==1 or i%100==0 or i==(args.max_images or len(samples)):print('WIDER',i,'/',args.max_images or len(samples),flush=True)
            records=[json.loads(line) for line in (out/'predictions.jsonl').read_text().splitlines()] if (out/'predictions.jsonl').exists() else [];assert len({(r['event'],r['filename']) for r in records})==len(records)
            status.update(state='partial_complete',smoke_passed=True,completed_images=len(records),total_images=3226)
            if args.max_images==0:
                assert {(r['event'],r['filename']) for r in records}==set(samples)
                predictions=out/'wider_predictions'
                for r in records:
                    folder=predictions/r['event'];folder.mkdir(parents=True,exist_ok=True)
                    (folder/(r['filename']+'.txt')).write_text(r['filename']+'.jpg\n'+str(len(r['boxes']))+'\n'+''.join(' '.join(f'{v:.8f}' for v in b)+'\n' for b in r['boxes']))
                sys.path.insert(0,str(EVAL));import evaluation
                aps=evaluation.evaluation(str(predictions),str(gt_dir));assert len(aps)==3 and np.isfinite(aps).all()
                score={name:float(ap) for name,ap in zip(['easy_AP','medium_AP','hard_AP'],aps)}
                score.update(unit='fraction_0_to_1',evaluation_images=3226,protocol_fingerprint=fingerprint,reference_check='same benchmark; author inference settings incompletely specified')
                core.atomic_json(out/'metrics.json',score);status.update(state='complete',metrics=score);print(score,flush=True)
        except BaseException as exc:status.update(state='failed',error=repr(exc));raise
        finally:status['elapsed_seconds']=time.time()-status['started_at'];core.atomic_json(path,status)

if __name__=='__main__':main()
