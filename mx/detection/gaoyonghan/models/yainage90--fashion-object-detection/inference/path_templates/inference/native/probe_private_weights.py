"""Verify private-data model binaries on ordinary COCO images; never report accuracy without gold."""
import io
import json
from pathlib import Path
import zipfile
from PIL import Image
from ultralytics import YOLO
from run_model import core,file_hash

ROOT=Path('__MODEL_ROOT__/weights')
OUT=Path('__MODEL_ROOT__/resources/runtime/assets/workspace/results/remaining25_private_probe')
with zipfile.ZipFile('__DATASET_ROOT__/COCO2017/val2017.zip') as z:
    names=sorted(n for n in z.namelist() if n.endswith('.jpg'))[:10]
    images=[Image.open(io.BytesIO(z.read(n))).convert('RGB') for n in names]
for sequence,repo in [(8,'Anzhc/Anzhcs_YOLOs'),(31,'erax-ai/EraX-NSFW-V1.0')]:
    rows=[]
    for weight in sorted((ROOT/repo).glob('*.pt')):
        row={'weight':weight.name,'sha256':file_hash(weight),'state':'failed'}
        try:
            model=YOLO(str(weight));row.update(task=model.task,labels=model.names)
            size=224 if model.task=='classify' else (1024 if '1024' in weight.name or 'seg-hd' in weight.name else 640)
            counts=[]
            for image in images:
                result=model.predict(image,imgsz=size,device='cuda:0',conf=.25,iou=.7,verbose=False)[0]
                counts.append({'boxes':len(result.boxes) if result.boxes is not None else None,
                               'mask_instances':len(result.masks) if result.masks is not None else 0,
                               'classification_probabilities':result.probs.data.cpu().tolist() if result.probs is not None else None})
            row.update(state='smoke_passed',input_size=size,completed_images=10,prediction_counts=counts)
        except Exception as exc:row['error']=repr(exc)
        rows.append(row);print(sequence,weight.name,row['state'],flush=True)
        core.atomic_json(OUT/f'{sequence}.json',{'sequence':sequence,'model_id':repo,'state':'blocked','all_model_tasks_complete':False,'probe_images':names,
          'purpose':'Load and inference diagnostics only; ordinary COCO images have no task-specific gold for these models. No AP/accuracy computed.',
          'weights':rows,'script_sha256':file_hash(Path(__file__)),'runtime_versions':core.runtime_versions()})
