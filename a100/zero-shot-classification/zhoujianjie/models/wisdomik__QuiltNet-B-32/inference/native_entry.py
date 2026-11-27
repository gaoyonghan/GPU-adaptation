import capture_environment
import os
os.environ.update(USE_TF="0",USE_FLAX="0",HF_HUB_OFFLINE="1",TRANSFORMERS_OFFLINE="1")
import importlib.util
_original_find_spec = importlib.util.find_spec
_disabled_optional = {"flash_attn", "deepspeed", "transformer_engine"}
def _find_spec(name, package=None):
    if name.split(".")[0] in _disabled_optional:
        return None
    return _original_find_spec(name, package)
importlib.util.find_spec = _find_spec
import transformers.utils.import_utils as iu
import transformers.utils as tu
# Broken optional system CUDA extension; this run does not request flash_attention_2.
iu.is_flash_attn_2_available=lambda:False
tu.is_flash_attn_2_available=lambda:False

import os,sys,json,time,runpy
from pathlib import Path
MODEL_DIR=Path(__file__).resolve().parent.parent
base=MODEL_DIR.parents[1]
root=MODEL_DIR.parents[3]
from shared.handoff_runtime import configure_legacy
from shared.standard_dataset import StandardDataset
b01,_=configure_legacy()
data=StandardDataset(Path(os.environ['ZERO_SHOT_DATASET_ROOT'])/'standardized/pcam_test')
print(json.dumps({'model':'wisdomik/QuiltNet-B-32','dataset':data.validate(),'purpose':'A100_P800_paired_baseline','official_acceptance':False}),flush=True)
import torch
from sklearn.metrics import f1_score,precision_score,recall_score,matthews_corrcoef
adapter=b01.load_adapter('wisdomik/QuiltNet-B-32','OpenCLIP')
labels=['lymph node','lymph node containing metastatic tumor tissue']
templates=['a histopathology slide showing {}','histopathology image of {}','pathology tissue showing {}','presence of {} tissue on image']
classifier=torch.stack([torch.nn.functional.normalize(adapter.text_features([t.format(label) for t in templates]).mean(0),dim=0) for label in labels],dim=1)
from datetime import datetime
out=Path(os.environ['ZERO_SHOT_RUN_DIR'])
targets=[];preds=[];hit5=0;start=time.perf_counter()
delivery_smoke_limit=int(os.environ.get('ZERO_SHOT_SMOKE_LIMIT','0'))
with (out/'predictions.jsonl').open('w') as f:
 for rows,images in data.batches(64):
  if delivery_smoke_limit:
   rows,images=rows[:max(0,delivery_smoke_limit-len(targets))],images[:max(0,delivery_smoke_limit-len(targets))]
   if not rows:break
  features=adapter.image_features(images);rank=(features@classifier).topk(min(5,len(data.categories)),dim=1).indices.cpu().tolist()
  for row,ids in zip(rows,rank):
   y=row['label'];targets.append(y);preds.append(ids[0]);hit5+=y in ids
   f.write(json.dumps({'sample_id':row['sample_id'],'target':y,'prediction':ids[0],'top5_predictions':ids})+'\n')
  if delivery_smoke_limit and len(targets)>=delivery_smoke_limit:break
  print('PROGRESS',len(targets),data.metadata['sample_count'],flush=True)
assert len(targets)==(delivery_smoke_limit or data.metadata['sample_count'])
metrics={'top1_accuracy':sum(a==b for a,b in zip(targets,preds))/len(targets),'top5_accuracy':None,'weighted_f1':f1_score(targets,preds,average='weighted'),'weighted_precision':precision_score(targets,preds,average='weighted',zero_division=0),'weighted_recall':recall_score(targets,preds,average='weighted',zero_division=0),'mcc':matthews_corrcoef(targets,preds)}
result={'model':'wisdomik/QuiltNet-B-32','sample_count':len(targets),'metrics':metrics,'forward_seconds':adapter.forward_seconds,'forward_ms_per_sample':adapter.forward_seconds*1000/len(targets),'pipeline_seconds':time.perf_counter()-start,'batch_size':64,'precision':'fp16','purpose':'A100_P800_paired_baseline','official_acceptance_assessed':False,'gpu':torch.cuda.get_device_name(0)}
(out/'metrics.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)

from export_bundle import export_bundle
export_bundle(out,data.categories,[json.loads(line) for line in (out/"predictions.jsonl").read_text().splitlines()],'wisdomik/QuiltNet-B-32',data.metadata["source"])
