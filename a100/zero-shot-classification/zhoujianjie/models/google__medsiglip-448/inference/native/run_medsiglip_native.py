from model_paths import legacy as _legacy, legacy_str as _legacy_str, relative as _relative, revision as _revision, weight as _weight, result as _result
import sys,runpy,json,time
from pathlib import Path
from datetime import datetime
r=_legacy('/data/zhoujianjie');mid='google/medsiglip-448'
sys.argv=['bootstrap','--check-imports']
try:runpy.run_path(str(_legacy('/data/zhoujianjie/environment_setup/run_model.py')),run_name='__main__')
except SystemExit as e:
 if e.code:raise
sys.path.insert(0,str(_legacy('/data/zhoujianjie/environment_setup')))
import medsiglip_native_a100 as n
import torch,numpy as np
from collections import OrderedDict
frozen=json.loads((_legacy('/data/zhoujianjie/environment_setup/medsiglip_4090_frozen.json')).read_text());selection=frozen['selection']
assert frozen['prompt_registry_sha256']==n.registry_sha256()
receipt=json.loads((n.MODEL_ROOT/'download_receipt.json').read_text());assert receipt['revision']==n.MODEL_REVISION
verified=n.inspect_h5('test')
evidence=json.loads((_legacy('/data/zhoujianjie/environment_setup/original_metadata/medsiglip_pcam_engineering_preflight.json')).read_text())
for k in ['x_sha256','y_sha256']:assert verified[k]==evidence['dataset']['files']['test'][k],k
out=_result(mid)/('native_'+datetime.now().strftime('%Y%m%dT%H%M%S'));out.mkdir(parents=True)
(out/'config.json').write_text(json.dumps({'model':mid,'selection_frozen_from_4090':selection,'dataset':verified,'batch_size':16,'original_source':'evaluate_medsiglip_pcam_engineering.py'},indent=2))
np.random.seed(n.SEED);torch.manual_seed(n.SEED);torch.cuda.manual_seed_all(n.SEED);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
print('MAPPING',mid,'PCam test32768; frozen4090 prompt/preprocessing/threshold; FP16 original adapter; no validation retuning',flush=True)
a=n.MedSigLipAdapter();classifiers=n.prompt_classifiers(a);name=selection['selected_prompt'];prep=selection['selected_preprocessing'];threshold=selection['selected_threshold_from_validation_youden_j']
orig=a.image_features;forward=[0.0]
def timed(*args,**kwargs):
 torch.cuda.synchronize();tick=time.perf_counter();result=orig(*args,**kwargs);torch.cuda.synchronize();forward[0]+=time.perf_counter()-tick;return result
a.image_features=timed
targets,scores,elapsed=n.score_split(a,'test',prep,OrderedDict([(name,classifiers[name])]),16)
score=scores[name];np.savez(out/'predictions.npz',targets=targets,scores=score,predictions=(score>=threshold).astype(np.int64))
d={'status':'completed','model_id':mid,'num_samples':len(targets),'selection':selection,'test':{**n.auc_with_ci(targets,score),'validation_frozen_threshold':n.threshold_metrics(targets,score,threshold)},'inference_seconds':elapsed,'synchronized_image_pipeline_seconds':forward[0],'synchronized_image_pipeline_ms_per_sample':forward[0]*1000/len(targets),'strict_official_acceptance':False,'purpose':'A100 same4090 frozen engineering baseline'}
(out/'metrics.json').write_text(json.dumps(d,indent=2))
