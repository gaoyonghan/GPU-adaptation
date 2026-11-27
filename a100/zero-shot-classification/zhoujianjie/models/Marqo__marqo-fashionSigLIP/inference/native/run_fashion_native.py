from model_paths import legacy as _legacy, legacy_str as _legacy_str, relative as _relative, revision as _revision, weight as _weight, result as _result
import sys,runpy,json,os
from pathlib import Path
from datetime import datetime
r=_legacy('/data/zhoujianjie');mid=sys.argv[1];small='--small' in sys.argv

sys.argv=['bootstrap','--check-imports']
try:runpy.run_path(str(_legacy('/data/zhoujianjie/environment_setup/run_model.py')),run_name='__main__')
except SystemExit as e:
 if e.code:raise
sys.path.insert(0,str(_legacy('/data/zhoujianjie/environment_setup')))
import fashion_native_a100 as n
for m,v in n.MODELS.items():v['path']=_weight(m)
n.common.MODELS=n.MODELS
out=_result(mid)/('native_'+datetime.now().strftime('%Y%m%dT%H%M%S'))
out.mkdir(parents=True);os.environ['NATIVE_OUTPUT']=str(out)
(out/'config.json').write_text(json.dumps({'model':mid,'original_script':'evaluate_marqo_deepfashion_official.py','datasets':list(n.DATASETS)},indent=2))
for key,d in n.DATASETS.items():
 assert d['path'].is_dir(),str(d['path'])
 for task in d['tasks']:
  suffix='image' if task=='text' else 'image+text';assert (d['gt']/('ground_truth_'+task+'-'+suffix+'.json')).is_file()
 print('MAPPING',mid,key,d,flush=True)
if small:
 import torch
 model,preprocess,tokenizer=n.common.load_local_model(mid)
 n.common.DATASET=n.DATASETS['inshop']['path']
 features,texts,ids,elapsed=n.common.encode_documents(model,preprocess,tokenizer,8,8,'smoke')
 torch.save({'features':features,'document_ids':ids},__import__('pathlib').Path(os.environ['ZERO_SHOT_RUN_DIR'])/'document_encoding_smoke.pt')
 assert len(ids)==8
 print('SMOKE_OK',mid,len(ids),flush=True)
 raise SystemExit(0)
results=[n.evaluate(key,mid) for key in n.DATASETS]
(out/'metrics.json').write_text(json.dumps({'status':'completed' if all(x['status']=='passed' for x in results) else 'failed','model_id':mid,'results':results},indent=2))
raise SystemExit(0 if all(x['status']=='passed' for x in results) else 1)
