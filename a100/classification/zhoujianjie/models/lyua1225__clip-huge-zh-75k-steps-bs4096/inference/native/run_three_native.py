from model_paths import legacy as _legacy, legacy_str as _legacy_str, relative as _relative, revision as _revision, weight as _weight, result as _result
import sys,runpy,json,time
from pathlib import Path
from datetime import datetime
r=_legacy('/data/zhoujianjie');mid=sys.argv[1];assert mid in ['geolocal/StreetCLIP','lyua1225/clip-huge-zh-75k-steps-bs4096','Bingsu/clip-vit-large-patch14-ko']
sys.argv=['bootstrap','--check-imports']
try:runpy.run_path(str(_legacy('/data/zhoujianjie/environment_setup/run_model.py')),run_name='__main__')
except SystemExit as e:
 if e.code:raise
sys.path.insert(0,str(_legacy('/data/zhoujianjie/environment_setup/native_sources/evaluation_audit')));sys.path.insert(0,str(_legacy('/data/zhoujianjie/environment_setup')))
import three_native_a100 as n
import evaluate_xtd10_clip as x
import torch
raw=_legacy('/data/zhoujianjie/model_inference_for_GPU_adaptation/dataset/raw')
n.b01.model_dir=lambda m:_weight(m)
n.COUNTRY_PARQUET=_legacy('/data/zhoujianjie/model_inference_for_GPU_adaptation/dataset/raw/mmeb_country_1/test-00000-of-00001.parquet');n.COUNTRY_IMAGES=_legacy('/data/zhoujianjie/model_inference_for_GPU_adaptation/dataset/raw/mmeb_country_2');n.XTD_ROOT=_legacy('/data/zhoujianjie/model_inference_for_GPU_adaptation/dataset/raw/xtd10');x.DATA_ROOT=n.XTD_ROOT;x.CACHE_ROOT=_legacy('/data/zhoujianjie/environment_setup/cache/xtd10_frozen')
out=_result(mid)/('native_'+datetime.now().strftime('%Y%m%dT%H%M%S'));out.mkdir(parents=True);n.PREDICTION_ROOT=out
spec=n.MODEL_SPECS[mid];receipt=json.loads((n.b01.model_dir(mid)/'download_receipt.json').read_text())
(out/'config.json').write_text(json.dumps({'model':mid,'spec':spec,'receipt':receipt,'source':'evaluate_remaining_11_proxy_diagnostics.py','batch_size':16,'text_batch_size':128},indent=2))
if spec['task']=='xtd10':
 for lang in spec['languages']:
  q,c,t=x.records(lang);assert len(q)==len(c)==len(t)==999
else:
 import pyarrow.parquet as pq
 assert pq.ParquetFile(n.COUNTRY_PARQUET).metadata.num_rows==1000
 assert len(list(n.COUNTRY_IMAGES.iterdir()))==1000
print('MAPPING',mid,spec,'original4090 route; no controls rerun',flush=True)
torch.manual_seed(n.SEED);a=n.load_adapter(mid,'Transformers AutoModel/AutoProcessor');timing={}
for name in ['image_features','text_features']:
 old=getattr(a,name)
 def wrap(*args,_old=old,_name=name,**kw):
  torch.cuda.synchronize();tick=time.perf_counter();v=_old(*args,**kw);torch.cuda.synchronize();timing[_name]=timing.get(_name,0.0)+time.perf_counter()-tick;return v
 setattr(a,name,wrap)
d=n.evaluate_country211(a,16) if spec['task']=='country211' else n.evaluate_xtd10(a,spec['languages'],16,128)
(out/'metrics.json').write_text(json.dumps({'status':'completed','model_id':mid,'model_revision':receipt.get('revision'),'result':d,'synchronized_feature_pipeline_seconds':timing,'purpose':'A100 same4090 original diagnostic baseline','strict_official_acceptance':False},indent=2))
