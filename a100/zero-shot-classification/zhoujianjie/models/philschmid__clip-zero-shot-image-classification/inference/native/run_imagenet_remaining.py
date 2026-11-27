from model_paths import legacy as _legacy, legacy_str as _legacy_str, relative as _relative, revision as _revision, weight as _weight, result as _result
import sys,runpy,json,argparse,time
from pathlib import Path
from datetime import datetime
r=_legacy('/data/zhoujianjie');mid=sys.argv[1];assert mid in ['Xenova/clip-vit-base-patch32','philschmid/clip-zero-shot-image-classification']
sys.argv=['bootstrap','--check-imports']
try:runpy.run_path(str(_legacy('/data/zhoujianjie/environment_setup/run_model.py')),run_name='__main__')
except SystemExit as e:
 if e.code:raise
sys.path.insert(0,str(_legacy('/data/zhoujianjie/environment_setup/native_sources/evaluation_audit')));sys.path.insert(0,str(_legacy('/data/zhoujianjie/environment_setup')))
import imagenet_native_a100 as n
import evaluate_b01 as b
import torch
sys.modules['evaluate_imagenet1k_official']=n
b.model_dir=lambda m:_weight(m)
out=_result(mid)/('native_'+datetime.now().strftime('%Y%m%dT%H%M%S'));out.mkdir(parents=True)
(out/'config.json').write_text(json.dumps({'model':mid,'protocol':'original ImageNet validation50000 CLIP80','batch_size':128 if mid.startswith('Xenova') else 32},indent=2))
if mid.startswith('Xenova'):
 import xenova_native_a100 as x
 x.OUT=out/'metrics.json';sys.argv=['xenova'];code=x.main();raise SystemExit(code)
else:
 import copy,hashlib
 proof=json.loads((_legacy('/data/zhoujianjie/environment_setup/original_metadata/philschmid_openai_clip_equivalence.json')).read_text())
 digest=hashlib.sha256()
 with (b.model_dir(mid)/'pytorch_model.bin').open('rb') as f:
  for chunk in iter(lambda:f.read(8*1024*1024),b''):digest.update(chunk)
 assert digest.hexdigest()==proof['fork_pytorch_bin_sha256']
 n.claim_spec.MODEL_SPECS[mid]=copy.deepcopy(n.claim_spec.MODEL_SPECS['openai/clip-vit-base-patch32'])
 args=argparse.Namespace(worker_model_id=mid,worker_backend='Transformers AutoModel/AutoProcessor',worker_output=out/'metrics.json',batch_size=32,text_batch_size=512)
 code=n.worker(args);raise SystemExit(code)
