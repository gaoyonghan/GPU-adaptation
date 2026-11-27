from model_paths import legacy as _legacy, legacy_str as _legacy_str, relative as _relative, revision as _revision, weight as _weight, result as _result
import sys,runpy,json,hashlib
from pathlib import Path
saved_args=sys.argv[:]
root=_legacy('/data/zhoujianjie');sys.argv=['bootstrap','--check-imports']
try:runpy.run_path(str(_legacy('/data/zhoujianjie/environment_setup/run_model.py')),run_name='__main__')
except SystemExit as e:
 if e.code:raise
sys.argv=saved_args
sys.path.insert(0,str(_legacy('/data/zhoujianjie/environment_setup')))
import evaluate_clip_rsicd_a100 as native
model=_legacy('/data/zhoujianjie/model_inference_for_GPU_adaptation/classification/zhoujianjie/models/flax-community__clip-rsicd-v2')
receipt=json.loads((_legacy('/data/zhoujianjie/model_inference_for_GPU_adaptation/classification/zhoujianjie/models/flax-community__clip-rsicd-v2/weights/download_receipt.json')).read_text()); config=json.loads((_legacy('/data/zhoujianjie/model_inference_for_GPU_adaptation/classification/zhoujianjie/models/flax-community__clip-rsicd-v2/model_config.json')).read_text())
assert (receipt.get('revision') or receipt.get('resolved_revision'))==config['revision']
manifest=json.loads(native.MANIFEST.read_text())
missing=[r['filename'] for r in manifest['records'] if not (native.IMAGES/r['filename']).is_file()]
assert not missing,missing[:3]
print('MAPPING',config['model_id'],config['revision'],'RSICD',len(manifest['records']),'Aerial photograph of {category}; FP32; Top1/3/5/10; paired baseline',flush=True)
if '--preflight-only' in sys.argv:raise SystemExit(0)
# Small validation runs the original dataset and model path on its first eight records.
# It never changes the full manifest on disk.
if '--small' in sys.argv:
 original=native.json.loads
 def loads(s,*a,**kw):
  d=original(s,*a,**kw)
  if isinstance(d,dict) and 'records' in d and 'classes' in d:d={**d,'records':d['records'][:int(__import__('os').environ['ZERO_SHOT_SMOKE_LIMIT'])]}
  return d
 native.json.loads=loads
sys.argv=['evaluate_clip_rsicd_a100.py','--model-id','flax-community/clip-rsicd-v2','--workers','0','--batch-size','32']
raise SystemExit(native.main())
