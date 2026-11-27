from model_paths import legacy as _legacy, legacy_str as _legacy_str, relative as _relative, revision as _revision, weight as _weight, result as _result
import sys,runpy,json,os
from pathlib import Path
from datetime import datetime
os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
r=_legacy('/data/zhoujianjie');mid=sys.argv[1];small=False
sys.argv=['bootstrap','--check-imports']
try:runpy.run_path(str(_legacy('/data/zhoujianjie/environment_setup/run_model.py')),run_name='__main__')
except SystemExit as e:
 if e.code:raise
sys.path.insert(0,str(_legacy('/data/zhoujianjie/environment_setup')))
import openclip_native_a100 as n
checkpoint=n.MODELS[mid]['checkpoint'].name
n.MODELS[mid]['checkpoint']=_weight(mid)/checkpoint
out=_result(mid)/ (('preflight_' if small else 'native_')+datetime.now().strftime('%Y%m%dT%H%M%S'))
if not small:
 candidates=[]
 for prior in out.parent.glob('native_*'):
  config=prior/'config.json'
  if (prior/'completion.json').exists() or not config.exists():continue
  try:d=json.loads(config.read_text())
  except ValueError:continue
  if d.get('model')==mid and d.get('suites')==['batch3','batch4'] and d.get('preflight_only') is False:candidates.append(prior)
 if candidates:out=sorted(candidates)[-1]
out.mkdir(parents=True,exist_ok=True);n.RESULT_DIR=out
(out/'config.json').write_text(json.dumps({'model':mid,'original_script':'evaluate_openclip_new_suite_official.py','suites':['batch3','batch4'],'preflight_only':small},indent=2))
for suite in (['batch3'] if int(os.environ.get('ZERO_SHOT_SMOKE_LIMIT','0')) else ['batch3','batch4']):
 sys.argv=['openclip','--model-key',mid,'--suite',suite,'--num-workers','0','--skip-existing']+(['--preflight-only'] if small else [])
 code=n.main()
 if code:raise SystemExit(code)
if not small:(out/'completion.json').write_text(json.dumps({'status':'completed','model_id':mid,'suites':['batch3','batch4']}))
