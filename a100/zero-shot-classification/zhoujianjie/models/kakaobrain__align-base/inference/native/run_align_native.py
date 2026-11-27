from model_paths import legacy as _legacy, legacy_str as _legacy_str, relative as _relative, revision as _revision, weight as _weight, result as _result
import sys,runpy,json,os
from pathlib import Path
from datetime import datetime
r=_legacy('/data/zhoujianjie');small='--small' in sys.argv;mid='kakaobrain/align-base'
sys.argv=['bootstrap','--check-imports']
try:runpy.run_path(str(_legacy('/data/zhoujianjie/environment_setup/run_model.py')),run_name='__main__')
except SystemExit as e:
 if e.code:raise
sys.path.insert(0,str(_legacy('/data/zhoujianjie/environment_setup')))
import align_native_a100 as n
out=_legacy('/data/zhoujianjie/model_inference_for_GPU_adaptation/classification/zhoujianjie/models/kakaobrain__align-base/result')/ (('small_validation_' if small else 'native_')+datetime.now().strftime('%Y%m%dT%H%M%S'))
out.mkdir(parents=True,exist_ok=False);n.OUTPUT=out/'metrics.json';n.PREFLIGHT=out/'preflight.json'
assert n.KARPATHY_JSON.is_file(),str(n.KARPATHY_JSON)
records=n.load_records();n.run_preflight(records)
(out/'config.json').write_text(json.dumps({'model_id':mid,'small_validation':small,'weights':str(n.MODEL_DIR),'dataset':str(n.DATASET_DIR),'image_batch_size':16,'text_batch_size':256,'precision':'FP32','original_script':'evaluate_kakao_align_flickr30k_official.py'},indent=2))
result=n.evaluate(records[:16] if small else records,16,256)
result['small_validation']=small
n.OUTPUT.write_text(json.dumps(result,indent=2))
raise SystemExit(0 if result.get('status')=='passed' else 1)
