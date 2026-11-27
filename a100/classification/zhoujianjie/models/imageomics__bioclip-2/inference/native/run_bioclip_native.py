from model_paths import legacy as _legacy, legacy_str as _legacy_str, relative as _relative, revision as _revision, weight as _weight, result as _result
import sys,runpy,json
from pathlib import Path
from datetime import datetime
r=_legacy('/data/zhoujianjie');mid=sys.argv[1];small=False
sys.argv=['bootstrap','--check-imports']
try:runpy.run_path(str(_legacy('/data/zhoujianjie/environment_setup/run_model.py')),run_name='__main__')
except SystemExit as e:
 if e.code:raise
sys.path.insert(0,str(_legacy('/data/zhoujianjie/environment_setup')))
import bioclip_native_a100 as n
n.ROOT=r
n.SPEC_PATH=_legacy('/data/zhoujianjie/environment_setup/bioclip_spec_fixed.json')
n.MANIFEST_PATH=_legacy('/data/zhoujianjie/environment_setup/original_metadata/bioclip_meta_album_eval_manifest.json')
n.EVALUATOR_ROOT=_legacy('/data/zhoujianjie/environment_setup/native_sources/evaluation_audit/reference_code/bioclip-2')
out=_result(mid)/ (('preflight_' if small else 'native_')+datetime.now().strftime('%Y%m%dT%H%M%S'))
out.mkdir(parents=True);n.RESULT_ROOT=out
(out/'config.json').write_text(json.dumps({'model':mid,'original_script':'evaluate_bioclip_meta_album_official.py','preflight_only':small,'batch_size':256,'workers':0},indent=2))
sys.argv=['bioclip','--model',mid,'--workers','0']+([] if small else ['--run'])
n.main()
if not small:(out/'completion.json').write_text(json.dumps({'status':'completed','model':mid}))
