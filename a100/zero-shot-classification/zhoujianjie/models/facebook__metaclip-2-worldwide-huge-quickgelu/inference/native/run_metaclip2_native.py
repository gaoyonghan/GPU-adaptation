from model_paths import legacy as _legacy, legacy_str as _legacy_str, relative as _relative, revision as _revision, weight as _weight, result as _result
import sys,runpy,json
from pathlib import Path
from datetime import datetime
r=_legacy('/data/zhoujianjie');mid='facebook/metaclip-2-worldwide-huge-quickgelu'
sys.path.insert(0,str(_legacy('/data/zhoujianjie/.venvs/zeroshot-main/lib/python3.10/site-packages')))
sys.argv=['bootstrap','--check-imports']
try:runpy.run_path(str(_legacy('/data/zhoujianjie/environment_setup/run_model_metaclip2.py')),run_name='__main__')
except SystemExit as e:
 if e.code:raise
sys.path.insert(0,str(_legacy('/data/zhoujianjie/environment_setup')))
import metaclip2_native_a100 as n
out=_result(mid)/('native_'+datetime.now().strftime('%Y%m%dT%H%M%S'))
out.mkdir(parents=True);n.OUT=out/'metrics.json';n.PROGRESS=out/'progress.json'
(out/'config.json').write_text(json.dumps({'model':mid,'source':'evaluate_metaclip2_cvqa_official.py','batch_size':16,'precision':'bfloat16','attention':'sdpa'},indent=2))
sys.argv=['metaclip2','--run'];n.main()
