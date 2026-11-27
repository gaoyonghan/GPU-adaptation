from model_paths import legacy as _legacy, legacy_str as _legacy_str, relative as _relative, revision as _revision, weight as _weight, result as _result
import sys,runpy,os,json
from pathlib import Path
r=_legacy('/data/zhoujianjie');args=sys.argv[1:];mid=args[0];small='--small' in args
sys.argv=['bootstrap','--check-imports']
try:runpy.run_path(str(_legacy('/data/zhoujianjie/environment_setup/run_model.py')),run_name='__main__')
except SystemExit as e:
 if e.code:raise
sys.path.insert(0,str(_legacy('/data/zhoujianjie/environment_setup')))
import chinese_native_a100 as n
if small:os.environ['NATIVE_SMALL']='1'
for dataset in (list(n.DATASETS)[:1] if small else n.DATASETS):
 print('MAPPING',mid,dataset,'original cn_clip FP32 / original templates / context_length52',flush=True)
 n.preflight(mid,dataset)
for dataset in (list(n.DATASETS)[:1] if small else n.DATASETS):
 result_root=_result(mid)
 complete=[]
 if not small:
  for f in result_root.glob('native_*/metrics.json'):
   d=json.loads(f.read_text())
   if d.get('model_id')==mid and d.get('dataset')==dataset and d.get('status')=='completed' and not d.get('small_validation') and d.get('dataset_count')==n.DATASETS[dataset]:complete.append(str(f))
 if complete:print('SKIP_COMPLETED_DATASET',mid,dataset,complete,flush=True);continue
 output=n.evaluate(mid,dataset,64)
 (output.parent/'config.json').write_text(json.dumps({'model_id':mid,'dataset':dataset,'batch_size':64,'small_validation':small,'original_script':'evaluate_chinese_clip_original_official.py','native_adapter':str(n.__file__),'weights':str(_weight(mid)),'precision':'fp32','context_length':52},indent=2))

