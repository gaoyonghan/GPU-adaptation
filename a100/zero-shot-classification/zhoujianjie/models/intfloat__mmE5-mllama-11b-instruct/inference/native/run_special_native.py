from model_paths import legacy as _legacy, legacy_str as _legacy_str, relative as _relative, revision as _revision, weight as _weight, result as _result
import sys,runpy,json,os,tarfile
from pathlib import Path
from datetime import datetime
r=_legacy('/data/zhoujianjie');mid=sys.argv[1];mapping={'qihoo360/fg-clip2-base':('fgclip','FG-CLIP','fgclip2_boxclass_cn_official_spec.json','fgclip_box'),'q-future/one-align':('onealign','Q-Align','onealign_unseen_official_spec.json','onealign'),'intfloat/mmE5-mllama-11b-instruct':('mme5','mmE5_official_4818cdf',None,None)};key,repo,specname,raw=mapping[mid]
if key=='onealign':
 sys.path.insert(0,str(_legacy('/data/zhoujianjie/.venvs/zeroshot-main/lib/python3.10/site-packages')));sys.path.insert(0,str(_legacy('/data/zhoujianjie/.venvs/zeroshot-onealign/lib/python3.10/site-packages')))
 os.environ.update(USE_TF='0',USE_FLAX='0',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
 import importlib.util
 _find=importlib.util.find_spec
 importlib.util.find_spec=lambda name,package=None:None if name.split('.')[0]=='flash_attn' else _find(name,package)
elif key=='mme5':
 sys.path.insert(0,str(_legacy('/data/zhoujianjie/.venvs/zeroshot-main/lib/python3.10/site-packages')));sys.path.insert(0,str(_legacy('/data/zhoujianjie/.venvs/zeroshot-mme5/lib/python3.10/site-packages')))
 os.environ.update(USE_TF='0',USE_FLAX='0',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
 import importlib.util
 _find=importlib.util.find_spec
 importlib.util.find_spec=lambda name,package=None:None if name.split('.')[0]=='flash_attn' else _find(name,package)
else:
 sys.argv=['bootstrap','--check-imports']
 try:runpy.run_path(str(_legacy('/data/zhoujianjie/environment_setup/run_model.py')),run_name='__main__')
 except SystemExit as e:
  if e.code:raise
sys.path.insert(0,str(_legacy('/data/zhoujianjie/environment_setup/native_sources/evaluation_audit')));sys.path.insert(0,str(_legacy('/data/zhoujianjie/environment_setup')))
n=__import__(key+'_native_a100')
out=_result(mid)/('native_'+datetime.now().strftime('%Y%m%dT%H%M%S'));out.mkdir(parents=True);os.environ['NATIVE_OUTPUT']=str(out)
if specname:
 spec=json.loads((_legacy('/data/zhoujianjie/environment_setup/original_metadata')/specname).read_text());spec['model']['local_dir']=str(_weight(mid));spec['dataset']['local_root']=str(_legacy('/data/zhoujianjie/model_inference_for_GPU_adaptation/dataset/raw')/raw);spec['dataset']['manifest']=str(_legacy('/data/zhoujianjie/environment_setup/original_metadata')/Path(spec['dataset']['manifest']).name);spec['evaluator']['local_dir']=str(_legacy('/data/zhoujianjie/environment_setup/native_sources/evaluation_audit/reference_code')/repo)
 n.SPEC_PATH=out/'spec.json';n.SPEC_PATH.write_text(json.dumps(spec,indent=2));n.RESULT_ROOT=out
else:n.OUT_DIR=out
(out/'config.json').write_text(json.dumps({'model':mid,'original_module':key,'output':str(out)},indent=2))
sys.argv=[key,'--run'] if key!='mme5' else [key]
n.main()
(out/'completion.json').write_text(json.dumps({'status':'completed','model_id':mid}))
