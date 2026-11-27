from model_paths import legacy as _legacy, legacy_str as _legacy_str, relative as _relative, revision as _revision, weight as _weight, result as _result
import sys,os,runpy,importlib.util
sys.path.insert(0,_legacy_str('/data/zhoujianjie/.venvs/zeroshot-main/lib/python3.10/site-packages'))
sys.path.insert(0,_legacy_str('/data/zhoujianjie/.venvs/zeroshot-mme5/lib/python3.10/site-packages'))
sys.path.insert(0,os.getcwd());sys.path.insert(0,os.getcwd()+'/src')
f=importlib.util.find_spec
importlib.util.find_spec=lambda name,package=None:None if name.split('.')[0]=='flash_attn' else f(name,package)
a=sys.argv[1:];sys.argv=a
import torch,time,json
from pathlib import Path
from src.model import MMEBModel
output=Path(a[a.index('--encode_output_path')+1]);output.mkdir(parents=True,exist_ok=True)
original=MMEBModel.forward
seen=[False]
def measured(self,*aa,**kk):
 # Verify the first real batch with the original method before consuming it in full evaluation.
 if not seen[0]:
  with torch.no_grad():s=original(self,*aa,**kk)
  saved={k:v.detach().cpu() for k,v in s.items() if torch.is_tensor(v)}
  assert saved and all(torch.isfinite(v).all() for v in saved.values()), 'invalid mmE5 smoke output'
  torch.save(saved,output/'smoke.pt');seen[0]=True;print('MME5_SMOKE_PASSED',flush=True)
 torch.cuda.synchronize();tick=time.perf_counter();result=original(self,*aa,**kk);torch.cuda.synchronize()
 sizes={k:len(v) for k,v in result.items() if torch.is_tensor(v)}
 with (output/'timing.jsonl').open('a') as f:f.write(json.dumps({'forward_seconds':time.perf_counter()-tick,'representation_counts':sizes})+'\n')
 return result
MMEBModel.forward=measured
runpy.run_path(a[0],run_name='__main__')
