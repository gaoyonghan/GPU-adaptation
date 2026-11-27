from model_paths import legacy as _legacy, legacy_str as _legacy_str, relative as _relative, revision as _revision, weight as _weight, result as _result
import sys,runpy
import os
original_cwd=os.getcwd()
args=sys.argv[1:];sys.argv=['bootstrap','--check-imports']
try:runpy.run_path(_legacy_str('/data/zhoujianjie/environment_setup/run_model_fgclip.py'),run_name='__main__')
except SystemExit as e:
 if e.code:raise
import os
sys.path.insert(0,_legacy_str('/data/zhoujianjie/environment_setup/runtime_overlays/openclip'))
os.chdir(original_cwd)
sys.path.insert(0,original_cwd)
os.environ["PYTHONPATH"]=os.pathsep.join([original_cwd,original_cwd+"/src",str(__import__("pathlib").Path(_legacy_str('/data/zhoujianjie/.venvs/zeroshot-main/lib/python3.10/site-packages'))),os.environ.get("PYTHONPATH","")])
if args[:2]==['-m','clip_benchmark.cli']:
 import open_clip.tokenizer as ot
 old_init=ot.HFTokenizer.__init__
 def local_hf_init(self,tokenizer_name,*a,**kw):
  if tokenizer_name=='timm/ViT-B-16-SigLIP':
   tokenizer_name=_legacy_str('/data/zhoujianjie/model_inference_for_GPU_adaptation/classification/zhoujianjie/models/timm__ViT-B-16-SigLIP/weights')
  return old_init(self,tokenizer_name,*a,**kw)
 ot.HFTokenizer.__init__=local_hf_init
 import torch
 old_load=torch.load
 trusted=_legacy_str('/data/zhoujianjie/model_inference_for_GPU_adaptation/classification/zhoujianjie/models/openai__clip-vit-large-patch14/weights/ViT-L-14.pt')
 def compatible_load(f,*a,**kw):
  if str(f)==trusted:kw.setdefault('weights_only',False)
  return old_load(f,*a,**kw)
 torch.load=compatible_load

if args[:2]==['-m','clip_benchmark.cli']:
 import torch,time,json
 from pathlib import Path
 import clip_benchmark.metrics.zeroshot_classification as metrics
 original=metrics.run_classification
 output=Path(args[args.index('--output')+1])
 def capture(model,*a,**kw):
  classifier,dataloader,device=a[:3]
  first=next(iter(dataloader));small=[(first[0][:2],first[1][:2])]
  import random,numpy as np
  rng=(random.getstate(),np.random.get_state(),torch.get_rng_state(),torch.cuda.get_rng_state_all())
  try:
   sp,st=original(model,classifier,small,device,**kw)
   assert len(st)>0 and torch.isfinite(sp).all(), 'smoke produced nonfinite or empty prediction'
   torch.save({'logits':sp,'targets':st,'purpose':'smoke only, excluded from full metrics'},output.with_suffix('.smoke.pt'))
   print('SMOKE_PASSED',len(st),str(output),flush=True)
  finally:
   random.setstate(rng[0]);np.random.set_state(rng[1]);torch.set_rng_state(rng[2]);torch.cuda.set_rng_state_all(rng[3])
  seconds=[0.0];old=model.encode_image
  def timed(*aa,**kk):
   torch.cuda.synchronize();tick=time.perf_counter();result=old(*aa,**kk);torch.cuda.synchronize();seconds[0]+=time.perf_counter()-tick;return result
  model.encode_image=timed
  try:pred,true=original(model,*a,**kw)
  finally:model.encode_image=old
  torch.save({'logits':pred,'targets':true},output.with_suffix('.predictions.pt'))
  output.with_suffix('.timing.json').write_text(json.dumps({'num_samples':len(true),'image_forward_total_seconds':seconds[0],'image_forward_ms_per_sample':seconds[0]*1000/len(true)}))
  return pred,true
 metrics.run_classification=capture

if args[:2]==['-m','torch.distributed.run']:
 args=args[:2]+[x for x in args[2:] if x!='-m'];i=args.index('--nproc_per_node')+2
 args.insert(i,_legacy_str('/data/zhoujianjie/environment_setup/native_fgclip_python.py'));args.insert(i+1,'-m')
if args[:2]==['-m','src.evaluation.classification']:
 import src.open_clip as oc,torch,json
 from pathlib import Path
 original_factory=oc.create_model_and_transforms
 def factory_smoke(*a,**kw):
  result=original_factory(*a,**kw);model=result[0];model.eval()
  from src.evaluation.data import DatasetFromFile
  root=args[args.index('--data_root')+1]
  ds=DatasetFromFile(root,transform=result[2]);image,target=ds[0]
  with torch.inference_mode(),torch.autocast('cuda'):
   features=model.encode_image(image.unsqueeze(0).to(next(model.parameters()).device))
  assert torch.isfinite(features).all(), 'smoke produced nonfinite features'
  out=Path(args[args.index('--logs')+1]);out.mkdir(parents=True,exist_ok=True)
  torch.save({'features':features.cpu(),'target':int(target),'purpose':'smoke only'},out/'smoke.pt')
  print('BIOCLIP_SMOKE_PASSED',root,flush=True)
  return result
 oc.create_model_and_transforms=factory_smoke

if args[0]=='-m':sys.argv=args[1:];runpy.run_module(args[1],run_name='__main__',alter_sys=True)
else:sys.argv=args;runpy.run_path(args[0],run_name='__main__')
