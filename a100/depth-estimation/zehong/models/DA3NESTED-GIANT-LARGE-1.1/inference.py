"""Native depth inference; independent task evaluate.py reads saved predictions."""
import argparse,json,sys,os
from pathlib import Path
from datetime import datetime
HERE=Path(__file__).resolve().parent
def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--weights',type=Path,default=HERE/'weights')
 p.add_argument('--pipeline-weights',type=Path,help='DepthCrafter SVD pipeline snapshot')
 p.add_argument('--dataset-root',type=Path,default=HERE.parents[3]/'dataset')
 p.add_argument('--dataset',choices=['KITTI','NYUv2','DIODE','Sintel'],default='NYUv2')
 p.add_argument('--output-root','--output-dir',type=Path,dest="output_dir",default=HERE/'result'/'run_001')
 p.add_argument('--num-workers','--workers',type=int,default=8)
 p.add_argument('--warmup-steps','--warmup',type=int,default=3)
 p.add_argument('--max-samples',type=int)
 p.add_argument('--num-inference-steps',type=int,default=5,help='DepthCrafter only')
 a=p.parse_args()
 # Asset paths are model-relative; absolute paths remain supported.
 for _name in ('weights', 'dataset_root', 'dataset', 'bundle', 'vision_weights', 'hf_cache', 'image'):
     _path = getattr(a, _name, None)
     if isinstance(_path, Path) and not _path.is_absolute():
         setattr(a, _name, (HERE / _path).resolve())
 # CLI paths are relative to this model directory, independent of cwd.
 if a.weights is not None and not a.weights.is_absolute(): a.weights=(HERE/a.weights).resolve()
 if a.dataset_root is not None and not a.dataset_root.is_absolute(): a.dataset_root=(HERE/a.dataset_root).resolve()
 if a.pipeline_weights is not None and not a.pipeline_weights.is_absolute(): a.pipeline_weights=(HERE/a.pipeline_weights).resolve()
 if a.output_dir is not None and not a.output_dir.is_absolute(): a.output_dir=(HERE/a.output_dir).resolve()
 if a.output_dir.exists(): p.error("Output directory already exists; choose a new --output-root")
 if a.output_dir.exists():p.error('Output already exists')
 if a.num_workers<0 or a.warmup_steps<0 or (a.max_samples is not None and a.max_samples<1):p.error('Invalid workers/warmup/max-samples')
 os.environ['HF_HUB_OFFLINE']='1'
 sys.path.insert(0,str(HERE/'inference'));sys.path.insert(0,str(HERE/'inference/src'))
 import torch
 pass
 device=torch.device('cuda:0');api=getattr(torch,device.type)
 if not api.is_available():raise RuntimeError('Requested accelerator unavailable; no CPU fallback')
 api.set_device(device)
 from inference.timer import NetworkTimer
 from inference import runner
 timer=NetworkTimer(device);runner.TIMER=timer
 from depth_anything_3.api import DepthAnything3
 model=DepthAnything3.from_pretrained(str(a.weights),local_files_only=True).to(device).eval()
 timer.wrap(model.model,'forward')
 a.output_dir.mkdir(parents=True,exist_ok=False)
 try:
  runner.run_inference(model,datasets=[a.dataset],dataset_root=a.dataset_root,output_dir=a.output_dir,batch_size=1,num_workers=a.num_workers,input_size=504,warmup_runs=a.warmup_steps,max_samples=a.max_samples)
  config=json.loads((a.output_dir/'evaluation_config.json').read_text())
  config['dataset_root']=os.path.relpath(a.dataset_root.absolute(),a.output_dir.absolute())
  config['evaluation_scope']='functional_subset' if a.max_samples else 'full_native_split'
  config['timing']={'forward':'actual network forward only; excludes output selection/interpolation/D2H/save','end_to_end':'formal dataset read through saves; excludes loading,warmup,offline scoring'}
  (a.output_dir/'evaluation_config.json').write_text(json.dumps(config,indent=2))
  (a.output_dir/'model_structure.txt').write_text(str(model),encoding='utf-8')
  (a.output_dir/'model_environment.json').write_text(json.dumps({'python':sys.version,'torch':torch.__version__,'device':str(device)}))
  (a.output_dir/'inference_output.json').write_text(json.dumps({'state':'completed','task':'depth-estimation','datasets':[a.dataset]}))
 except Exception:
  (a.output_dir/'inference_output.json').write_text(json.dumps({'state':'failed','task':'depth-estimation'}));raise
 finally:timer.close()
if __name__=='__main__':main()
