"""模型推理入口；保存原始预测与计时，另行运行任务evaluate.py评分。"""
from pathlib import Path
import argparse,os,sys
from datetime import datetime
HERE=Path(__file__).resolve().parent
KIND='v2_vitl'
BACKEND='cuda'
def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--weights',type=Path,default=HERE/'weights')
 p.add_argument('--dataset-root',type=Path,default=HERE.parents[3]/'dataset')
 p.add_argument('--dataset',choices=['KITTI','NYUv2','DIODE','Sintel'],default='NYUv2')
 p.add_argument('--output-root','--output-dir',type=Path,dest="output_dir",default=HERE/'result'/'run_001')
 p.add_argument('--num-workers','--workers',type=int,default=4)
 p.add_argument('--warmup-steps','--warmup',type=int,default=3)
 p.add_argument('--max-samples',type=int,help='Only for debug; omit for the whole dataset')
 p.add_argument('--seed',type=int,default=0)
 p.add_argument('--lotus-dtype',choices=['float16','float32'],default='float16')
 p.add_argument('--attention',choices=['eager','sdpa'],default='eager' if KIND=='zoe' and BACKEND=='npu' else None)
 a=p.parse_args()
 # Asset paths are model-relative; absolute paths remain supported.
 for _name in ('weights', 'dataset_root', 'dataset', 'bundle', 'vision_weights', 'hf_cache', 'image'):
     _path = getattr(a, _name, None)
     if isinstance(_path, Path) and not _path.is_absolute():
         setattr(a, _name, (HERE / _path).resolve())
 # CLI paths are relative to this model directory, independent of cwd.
 if a.weights is not None and not a.weights.is_absolute(): a.weights=(HERE/a.weights).resolve()
 if a.dataset_root is not None and not a.dataset_root.is_absolute(): a.dataset_root=(HERE/a.dataset_root).resolve()
 if a.output_dir is not None and not a.output_dir.is_absolute(): a.output_dir=(HERE/a.output_dir).resolve()
 if a.output_dir.exists(): p.error("Output directory already exists; choose a new --output-root")
 if a.num_workers<0 or a.warmup_steps<0 or (a.max_samples is not None and a.max_samples<1):p.error('Invalid workers/warmup/max-samples')
 if a.output_dir.exists():p.error('Output directory already exists; choose a new run')
 if not a.weights.exists():p.error('Missing local weights; see README')
 os.environ['HF_HOME']=str(HERE/'cache/huggingface');os.environ['HF_HUB_CACHE']=str(HERE/'cache/huggingface/hub')
 os.environ['HF_MODULES_CACHE']=str(HERE/'cache/huggingface/modules');os.environ['TORCH_HOME']=str(HERE/'cache/torch')
 os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
 sys.path.insert(0,str(HERE/'inference'))
 import torch
 if BACKEND=='npu':import torch_npu
 api=getattr(torch,BACKEND)
 if not api.is_available():raise RuntimeError('Requested accelerator unavailable; no CPU fallback')
 device=torch.device(BACKEND+':0');api.set_device(device);torch.manual_seed(a.seed);api.manual_seed_all(a.seed)
 from inference.adapter import Adapter
 from inference.runner import run
 adapter=Adapter(KIND,a.weights.resolve(),device,a.lotus_dtype,a.attention)
 try:run(adapter,a,api)
 except Exception:
  if a.output_dir.exists():
   import json
   (a.output_dir/'inference_output.json').write_text(json.dumps({'task':'depth-estimation','state':'failed'}))
  raise
if __name__=='__main__':main()
