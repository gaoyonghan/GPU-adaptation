"""Model native inference and standardized outputs; score with task-level evaluator."""
import argparse,os,json
from pathlib import Path
from datetime import datetime
HERE=Path(__file__).resolve().parent
def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--hf-cache',type=Path,default=HERE/'weights/.runtime_cache',help='Explicit external HF cache for nested model dependencies')
 p.add_argument('--weights',type=Path,default=HERE/'weights')
 p.add_argument('--vision-weights',type=Path,help='CLIP backbone local snapshot, only for clip-face-expression')
 p.add_argument('--dataset-root',type=Path,help='Override this model native dataset root')
 p.add_argument('--output-root', '--output-dir', dest='output_dir', type=Path, default=HERE/'result'/'run_001', help='Run directory; relative paths resolve against this model folder (--output-dir is a compatibility alias)')
 p.add_argument('--batch-size',type=int,default=64)
 p.add_argument('--num-workers', '--workers',type=int,default=8)
 p.add_argument('--warmup-steps', '--warmup',type=int,default=3)
 p.add_argument('--max-samples',type=int)
 a = p.parse_args()
 # Asset paths are model-relative; absolute paths remain supported.
 for _name in ('weights', 'dataset_root', 'dataset', 'bundle', 'vision_weights', 'hf_cache', 'image'):
     _path = getattr(a, _name, None)
     if isinstance(_path, Path) and not _path.is_absolute():
         setattr(a, _name, (HERE / _path).resolve())
 if not a.output_dir.is_absolute():
     a.output_dir = HERE / a.output_dir
 a.output_dir = a.output_dir.resolve()
 if a.output_dir.exists():
     p.error('Output directory already exists; use a new --output-root to preserve previous results')
 a.device='cuda:0';a.model_name='eva02_base_patch14_448.mim_in22k_ft_in22k_in1k'
 if a.batch_size<1 or a.num_workers<0 or a.warmup_steps<0 or (a.max_samples is not None and a.max_samples<1):p.error('Invalid batch/workers/warmup/max-samples')
 if a.output_dir.exists():p.error('Output directory already exists')
 os.environ['HF_HUB_OFFLINE']='1'
 os.environ['HF_HUB_CACHE']=str(a.hf_cache.absolute())
 from inference import options
 options.ARGS=a
 import torch
 pass
 api=getattr(torch,a.device.split(':')[0])
 if not api.is_available():raise RuntimeError('Requested device unavailable; no CPU fallback')
 api.set_device(a.device)
 try:
  import inference.model as native
  if hasattr(native,'main'):native.main()
 except Exception:
  if a.output_dir.exists():(a.output_dir/'inference_output.json').write_text(json.dumps({'state':'failed','task':'classification'}))
  raise
if __name__=='__main__':main()
