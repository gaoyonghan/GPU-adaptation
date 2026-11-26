"""Native model inference; task evaluate.py scores two complete result roots."""
import argparse,os,json
from pathlib import Path
from datetime import datetime
HERE=Path(__file__).resolve().parent
def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--weights',type=Path,default=HERE/'weights')
 p.add_argument('--bundle',type=Path,default=HERE.parents[3]/'dataset'/'aimv2_features/imagenet_validation_1000')
 p.add_argument('--output-root','--output-dir',type=Path,dest="output_dir",default=HERE/'result'/'run_001')
 p.add_argument('--batch-size',type=int,default=64)
 p.add_argument('--num-workers','--workers',type=int,default=8)
 p.add_argument('--warmup-steps','--warmup',type=int,default=3)
 p.add_argument('--max-samples',type=int)
 a=p.parse_args()
 # Asset paths are model-relative; absolute paths remain supported.
 for _name in ('weights', 'dataset_root', 'dataset', 'bundle', 'vision_weights', 'hf_cache', 'image'):
     _path = getattr(a, _name, None)
     if isinstance(_path, Path) and not _path.is_absolute():
         setattr(a, _name, (HERE / _path).resolve())
 # CLI paths are relative to this model directory, independent of cwd.
 if a.weights is not None and not a.weights.is_absolute(): a.weights=(HERE/a.weights).resolve()
 if a.bundle is not None and not a.bundle.is_absolute(): a.bundle=(HERE/a.bundle).resolve()
 if a.output_dir is not None and not a.output_dir.is_absolute(): a.output_dir=(HERE/a.output_dir).resolve()
 if a.output_dir.exists(): p.error("Output directory already exists; choose a new --output-root")
 if a.max_samples is not None and a.max_samples<1:p.error('--max-samples must be positive')
 if a.batch_size<1 or a.num_workers<0 or a.warmup_steps<0:p.error('Invalid batch/workers/warmup')
 os.environ['HF_HUB_OFFLINE']='1'
 import inference.model as native
 native.RUN_ARGS=a;native.ROOT=HERE;native.BUNDLE_DIR=a.bundle.absolute()
 native.BATCH_SIZE=a.batch_size;native.NUM_WORKERS=a.num_workers;native.WARMUP_STEPS=a.warmup_steps
 from inference.utils import evaluate_model as runner
 runner.MAX_SAMPLES=a.max_samples
 try:
  native.main()
  report=json.loads((a.output_dir/'summary.json').read_text())
  if report['status']!='complete':raise RuntimeError('Invalid features')
 except Exception:
  if a.output_dir.exists():(a.output_dir/'inference_output.json').write_text(json.dumps({'state':'failed','task':'image-feature-extraction'}))
  raise
 (a.output_dir/'inference_output.json').write_text(json.dumps({'state':'completed','task':'image-feature-extraction','format':'raw-tensor-features-v4'}))
if __name__=='__main__':main()
