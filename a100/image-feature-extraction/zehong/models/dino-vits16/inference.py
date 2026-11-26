"""Offline feature extraction. Compare completed runs using the task-root evaluate.py."""
import argparse,json,os,sys,traceback
from pathlib import Path
from datetime import datetime,timezone
HERE=Path(__file__).resolve().parent
def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--weights',type=Path,default=HERE/'weights',help='Complete local checkpoint directory')
 p.add_argument('--bundle',type=Path,default=HERE.parents[3]/'dataset'/'aimv2_features/imagenet_validation_1000')
 p.add_argument('--output-root','--output-dir',type=Path,dest="output_dir",default=HERE/'result'/'run_001')
 p.add_argument('--batch-size',type=int,default=1)
 p.add_argument('--num-workers','--workers',type=int,default=0)
 p.add_argument('--warmup-steps','--warmup',type=int,default=0)
 p.add_argument('--max-samples','--max-images',type=int)
 p.add_argument('--device',default='cuda:0')
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
 if a.batch_size<1 or a.num_workers<0 or a.warmup_steps<0 or (a.max_samples is not None and a.max_samples<1):p.error('Invalid sample/batch/workers/warmup')
 if a.device.split(':')[0]!='cuda':p.error('This implementation requires cuda; no CPU fallback')
 if 'native' in HERE.name and a.batch_size!=1:p.error('AIMv2 native requires batch-size=1 for variable image sizes')
 a.weights=a.weights.resolve()
 if not a.weights.is_dir():raise FileNotFoundError(a.weights)
 if not (a.bundle/'input_manifest.jsonl').is_file():raise FileNotFoundError(a.bundle/'input_manifest.jsonl')
 os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
 a.output_dir=a.output_dir.resolve();a.output_dir.mkdir(parents=True,exist_ok=False)
 record={'state':'running','model':'facebook/dino-vits16','argv':sys.argv,'started_at':datetime.now(timezone.utc).isoformat(),'weights':str(a.weights),'bundle':str(a.bundle)}
 try:
  from inference.model import extract
  extract(a)
  record['state']='completed'
 except Exception:
  record['state']='failed';record['error']=traceback.format_exc();raise
 finally:
  (a.output_dir/'inference_output.json').write_text(json.dumps(record,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
