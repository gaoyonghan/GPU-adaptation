MODEL_ID='facebook/convnext-small-224'
REVISION='b34f016dbc4949fc8cd41b53c828e64186baae2b'
KIND='imagenet'
DATASET_REL='dataset/imagenet-1k'
"""Official model loading and inference. Run task evaluate.py separately."""
import argparse,json,os,sys
from pathlib import Path
from datetime import datetime
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[3]
# Default assets follow the project; explicit paths can point to AFS or AOSS.
WEIGHTS=HERE/'weights'
DATASET_ROOT=ROOT/DATASET_REL
BATCH_SIZE,NUM_WORKERS,WARMUP_STEPS=64,8,10
os.environ.setdefault('HF_HOME',str(HERE/'weights/.runtime_cache'))
os.environ.setdefault('HF_DATASETS_CACHE',str(HERE/'cache/datasets'))
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--weights',type=Path,default=WEIGHTS)
    p.add_argument('--dataset-root',type=Path,default=DATASET_ROOT)
    p.add_argument('--output-root', '--output-dir', dest='output_dir', type=Path, default=HERE/'result'/'run_001', help='Run directory; relative paths resolve against this model folder (--output-dir is a compatibility alias)')
    p.add_argument('--batch-size',type=int,default=BATCH_SIZE)
    p.add_argument('--num-workers', '--workers',type=int,default=NUM_WORKERS)
    p.add_argument('--max-samples',type=int,help='Debug subset; omit only for an intended full evaluation')
    p.add_argument('--preflight',action='store_true',help='Check paths only; no model loading or inference')
    a = p.parse_args()
    # Asset paths are model-relative; absolute paths remain supported.
    for _name in ('weights', 'dataset_root', 'dataset', 'bundle', 'vision_weights', 'hf_cache'):
        _path = getattr(a, _name, None)
        if isinstance(_path, Path) and not _path.is_absolute():
            setattr(a, _name, (HERE / _path).resolve())
    if not a.output_dir.is_absolute():
        a.output_dir = HERE / a.output_dir
    a.output_dir = a.output_dir.resolve()
    if a.output_dir.exists():
        p.error('Output directory already exists; use a new --output-root to preserve previous results')
    if a.max_samples is not None and a.max_samples<1:p.error('--max-samples must be positive')
    if a.preflight:
        required=[a.weights, a.dataset_root]
        missing=[str(path) for path in required if not path.exists()]
        print(json.dumps({'model':MODEL_ID,'missing_paths':missing,'state':'missing_assets' if missing else 'paths_present','note':'path check only; no model or dataset validation'}))
        if missing:raise SystemExit(2)
        return
    import torch
    if not torch.cuda.is_available():raise RuntimeError('CUDA unavailable; no CPU fallback')
    device=torch.device('cuda:0');torch.cuda.set_device(device);api=torch.cuda
    processor=None
    if KIND=='hqq':
        from inference.hqq_model import load_model,forward
        processor,model=load_model(str(a.weights/'visual'),True,a.weights/'text/open_clip_model.safetensors')
    else:
        from transformers import AutoImageProcessor,AutoModelForImageClassification
        processor=AutoImageProcessor.from_pretrained(a.weights,local_files_only=True)
        model=AutoModelForImageClassification.from_pretrained(a.weights,local_files_only=True).to(device).eval()
        forward=None
    model_info={'id':MODEL_ID,'revision':REVISION,'source':str(a.weights),'device':str(device)}
    a.output_dir.mkdir(parents=True,exist_ok=False)
    (a.output_dir/'model_structure.txt').write_text(str(model))
    import importlib.metadata
    versions={}
    for pkg in ['torch','torch_npu','transformers','datasets','numpy','Pillow','scikit-learn','hqq','open_clip_torch']:
        try:versions[pkg]=importlib.metadata.version(pkg)
        except importlib.metadata.PackageNotFoundError:pass
    (a.output_dir/'model_environment.json').write_text(json.dumps({'python':sys.version,'device':str(device),'packages':versions},indent=2))
    from inference.data import load,LimitedImages
    dataset,categories,dataset_info=load(a.dataset_root,model)
    if a.max_samples is not None:
        dataset=LimitedImages(dataset,a.max_samples)
        dataset_info=dict(dataset_info,scope='debug_subset',max_samples=a.max_samples)
    if KIND in ['chest','clothes']:
        from inference.model_data import ValidationProcessor
        processor=ValidationProcessor(processor)
    from inference.run_inference import run_inference
    kwargs={'forward_fn':forward} if forward is not None else {}
    run_inference(model,processor,dataset,categories,a.output_dir,batch_size=a.batch_size,num_workers=a.num_workers,
        warmup_steps=WARMUP_STEPS,model_info=model_info,dataset_info=dataset_info,**kwargs)
    from inference.prepare import publish
    publish(a.output_dir,model_info,dataset_info)
    print('Inference/package complete:',a.output_dir,'; run task evaluate.py separately')
if __name__=='__main__':main()
