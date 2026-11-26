MODEL_NAME='vit_base_patch16_clip_384.laion2b_ft_in1k'
REVISION='f8778660141556288e90110db0ec7527ea0c1f32'
BATCH_SIZE,NUM_WORKERS,WARMUP_STEPS=64,8,3
DATASET_REL='dataset/imagenet-1k'
"""timm official model and transform; inference/package generation only."""
import argparse,json,os,sys
from pathlib import Path
from datetime import datetime
from types import SimpleNamespace
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[3]
os.environ.setdefault('HF_HUB_OFFLINE','1')
os.environ.setdefault('HF_DATASETS_CACHE',str(HERE/'cache/datasets'))

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--weights',type=Path,default=HERE/'weights',help='Complete local model.safetensors directory or explicit checkpoint file')
    p.add_argument('--dataset-root',type=Path,default=ROOT/DATASET_REL)
    p.add_argument('--output-root', '--output-dir', dest='output_dir', type=Path, default=HERE/'result'/'run_001', help='Run directory; relative paths resolve against this model folder (--output-dir is a compatibility alias)')
    p.add_argument('--batch-size',type=int,default=BATCH_SIZE)
    p.add_argument('--num-workers', '--workers',type=int,default=NUM_WORKERS)
    p.add_argument('--max-samples',type=int,help='Debug subset only')
    p.add_argument('--preflight',action='store_true',help='Check local paths only; do not load or infer')
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
    checkpoint=a.weights if a.weights.is_file() else a.weights/'model.safetensors'
    if a.preflight:
        missing=[str(x) for x in [checkpoint,a.dataset_root] if not x.exists()]
        print(json.dumps({'model':MODEL_NAME,'missing_paths':missing,'state':'missing_assets' if missing else 'paths_present'}))
        if missing:raise SystemExit(2)
        return
    if a.max_samples is not None and a.max_samples<1:p.error('--max-samples must be positive')
    if not checkpoint.is_file():raise FileNotFoundError(checkpoint)
    import torch
    if not torch.cuda.is_available():raise RuntimeError('CUDA unavailable; no CPU fallback')
    device=torch.device('cuda:0');torch.cuda.set_device(device)
    import timm
    from inference.timm_adapter import Processor,forward
    from inference.data import load,LimitedImages
    from inference.run_inference import run_inference
    from inference.prepare import publish
    # Use timm's native model-specific pretrained filters, overriding only the file source.
    # This avoids depending on mutable HF cache refs/main or downloading another revision.
    model=timm.create_model(MODEL_NAME,pretrained=True,
        pretrained_cfg_overlay={'source':'','file':str(checkpoint.resolve()),'custom_load':False}).to(device).eval()
    if model.num_classes!=1000:raise ValueError('Expected ImageNet-1K 1000 outputs')
    model.config=SimpleNamespace(id2label={i:'ImageNet-1K index '+str(i) for i in range(1000)})
    processor=Processor(model)
    dataset,categories,dataset_info=load(a.dataset_root,model)
    if a.max_samples is not None:
        dataset=LimitedImages(dataset,a.max_samples);dataset_info=dict(dataset_info,scope='debug_subset',max_samples=a.max_samples)
    model_info={'source':'timm','id':'timm/'+MODEL_NAME,'revision':REVISION,'weights':str(checkpoint.resolve()),'data_config':processor.config}
    a.output_dir.mkdir(parents=True,exist_ok=False)
    (a.output_dir/'model_structure.txt').write_text(str(model))
    import importlib.metadata
    versions={}
    for pkg in ['torch','torch_npu','timm','torchvision','datasets','numpy','Pillow','scikit-learn']:
        try:versions[pkg]=importlib.metadata.version(pkg)
        except importlib.metadata.PackageNotFoundError:pass
    (a.output_dir/'model_environment.json').write_text(json.dumps({'python':sys.version,'device':str(device),'packages':versions},indent=2))
    try:
        run_inference(model,processor,dataset,categories,a.output_dir,batch_size=a.batch_size,num_workers=a.num_workers,
            warmup_steps=WARMUP_STEPS,model_info=model_info,dataset_info=dataset_info,forward_fn=forward)
        publish(a.output_dir,model_info,dataset_info)
    except Exception:
        (a.output_dir/'inference_output.json').write_text(json.dumps({'state':'failed','task':'classification'}))
        raise
    print('Inference/package complete:',a.output_dir)
if __name__=='__main__':main()
