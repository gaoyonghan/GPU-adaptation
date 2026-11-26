MODE='hf'
"""Official model inference; run task evaluate.py separately to compute accuracy."""
import argparse, json, sys
from pathlib import Path
from datetime import datetime
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[3]
# Paths are based on the project root, never the terminal working directory.
DEFAULT_WEIGHTS=HERE/'weights'
DEFAULT_DATASET=ROOT/'dataset'
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--weights',type=Path,default=DEFAULT_WEIGHTS,help='Complete HF directory or directory containing native .pth')
    p.add_argument('--dataset-root',type=Path,default=DEFAULT_DATASET,help='Parent of KITTI/NYUv2/DIODE/Sintel')
    p.add_argument('--dataset',choices=['KITTI','NYUv2','DIODE','Sintel'],default='NYUv2')
    p.add_argument('--output-root','--output-dir',type=Path,dest="output_dir",default=HERE/'result'/'run_001')
    p.add_argument('--device',default='cuda:0')
    p.add_argument('--num-workers','--workers',type=int,default=4)
    p.add_argument('--warmup-runs','--warmup',type=int,default=3)
    p.add_argument('--max-samples',type=int,help='Debug subset only; omit for full dataset')
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
    if a.max_samples is not None and a.max_samples<1: p.error('--max-samples must be positive')
    import torch
    if not torch.cuda.is_available(): raise RuntimeError('CUDA unavailable; no silent CPU fallback')
    device=torch.device(a.device)
    if device.type!='cuda': p.error('This delivery targets NVIDIA CUDA')
    torch.cuda.set_device(device)
    # Retain the existing DA-V2 FP32 baseline; no AMP or implicit offload.
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    processor=None
    if MODE=='hf':
        from transformers import AutoImageProcessor,AutoModelForDepthEstimation
        processor=AutoImageProcessor.from_pretrained(a.weights,local_files_only=True)
        model=AutoModelForDepthEstimation.from_pretrained(a.weights,local_files_only=True)
    else:
        sys.path.insert(0,str(HERE/'inference'))
        from depth_anything_v2.dpt import DepthAnythingV2
        cfg={'vits':dict(features=64,out_channels=[48,96,192,384]),'vitb':dict(features=128,out_channels=[96,192,384,768])}
        model=DepthAnythingV2(encoder=MODE,**cfg[MODE])
        model.load_state_dict(torch.load(a.weights/('depth_anything_v2_'+MODE+'.pth'),map_location='cpu',weights_only=True))
    model=model.to(device).eval()
    from inference.runner import run
    try:
        run(model,processor,a.dataset_root,a.output_dir,a.num_workers,a.warmup_runs,a.dataset,a.max_samples)
    except Exception:
        if a.output_dir.exists(): (a.output_dir/'inference_output.json').write_text(json.dumps({'task':'depth-estimation','state':'failed'}))
        raise
    (a.output_dir/'inference_output.json').write_text(json.dumps({'task':'depth-estimation','state':'completed','datasets':[a.dataset],'result':'.'},indent=2))
if __name__=='__main__': main()
