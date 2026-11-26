MODEL_ID='prs-eth/marigold-depth-lcm-v1-0'
DTYPE='float16'
STEPS=4
import argparse,json
from pathlib import Path
from datetime import datetime
HERE=Path(__file__).resolve().parent
PROJECT=HERE.parents[3]
def main():
    p=argparse.ArgumentParser(description='Marigold推理/保存；公共evaluate.py独立评分')
    p.add_argument('--weights',type=Path,default=HERE/'weights')
    p.add_argument('--dataset-root',type=Path,default=PROJECT/'dataset')
    p.add_argument('--dataset',choices=['KITTI','NYUv2','DIODE','Sintel'],default='NYUv2')
    p.add_argument('--output-root','--output-dir',type=Path,dest="output_dir",default=HERE/'result'/'run_001')
    p.add_argument('--max-samples',type=int)
    p.add_argument('--num-workers','--workers',type=int,default=4)
    p.add_argument('--warmup-runs','--warmup',type=int,default=1)
    p.add_argument('--steps',type=int,default=STEPS)
    p.add_argument('--ensemble',type=int,default=10)
    p.add_argument('--processing-resolution',type=int,default=768)
    p.add_argument('--seed',type=int,default=2024)
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
    if a.steps<1 or a.ensemble<1 or a.warmup_runs<0 or (a.max_samples is not None and a.max_samples<1):p.error('Invalid steps/ensemble/warmup/sample count')
    import torch
    if not torch.cuda.is_available():raise RuntimeError('CUDA unavailable')
    device=torch.device('cuda:0');torch.cuda.set_device(device)
    from diffusers import MarigoldDepthPipeline
    pipe=MarigoldDepthPipeline.from_pretrained(a.weights,torch_dtype=getattr(torch,DTYPE),local_files_only=True).to(device)
    for component in ['unet','vae','text_encoder']:
        module=getattr(pipe,component,None)
        if module is not None:module.eval().requires_grad_(False)
    from inference.runner import run
    try:run(pipe,a,device)
    except Exception:
        if a.output_dir.is_dir():(a.output_dir/'inference_output.json').write_text(json.dumps({'state':'failed','task':'depth-estimation'}))
        raise
if __name__=='__main__':main()
