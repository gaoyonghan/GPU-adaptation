"""Native model inference; offline scoring uses the task-level evaluate.py."""
import argparse
import json
import os
from pathlib import Path
from datetime import datetime
HERE = Path(__file__).resolve().parent

def main():
    cfg = json.loads((HERE/'inference/config.json').read_text(encoding='utf-8'))
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--weights', type=Path, default=Path(cfg['weights']))
    p.add_argument('--dataset-root', type=Path, default=Path(cfg['dataset_root']))
    p.add_argument('--output-root', '--output-dir', dest='output_dir', type=Path, default=HERE/'result'/'run_001', help='Run directory; relative paths resolve against this model folder (--output-dir is a compatibility alias)')
    p.add_argument('--device', default=cfg['device'])
    p.add_argument('--batch-size', type=int, default=64)
    p.add_argument('--num-workers', '--workers', type=int, default=8)
    p.add_argument('--warmup-steps', '--warmup', type=int, default=cfg['warmup_steps'])
    p.add_argument('--max-samples', type=int, help='Explicit debug subset; omitted means native split')
    p.add_argument('--timing-mode', choices=['batched','single'], default='batched')
    a = p.parse_args()
    # Asset paths are model-relative; absolute paths remain supported.
    for _name in ('weights', 'dataset_root', 'dataset', 'bundle', 'vision_weights', 'hf_cache', 'image'):
        _path = getattr(a, _name, None)
        if isinstance(_path, Path) and not _path.is_absolute():
            setattr(a, _name, (HERE / _path).resolve())
    if not a.weights.is_absolute(): a.weights = HERE / a.weights
    if not a.dataset_root.is_absolute(): a.dataset_root = HERE / a.dataset_root
    if not a.output_dir.is_absolute():
        a.output_dir = HERE / a.output_dir
    a.output_dir = a.output_dir.resolve()
    if a.output_dir.exists():
        p.error('Output directory already exists; use a new --output-root to preserve previous results')
    if a.batch_size < 1 or a.num_workers < 0 or a.warmup_steps < 0 or (a.max_samples is not None and a.max_samples < 1):
        p.error('Invalid batch/workers/warmup/max-samples')
    if a.output_dir.exists(): p.error('Output directory already exists; keep previous results')
    if a.device.split(':')[0] != cfg['device'].split(':')[0]: p.error('Wrong accelerator backend; CPU fallback is disabled')
    if not a.weights.exists() or not a.dataset_root.is_dir(): p.error('Local weights or dataset root missing')
    if a.timing_mode == 'single': a.batch_size = 1
    a.model_name = cfg['model_name']
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['HF_DATASETS_OFFLINE'] = '1'
    os.environ['HF_HOME'] = str(HERE/'weights/.runtime_cache')
    os.environ['HF_DATASETS_CACHE'] = str(HERE/'weights/.runtime_cache/datasets')
    from inference import options
    options.ARGS = a
    import torch
    if cfg['device'].startswith('npu'):
        import torch_npu
    api = getattr(torch, a.device.split(':')[0])
    if not api.is_available(): raise RuntimeError('Requested accelerator is unavailable')
    api.set_device(a.device)
    from inference.model import load_model
    from inference.runner import evaluate_model
    model, processor = load_model(a.weights)
    model = model.to(torch.device(a.device)).eval()
    evaluate_model(model, dataset_name=cfg['dataset_name'], image_processor=processor,
                   timing_mode=a.timing_mode)

if __name__ == '__main__':
    main()
