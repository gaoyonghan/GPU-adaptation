"""YOLOv8 classification inference; run the task evaluate.py separately."""
import argparse
import json
import os
import sys
from pathlib import Path

MODEL_ID = 'keremberke/yolov8m-chest-xray-classification'
DATASET_NAME = 'chest-xray-classification'
DEFAULT_SPLIT = 'valid'
BATCH_SIZE, NUM_WORKERS, WARMUP_STEPS = 64, 8, 3

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]  # jingzhi/
os.environ.setdefault('YOLO_CONFIG_DIR', str(HERE / 'weights/.runtime_cache/ultralytics-config'))
os.environ.setdefault('HF_HUB_OFFLINE', '1')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--weights', type=Path, default=HERE / 'weights')
    parser.add_argument('--dataset-root', type=Path, default=HERE.parents[3] / 'dataset' / DATASET_NAME)
    parser.add_argument('--split', default=DEFAULT_SPLIT)
    parser.add_argument('--output-root', '--output-dir', dest='output_dir', type=Path, default=HERE/'result'/'run_001', help='Run directory; relative paths resolve against this model folder (--output-dir is a compatibility alias)')
    parser.add_argument('--batch-size', type=int, default=BATCH_SIZE)
    parser.add_argument('--num-workers', '--workers', type=int, default=NUM_WORKERS)
    parser.add_argument('--warmup-steps', '--warmup', type=int, default=WARMUP_STEPS)
    parser.add_argument('--max-samples', type=int)
    parser.add_argument('--preflight', action='store_true', help='Check paths only; no model loading or inference')
    args = parser.parse_args()
    # Asset paths are model-relative; absolute paths remain supported.
    for _name in ('weights', 'dataset_root', 'dataset', 'bundle', 'vision_weights', 'hf_cache', 'image'):
        _path = getattr(args, _name, None)
        if isinstance(_path, Path) and not _path.is_absolute():
            setattr(args, _name, (HERE / _path).resolve())
    if not args.output_dir.is_absolute():
        args.output_dir = HERE / args.output_dir
    args.output_dir = args.output_dir.resolve()
    if args.output_dir.exists():
        parser.error('Output directory already exists; use a new --output-root to preserve previous results')
    if args.max_samples is not None and args.max_samples < 1:
        parser.error('--max-samples must be positive')
    if args.batch_size < 1 or args.num_workers < 0 or args.warmup_steps < 0:
        parser.error('Invalid batch/workers/warmup')

    weights = args.weights / 'best.pt'
    dataset_dir = args.dataset_root / args.split
    missing = [str(p) for p in (weights, dataset_dir) if not p.exists()]
    if args.preflight:
        print(json.dumps({'model': MODEL_ID, 'weights': str(weights), 'dataset': str(dataset_dir),
                          'missing_paths': missing,
                          'state': 'missing_assets' if missing else 'paths_present',
                          'note': 'path check only; no model or dataset validation'}))
        if missing:
            raise SystemExit(2)
        return
    if missing:
        raise SystemExit('Missing assets: ' + ', '.join(missing))

    from inference.model import load_checkpoint, build_transform, resolve_device
    from inference.runner import run

    device = resolve_device()
    net, ckpt = load_checkpoint(weights)
    imgsz = int(ckpt['train_args'].get('imgsz', 224))
    transform = build_transform(imgsz)
    net.to(device)
    output_dir = args.output_dir.absolute()
    output_dir.mkdir(parents=True, exist_ok=False)
    summary = run(net, ckpt, transform, imgsz, device, output_dir,
                  dataset_root=args.dataset_root.absolute(), split=args.split, dataset_name=DATASET_NAME,
                  batch_size=args.batch_size, num_workers=args.num_workers,
                  warmup_steps=args.warmup_steps, max_samples=args.max_samples,
                  model_info={'id': MODEL_ID, 'source': str(weights),
                              'revision': ckpt.get('version'), 'date': ckpt.get('date')})
    print('Results:', output_dir)
    print('Status:', summary['status'], 'images:', summary['sample_count'])
    print('Performance:', summary['performance'])


if __name__ == '__main__':
    main()
