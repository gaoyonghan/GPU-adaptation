BUNDLE_NAME='NYUv2_bundle_1000'
DEFAULT_BATCH=64
"""模型推理入口；公共evaluate.py独立比较两个结果目录。"""
import argparse,os
from pathlib import Path
from datetime import datetime
HERE=Path(__file__).resolve().parent
PROJECT=HERE.parents[3]
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--weights',type=Path,default=HERE/'weights',help='完整本地模型目录；不得只复制HF软链接')
    p.add_argument('--bundle',type=Path,default=PROJECT/'dataset'/BUNDLE_NAME)
    p.add_argument('--output-root','--output-dir',type=Path,dest="output_dir",default=HERE/'result'/'run_001')
    p.add_argument('--batch-size',type=int,default=DEFAULT_BATCH)
    p.add_argument('--num-workers','--workers',type=int,default=8)
    p.add_argument('--warmup-steps','--warmup',type=int,default=3)
    p.add_argument('--max-samples',type=int,help='仅调试；省略为全量')
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
    if not a.weights.is_dir():p.error('weights must be a complete local directory')
    os.environ.setdefault('HF_HUB_OFFLINE','1')
    import inference.model as native
    native.RUN_ARGS=a
    native.ROOT=HERE
    native.BUNDLE_DIR=a.bundle.resolve()
    native.BATCH_SIZE=a.batch_size
    native.NUM_WORKERS=a.num_workers
    native.WARMUP_STEPS=a.warmup_steps
    if hasattr(native,'MODEL_DIR'):native.MODEL_DIR=a.weights.resolve()
    from inference.utils import evaluate_model as evaluator
    evaluator.MAX_SAMPLES=a.max_samples
    native.main()
    import json
    report=json.loads((a.output_dir/'summary.json').read_text())
    (a.output_dir/'inference_output.json').write_text(json.dumps({'state':'completed' if report['status']=='complete' else 'failed','task':'image-feature-extraction'}))
if __name__=='__main__':main()
