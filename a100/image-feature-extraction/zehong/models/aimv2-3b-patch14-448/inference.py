"""AIMv2官方加载→逐图完整特征和计时；评分由任务evaluate.py独立执行。"""
import argparse
import json
import os
from datetime import datetime
from pathlib import Path

MODEL_DIR = Path(__file__).resolve().parent
ROOT = MODEL_DIR.parents[3]
MODEL_ID = 'apple/aimv2-3B-patch14-448'
MODEL_REVISION = '70810b618e4456b724bacc8ef4d2d038060ceda6'
BUNDLE_DIR = ROOT/'dataset/aimv2_features/imagenet_validation_1000'
BATCH_SIZE, NUM_WORKERS, WARMUP_STEPS = 8, 8, 3
os.environ.setdefault('CUDA_VISIBLE_DEVICES','0')
os.environ.setdefault('HF_HOME',str(MODEL_DIR/'weights/.runtime_cache'))


class PrepareBatch:
    """沿用官方processor，逐图处理后拼batch，支持spawn worker。"""
    def __init__(self,processor): self.processor=processor
    def __call__(self,images):
        import torch
        samples=[self.processor(images=image,return_tensors='pt') for image in images]
        return {key:torch.cat([s[key] for s in samples]) for key in samples[0]}


def forward_fn(model,inputs): return model(**inputs)


def select_output(outputs): return outputs.last_hidden_state


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--weights',type=Path,default=MODEL_DIR/'weights')
    parser.add_argument('--bundle',type=Path,default=BUNDLE_DIR)
    parser.add_argument('--output-root','--output-dir',type=Path,dest="output_dir",default=MODEL_DIR/'result'/'run_001')
    parser.add_argument('--batch-size',type=int,default=BATCH_SIZE)
    parser.add_argument('--num-workers','--workers',type=int,default=NUM_WORKERS)
    args=parser.parse_args()
    # Asset paths are model-relative; absolute paths remain supported.
    for _name in ('weights', 'dataset_root', 'dataset', 'bundle', 'vision_weights', 'hf_cache', 'image'):
        _path = getattr(args, _name, None)
        if isinstance(_path, Path) and not _path.is_absolute():
            setattr(args, _name, (MODEL_DIR / _path).resolve())
    # CLI paths are relative to this model directory, independent of cwd.
    if args.weights is not None and not args.weights.is_absolute(): args.weights=(MODEL_DIR/args.weights).resolve()
    if args.bundle is not None and not args.bundle.is_absolute(): args.bundle=(MODEL_DIR/args.bundle).resolve()
    if args.output_dir is not None and not args.output_dir.is_absolute(): args.output_dir=(MODEL_DIR/args.output_dir).resolve()
    if args.output_dir.exists(): parser.error("Output directory already exists; choose a new --output-root")
    import torch
    from transformers import AutoImageProcessor,AutoModel
    from inference.evaluate_model import evaluate_model
    from inference.print_model_info import print_model_and_environment
    from inference.protocol import checkpoint_fingerprints,processor_protocol
    if not torch.cuda.is_available(): raise RuntimeError('CUDA不可用；请检查执行环境的GPU访问权限，不自动退CPU。')
    processor=AutoImageProcessor.from_pretrained(args.weights,local_files_only=True)
    model=AutoModel.from_pretrained(args.weights,trust_remote_code=True,local_files_only=True).cuda().eval()
    # 不覆盖官方加载dtype、AMP、TF32、CPU分支或线程数；记录实际条件。
    args.output_dir.mkdir(parents=True,exist_ok=False)
    print_model_and_environment(model,args.output_dir)
    try:
        summary=evaluate_model(model,PrepareBatch(processor),forward_fn,select_output,
            bundle_dir=args.bundle,output_dir=args.output_dir,
            model_info={'source':'huggingface','id':MODEL_ID,'revision':MODEL_REVISION,
                        'checkpoint_files':checkpoint_fingerprints(args.weights)},
            preprocessing_info=processor_protocol(processor),output_name='last_hidden_state',
            batch_size=args.batch_size,num_workers=args.num_workers,warmup_steps=WARMUP_STEPS)
        if summary['status']!='complete': raise RuntimeError('存在无效特征，查看summary.json')
    except Exception:
        (args.output_dir/'inference_output.json').write_text(json.dumps({'state':'failed','task':'feature_extraction'}))
        raise
    (args.output_dir/'inference_output.json').write_text(json.dumps({'state':'completed','task':'feature_extraction','format':'raw-tensor-features-v4','result':'.'},indent=2))
    print('Result:',args.output_dir)
    print('Performance:',summary['performance'])


if __name__=='__main__': main()
