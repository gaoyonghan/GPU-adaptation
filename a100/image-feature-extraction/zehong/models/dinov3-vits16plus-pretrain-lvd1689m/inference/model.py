"""NVIDIA入口：官方模型加载 → 结构/环境记录 → 全部图片提取。"""
import os
from datetime import datetime
from pathlib import Path
ROOT = Path(__file__).resolve().parent
GPU_ID = '0'
MODEL_ID = 'facebook/dinov3-vits16plus-pretrain-lvd1689m'
BUNDLE_DIR = (Path(__file__).resolve().parents[5] / 'dataset' / 'aimv2_features/imagenet_validation_1000')
BATCH_SIZE = 64
NUM_WORKERS = 8
WARMUP_STEPS = 3
os.environ.setdefault('CUDA_VISIBLE_DEVICES', GPU_ID)
os.environ.setdefault('HF_ENDPOINT', 'https://hf-mirror.com')
os.environ.setdefault('HF_HUB_CACHE', str(Path(__file__).resolve().parents[1] / 'weights/.runtime_cache'))
import torch
from transformers import AutoImageProcessor, AutoModel
from inference.utils.evaluate_model import evaluate_model
from inference.utils.print_model_info import print_model_and_environment
from inference.utils.protocol import checkpoint_fingerprints, processor_protocol

class PrepareBatch:
    """官方processor逐图处理并拼批；类在顶层以支持spawn worker。"""

    def __init__(self, processor):
        self.processor = processor

    def __call__(self, images):
        samples = [self.processor(images=image, return_tensors='pt') for image in images]
        return {key: torch.cat([sample[key] for sample in samples]) for key in samples[0]}

def forward_fn(model, inputs):
    return model(**inputs)

def select_output(outputs):
    return outputs.pooler_output

def main():
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA不可用，请检查环境；不自动改用CPU')
    snapshot = RUN_ARGS.weights.resolve()
    processor = AutoImageProcessor.from_pretrained(snapshot, local_files_only=True, use_fast=True)
    model = AutoModel.from_pretrained(snapshot, local_files_only=True)
    model.cuda().eval()
    model_revision = snapshot.name if snapshot.parent.name == 'snapshots' else None
    output_dir = RUN_ARGS.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    print_model_and_environment(model, output_dir)
    summary = evaluate_model(model, PrepareBatch(processor), forward_fn, select_output, bundle_dir=BUNDLE_DIR, output_dir=output_dir, model_info={'source': 'huggingface', 'id': MODEL_ID, 'revision': model_revision, 'checkpoint_files': checkpoint_fingerprints(snapshot)}, preprocessing_info=processor_protocol(processor), output_name='pooler_output', batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, warmup_steps=WARMUP_STEPS)
    print('Results:', output_dir)
    print('Status:', summary['status'], 'images:', summary['num_images'])
    print('Performance:', summary['performance'])
    if summary['status'] != 'complete':
        raise RuntimeError('存在无效输出，请查看summary和feature_index')
RUN_ARGS = None

def snapshot_download(*args, **kwargs):
    return str(RUN_ARGS.weights.resolve())
if __name__ == '__main__':
    main()
