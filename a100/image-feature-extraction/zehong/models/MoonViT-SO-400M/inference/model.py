"""NVIDIA入口：官方模型加载 → 结构/环境记录 → 全部图片提取。"""
import os
from datetime import datetime
from pathlib import Path
ROOT = Path(__file__).resolve().parent
GPU_ID = '0'
MODEL_ID = 'moonshotai/MoonViT-SO-400M'
BUNDLE_DIR = (Path(__file__).resolve().parents[5] / 'dataset' / 'infovqa_bundle_1000')
BATCH_SIZE = 1
NUM_WORKERS = 8
WARMUP_STEPS = 3
os.environ.setdefault('CUDA_VISIBLE_DEVICES', GPU_ID)
os.environ.setdefault('HF_ENDPOINT', 'https://hf-mirror.com')
os.environ.setdefault('HF_HUB_CACHE', str(Path(__file__).resolve().parents[1] / 'weights/.runtime_cache'))
import torch
from huggingface_hub import snapshot_download
from transformers import AutoImageProcessor, AutoModel
from inference.utils.evaluate_model import evaluate_model
from inference.utils.print_model_info import print_model_and_environment
from inference.utils.protocol import checkpoint_fingerprints, processor_protocol

class PrepareBatch:
    """单图官方预处理；仅增加适配评测器的外层batch维。"""

    def __init__(self, processor, dtype):
        self.processor = processor
        self.dtype = dtype

    def __call__(self, images):
        if len(images) != 1:
            raise ValueError('MoonViT minimal adapter requires batch_size=1')
        inputs = self.processor(images=images, return_tensors='pt')
        return {'pixel_values': inputs['pixel_values'].to(dtype=self.dtype).unsqueeze(0), 'image_grid_hws': inputs['image_grid_hws']}

def forward_fn(model, inputs):
    return model(inputs['pixel_values'].squeeze(0), inputs['image_grid_hws'])

def select_output(outputs):
    if not isinstance(outputs, (list, tuple)) or len(outputs) != 1:
        raise TypeError('Expected one image feature tensor')
    return outputs[0].unsqueeze(0)

def main():
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA不可用，请检查环境；不自动改用CPU')
    snapshot = snapshot_download(MODEL_ID, allow_patterns=['*.safetensors', '*.bin', '*.json', '*.py'])
    processor = AutoImageProcessor.from_pretrained(snapshot, trust_remote_code=True)
    model = AutoModel.from_pretrained(snapshot, trust_remote_code=True, torch_dtype='auto', attn_implementation='sdpa')
    model.cuda().eval()
    output_dir = RUN_ARGS.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    print_model_and_environment(model, output_dir)
    summary = evaluate_model(model, PrepareBatch(processor, model.dtype), forward_fn, select_output, bundle_dir=BUNDLE_DIR, output_dir=output_dir, model_info={'source': 'huggingface', 'id': MODEL_ID, 'revision': Path(snapshot).name, 'checkpoint_files': checkpoint_fingerprints(snapshot)}, preprocessing_info={**processor_protocol(processor), 'floating_input_dtype': str(model.dtype), 'batch_adapter': 'single_image_outer_batch_axis', 'attention_implementation': 'sdpa'}, output_name='forward_patch_merger_features', batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, warmup_steps=WARMUP_STEPS, dynamic_single_image=True)
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
