"""NVIDIA入口：官方模型加载 → 结构/环境记录 → 全部图片提取。"""
import os
from datetime import datetime
from pathlib import Path
ROOT = Path(__file__).resolve().parent
GPU_ID = '0'
MODEL_ID = 'nvidia/RADIO'
BUNDLE_DIR = (Path(__file__).resolve().parents[5] / 'dataset' / 'aimv2_features/imagenet_validation_1000')
BATCH_SIZE = 64
NUM_WORKERS = 8
WARMUP_STEPS = 3
os.environ.setdefault('CUDA_VISIBLE_DEVICES', GPU_ID)
os.environ.setdefault('HF_ENDPOINT', 'https://hf-mirror.com')
os.environ.setdefault('HF_HUB_CACHE', str(Path(__file__).resolve().parents[1] / 'weights/.runtime_cache'))
import torch
from huggingface_hub import snapshot_download
from transformers import CLIPImageProcessor, AutoModel, AutoConfig
import torch.nn.functional as F
from inference.utils.evaluate_model import evaluate_model
from inference.utils.print_model_info import print_model_and_environment
from inference.utils.protocol import checkpoint_fingerprints, processor_protocol

class PrepareBatch:

    def __init__(self, processor, resolution):
        self.processor = processor
        self.resolution = tuple(resolution)

    def __call__(self, images):
        samples = []
        for image in images:
            x = self.processor(images=image, return_tensors='pt').pixel_values
            x = F.interpolate(x, size=self.resolution, mode='bilinear', align_corners=False, antialias=False)
            samples.append(x)
        return torch.cat(samples, dim=0)

def forward_fn(model, inputs):
    return model(inputs)

def select_output(outputs):
    return outputs.features

def load_radio_model(snapshot):
    """保留官方类和BF16；整包strict加载兼容其依赖完整state_dict的自定义钩子。"""
    from safetensors.torch import load_file
    config = AutoConfig.from_pretrained(snapshot, trust_remote_code=True, local_files_only=True)
    model = AutoModel.from_config(config, trust_remote_code=True, torch_dtype=torch.bfloat16)
    state = load_file(str(Path(snapshot) / 'model.safetensors'), device='cpu')
    expected = model.state_dict()
    if set(state) != set(expected) or any(state[k].shape != expected[k].shape for k in expected):
        raise RuntimeError('RADIO checkpoint keys or shapes differ from the official configured model')
    model.load_state_dict(state, strict=True)
    return model

def main():
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA不可用，请检查环境；不自动改用CPU')
    snapshot = snapshot_download(MODEL_ID, allow_patterns=['*.safetensors', '*.bin', '*.json', '*.py'])
    processor = CLIPImageProcessor.from_pretrained(snapshot)
    model = load_radio_model(snapshot)
    model.cuda().eval()
    output_dir = RUN_ARGS.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    print_model_and_environment(model, output_dir)
    resolution = tuple(model.config.preferred_resolution)
    preprocessing_info = processor_protocol(processor)
    preprocessing_info['external_resize'] = {'implementation': 'torch.nn.functional.interpolate', 'size': list(resolution), 'mode': 'bilinear', 'align_corners': False, 'antialias': False, 'device': 'cpu', 'dtype': 'torch.float32'}
    summary = evaluate_model(model, PrepareBatch(processor, resolution), forward_fn, select_output, bundle_dir=BUNDLE_DIR, output_dir=output_dir, model_info={'source': 'huggingface', 'id': MODEL_ID, 'revision': Path(snapshot).name, 'checkpoint_files': checkpoint_fingerprints(snapshot)}, preprocessing_info=preprocessing_info, output_name='spatial_features', batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, warmup_steps=WARMUP_STEPS)
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
