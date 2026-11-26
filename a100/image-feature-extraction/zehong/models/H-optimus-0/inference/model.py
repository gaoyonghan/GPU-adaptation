"""NVIDIA入口：官方模型加载 → 结构/环境记录 → 全部图片提取。"""
import os
from datetime import datetime
from pathlib import Path
ROOT = Path(__file__).resolve().parent
GPU_ID = '0'
MODEL_ID = 'bioptimus/H-optimus-0'
BUNDLE_DIR = (Path(__file__).resolve().parents[5] / 'dataset' / 'pathology/crc_val_he_7k_bundle')
BATCH_SIZE = 64
NUM_WORKERS = 8
WARMUP_STEPS = 3
os.environ.setdefault('CUDA_VISIBLE_DEVICES', GPU_ID)
os.environ.setdefault('HF_ENDPOINT', 'https://hf-mirror.com')
os.environ.setdefault('HF_HUB_CACHE', str(Path(__file__).resolve().parents[1] / 'weights/.runtime_cache'))
import torch
from huggingface_hub import snapshot_download
import timm
from torchvision import transforms
from inference.utils.evaluate_model import evaluate_model
from inference.utils.print_model_info import print_model_and_environment
from inference.utils.protocol import checkpoint_fingerprints, processor_protocol

class PrepareBatch:
    """按 H-optimus-0 官方归一化方式处理并拼批。"""

    def __init__(self, transform):
        self.transform = transform

    def __call__(self, images):
        return torch.stack([self.transform(image) for image in images])

def forward_fn(model, inputs):
    return model(inputs)

def select_output(outputs):
    return outputs

def main():
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA不可用，请检查环境；不自动改用CPU')
    snapshot = snapshot_download(MODEL_ID, allow_patterns=['*.safetensors', '*.bin', '*.json', '*.py'])
    transform = transforms.Compose([transforms.Resize((224, 224), antialias=True), transforms.ToTensor(), transforms.Normalize(mean=(0.707223, 0.578729, 0.703617), std=(0.211883, 0.230117, 0.177517))])
    model = local_timm_create(snapshot, init_values=1e-05, dynamic_img_size=False)
    model.cuda().eval()
    output_dir = RUN_ARGS.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    print_model_and_environment(model, output_dir)
    summary = evaluate_model(model, PrepareBatch(transform), forward_fn, select_output, bundle_dir=BUNDLE_DIR, output_dir=output_dir, model_info={'source': 'huggingface', 'id': MODEL_ID, 'revision': Path(snapshot).name, 'checkpoint_files': checkpoint_fingerprints(snapshot)}, preprocessing_info={'class': 'torchvision.transforms.Compose', 'input_size': [224, 224], 'mean': [0.707223, 0.578729, 0.703617], 'std': [0.211883, 0.230117, 0.177517], 'decode': 'Pillow open, first frame, convert RGB; no EXIF transpose'}, output_name='image_features', batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, warmup_steps=WARMUP_STEPS, precision='amp-fp16')
    print('Results:', output_dir)
    print('Status:', summary['status'], 'images:', summary['num_images'])
    print('Performance:', summary['performance'])
    if summary['status'] != 'complete':
        raise RuntimeError('存在无效输出，请查看summary和feature_index')
RUN_ARGS = None

def snapshot_download(*args, **kwargs):
    return str(RUN_ARGS.weights.resolve())

def local_timm_create(snapshot, **overrides):
    import json, timm
    config = json.loads((Path(snapshot) / 'config.json').read_text())
    cfg = dict(config.get('pretrained_cfg', {}))
    for key in ['num_classes', 'label_names', 'label_descriptions']:
        if key in config:
            cfg[key] = config[key]
    model_args = dict(config.get('model_args', {}))
    model_args.update(overrides)
    files = [Path(snapshot) / 'model.safetensors', Path(snapshot) / 'pytorch_model.bin']
    weight = next((p for p in files if p.is_file()), None)
    if weight is None:
        raise FileNotFoundError('No local timm checkpoint')
    return timm.create_model(config['architecture'], pretrained=True, pretrained_cfg=cfg, pretrained_cfg_overlay={'source': '', 'file': str(weight.absolute()), 'custom_load': False}, **model_args)
if __name__ == '__main__':
    main()
