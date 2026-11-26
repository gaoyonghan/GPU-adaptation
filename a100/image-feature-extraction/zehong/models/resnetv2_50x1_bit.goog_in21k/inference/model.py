"""timm/resnetv2_50x1_bit.goog_in21k：官方timm加载 → 结构/环境记录 → 全量图片特征提取。"""
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
MODEL_ID = 'timm/resnetv2_50x1_bit.goog_in21k'
ARCH_ALIASES = {}
BUNDLE_DIR = Path(__file__).resolve().parents[5]/'dataset'/'aimv2_features'/'imagenet_validation_1000'
BATCH_SIZE = 64
NUM_WORKERS = 8
WARMUP_STEPS = 3
OUTPUT_NAME = 'forward_features'

os.environ.setdefault('CUDA_VISIBLE_DEVICES', '0')
os.environ.setdefault('HF_ENDPOINT', 'https://hf-mirror.com')
os.environ.setdefault('HF_HUB_CACHE', str(ROOT.parent / 'weights/.runtime_cache'))

import torch
try:
    import torch_npu  # noqa: F401  910B上注册npu后端；A100环境没有该包
except Exception:
    pass
import timm

from inference.utils.evaluate_model import evaluate_model
from inference.utils.print_model_info import print_model_and_environment
from inference.utils.protocol import checkpoint_fingerprints


class PrepareBatch:
    """timm官方transforms逐图处理并拼批；类在顶层以支持spawn worker。"""

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, images):
        tensors = [self.transforms(image) for image in images]
        return torch.stack(tensors, dim=0)


def forward_fn(model, inputs):
    """只跑到layer4输出，保留空间维度，不做全局池化。"""
    return model.forward_features(inputs)


def select_output(outputs):
    return outputs


def resolve_device():
    if torch.cuda.is_available():
        return torch.device('cuda:0')
    npu = getattr(torch, 'npu', None)
    if npu is not None and npu.is_available():
        return torch.device('npu:0')
    raise RuntimeError('CUDA/NPU不可用，请检查环境；不自动改用CPU')


def checkpoint_path(snapshot):
    for name in ('model.safetensors', 'pytorch_model.bin'):
        path = Path(snapshot) / name
        if path.is_file():
            return path
    raise FileNotFoundError(f'本地权重缺失，--weights需指向完整snapshot: {snapshot}')


def pretrained_cfg_from_config(config):
    """该仓库是新格式：config.json内直接带pretrained_cfg（tag=goog_in21k）。"""
    cfg = dict(config.get('pretrained_cfg') or {})
    if cfg:
        return cfg
    cfg = {key: config[key] for key in ('input_size', 'mean', 'std', 'interpolation',
                                        'crop_pct', 'first_conv', 'classifier', 'num_classes')
           if key in config}
    cfg.update({'tag': 'goog_in21k', 'crop_mode': 'center'})
    return cfg


def create_model(snapshot):
    config = json.loads((Path(snapshot) / 'config.json').read_text())
    arch = ARCH_ALIASES.get(config['architecture'], config['architecture'])
    model_args = dict(config.get('model_args') or {})
    weight = checkpoint_path(snapshot)
    return timm.create_model(
        arch, pretrained=True, pretrained_cfg=pretrained_cfg_from_config(config),
        pretrained_cfg_overlay={'source': '', 'file': str(weight.absolute()), 'custom_load': False},
        **model_args,
    )


def checkpoint_revision(path):
    import re
    path = Path(path)
    if re.fullmatch(r'[0-9a-f]{40}', path.name):
        return path.name
    config = path / 'config.json'
    if config.is_file():
        value = json.loads(config.read_text()).get('_commit_hash')
        if isinstance(value, str) and re.fullmatch(r'[0-9a-f]{40}', value):
            return value
    return None


def main():
    device = resolve_device()
    model = create_model(RUN_ARGS.weights)
    model.to(device).eval()
    data_config = timm.data.resolve_model_data_config(model)
    transforms = timm.data.create_transform(**data_config, is_training=False)
    output_dir = RUN_ARGS.output_dir.absolute()
    output_dir.mkdir(parents=True, exist_ok=False)
    print_model_and_environment(model, output_dir)
    summary = evaluate_model(
        model, PrepareBatch(transforms), forward_fn, select_output,
        bundle_dir=BUNDLE_DIR, output_dir=output_dir,
        model_info={'source': 'huggingface', 'id': MODEL_ID,
                    'revision': checkpoint_revision(RUN_ARGS.weights),
                    'checkpoint_files': checkpoint_fingerprints(RUN_ARGS.weights)},
        preprocessing_info=data_config, output_name=OUTPUT_NAME,
        batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, warmup_steps=WARMUP_STEPS)
    print('Results:', output_dir)
    print('Status:', summary['status'], 'images:', summary['num_images'])
    print('Performance:', summary['performance'])
    if summary['status'] != 'complete':
        raise RuntimeError('存在无效输出，请查看summary和feature_index')


RUN_ARGS = None
