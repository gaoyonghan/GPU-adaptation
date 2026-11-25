"""Ultralytics YOLOv8 classification checkpoint loading and preprocessing."""
from pathlib import Path

import torch

_ORIGINAL_TORCH_LOAD = torch.load


def _load_compat(*args, **kwargs):
    """8.0.x checkpoint 为完整 nn.Module pickle，torch>=2.6 默认 weights_only=True 需放开。"""
    kwargs.setdefault('weights_only', False)
    return _ORIGINAL_TORCH_LOAD(*args, **kwargs)


def load_checkpoint(weights):
    torch.load = _load_compat
    try:
        import ultralytics  # noqa: F401  checkpoint 的 pickle 需要该包的类
        ckpt = torch.load(str(Path(weights)), map_location='cpu')
    finally:
        torch.load = _ORIGINAL_TORCH_LOAD
    net = ckpt['model'].float().eval()
    return net, ckpt


def build_transform(imgsz):
    """官方分类推理预处理；新旧包路径都兼容，不自行改写 resize/crop/normalize。"""
    try:
        from ultralytics.yolo.data.augment import classify_transforms
    except ImportError:
        from ultralytics.data.augment import classify_transforms
    return classify_transforms(int(imgsz))


def resolve_device():
    if torch.cuda.is_available():
        return torch.device('cuda:0')
    npu = getattr(torch, 'npu', None)
    if npu is not None and npu.is_available():
        return torch.device('npu:0')
    raise RuntimeError('CUDA/NPU不可用，请检查环境；不自动改用CPU')
