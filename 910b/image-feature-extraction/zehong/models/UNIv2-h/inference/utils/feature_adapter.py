"""输入/输出的轻量检查；不假设必须是patch，也不剔除CLS/register。"""
from collections.abc import Mapping
import torch


def named_inputs(inputs):
    if isinstance(inputs, torch.Tensor):
        # timm等单Tensor输入也有稳定名字，以便两端计算相同指纹。
        return {"images": inputs}
    if isinstance(inputs, Mapping) and inputs and all(isinstance(v, torch.Tensor) for v in inputs.values()):
        return dict(inputs)
    raise TypeError("prepare_batch must return a Tensor or nonempty dict[str, Tensor]")


def select_field(outputs, key):
    """默认字段读取；裸Tensor/tuple/中间层请替换start.py的select_output。"""
    value = outputs
    for part in key.split("."):
        value = value[part] if isinstance(value, Mapping) else getattr(value, part)
    return value


def describe_output(value, batch_size, device):
    if not isinstance(value, torch.Tensor) or not value.is_floating_point():
        raise TypeError("select_output must return a floating-point Tensor")
    if value.ndim < 2 or value.shape[0] != batch_size or any(n <= 0 for n in value.shape):
        raise ValueError("Output needs batch first: [B,D], [B,N,D] or [B,C,H,W]")
    if value.device != device:
        raise RuntimeError(f"Selected output is on {value.device}, expected {device}")
    # NumPy没有原生BF16；转FP32可完整保留BF16数值，其余常用浮点类型原样保存。
    storage = {torch.float16: "float16", torch.bfloat16: "float32",
               torch.float32: "float32", torch.float64: "float64"}
    if value.dtype not in storage:
        raise TypeError(f"Unsupported output dtype for NPY storage: {value.dtype}")
    return {"shape": list(value.shape[1:]), "output_dtype": str(value.dtype),
            "saved_dtype": storage[value.dtype]}
