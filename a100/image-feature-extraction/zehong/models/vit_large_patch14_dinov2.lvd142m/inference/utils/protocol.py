"""文件、样本身份和输入指纹；不依赖PyTorch，离线比较只需要NumPy。"""
import hashlib
import json
import re
from pathlib import Path
import numpy as np

SCHEMA = "raw-tensor-features-v4"


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def inside(root, relative):
    root = Path(root).resolve()
    path = (root / relative).resolve()
    if Path(relative).is_absolute() or not path.is_relative_to(root):
        raise ValueError(f"Path outside root: {relative}")
    return path


def read_manifest(path):
    rows, seen = [], set()
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        sid = row["sample_id"]
        if not isinstance(sid, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", sid):
            raise ValueError(f"Unsafe sample_id: {sid!r}")
        if sid in seen:
            raise ValueError(f"Duplicate sample_id: {sid}")
        if not isinstance(row["relative_path"], str):
            raise ValueError(f"Invalid image path: {sid}")
        inside(Path(path).parent, row["relative_path"])
        if not re.fullmatch(r"[0-9a-f]{64}", row["image_sha256"]):
            raise ValueError(f"Invalid image hash: {sid}")
        seen.add(sid)
        rows.append(row)
    if not rows:
        raise ValueError("Empty input manifest")
    return rows


def tensor_fingerprint(inputs):
    """每图、无batch维；保留原dtype和原始位，BF16也不先转FP32。"""
    import sys
    import torch  # 仅提取时调用；离线比较不需要安装PyTorch。
    digest = hashlib.sha256()
    for name in sorted(inputs):
        tensor = inputs[name].detach().cpu().contiguous()
        byte_array = tensor.reshape(-1).view(torch.uint8).numpy()
        if sys.byteorder == "big" and tensor.element_size() > 1:
            byte_array = byte_array.reshape(-1, tensor.element_size())[:, ::-1].copy()
        header = json.dumps({"name": name, "shape": list(tensor.shape), "dtype": str(tensor.dtype)},
                            sort_keys=True, separators=(",", ":")).encode()
        for payload in (header, byte_array.tobytes(order="C")):
            digest.update(len(payload).to_bytes(8, "little"))
            digest.update(payload)
    return digest.hexdigest()


def processor_protocol(processor):
    config = dict(processor.to_dict())
    for key in ("_name_or_path", "name_or_path", "_processor_class"):
        config.pop(key, None)
    return {"class": type(processor).__name__, "module": type(processor).__module__, "config": config,
            "decode": "Pillow open, first frame, convert RGB; no EXIF transpose",
            "input_hash_protocol": "named-tensors-original-dtype-le-v2"}


def checkpoint_fingerprints(directory):
    """权重及官方配置/代码的真实内容；GitHub单文件权重也可直接传入。"""
    root = Path(directory)
    if root.is_file():
        return {root.name: sha256_file(root)}
    weights = sorted(root.glob("*.safetensors")) or sorted(root.glob("*.bin")) or sorted(root.glob("*.pth")) or sorted(root.glob("*.pt"))
    if not weights:
        raise FileNotFoundError(f"No checkpoint weights in {root}")
    files = sorted({*weights, *root.glob("*.json"), *root.glob("*.py")})
    return {p.name: sha256_file(p) for p in files}
