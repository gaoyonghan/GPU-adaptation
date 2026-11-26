"""结构只写TXT，环境只写JSON；在正式计时之前调用。"""
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path

import torch
from .protocol import write_json


def version(name):
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def command(args):
    try:
        result = subprocess.run(args, capture_output=True, text=True, timeout=15)
        return {"exit_code": result.returncode, "output": (result.stdout + result.stderr).strip()}
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"error": str(exc)}


def cpu_quota():
    for name in ("/sys/fs/cgroup/cpu.max", "/sys/fs/cgroup/cpu/cpu.cfs_quota_us", "/sys/fs/cgroup/cpu/cpu.cfs_period_us"):
        path = Path(name)
        if path.is_file():
            yield name, path.read_text().strip()


def print_model_and_environment(model, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for name in ("model_structure.txt", "model_environment.json"):
        if (output_dir / name).exists():
            raise FileExistsError(output_dir / name)
    parameters = list(model.parameters())
    device = parameters[0].device
    model_info = {"class": f"{type(model).__module__}.{type(model).__name__}",
                  "parameter_count": sum(p.numel() for p in parameters),
                  "parameter_bytes": sum(p.numel() * p.element_size() for p in parameters),
                  "parameter_devices": sorted({str(p.device) for p in parameters}),
                  "parameter_dtypes": sorted({str(p.dtype) for p in parameters}),
                  "training": model.training}
    packages = ("torch", "torchvision", "torch_npu", "transformers", "timm", "numpy", "Pillow",
                "safetensors", "huggingface-hub", "tqdm")
    if device.type == "cuda":
        math = {"matmul_reduced_precision": bool(torch.backends.cuda.matmul.allow_tf32),
                "conv_reduced_precision": bool(torch.backends.cudnn.allow_tf32)}
    elif device.type == "npu":
        math = {"matmul_reduced_precision": bool(torch.npu.matmul.allow_hf32),
                "conv_reduced_precision": bool(torch.npu.conv.allow_hf32)}
    else:
        math = {"matmul_reduced_precision": False, "conv_reduced_precision": False}
    environment = {"collected_at": datetime.now(timezone.utc).isoformat(),
                   "python": {"executable": sys.executable, "version": sys.version},
                   "system": {"platform": platform.platform(), "hostname": platform.node(),
                              "cpu_count": os.cpu_count(), "cpu_quota": dict(cpu_quota())},
                   "packages": {name: version(name) for name in packages},
                   "device": str(device), "math": math,
                   "torch_num_threads": torch.get_num_threads()}
    if device.type == "cuda":
        props = torch.cuda.get_device_properties(device)
        environment["cuda"] = {"name": props.name, "uuid": str(getattr(props, "uuid", "")),
                               "total_memory_bytes": props.total_memory, "runtime": torch.version.cuda,
                               "cudnn": torch.backends.cudnn.version(),
                               "visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES")}
        environment["device_status"] = command(["nvidia-smi", "--query-gpu=index,name,uuid,driver_version,memory.used,utilization.gpu", "--format=csv"])
    elif device.type == "npu":
        environment["npu"] = {"name": torch.npu.get_device_name(device),
                              "ASCEND_HOME_PATH": os.environ.get("ASCEND_HOME_PATH"),
                              "ASCEND_RT_VISIBLE_DEVICES": os.environ.get("ASCEND_RT_VISIBLE_DEVICES")}
        environment["device_status"] = command(["npu-smi", "info"])
        home = Path(os.environ.get("ASCEND_HOME_PATH", "/usr/local/Ascend/ascend-toolkit/latest"))
        version_files = [home / "version.cfg", home / "version.info", home.parent / "version.cfg"]
        environment["npu"]["cann_version_files"] = {str(p): p.read_text(errors="replace")[:4096] for p in version_files if p.is_file()}
    (output_dir / "model_structure.txt").write_text(
        f"Class: {model_info['class']}\nParameters: {model_info['parameter_count']:,}\n\n{model}\n", encoding="utf-8")
    write_json(output_dir / "model_environment.json", {"model": model_info, "environment": environment})
    print(f"Model: {model_info['class']}; parameters: {model_info['parameter_count']:,}; device: {device}")
    print("Structure/environment:", output_dir)
