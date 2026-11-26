#!/usr/bin/env python3
"""Check imports, official package origins and GPU runtime without loading weights."""
import argparse
import ctypes
import importlib
import json
from pathlib import Path
import site
import sys


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    args = parser.parse_args()
    profile = Path(sys.prefix).name.removeprefix("object-detection-").removesuffix("-v2")
    require(not site.ENABLE_USER_SITE, "请设置 PYTHONNOUSERSITE=1，避免混入用户级包")
    require(profile in {"hf", "yolo", "yolov5", "yolov10", "yoloe", "yolov13",
                        "rfdetr", "tensorflow", "onnx", "wilddet3d"}, f"未知环境: {sys.prefix}")
    for name in ("numpy", "PIL.Image", "pycocotools.coco", "pycocotools.cocoeval"):
        importlib.import_module(name)
    import numpy as np
    require(np.__version__ == "1.26.4", f"NumPy 偏离环境约束: {np.__version__}，应为 1.26.4")
    if profile == "tensorflow":
        import tensorflow as tf
        gpu_count = len(tf.config.list_physical_devices("GPU"))
        require(gpu_count > 0, "TensorFlow 未发现 GPU；请检查本环境的 CUDA 12.2/cuDNN 8 运行库")
    elif profile == "onnx":
        import onnxruntime as ort
        from transformers import AutoImageProcessor
        # Provider 列表不证明共享库能加载，必须同时核查动态链接。
        ctypes.CDLL(str(Path(ort.__file__).parent / "capi/libonnxruntime_providers_cuda.so"))
        driver = ctypes.CDLL("libcuda.so.1")
        count = ctypes.c_int()
        require(driver.cuInit(0) == 0 and driver.cuDeviceGetCount(ctypes.byref(count)) == 0,
                "CUDA 驱动初始化失败")
        gpu_count = count.value
        require(gpu_count > 0 and "CUDAExecutionProvider" in ort.get_available_providers(),
                "ONNX Runtime 未发现 CUDA provider 或 GPU")
    else:
        import torch
        import torchvision
        array = np.zeros(1, dtype=np.float32)
        require(np.array_equal(torch.from_numpy(array).numpy(), array), "PyTorch / NumPy 数据转换失败")
        require(torch.cuda.is_available(), "PyTorch 未发现 GPU；请检查驱动及 CUDA_VISIBLE_DEVICES")
        gpu_count = torch.cuda.device_count()
        if profile == "hf":
            from transformers import AutoImageProcessor, AutoModelForObjectDetection
            import timm
            import pyarrow
            import fitz
        elif profile in {"yolo", "yolov10", "yoloe", "yolov13"}:
            import ultralytics
            if profile != "yolo":
                expected = (args.source_root / profile).resolve()
                require(Path(ultralytics.__file__).resolve().is_relative_to(expected),
                        f"错误的 Ultralytics 来源: {ultralytics.__file__}，应来自 {expected}")
            if profile == "yoloe":
                import clip
                import mobileclip
                import lvis
                from ultralytics.models.yolo.yoloe.val_pe_free import YOLOEPEFreeDetectValidator
                require(Path(lvis.__file__).resolve().is_relative_to(expected / "third_party/lvis-api"),
                        "YOLOE 必须使用官方仓库内的 LVIS Fixed evaluator")
        elif profile == "yolov5":
            import yolov5
        elif profile == "rfdetr":
            from rfdetr.detr import RFDETRMedium
            import pyarrow
        elif profile == "wilddet3d":
            import vis4d_cuda_ops
            require(hasattr(vis4d_cuda_ops, "iou_box3d"), "缺少 WildDet3D 必需的 CUDA 扩展接口")
            sys.path.insert(0, str(args.source_root / "WildDet3D"))
            from wilddet3d import build_model, preprocess
            importlib.import_module("configs.eval.in_the_wild.text")
    print(json.dumps({"status": "environment_ready", "profile": profile,
                      "python": sys.executable, "gpu_count": gpu_count}, ensure_ascii=False))


if __name__ == "__main__":
    main()
