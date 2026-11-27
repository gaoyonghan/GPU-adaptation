#!/usr/bin/env python3
"""在本地 CIFAR-10 test split 上评测 edadaltocg/resnet18_cifar10。"""

import os as _portable_os
from common import configure_native, select_torch_device as _select_torch_device, tensorflow_gpus as _tensorflow_gpus, onnx_providers as _onnx_providers, verify_onnx_session as _verify_onnx_session
if __name__ == "__main__":
    _runtime_options = configure_native()
from common import PROJECT_ROOT as _PACKAGE_ROOT, WEIGHTS_ROOT as _WEIGHTS_ROOT, DATASET_ROOT as _DATASET_ROOT, OUTPUT_ROOT as _OUTPUT_ROOT
from timing import warmup_steps as _warmup_steps


# 原脚本直接保存逐样本结果，不新增推理入口或模型前向。
if __name__ == "__main__":
    import sys as _records_sys
    from pathlib import Path as _RecordsPath
    _records_sys.path.insert(0, str(_RecordsPath(__file__).resolve().parents[0]))
    from common import start as _start_records
    _test_record = _start_records(110, __file__, globals())


import json
import pickle
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.models import resnet18


MODEL_ID = "edadaltocg/resnet18_cifar10"
PROJECT_ROOT = _PACKAGE_ROOT
MODEL_DIR = _WEIGHTS_ROOT
WEIGHTS_PATH = MODEL_DIR / "pytorch_model.bin"
DATASET_DIR = _DATASET_ROOT/'CIFAR-10'
TEST_BATCH_PATH = DATASET_DIR / "test_batch"
META_PATH = DATASET_DIR / "batches.meta"
RESULTS_PATH = _OUTPUT_ROOT/_portable_os.environ.get('MODEL_PHASE','full')/'m110/metrics.json'
if __name__ == "__main__":
    RESULTS_PATH = _test_record.output / "metrics.json"

BATCH_SIZE = int(_portable_os.environ.get('MODEL_BATCH_SIZE',256))
NUM_WORKERS = int(_portable_os.environ.get('MODEL_NUM_WORKERS',4))
WARMUP_STEPS = _warmup_steps(10)
MEAN = np.asarray([0.4914, 0.4822, 0.4465], dtype=np.float32)
STD = np.asarray([0.2023, 0.1994, 0.2010], dtype=np.float32)


def read_cifar10_test():
    if not TEST_BATCH_PATH.is_file():
        raise FileNotFoundError(f"未找到 CIFAR-10 test 文件：{TEST_BATCH_PATH}")
    with TEST_BATCH_PATH.open("rb") as file:
        batch = pickle.load(file, encoding="bytes")
    images = np.asarray(batch[b"data"], dtype=np.uint8).reshape(-1, 3, 32, 32)
    labels = np.asarray(batch[b"labels"], dtype=np.int64)
    if images.shape != (10000, 3, 32, 32) or labels.shape != (10000,):
        raise ValueError(
            f"CIFAR-10 test 数据应为 10000 张 3x32x32 图像，实际为 {images.shape}、{labels.shape}"
        )

    class_names = [str(index) for index in range(10)]
    if META_PATH.is_file():
        with META_PATH.open("rb") as file:
            metadata = pickle.load(file, encoding="bytes")
        names = metadata.get(b"label_names", [])
        class_names = [name.decode("utf-8") if isinstance(name, bytes) else str(name) for name in names]
        if len(class_names) != 10:
            class_names = [str(index) for index in range(10)]

    images = images.astype(np.float32) / 255.0
    images = (images - MEAN[None, :, None, None]) / STD[None, :, None, None]
    return torch.from_numpy(images), torch.from_numpy(labels), class_names


class CIFAR10TensorDataset(Dataset):
    def __init__(self, images, labels):
        self.images = images
        self.labels = labels

    def __len__(self):
        return self.labels.shape[0]

    def __getitem__(self, index):
        return self.images[index], self.labels[index]


def load_model():
    if not WEIGHTS_PATH.is_file():
        raise FileNotFoundError(f"未找到模型权重：{WEIGHTS_PATH}")
    model = resnet18(num_classes=10)
    # 该模型按 CIFAR-10 的 32x32 图像训练，首层和 ImageNet ResNet-18 不同。
    model.conv1 = torch.nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = torch.nn.Identity()
    state_dict = torch.load(WEIGHTS_PATH, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict, strict=True)
    return model


def save_result(result):
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = RESULTS_PATH.with_suffix(".json.tmp")
    temporary_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary_path.replace(RESULTS_PATH)


def main():
    device = torch.device(_select_torch_device(torch, 'cuda' if torch.cuda.is_available() else 'cpu'))
    device_name = torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU"
    print(f"使用设备: {device_name}")
    print(f"模型: {MODEL_ID}")
    print("数据集: CIFAR-10 test split")

    images, labels, class_names = read_cifar10_test()
    dataset = CIFAR10TensorDataset(images, labels)
    dataloader = _test_record.loader(DataLoader, 
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=device.type == "cuda",
        persistent_workers=NUM_WORKERS > 0,
    )
    model = load_model().to(device)
    model.eval()
    parameter_count_millions = sum(parameter.numel() for parameter in model.parameters()) / 1e6
    print(f"测试图片数量: {len(dataset)}")
    print(f"模型参数量: {parameter_count_millions:.4f} M")
    print(f"batch size: {BATCH_SIZE}")
    print(f"类别: {', '.join(class_names)}")
    print("预处理: 32x32 RGB，除以 255 后按模型配置标准化")

    warmup_pixel_values, _ = next(iter(dataloader))
    warmup_pixel_values = warmup_pixel_values.to(device, non_blocking=True)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.cuda.reset_peak_memory_stats(device)
    with torch.inference_mode():
        for _ in range(WARMUP_STEPS):
            model(warmup_pixel_values)
    if device.type == "cuda":
        torch.cuda.synchronize()
    del warmup_pixel_values

    correct_top1 = 0
    correct_top5 = 0
    class_correct = np.zeros(10, dtype=np.int64)
    class_total = np.zeros(10, dtype=np.int64)
    forward_seconds = 0.0
    wall_start = time.perf_counter()
    with torch.inference_mode():
        for batch_index, (pixel_values, batch_labels) in enumerate(dataloader, start=1):
            _test_record.before_batch(locals())
            pixel_values = pixel_values.to(device, non_blocking=True)
            batch_labels = batch_labels.to(device, non_blocking=True)
            if device.type == "cuda":
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)
                start_event.record()
                logits = model(pixel_values)
                end_event.record()
                torch.cuda.synchronize()
                forward_seconds += start_event.elapsed_time(end_event) / 1000.0
            else:
                forward_start = time.perf_counter()
                logits = model(pixel_values)
                forward_seconds += time.perf_counter() - forward_start

            top5 = logits.topk(k=5, dim=1).indices
            predictions = top5[:, 0]
            correct_top1 += predictions.eq(batch_labels).sum().item()
            correct_top5 += top5.eq(batch_labels.unsqueeze(1)).any(dim=1).sum().item()
            for class_index in range(10):
                mask = batch_labels == class_index
                class_total[class_index] += mask.sum().item()
                class_correct[class_index] += (predictions[mask] == class_index).sum().item()
            if batch_index % 10 == 0:
                print(f"已处理: {min(batch_index * BATCH_SIZE, len(dataset))}/{len(dataset)}")
            _test_record.capture(locals())
    wall_seconds = time.perf_counter() - wall_start

    image_count = len(dataset)
    top1_pct = correct_top1 / image_count * 100.0
    top5_pct = correct_top5 / image_count * 100.0
    per_class_accuracy = {
        class_names[index]: round(float(class_correct[index] / class_total[index] * 100.0), 4)
        for index in range(10)
    }
    result = {
        "recorded_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "model_id": MODEL_ID,
        "model_path": str(MODEL_DIR),
        "weights_path": str(WEIGHTS_PATH),
        "script_path": str(Path(__file__).resolve()),
        "model": {
            "id": MODEL_ID,
            "architecture": "ResNet-18 (CIFAR-10 stem)",
            "parameter_count_millions": round(float(parameter_count_millions), 4),
            "num_labels": 10,
        },
        "dataset": {
            "name": "CIFAR-10",
            "split": "test",
            "path": str(DATASET_DIR),
            "image_count": image_count,
            "class_count": 10,
            "class_names": class_names,
        },
        "official": {
            "source": "Hugging Face model card",
            "test_accuracy_pct": 94.98,
        },
        "environment": {
            "device": device_name,
            "precision": "FP32",
            "batch_size": BATCH_SIZE,
            "warmup_steps": WARMUP_STEPS,
            "num_workers": NUM_WORKERS,
        },
        "preprocessing": {
            "input_size": [3, 32, 32],
            "mean": MEAN.tolist(),
            "std": STD.tolist(),
        },
        "accuracy": {
            "image_count": image_count,
            "top1_pct": round(top1_pct, 4),
            "top5_pct": round(top5_pct, 4),
            "per_class_top1_pct": per_class_accuracy,
            "macro_top1_pct": round(float(np.mean(list(per_class_accuracy.values()))), 4),
        },
        "timing": {
            "model_latency_ms_per_image": round(forward_seconds / image_count * 1000.0, 4),
            "model_throughput_images_per_second": round(image_count / forward_seconds, 4),
            "end_to_end_latency_ms_per_image": round(wall_seconds / image_count * 1000.0, 4),
            "end_to_end_throughput_images_per_second": round(image_count / wall_seconds, 4),
        },
    }
    if device.type == "cuda":
        result["memory"] = {
            "peak_allocated_gib": round(torch.cuda.max_memory_allocated(device) / 2**30, 4),
            "peak_reserved_gib": round(torch.cuda.max_memory_reserved(device) / 2**30, 4),
        }
    save_result(result)
    print("=" * 60)
    print(f"Top-1 accuracy: {top1_pct:.2f}%")
    print(f"Top-5 accuracy: {top5_pct:.2f}%")
    print(f"Macro per-class accuracy: {result['accuracy']['macro_top1_pct']:.2f}%")
    print(f"模型平均推理时间: {result['timing']['model_latency_ms_per_image']:.4f} ms/image")
    print(f"模型吞吐量: {result['timing']['model_throughput_images_per_second']:.2f} images/s")
    print(f"评测结果已更新: {RESULTS_PATH}")
    _test_record.finish(locals())


if __name__ == "__main__":
    main()
