#!/usr/bin/env python3
"""在本地 Stanford Dogs test split 上评测 skyau/dog-breed-classifier-vit。"""

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
    _test_record = _start_records(107, __file__, globals())


import json
import os
import time
from datetime import datetime
from pathlib import Path

import torch
from datasets import Dataset as HFDataset, DatasetDict, load_from_disk, load_dataset
from torch.utils.data import DataLoader, Dataset
from transformers import AutoImageProcessor, AutoModelForImageClassification


MODEL_ID = "skyau/dog-breed-classifier-vit"
PROJECT_ROOT = _PACKAGE_ROOT
MODEL_CACHE_DIR = _WEIGHTS_ROOT
DATASET_DIR = _DATASET_ROOT/'stanford-dogs'
RESULTS_PATH = _OUTPUT_ROOT/_portable_os.environ.get('MODEL_PHASE','full')/'m107/metrics.json'
if __name__ == "__main__":
    RESULTS_PATH = _test_record.output / "metrics.json"
BATCH_SIZE = int(_portable_os.environ.get('MODEL_BATCH_SIZE',128))
NUM_WORKERS = int(_portable_os.environ.get('MODEL_NUM_WORKERS',4))
WARMUP_STEPS = _warmup_steps(10)
NUM_CLASSES = 120
LABEL_ALIASES = {
    "shih_tzu": "shih",
    "black_and_tan_coonhound": "black",
    "wire_haired_fox_terrier": "wire",
    "soft_coated_wheaten_terrier": "soft",
    "flat_coated_retriever": "flat",
    "curly_coated_retriever": "curly",
    "german_short_haired_pointer": "german_short",
}


def normalize_breed_name(name):
    """统一 Stanford Dogs 的 synset 名称与模型 config 中的犬种名称。"""
    value = str(name).strip().lower()
    prefix, separator, breed_name = value.partition("-")
    if separator and prefix.startswith("n") and prefix[1:].isdigit():
        value = breed_name
    value = value.replace("-", "_").replace(" ", "_")
    return LABEL_ALIASES.get(value, value)


def build_label_mapping(dataset, model):
    """返回 dataset label id -> model logits id 的映射。"""
    if "label" not in dataset.features or not hasattr(dataset.features["label"], "names"):
        raise ValueError("Stanford Dogs 数据必须提供带 names 的 label ClassLabel")

    model_label_to_id = {}
    for model_id, model_name in model.config.id2label.items():
        normalized_name = normalize_breed_name(model_name)
        if normalized_name in model_label_to_id:
            raise ValueError(f"模型标签存在重复名称，无法映射：{model_name}")
        model_label_to_id[normalized_name] = int(model_id)

    mapping = []
    missing = []
    for dataset_name in dataset.features["label"].names:
        normalized_name = normalize_breed_name(dataset_name)
        if normalized_name not in model_label_to_id:
            missing.append(dataset_name)
        else:
            mapping.append(model_label_to_id[normalized_name])
    if missing:
        raise ValueError(f"模型缺少 Stanford Dogs 类别：{missing}")
    if len(mapping) != len(set(mapping)):
        raise ValueError("数据集类别映射到了重复的模型输出类别")
    return mapping


def load_test_split():
    """兼容三种本地布局：save_to_disk / parquet / imagefolder。"""
    if not DATASET_DIR.is_dir():
        raise FileNotFoundError(f"未找到本地 Stanford Dogs 数据：{DATASET_DIR}")

    # 1) save_to_disk 格式：目录下有 dataset_info.json
    if (DATASET_DIR / "dataset_info.json").is_file():
        loaded = load_from_disk(str(DATASET_DIR))
        if isinstance(loaded, DatasetDict):
            if "test" not in loaded:
                raise ValueError(f"数据集没有 test split，可用 split：{list(loaded)}")
            loaded = loaded["test"]
        return loaded

    # 2) parquet 格式：包含 test-*.parquet
    parquet_files = sorted(DATASET_DIR.rglob("test-*.parquet"))
    if parquet_files:
        loaded = HFDataset.from_parquet(
            [str(p) for p in parquet_files],
            cache_dir=_portable_os.environ["HF_DATASETS_CACHE"],
        )
        return loaded

    # 3) imagefolder 格式：按类别组织图片目录
    test_dir = DATASET_DIR / "test"
    if not test_dir.is_dir():
        test_dir = DATASET_DIR
    loaded = load_dataset("imagefolder", data_dir=str(test_dir), split="train")
    if "image" not in loaded.column_names or "label" not in loaded.column_names:
        raise ValueError(f"数据列应包含 image 和 label，实际为：{loaded.column_names}")
    return loaded


class DogBreedDataset(Dataset):
    def __init__(self, dataset, processor, label_mapping):
        self.dataset = dataset
        self.processor = processor
        self.label_mapping = label_mapping

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        example = self.dataset[index]
        image = example["image"].convert("RGB")
        pixel_values = self.processor(image, return_tensors="pt")["pixel_values"].squeeze(0)
        dataset_label = int(example["label"])
        return pixel_values, self.label_mapping[dataset_label]


def save_result(result):
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = RESULTS_PATH.with_suffix(".json.tmp")
    temporary_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    os.replace(temporary_path, RESULTS_PATH)


def main():
    device = torch.device(_select_torch_device(torch, 'cuda' if torch.cuda.is_available() else 'cpu'))
    print(f"使用设备: {torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'}")
    print(f"模型: {MODEL_ID}")
    print("数据集: 本地 Stanford Dogs test split")

    processor = AutoImageProcessor.from_pretrained(
        str(MODEL_CACHE_DIR), local_files_only=True
    )
    model = AutoModelForImageClassification.from_pretrained(
        str(MODEL_CACHE_DIR), local_files_only=True
    ).to(device)
    model.eval()
    if model.config.num_labels != NUM_CLASSES:
        raise ValueError(
            f"模型分类头应输出 {NUM_CLASSES} 类，实际为 {model.config.num_labels} 类"
        )
    parameter_count_millions = sum(p.numel() for p in model.parameters()) / 1e6

    dataset = load_test_split()
    label_mapping = build_label_mapping(dataset, model)
    print(f"标签映射: {len(label_mapping)} 个 Stanford Dogs 类别已按犬种名称对齐")
    wrapped_dataset = DogBreedDataset(dataset, processor, label_mapping)
    dataloader = _test_record.loader(DataLoader, 
        wrapped_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=device.type == "cuda",
        persistent_workers=NUM_WORKERS > 0,
    )
    print(f"测试图片数量: {len(wrapped_dataset)}")
    print(f"batch size: {BATCH_SIZE}")
    print(f"预处理: {processor.__class__.__name__}")

    warmup_pixel_values, _ = next(iter(dataloader))
    warmup_pixel_values = warmup_pixel_values[:BATCH_SIZE].to(device, non_blocking=True)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.cuda.reset_peak_memory_stats(device)
    with torch.inference_mode():
        for _ in range(WARMUP_STEPS):
            model(pixel_values=warmup_pixel_values)
    if device.type == "cuda":
        torch.cuda.synchronize()
    del warmup_pixel_values

    correct_top1 = 0
    correct_top5 = 0
    forward_seconds = 0.0
    wall_start = time.perf_counter()
    with torch.inference_mode():
        for batch_index, (pixel_values, labels) in enumerate(dataloader, start=1):
            _test_record.before_batch(locals())
            pixel_values = pixel_values.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            if device.type == "cuda":
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)
                start_event.record()
                logits = model(pixel_values=pixel_values).logits
                end_event.record()
                torch.cuda.synchronize()
                forward_seconds += start_event.elapsed_time(end_event) / 1000.0
            else:
                forward_start = time.perf_counter()
                logits = model(pixel_values=pixel_values).logits
                forward_seconds += time.perf_counter() - forward_start

            top5 = logits.topk(k=5, dim=1).indices
            correct_top1 += top5[:, 0].eq(labels).sum().item()
            correct_top5 += top5.eq(labels.unsqueeze(1)).any(dim=1).sum().item()
            if batch_index % 10 == 0:
                print(
                    f"已处理: {min(batch_index * BATCH_SIZE, len(wrapped_dataset))}"
                    f"/{len(wrapped_dataset)}"
                )
            _test_record.capture(locals())
    wall_seconds = time.perf_counter() - wall_start

    image_count = len(wrapped_dataset)
    top1_pct = correct_top1 / image_count * 100.0
    top5_pct = correct_top5 / image_count * 100.0
    result = {
        "recorded_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "model_id": MODEL_ID,
        "model_path": str(MODEL_CACHE_DIR),
        "script_path": str(Path(__file__).resolve()),
        "model": {
            "id": MODEL_ID,
            "parameter_count_millions": round(parameter_count_millions, 4),
            "num_labels": NUM_CLASSES,
        },
        "dataset": {
            "name": "Stanford Dogs",
            "split": "test",
            "path": str(DATASET_DIR),
            "image_count": image_count,
            "class_count": NUM_CLASSES,
            "label_mapping": "按犬种名称对齐 dataset ClassLabel 与 model config.id2label",
        },
        "environment": {
            "device": torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU",
            "precision": "FP32",
            "batch_size": BATCH_SIZE,
            "warmup_steps": WARMUP_STEPS,
            "num_workers": NUM_WORKERS,
        },
        "preprocessing": {
            "processor": processor.__class__.__name__,
        },
        "accuracy": {
            "image_count": image_count,
            "top1_pct": round(top1_pct, 4),
            "top5_pct": round(top5_pct, 4),
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
    print(f"模型平均推理时间: {result['timing']['model_latency_ms_per_image']:.4f} ms/image")
    print(f"模型吞吐量: {result['timing']['model_throughput_images_per_second']:.2f} images/s")
    print(f"评测结果已更新: {RESULTS_PATH}")
    _test_record.finish(locals())


if __name__ == "__main__":
    main()
