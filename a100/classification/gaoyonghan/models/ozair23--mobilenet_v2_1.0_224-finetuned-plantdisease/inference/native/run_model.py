#!/usr/bin/env python3
"""在 PlantVillage color test split 上评测 ozair23/mobilenet_v2_1.0_224-finetuned-plantdisease。"""

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
    _test_record = _start_records(145, __file__, globals())


import io
import json
import os
import time
import zipfile
from datetime import datetime
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from transformers import AutoImageProcessor, AutoModelForImageClassification

from common import (
    balanced_accuracy_from_confusion,
)


MODEL_ID = "ozair23/mobilenet_v2_1.0_224-finetuned-plantdisease"
PROJECT_ROOT = _PACKAGE_ROOT
MODEL_CACHE_DIR = _WEIGHTS_ROOT
DATASET_DIR = _DATASET_ROOT/'PlantVillage'
DATA_ZIP = DATASET_DIR / "data.zip"
TEST_LIST = DATASET_DIR / "splits/color_test.txt"
RESULTS_PATH = _OUTPUT_ROOT/_portable_os.environ.get('MODEL_PHASE','full')/'m145/metrics.json'
if __name__ == "__main__":
    RESULTS_PATH = _test_record.output / "metrics.json"
NUM_CLASSES = 38
BATCH_SIZE = int(_portable_os.environ.get('MODEL_BATCH_SIZE',128))
NUM_WORKERS = int(_portable_os.environ.get('MODEL_NUM_WORKERS',0))
WARMUP_STEPS = _warmup_steps(3)


def load_test_records():
    if not DATA_ZIP.is_file():
        raise FileNotFoundError(f"未找到 PlantVillage 压缩包：{DATA_ZIP}")
    if not TEST_LIST.is_file():
        raise FileNotFoundError(f"未找到 PlantVillage color test 列表：{TEST_LIST}")

    records = []
    with TEST_LIST.open(encoding="utf-8") as handle:
        for line in handle:
            archive_path = line.strip()
            if not archive_path:
                continue
            parts = archive_path.split("/")
            if len(parts) < 4 or parts[0:2] != ["raw", "color"]:
                raise ValueError(f"异常的 PlantVillage 路径：{archive_path}")
            records.append((archive_path, parts[2]))
    if len(records) != 10709:
        raise ValueError(f"PlantVillage color test 应有 10709 张图片，实际为 {len(records)} 张")
    return records


class PlantVillageTestDataset(Dataset):
    def __init__(self, records, processor, archive):
        self.records = records
        self.processor = processor
        self.archive = archive

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        archive_path, class_name = self.records[index]
        image_bytes = self.archive.read(archive_path)
        with Image.open(io.BytesIO(image_bytes)) as image:
            pixel_values = self.processor(
                images=image.convert("RGB"), return_tensors="pt"
            )["pixel_values"].squeeze(0)
        return pixel_values, class_name


def calculate_metrics(confusion, top1_correct, top5_correct, class_names):
    true_positive = confusion.diag().to(torch.float64)
    support = confusion.sum(dim=1).to(torch.float64)
    predicted = confusion.sum(dim=0).to(torch.float64)
    precision = torch.where(
        predicted > 0, true_positive / predicted, torch.zeros_like(predicted)
    )
    recall = torch.where(
        support > 0, true_positive / support, torch.zeros_like(support)
    )
    f1 = torch.where(
        precision + recall > 0,
        2 * precision * recall / (precision + recall),
        torch.zeros_like(precision),
    )
    image_count = int(support.sum().item())
    per_class = {}
    for index, name in enumerate(class_names):
        per_class[name] = {
            "support": int(support[index].item()),
            "precision_pct": round(precision[index].item() * 100.0, 4),
            "recall_pct": round(recall[index].item() * 100.0, 4),
            "f1_pct": round(f1[index].item() * 100.0, 4),
        }

    off_diagonal = confusion.clone()
    off_diagonal.fill_diagonal_(0)
    flattened = off_diagonal.flatten()
    pairs = []
    for flat_index in torch.argsort(flattened, descending=True).tolist():
        count = int(flattened[flat_index].item())
        if count == 0 or len(pairs) >= 20:
            break
        true_index = flat_index // NUM_CLASSES
        predicted_index = flat_index % NUM_CLASSES
        pairs.append(
            {
                "true": class_names[true_index],
                "predicted": class_names[predicted_index],
                "count": count,
            }
        )

    return {
        "image_count": image_count,
        "top1_accuracy_pct": round(top1_correct / image_count * 100.0, 4),
        "top5_accuracy_pct": round(top5_correct / image_count * 100.0, 4),
        "macro_precision_pct": round(precision.mean().item() * 100.0, 4),
        "macro_recall_pct": round(recall.mean().item() * 100.0, 4),
        "balanced_accuracy_pct": round(100.0 * balanced_accuracy_from_confusion(confusion), 4),
        "macro_f1_pct": round(f1.mean().item() * 100.0, 4),
        "per_class": per_class,
        "top_confusion_pairs": pairs,
    }


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
    print("数据集: PlantVillage color test split")

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

    records = load_test_records()
    model_class_names = [
        model.config.id2label[index]
        if index in model.config.id2label
        else model.config.id2label[str(index)]
        for index in range(NUM_CLASSES)
    ]
    record_class_names = sorted({class_name for _path, class_name in records})
    if set(record_class_names) != set(model_class_names):
        missing = sorted(set(record_class_names) - set(model_class_names))
        extra = sorted(set(model_class_names) - set(record_class_names))
        raise ValueError(f"标签不匹配，数据集多出 {missing}，模型多出 {extra}")
    label_to_id = {name: index for index, name in enumerate(model_class_names)}

    archive = zipfile.ZipFile(DATA_ZIP, "r")
    archive_names = set(archive.namelist())
    missing_files = [path for path, _class_name in records if path not in archive_names]
    if missing_files:
        archive.close()
        raise FileNotFoundError(f"压缩包缺少测试图片，例如：{missing_files[:3]}")
    dataset = PlantVillageTestDataset(records, processor, archive)
    dataloader = _test_record.loader(DataLoader, 
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=device.type == "cuda",
    )
    print(f"测试图片数量: {len(dataset)}")
    print(f"类别数量: {NUM_CLASSES}")
    print(f"标签映射: 使用模型 config.id2label 的 PlantVillage 类别顺序")
    print(f"batch size: {BATCH_SIZE}")

    warmup_pixel_values, _class_names = next(iter(dataloader))
    warmup_pixel_values = warmup_pixel_values.to(device, non_blocking=True)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.cuda.reset_peak_memory_stats(device)
    with torch.inference_mode():
        for _ in range(WARMUP_STEPS):
            model(pixel_values=warmup_pixel_values)
    if device.type == "cuda":
        torch.cuda.synchronize()
    del warmup_pixel_values

    confusion = torch.zeros((NUM_CLASSES, NUM_CLASSES), dtype=torch.long)
    top1_correct = 0
    top5_correct = 0
    forward_seconds = 0.0
    wall_start = time.perf_counter()
    with torch.inference_mode():
        for batch_index, (pixel_values, batch_class_names) in enumerate(dataloader, start=1):
            _test_record.before_batch(locals())
            labels = torch.tensor(
                [label_to_id[name] for name in batch_class_names],
                dtype=torch.long,
                device=device,
            )
            pixel_values = pixel_values.to(device, non_blocking=True)
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
            top1_correct += top5[:, 0].eq(labels).sum().item()
            top5_correct += top5.eq(labels.unsqueeze(1)).any(dim=1).sum().item()
            for true_label, predicted_label in zip(
                labels.cpu().tolist(), top5[:, 0].cpu().tolist()
            ):
                confusion[true_label, predicted_label] += 1
            if batch_index % 25 == 0:
                print(f"已处理: {min(batch_index * BATCH_SIZE, len(dataset))}/{len(dataset)}")
            _test_record.capture(locals())
    if device.type == "cuda":
        torch.cuda.synchronize()
    wall_seconds = time.perf_counter() - wall_start
    archive.close()

    metrics = calculate_metrics(confusion, top1_correct, top5_correct, model_class_names)
    image_count = len(dataset)
    result = {
        "recorded_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "model_id": MODEL_ID,
        "model_path": str(MODEL_CACHE_DIR),
        "script_path": str(Path(__file__).resolve()),
        "model": {
            "id": MODEL_ID,
            "architecture": model.__class__.__name__,
            "parameter_count_millions": round(sum(p.numel() for p in model.parameters()) / 1e6, 4),
            "num_labels": NUM_CLASSES,
            "id2label": {str(key): value for key, value in model.config.id2label.items()},
        },
        "dataset": {
            "name": "PlantVillage",
            "source": "mohanty/PlantVillage color",
            "path": str(DATASET_DIR),
            "split": "color/test",
            "split_note": "predefined leaf-grouped color_test.txt split; 10709 images",
            "label_column": "path-derived class name",
            "image_count": image_count,
            "class_count": NUM_CLASSES,
            "class_names": model_class_names,
        },
        "environment": {
            "device": torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU",
            "precision": "FP32",
            "batch_size": BATCH_SIZE,
            "warmup_steps": WARMUP_STEPS,
            "num_workers": NUM_WORKERS,
        },
        "preprocessing": {"processor": processor.__class__.__name__},
        "metrics": metrics,
        "accuracy": {
            "top1_pct": metrics["top1_accuracy_pct"],
            "top5_pct": metrics["top5_accuracy_pct"],
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
    print("=" * 64)
    print(f"Top-1 accuracy: {metrics['top1_accuracy_pct']:.2f}%")
    print(f"Top-5 accuracy: {metrics['top5_accuracy_pct']:.2f}%")
    print(f"Macro-F1: {metrics['macro_f1_pct']:.2f}%")
    print(f"Balanced accuracy: {metrics['balanced_accuracy_pct']:.2f}%")
    print(f"模型平均推理时间: {result['timing']['model_latency_ms_per_image']:.4f} ms/image")
    print(f"模型吞吐量: {result['timing']['model_throughput_images_per_second']:.2f} images/s")
    print(f"评测结果已更新: {RESULTS_PATH}")
    _test_record.finish(locals())


if __name__ == "__main__":
    main()
