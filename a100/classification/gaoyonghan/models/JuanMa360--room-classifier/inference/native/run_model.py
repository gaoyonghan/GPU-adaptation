#!/usr/bin/env python3
"""在本地 House_Room_Dataset 上评测 JuanMa360/room-classifier。

数据集只覆盖模型七个类别中的五个类别。Dinning/Livingroom 是数据集目录名，
分别映射到模型标签 dining_room/living_room；Exterior、Interior 没有真实样本，
因此主指标只在共同的五类 logits 子集上计算。
"""

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
    _test_record = _start_records(141, __file__, globals())


import json
import os
import time
from datetime import datetime
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from transformers import AutoImageProcessor, AutoModelForImageClassification

from common import (
    balanced_accuracy_from_confusion,
)


MODEL_ID = "JuanMa360/room-classifier"
PROJECT_ROOT = _PACKAGE_ROOT
MODEL_CACHE_DIR = _WEIGHTS_ROOT
DATASET_DIR = _DATASET_ROOT/'house-rooms'
RESULTS_PATH = _OUTPUT_ROOT/_portable_os.environ.get('MODEL_PHASE','full')/'m141/metrics.json'
if __name__ == "__main__":
    RESULTS_PATH = _test_record.output / "metrics.json"

BATCH_SIZE = int(_portable_os.environ.get('MODEL_BATCH_SIZE',int(os.environ.get('BATCH_SIZE', '64'))))
NUM_WORKERS = int(_portable_os.environ.get('MODEL_NUM_WORKERS',0))
WARMUP_STEPS = _warmup_steps(3)
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
MODEL_LABELS = [
    "Exterior",
    "Interior",
    "bathroom",
    "bedroom",
    "dining_room",
    "kitchen",
    "living_room",
]
EVALUATED_LABELS = ["bathroom", "bedroom", "dining_room", "kitchen", "living_room"]
FOLDER_TO_LABEL = {
    "Bathroom": "bathroom",
    "Bedroom": "bedroom",
    "Dinning": "dining_room",
    "Kitchen": "kitchen",
    "Livingroom": "living_room",
}


def safe_divide(numerator, denominator):
    return float(numerator) / float(denominator) if denominator else 0.0


def labels_in_order(config):
    labels = []
    for index in range(config.num_labels):
        if index in config.id2label:
            labels.append(config.id2label[index])
        elif str(index) in config.id2label:
            labels.append(config.id2label[str(index)])
        else:
            raise ValueError(f"模型缺少 id2label[{index}]")
    return labels


class HouseRoomsDataset(Dataset):
    def __init__(self, records, processor):
        self.records = records
        self.processor = processor

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        image_path, label = self.records[index]
        with Image.open(image_path) as image:
            pixel_values = self.processor(
                images=image.convert("RGB"), return_tensors="pt"
            )["pixel_values"].squeeze(0)
        return pixel_values, label


def collect_records(dataset_root):
    records = []
    class_counts = {}
    for folder_name, label_name in FOLDER_TO_LABEL.items():
        folder = dataset_root / folder_name
        if not folder.is_dir():
            raise FileNotFoundError(f"缺少类别目录：{folder}")
        image_paths = sorted(
            path
            for path in folder.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        )
        if not image_paths:
            raise ValueError(f"类别目录没有可读取图片：{folder}")
        class_counts[label_name] = len(image_paths)
        records.extend((path, EVALUATED_LABELS.index(label_name)) for path in image_paths)
    return records, class_counts


def classification_metrics(
    labels, predictions, top3_predictions, losses, confidences, ranks, class_names
):
    class_count = len(class_names)
    confusion = torch.zeros((class_count, class_count), dtype=torch.long)
    for label, prediction in zip(labels, predictions):
        confusion[label, prediction] += 1

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
    per_class = {
        class_name: {
            "support": int(support[index].item()),
            "precision_pct": round(100 * precision[index].item(), 4),
            "recall_pct": round(100 * recall[index].item(), 4),
            "f1_pct": round(100 * f1[index].item(), 4),
        }
        for index, class_name in enumerate(class_names)
    }

    ece = 0.0
    ece_bins = 15
    for bin_index in range(ece_bins):
        lower = bin_index / ece_bins
        upper = (bin_index + 1) / ece_bins
        indices = [
            index
            for index, confidence in enumerate(confidences)
            if lower <= confidence < upper
            or (bin_index == ece_bins - 1 and confidence == upper)
        ]
        if indices:
            accuracy = sum(
                labels[index] == predictions[index] for index in indices
            ) / len(indices)
            mean_confidence = sum(confidences[index] for index in indices) / len(indices)
            ece += len(indices) / len(labels) * abs(accuracy - mean_confidence)

    top1_correct = sum(
        label == prediction for label, prediction in zip(labels, predictions)
    )
    top3_correct = sum(
        label in candidates for label, candidates in zip(labels, top3_predictions)
    )
    off_diagonal = confusion.clone()
    off_diagonal.fill_diagonal_(0)
    top_confusions = []
    for flat_index in torch.argsort(off_diagonal.flatten(), descending=True).tolist():
        count = int(off_diagonal.flatten()[flat_index].item())
        if count == 0 or len(top_confusions) >= 20:
            break
        true_index = flat_index // class_count
        predicted_index = flat_index % class_count
        top_confusions.append(
            {
                "true": class_names[true_index],
                "predicted": class_names[predicted_index],
                "count": count,
            }
        )

    image_count = len(labels)
    return {
        "image_count": image_count,
        "top1_accuracy_pct": round(100 * safe_divide(top1_correct, image_count), 4),
        "top3_accuracy_pct": round(100 * safe_divide(top3_correct, image_count), 4),
        "mean_reciprocal_rank": round(
            safe_divide(sum(1.0 / rank for rank in ranks), image_count), 6
        ),
        "macro_precision_pct": round(100 * precision.mean().item(), 4),
        "macro_recall_pct": round(100 * recall.mean().item(), 4),
        "balanced_accuracy_pct": round(100.0 * balanced_accuracy_from_confusion(confusion), 4),
        "macro_f1_pct": round(100 * f1.mean().item(), 4),
        "weighted_f1_pct": round(
            100
            * safe_divide(
                sum(f1[index].item() * support[index].item() for index in range(class_count)),
                image_count,
            ),
            4,
        ),
        "cross_entropy_loss": round(safe_divide(sum(losses), image_count), 6),
        "expected_calibration_error": round(ece, 6),
        "per_class": per_class,
        "top_confusion_pairs": top_confusions,
        "confusion_matrix": confusion.tolist(),
        "confusion_matrix_labels": class_names,
    }


def save_result(result):
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = RESULTS_PATH.with_suffix(".json.tmp")
    temporary_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    os.replace(temporary_path, RESULTS_PATH)


def main():
    if not MODEL_CACHE_DIR.is_dir():
        raise FileNotFoundError(f"未找到模型缓存：{MODEL_CACHE_DIR}")
    dataset_root = DATASET_DIR / "House_Room_Dataset"
    if not dataset_root.is_dir():
        raise FileNotFoundError(f"未找到 House_Room_Dataset：{dataset_root}")

    processor = AutoImageProcessor.from_pretrained(
        str(MODEL_CACHE_DIR), local_files_only=True
    )
    model = AutoModelForImageClassification.from_pretrained(
        str(MODEL_CACHE_DIR), local_files_only=True
    )
    model_class_names = labels_in_order(model.config)
    if model_class_names != MODEL_LABELS:
        raise ValueError(
            f"模型类别顺序应为 {MODEL_LABELS}，实际为 {model_class_names}"
        )
    if model.config.problem_type != "single_label_classification":
        raise ValueError(f"模型 problem_type 异常：{model.config.problem_type}")

    records, class_counts = collect_records(dataset_root)
    model_eval_indices = [MODEL_LABELS.index(label) for label in EVALUATED_LABELS]
    dataset = HouseRoomsDataset(records, processor)
    dataloader = _test_record.loader(DataLoader, 
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
    )
    device = torch.device(_select_torch_device(torch, 'cuda' if torch.cuda.is_available() else 'cpu'))
    device_name = torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU"
    model = model.to(device).eval()

    print(f"使用设备: {device_name}")
    print(f"模型: {MODEL_ID}")
    print(f"数据集: {dataset_root}（本地完整目录，未发现独立 test split）")
    print(f"评测类别: {EVALUATED_LABELS}")
    print("排除模型类别: Exterior、Interior（数据集无对应样本）")
    print(f"测试图片数量: {len(dataset)}")
    print(f"类别分布: {class_counts}")
    print(f"模型参数量: {sum(parameter.numel() for parameter in model.parameters()) / 1e6:.4f} M")
    print(f"batch size: {BATCH_SIZE}")

    warmup_pixel_values, _ = next(iter(dataloader))
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

    labels, predictions, top3_predictions = [], [], []
    losses, confidences, ranks = [], [], []
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
                logits = model(pixel_values=pixel_values).logits
                end_event.record()
                torch.cuda.synchronize()
                forward_seconds += start_event.elapsed_time(end_event) / 1000.0
            else:
                forward_start = time.perf_counter()
                logits = model(pixel_values=pixel_values).logits
                forward_seconds += time.perf_counter() - forward_start

            # 只在数据集覆盖的五个模型类别之间比较，避免缺失类别改变主指标。
            logits = logits[:, model_eval_indices]
            probabilities = F.softmax(logits, dim=1)
            confidence, batch_predictions = probabilities.max(dim=1)
            batch_top3 = logits.topk(3, dim=1).indices
            batch_loss = F.cross_entropy(logits, batch_labels, reduction="none")
            true_logits = logits.gather(1, batch_labels.unsqueeze(1))
            batch_ranks = logits.gt(true_logits).sum(dim=1).add(1)
            labels.extend(batch_labels.cpu().tolist())
            predictions.extend(batch_predictions.cpu().tolist())
            top3_predictions.extend(batch_top3.cpu().tolist())
            losses.extend(batch_loss.cpu().tolist())
            confidences.extend(confidence.cpu().tolist())
            ranks.extend(batch_ranks.cpu().tolist())
            if batch_index % 10 == 0 or batch_index == len(dataloader):
                print(f"已处理: {min(batch_index * BATCH_SIZE, len(dataset))}/{len(dataset)}")
            _test_record.capture(locals())

    if device.type == "cuda":
        torch.cuda.synchronize()
    wall_seconds = time.perf_counter() - wall_start
    metrics = classification_metrics(
        labels,
        predictions,
        top3_predictions,
        losses,
        confidences,
        ranks,
        EVALUATED_LABELS,
    )
    image_count = len(dataset)
    result = {
        "recorded_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "model_id": MODEL_ID,
        "model_path": str(MODEL_CACHE_DIR),
        "script_path": str(Path(__file__).resolve()),
        "model": {
            "id": MODEL_ID,
            "architecture": model.__class__.__name__,
            "parameter_count_millions": round(sum(parameter.numel() for parameter in model.parameters()) / 1e6, 4),
            "num_labels": model.config.num_labels,
            "class_names": MODEL_LABELS,
        },
        "dataset": {
            "name": "house-rooms",
            "source": "local House_Room_Dataset",
            "path": str(DATASET_DIR),
            "resolved_root": str(dataset_root),
            "split": "full_local_dataset",
            "split_note": (
                "本地目录未提供独立 test split；本次使用全部可用图片，结果不等价于独立泛化性能。"
            ),
            "image_count": image_count,
            "original_class_names": sorted(FOLDER_TO_LABEL),
            "evaluated_class_names": EVALUATED_LABELS,
            "class_count": len(EVALUATED_LABELS),
            "class_counts": class_counts,
            "label_mapping": FOLDER_TO_LABEL,
            "excluded_model_classes": ["Exterior", "Interior"],
            "exclusion_note": "数据集没有 Exterior、Interior 样本；主指标在共同五类 logits 子集上计算。",
        },
        "evaluation": {
            "task": "five-class room image classification",
            "logit_scope": EVALUATED_LABELS,
            "primary_metrics": [
                "top1_accuracy_pct",
                "top3_accuracy_pct",
                "macro_f1_pct",
                "balanced_accuracy_pct",
            ],
            "independence_warning": True,
        },
        "environment": {
            "device": device_name,
            "precision": "FP32",
            "batch_size": BATCH_SIZE,
            "warmup_steps": WARMUP_STEPS,
            "num_workers": NUM_WORKERS,
        },
        "preprocessing": {"processor": processor.__class__.__name__},
        "metrics": metrics,
        "accuracy": {
            "top1_pct": metrics["top1_accuracy_pct"],
            "top3_pct": metrics["top3_accuracy_pct"],
        },
        "timing": {
            "model_forward_total_seconds": round(forward_seconds, 6),
            "model_latency_ms_per_image": round(forward_seconds / image_count * 1000.0, 6),
            "model_throughput_images_per_second": round(image_count / forward_seconds, 6),
            "end_to_end_total_seconds": round(wall_seconds, 6),
            "end_to_end_latency_ms_per_image": round(wall_seconds / image_count * 1000.0, 6),
            "end_to_end_throughput_images_per_second": round(image_count / wall_seconds, 6),
        },
    }
    if device.type == "cuda":
        result["memory"] = {
            "peak_allocated_mib": round(torch.cuda.max_memory_allocated(device) / 1024**2, 4),
            "peak_reserved_mib": round(torch.cuda.max_memory_reserved(device) / 1024**2, 4),
        }
    save_result(result)

    print("=" * 64)
    print(f"Top-1 accuracy: {metrics['top1_accuracy_pct']:.2f}%")
    print(f"Top-3 accuracy: {metrics['top3_accuracy_pct']:.2f}%")
    print(f"Macro-F1: {metrics['macro_f1_pct']:.2f}%")
    print(f"Balanced accuracy: {metrics['balanced_accuracy_pct']:.2f}%")
    print(f"模型平均推理时间: {result['timing']['model_latency_ms_per_image']:.4f} ms/image")
    print(f"模型吞吐量: {result['timing']['model_throughput_images_per_second']:.2f} images/s")
    print(f"评测结果已更新: {RESULTS_PATH}")
    _test_record.finish(locals())


if __name__ == "__main__":
    main()
