#!/usr/bin/env python3
"""Evaluate qipchip31/electronic-components-model on its electronic_components test split."""

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
    _test_record = _start_records(121, __file__, globals())


import json
import math
import os
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from datasets import Dataset as HFDataset
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForImageClassification, ViTImageProcessor

from common import (
    balanced_accuracy_from_confusion,
)


MODEL_ID = "qipchip31/electronic-components-model"
PROJECT_ROOT = _PACKAGE_ROOT
MODEL_CACHE_DIR = _WEIGHTS_ROOT
DATASET_DIR = _DATASET_ROOT/'electronic_components'
TEST_PARQUET = DATASET_DIR / "data/test-00000-of-00001.parquet"
RESULTS_PATH = _OUTPUT_ROOT/_portable_os.environ.get('MODEL_PHASE','full')/'m121/metrics.json'
if __name__ == "__main__":
    RESULTS_PATH = _test_record.output / "metrics.json"
CLASS_NAMES = ["resistor", "capacitor", "inductor", "transistor"]
BATCH_SIZE = int(_portable_os.environ.get('MODEL_BATCH_SIZE',32))
NUM_WORKERS = int(_portable_os.environ.get('MODEL_NUM_WORKERS',0))
WARMUP_STEPS = _warmup_steps(3)
ECE_BINS = 15


class ElectronicComponentsTestDataset(Dataset):
    def __init__(self, dataset, processor):
        self.dataset = dataset
        self.processor = processor

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        image = Image.fromarray(
            np.asarray(self.dataset[index]["image"], dtype=np.uint8)
        ).convert("RGB")
        pixel_values = self.processor(images=image, return_tensors="pt")[
            "pixel_values"
        ].squeeze(0)
        return pixel_values, self.dataset[index]["label"]


def safe_divide(numerator, denominator):
    return float(numerator) / float(denominator) if denominator else 0.0


def expected_calibration_error(labels, predictions, confidences):
    ece = 0.0
    for bin_index in range(ECE_BINS):
        lower = bin_index / ECE_BINS
        upper = (bin_index + 1) / ECE_BINS
        indices = [
            index
            for index, confidence in enumerate(confidences)
            if lower <= confidence < upper
            or (bin_index == ECE_BINS - 1 and confidence == upper)
        ]
        if indices:
            accuracy = sum(labels[index] == predictions[index] for index in indices) / len(
                indices
            )
            mean_confidence = sum(confidences[index] for index in indices) / len(indices)
            ece += len(indices) / len(labels) * abs(accuracy - mean_confidence)
    return round(ece, 6)


def classification_metrics(labels, predictions, top3_predictions, losses, confidences):
    class_count = len(CLASS_NAMES)
    confusion = [[0 for _ in range(class_count)] for _ in range(class_count)]
    for label, prediction in zip(labels, predictions):
        confusion[label][prediction] += 1

    per_class = {}
    precisions, recalls, f1_scores = [], [], []
    for class_index, class_name in enumerate(CLASS_NAMES):
        true_positive = confusion[class_index][class_index]
        support = sum(confusion[class_index])
        predicted = sum(row[class_index] for row in confusion)
        precision = safe_divide(true_positive, predicted)
        recall = safe_divide(true_positive, support)
        f1 = safe_divide(2.0 * precision * recall, precision + recall)
        precisions.append(precision)
        recalls.append(recall)
        f1_scores.append(f1)
        per_class[class_name] = {
            "support": support,
            "precision_pct": round(100.0 * precision, 4),
            "recall_pct": round(100.0 * recall, 4),
            "f1_pct": round(100.0 * f1, 4),
        }

    image_count = len(labels)
    top1_correct = sum(label == prediction for label, prediction in zip(labels, predictions))
    top3_correct = sum(
        label in candidates for label, candidates in zip(labels, top3_predictions)
    )
    return {
        "image_count": image_count,
        "top1_accuracy_pct": round(100.0 * safe_divide(top1_correct, image_count), 4),
        "top3_accuracy_pct": round(100.0 * safe_divide(top3_correct, image_count), 4),
        "macro_precision_pct": round(100.0 * sum(precisions) / class_count, 4),
        "macro_recall_pct": round(100.0 * sum(recalls) / class_count, 4),
        "balanced_accuracy_pct": round(100.0 * balanced_accuracy_from_confusion(confusion), 4),
        "macro_f1_pct": round(100.0 * sum(f1_scores) / class_count, 4),
        "cross_entropy_loss": round(safe_divide(sum(losses), image_count), 6),
        "expected_calibration_error": expected_calibration_error(
            labels, predictions, confidences
        ),
        "mean_prediction_confidence_pct": round(
            100.0 * safe_divide(sum(confidences), image_count), 4
        ),
        "per_class": per_class,
        "confusion_matrix_rows_true_cols_pred": confusion,
        "confusion_matrix_labels": CLASS_NAMES,
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
        raise FileNotFoundError(f"未找到模型目录：{MODEL_CACHE_DIR}")
    if not TEST_PARQUET.is_file():
        raise FileNotFoundError(f"未找到测试 Parquet：{TEST_PARQUET}")

    # The model directory does not include preprocessor_config.json. These are
    # the default ViTImageProcessor settings used by google/vit-base-patch16-224.
    processor = ViTImageProcessor()
    model = AutoModelForImageClassification.from_pretrained(
        str(MODEL_CACHE_DIR), local_files_only=True
    )
    if model.config.num_labels != len(CLASS_NAMES):
        raise ValueError(
            f"模型应输出 {len(CLASS_NAMES)} 类，实际为 {model.config.num_labels} 类"
        )
    model_labels = [
        model.config.id2label[index]
        if index in model.config.id2label
        else model.config.id2label[str(index)]
        for index in range(model.config.num_labels)
    ]

    full_test = HFDataset.from_parquet(str(TEST_PARQUET))
    if full_test.column_names != ["image", "label"]:
        raise ValueError(f"测试列应为 image、label，实际为 {full_test.column_names}")
    if len(full_test) != 74:
        raise ValueError(f"电子元件测试集应有 74 张图片，实际为 {len(full_test)} 张")
    observed_labels = sorted(set(full_test["label"]))
    if observed_labels != list(range(len(CLASS_NAMES))):
        raise ValueError(f"测试标签应为 0..3，实际为 {observed_labels}")

    dataset = ElectronicComponentsTestDataset(full_test, processor)
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
    print("数据集: qipchip31/electronic_components test split")
    print(f"测试图片数量: {len(dataset)}")
    print(f"类别顺序: {CLASS_NAMES}")
    print(f"模型 config.id2label: {model_labels}")
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
    losses, confidences = [], []
    forward_seconds = 0.0
    wall_start = time.perf_counter()
    with torch.inference_mode():
        for pixel_values, batch_labels in dataloader:
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

            probabilities = F.softmax(logits, dim=1)
            batch_confidence, batch_predictions = probabilities.max(dim=1)
            batch_top3 = logits.topk(3, dim=1).indices
            batch_losses = F.cross_entropy(logits, batch_labels, reduction="none")
            labels.extend(batch_labels.cpu().tolist())
            predictions.extend(batch_predictions.cpu().tolist())
            top3_predictions.extend(batch_top3.cpu().tolist())
            losses.extend(batch_losses.cpu().tolist())
            confidences.extend(batch_confidence.cpu().tolist())
            _test_record.capture(locals())

    if device.type == "cuda":
        torch.cuda.synchronize()
    wall_seconds = time.perf_counter() - wall_start
    metrics = classification_metrics(
        labels, predictions, top3_predictions, losses, confidences
    )
    label_counts = {CLASS_NAMES[index]: labels.count(index) for index in range(len(CLASS_NAMES))}
    prediction_counts = {
        CLASS_NAMES[index]: predictions.count(index) for index in range(len(CLASS_NAMES))
    }
    result = {
        "recorded_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "model_id": MODEL_ID,
        "model_path": str(MODEL_CACHE_DIR),
        "script_path": str(Path(__file__).resolve()),
        "model": {
            "id": MODEL_ID,
            "architecture": model.__class__.__name__,
            "parameter_count_millions": round(
                sum(parameter.numel() for parameter in model.parameters()) / 1e6, 4
            ),
            "num_labels": model.config.num_labels,
            "config_id2label": {
                str(index): label for index, label in enumerate(model_labels)
            },
            "dataset_label_mapping": {
                str(index): CLASS_NAMES[index] for index in range(len(CLASS_NAMES))
            },
        },
        "dataset": {
            "name": "qipchip31/electronic_components",
            "path": str(DATASET_DIR),
            "split": "test",
            "source_parquet": str(TEST_PARQUET),
            "image_count": len(dataset),
            "class_count": len(CLASS_NAMES),
            "class_names": CLASS_NAMES,
            "label_counts": label_counts,
            "small_test_set_note": "The repository test split contains only 74 images; metrics have high sampling uncertainty.",
        },
        "environment": {
            "device": device_name,
            "precision": "FP32",
            "batch_size": BATCH_SIZE,
            "warmup_steps": WARMUP_STEPS,
            "num_workers": NUM_WORKERS,
        },
        "preprocessing": {
            "processor": "ViTImageProcessor default configuration for google/vit-base-patch16-224",
            "details": "RGB conversion, resize to 224x224, bilinear resampling, rescale by 1/255, normalize with mean/std 0.5",
        },
        "metrics": metrics,
        "prediction_counts": prediction_counts,
        "timing": {
            "model_latency_ms_per_image": round(
                forward_seconds / len(dataset) * 1000.0, 4
            ),
            "model_throughput_images_per_second": round(
                len(dataset) / forward_seconds, 4
            ),
            "end_to_end_latency_ms_per_image": round(
                wall_seconds / len(dataset) * 1000.0, 4
            ),
            "end_to_end_throughput_images_per_second": round(
                len(dataset) / wall_seconds, 4
            ),
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
    print(f"Top-3 accuracy: {metrics['top3_accuracy_pct']:.2f}%")
    print(f"Macro-F1: {metrics['macro_f1_pct']:.2f}%")
    print(f"Balanced accuracy: {metrics['balanced_accuracy_pct']:.2f}%")
    print(f"模型平均推理时间: {result['timing']['model_latency_ms_per_image']:.4f} ms/image")
    print(f"模型吞吐量: {result['timing']['model_throughput_images_per_second']:.2f} images/s")
    print(f"评测结果已保存: {RESULTS_PATH}")
    _test_record.finish(locals())


if __name__ == "__main__":
    main()
