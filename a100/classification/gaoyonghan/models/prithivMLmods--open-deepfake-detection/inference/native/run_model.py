#!/usr/bin/env python3
"""Evaluate prithivMLmods/open-deepfake-detection on AIGCDetectionBenchmark."""

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
    _test_record = _start_records(148, __file__, globals())


import json
import math
import os
import time
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image, ImageFile
from torch.utils.data import DataLoader, Dataset
from transformers import AutoImageProcessor, AutoModelForImageClassification

from common import (
    balanced_accuracy_from_confusion,
    binary_roc_auc as standard_binary_roc_auc,
    rounded_or_none,
)


MODEL_ID = "prithivMLmods/open-deepfake-detection"
PROJECT_ROOT = _PACKAGE_ROOT
MODEL_CACHE_DIR = _WEIGHTS_ROOT
DATASET_DIR = _DATASET_ROOT/'AIGCDetectionBenchmark/test'
RESULTS_PATH = _OUTPUT_ROOT/_portable_os.environ.get('MODEL_PHASE','full')/'m148/metrics.json'
if __name__ == "__main__":
    RESULTS_PATH = _test_record.output / "metrics.json"

BATCH_SIZE = int(_portable_os.environ.get('MODEL_BATCH_SIZE',512))
NUM_WORKERS = int(_portable_os.environ.get('MODEL_NUM_WORKERS',4))
WARMUP_STEPS = _warmup_steps(3)
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
REAL_LABEL = 0
FAKE_LABEL = 1
MODEL_FAKE_INDEX = 0
MODEL_REAL_INDEX = 1

# The local benchmark contains at least one truncated JPEG.
ImageFile.LOAD_TRUNCATED_IMAGES = True


class BenchmarkDataset(Dataset):
    def __init__(self, records, processor):
        self.records = records
        self.processor = processor

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        path, _source, _subcategory, label = self.records[index]
        with Image.open(path) as image:
            pixel_values = self.processor(
                images=image.convert("RGB"), return_tensors="pt"
            )["pixel_values"].squeeze(0)
        return pixel_values, label


def safe_divide(numerator, denominator):
    return float(numerator) / float(denominator) if denominator else 0.0


def collect_records():
    if not DATASET_DIR.is_dir():
        raise FileNotFoundError(f"未找到 AIGCDetectionBenchmark 测试目录：{DATASET_DIR}")
    records = []
    for path in sorted(DATASET_DIR.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        parts = path.relative_to(DATASET_DIR).parts
        label_index = next(
            (index for index, part in enumerate(parts) if part in {"0_real", "1_fake"}),
            None,
        )
        if label_index is None or label_index == 0:
            raise ValueError(f"图片路径缺少有效的生成器/标签目录：{path}")
        source = parts[0]
        subcategory = "/".join(parts[1:label_index]) or None
        label = REAL_LABEL if parts[label_index] == "0_real" else FAKE_LABEL
        records.append((path, source, subcategory, label))
    if not records:
        raise ValueError(f"测试目录中没有可识别的图片：{DATASET_DIR}")
    return records


def binary_metrics(labels, predictions, fake_scores):
    true_negative = sum(
        label == REAL_LABEL and prediction == REAL_LABEL
        for label, prediction in zip(labels, predictions)
    )
    false_positive = sum(
        label == REAL_LABEL and prediction == FAKE_LABEL
        for label, prediction in zip(labels, predictions)
    )
    false_negative = sum(
        label == FAKE_LABEL and prediction == REAL_LABEL
        for label, prediction in zip(labels, predictions)
    )
    true_positive = sum(
        label == FAKE_LABEL and prediction == FAKE_LABEL
        for label, prediction in zip(labels, predictions)
    )
    image_count = len(labels)
    real_count = sum(label == REAL_LABEL for label in labels)
    fake_count = sum(label == FAKE_LABEL for label in labels)
    real_recall = safe_divide(true_negative, real_count)
    fake_recall = safe_divide(true_positive, fake_count)
    fake_precision = safe_divide(true_positive, true_positive + false_positive)
    fake_f1 = safe_divide(
        2.0 * fake_precision * fake_recall, fake_precision + fake_recall
    )
    binary_log_loss = safe_divide(
        sum(
            -math.log(max(min(score, 1.0 - 1e-15), 1e-15))
            if label == FAKE_LABEL
            else -math.log(max(min(1.0 - score, 1.0 - 1e-15), 1e-15))
            for label, score in zip(labels, fake_scores)
        ),
        image_count,
    )
    return {
        "image_count": image_count,
        "real_count": real_count,
        "fake_count": fake_count,
        "accuracy_pct": round(
            100.0 * safe_divide(true_negative + true_positive, image_count), 4
        ),
        "balanced_accuracy_pct": round(100.0 * balanced_accuracy_from_confusion([[true_negative, false_positive], [false_negative, true_positive]]), 4),
        "fake_precision_pct": round(100.0 * fake_precision, 4),
        "fake_recall_pct": round(100.0 * fake_recall, 4),
        "fake_f1_pct": round(100.0 * fake_f1, 4),
        "real_specificity_pct": round(100.0 * real_recall, 4),
        "binary_log_loss": round(binary_log_loss, 6),
        "confusion_matrix_rows_true_cols_pred": [
            [true_negative, false_positive],
            [false_negative, true_positive],
        ],
        "confusion_matrix_labels": ["Real", "Fake"],
        "mean_fake_probability_pct": round(
            100.0 * safe_divide(sum(fake_scores), image_count), 4
        ),
    }


def binary_auroc(labels, scores):
    return rounded_or_none(
        standard_binary_roc_auc([int(label == FAKE_LABEL) for label in labels], scores)
    )


def expected_calibration_error(labels, predictions, confidences, bins=15):
    ece = 0.0
    for bin_index in range(bins):
        lower = bin_index / bins
        upper = (bin_index + 1) / bins
        indices = [
            index
            for index, confidence in enumerate(confidences)
            if lower <= confidence < upper
            or (bin_index == bins - 1 and confidence == upper)
        ]
        if indices:
            accuracy = sum(labels[index] == predictions[index] for index in indices) / len(
                indices
            )
            mean_confidence = sum(confidences[index] for index in indices) / len(indices)
            ece += len(indices) / len(labels) * abs(accuracy - mean_confidence)
    return round(ece, 6)


def save_result(result):
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = RESULTS_PATH.with_suffix(".json.tmp")
    temporary_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    os.replace(temporary_path, RESULTS_PATH)


def build_source_metrics(source_states):
    result = {}
    for source in sorted(source_states):
        state = source_states[source]
        metrics = binary_metrics(state["labels"], state["predictions"], state["scores"])
        metrics["auroc"] = binary_auroc(state["labels"], state["scores"])
        metrics["subcategories"] = {}
        for name, value in sorted(state["subcategories"].items()):
            sub_metrics = binary_metrics(
                value["labels"], value["predictions"], value["scores"]
            )
            sub_metrics["auroc"] = binary_auroc(value["labels"], value["scores"])
            metrics["subcategories"][name] = sub_metrics
        result[source] = metrics
    return result


def main():
    if not MODEL_CACHE_DIR.is_dir():
        raise FileNotFoundError(f"未找到模型缓存：{MODEL_CACHE_DIR}")
    processor = AutoImageProcessor.from_pretrained(
        str(MODEL_CACHE_DIR), local_files_only=True
    )
    model = AutoModelForImageClassification.from_pretrained(
        str(MODEL_CACHE_DIR), local_files_only=True
    )
    id2label = {int(index): label for index, label in model.config.id2label.items()}
    model_labels = [id2label[index] for index in range(model.config.num_labels)]
    if model_labels != ["Fake", "Real"]:
        raise ValueError(f"模型标签应按 [Fake, Real] 排列，实际为 {model_labels}")

    records = collect_records()
    dataset = BenchmarkDataset(records, processor)
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

    source_counts = Counter(source for _path, source, _subcategory, _label in records)
    label_counts = Counter(label for _path, _source, _subcategory, label in records)
    print(f"使用设备: {device_name}")
    print(f"模型: {MODEL_ID}")
    print("数据集: AIGCDetectionBenchmark/test")
    print(f"测试图片数量: {len(dataset)}")
    print(f"真实/伪造数量: {label_counts[REAL_LABEL]}/{label_counts[FAKE_LABEL]}")
    print(f"生成器数量: {len(source_counts)}")
    print(f"类别顺序: {model_labels}")
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

    labels, predictions, fake_scores, confidences = [], [], [], []
    raw_prediction_counts = Counter()
    source_states = defaultdict(
        lambda: {
            "labels": [],
            "predictions": [],
            "scores": [],
            "subcategories": defaultdict(
                lambda: {"labels": [], "predictions": [], "scores": []}
            ),
        }
    )
    forward_seconds = 0.0
    wall_start = time.perf_counter()
    with torch.inference_mode():
        for batch_index, (pixel_values, batch_labels) in enumerate(
            dataloader, start=1
        ):
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
            confidence, batch_predictions = probabilities.max(dim=1)
            batch_fake_scores = probabilities[:, MODEL_FAKE_INDEX]
            # Model indices are Fake=0/Real=1; metrics use Real=0/Fake=1.
            batch_binary_predictions = torch.where(
                batch_predictions.eq(MODEL_REAL_INDEX), REAL_LABEL, FAKE_LABEL
            )
            labels.extend(batch_labels.cpu().tolist())
            predictions.extend(batch_binary_predictions.cpu().tolist())
            fake_scores.extend(batch_fake_scores.cpu().tolist())
            confidences.extend(confidence.cpu().tolist())
            raw_prediction_counts.update(batch_predictions.cpu().tolist())

            start_index = (batch_index - 1) * BATCH_SIZE
            for record_index, (label, prediction, score) in enumerate(
                zip(
                    batch_labels.cpu().tolist(),
                    batch_binary_predictions.cpu().tolist(),
                    batch_fake_scores.cpu().tolist(),
                ),
                start=start_index,
            ):
                _path, source, subcategory, _record_label = records[record_index]
                source_state = source_states[source]
                source_state["labels"].append(label)
                source_state["predictions"].append(prediction)
                source_state["scores"].append(score)
                subcategory_name = subcategory or "(root)"
                subcategory_state = source_state["subcategories"][subcategory_name]
                subcategory_state["labels"].append(label)
                subcategory_state["predictions"].append(prediction)
                subcategory_state["scores"].append(score)

            if batch_index % 25 == 0 or batch_index == len(dataloader):
                print(f"已处理: {min(batch_index * BATCH_SIZE, len(dataset))}/{len(dataset)}")
            _test_record.capture(locals())

    if device.type == "cuda":
        torch.cuda.synchronize()
    wall_seconds = time.perf_counter() - wall_start
    metrics = binary_metrics(labels, predictions, fake_scores)
    metrics["auroc"] = binary_auroc(labels, fake_scores)
    metrics["expected_calibration_error"] = expected_calibration_error(
        labels, predictions, confidences
    )

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
            "id2label": {str(index): label for index, label in id2label.items()},
        },
        "dataset": {
            "name": "AIGCDetectionBenchmark",
            "path": str(DATASET_DIR),
            "split": "test",
            "image_count": len(dataset),
            "source_count": len(source_counts),
            "source_image_counts": dict(sorted(source_counts.items())),
            "binary_label_counts": {
                "Real": label_counts[REAL_LABEL],
                "Fake": label_counts[FAKE_LABEL],
            },
            "label_protocol": (
                "0_real and 1_fake directory labels are mapped directly to Real and Fake."
            ),
        },
        "evaluation": {
            "task": "binary real-vs-fake classification",
            "model_label_mapping": "model class 0 Fake, class 1 Real",
        },
        "environment": {
            "device": device_name,
            "precision": "FP32",
            "batch_size": BATCH_SIZE,
            "warmup_steps": WARMUP_STEPS,
            "num_workers": NUM_WORKERS,
        },
        "preprocessing": {
            "processor": processor.__class__.__name__,
            "details": "model-local processor: resize 224x224, rescale by 1/255, normalize with mean/std 0.5, RGB",
            "truncated_image_policy": "PIL LOAD_TRUNCATED_IMAGES=True; truncated files are evaluated using decodable content",
        },
        "metrics": metrics,
        "raw_predictions": {
            "top1_prediction_counts": {
                model_labels[index]: raw_prediction_counts[index]
                for index in range(len(model_labels))
            },
            "top1_prediction_pct": {
                model_labels[index]: round(
                    100.0 * safe_divide(raw_prediction_counts[index], len(dataset)), 4
                )
                for index in range(len(model_labels))
            },
        },
        "per_source": build_source_metrics(source_states),
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
    print(f"Binary accuracy: {metrics['accuracy_pct']:.2f}%")
    print(f"Balanced accuracy: {metrics['balanced_accuracy_pct']:.2f}%")
    print(f"Fake F1: {metrics['fake_f1_pct']:.2f}%")
    print(f"AUROC: {metrics['auroc']:.6f}")
    print(f"模型平均推理时间: {result['timing']['model_latency_ms_per_image']:.4f} ms/image")
    print(f"模型吞吐量: {result['timing']['model_throughput_images_per_second']:.2f} images/s")
    print(f"评测结果已保存: {RESULTS_PATH}")
    _test_record.finish(locals())


if __name__ == "__main__":
    main()
