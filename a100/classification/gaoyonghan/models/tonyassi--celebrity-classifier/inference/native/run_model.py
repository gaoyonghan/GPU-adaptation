#!/usr/bin/env python3
"""Evaluate tonyassi/celebrity-classifier on a local celebrity-1000 holdout.

The public celebrity-1000 dataset contains one labelled train split only, so
this script creates a fixed stratified 20% holdout and records that scope in
the result JSON.
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
    _test_record = _start_records(117, __file__, globals())


import json
import os
import time
from datetime import datetime
from pathlib import Path

import torch
import torch.nn.functional as F
from datasets import Dataset as HFDataset
from torch.utils.data import DataLoader, Dataset
from transformers import AutoImageProcessor, AutoModelForImageClassification

from common import (
    balanced_accuracy_from_confusion,
)


MODEL_ID = "tonyassi/celebrity-classifier"
PROJECT_ROOT = _PACKAGE_ROOT
MODEL_CACHE_DIR = _WEIGHTS_ROOT
DATASET_DIR = _DATASET_ROOT/'celebrity-1000'
RESULTS_PATH = _OUTPUT_ROOT/_portable_os.environ.get('MODEL_PHASE','full')/'m117/metrics.json'
if __name__ == "__main__":
    RESULTS_PATH = _test_record.output / "metrics.json"

BATCH_SIZE = int(_portable_os.environ.get('MODEL_BATCH_SIZE',64))
NUM_WORKERS = int(_portable_os.environ.get('MODEL_NUM_WORKERS',0))
WARMUP_STEPS = _warmup_steps(3)
HOLDOUT_FRACTION = 0.2
SPLIT_SEED = 42
ECE_BINS = 15


def safe_divide(numerator, denominator):
    return float(numerator) / float(denominator) if denominator else 0.0


def find_train_shards():
    shards = sorted(DATASET_DIR.glob("data/train-*.parquet"))
    if not shards:
        raise FileNotFoundError(f"未找到 celebrity-1000 parquet：{DATASET_DIR}")
    return shards


class CelebrityDataset(Dataset):
    def __init__(self, dataset, processor):
        self.dataset = dataset
        self.processor = processor

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        example = self.dataset[index]
        pixel_values = self.processor(
            images=example["image"].convert("RGB"), return_tensors="pt"
        )["pixel_values"].squeeze(0)
        return pixel_values, int(example["label"])


def classification_metrics(
    labels, predictions, top5_predictions, losses, confidences, ranks, class_names
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
            accuracy = sum(
                labels[index] == predictions[index] for index in indices
            ) / len(indices)
            average_confidence = sum(confidences[index] for index in indices) / len(
                indices
            )
            ece += len(indices) / len(labels) * abs(accuracy - average_confidence)

    top1_correct = sum(
        label == prediction for label, prediction in zip(labels, predictions)
    )
    top5_correct = sum(
        label in candidates for label, candidates in zip(labels, top5_predictions)
    )
    off_diagonal = confusion.clone()
    off_diagonal.fill_diagonal_(0)
    top_confusions = []
    for flat_index in torch.argsort(off_diagonal.flatten(), descending=True).tolist():
        count = int(off_diagonal.flatten()[flat_index].item())
        if count == 0 or len(top_confusions) == 30:
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
        "top5_accuracy_pct": round(100 * safe_divide(top5_correct, image_count), 4),
        "mean_reciprocal_rank": round(
            safe_divide(sum(1.0 / rank for rank in ranks), image_count), 6
        ),
        "macro_precision_pct": round(100 * precision.mean().item(), 4),
        "macro_recall_pct": round(100 * recall.mean().item(), 4),
        "balanced_accuracy_pct": round(100.0 * balanced_accuracy_from_confusion(confusion), 4),
        "macro_f1_pct": round(100 * f1.mean().item(), 4),
        "cross_entropy_loss": round(safe_divide(sum(losses), image_count), 6),
        "expected_calibration_error": round(ece, 6),
        "per_class": per_class,
        "top_confusion_pairs": top_confusions,
        "confusion_matrix_note": "997×997 matrix omitted from JSON; see per_class and top_confusion_pairs.",
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
    shards = find_train_shards()
    processor = AutoImageProcessor.from_pretrained(
        str(MODEL_CACHE_DIR), local_files_only=True
    )
    model = AutoModelForImageClassification.from_pretrained(
        str(MODEL_CACHE_DIR), local_files_only=True
    )
    model_class_names = [
        model.config.id2label[index] for index in range(model.config.num_labels)
    ]

    dataset = HFDataset.from_parquet(
        [str(path) for path in shards], cache_dir=_portable_os.environ["HF_DATASETS_CACHE"]
    )
    if set(dataset.column_names) != {"image", "label"}:
        raise ValueError(
            f"celebrity-1000 列应为 image、label，实际为：{dataset.column_names}"
        )
    dataset_class_names = list(dataset.features["label"].names)
    if dataset_class_names != model_class_names:
        raise ValueError("数据集和模型的类别名称或顺序不一致，不能直接评测")

    split = dataset.train_test_split(
        test_size=HOLDOUT_FRACTION,
        seed=SPLIT_SEED,
        stratify_by_column="label",
    )
    wrapped_dataset = CelebrityDataset(split["test"], processor)
    device = torch.device(_select_torch_device(torch, 'cuda' if torch.cuda.is_available() else 'cpu'))
    device_name = torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU"
    model = model.to(device)
    model.eval()
    dataloader = _test_record.loader(DataLoader, 
        wrapped_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=device.type == "cuda",
    )

    print(f"使用设备: {device_name}")
    print(f"模型: {MODEL_ID}")
    print("数据集: celebrity-1000 的固定分层 20% holdout（无官方 test split）")
    print(f"原始图片数量: {len(dataset)}")
    print(f"测试图片数量: {len(wrapped_dataset)}")
    print(f"类别数量: {len(model_class_names)}")
    print(
        f"模型参数量: {sum(parameter.numel() for parameter in model.parameters()) / 1e6:.4f} M"
    )
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

    labels = []
    predictions = []
    top5_predictions = []
    losses = []
    confidences = []
    ranks = []
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
            batch_top5 = logits.topk(5, dim=1).indices
            batch_loss = F.cross_entropy(logits, batch_labels, reduction="none")
            true_logits = logits.gather(1, batch_labels.unsqueeze(1))
            batch_ranks = logits.gt(true_logits).sum(dim=1).add(1)
            labels.extend(batch_labels.cpu().tolist())
            predictions.extend(batch_predictions.cpu().tolist())
            top5_predictions.extend(batch_top5.cpu().tolist())
            losses.extend(batch_loss.cpu().tolist())
            confidences.extend(confidence.cpu().tolist())
            ranks.extend(batch_ranks.cpu().tolist())
            if batch_index % 10 == 0 or batch_index == len(dataloader):
                print(
                    f"已处理: {min(batch_index * BATCH_SIZE, len(wrapped_dataset))}"
                    f"/{len(wrapped_dataset)}"
                )
            _test_record.capture(locals())
    if device.type == "cuda":
        torch.cuda.synchronize()
    wall_seconds = time.perf_counter() - wall_start

    metrics = classification_metrics(
        labels,
        predictions,
        top5_predictions,
        losses,
        confidences,
        ranks,
        model_class_names,
    )
    image_count = len(wrapped_dataset)
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
            "num_labels": len(model_class_names),
        },
        "dataset": {
            "name": "celebrity-1000",
            "source": "tonyassi/celebrity-1000 (local parquet cache)",
            "path": str(DATASET_DIR),
            "source_split": "train",
            "evaluated_split": "stratified_holdout",
            "split_note": (
                "公开数据集只有 train split；按 seed=42 分层抽取 20% 作为测试 holdout，"
                "非官方 test split。"
            ),
            "split_seed": SPLIT_SEED,
            "holdout_fraction": HOLDOUT_FRACTION,
            "source_image_count": len(dataset),
            "image_count": image_count,
            "class_count": len(model_class_names),
        },
        "environment": {
            "device": device_name,
            "precision": "FP32",
            "batch_size": BATCH_SIZE,
            "warmup_steps": WARMUP_STEPS,
            "num_workers": NUM_WORKERS,
        },
        "preprocessing": {"processor": processor.__class__.__name__},
        "official_reference": {
            "model_card_eval_accuracy_pct": 79.82,
            "model_card_eval_loss": 0.9089,
            "comparability": "仅作参考；模型卡的 evaluation split 与本 holdout 不保证相同。",
        },
        "accuracy": metrics,
        "timing": {
            "model_forward_total_seconds": round(forward_seconds, 6),
            "model_latency_ms_per_image": round(
                forward_seconds / image_count * 1000.0, 6
            ),
            "model_throughput_images_per_second": round(
                image_count / forward_seconds, 6
            ),
            "end_to_end_total_seconds": round(wall_seconds, 6),
            "end_to_end_latency_ms_per_image": round(
                wall_seconds / image_count * 1000.0, 6
            ),
            "end_to_end_throughput_images_per_second": round(
                image_count / wall_seconds, 6),
        },
    }
    if device.type == "cuda":
        result["environment"]["peak_gpu_memory_mib"] = round(
            torch.cuda.max_memory_allocated(device) / 1024**2, 4
        )
    save_result(result)

    print("=" * 60)
    print(f"Top-1 accuracy: {metrics['top1_accuracy_pct']:.2f}%")
    print(f"Top-5 accuracy: {metrics['top5_accuracy_pct']:.2f}%")
    print(f"MRR: {metrics['mean_reciprocal_rank']:.6f}")
    print(f"Macro-F1: {metrics['macro_f1_pct']:.2f}%")
    print(f"Balanced accuracy: {metrics['balanced_accuracy_pct']:.2f}%")
    print(f"Cross-entropy loss: {metrics['cross_entropy_loss']:.6f}")
    print(
        f"模型平均推理时间: {result['timing']['model_latency_ms_per_image']:.4f} ms/image"
    )
    print(
        f"模型吞吐量: {result['timing']['model_throughput_images_per_second']:.2f} images/s"
    )
    print(f"评测结果已更新: {RESULTS_PATH}")
    _test_record.finish(locals())


if __name__ == "__main__":
    main()
