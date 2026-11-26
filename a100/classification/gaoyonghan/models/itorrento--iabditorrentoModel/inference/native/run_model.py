#!/usr/bin/env python3
"""评测 itorrento/iabditorrentoModel。"""

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
    _test_record = _start_records(123, __file__, globals())


import json
import os
import time
from datetime import datetime
from pathlib import Path

import torch
from datasets import Dataset as HFDataset
from torch.utils.data import DataLoader, Dataset
from transformers import AutoImageProcessor, AutoModelForImageClassification

from common import (
    balanced_accuracy_from_confusion,
    binary_roc_auc as standard_binary_roc_auc,
)


MODEL_ID = "itorrento/iabditorrentoModel"
PROJECT_ROOT = _PACKAGE_ROOT
MODEL_CACHE_DIR = _WEIGHTS_ROOT
DATASET_DIR = _DATASET_ROOT/'cats_vs_dogs'
RESULTS_PATH = _OUTPUT_ROOT/_portable_os.environ.get('MODEL_PHASE','full')/'m123/metrics.json'
if __name__ == "__main__":
    RESULTS_PATH = _test_record.output / "metrics.json"
BATCH_SIZE = int(_portable_os.environ.get('MODEL_BATCH_SIZE',64))
NUM_WORKERS = int(_portable_os.environ.get('MODEL_NUM_WORKERS',0))
WARMUP_STEPS = _warmup_steps(3)
HOLDOUT_SIZE = 0.2
HOLDOUT_SEED = 42


def load_test_split():
    """加载 test；若上游只有 train，则生成固定的分层测试留出。"""
    if not DATASET_DIR.is_dir():
        raise FileNotFoundError(f"未找到 Cats vs Dogs 数据：{DATASET_DIR}")
    test_files = sorted(DATASET_DIR.rglob("test-*.parquet"))
    if test_files:
        dataset = HFDataset.from_parquet(
            [str(path) for path in test_files],
            cache_dir=_portable_os.environ["HF_DATASETS_CACHE"],
        )
        split_name = "test"
        split_note = "local explicit test parquet split"
    else:
        train_files = sorted(DATASET_DIR.rglob("train-*.parquet"))
        if not train_files:
            raise FileNotFoundError(
                f"{DATASET_DIR} 下未找到 test-*.parquet 或 train-*.parquet"
            )
        train = HFDataset.from_parquet(
            [str(path) for path in train_files],
            cache_dir=_portable_os.environ["HF_DATASETS_CACHE"],
        )
        label_column = "labels" if "labels" in train.column_names else "label"
        dataset = train.train_test_split(
            test_size=HOLDOUT_SIZE,
            seed=HOLDOUT_SEED,
            stratify_by_column=label_column,
        )["test"]
        split_name = "deterministic_20pct_holdout_from_train"
        split_note = (
            "microsoft/cats_vs_dogs has no official test split; stratified holdout "
            f"test_size={HOLDOUT_SIZE}, seed={HOLDOUT_SEED}"
        )

    label_column = "labels" if "labels" in dataset.column_names else "label"
    if "image" not in dataset.column_names or label_column not in dataset.column_names:
        raise ValueError(f"数据列应包含 image 和 labels/label，实际为：{dataset.column_names}")
    names = [str(name).lower() for name in dataset.features[label_column].names]
    if names != ["cat", "dog"]:
        raise ValueError(f"预期标签顺序为 ['cat', 'dog']，实际为：{names}")
    return dataset, label_column, split_name, split_note


class CatsVsDogsDataset(Dataset):
    def __init__(self, dataset, label_column, processor):
        self.dataset = dataset
        self.label_column = label_column
        self.processor = processor

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        example = self.dataset[index]
        pixel_values = self.processor(
            images=example["image"].convert("RGB"), return_tensors="pt"
        )["pixel_values"].squeeze(0)
        return pixel_values, int(example[self.label_column])


def safe_divide(numerator, denominator):
    return numerator / denominator if denominator else 0.0


def binary_metrics(labels, predictions, positive_scores):
    tn = fp = fn = tp = 0
    for label, prediction in zip(labels, predictions):
        if label == 0 and prediction == 0:
            tn += 1
        elif label == 0 and prediction == 1:
            fp += 1
        elif label == 1 and prediction == 0:
            fn += 1
        else:
            tp += 1
    precision_cat = safe_divide(tn, tn + fn)
    recall_cat = safe_divide(tn, tn + fp)
    precision_dog = safe_divide(tp, tp + fp)
    recall_dog = safe_divide(tp, tp + fn)
    f1_cat = safe_divide(2 * precision_cat * recall_cat, precision_cat + recall_cat)
    f1_dog = safe_divide(2 * precision_dog * recall_dog, precision_dog + recall_dog)

    roc_auc = standard_binary_roc_auc(labels, positive_scores)
    image_count = len(labels)
    return {
        "image_count": image_count,
        "accuracy_pct": round(safe_divide(tp + tn, image_count) * 100.0, 4),
        "balanced_accuracy_pct": round(100.0 * balanced_accuracy_from_confusion([[tn, fp], [fn, tp]]), 4),
        "macro_f1_pct": round((f1_cat + f1_dog) * 50.0, 4),
        "roc_auc": None if roc_auc is None else round(roc_auc, 6),
        "confusion_matrix_rows_true_cat_dog_cols_pred_cat_dog": [[tn, fp], [fn, tp]],
        "per_class": {
            "cat": {
                "precision_pct": round(precision_cat * 100.0, 4),
                "recall_pct": round(recall_cat * 100.0, 4),
                "f1_pct": round(f1_cat * 100.0, 4),
            },
            "dog": {
                "precision_pct": round(precision_dog * 100.0, 4),
                "recall_pct": round(recall_dog * 100.0, 4),
                "f1_pct": round(f1_dog * 100.0, 4),
            },
        },
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
    print("数据集: Cats vs Dogs")
    processor = AutoImageProcessor.from_pretrained(
        str(MODEL_CACHE_DIR), local_files_only=True
    )
    model = AutoModelForImageClassification.from_pretrained(
        str(MODEL_CACHE_DIR), local_files_only=True
    ).to(device)
    model.eval()
    if model.config.num_labels != 2:
        raise ValueError(f"模型分类头应输出 2 类，实际为 {model.config.num_labels} 类")

    dataset, label_column, split_name, split_note = load_test_split()
    wrapped_dataset = CatsVsDogsDataset(dataset, label_column, processor)
    dataloader = _test_record.loader(DataLoader, 
        wrapped_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=device.type == "cuda",
    )
    print(f"评测图片数量: {len(wrapped_dataset)}")
    print(f"评测划分: {split_name}")
    print(f"说明: {split_note}")
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

    labels, predictions, positive_scores = [], [], []
    forward_seconds = 0.0
    wall_start = time.perf_counter()
    with torch.inference_mode():
        for batch_index, (pixel_values, batch_labels) in enumerate(dataloader, start=1):
            _test_record.before_batch(locals())
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
            labels.extend(batch_labels.tolist())
            predictions.extend(logits.argmax(dim=1).cpu().tolist())
            positive_scores.extend(logits.softmax(dim=1)[:, 1].cpu().tolist())
            if batch_index % 10 == 0:
                print(f"已处理: {min(batch_index * BATCH_SIZE, len(wrapped_dataset))}/{len(wrapped_dataset)}")
            _test_record.capture(locals())
    wall_seconds = time.perf_counter() - wall_start

    metrics = binary_metrics(labels, predictions, positive_scores)
    image_count = len(wrapped_dataset)
    result = {
        "recorded_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "model_id": MODEL_ID,
        "model_path": str(MODEL_CACHE_DIR),
        "script_path": str(Path(__file__).resolve()),
        "model": {
            "id": MODEL_ID,
            "parameter_count_millions": round(sum(p.numel() for p in model.parameters()) / 1e6, 4),
            "num_labels": model.config.num_labels,
            "id2label": {str(key): value for key, value in model.config.id2label.items()},
        },
        "dataset": {
            "name": "Cats vs Dogs",
            "source": "microsoft/cats_vs_dogs",
            "path": str(DATASET_DIR),
            "split": split_name,
            "split_note": split_note,
            "label_column": label_column,
            "class_names": ["cat", "dog"],
            "image_count": image_count,
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
        "accuracy": {"top1_pct": metrics["accuracy_pct"]},
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
    print(f"Accuracy: {metrics['accuracy_pct']:.2f}%")
    print(f"Balanced accuracy: {metrics['balanced_accuracy_pct']:.2f}%")
    print(f"Macro-F1: {metrics['macro_f1_pct']:.2f}%")
    print(f"ROC-AUC: {metrics['roc_auc']}")
    print(f"模型平均推理时间: {result['timing']['model_latency_ms_per_image']:.4f} ms/image")
    print(f"模型吞吐量: {result['timing']['model_throughput_images_per_second']:.2f} images/s")
    print(f"评测结果已更新: {RESULTS_PATH}")
    _test_record.finish(locals())


if __name__ == "__main__":
    main()
