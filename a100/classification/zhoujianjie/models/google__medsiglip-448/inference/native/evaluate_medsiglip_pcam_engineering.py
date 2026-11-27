#!/usr/bin/env python3
"""Tune a leakage-free MedSigLIP zero-shot PCam engineering diagnostic.

This is deliberately not an official MedSigLIP metric reproduction. Prompt and
preprocessing choices are made on the official PCam validation split, then the
chosen configuration is evaluated once on the PCam test split.
"""

from __future__ import annotations
from model_paths import legacy as _legacy, legacy_str as _legacy_str, relative as _relative, revision as _revision, weight as _weight, result as _result

import argparse
import hashlib
import json
import platform
import time
from collections import OrderedDict
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
MODEL_ID = "google/medsiglip-448"
MODEL_REVISION = "9cea28a1a1195f665105faa6e8544c112fd960a4"
MODEL_ROOT = ROOT / "models/google/medsiglip-448"
DATA_ROOT = ROOT / "datasets/evaluation/biomedclip_pcam_official_original"
PREFLIGHT_PATH = ROOT / "evaluation_audit/medsiglip_pcam_engineering_preflight.json"
RESULT_PATH = ROOT / "results/medsiglip_pcam_engineering_optimization/summary.json"
SEED = 20260914


# This registry is fixed in source before looking at the PCam validation labels.
# The two published PCam prompt routes are kept separate from the additional
# medically direct engineering candidates so the selected route remains auditable.
PROMPT_CONFIGS: OrderedDict[str, dict[str, Any]] = OrderedDict([
    ("legacy_proxy", {
        "source": "previous local MedSigLIP proxy diagnostic",
        "negative": [
            "a histopathology image without metastatic tumor tissue",
            "benign lymph node tissue",
        ],
        "positive": [
            "a histopathology image containing metastatic tumor tissue",
            "metastatic tumor tissue",
        ],
    }),
    ("clip_benchmark_pcam", {
        "source": "CLIP benchmark PCam class names and English templates",
        "negative": [
            "a histopathology slide showing lymph node",
            "histopathology image of lymph node",
        ],
        "positive": [
            "a histopathology slide showing lymph node containing metastatic tumor tissue",
            "histopathology image of lymph node containing metastatic tumor tissue",
        ],
    }),
    ("biomedclip_pcam", {
        "source": "BiomedCLIP paper Supplementary Table 10 PCam route",
        "negative": [
            "this is an image of normal lymph node",
            "normal lymph node presented in image",
        ],
        "positive": [
            "this is an image of lymph node metastasis",
            "lymph node metastasis presented in image",
        ],
    }),
    ("direct_tissue", {
        "source": "predeclared engineering prompt pair",
        "negative": ["normal lymph node tissue", "benign lymph node tissue"],
        "positive": [
            "lymph node metastasis",
            "lymph node tissue containing metastatic tumor",
        ],
    }),
    ("breast_carcinoma", {
        "source": "predeclared engineering prompt pair using PCam clinical context",
        "negative": [
            "benign lymph node without metastatic breast carcinoma",
            "normal lymph node histopathology",
        ],
        "positive": [
            "metastatic breast carcinoma in a lymph node",
            "lymph node histopathology with metastatic carcinoma",
        ],
    }),
    ("pathology_finding", {
        "source": "predeclared engineering prompt pair",
        "negative": [
            "histopathology of benign lymph node tissue",
            "histopathology negative for metastatic carcinoma",
        ],
        "positive": [
            "histopathology of metastatic tumor in lymph node tissue",
            "histopathology positive for metastatic carcinoma",
        ],
    }),
    ("direct_ensemble", {
        "source": "predeclared union of direct PCam descriptions",
        "negative": [
            "normal lymph node tissue",
            "benign lymph node tissue",
            "normal lymph node histopathology",
            "histopathology of benign lymph node tissue",
        ],
        "positive": [
            "lymph node metastasis",
            "lymph node tissue containing metastatic tumor",
            "metastatic breast carcinoma in a lymph node",
            "histopathology of metastatic tumor in lymph node tissue",
        ],
    }),
])

PREPROCESSING_CONFIGS = OrderedDict([
    ("transformers_pil_bicubic", {
        "description": "checkpoint's Transformers processor default: PIL bicubic resize to 448",
        "model_card_exact": False,
    }),
    ("model_card_bilinear_no_antialias_torch_compatible", {
        "description": (
            "448px bilinear resize, antialias disabled, half-pixel coordinates and uint8 "
            "quantization; PyTorch-compatible implementation because TensorFlow is unavailable"
        ),
        "model_card_exact": False,
        "difference_from_model_card": "resize engine is PyTorch rather than tf.image.resize",
    }),
])


def now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def registry_sha256() -> str:
    encoded = json.dumps(PROMPT_CONFIGS, ensure_ascii=False, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def feature_tensor(value: Any):
    """Handle both tensor and Transformers model-output return conventions."""
    import torch

    if isinstance(value, torch.Tensor):
        return value
    for key in ("pooler_output", "text_embeds", "image_embeds", "last_hidden_state"):
        candidate = getattr(value, key, None)
        if isinstance(candidate, torch.Tensor):
            return candidate[:, 0] if key == "last_hidden_state" else candidate
    if isinstance(value, (tuple, list)) and value and isinstance(value[0], torch.Tensor):
        return value[0]
    raise TypeError(f"cannot extract feature tensor from {type(value).__name__}")


def split_paths(split: str) -> tuple[Path, Path]:
    return (
        DATA_ROOT / f"camelyonpatch_level_2_split_{split}_x.h5",
        DATA_ROOT / f"camelyonpatch_level_2_split_{split}_y.h5",
    )


def inspect_h5(split: str) -> dict[str, Any]:
    import h5py

    x_path, y_path = split_paths(split)
    with h5py.File(x_path, "r") as x_file, h5py.File(y_path, "r") as y_file:
        x, y = x_file["x"], y_file["y"]
        labels = np.asarray(y).reshape(-1).astype(np.int64)
        if x.shape != (32768, 96, 96, 3) or y.shape[0] != 32768:
            raise RuntimeError(f"unexpected PCam {split} shape: x={x.shape}, y={y.shape}")
        counts = np.bincount(labels, minlength=2).tolist()
    return {
        "x_path": str(_relative(x_path,ROOT)),
        "y_path": str(_relative(y_path,ROOT)),
        "x_bytes": x_path.stat().st_size,
        "y_bytes": y_path.stat().st_size,
        "x_sha256": sha256_file(x_path),
        "y_sha256": sha256_file(y_path),
        "image_shape": [32768, 96, 96, 3],
        "label_counts": counts,
    }


def preflight() -> dict[str, Any]:
    errors: list[str] = []
    receipt_path = MODEL_ROOT / "download_receipt.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8")) if receipt_path.is_file() else {}
    if receipt.get("revision") != MODEL_REVISION:
        errors.append("checkpoint receipt revision mismatch")
    for name in ("model.safetensors", "config.json", "preprocessor_config.json"):
        if not (MODEL_ROOT / name).is_file():
            errors.append(f"missing checkpoint artifact: {name}")
    datasets: dict[str, Any] = {}
    for split in ("valid", "test"):
        x_path, y_path = split_paths(split)
        missing = [str(path) for path in (x_path, y_path) if not path.is_file()]
        if missing:
            errors.extend(f"missing PCam {split} file: {path}" for path in missing)
        else:
            datasets[split] = inspect_h5(split)

    payload = {
        "generated_at": now(),
        "status": "failed" if errors else "passed",
        "evaluation_class": "custom_engineering_diagnostic_only",
        "strict_official_acceptance": False,
        "changes_strict_acceptance_count": False,
        "rtx4090_stage1_engineering_only": True,
        "model": {
            "model_id": MODEL_ID,
            "revision": MODEL_REVISION,
            "local_root": str(_relative(MODEL_ROOT,ROOT)),
            "checkpoint_sha256": sha256_file(MODEL_ROOT / "model.safetensors")
            if (MODEL_ROOT / "model.safetensors").is_file() else None,
        },
        "dataset": {
            "name": "PatchCamelyon (PCam)",
            "source": "Zenodo record 2546921",
            "selection_split": "official validation, all 32,768 patches",
            "final_split": "official test, all 32,768 patches, evaluated once after selection",
            "files": datasets,
        },
        "protocol": {
            "task": "zero-shot binary image classification",
            "selection_metric": "validation AUROC",
            "threshold_selection": "validation Youden J (maximizes sensitivity + specificity - 1)",
            "final_metrics": ["AUROC", "accuracy", "balanced accuracy", "macro F1", "sensitivity", "specificity"],
            "prompt_registry_sha256": registry_sha256(),
            "prompt_candidates": PROMPT_CONFIGS,
            "preprocessing_candidates": PREPROCESSING_CONFIGS,
            "test_set_tuning_forbidden": True,
        },
        "official_evidence_gap": {
            "same_checkpoint_official_pcam_reference": None,
            "reason": (
                "MedSigLIP's model card reports other private/unspecified medical evaluation tasks; "
                "it does not publish a same-checkpoint PCam reference and complete PCam protocol"
            ),
            "model_card_path": "models/google/medsiglip-448/README.md",
            "published_pathology_auc_range_diagnostic_context_only": [0.721, 0.933],
            "published_pathology_average_auc_diagnostic_context_only": 0.870,
        },
        "user_decision": (
            "When no public same-protocol metric exists, retain the RTX 4090 result as a self-defined "
            "stage-one engineering baseline with an explicit note"
        ),
        "errors": errors,
    }
    PREFLIGHT_PATH.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print("PREFLIGHT MODEL->DATASET->REFERENCE MAPPING", flush=True)
    print(json.dumps(payload, indent=2, ensure_ascii=False), flush=True)
    if errors:
        raise RuntimeError("preflight failed: " + "; ".join(errors))
    return payload


class MedSigLipAdapter:
    def __init__(self) -> None:
        import torch
        from transformers import AutoModel, AutoProcessor

        self.torch = torch
        self.processor = AutoProcessor.from_pretrained(MODEL_ROOT, local_files_only=True)
        self.model = AutoModel.from_pretrained(
            MODEL_ROOT,
            local_files_only=True,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
        ).eval().cuda()
        self.text_length = int(self.model.config.text_config.max_position_embeddings)

    def text_features(self, prompts: list[str]):
        torch = self.torch
        values = self.processor(
            text=prompts,
            padding="max_length",
            truncation=True,
            max_length=self.text_length,
            return_tensors="pt",
        )
        values = {key: value.cuda(non_blocking=True) for key, value in values.items()}
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
            features = feature_tensor(self.model.get_text_features(**values))
        return torch.nn.functional.normalize(features.float(), dim=-1)

    def image_features(self, image_values: np.ndarray, preprocessing: str):
        torch = self.torch
        if preprocessing == "transformers_pil_bicubic":
            from PIL import Image

            images = [Image.fromarray(value, mode="RGB") for value in image_values]
            pixels = self.processor(images=images, return_tensors="pt")["pixel_values"]
            pixels = pixels.cuda(non_blocking=True)
        elif preprocessing == "model_card_bilinear_no_antialias_torch_compatible":
            pixels = torch.from_numpy(np.ascontiguousarray(image_values)).permute(0, 3, 1, 2)
            pixels = pixels.cuda(non_blocking=True).float()
            pixels = torch.nn.functional.interpolate(
                pixels,
                size=(448, 448),
                mode="bilinear",
                align_corners=False,
                antialias=False,
            )
            # Match the model-card example's `.numpy().astype(np.uint8)` before
            # SigLIP rescaling and normalization to [-1, 1].
            pixels = pixels.to(torch.uint8).to(torch.float32).mul_(2.0 / 255.0).sub_(1.0)
        else:
            raise ValueError(f"unknown preprocessing: {preprocessing}")
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
            features = feature_tensor(self.model.get_image_features(pixel_values=pixels))
        return torch.nn.functional.normalize(features.float(), dim=-1)


def prompt_classifiers(adapter: MedSigLipAdapter) -> OrderedDict[str, Any]:
    torch = adapter.torch
    classifiers: OrderedDict[str, Any] = OrderedDict()
    for name, config in PROMPT_CONFIGS.items():
        per_class = []
        for label in ("negative", "positive"):
            features = adapter.text_features(config[label])
            per_class.append(torch.nn.functional.normalize(features.mean(dim=0), dim=-1))
        classifiers[name] = torch.stack(per_class, dim=1)
    return classifiers


def score_split(
    adapter: MedSigLipAdapter,
    split: str,
    preprocessing: str,
    classifiers: OrderedDict[str, Any],
    batch_size: int,
) -> tuple[np.ndarray, OrderedDict[str, np.ndarray], float]:
    import h5py

    x_path, y_path = split_paths(split)
    score_parts: dict[str, list[np.ndarray]] = {name: [] for name in classifiers}
    started = time.perf_counter()
    with h5py.File(x_path, "r") as x_file, h5py.File(y_path, "r") as y_file:
        x, y = x_file["x"], y_file["y"]
        targets = np.asarray(y).reshape(-1).astype(np.int64)
        for offset in range(0, len(x), batch_size):
            features = adapter.image_features(x[offset:offset + batch_size], preprocessing)
            for name, classifier in classifiers.items():
                logits = features @ classifier
                score_parts[name].append((logits[:, 1] - logits[:, 0]).cpu().numpy())
            seen = min(offset + batch_size, len(x))
            if seen % 4096 < batch_size:
                print(f"{split.upper()} {preprocessing} {seen:,}/{len(x):,}", flush=True)
    scores = OrderedDict((name, np.concatenate(parts)) for name, parts in score_parts.items())
    return targets, scores, time.perf_counter() - started


def threshold_metrics(targets: np.ndarray, scores: np.ndarray, threshold: float) -> dict[str, Any]:
    from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score

    predictions = (scores >= threshold).astype(np.int64)
    confusion = confusion_matrix(targets, predictions, labels=[0, 1])
    tn, fp, fn, tp = confusion.ravel()
    return {
        "threshold": float(threshold),
        "accuracy_percent": round(100 * accuracy_score(targets, predictions), 4),
        "balanced_accuracy_percent": round(100 * balanced_accuracy_score(targets, predictions), 4),
        "macro_f1_percent": round(100 * f1_score(targets, predictions, average="macro"), 4),
        "specificity_percent": round(100 * tn / (tn + fp), 4),
        "sensitivity_percent": round(100 * tp / (tp + fn), 4),
        "predicted_positive_percent": round(100 * predictions.mean(), 4),
        "confusion_matrix_rows_target_columns_prediction": confusion.tolist(),
    }


def auc_with_ci(targets: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    from sklearn.metrics import roc_auc_score

    auc = float(roc_auc_score(targets, scores))
    positive = int(targets.sum())
    negative = int(len(targets) - positive)
    q1 = auc / (2.0 - auc)
    q2 = 2.0 * auc * auc / (1.0 + auc)
    variance = (
        auc * (1.0 - auc)
        + (positive - 1) * (q1 - auc * auc)
        + (negative - 1) * (q2 - auc * auc)
    ) / (positive * negative)
    standard_error = float(np.sqrt(max(variance, 0.0)))
    return {
        "auroc_fraction": auc,
        "auroc_percent": round(100 * auc, 4),
        "hanley_mcneil_95ci_low_percent": round(100 * max(0.0, auc - 1.96 * standard_error), 4),
        "hanley_mcneil_95ci_high_percent": round(100 * min(1.0, auc + 1.96 * standard_error), 4),
        "auroc_above_random_at_95_percent_confidence": auc - 1.96 * standard_error > 0.5,
    }


def validation_result(targets: np.ndarray, scores: np.ndarray) -> dict[str, Any]:
    from sklearn.metrics import roc_curve

    fpr, tpr, thresholds = roc_curve(targets, scores)
    youden = tpr - fpr
    best = np.flatnonzero(np.isclose(youden, youden.max(), rtol=0.0, atol=1e-12))
    # Deterministic tie break: closest threshold to the zero-shot default.
    index = int(best[np.argmin(np.abs(thresholds[best]))])
    selected_threshold = float(thresholds[index])
    return {
        **auc_with_ci(targets, scores),
        "zero_threshold": threshold_metrics(targets, scores, 0.0),
        "validation_selected_threshold": threshold_metrics(targets, scores, selected_threshold),
        "youden_j": round(float(youden[index]), 8),
    }


def run(batch_size: int, preflight_payload: dict[str, Any]) -> dict[str, Any]:
    import torch

    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.reset_peak_memory_stats()
    started_at = now()
    overall_started = time.perf_counter()

    adapter = MedSigLipAdapter()
    classifiers = prompt_classifiers(adapter)
    validation: OrderedDict[str, Any] = OrderedDict()
    validation_targets: np.ndarray | None = None

    for preprocessing in PREPROCESSING_CONFIGS:
        targets, scores_by_prompt, elapsed = score_split(
            adapter, "valid", preprocessing, classifiers, batch_size,
        )
        if validation_targets is not None and not np.array_equal(validation_targets, targets):
            raise RuntimeError("validation labels changed between preprocessing passes")
        validation_targets = targets
        validation[preprocessing] = {"elapsed_seconds": round(elapsed, 4), "prompts": OrderedDict()}
        for prompt_name, scores in scores_by_prompt.items():
            validation[preprocessing]["prompts"][prompt_name] = validation_result(targets, scores)

    candidates = [
        (
            result["auroc_fraction"],
            preprocessing,
            prompt_name,
            result,
        )
        for preprocessing, prep_result in validation.items()
        for prompt_name, result in prep_result["prompts"].items()
    ]
    _, selected_preprocessing, selected_prompt, selected_validation = max(
        candidates,
        key=lambda item: (item[0], -list(PREPROCESSING_CONFIGS).index(item[1]), -list(PROMPT_CONFIGS).index(item[2])),
    )
    threshold = selected_validation["validation_selected_threshold"]["threshold"]

    selected_classifier = OrderedDict([(selected_prompt, classifiers[selected_prompt])])
    test_targets, test_scores_by_prompt, test_elapsed = score_split(
        adapter, "test", selected_preprocessing, selected_classifier, batch_size,
    )
    test_scores = test_scores_by_prompt[selected_prompt]
    test_auc = auc_with_ci(test_targets, test_scores)
    test = {
        **test_auc,
        "zero_threshold": threshold_metrics(test_targets, test_scores, 0.0),
        "validation_frozen_threshold": threshold_metrics(test_targets, test_scores, threshold),
        "elapsed_seconds": round(test_elapsed, 4),
        "evaluated_once_after_selection": True,
    }
    prior_auc = 76.7946
    result = {
        "status": "completed_engineering_diagnostic_only",
        "strict_official_acceptance": False,
        "changes_strict_acceptance_count": False,
        "rtx4090_stage1_engineering_baseline": True,
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "seed": SEED,
        "started_at": started_at,
        "finished_at": now(),
        "preflight": str(_relative(PREFLIGHT_PATH,ROOT)),
        "preflight_status": preflight_payload["status"],
        "prompt_registry_sha256": registry_sha256(),
        "selection": {
            "split": "PCam official validation",
            "primary_metric": "AUROC",
            "selected_preprocessing": selected_preprocessing,
            "selected_prompt": selected_prompt,
            "selected_prompts": PROMPT_CONFIGS[selected_prompt],
            "selected_threshold_from_validation_youden_j": threshold,
        },
        "validation_candidates": validation,
        "test": test,
        "comparison_to_previous_local_proxy": {
            "previous_test_auroc_percent": prior_auc,
            "optimized_test_auroc_percent": test_auc["auroc_percent"],
            "absolute_delta_percentage_points": round(test_auc["auroc_percent"] - prior_auc, 4),
            "caution": "the old route did not use validation selection and used only default preprocessing",
        },
        "random_baseline_assessment": {
            "auroc_random_percent": 50.0,
            "test_auroc_95ci_excludes_random": test_auc["auroc_above_random_at_95_percent_confidence"],
            "engineering_effect_is_significant": test_auc["auroc_above_random_at_95_percent_confidence"],
        },
        "limitations": [
            "PCam is not a published same-checkpoint MedSigLIP evaluation route.",
            "The TensorFlow resize package is unavailable; the model-card-like candidate uses the compatible PyTorch operation.",
            "Validation-selected prompt and threshold must be frozen unchanged for the later P800 comparison.",
        ],
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0),
            "batch_size": batch_size,
            "peak_gpu_memory_mb": round(torch.cuda.max_memory_allocated() / 1024**2, 2),
        },
        "elapsed_seconds": round(time.perf_counter() - overall_started, 4),
    }
    RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULT_PATH.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": result["status"],
        "selected_preprocessing": selected_preprocessing,
        "selected_prompt": selected_prompt,
        "validation_auroc_percent": selected_validation["auroc_percent"],
        "test_auroc_percent": test_auc["auroc_percent"],
        "test_balanced_accuracy_percent": test["validation_frozen_threshold"]["balanced_accuracy_percent"],
        "test_auroc_delta_vs_old_percentage_points": result["comparison_to_previous_local_proxy"]["absolute_delta_percentage_points"],
        "output": str(RESULT_PATH),
    }, indent=2, ensure_ascii=False), flush=True)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true", help="run GPU evaluation after successful preflight")
    parser.add_argument("--batch-size", type=int, default=16)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    PREFLIGHT_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = preflight()
    if not args.run:
        print("Preflight passed. No GPU evaluation started; pass --run to evaluate.", flush=True)
        return 0
    run(args.batch_size, payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
