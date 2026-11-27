#!/usr/bin/env python3
"""Offline, auditable model 174 inference on the visually reviewed 3-class proxy."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import importlib.metadata
import io
import json
import os
from pathlib import Path
import sys
import time
import traceback

for key in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE"):
    os.environ[key] = "1"
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

REVISION = "49c5c2cc437566416b56364095cdc9e6b357e7cf"
WEIGHT_SHA256 = "c25b1650605916840a2e116e856ff8a962ac354248e090e2ca5067659f5111af"
LABELS = {0: "skinny", 1: "ordinary", 2: "overweight", 3: "hulk"}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path: Path, obj: dict) -> None:
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def read_rows(index: Path, root: Path) -> list[dict]:
    rows = [json.loads(line) for line in index.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(rows) != 36 or Counter(row["target"] for row in rows) != {"skinny": 12, "ordinary": 12, "overweight": 12}:
        raise ValueError("Expected the frozen 36-image proxy: 12 images per class")
    if len({row["id"] for row in rows}) != 36:
        raise ValueError("Duplicate sample IDs")
    for row in rows:
        if row["target_id"] not in (0, 1, 2) or LABELS[row["target_id"]] != row["target"]:
            raise ValueError(f"Target mapping mismatch: {row['id']}")
        path = root / row["member"]
        if not path.is_file() or sha256(path) != row["sha256"]:
            raise ValueError(f"Missing or changed image: {path}")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--phase", choices=("smoke", "full"), required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args()
    if args.batch_size < 1 or args.threads < 1:
        raise ValueError("batch-size and threads must be positive")

    import numpy as np
    from PIL import Image
    import torch
    import transformers
    from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, precision_recall_fscore_support

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable; no silent CPU fallback")
    device = torch.device(args.device)
    torch.set_num_threads(args.threads)
    index = args.dataset_root / "index.jsonl"
    rows = read_rows(index, args.dataset_root)
    model_files = {
        "pytorch_model.bin": sha256(args.model / "pytorch_model.bin"),
        "config.json": sha256(args.model / "config.json"),
        "preprocessor_config.json": sha256(args.model / "preprocessor_config.json"),
    }
    if model_files["pytorch_model.bin"] != WEIGHT_SHA256:
        raise ValueError("Local model weight differs from the verified published revision")
    config = json.loads((args.model / "config.json").read_text())
    if {int(k): v for k, v in config["id2label"].items()} != LABELS:
        raise ValueError("Model label order changed")
    protocol = {
        "sequence": 174, "model_id": "glazzova/body_type", "model_revision": REVISION,
        "model_files_sha256": model_files, "dataset_relation": "public_replacement",
        "dataset_name": "model174_proxy_3class_v1", "dataset_index_sha256": sha256(index),
        "dataset_image_hashes": [row["sha256"] for row in rows],
        "labels": LABELS, "missing_gold_class": "hulk", "label_basis": "single-reviewer visual labels",
        "preprocessing": "publisher AutoImageProcessor, use_fast=False", "dtype": "float32",
        "batch_size": args.batch_size, "device": str(device), "cpu_threads": args.threads if device.type == "cpu" else None,
        "python": sys.version, "torch": torch.__version__, "transformers": transformers.__version__,
        "pillow": importlib.metadata.version("Pillow"), "scikit_learn": importlib.metadata.version("scikit-learn"),
    }
    fingerprint = hashlib.sha256(json.dumps(protocol, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    out = args.output_root / args.phase
    if args.phase == "full":
        smoke_status_path = args.output_root / "smoke" / "status.json"
        if not smoke_status_path.is_file():
            raise ValueError("A completed smoke run is required before the full run")
        smoke_status = json.loads(smoke_status_path.read_text())
        if smoke_status.get("status") != "complete" or smoke_status.get("protocol_fingerprint") != fingerprint:
            raise ValueError("Smoke run is incomplete or protocol fingerprint changed")
    out.mkdir(parents=True, exist_ok=False)
    write_json(out / "status.json", {"status": "running", "phase": args.phase, "protocol_fingerprint": fingerprint, "protocol": protocol})

    try:
        load_start = time.perf_counter()
        model, loading = transformers.AutoModelForImageClassification.from_pretrained(
            str(args.model), local_files_only=True, output_loading_info=True
        )
        if any(loading.get(key) for key in ("missing_keys", "unexpected_keys", "mismatched_keys")):
            raise ValueError(f"Checkpoint did not load exactly: {loading}")
        processor = transformers.AutoImageProcessor.from_pretrained(
            str(args.model), local_files_only=True, use_fast=False
        )
        model = model.to(device=device, dtype=torch.float32).eval()
        model_load_seconds = time.perf_counter() - load_start
        selected = rows[:10] if args.phase == "smoke" else rows
        def read_images(chunk: list[dict]) -> list[Image.Image]:
            images = []
            for row in chunk:
                with Image.open(args.dataset_root / row["member"]) as original:
                    images.append(original.convert("RGB"))
            return images
        def synchronize() -> None:
            if device.type == "cuda":
                torch.cuda.synchronize(device)
        with torch.inference_mode():
            warm_start = time.perf_counter()
            warm_images = read_images(selected[:args.batch_size])
            warm_inputs = processor(images=warm_images, return_tensors="pt").to(device)
            _ = model(**warm_inputs).logits
            synchronize()
            warmup_seconds = time.perf_counter() - warm_start
            records = []
            forward_seconds = 0.0
            e2e_seconds = 0.0
            with (out / "batches.jsonl").open("w", encoding="utf-8") as batch_file:
                for offset in range(0, len(selected), args.batch_size):
                    chunk = selected[offset:offset + args.batch_size]
                    synchronize()
                    batch_start = time.perf_counter()
                    images = read_images(chunk)
                    inputs = processor(images=images, return_tensors="pt").to(device)
                    synchronize()
                    forward_start = time.perf_counter()
                    logits = model(**inputs).logits
                    synchronize()
                    forward = time.perf_counter() - forward_start
                    probabilities = torch.softmax(logits.float(), dim=-1).cpu().numpy()
                    if probabilities.shape != (len(chunk), 4) or not np.isfinite(probabilities).all():
                        raise ValueError("Invalid native four-class output")
                    batch_records = []
                    for row, prob in zip(chunk, probabilities):
                        top = np.argsort(-prob, kind="stable").tolist()
                        prediction = int(top[0])
                        batch_records.append({
                            "id": row["id"], "source_row_idx": row["source_row_idx"],
                            "target": row["target"], "target_id": row["target_id"],
                            "prediction": prediction, "predicted_label": LABELS[prediction],
                            "probabilities": {LABELS[i]: float(prob[i]) for i in range(4)},
                            "confidence": float(prob[prediction]),
                        })
                    synchronize()
                    e2e = time.perf_counter() - batch_start
                    forward_seconds += forward
                    e2e_seconds += e2e
                    batch = {"offset": offset, "images": len(chunk), "forward_seconds": forward,
                             "inference_e2e_seconds": e2e, "records": batch_records}
                    batch_file.write(json.dumps(batch, ensure_ascii=False, allow_nan=False) + "\n")
                    batch_file.flush()
                    records.extend(batch_records)
                    print(f"{args.phase}: {len(records)}/{len(selected)}", flush=True)
        if [r["id"] for r in records] != [r["id"] for r in selected]:
            raise ValueError("Prediction order changed")
        if args.phase == "full":
            smoke_records = [json.loads(line) for line in (args.output_root / "smoke" / "predictions.jsonl").read_text().splitlines()]
            for a, b in zip(records[:10], smoke_records, strict=True):
                if a["id"] != b["id"] or a["prediction"] != b["prediction"]:
                    raise ValueError("Full predictions differ from smoke run")
                if any(abs(a["probabilities"][name] - b["probabilities"][name]) > 1e-5 for name in LABELS.values()):
                    raise ValueError("Full probabilities differ from smoke run")
        (out / "predictions.jsonl").write_text(
            "".join(json.dumps(r, ensure_ascii=False, allow_nan=False) + "\n" for r in records), encoding="utf-8"
        )
        timing = {
            "device": str(device), "batch_size": args.batch_size, "images": len(records),
            "model_load_seconds": model_load_seconds, "warmup_seconds": warmup_seconds,
            "forward_seconds": forward_seconds, "inference_e2e_seconds": e2e_seconds,
            "forward_ms_per_image": 1000 * forward_seconds / len(records),
            "e2e_ms_per_image": 1000 * e2e_seconds / len(records),
            "forward_method": "perf_counter synchronized" if device.type == "cuda" else "perf_counter",
            "e2e_scope": "image read, publisher processor, model forward, native four-class softmax and CPU probabilities",
        }
        write_json(out / "timing.json", timing)
        result = {
            "status": "complete", "phase": args.phase, "sequence": 174,
            "model_id": "glazzova/body_type", "dataset_relation": "public_replacement",
            "fidelity": "reference_baseline", "images": len(records), "protocol_fingerprint": fingerprint,
            "device": str(device), "prediction_counts": dict(Counter(r["predicted_label"] for r in records)),
            "timing": timing,
        }
        if args.phase == "full":
            truth = [r["target_id"] for r in records]
            pred = [r["prediction"] for r in records]
            precision, recall, f1, support = precision_recall_fscore_support(
                truth, pred, labels=[0, 1, 2], zero_division=0
            )
            matrix = confusion_matrix(truth, pred, labels=[0, 1, 2, 3])[:3].tolist()
            metrics = {
                "accuracy": float(accuracy_score(truth, pred)),
                "balanced_accuracy": float(balanced_accuracy_score(truth, pred)),
                "macro_f1_over_three_gold_classes": float(f1.mean()),
                "correct": sum(a == b for a, b in zip(truth, pred)),
                "images": len(records),
                "per_class": {LABELS[i]: {"precision": float(precision[i]), "recall": float(recall[i]),
                                          "f1": float(f1[i]), "support": int(support[i])} for i in range(3)},
                "confusion_matrix": matrix,
                "confusion_rows_gold": [LABELS[i] for i in range(3)],
                "confusion_columns_predicted": [LABELS[i] for i in range(4)],
                "hulk_predictions_counted_wrong": sum(p == 3 for p in pred),
                "comparison_to_model_card": None,
                "note": "Small, visually labeled three-class proxy; not an official four-class score.",
            }
            write_json(out / "metrics.json", metrics)
            result["metrics"] = metrics
        write_json(out / "result.json", result)
        write_json(out / "status.json", {"status": "complete", "phase": args.phase,
                                          "protocol_fingerprint": fingerprint, "protocol": protocol,
                                          "images": len(records), "device": str(device)})
        print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    except Exception as exc:
        write_json(out / "status.json", {"status": "failed", "phase": args.phase,
                                          "protocol_fingerprint": fingerprint,
                                          "error": str(exc), "traceback": traceback.format_exc()})
        raise


if __name__ == "__main__":
    main()
