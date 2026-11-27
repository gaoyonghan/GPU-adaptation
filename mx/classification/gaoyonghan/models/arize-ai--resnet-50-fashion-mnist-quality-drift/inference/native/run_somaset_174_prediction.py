#!/usr/bin/env python3
"""Run model 174 on the frozen SOMASet prediction-only sample."""
from __future__ import annotations

import argparse
import io
import json
import time
import zipfile
from collections import Counter
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    import torch
    from PIL import Image
    from transformers import AutoImageProcessor, AutoModelForImageClassification

    rows = [json.loads(line) for line in args.index.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(rows) != 600 or any(row.get("target") is not None for row in rows):
        raise SystemExit("Frozen SOMASet index must contain exactly 600 prediction-only rows")
    device = torch.device(args.device if args.device.startswith("cuda") and torch.cuda.is_available() else "cpu")
    args.output.mkdir(parents=True, exist_ok=True)

    load_start = time.perf_counter()
    processor = AutoImageProcessor.from_pretrained(str(args.model), local_files_only=True)
    model = AutoModelForImageClassification.from_pretrained(str(args.model), local_files_only=True)
    model.eval().to(device)
    model_load_seconds = time.perf_counter() - load_start
    labels = model.config.id2label

    def label_for(index: int) -> str:
        return labels.get(str(index), labels.get(index, str(index)))

    records = []
    forward_seconds = 0.0
    e2e_start = time.perf_counter()
    with zipfile.ZipFile(args.archive) as archive, torch.inference_mode():
        for offset in range(0, len(rows), args.batch_size):
            chunk = rows[offset:offset + args.batch_size]
            images = [Image.open(io.BytesIO(archive.read(row["member"]))).convert("RGB") for row in chunk]
            inputs = processor(images=images, return_tensors="pt")
            inputs = {key: value.to(device) for key, value in inputs.items()}
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            start = time.perf_counter()
            logits = model(**inputs).logits
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            forward_seconds += time.perf_counter() - start
            probabilities = torch.softmax(logits.float(), dim=-1).cpu()
            for row, probability in zip(chunk, probabilities):
                top = torch.argsort(probability, descending=True).tolist()
                records.append({
                    "id": row["id"],
                    "member": row["member"],
                    "prototype": row["prototype"],
                    "garment": row["garment"],
                    "pose": row["pose"],
                    "target": None,
                    "prediction": int(top[0]),
                    "predicted_label": label_for(top[0]),
                    "confidence": float(probability[top[0]]),
                    "top5": [{"id": int(index), "label": label_for(index), "probability": float(probability[index])} for index in top[:5]],
                })
    e2e_seconds = time.perf_counter() - e2e_start

    by_label = Counter(record["predicted_label"] for record in records)
    by_prototype = {prototype: dict(Counter(record["predicted_label"] for record in records if record["prototype"] == prototype)) for prototype in sorted({record["prototype"] for record in records})}
    (args.output / "predictions.jsonl").write_text("".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records), encoding="utf-8")
    result = {
        "sequence": 174,
        "model_id": "glazzova/body_type",
        "model_path": str(args.model),
        "dataset": "SOMASet",
        "dataset_relation": "public_replacement",
        "status": "prediction_only",
        "accuracy": None,
        "reason": "No auditable three-class ground truth exists for the mixed somatotype prototypes.",
        "images": len(records),
        "device": str(device),
        "batch_size": args.batch_size,
        "model_load_seconds": model_load_seconds,
        "forward_seconds": forward_seconds,
        "inference_e2e_seconds": e2e_seconds,
        "forward_ms_per_image": 1000 * forward_seconds / len(records),
        "forward_images_per_second": len(records) / forward_seconds,
        "e2e_images_per_second": len(records) / e2e_seconds,
        "predicted_label_counts": dict(by_label),
        "predicted_label_counts_by_prototype": by_prototype,
        "target_mapping": None,
    }
    (args.output / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
