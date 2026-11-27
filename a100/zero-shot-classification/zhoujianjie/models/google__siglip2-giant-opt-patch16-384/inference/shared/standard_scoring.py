"""Shared JSON scoring package; Top-1 accuracy in fraction units."""
from __future__ import annotations

import json
from pathlib import Path

try:
    from .classification_contract import publish, load_bundle, write, file_hash
except ImportError:
    from classification_contract import publish, load_bundle, write, file_hash


def records(path):
    rows = []
    with Path(path).open() as handle:
        for line in handle:
            if not line.strip():
                raise ValueError("Blank prediction/annotation record")
            rows.append(json.loads(line))
    return rows


def export_jsonl(prediction_path, annotation_path, output, categories, metadata):
    predictions = []
    for row in records(prediction_path):
        # Never synthesize missing ranks, confidence values, or probabilities.
        ranked = row.get("top5")
        if ranked is None:
            ranked = [row["prediction"]]
        predictions.append({"sample_id": row["sample_id"], "prediction": row["prediction"],
                            "topk": ranked, "scores": row.get("scores")})
    annotation = {"categories": categories,
                  "samples": [{"sample_id": row["sample_id"], "label": row["target"]}
                              for row in records(annotation_path)]}
    return publish(output, annotation, predictions, metadata)


def evaluate(bundle):
    annotation, predictions, protocol = load_bundle(bundle)
    lookup = {row["sample_id"]: row for row in predictions}
    samples = annotation["samples"]
    ordered = [lookup[row["sample_id"]] for row in samples]
    top1 = sum(row["label"] == pred["prediction"] for row, pred in zip(samples, ordered)) / len(samples)
    result = {"schema": "model-eval/metrics/v1", "task": "zero_shot_image_classification",
              "model": protocol.get("model"), "dataset": protocol.get("dataset"),
              "metric_unit": "fraction", "primary_metric_names": ["top1_accuracy"],
              "metrics": {"top1_accuracy": top1},
              "metric_status": {"top1_accuracy": "computed"},
              "scope": {"images": len(samples), "categories": len(annotation["categories"])},
              "qualification": protocol.get("qualification"), "acceptance_assessed": False,
              "input_sha256": protocol["artifact_sha256"],
              "protocol_sha256": file_hash(Path(bundle) / "protocol.json"), "inference_executed": False}
    write(Path(bundle) / "metrics.json", result)
    return result
