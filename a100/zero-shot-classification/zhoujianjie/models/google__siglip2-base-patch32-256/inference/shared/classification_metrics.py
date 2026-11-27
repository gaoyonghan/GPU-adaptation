#!/usr/bin/env python3
"""CPU-only, strict single-label Top-1/Top-5 scoring of JSONL predictions."""
from __future__ import annotations

import json
from pathlib import Path


def read_records(path: Path) -> dict[str, dict]:
    records = {}
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                raise ValueError(f"Blank JSONL record: {path}:{line_no}")
            row = json.loads(line)
            identifier = row.get("sample_id")
            if not isinstance(identifier, str) or not identifier or identifier in records:
                raise ValueError(f"Missing/duplicate sample_id: {path}:{line_no}")
            records[identifier] = row
    if not records:
        raise ValueError(f"Empty records: {path}")
    return records


def label(value, class_count: int) -> int:
    if type(value) is not int or not 0 <= value < class_count:
        raise ValueError(f"Invalid class index {value!r} for {class_count} classes")
    return value


def score(predictions: dict[str, dict], annotations: dict[str, dict], class_count: int) -> dict:
    if type(class_count) is not int or class_count < 2:
        raise ValueError("class_count must be an integer >= 2")
    if not annotations or predictions.keys() != annotations.keys():
        raise ValueError("Prediction IDs and annotation IDs must match exactly and be nonempty")
    correct1 = correct5 = 0
    for identifier, annotation in annotations.items():
        target = label(annotation["target"], class_count)
        row = predictions[identifier]
        prediction = label(row["prediction"], class_count)
        top5 = row["top5"]
        if not isinstance(top5, list):
            raise ValueError(f"top5 must be a list: {identifier}")
        values = [label(value, class_count) for value in top5]
        if len(values) != min(5, class_count) or len(set(values)) != len(values) or values[0] != prediction:
            raise ValueError(f"Invalid ranked top5: {identifier}")
        correct1 += prediction == target
        correct5 += target in values
    count = len(annotations)
    return {
        "sample_count": count,
        "class_count": class_count,
        "top1_correct": correct1,
        "top5_correct": correct5,
        "top1_percent": 100 * correct1 / count,
        "top5_percent": 100 * correct5 / count,
        "top5_effective_k": min(5, class_count),
    }
