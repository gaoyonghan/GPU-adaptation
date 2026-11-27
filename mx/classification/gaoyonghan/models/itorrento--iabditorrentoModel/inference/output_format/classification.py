"""Classification exchange and scoring contracts; no model inference or GPU imports."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

SCHEMA = "model-eval/score-classification/v1"
EXCHANGE_SCHEMA = "model-eval/classification-predictions/v1"
POLICY = {
    "task": "single_label_classification", "top_k": [1, 5],
    "top5_when_classes_below_five": "not_applicable",
    "top5_when_ranking_missing": "unavailable",
    "ranking": "preserve_native_order_no_new_tie_break",
    "macro_labels": "observed_ground_truth_classes", "zero_division": 0,
    "unit": "fraction", "sample_weight": "uniform",
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def integer(value):
    return isinstance(value, int) and not isinstance(value, bool)


def validate_predictions(rows, categories):
    require(isinstance(rows, list) and rows, "empty or invalid prediction list")
    ids = set()
    for row in rows:
        sid = row.get("sample_id")
        require(isinstance(sid, str) and sid and sid not in ids, "invalid/duplicate prediction sample_id")
        ids.add(sid)
        pred = row.get("prediction")
        require(integer(pred) and pred in categories, "prediction outside declared class space")
        ranked = row.get("topk")
        require(isinstance(ranked, list) and ranked and ranked[0] == pred, "topk must start with native prediction")
        require(all(integer(v) and v in categories for v in ranked) and len(set(ranked)) == len(ranked), "invalid/repeated topk classes")
        scores = row.get("scores")
        if scores is not None:
            require(isinstance(scores, list) and len(scores) == len(ranked), "scores must align with topk")
            require(all(isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) for v in scores), "nonfinite scores")
            require(all(a >= b for a, b in zip(scores, scores[1:])), "scores disagree with native ranking")
    return ids


def validate_annotation(annotation):
    require(isinstance(annotation, dict), "annotation must be an object")
    cats = annotation.get("categories")
    require(isinstance(cats, list) and cats, "full declared category space required")
    cat_ids = [r.get("id") for r in cats]
    require(all(integer(v) and v >= 0 for v in cat_ids) and len(set(cat_ids)) == len(cat_ids), "invalid/duplicate categories")
    require(all(isinstance(c.get("name"), str) and c["name"] for c in cats), "category names required")
    category_ids = set(cat_ids)
    samples = annotation.get("samples")
    require(isinstance(samples, list) and samples, "empty selected GT scope")
    ids = set()
    for row in samples:
        sid = row.get("sample_id")
        require(isinstance(sid, str) and sid and sid not in ids, "invalid/duplicate GT sample_id")
        ids.add(sid)
        require(integer(row.get("label")) and row["label"] in category_ids, "GT label outside declared classes")
    return ids, set(cat_ids)


def load_bundle(bundle):
    bundle = Path(bundle)
    p = read(bundle / "protocol.json")
    require(p.get("schema") == SCHEMA and p.get("evaluation") == POLICY, "classification scoring protocol mismatch")
    require(p.get("scoring_ready") is True, "classification bundle not ready")
    hashes = p.get("artifact_sha256", {})
    require(set(hashes) == {"annotations.json", "predictions.json"}, "input hash manifest incomplete")
    for name, value in hashes.items():
        require(file_hash(bundle / name) == value, "input hash mismatch: " + name)
    annotation, predictions = read(bundle / "annotations.json"), read(bundle / "predictions.json")
    ids, categories = validate_annotation(annotation)
    require(validate_predictions(predictions, categories) == ids, "prediction/GT coverage mismatch")
    completed = p.get("completed_sample_ids", [])
    require(len(completed) == len(ids) and set(completed) == ids, "completed sample coverage mismatch")
    require(p.get("prediction_count") == len(predictions), "prediction count mismatch")
    return annotation, predictions, p


def publish(output, annotation, predictions, metadata):
    import tempfile
    import shutil
    output = Path(output)
    require(not output.exists(), "output exists; use a new output directory")
    ids, categories = validate_annotation(annotation)
    require(validate_predictions(predictions, categories) == ids, "prediction/GT coverage mismatch")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix="." + output.name + "-", dir=output.parent))
    try:
        write(temporary / "annotations.json", annotation)
        write(temporary / "predictions.json", predictions)
        protocol = dict(metadata, schema=SCHEMA, scoring_ready=True, evaluation=POLICY,
                        completed_sample_ids=sorted(ids), prediction_count=len(predictions),
                        artifact_sha256={n: file_hash(temporary / n) for n in ("annotations.json", "predictions.json")})
        write(temporary / "protocol.json", protocol)
        temporary.rename(output)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return protocol
