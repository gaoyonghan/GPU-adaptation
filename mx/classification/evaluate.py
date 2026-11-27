#!/usr/bin/env python3
"""Evaluate standardized classification outputs using the existing scoring rules.

Output conversion is performed by each model's inference.py. This script only
validates completed bundles, computes metrics, and saves evaluation results.
"""
from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent  # Relative inputs stay task-relative.

SCHEMA = "model-eval/score-classification/v1"

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

def evaluate(bundle):
    import sklearn
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support

    annotation, predictions, protocol = load_bundle(bundle)
    lookup = {r["sample_id"]: r for r in predictions}
    samples = annotation["samples"]
    truth = [r["label"] for r in samples]
    ordered = [lookup[r["sample_id"]] for r in samples]
    predicted = [r["prediction"] for r in ordered]
    labels = sorted(set(truth))
    precision, recall, f1, support = precision_recall_fscore_support(truth, predicted, labels=labels, zero_division=0)
    acc = float(accuracy_score(truth, predicted))
    metrics = dict(accuracy=acc, top1_accuracy=acc, top5_accuracy=None,
                   balanced_accuracy=float(recall.mean()),
                   macro_precision_over_evaluated_target_classes=float(precision.mean()),
                   macro_recall_over_evaluated_target_classes=float(recall.mean()),
                   macro_f1_over_evaluated_target_classes=float(f1.mean()),
                   weighted_f1=float((f1 * support).sum() / support.sum()))
    if len(annotation["categories"]) < 5:
        top5_status = "not_applicable_fewer_than_five_classes"
    elif not all(len(r["topk"]) >= 5 for r in ordered):
        top5_status = "unavailable_native_output_missing_top5"
    else:
        # Preserve the native ranking/tie behavior. The hit vector is evaluated
        # by sklearn; no invented logits or probabilities are reconstructed.
        hit = [t in r["topk"][:5] for t, r in zip(truth, ordered)]
        metrics["top5_accuracy"] = float(accuracy_score([True] * len(hit), hit))
        top5_status = "computed"
    result = dict(schema="model-eval/metrics/v1", task="image_classification", protocol=protocol["schema"],
                  model=protocol.get("model"), dataset=protocol.get("dataset"), metric_unit="fraction", metrics=metrics,
                  metric_status={"top5_accuracy": top5_status},
                  per_class=[dict(category_id=int(c), support=int(n), precision=float(p), recall=float(r), f1=float(f))
                             for c, n, p, r, f in zip(labels, support, precision, recall, f1)],
                  scope=dict(images=len(samples), completed_images=len(ordered), categories=len(annotation["categories"])),
                  evaluation=protocol["evaluation"], qualification=protocol.get("qualification"),
                  native_execution=protocol.get('native_execution', []),
                  limitations=protocol.get("limitations", []),
                  evaluator=dict(name="scikit-learn.metrics", version=sklearn.__version__,
                                 functions=["accuracy_score", "precision_recall_fscore_support"],
                                 top5="accuracy_score of GT membership in native ordered top5",
                                 source_sha256=file_hash(inspect.getfile(accuracy_score)),
                                 reference="https://scikit-learn.org/stable/modules/model_evaluation.html"),
                  input_sha256=protocol["artifact_sha256"], protocol_sha256=file_hash(Path(bundle) / "protocol.json"),
                  inference_executed=False)
    write(Path(bundle) / "metrics.json", result)
    return result

def resolve_input(path):
    return (path if path.is_absolute() else HERE / path).resolve()

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--bundle', type=Path, help='One standardized scoring bundle')
    source.add_argument('--run-dir', type=Path, help='One run containing inference_output.json')
    args = parser.parse_args()
    task = 'classification'
    if args.bundle:
        bundles = [resolve_input(args.bundle)]
        output = None
    else:
        root = resolve_input(args.run_dir)
        manifest = json.loads((root / 'inference_output.json').read_text())
        if manifest.get('state') != 'completed' or manifest.get('task') != task:
            raise SystemExit('Inference output is incomplete or belongs to another task')
        bundles = [(root / p).resolve() for p in manifest['bundles']]
        if any(not p.is_relative_to(root) for p in bundles):
            raise SystemExit('Bundle path escapes its run directory')
        output = root / 'evaluation_summary.json'
    if not bundles or len(bundles) != len(set(bundles)):
        raise SystemExit('No unique scoring bundles provided')
    results = []
    for bundle in bundles:
        protocol = json.loads((bundle / 'protocol.json').read_text())
        if protocol.get('schema') != SCHEMA:
            raise SystemExit('Unsupported scoring protocol: ' + str(protocol.get('schema')))
        evaluate(bundle)
        results.append({'bundle': str(bundle), 'metrics': json.loads((bundle / 'metrics.json').read_text())})
    if output:
        output.write_text(json.dumps({'state': 'completed', 'task': task, 'results': results}, ensure_ascii=False, indent=2))
    print(json.dumps({'state': 'completed', 'task': task, 'scored_bundles': len(results)}, ensure_ascii=False))

if __name__ == '__main__':
    main()
