#!/usr/bin/env python3
"""Frozen Oxford-IIIT Pet replacement baseline for seven local HF model IDs."""
import argparse
from collections import Counter
from contextlib import contextmanager
import fcntl
import hashlib
import importlib.metadata
import io
import json
import os
from pathlib import Path
import sys
import tarfile
import tempfile
import time
import traceback

HERE = Path(__file__).resolve().parent
MANIFEST = json.loads((HERE / "models.json").read_text())
RUNTIME = (HERE / ".runtime").resolve()


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8388608), b""):
            h.update(block)
    return h.hexdigest()


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp = tempfile.mkstemp(dir=path.parent, prefix=".writing-")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
            f.write("\n")
        os.replace(temp, path)
    finally:
        if os.path.exists(temp):
            os.unlink(temp)


@contextmanager
def lock(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        yield


def check_environment():
    from _direct_full_policy import record_environment
    return record_environment(MANIFEST["versions"])



def verify_assets(root, files):
    for name, expected in files.items():
        p = root / name
        assert p.stat().st_size == expected["size"] and digest(p) == expected["sha256"], f"Changed asset: {p}"


def prepare_data():
    """Reuse the frozen test order and labels; resolve only machine-local paths."""
    cache = RUNTIME / "pet_test"
    data = json.loads((cache / "index.json").read_text())
    for row in data["rows"]:
        row["path"] = str(cache / "images" / (row["id"] + ".jpg"))
    spec = MANIFEST["dataset"]
    source = Path(spec["root"])
    data["sources"] = {n: [str(source / n), (source / n).stat().st_size,
                           (source / n).stat().st_mtime_ns] for n in spec["assets"]}
    data["protocol"] = spec
    data.pop("fingerprint", None)
    data["fingerprint"] = fingerprint(data)
    return data


def read_predictions(path, rows):
    records = [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []
    expected = {r["id"]: r["target"] for r in rows}
    assert len(records) == len({r["id"] for r in records}), "Duplicate resume IDs"
    for r in records:
        assert r["id"] in expected and r["target"] == expected[r["id"]]
        assert r["prediction"] in (0, 1) and 0 <= r["p_dog"] <= 1
    return records


def metrics_for(records):
    from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
    y = [r["target"] for r in records]
    pred = [r["prediction"] for r in records]
    return {"accuracy_percent": 100 * accuracy_score(y, pred), "balanced_accuracy_percent": 100 * balanced_accuracy_score(y, pred),
            "macro_f1": f1_score(y, pred, average="macro", zero_division=0),
            "dog_precision": precision_score(y, pred, zero_division=0), "dog_recall": recall_score(y, pred, zero_division=0),
            "auroc_dog_positive": roc_auc_score(y, [r["p_dog"] for r in records]),
            "confusion_matrix_cat_dog": confusion_matrix(y, pred, labels=[0, 1]).tolist()}


def run(spec, args):
    output = args.output_root / spec["model_key"]
    with lock(output / ".lock"):
        status_path = output / "status.json"
        old = json.loads(status_path.read_text()) if status_path.exists() else None
        status = {"sequence": spec["sequence"], "model_id": spec["model_id"], "manifest_snapshot": spec}
        started = time.time()
        current_fingerprint = None
        try:
            versions = check_environment()
            model_root = Path(MANIFEST["model_root"]) / spec["model_id"]
            verify_assets(model_root, spec["files"])
            data = prepare_data()
            assert spec["config"]["id2label"] == {"0": "cat", "1": "dog"}
            current_fingerprint = fingerprint({"model": spec, "dataset": data["fingerprint"], "versions": versions, "code": digest(__file__), "device": args.device})
            if old and old.get("run_fingerprint") and old["run_fingerprint"] != current_fingerprint:
                raise RuntimeError("Protocol changed: use NEW OUTPUT_ROOT; old evidence preserved")
            status.update(run_fingerprint=current_fingerprint, dataset_fingerprint=data["fingerprint"], runtime_versions=versions,
                          python_version=sys.version, requested_images=args.max_images, total_images=len(data["rows"]), started_at=started)
            if args.preflight:
                print(f"[{spec['sequence']}] preflight_ready: {len(data['rows'])} images")
                return True
            limit = len(data["rows"]) if not args.max_images else min(args.max_images, len(data["rows"]))
            from classification_capture import snapshot_gt
            snapshot_gt(output, data["rows"][:limit])
            records = read_predictions(output / "predictions.jsonl", data["rows"])
            if old and old["state"] == "complete":
                saved = json.loads((output / "metrics.json").read_text())
                assert saved["run_fingerprint"] == current_fingerprint and saved["evaluation_images"] == len(data["rows"]) == len(records)
                print(f"[{spec['sequence']}] 已全量完成，跳过", flush=True)
                return True
            if old and old["state"] == "partial_complete" and args.max_images and len(records) >= limit:
                print(f"[{spec['sequence']}] 已完成当前小样本范围，跳过", flush=True)
                return True
            import torch
            from PIL import Image
            from transformers import AutoImageProcessor, AutoModelForImageClassification
            assert args.device.startswith("cuda") and torch.cuda.is_available(), "CUDA required; no silent CPU fallback"
            torch.set_num_threads(2)
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.cuda.reset_peak_memory_stats()
            status.update(state="running", gpu_name=torch.cuda.get_device_name(), device=args.device)
            save(status_path, status)
            processor = AutoImageProcessor.from_pretrained(model_root, local_files_only=True, use_fast=False)
            model, loading = AutoModelForImageClassification.from_pretrained(model_root, local_files_only=True,
                use_safetensors=spec["weight_file"].endswith(".safetensors"), output_loading_info=True)
            assert not any(loading.get(k) for k in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")), str(loading)
            model.to(args.device).eval()
            done = {r["id"] for r in records}
            pending = [r for r in data["rows"][:limit] if r["id"] not in done]
            with (output / "predictions.jsonl").open("a") as handle:
                for i in range(0, len(pending), spec["batch_size"]):
                    batch = pending[i:i + spec["batch_size"]]
                    images = []
                    for row in batch:
                        raw = Path(row["path"]).read_bytes()
                        assert hashlib.sha256(raw).hexdigest() == row["sha256"], f"Changed image: {row['id']}"
                        with Image.open(io.BytesIO(raw)) as im:
                            images.append(im.convert("RGB"))
                    inputs = processor(images=images, return_tensors="pt").to(args.device)
                    with torch.inference_mode():
                        logits = model(**inputs).logits.float()
                    assert logits.shape == (len(batch), 2) and torch.isfinite(logits).all()
                    preds = logits.argmax(-1).cpu().tolist()
                    scores = logits.softmax(-1)[:, 1].cpu().tolist()
                    for row, pred, score in zip(batch, preds, scores):
                        handle.write(json.dumps({"id": row["id"], "target": row["target"], "prediction": pred, "p_dog": score}, allow_nan=False) + "\n")
                    handle.flush()
                    os.fsync(handle.fileno())
                    completed = len(records) + i + len(batch)
                    if i == 0 or completed % 200 < spec["batch_size"] or i + len(batch) == len(pending):
                        print(f"[{spec['sequence']}] {completed}/{limit}", flush=True)
            records = read_predictions(output / "predictions.jsonl", data["rows"])
            assert {r["id"] for r in data["rows"][:limit]} <= {r["id"] for r in records}
            if not args.max_images:
                assert len(records) == len(data["rows"])
                save(output / "metrics.json", {"metrics": metrics_for(records), "evaluation_images": len(records),
                     "dataset_relation": "public_replacement", "fidelity": "reference_baseline", "reference_check": "not_officially_comparable",
                     "official_reference": None, "run_fingerprint": current_fingerprint})
            status.update(state="partial_complete" if args.max_images else "complete", completed_images=len(records), finished_at=time.time(),
                          elapsed_seconds=time.time() - started, peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                          processor=processor.to_dict(), backend_flags={"cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
                          "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32, "cudnn_deterministic": torch.backends.cudnn.deterministic})
            save(status_path, status)
            print(f"[{spec['sequence']}] {status['state']} -> {output}", flush=True)
            return True
        except Exception as error:
            traceback.print_exc()
            if old and old.get("run_fingerprint") and old["run_fingerprint"] != current_fingerprint:
                return False
            status.update(state="failed", error=str(error), traceback=traceback.format_exc(), elapsed_seconds=time.time() - started)
            save(status_path, status)
            return False


def summarize(root):
    rows = []
    for spec in MANIFEST["models"]:
        path = root / spec["model_key"]
        state = json.loads((path / "status.json").read_text()) if (path / "status.json").exists() else {"state": "not_started"}
        metric = json.loads((path / "metrics.json").read_text()) if (path / "metrics.json").exists() else {}
        rows.append({"sequence": spec["sequence"], "model_id": spec["model_id"], "state": state["state"],
                     "completed_images": state.get("completed_images", 0), "metrics": metric.get("metrics"),
                     "dataset_relation": spec["dataset_relation"], "fidelity": spec["fidelity"], "case_type": spec["case_type"],
                     "special_tags": spec["special_tags"], "reference_check": "not_officially_comparable",
                     "elapsed_seconds": state.get("elapsed_seconds"), "peak_allocated_bytes": state.get("peak_allocated_bytes"), "error": state.get("error")})
    result = {"scope": MANIFEST["scope"], "models": rows, "counts": dict(Counter(r["state"] for r in rows)),
              "fidelity_counts": dict(Counter(r["fidelity"] for r in rows)), "dataset_relation_counts": dict(Counter(r["dataset_relation"] for r in rows)),
              "case_type_counts": dict(Counter(r["case_type"] for r in rows)), "special_tag_counts": dict(Counter(t for r in rows for t in r["special_tags"])),
              "note": "仅本批 7 个唯一模型 ID；不代表 7 种独立架构或全项目完成数。"}
    save(root / "audit_summary.json", result)
    print(json.dumps({"counts": result["counts"], "summary": str(root / "audit_summary.json")}, ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--describe", action="store_true")
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--check-environment", action="store_true")
    parser.add_argument("--summarize", action="store_true")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-images", type=int, default=10)
    parser.add_argument("--output-root", type=Path, default=Path(MANIFEST["result_root"]))
    args = parser.parse_args()
    if args.max_images < 0:
        parser.error("max-images must be >=0")
    if args.check_environment:
        print(check_environment()); return 0
    if args.summarize:
        summarize(args.output_root); return 0
    if len(MANIFEST["models"]) != 1:
        raise ValueError("Expected one model in this folder")
    specs = MANIFEST["models"]
    if args.describe:
        print(json.dumps({"model": specs[0], "dataset": MANIFEST["dataset"]}, ensure_ascii=False, indent=2)); return 0
    return 0 if run(specs[0], args) else 1


if __name__ == "__main__":
    raise SystemExit(main())
