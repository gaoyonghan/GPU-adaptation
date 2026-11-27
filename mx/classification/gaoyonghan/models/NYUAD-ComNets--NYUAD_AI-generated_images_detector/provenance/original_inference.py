#!/usr/bin/env python3
"""Model-card pipeline, frozen replacement dataset, resumable three-class evaluation."""
import argparse
from collections import Counter, defaultdict
import fcntl
import hashlib
import importlib.metadata
import io
import json
import os
from pathlib import Path
import platform
import time
import zipfile

ROOT = Path(__file__).resolve().parent
SPEC = json.loads((ROOT / "models.json").read_text())


def canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def sha(path):
    with open(path, "rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def save(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    os.replace(temporary, path)


def environment():
    actual = {name: importlib.metadata.version(name) for name in SPEC["environment"]}
    if actual != SPEC["environment"]:
        raise RuntimeError(f"环境版本不符，禁止修改其他任务环境: {actual}")
    return {**actual, "python": platform.python_version()}


def dataset_index():
    groups = defaultdict(list)
    identities = {}
    for source, entry in SPEC["dataset"]["archives"].items():
        path = Path(entry["path"])
        stat = path.stat()
        if stat.st_size != entry["size"]:
            raise RuntimeError(f"数据归档大小不符: {path}")
        identities[source] = {"path": str(path), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}
        with zipfile.ZipFile(path) as archive:
            for item in archive.infolist():
                member = item.filename
                if not member.lower().endswith((".jpg", ".jpeg", ".png", ".tif", ".tiff")):
                    continue
                group = Path(member).parent.name if source == "synthbuster" else "real"
                if group in SPEC["dataset"]["group_order"]:
                    groups[group].append({"source": source, "member": member, "crc32": item.CRC, "size": item.file_size})
    names = SPEC["dataset"]["group_order"]
    for group in names:
        expected = 300 if group == "real" else 1000
        if len(groups[group]) != expected:
            raise RuntimeError(f"{group} 原始图像数不符: {len(groups[group])} != {expected}")
        groups[group] = sorted(groups[group], key=lambda row: hashlib.sha256(row["member"].encode()).hexdigest())[:300]
    rows = [dict(groups[group][i], target=label, group=group) for i in range(300) for label, group in enumerate(names)]
    if canonical(rows) != SPEC["dataset"]["selected_index_sha256"]:
        raise RuntimeError("样本选择或标签索引不符")
    return {"archives": identities, "rows": rows, "selected_index_sha256": canonical(rows)}


def metrics(records):
    from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, roc_auc_score
    targets = [row["target"] for row in records]
    predictions = [row["prediction"] for row in records]
    return {"accuracy_percent": 100 * accuracy_score(targets, predictions),
            "macro_f1": f1_score(targets, predictions, labels=[0, 1, 2], average="macro", zero_division=0),
            "macro_ovr_auroc": roc_auc_score(targets, [row["probabilities"] for row in records],
                                            labels=[0, 1, 2], average="macro", multi_class="ovr"),
            "confusion_matrix": confusion_matrix(targets, predictions, labels=[0, 1, 2]).tolist(),
            "label_order": SPEC["labels"], "confusion_matrix_axes": "rows=target, columns=prediction",
            "units": {"accuracy_percent": "%", "macro_f1": "0..1", "macro_ovr_auroc": "0..1"}}


def self_test():
    records = [{"target": i, "prediction": i, "probabilities": [float(j == i) for j in range(3)]}
               for _ in range(2) for i in range(3)]
    result = metrics(records)
    assert result["accuracy_percent"] == 100 and result["macro_f1"] == 1 and result["macro_ovr_auroc"] == 1
    assert result["confusion_matrix"] == [[2, 0, 0], [0, 2, 0], [0, 0, 2]]
    for row in records:
        row["prediction"] = (row["target"] + 1) % 3
        row["probabilities"] = [float(j == row["prediction"]) for j in range(3)]
    assert metrics(records)["accuracy_percent"] == 0 and metrics(records)["macro_f1"] == 0
    print("[72] 三分类指标自检通过")


def run(args):
    import numpy as np
    import torch
    from PIL import Image
    from transformers import AutoImageProcessor, AutoModelForImageClassification, pipeline
    versions = environment()
    model_root = Path(SPEC["model_directory"])
    for name, expected in SPEC["files"].items():
        if sha(model_root / name) != expected:
            raise RuntimeError(f"冻结模型文件已变化: {name}")
    config = json.loads((model_root / "config.json").read_text())
    if [config["id2label"][str(i)] for i in range(3)] != SPEC["labels"]:
        raise RuntimeError("模型类别顺序与清单不符")
    data = dataset_index()
    rows = data["rows"]
    fingerprint = canonical({"manifest": SPEC, "environment": versions, "dataset": data,
                             "runner_sha256": sha(Path(__file__))})
    out = Path(args.output_root)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / ".model.lock", "a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        status_path = out / "status.json"
        prediction_path = out / "predictions.json"
        previous = json.loads(status_path.read_text()) if status_path.exists() else {}
        if previous and previous.get("protocol_fingerprint") != fingerprint:
            raise RuntimeError("模型、数据、环境或脚本改变；请使用新的 OUTPUT_ROOT，旧结果不覆盖")
        if not previous and (prediction_path.exists() or (out / "metrics.json").exists()):
            raise RuntimeError("旧结果缺少协议记录，拒绝覆盖")
        records = []
        if prediction_path.exists():
            saved = json.loads(prediction_path.read_text())
            if saved["protocol_fingerprint"] != fingerprint:
                raise RuntimeError("预测断点指纹不符")
            records = saved["records"]
            if len(records) > len(rows):
                raise RuntimeError("预测数量超出评测集")
            for record, row in zip(records, rows):
                scores = np.asarray(record["probabilities"])
                if any(record[key] != row[key] for key in ("source", "member", "target", "group")):
                    raise RuntimeError("断点样本顺序或标签错误")
                if scores.shape != (3,) or not np.isfinite(scores).all() or (scores < 0).any() or not np.isclose(scores.sum(), 1):
                    raise RuntimeError("断点概率无效")
                if record["prediction"] != int(scores.argmax()):
                    raise RuntimeError("断点预测与概率不一致")
        if previous.get("status") == "complete":
            saved_metrics = json.loads((out / "metrics.json").read_text())
            if len(records) != 900 or saved_metrics["protocol_fingerprint"] != fingerprint:
                raise RuntimeError("完成记录不完整")
            print("[72] 已完整完成，跳过")
            return
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA 不可用，不静默改用 CPU")
        target = args.max_images or 900
        state = {"sequence": 72, "model_id": SPEC["model_id"], "manifest": SPEC,
                 "protocol_fingerprint": fingerprint, "environment": versions,
                 "total_images": 900, "gpu": torch.cuda.get_device_name(0),
                 "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                 "peak_allocated_bytes": previous.get("peak_allocated_bytes", 0)}
        started = time.monotonic()
        elapsed_before = previous.get("elapsed_seconds", 0)

        def status(value):
            state.update(status=value, completed_images=len(records), elapsed_seconds=elapsed_before + time.monotonic() - started)
            save(status_path, state)
            save(out / "audit_summary.json", {"models": [state], "complete": int(value == "complete")})

        def checkpoint():
            save(prediction_path, {"protocol_fingerprint": fingerprint, "records": records})
            status("running")

        save(out / "dataset_index.json", data)
        status("running")
        archives = {}
        try:
            if len(records) < target:
                torch.set_num_threads(2)
                torch.backends.cuda.matmul.allow_tf32 = False
                torch.backends.cudnn.allow_tf32 = False
                torch.backends.cudnn.benchmark = False
                torch.cuda.reset_peak_memory_stats()
                processor = AutoImageProcessor.from_pretrained(model_root, local_files_only=True, use_fast=False)
                model, loading = AutoModelForImageClassification.from_pretrained(
                    model_root, local_files_only=True, use_safetensors=True, output_loading_info=True,
                    torch_dtype=torch.float32, attn_implementation="eager")
                if any(loading.get(key) for key in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")):
                    raise RuntimeError(f"模型未严格完整加载: {loading}")
                model.eval()
                classifier = pipeline("image-classification", model=model, image_processor=processor, device=0)
                label_indices = {name: i for i, name in enumerate(SPEC["labels"])}
                archives = {source: zipfile.ZipFile(entry["path"]) for source, entry in SPEC["dataset"]["archives"].items()}
                for row in rows[len(records):target]:
                    raw = archives[row["source"]].read(row["member"])
                    with Image.open(io.BytesIO(raw)) as image, torch.inference_mode():
                        results = classifier(image.convert("RGB"), top_k=None, function_to_apply="softmax")
                    if len(results) != 3 or {item["label"] for item in results} != set(SPEC["labels"]):
                        raise RuntimeError("模型未返回预期三类别")
                    scores = np.zeros(3, dtype=np.float64)
                    for item in results:
                        scores[label_indices[item["label"]]] = item["score"]
                    if not np.isfinite(scores).all() or (scores < 0).any() or not np.isclose(scores.sum(), 1):
                        raise RuntimeError("模型输出概率无效")
                    records.append({**row, "input_sha256": hashlib.sha256(raw).hexdigest(),
                                    "prediction": int(scores.argmax()), "probabilities": scores.tolist()})
                    if len(records) % 25 == 0 or len(records) == target:
                        checkpoint()
                        print(f"[72] {len(records)}/900", flush=True)
                state["peak_allocated_bytes"] = max(state["peak_allocated_bytes"], torch.cuda.max_memory_allocated())
            if len(records) < 900:
                status("partial_complete")
                print("[72] 小样本通过；不报告小样本指标", flush=True)
                return
            if Counter(row["target"] for row in records) != {0: 300, 1: 300, 2: 300}:
                raise RuntimeError("完整三类覆盖检查失败")
            result = {"sequence": 72, "model_id": SPEC["model_id"], "images": 900,
                      "protocol_fingerprint": fingerprint, "metrics": metrics(records),
                      "fidelity": SPEC["fidelity"], "dataset_relation": SPEC["dataset_relation"],
                      "official_reference": SPEC["official_reference"], "comparison": "not_officially_comparable", "delta": None}
            save(out / "metrics.json", result)
            state["metrics"] = result
            status("complete")
            print(json.dumps(result, indent=2, ensure_ascii=False), flush=True)
        except Exception as exc:
            checkpoint()
            state["error"] = f"{type(exc).__name__}: {exc}"
            status("failed")
            raise
        finally:
            for archive in archives.values():
                archive.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--describe", choices=["72", "72-nyuad-ai-detector"])
    parser.add_argument("--check-environment", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--max-images", type=int, default=0)
    parser.add_argument("--output-root", default="/home/ubuntu/gaoyonghan/gpu/results/nyuad72_v1")
    args = parser.parse_args()
    if args.describe:
        print(json.dumps(SPEC, ensure_ascii=False, indent=2))
    elif args.check_environment:
        print(json.dumps(environment()))
    elif args.self_test:
        self_test()
    else:
        if args.max_images != 0 and not 10 <= args.max_images <= 900:
            parser.error("max-images 必须为 0（900 图全量）或 10..900")
        run(args)


if __name__ == "__main__":
    main()
