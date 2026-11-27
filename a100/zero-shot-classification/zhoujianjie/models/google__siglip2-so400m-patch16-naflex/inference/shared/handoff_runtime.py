#!/usr/bin/env python3
"""Per-model entries for the first frozen ImageNet hand-off batch.

The pretrained model adapters, templates and image preprocessing are reused
from the existing evaluators. This module separates inference from CPU scoring.
No model is loaded until the checkpoint/dataset/protocol/reference preflight passes.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import importlib
import importlib.metadata
import json
import os
import platform
import sys
import time
from datetime import datetime
from pathlib import Path

sys.dont_write_bytecode = True
MODEL_DIR = Path(__file__).resolve().parents[2]
CONTRIBUTOR = MODEL_DIR.parents[1]
BUNDLE = MODEL_DIR.parents[3]
DATASET_ROOT = Path(os.environ.get("ZERO_SHOT_DATASET_ROOT", BUNDLE / "dataset"))


def model_directory(model_id: str) -> Path:
    return CONTRIBUTOR / "models" / model_id.replace("/", "__")


def digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def save(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def configure_legacy():
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    legacy = MODEL_DIR / "inference/shared/legacy"
    if str(legacy) not in sys.path:
        sys.path.insert(0, str(legacy))
    stage2 = importlib.import_module("validate_stage2_models")
    b01 = importlib.import_module("evaluate_b01")
    official = importlib.import_module("evaluate_imagenet1k_official")
    weights = lambda model_id: Path(os.environ["ZERO_SHOT_WEIGHTS"]) if model_id == model_config(MODEL_DIR)["model_id"] else model_directory(model_id) / "weights"
    stage2.ROOT = b01.ROOT = official.ROOT = BUNDLE
    stage2.MODEL_ROOT = b01.MODEL_ROOT = CONTRIBUTOR / "models"
    stage2.local_dir = b01.model_dir = weights
    official.DATA_DIR = DATASET_ROOT / "raw/imagenet1k_validation/data"
    official.CLASS_NAMES_FILE = legacy / "class_names.py"
    official.PROMPTS_FILE = legacy / "prompt_engineering_constants.py"
    return b01, official


def model_config(directory: Path) -> dict:
    return json.loads((directory / "model_config.json").read_text(encoding="utf-8"))


def verify_dataset(official, deep_hash: bool) -> dict:
    """Equivalent metadata checks, with paths relative to the portable Dataset/."""
    import pyarrow.parquet as pq
    directory = official.DATA_DIR
    shards = sorted(directory.glob("validation-*.parquet"))
    if [path.name for path in shards] != [f"validation-{i:05d}-of-00014.parquet" for i in range(14)]:
        raise ValueError("Expected exactly 14 official validation shards")
    rows = 0
    schemas = set()
    files = []
    for path in shards:
        parquet = pq.ParquetFile(path)
        rows += parquet.metadata.num_rows
        schemas.add(str(parquet.schema_arrow))
        metadata = directory.parent / ".cache/huggingface/download/data" / f"{path.name}.metadata"
        lines = metadata.read_text().splitlines()
        if len(lines) < 2 or lines[0] != official.claim_spec.DATASET["revision"]:
            raise ValueError(f"Dataset revision metadata mismatch: {path.name}")
        record = {"name": path.name, "bytes": path.stat().st_size, "expected_sha256": lines[1]}
        if deep_hash:
            record["actual_sha256"] = digest(path)
            if record["actual_sha256"] != record["expected_sha256"]:
                raise ValueError(f"Dataset content hash mismatch: {path.name}")
        files.append(record)
    if rows != 50000 or len(schemas) != 1:
        raise ValueError("Official dataset row count/schema mismatch")
    from shared.standard_dataset import StandardDataset
    normalized = StandardDataset(DATASET_ROOT / "standardized/imagenet1k_validation")
    if normalized.metadata["source"] != official.claim_spec.DATASET or normalized.metadata["sample_count"] != rows or len(normalized.categories) != 1000:
        raise ValueError("Normalized index does not match the official ImageNet split")
    return {**official.claim_spec.DATASET, "rows_verified": rows, "files": files,
            "deep_hash_checked": deep_hash, "standard_index": normalized.validate()}


def preflight(directory: Path, *, dataset=None, deep_hash=False) -> dict:
    config = model_config(directory)
    mapping = {key: config[key] for key in ("model_id", "revision", "handoff_status", "historical_task")}
    mapping.update({"dataset": config.get("official_dataset"), "reference": config.get("official_claim"), "errors": []})
    errors = mapping["errors"]
    if config["handoff_status"] != "first_batch_framework_prepared":
        errors.extend(config["remaining_work"])
        mapping["status"] = "blocked"
        return mapping
    if config.get("standard_dataset") != "dataset/standardized/imagenet1k_validation":
        errors.append("统一数据索引未按模型配置固定")
    weights = Path(os.environ.get("ZERO_SHOT_WEIGHTS", directory / "weights"))
    if not weights.is_dir():
        errors.append("权重目录缺失；本地软链接在A100上必须替换为实际文件")
    else:
        try:
            receipt = json.loads((weights / "download_receipt.json").read_text())
            if (receipt.get("revision") or receipt.get("resolved_revision")) != config["revision"]:
                errors.append("checkpoint revision与冻结配置不一致")
            if not any(weights.glob("*.safetensors")) and not any(weights.glob("*.bin")):
                errors.append("缺少可加载的模型权重")
            for name, expected in config["model_asset_sha256"].items():
                if not (weights / name).is_file() or digest(weights / name) != expected:
                    errors.append(f"模型配置或tokenizer摘要不一致：{name}")
        except (OSError, ValueError) as error:
            errors.append(f"模型下载凭证无法读取：{error}")
    for name, expected in config["framework_sha256"].items():
        if not (BUNDLE / name).is_file() or digest(BUNDLE / name) != expected:
            errors.append(f"冻结评测代码摘要不一致：{name}")
    try:
        _, official = configure_legacy()
        spec = official.claim_spec.MODEL_SPECS[config["model_id"]]
        if spec != config["official_claim"] or spec["reference_status"] != "verified":
            errors.append("同checkpoint官方协议或指标参考值不一致")
        history = json.loads((directory / "inference/provenance/imagenet1k_4090.json").read_text())
        if history.get("model_revision") != config["revision"] or history.get("dataset") != config["official_dataset"]:
            errors.append("历史官方结果的checkpoint/dataset与交接配置不一致")
        historical_claim = history.get("claim", {})
        if any(historical_claim.get(key) != spec.get(key) for key in ("protocol", "resolution", "official_top1_percent", "reference_status")) or any(spec.get(key) != value for key, value in historical_claim.items()):
            errors.append("历史官方结果的协议与当前冻结参考不一致")
        if spec.get("official_top5_percent") is not None:
            top5 = history.get("official_metric_alignment", {}).get("top5_percent", {})
            if top5.get("reference_status") != "verified" or top5.get("official_value") != spec["official_top5_percent"] or top5.get("evidence") != spec.get("top5_evidence"):
                errors.append("历史记录缺少匹配的官方Top-5参考")
        if dataset is None:
            dataset = verify_dataset(official, deep_hash)
        if any(dataset.get(key) != value for key, value in config["official_dataset"].items()):
            errors.append("官方数据集版本/split/规模不匹配")
        mapping["dataset_check"] = dataset
    except (OSError, ValueError, KeyError, RuntimeError, ImportError) as error:
        errors.append(str(error))
    mapping["status"] = "pass" if not errors else "blocked"
    mapping["weight_content_sha256_recomputed"] = False
    mapping["environment_compatibility_checked"] = False
    mapping["gpu_used"] = False
    return mapping


def inference_main(directory: Path) -> int:
    parser = argparse.ArgumentParser(description="冻结ImageNet原始协议；默认完整推理")
    parser.add_argument("--run", action="store_true", default=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--text-batch-size", type=int, default=512)
    args = parser.parse_args()
    if args.batch_size < 1 or args.text_batch_size < 1:
        parser.error("batch sizes must be positive")
    # Deep dataset verification is required before a real GPU run.
    smoke_limit = int(os.environ.get("ZERO_SHOT_SMOKE_LIMIT", "0"))
    check = preflight(directory, deep_hash=args.run and not smoke_limit)
    print(json.dumps(check, ensure_ascii=False, indent=2), flush=True)
    if check["status"] != "pass":
        return 2
    if not args.run:
        return 0
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    config = model_config(directory)
    stamp = datetime.now().astimezone().strftime("%Y%m%dT%H%M%S%f")
    run_dir = Path(os.environ.get("ZERO_SHOT_RUN_DIR", directory / "result" / stamp))
    run_dir.mkdir(parents=True, exist_ok=True)
    save(run_dir / "preflight.json", check)
    save(run_dir / "run_metadata.json", {"status": "running", "model_id": config["model_id"]})
    b01, official = configure_legacy()
    spec = config["official_claim"]
    labels, templates = official.protocol_text(spec["protocol"])
    adapter = b01.load_adapter(config["model_id"], config["runtime_backend"])
    classifier, _ = official.build_classifier(adapter, labels, templates, args.text_batch_size)
    from shared.standard_dataset import StandardDataset
    normalized = StandardDataset(DATASET_ROOT / config["standard_dataset"].removeprefix("dataset/"))
    seen = 0
    started = time.perf_counter()
    with (run_dir / "predictions.jsonl").open("w", encoding="utf-8") as pred_file, (run_dir / "ground_truth.jsonl").open("w", encoding="utf-8") as gt_file:
        for records, images in normalized.batches(args.batch_size):
            if smoke_limit:
                records, images = records[:max(0, smoke_limit-seen)], images[:max(0, smoke_limit-seen)]
                if not records:break
            if spec["protocol"] == "bigvision_81":
                features = official.bigvision_image_features(adapter, images, int(spec["resolution"]))
            else:
                if spec["protocol"] == "bigvision_naflex_256":
                    processor = getattr(adapter.processor, "image_processor", adapter.processor)
                    if processor.max_num_patches != spec["max_num_patches"]:
                        raise RuntimeError("NaFlex patch limit mismatch")
                features = adapter.image_features(images)
            ranked = (features @ classifier.T).topk(1, dim=1).indices.cpu().tolist()
            for record, values in zip(records, ranked):
                target, identifier = record["label"], record["sample_id"]
                pred_file.write(json.dumps({"sample_id": identifier, "prediction": values[0]}) + "\n")
                gt_file.write(json.dumps({"sample_id": identifier, "target": target, "image": record["image"]}) + "\n")
                seen += 1
            if smoke_limit and seen>=smoke_limit:break
            if seen % 1000 < len(records):
                print(f"{config['model_id']} {seen}/50000", flush=True)
    if seen != (smoke_limit or 50000):
        raise RuntimeError(f"Expected 50000 samples, got {seen}; incomplete run is not scoreable")
    forward_seconds = float(getattr(adapter, "forward_seconds", 0.0))
    forward_samples = int(getattr(adapter, "forward_samples", 0))
    if forward_samples != seen or forward_seconds <= 0:
        raise RuntimeError("Forward timing coverage does not match the evaluated sample set")
    metadata = {
        "status": "inference_completed", "model_id": config["model_id"], "revision": config["revision"],
        "purpose": "delivery_smoke_only" if smoke_limit else "frozen_official_protocol_measurement", "smoke_only": bool(smoke_limit), "sample_count": seen, "class_count": len(labels),
        "config_sha256": digest(directory / "model_config.json"),
        "predictions_sha256": digest(run_dir / "predictions.jsonl"),
        "ground_truth_sha256": digest(run_dir / "ground_truth.jsonl"),
        "dataset_index_sha256": digest(normalized.directory / "dataset.json"),
        "elapsed_pipeline_seconds": time.perf_counter() - started,
        "forward_total_seconds": forward_seconds,
        "forward_timed_samples": forward_samples,
        "forward_ms_per_sample": forward_seconds * 1000 / forward_samples,
        "forward_timing_scope": "CUDA-synchronized image encoder forward only; excludes decode, preprocessing, host-to-device transfer, text classifier construction, similarity scoring and metrics",
        "batch_size": args.batch_size,
        "precision": "fp16", "gpu": torch.cuda.get_device_name(0), "torch": torch.__version__,
        "python": platform.python_version(), "acceptance_assessed": False,
        "packages": {name: importlib.metadata.version(name) for name in ["torchvision", "transformers", "open_clip_torch", "timm", "Pillow", "numpy", "pyarrow"]},
    }
    save(run_dir / "run_metadata.json", metadata)
    from export_bundle import export_bundle
    rows = [json.loads(line) for line in (run_dir / "predictions.jsonl").read_text().splitlines()]
    targets = {r["sample_id"]: r["target"] for r in (json.loads(line) for line in (run_dir / "ground_truth.jsonl").read_text().splitlines())}
    rows = [{**r, "target": targets[r["sample_id"]]} for r in rows]
    bundle = export_bundle(run_dir, normalized.categories, rows, config["model_id"], config["official_dataset"])
    result = {"status": "inference_completed", "bundle": str(bundle), "sample_count": seen}
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"Inference completed; standard scoring bundle: {run_dir}")
    return 0


