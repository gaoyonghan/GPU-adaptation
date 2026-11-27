#!/usr/bin/env python3
"""Strict ImageNet-1k validation for reference-ready B01 checkpoints.

This runner refuses proxies, incomplete datasets, missing checkpoint reference
values, and protocol mismatches before launching a model worker.
"""

from __future__ import annotations
from model_paths import legacy as _legacy, legacy_str as _legacy_str, relative as _relative, revision as _revision, weight as _weight, result as _result

import argparse
import gc
import hashlib
import io
import json
import platform
import re
import runpy
import string
import subprocess
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any

import imagenet1k_official_spec as claim_spec
from official_22_models import IMAGENET_MODELS


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "datasets/evaluation/imagenet1k_val_hf/data"
RESULT_ROOT = ROOT / "results/imagenet1k_official"
# PyTorch >=2.6 is required by current Transformers to safely load legacy
# PyTorch .bin checkpoints (CVE-2025-32434 guard). This environment is also
# used by the already-validated Chinese-CLIP and AltCLIP official runs.
DEFAULT_PYTHON = _legacy('/home/dell/anaconda3/envs/hengxiang_stage2_clean/bin/python')
# MetaCLIP 2 was released against Transformers-from-source and needs the
# native MetaCLIP2Model/XLM-R tokenizer support absent from the frozen 4.36
# environment used by the older checkpoints.  The UMind environment provides
# Transformers 4.57.1, which contains that official model implementation.
MODEL_PYTHONS = {
    "facebook/metaclip-2-worldwide-huge-quickgelu": _legacy('/home/dell/anaconda3/envs/UMind_agent/bin/python'),
    "google/siglip2-base-patch16-naflex": _legacy('/home/dell/anaconda3/envs/UMind_agent/bin/python'),
    "google/siglip2-so400m-patch16-naflex": _legacy('/home/dell/anaconda3/envs/UMind_agent/bin/python'),
    "google/siglip2-giant-opt-patch16-384": _legacy('/home/dell/anaconda3/envs/UMind_agent/bin/python'),
}
BIGVISION_ROOT = ROOT / "evaluation_audit/reference_code/big_vision"
CLASS_NAMES_FILE = BIGVISION_ROOT / "big_vision/datasets/imagenet/class_names.py"
PROMPTS_FILE = BIGVISION_ROOT / "big_vision/evaluators/proj/image_text/prompt_engineering_constants.py"


def now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", action="append", help="repeat to select verified models")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--text-batch-size", type=int, default=512)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument(
        "--reconcile-existing", action="store_true",
        help="add all-official-metrics acceptance fields to completed summaries without GPU inference",
    )
    parser.add_argument("--worker-model-id", help=argparse.SUPPRESS)
    parser.add_argument("--worker-backend", help=argparse.SUPPRESS)
    parser.add_argument("--worker-output", type=Path, help=argparse.SUPPRESS)
    return parser.parse_args()


def dataset_shards() -> list[Path]:
    return sorted(DATA_DIR.glob("validation-*.parquet"))


def verify_dataset(*, deep_hash: bool = False) -> dict[str, Any]:
    import pyarrow.parquet as pq

    shards = dataset_shards()
    expected_names = [f"validation-{index:05d}-of-00014.parquet" for index in range(14)]
    if [path.name for path in shards] != expected_names:
        raise RuntimeError(
            "official dataset preflight failed: expected exactly 14 ImageNet-1k validation shards; "
            f"found {[path.name for path in shards]}"
        )
    rows = 0
    schemas = set()
    file_records = []
    for path in shards:
        parquet = pq.ParquetFile(path)
        rows += parquet.metadata.num_rows
        schemas.add(str(parquet.schema_arrow))
        metadata_path = (ROOT / "datasets/evaluation/imagenet1k_val_hf/.cache/huggingface/download/data"
                         / f"{path.name}.metadata")
        if not metadata_path.is_file():
            raise RuntimeError(f"official dataset preflight failed: download metadata missing for {path.name}")
        metadata_lines = metadata_path.read_text(encoding="utf-8").splitlines()
        if len(metadata_lines) < 2 or metadata_lines[0] != claim_spec.DATASET["revision"]:
            raise RuntimeError(f"official dataset preflight failed: revision mismatch for {path.name}")
        record = {"name": path.name, "bytes": path.stat().st_size,
                  "expected_sha256": metadata_lines[1]}
        if deep_hash:
            digest = hashlib.sha256()
            with path.open("rb") as stream:
                for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                    digest.update(chunk)
            record["actual_sha256"] = digest.hexdigest()
            if record["actual_sha256"] != record["expected_sha256"]:
                raise RuntimeError(f"official dataset preflight failed: SHA-256 mismatch for {path.name}")
        file_records.append(record)
    if rows != claim_spec.DATASET["rows"]:
        raise RuntimeError(f"official dataset preflight failed: expected 50000 rows, found {rows}")
    if len(schemas) != 1:
        raise RuntimeError("official dataset preflight failed: validation shard schemas differ")
    return {
        **claim_spec.DATASET,
        "local_dir": str(_relative(DATA_DIR,ROOT)),
        "files": file_records,
        "bytes": sum(path.stat().st_size for path in shards),
        "rows_verified_from_parquet_metadata": rows,
        "schema_sha256": hashlib.sha256(next(iter(schemas)).encode()).hexdigest(),
    }


def canonicalize(text: str, keep: str | None = None) -> str:
    text = text.replace("_", " ")
    if keep:
        text = keep.join(part.translate(str.maketrans("", "", string.punctuation)) for part in text.split(keep))
    else:
        text = text.translate(str.maketrans("", "", string.punctuation))
    return re.sub(r"\s+", " ", text.lower()).strip()


def protocol_text(protocol: str) -> tuple[list[str], list[str]]:
    class_values = runpy.run_path(str(CLASS_NAMES_FILE))["CLIP_IMAGENET_CLASS_NAMES"]
    prompt_values = runpy.run_path(str(PROMPTS_FILE))["CLIP_PAPER_PROMPT_TEMPLATES"]
    if len(class_values) != 1000 or len(prompt_values) != 81 or prompt_values[-1] != "{}":
        raise RuntimeError("official reference-code constants changed unexpectedly")
    if protocol == "clip_80":
        return list(class_values), list(prompt_values[:-1])
    if protocol in {"bigvision_81", "bigvision_naflex_256"}:
        return (
            [canonicalize(value, keep=",") for value in class_values],
            [canonicalize(value, keep="{}") for value in prompt_values],
        )
    raise RuntimeError(f"unsupported official zero-shot protocol: {protocol}")


def protocol_source_hashes() -> dict[str, str]:
    return {
        str(_relative(path,ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in (CLASS_NAMES_FILE, PROMPTS_FILE)
    }


def preflight_models(requested: list[str] | None) -> tuple[list[str], list[dict[str, Any]]]:
    requested_ids = list(requested or IMAGENET_MODELS)
    unknown = sorted(set(requested_ids) - set(claim_spec.MODEL_SPECS))
    if unknown:
        raise RuntimeError(f"models outside frozen official scope: {unknown}")
    runnable, blocked = [], []
    for model_id in requested_ids:
        spec = claim_spec.MODEL_SPECS[model_id]
        if spec["reference_status"] != "verified" or spec["protocol"] not in {
            "clip_80", "bigvision_81", "bigvision_naflex_256"
        }:
            blocked.append({"model_id": model_id, **spec})
        else:
            runnable.append(model_id)
    # An explicitly requested blocked model is a hard error. The default full
    # scope reports it and runs only the independently verified entries.
    if requested and blocked:
        raise RuntimeError(f"official preflight blocked requested models: {blocked}")
    return runnable, blocked


def backend_for(model_id: str) -> str:
    from evaluate_b01 import backend_for as ledger_backend_for, ledger_rows

    matches = [row for row in ledger_rows() if str(row["模型名称"]) == model_id]
    if len(matches) != 1:
        raise RuntimeError(f"expected one ledger row for {model_id}, found {len(matches)}")
    backend = ledger_backend_for(matches[0])
    if backend not in {"Transformers AutoModel/AutoProcessor", "OpenCLIP"}:
        raise RuntimeError(f"official evaluator has no approved backend for {model_id}: {backend}")
    return backend


def result_path(model_id: str) -> Path:
    return RESULT_ROOT.joinpath(*model_id.split("/"), "summary.json")


def metric_alignment(top1: float, top5: float, model_id: str) -> dict[str, Any]:
    """Compare measurements only where a same-checkpoint official value exists."""
    measured = {"top1_percent": top1, "top5_percent": top5}
    rows: dict[str, dict[str, Any]] = {}
    for name, claim in claim_spec.official_metric_claims(model_id).items():
        value = round(float(measured[name]), 4)
        official = claim["official_value"]
        row = {"measured_value": value, **claim}
        if claim["reference_status"] == "verified" and official is not None:
            official_float = float(official)
            gap = value - official_float
            relative_gap = abs(gap) / official_float * 100
            row.update({
                "absolute_gap_percentage_points": round(gap, 4),
                "relative_gap_percent": round(relative_gap, 4),
                "within_5_percent_relative": relative_gap <= 5.0,
            })
        else:
            row.update({
                "absolute_gap_percentage_points": None,
                "relative_gap_percent": None,
                "within_5_percent_relative": None,
            })
        rows[name] = row
    validated = [name for name, row in rows.items() if row["reference_status"] == "verified"]
    diagnostics = [name for name, row in rows.items() if row["reference_status"] != "verified"]
    return {
        "metrics": rows,
        "validated_official_metrics": validated,
        "diagnostic_metrics_without_official_reference": diagnostics,
        "within_5_percent_relative_all_official_metrics": all(
            rows[name]["within_5_percent_relative"] is True for name in validated
        ),
    }


def reconcile_result_metrics(result: dict[str, Any]) -> dict[str, Any]:
    if result.get("status") != "passed":
        return result
    alignment = metric_alignment(
        float(result["top1_percent"]), float(result["top5_percent"]), str(result["model_id"])
    )
    result["official_metric_alignment"] = alignment["metrics"]
    result["validated_official_metrics"] = alignment["validated_official_metrics"]
    result["diagnostic_metrics_without_official_reference"] = alignment[
        "diagnostic_metrics_without_official_reference"
    ]
    result["within_5_percent_relative_all_official_metrics"] = alignment[
        "within_5_percent_relative_all_official_metrics"
    ]
    return result


def decode_image(value: Any):
    from PIL import Image

    if isinstance(value, dict):
        payload = value.get("bytes")
        if payload is None and value.get("path"):
            return Image.open(value["path"]).convert("RGB")
    else:
        payload = value
    if not isinstance(payload, (bytes, bytearray, memoryview)):
        raise TypeError(f"unexpected ImageNet image parquet value: {type(value).__name__}")
    return Image.open(io.BytesIO(bytes(payload))).convert("RGB")


def parquet_batches(batch_size: int):
    import pyarrow.parquet as pq

    for shard in dataset_shards():
        parquet = pq.ParquetFile(shard)
        for batch in parquet.iter_batches(batch_size=batch_size, columns=["image", "label"]):
            values = batch.to_pydict()
            images = [decode_image(value) for value in values["image"]]
            yield shard.name, images, [int(value) for value in values["label"]]


def build_classifier(adapter: Any, labels: list[str], templates: list[str], text_batch_size: int):
    import torch

    # Prompt order is class-major, matching the official evaluators.
    prompts = [template.format(label) for label in labels for template in templates]
    chunks = []
    for start in range(0, len(prompts), text_batch_size):
        chunks.append(adapter.text_features(prompts[start:start + text_batch_size]).cpu())
    features = torch.cat(chunks).reshape(len(labels), len(templates), -1).mean(dim=1)
    return torch.nn.functional.normalize(features, dim=-1).cuda(), prompts


def bigvision_image_features(adapter: Any, images: list[Any], resolution: int):
    """Equivalent implementation of Big Vision's square875 TF preprocessing."""
    import numpy as np
    import torch
    import torch.nn.functional as functional

    if adapter.__class__.__name__ != "TransformersAdapter":
        raise RuntimeError("bigvision_81 requires the pinned Transformers checkpoint adapter")
    tensors = []
    resized_size = int(resolution / 0.875)
    for image in images:
        array = np.asarray(image, dtype=np.uint8).copy()
        tensor = torch.from_numpy(array).permute(2, 0, 1).float().unsqueeze(0)
        tensor = functional.interpolate(
            tensor, size=(resized_size, resized_size), mode="bilinear",
            align_corners=False, antialias=False,
        )
        # Big Vision's resize op clips and casts back to the input uint8 dtype.
        tensor = tensor.clamp(0, 255).to(torch.uint8).float()
        offset = (resized_size - resolution) // 2
        tensor = tensor[:, :, offset:offset + resolution, offset:offset + resolution]
        tensors.append(tensor.squeeze(0) / 127.5 - 1.0)
    pixels = torch.stack(tensors).cuda(non_blocking=True)
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
        features = adapter.model.get_image_features(pixel_values=pixels)
        from evaluate_b01 import tensor_from_feature
        features = tensor_from_feature(features)
    return torch.nn.functional.normalize(features.float(), dim=-1)


def worker(args: argparse.Namespace) -> int:
    import torch
    from evaluate_b01 import load_adapter, model_dir

    model_id = args.worker_model_id
    output = args.worker_output
    assert model_id and args.worker_backend and output
    spec = claim_spec.MODEL_SPECS[model_id]
    result: dict[str, Any] = {
        "purpose": "same_checkpoint_official_imagenet1k_metric_acceptance",
        "status": "failed", "model_id": model_id, "started_at": now(),
        "claim": spec, "dataset": claim_spec.DATASET,
    }
    adapter = None
    try:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable")
        if spec["reference_status"] != "verified":
            raise RuntimeError(f"reference is not verified: {spec['reference_status']}")
        verify_dataset(deep_hash=False)
        labels, templates = protocol_text(spec["protocol"])
        adapter = load_adapter(model_id, args.worker_backend)
        receipt = model_dir(model_id) / "download_receipt.json"
        if not receipt.is_file():
            raise RuntimeError("pinned model download receipt missing")
        receipt_data = json.loads(receipt.read_text(encoding="utf-8"))
        result["model_revision"] = receipt_data.get("revision") or receipt_data.get("resolved_revision")
        result["backend"] = adapter.name
        result["protocol_source_sha256"] = protocol_source_hashes()
        result["prompt_templates"] = templates
        result["class_names_sha256"] = hashlib.sha256(json.dumps(labels).encode()).hexdigest()
        classifier, prompts = build_classifier(adapter, labels, templates, args.text_batch_size)
        result["prompt_count"] = len(prompts)
        correct1 = correct5 = seen = 0
        next_progress = 1_000
        started = time.perf_counter()
        for shard, images, targets in parquet_batches(args.batch_size):
            if spec["protocol"] == "bigvision_81":
                image_features = bigvision_image_features(adapter, images, int(spec["resolution"]))
            elif spec["protocol"] == "bigvision_naflex_256":
                image_processor = getattr(adapter.processor, "image_processor", adapter.processor)
                processor_limit = getattr(image_processor, "max_num_patches", None)
                if processor_limit != int(spec["max_num_patches"]):
                    raise RuntimeError(
                        f"NaFlex processor max_num_patches mismatch: {processor_limit} "
                        f"!= {spec['max_num_patches']}"
                    )
                image_features = adapter.image_features(images)
            else:
                image_features = adapter.image_features(images)
            top5 = (image_features @ classifier.T).topk(5, dim=1).indices.cpu()
            target = torch.tensor(targets)
            correct1 += int((top5[:, 0] == target).sum())
            correct5 += int((top5 == target[:, None]).any(dim=1).sum())
            seen += len(targets)
            if seen >= next_progress:
                print(f"PROGRESS {model_id} {seen}/50000 top1={100*correct1/seen:.3f}", flush=True)
                next_progress = ((seen // 1_000) + 1) * 1_000
        if seen != 50_000:
            raise RuntimeError(f"expected 50000 validation images, evaluated {seen}")
        top1 = 100 * correct1 / seen
        official = float(spec["official_top1_percent"])
        relative_gap = abs(top1 - official) / official * 100
        result.update({
            "status": "passed", "sample_count": seen,
            "top1_percent": round(top1, 4), "top5_percent": round(100 * correct5 / seen, 4),
            "official_top1_percent": official,
            "absolute_gap_percentage_points": round(top1 - official, 4),
            "relative_gap_percent": round(relative_gap, 4),
            "within_5_percent_relative": relative_gap <= 5.0,
            "elapsed_seconds": round(time.perf_counter() - started, 3),
            "environment": {"python": platform.python_version(), "torch": torch.__version__,
                            "gpu": torch.cuda.get_device_name(0)},
        })
        reconcile_result_metrics(result)
    except Exception as error:
        result.update({"error_type": type(error).__name__, "error": str(error),
                       "traceback": traceback.format_exc(limit=20)})
    finally:
        result["finished_at"] = now()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        del adapter
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return 0 if result["status"] == "passed" else 1


def coordinator(args: argparse.Namespace) -> int:
    dataset = verify_dataset(deep_hash=True)
    runnable, blocked = preflight_models(args.model_id)
    report = {
        "generated_at": now(), "dataset": dataset,
        "runnable": [
            {
                "model_id": model_id,
                **claim_spec.MODEL_SPECS[model_id],
                "official_metric_claims": claim_spec.official_metric_claims(model_id),
            }
            for model_id in runnable
        ],
        "blocked": blocked,
    }
    report_path = ROOT / "evaluation_audit/imagenet1k_official_preflight.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    if args.preflight_only:
        return 0
    failures = 0
    for position, model_id in enumerate(runnable, 1):
        output = result_path(model_id)
        if output.is_file():
            old = json.loads(output.read_text(encoding="utf-8"))
            if old.get("status") == "passed":
                reconciled = reconcile_result_metrics(old)
                output.write_text(json.dumps(reconciled, ensure_ascii=False, indent=2), encoding="utf-8")
                action = "RECONCILE" if args.reconcile_existing else "SKIP"
                print(f"[{position}/{len(runnable)}] {action} {model_id}", flush=True)
                if args.reconcile_existing or not args.force:
                    continue
        if args.reconcile_existing:
            print(f"[{position}/{len(runnable)}] MISSING {model_id}", flush=True)
            failures += 1
            continue
        worker_python = MODEL_PYTHONS.get(model_id, DEFAULT_PYTHON)
        command = [str(worker_python), str(Path(__file__).resolve()),
                   "--worker-model-id", model_id, "--worker-backend", backend_for(model_id),
                   "--worker-output", str(output), "--batch-size", str(args.batch_size),
                   "--text-batch-size", str(args.text_batch_size)]
        print(f"[{position}/{len(runnable)}] RUN {model_id}", flush=True)
        completed = subprocess.run(command, cwd=ROOT, check=False)
        failures += completed.returncode != 0
    return 1 if failures else 0


def main() -> int:
    args = parse_args()
    if args.worker_model_id:
        return worker(args)
    return coordinator(args)


if __name__ == "__main__":
    raise SystemExit(main())
