#!/usr/bin/env python3
"""Strict single-dataset acceptance runner for mmE5 on MMEB/Country211."""

from __future__ import annotations
from model_paths import DATA as _data
from model_paths import legacy as _legacy, legacy_str as _legacy_str, relative as _relative, revision as _revision, weight as _weight, result as _result

import hashlib
import importlib.metadata
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import pyarrow.parquet as pq
from datasets import load_from_disk


ROOT = _legacy('/data/zhoujianjie')
MODEL_ID = "intfloat/mmE5-mllama-11b-instruct"
MODEL_REVISION = "cbb328b9bf9ff5362c852c3166931903226d46f1"
DATASET_REVISION = "2f069730be515ea60778413777816b53e2d2a697"
CODE_REVISION = "4818cdfebe46f761421ae7edff8aaabf1f947bad"
OFFICIAL_P1_PERCENT = 35.4
MODEL_DIR = _legacy('/data/zhoujianjie/model_inference_for_GPU_adaptation/classification/zhoujianjie/models/intfloat__mmE5-mllama-11b-instruct/weights')
DATASET_DIR = _data / 'raw/mmeb_country_1'
RUNTIME_DATASET_DIR = _legacy('/data/zhoujianjie/model_inference_for_GPU_adaptation/dataset/raw/mmeb_runtime')
IMAGE_DIR = _data / 'raw/mmeb_country_2'
CODE_DIR = _legacy('/data/zhoujianjie/environment_setup/native_sources/evaluation_audit/reference_code/mmE5_official_4818cdf')
OUT_DIR = _legacy('/data/zhoujianjie/results/mme5_mmeb_official/country211_strict')

REQUIRED_PACKAGES = {
    "datasets": "2.19.2",
    "peft": "0.10.0",
    "transformers": "4.48.3",
    "tokenizers": "0.21.0",
    "huggingface-hub": "0.28.1",
    "accelerate": "0.34.0",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def preflight() -> dict[str, object]:
    receipt = json.loads((_legacy('/data/zhoujianjie/model_inference_for_GPU_adaptation/classification/zhoujianjie/models/intfloat__mmE5-mllama-11b-instruct/weights/download_receipt.json')).read_text())
    if receipt["revision"] != MODEL_REVISION:
        raise RuntimeError("mmE5 checkpoint revision mismatch")

    parquet = DATASET_DIR / 'test-00000-of-00001.parquet'
    metadata = DATASET_DIR / '.cache/huggingface/download/Country211/test-00000-of-00001.parquet.metadata'
    revision, expected_hash, *_ = metadata.read_text().splitlines()
    if revision != DATASET_REVISION or sha256(parquet) != expected_hash:
        raise RuntimeError("MMEB Country211 revision/hash mismatch")
    if pq.ParquetFile(parquet).metadata.num_rows != 1000:
        raise RuntimeError("MMEB Country211 must contain exactly 1,000 test rows")

    code_revision = _revision(str(CODE_DIR),status=False).strip()
    if code_revision != CODE_REVISION:
        raise RuntimeError("mmE5 evaluator revision mismatch")
    code_changes = _revision(str(CODE_DIR),status=True).strip()
    if code_changes:
        raise RuntimeError("mmE5 official evaluator worktree is not clean")

    package_versions = {
        package: importlib.metadata.version(package) for package in REQUIRED_PACKAGES
    }
    mismatched_packages = {
        package: {"expected": expected, "actual": package_versions[package]}
        for package, expected in REQUIRED_PACKAGES.items()
        if package_versions[package] != expected
    }
    if mismatched_packages:
        raise RuntimeError(f"mmE5 dependency mismatch: {mismatched_packages}")

    saved_dataset = load_from_disk(str(_legacy('/data/zhoujianjie/model_inference_for_GPU_adaptation/dataset/raw/mmeb_runtime/Country211')))
    parquet_rows = pq.read_table(parquet).to_pylist()
    if len(saved_dataset) != 1000 or saved_dataset.to_list() != parquet_rows:
        raise RuntimeError("save_to_disk dataset is not identical to the frozen parquet")

    image_count = sum(1 for path in IMAGE_DIR.iterdir() if path.is_file())
    if image_count != 1000:
        raise RuntimeError(f"expected 1,000 extracted Country211 images, found {image_count}")

    mapping = {
        "model": f"{MODEL_ID}@{MODEL_REVISION}",
        "dataset": f"TIGER-Lab/MMEB-eval@{DATASET_REVISION}:Country211/test (1,000 rows)",
        "protocol": f"mmE5@{CODE_REVISION} official eval.py; normalized last-token embeddings",
        "metric": "Precision@1 (official runner names the JSON field acc)",
        "same_checkpoint_reference_percent": OFFICIAL_P1_PERCENT,
        "reference": "mmE5 paper arXiv:2502.08468v1, Appendix E Table 7, supervised mmE5 column",
        "runtime_dataset_format": "HF save_to_disk; row-for-row verified against frozen parquet",
        "dependency_versions": package_versions,
    }
    print("STRICT PREFLIGHT MODEL->DATASET->PROTOCOL->METRIC")
    print(json.dumps(mapping, indent=2))
    return mapping


def main() -> None:
    mapping = preflight()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable, "eval.py",
        "--model_name", str(MODEL_DIR),
        "--encode_output_path", str(OUT_DIR),
        "--max_len", "256",
        "--pooling", "last",
        "--normalize", "True",
        "--dataloader_num_workers", "0",
        "--dataset_path", str(RUNTIME_DATASET_DIR),
        "--subset_name", "Country211",
        "--dataset_split", "test",
        "--per_device_eval_batch_size", "2",
        "--image_dir", str(IMAGE_DIR),
        "--model_backbone", "mllama",
    ]
    env = os.environ.copy()
    env['PYTHONPATH'] = os.pathsep.join([str(Path(__file__).parent),str(CODE_DIR/'src'),str(CODE_DIR),env.get('PYTHONPATH','')])
    env["HF_HOME"] = str(_legacy('/data/zhoujianjie/.hf_runtime_cache'))
    command=[sys.executable,_legacy_str('/data/zhoujianjie/environment_setup/native_mme5_python.py')]+command[1:]
    subprocess.run(command, cwd=CODE_DIR, env=env, check=True)

    score_path = OUT_DIR / 'Country211_score.json'
    measured = json.loads(score_path.read_text())["acc"] * 100
    relative_gap = abs(measured - OFFICIAL_P1_PERCENT) / OFFICIAL_P1_PERCENT * 100
    summary = {
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "status": "passed" if relative_gap <= 5 else "failed",
        "mapping": mapping,
        "measured_precision_at_1_percent": measured,
        "official_precision_at_1_percent": OFFICIAL_P1_PERCENT,
        "relative_gap_percent": relative_gap,
        "within_5_percent_relative": relative_gap <= 5,
        "official_score_json": _relative(score_path,ROOT).as_posix(),
    }
    summary_path = OUT_DIR / 'summary.json'
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
