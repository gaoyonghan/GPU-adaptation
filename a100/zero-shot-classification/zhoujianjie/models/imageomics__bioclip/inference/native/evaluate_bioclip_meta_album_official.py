#!/usr/bin/env python3
"""Strict preflight and launcher for BioCLIP/BioCLIP 2 Meta-Album evaluation."""

from __future__ import annotations
from model_paths import legacy as _legacy, legacy_str as _legacy_str, relative as _relative, revision as _revision, weight as _weight, result as _result

import argparse
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC_PATH = ROOT / "evaluation_audit/bioclip_meta_album_official_spec.json"
MANIFEST_PATH = ROOT / "evaluation_audit/bioclip_meta_album_eval_manifest.json"
DATA_ROOT = ROOT / "datasets/evaluation/bioclip_meta_album_official"
EVALUATOR_ROOT = ROOT / "evaluation_audit/reference_code/bioclip-2"
RESULT_ROOT = ROOT / "evaluation_audit/results/bioclip_meta_album_official"


def git_head(path: Path) -> str:
    return _revision(str(path),status=False).strip()


def preflight(model_id: str) -> tuple[dict, list[str]]:
    spec = json.loads(SPEC_PATH.read_text(encoding="utf-8"))
    model_spec = spec["models"][model_id]
    receipt_path = ROOT / model_spec["local_dir"] / "download_receipt.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8")) if receipt_path.exists() else {}
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8")) if MANIFEST_PATH.exists() else {}
    errors = []
    if receipt.get("model_id") != model_id or receipt.get("revision") != model_spec["revision"]:
        errors.append("checkpoint ID/revision mismatch")
    if manifest.get("status") != "verified_eval_only" or manifest.get("total_images") != 16200:
        errors.append("Meta-Album evaluation-only manifest missing or invalid")
    if manifest.get("train_payload_downloaded") is not False:
        errors.append("manifest does not prove train_payload_downloaded=false")
    if git_head(EVALUATOR_ROOT) != spec["evaluator"]["commit"]:
        errors.append("official evaluator commit mismatch")
    for task, dataset in spec["datasets"].items():
        val_dir = DATA_ROOT / dataset["path"]
        if not val_dir.is_dir() or not (val_dir / "metadata.csv").is_file():
            errors.append(f"missing official evaluation set: {task}")

    mapping = {
        "checkpoint": f"{model_id}@{model_spec['revision']}",
        "datasets": spec["datasets"],
        "protocol": "official classification.py; asis labels; OpenAI templates; zero/1/5-shot Top-1",
        "official_reference": spec["official_top1_percent"][model_id],
        "diagnostic_only": spec["diagnostic_only"],
        "status": "PASS" if not errors else "HARD_FAIL",
        "errors": errors,
    }
    print(json.dumps(mapping, indent=2, ensure_ascii=False))
    return spec, errors


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["imageomics/bioclip", "imageomics/bioclip-2"], required=True)
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    spec, errors = preflight(args.model)
    if errors:
        raise SystemExit(2)
    if not args.run:
        print("Preflight passed. No GPU evaluation started; pass --run to execute.")
        return

    model_root = (ROOT / spec["models"][args.model]["local_dir"]).resolve()
    for task, dataset in spec["datasets"].items():
        val_dir = (DATA_ROOT / dataset["path"]).resolve()
        task_slug = task.lower().replace(" ", "_")
        command = [
            sys.executable,
            "-m",
            "src.evaluation.classification",
            "--model",
            f"local-dir:{model_root}",
            "--pretrained",
            "False",
            "--batch-size",
            str(args.batch_size),
            "--workers",
            str(args.workers),
            "--data_root",
            str(val_dir),
            "--label_filename",
            str(val_dir / "metadata.csv"),
            "--logs",
            str(RESULT_ROOT / args.model.replace("/", "__") / task_slug),
            "--name",
            task_slug,
            "--text_type",
            "asis",
            "--task_type",
            "all",
            "--classification-tasks",
            "zero_shot",
            "few_shot",
            "--nfold",
            "5",
            "--kshot_list",
            "1",
            "5",
        ]
        print("RUN:", " ".join(command), flush=True)
        subprocess.run(command, cwd=EVALUATOR_ROOT, check=True)


if __name__ == "__main__":
    main()
