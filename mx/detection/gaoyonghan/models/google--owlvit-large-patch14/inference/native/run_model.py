#!/usr/bin/env python3
"""Route model 70 to the upstream Scenic evaluator and expose its raw results."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--scenic-root", required=True)
    parser.add_argument("--native-data-root", required=True)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--model-root", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--phase", choices=["smoke", "full"], default="full")
    args = parser.parse_args()
    if not args.device.startswith("cuda:") or not args.device[5:].isdigit():
        raise SystemExit("The native Scenic route requires cuda:<physical GPU index>.")
    data = Path(args.native_data_root).resolve()
    required = [data / "checkpoint_hf_roundtrip/checkpoint_0",
                data / "checkpoint_conversion_report.json",
                data / "annotations/lvis_v1_val.json",
                data / "tfds/lvis/1.3.0/dataset_info.json"]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise SystemExit("Missing prepared native inputs; prepare them before running: " +
                         ", ".join(missing))
    shards = list((data / "tfds/lvis/1.3.0").glob("lvis-validation.tfrecord-*-of-00032"))
    if len(shards) != 32:
        raise SystemExit(f"Expected all 32 native LVIS validation shards, found {len(shards)}.")
    conversion = json.loads(required[1].read_text())
    if (conversion.get("status") != "verified" or
            conversion.get("source") != "google/owlvit-large-patch14" or
            conversion.get("roundtrip_exact") is not True or
            conversion.get("roundtrip_max_abs_error") != 0):
        raise SystemExit("The native checkpoint conversion is not verified.")
    output = Path(args.output_root).resolve()
    if args.phase == "full" and (output / "run_manifest.json").exists():
        raise SystemExit("A native run already exists here; select a fresh output_root.")
    environment = {key: value for key, value in os.environ.items()
                   if not key.startswith("CODEX_TIMING_") and key not in
                   {"PYTHONPATH", "PYTHONHOME", "JAX_PLATFORMS"}}
    environment.update(
        ENV_PREFIX=sys.prefix, SCENIC_ROOT=str(Path(args.scenic_root).resolve()),
        NATIVE_DATA_ROOT=str(data), DATASET_ROOT=str(Path(args.dataset_root).resolve()),
        HF_WEIGHTS=str(Path(args.model_root).resolve() /
                       "google/owlvit-large-patch14/pytorch_model.bin"),
        OUTPUT_ROOT=str(output), OWL_CONFIG="clip_l14", GPU_ID=args.device[5:],
        ALLOW_TFDS_DOWNLOAD="0", PREPARE_DATA_ONLY="0", PREPARE_WEIGHTS_ONLY="0",
        PREFLIGHT_ONLY="1" if args.phase == "smoke" else "0",
        PYTHONNOUSERSITE="1", PYTHONDONTWRITEBYTECODE="1",
        XLA_PYTHON_CLIENT_PREALLOCATE="false", HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1")
    subprocess.run(["bash", str(ROOT / "run_official.sh")], cwd=ROOT,
                   env=environment, check=True)
    if args.phase == "smoke":
        print("Native preflight completed; no inference metrics were produced.", flush=True)
        return
    manifest_path = output / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("status") != "completed" or manifest.get("exit_code") != 0:
        raise SystemExit("Upstream evaluation or result validation failed; see run_manifest.json.")
    result_path = output / "clip_l14/lvis/results_validation.json"
    result_bytes = result_path.read_bytes()
    if hashlib.sha256(result_bytes).hexdigest() != manifest["official_results_sha256"]:
        raise SystemExit("Official result hash does not match the run manifest.")
    payload = {
        "sequence": 70, "model_key": "owlvit-large-patch14",
        "fidelity": "upstream_evaluator", "execution_path": "upstream_evaluator",
        "metric_implementation": "Scenic official evaluator / lvis-api LVISEval",
        "metrics": json.loads(result_bytes), "metrics_percent": manifest["metrics_percent"],
        "weight_provenance": manifest["weight_provenance"],
        "scenic_revision": manifest["scenic_revision"],
        "run_manifest": str(manifest_path), "official_results_file": str(result_path),
        "official_results_sha256": manifest["official_results_sha256"],
        "prediction_validation": manifest["prediction_validation"],
        "reference_comparisons": manifest["reference_comparisons"],
    }
    (output / "metrics.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(payload["metrics_percent"], ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
