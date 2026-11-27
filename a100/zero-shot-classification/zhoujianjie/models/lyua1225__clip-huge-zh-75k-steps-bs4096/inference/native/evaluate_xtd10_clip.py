#!/usr/bin/env python3
"""Evaluate B01 multilingual dual encoders on the verified XTD-10 test set."""

from __future__ import annotations
from model_paths import legacy as _legacy, legacy_str as _legacy_str, relative as _relative, revision as _revision, weight as _weight, result as _result

import argparse
import gc
import json
import platform
import subprocess
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "evaluation_audit"))
import evaluate_b01 as b01
from official_22_models import MULTILINGUAL_MODELS

DATA_ROOT = ROOT / "datasets/evaluation/xtd10_hf"
CACHE_ROOT = ROOT / "evaluation_audit/cache/xtd10"
RESULT_ROOT = ROOT / "results/downloaded_official/xtd10"
REVISION = "63227af6ee1d13951d244b979254154d7babe463"
LANGUAGES = ("es", "it", "ko", "pl", "ru", "tr", "zh")
INSTRUCTION = "Retrieve an image of this caption. "
WORKER_PYTHON = _legacy_str('/home/dell/anaconda3/envs/hengxiang_stage2_clean/bin/python')


def now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def records(language: str):
    cache = CACHE_ROOT / f"{language}_retrieval.json"
    if cache.is_file():
        frozen = json.loads(cache.read_text(encoding="utf-8"))
        return frozen["queries"], frozen["candidates"], frozen["targets"]
    import pyarrow.parquet as pq
    rows = pq.read_table(DATA_ROOT / language / f"{language}.parquet",
                         columns=["qry_text", "tgt_img_path"]).to_pylist()
    if len(rows) != 999:
        raise RuntimeError(f"expected 999 {language} queries, found {len(rows)}")
    candidates = sorted({name for row in rows for name in row["tgt_img_path"]})
    if len(candidates) != 999:
        raise RuntimeError(f"expected 999 {language} candidate images, found {len(candidates)}")
    index = {name: position for position, name in enumerate(candidates)}
    queries, targets = [], []
    for row in rows:
        text = row["qry_text"].strip()
        if text.startswith(INSTRUCTION):
            text = text[len(INSTRUCTION):]
        queries.append(text)
        targets.append(index[row["tgt_img_path"][0]])
    CACHE_ROOT.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps({"language": language, "dataset_revision": REVISION,
                                 "queries": queries, "candidates": candidates,
                                 "targets": targets}, ensure_ascii=False) + "\n", encoding="utf-8")
    return queries, candidates, targets


def worker(args) -> int:
    import torch
    from PIL import Image

    output = Path(args.worker_output)
    result = {
        "status": "failed", "model_id": args.worker_model_id,
        "dataset": "Haon-Chen/XTD-10 official test", "dataset_revision": REVISION,
        "languages": list(LANGUAGES), "protocol": "raw-caption text-to-image Recall@1/5/10",
        "note": "MMEB retrieval instruction removed because these are CLIP-family encoders",
        "started_at": now(),
    }
    adapter = None
    try:
        adapter = b01.load_adapter(args.worker_model_id, args.worker_backend)
        per_language = {}
        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        for language in LANGUAGES:
            queries, candidates, targets = records(language)
            image_features = []
            for offset in range(0, len(candidates), args.image_batch_size):
                images = []
                for name in candidates[offset:offset + args.image_batch_size]:
                    with Image.open(DATA_ROOT / "XTD10_dataset" / name) as source:
                        images.append(source.convert("RGB").copy())
                image_features.append(adapter.image_features(images).cpu())
            image_features = torch.cat(image_features)
            hits = {1: 0, 5: 0, 10: 0}
            for offset in range(0, len(queries), args.text_batch_size):
                text_features = adapter.text_features(queries[offset:offset + args.text_batch_size]).cpu()
                ranked = (text_features @ image_features.T).topk(10, dim=1).indices.tolist()
                for target, prediction in zip(targets[offset:offset + args.text_batch_size], ranked):
                    for k in hits:
                        hits[k] += target in prediction[:k]
            per_language[language] = {
                f"text_to_image_recall_at_{k}_percent": round(100 * hits[k] / len(queries), 4)
                for k in hits
            }
            print(json.dumps({"language": language, **per_language[language]}), flush=True)
        macro = {
            f"text_to_image_recall_at_{k}_percent": round(
                sum(per_language[lang][f"text_to_image_recall_at_{k}_percent"] for lang in LANGUAGES)
                / len(LANGUAGES), 4)
            for k in (1, 5, 10)
        }
        result.update(
            status="passed", query_count_per_language=999, candidate_count=999,
            per_language=per_language, macro_average=macro,
            elapsed_seconds=round(time.perf_counter() - started, 4),
            peak_gpu_memory_mb=round(torch.cuda.max_memory_allocated() / 1024**2, 2),
            environment={"python": platform.python_version(), "torch": torch.__version__,
                         "cuda_runtime": torch.version.cuda, "gpu": torch.cuda.get_device_name(0)},
        )
    except Exception as error:
        result.update(error_type=type(error).__name__, error=str(error),
                      traceback=traceback.format_exc(limit=20))
    finally:
        result["finished_at"] = now()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        del adapter
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return 0 if result["status"] == "passed" else 1


def coordinator(args) -> int:
    # Freeze the small Parquet-derived manifests before entering the lean GPU
    # worker environment, which intentionally does not carry pyarrow.
    for language in LANGUAGES:
        records(language)
    rows = {str(row["模型名称"]): row for row in b01.ledger_rows()}
    models = args.model_id or list(MULTILINGUAL_MODELS)
    failures = 0
    for index, model_id in enumerate(models, 1):
        if model_id not in MULTILINGUAL_MODELS:
            raise SystemExit(f"model is outside the 22-model XTD-10 scope: {model_id}")
        output = RESULT_ROOT.joinpath(*model_id.split("/"), "summary.json")
        if output.is_file() and not args.force:
            old = json.loads(output.read_text(encoding="utf-8"))
            if old.get("status") == "passed":
                print(f"[{index}/{len(models)}] SKIP {model_id}", flush=True)
                continue
        backend = b01.backend_for(rows[model_id])
        command = [WORKER_PYTHON, str(Path(__file__).resolve()),
                   "--worker-model-id", model_id, "--worker-backend", backend,
                   "--worker-output", str(output), "--image-batch-size", str(args.image_batch_size),
                   "--text-batch-size", str(args.text_batch_size)]
        print(f"[{index}/{len(models)}] RUN {model_id}", flush=True)
        completed = subprocess.run(command, cwd=ROOT, check=False, timeout=args.timeout_seconds)
        failures += int(completed.returncode != 0)
    return 1 if failures else 0


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", action="append")
    parser.add_argument("--image-batch-size", type=int, default=32)
    parser.add_argument("--text-batch-size", type=int, default=256)
    parser.add_argument("--timeout-seconds", type=int, default=3600)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--worker-model-id", help=argparse.SUPPRESS)
    parser.add_argument("--worker-backend", help=argparse.SUPPRESS)
    parser.add_argument("--worker-output", help=argparse.SUPPRESS)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.worker_model_id:
        return worker(args)
    return coordinator(args)


if __name__ == "__main__":
    raise SystemExit(main())
