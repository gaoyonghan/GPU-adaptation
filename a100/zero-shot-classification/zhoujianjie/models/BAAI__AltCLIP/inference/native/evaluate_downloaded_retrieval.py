#!/usr/bin/env python3
"""Evaluate mapped dual encoders on verified Flickr30k/MSCOCO test sets."""

from __future__ import annotations
from model_paths import legacy as _legacy, legacy_str as _legacy_str, relative as _relative, revision as _revision, weight as _weight, result as _result

import argparse
import csv
import gc
import io
import json
import platform
import subprocess
import sys
import tarfile
import time
import traceback
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "evaluation_audit"))
import evaluate_b01 as b01

DATASETS = {
    "flickr30k": {
        "directory": ROOT / "datasets/evaluation/flickr30k_wds_hf/test",
        "revision": "f8af3ceee1b944bf761340682621dfafd23974d2",
        "images": 1000,
        "mapping_needle": "Flickr30k",
    },
    "mscoco_karpathy": {
        "directory": ROOT / "datasets/evaluation/mscoco_karpathy_wds_hf/test",
        "revision": "3bf9ad5d280b03126293bcc691585015e29ce691",
        "images": 5000,
        "mapping_needle": "MS-COCO",
    },
}
RESULT_ROOT = ROOT / "results/downloaded_retrieval"
WORKER_PYTHON = _legacy_str('/home/dell/anaconda3/envs/hengxiang_stage2_clean/bin/python')
OFFICIAL_CLAIMS = {
    ("BAAI/AltCLIP", "flickr30k"): {
        "text_to_image_recall_at_1_percent": 72.5,
        "text_to_image_recall_at_5_percent": 91.6,
        "text_to_image_recall_at_10_percent": 95.4,
        "image_to_text_recall_at_1_percent": 86.0,
        "image_to_text_recall_at_5_percent": 98.0,
        "image_to_text_recall_at_10_percent": 99.1,
    },
}


def now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def load_records(dataset: str):
    records = []
    for shard in sorted(DATASETS[dataset]["directory"].glob("*.tar")):
        with tarfile.open(shard) as archive:
            members = {Path(member.name).name: member for member in archive if member.isfile()}
            stems = sorted({Path(name).stem for name in members})
            for stem in stems:
                image_name, text_name = f"{stem}.jpg", f"{stem}.txt"
                if image_name not in members or text_name not in members:
                    raise RuntimeError(f"unpaired sample {stem} in {shard.name}")
                image_bytes = archive.extractfile(members[image_name]).read()
                captions = archive.extractfile(members[text_name]).read().decode("utf-8").splitlines()
                captions = [caption.strip() for caption in captions if caption.strip()]
                if not captions:
                    raise RuntimeError(f"no captions for {stem}")
                records.append((stem, image_bytes, captions))
    records.sort(key=lambda row: row[0])
    expected = DATASETS[dataset]["images"]
    if len(records) != expected:
        raise RuntimeError(f"expected {expected:,} images, found {len(records):,}")
    _limit=int(__import__('os').environ.get('ZERO_SHOT_SMOKE_LIMIT','0'))
    return records[:_limit] if _limit else records


def recall_metrics(image_features, text_features, captions, targets):
    """Compute exact Recall@K in chunks without materializing the full matrix."""
    import torch

    values = {}
    # Text -> image: at most 25k x 5k, chunk queries to cap temporary memory.
    text_hits = {1: 0, 5: 0, 10: 0}
    for offset in range(0, len(captions), 1024):
        scores = text_features[offset:offset + 1024] @ image_features.T
        ranked = scores.topk(min(10, scores.shape[1]), dim=1).indices.tolist()
        for target, predictions in zip(targets[offset:offset + 1024], ranked):
            for k in text_hits:
                text_hits[k] += target in predictions[:k]
    for k, hits in text_hits.items():
        values[f"text_to_image_recall_at_{k}_percent"] = round(100 * hits / len(captions), 4)
    # Image -> text: chunk images for the same reason.
    image_hits = {1: 0, 5: 0, 10: 0}
    for offset in range(0, len(image_features), 256):
        scores = image_features[offset:offset + 256] @ text_features.T
        ranked = scores.topk(min(10, scores.shape[1]), dim=1).indices.tolist()
        for local_index, predictions in enumerate(ranked):
            image_index = offset + local_index
            for k in image_hits:
                image_hits[k] += any(targets[candidate] == image_index for candidate in predictions[:k])
    for k, hits in image_hits.items():
        values[f"image_to_text_recall_at_{k}_percent"] = round(100 * hits / len(image_features), 4)
    return values


def worker(args) -> int:
    import torch
    from PIL import Image

    output = Path(args.worker_output)
    official = OFFICIAL_CLAIMS.get((args.worker_model_id, args.dataset))
    result = {
        "status": "failed",
        "purpose": ("same_checkpoint_official_metric_acceptance" if official
                    else "NVIDIA retrieval engineering baseline"),
        "model_id": args.worker_model_id, "backend": args.worker_backend,
        "dataset": args.dataset, "dataset_revision": DATASETS[args.dataset]["revision"],
        "started_at": now(),
    }
    adapter = None
    try:
        records = load_records(args.dataset)
        adapter = b01.load_adapter(args.worker_model_id, args.worker_backend)
        image_features = []
        text_features = []
        captions = [caption for _, _, group in records for caption in group]
        targets = [index for index, record in enumerate(records) for _ in record[2]]
        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        for offset in range(0, len(records), args.image_batch_size):
            images = []
            for _, image_bytes, _ in records[offset:offset + args.image_batch_size]:
                with Image.open(io.BytesIO(image_bytes)) as source:
                    images.append(source.convert("RGB").copy())
            image_features.append(adapter.image_features(images).cpu())
            seen = min(offset + args.image_batch_size, len(records))
            if seen % 1000 < args.image_batch_size:
                print(f"encoded {seen:,} {args.dataset} images", flush=True)
        for offset in range(0, len(captions), args.text_batch_size):
            text_features.append(adapter.text_features(captions[offset:offset + args.text_batch_size]).cpu())
        image_features = torch.cat(image_features)
        text_features = torch.cat(text_features)
        metrics = recall_metrics(image_features, text_features, captions, targets)
        elapsed = time.perf_counter() - started
        result.update(
            status="passed", image_count=len(records), caption_count=len(captions), **metrics,
            images_and_captions_per_second=round((len(records) + len(captions)) / elapsed, 3),
            peak_gpu_memory_mb=round(torch.cuda.max_memory_allocated() / 1024**2, 2),
            environment={"python": platform.python_version(), "torch": torch.__version__,
                         "cuda_runtime": torch.version.cuda, "gpu": torch.cuda.get_device_name(0)},
        )
        if official:
            relative_gaps = {
                key: round(abs(metrics[key] - reference) / reference * 100, 4)
                for key, reference in official.items()
            }
            result.update(
                official_reference=official,
                official_evidence="BAAI/AltCLIP official model card and AltCLIP paper Table 2, Flickr30kEN",
                relative_gap_percent=relative_gaps,
                within_5_percent_relative_all_metrics=all(value <= 5 for value in relative_gaps.values()),
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
    print(json.dumps({"model_id": args.worker_model_id, "dataset": args.dataset,
                      "status": result["status"],
                      "t2i_r1": result.get("text_to_image_recall_at_1_percent")}), flush=True)
    return 0 if result["status"] == "passed" else 1


def mapped_rows(dataset: str):
    mapping = list(csv.DictReader((ROOT / "evaluation_audit/model_evaluation_mapping.csv")
                                  .open(encoding="utf-8-sig")))
    wanted = {row["模型名称"] for row in mapping
              if DATASETS[dataset]["mapping_needle"] in row["补充测试集"]}
    return [row for row in b01.ledger_rows() if str(row["模型名称"]) in wanted]


def coordinator(args) -> int:
    requested = set(args.model_id or [])
    failures = 0
    for dataset in args.dataset or DATASETS:
        # Explicit model selection is authoritative after the dataset/model
        # official-claim preflight; the older supplemental mapping must not
        # silently turn an explicitly requested formal run into a no-op.
        source_rows = (b01.ledger_rows() if requested else mapped_rows(dataset))
        rows = [row for row in source_rows if not requested or str(row["模型名称"]) in requested]
        missing = requested - {str(row["模型名称"]) for row in rows}
        if missing:
            raise RuntimeError(f"requested models missing from ledger: {sorted(missing)}")
        for index, row in enumerate(rows, 1):
            model_id = str(row["模型名称"])
            backend = b01.backend_for(row)
            output = RESULT_ROOT.joinpath(dataset, *model_id.split("/"), "summary.json")
            family_output = ROOT.joinpath(
                "results", "pe_core_flickr30k" if dataset == "flickr30k" else "pe_core_mscoco_karpathy",
                *model_id.split("/"), "summary.json")
            old_output = family_output if family_output.is_file() else output
            if old_output.is_file() and not args.force:
                old = json.loads(old_output.read_text(encoding="utf-8"))
                if old.get("status") == "passed":
                    print(f"[{dataset} {index}/{len(rows)}] SKIP {model_id}", flush=True)
                    continue
            if backend not in {"Transformers AutoModel/AutoProcessor", "OpenCLIP"}:
                output.parent.mkdir(parents=True, exist_ok=True)
                output.write_text(json.dumps({"status": "unsupported_backend", "model_id": model_id,
                    "dataset": dataset, "backend": backend, "finished_at": now()},
                    ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
                failures += 1
                continue
            command = [WORKER_PYTHON, str(Path(__file__).resolve()), "--dataset", dataset,
                       "--worker-model-id", model_id, "--worker-backend", backend,
                       "--worker-output", str(output), "--image-batch-size", str(args.image_batch_size),
                       "--text-batch-size", str(args.text_batch_size)]
            print(f"[{dataset} {index}/{len(rows)}] RUN {model_id}", flush=True)
            try:
                completed = subprocess.run(command, cwd=ROOT, check=False, timeout=args.timeout_seconds)
                failures += int(completed.returncode != 0)
            except subprocess.TimeoutExpired:
                output.parent.mkdir(parents=True, exist_ok=True)
                output.write_text(json.dumps({"status": "timeout", "model_id": model_id,
                    "dataset": dataset, "timeout_seconds": args.timeout_seconds, "finished_at": now()},
                    ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
                failures += 1
    return 1 if failures else 0


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", action="append", choices=DATASETS)
    parser.add_argument("--model-id", action="append")
    parser.add_argument("--image-batch-size", type=int, default=16)
    parser.add_argument("--text-batch-size", type=int, default=256)
    parser.add_argument("--timeout-seconds", type=int, default=7200)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--worker-model-id", help=argparse.SUPPRESS)
    parser.add_argument("--worker-backend", help=argparse.SUPPRESS)
    parser.add_argument("--worker-output", help=argparse.SUPPRESS)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.worker_model_id:
        if not args.dataset or len(args.dataset) != 1:
            raise SystemExit("worker requires exactly one --dataset")
        args.dataset = args.dataset[0]
        return worker(args)
    return coordinator(args)


if __name__ == "__main__":
    raise SystemExit(main())
