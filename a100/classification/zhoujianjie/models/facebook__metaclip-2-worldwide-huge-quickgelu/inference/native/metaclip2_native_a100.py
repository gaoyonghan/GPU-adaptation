#!/usr/bin/env python3
"""Strict MetaCLIP2 evaluation on the official CVQA LOCAL test task."""

from __future__ import annotations
from model_paths import legacy as _legacy, legacy_str as _legacy_str, relative as _relative, revision as _revision, weight as _weight, result as _result

import argparse
import hashlib
import io
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import torch
import transformers
from PIL import Image
from transformers import AutoModel, AutoProcessor


ROOT = _legacy('/data/zhoujianjie')
MODEL_ID = "facebook/metaclip-2-worldwide-huge-quickgelu"
MODEL_REVISION = "c139061af7b10fdb2e754b60d2b1182a3d5526c2"
DATASET_ID = "afaji/cvqa"
DATASET_REVISION = "6a38d71fa391ca7a5e061bbcf48ba714a9a9f4f1"
CODE_REVISION = "f47f7841f6a91cc5676729a3d125519393d87d1e"
OFFICIAL_LOCAL_ACCURACY_PERCENT = 57.4
EXPECTED_ROWS = 10374
MODEL_DIR = _legacy('/data/zhoujianjie/model_inference_for_GPU_adaptation/classification/zhoujianjie/models/facebook__metaclip-2-worldwide-huge-quickgelu/weights')
DATASET_DIR = _legacy('/data/zhoujianjie/model_inference_for_GPU_adaptation/dataset/raw/cvqa')
CODE_DIR = _legacy('/data/zhoujianjie/environment_setup/native_sources/evaluation_audit/reference_code/MetaCLIP')
OUT = _legacy('/data/zhoujianjie/results/metaclip2_cvqa_official/summary.json')
PROGRESS = _legacy('/data/zhoujianjie/results/metaclip2_cvqa_official/progress.json')


def load_pyarrow():
    try:
        import pyarrow.parquet as pq
    except ImportError:
        # The MetaCLIP2 runtime is isolated from the evaluation-data environment.
        sys.path.append(_legacy_str('/home/dell/anaconda3/envs/UMind_agent/lib/python3.10/site-packages'))
        import pyarrow.parquet as pq
    return pq


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parquet_files_and_metadata() -> list[tuple[Path, Path]]:
    pairs = []
    for index in range(10):
        name = f"test-{index:05d}-of-00010.parquet"
        pairs.append((
            _legacy('/data/zhoujianjie/model_inference_for_GPU_adaptation/dataset/raw/cvqa/data') / name,
            _legacy('/data/zhoujianjie/model_inference_for_GPU_adaptation/dataset/raw/cvqa/.cache/huggingface/download/data') / f"{name}.metadata",
        ))
    return pairs


def preflight() -> dict[str, object]:
    pq = load_pyarrow()
    errors = []
    receipt_path = _legacy('/data/zhoujianjie/model_inference_for_GPU_adaptation/classification/zhoujianjie/models/facebook__metaclip-2-worldwide-huge-quickgelu/weights/download_receipt.json')
    if not receipt_path.is_file():
        errors.append("checkpoint download receipt missing")
    else:
        receipt = json.loads(receipt_path.read_text())
        if receipt.get("model_id") != MODEL_ID or receipt.get("revision") != MODEL_REVISION:
            errors.append("checkpoint ID/revision mismatch")

    model_card = (_legacy('/data/zhoujianjie/model_inference_for_GPU_adaptation/classification/zhoujianjie/models/facebook__metaclip-2-worldwide-huge-quickgelu/weights/README.md')).read_text(errors="replace")
    if 'corresponds to "ViT-H-14-quickgelu-worldwide"' not in model_card:
        errors.append("official HF-to-original-checkpoint mapping is missing")

    code_revision = _revision(str(CODE_DIR),status=False).strip()
    code_changes = _revision(str(CODE_DIR),status=True).strip()
    if code_revision != CODE_REVISION:
        errors.append("MetaCLIP official evaluator revision mismatch")
    if code_changes:
        errors.append("MetaCLIP official evaluator worktree is not clean")

    rows = 0
    invalid_labels = 0
    invalid_options = 0
    files = []
    for parquet, metadata in parquet_files_and_metadata():
        if not parquet.is_file() or not metadata.is_file():
            errors.append(f"missing CVQA artifact: {parquet.name}")
            continue
        metadata_lines = metadata.read_text().splitlines()
        if len(metadata_lines) < 2 or metadata_lines[0] != DATASET_REVISION:
            errors.append(f"CVQA revision mismatch: {parquet.name}")
            continue
        expected_hash = metadata_lines[1]
        actual_hash = sha256(parquet)
        if actual_hash != expected_hash:
            errors.append(f"CVQA hash mismatch: {parquet.name}")
            continue
        table = pq.read_table(parquet, columns=["Label", "Options"])
        labels = table.column("Label").to_pylist()
        options = table.column("Options").to_pylist()
        rows += len(labels)
        invalid_labels += sum(not isinstance(value, int) or not 0 <= value <= 3 for value in labels)
        invalid_options += sum(len(value) != 4 for value in options)
        files.append({"path": _relative(parquet,ROOT).as_posix(), "sha256": actual_hash, "rows": len(labels)})
    if rows != EXPECTED_ROWS:
        errors.append(f"CVQA test row count mismatch: expected {EXPECTED_ROWS}, found {rows}")
    if invalid_labels:
        errors.append(f"CVQA contains {invalid_labels} labels outside [0, 3]")
    if invalid_options:
        errors.append(f"CVQA contains {invalid_options} rows without exactly four options")

    mapping = {
        "checkpoint": f"{MODEL_ID}@{MODEL_REVISION}",
        "checkpoint_mapping": "official model card -> ViT-H-14-quickgelu-worldwide",
        "dataset": f"{DATASET_ID}@{DATASET_REVISION}:test ({EXPECTED_ROWS} rows), LOCAL fields",
        "protocol": f"MetaCLIP@{CODE_REVISION} clipeval/cvqa/eval_cvqa.py; Question + space + each of four Options; native processor; normalized cosine argmax",
        "metric": "CVQA-LOCAL zero-shot accuracy",
        "same_checkpoint_reference_percent": OFFICIAL_LOCAL_ACCURACY_PERCENT,
        "reference": "MetaCLIP official README, Meta CLIP 2 pretrained-model table",
        "transformers_version": transformers.__version__,
        "dataset_files": files,
        "status": "PASS" if not errors else "HARD_FAIL",
        "errors": errors,
    }
    print("STRICT PREFLIGHT MODEL->DATASET->PROTOCOL->METRIC")
    print(json.dumps(mapping, indent=2))
    if errors:
        raise SystemExit(2)
    return mapping


def open_image(value: dict[str, object]) -> Image.Image:
    data = value.get("bytes")
    if data is None:
        path = value.get("path")
        if not path:
            raise ValueError("CVQA image has neither bytes nor path")
        image = Image.open(path)
    else:
        image = Image.open(io.BytesIO(data))
    return image.convert("RGB")


def evaluate(mapping: dict[str, object], batch_size: int) -> None:
    pq = load_pyarrow()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the full MetaCLIP2 CVQA evaluation")
    model = AutoModel.from_pretrained(
        MODEL_DIR, local_files_only=True, dtype=torch.bfloat16, attn_implementation="sdpa"
    ).cuda().eval()
    processor = AutoProcessor.from_pretrained(MODEL_DIR, local_files_only=True)

    timing = [0.0]
    progress_key = {
        "model_revision": MODEL_REVISION,
        "dataset_revision": DATASET_REVISION,
        "code_revision": CODE_REVISION,
    }
    if PROGRESS.is_file():
        progress = json.loads(PROGRESS.read_text())
        if progress.get("key") != progress_key:
            raise RuntimeError("saved CVQA progress belongs to a different frozen mapping")
        completed_shards = set(progress["completed_shards"])
        correct = int(progress["correct"])
        total = int(progress["total"])
        subset_counts = progress["subset_counts"]
        print(f"Resuming CVQA LOCAL from {total}/{EXPECTED_ROWS}", flush=True)
    else:
        completed_shards: set[str] = set()
        correct = total = 0
        subset_counts: dict[str, list[int]] = {}
    columns = ["image", "Question", "Options", "Label", "Subset"]
    with torch.inference_mode():
        for parquet, _ in parquet_files_and_metadata():
            if parquet.name in completed_shards:
                continue
            parquet_file = pq.ParquetFile(parquet)
            for batch_index, batch in enumerate(parquet_file.iter_batches(batch_size=batch_size, columns=columns)):
                values = batch.to_pydict()
                _limit=int(__import__('os').environ.get('ZERO_SHOT_SMOKE_LIMIT','0'))
                if _limit:values={k:v[:_limit] for k,v in values.items()}
                images = [open_image(value) for value in values["image"]]
                texts = [
                    question + " " + option
                    for question, options in zip(values["Question"], values["Options"])
                    for option in options
                ]
                image_inputs = processor(images=images, return_tensors="pt")
                text_inputs = processor(
                    text=texts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=model.config.text_config.max_position_embeddings,
                )
                torch.cuda.synchronize(); tick = __import__("time").perf_counter()
                image_features = model.get_image_features(
                    pixel_values=image_inputs["pixel_values"].cuda(non_blocking=True)
                ).pooler_output
                text_features = model.get_text_features(
                    input_ids=text_inputs["input_ids"].cuda(non_blocking=True),
                    attention_mask=text_inputs["attention_mask"].cuda(non_blocking=True),
                ).pooler_output
                image_features = torch.nn.functional.normalize(image_features.float(), dim=-1)
                text_features = torch.nn.functional.normalize(text_features.float(), dim=-1)
                predictions = torch.bmm(
                    image_features[:, None, :], text_features.view(len(images), 4, -1).transpose(1, 2)
                ).squeeze(1).argmax(1).cpu().tolist()
                torch.cuda.synchronize()
                timing[0] += __import__('time').perf_counter()-tick
                with (OUT.parent/'predictions.jsonl').open('a') as stream:
                    for index,(prediction,label,subset) in enumerate(zip(predictions,values['Label'],values['Subset'])):stream.write(json.dumps({'shard':parquet.name,'batch':batch_index,'index':index,'prediction':prediction,'target':label,'subset':subset})+'\n')
                if _limit:
                    (OUT.parent/'smoke_native.json').write_text(json.dumps({'state':'smoke_completed','samples':len(predictions)}))
                    return
                for prediction, label, subset in zip(predictions, values["Label"], values["Subset"]):
                    hit = int(prediction == label)
                    correct += hit
                    total += 1
                    counts = subset_counts.setdefault(subset, [0, 0])
                    counts[0] += hit
                    counts[1] += 1
                if batch_index % 16 == 0:
                    print(f"CVQA LOCAL: {total}/{EXPECTED_ROWS}", flush=True)
            completed_shards.add(parquet.name)
            PROGRESS.parent.mkdir(parents=True, exist_ok=True)
            PROGRESS.write_text(json.dumps({
                "key": progress_key,
                "completed_shards": sorted(completed_shards),
                "correct": correct,
                "total": total,
                "subset_counts": subset_counts,
            }, indent=2, ensure_ascii=False) + "\n")
            print(f"CVQA LOCAL shard complete: {total}/{EXPECTED_ROWS}", flush=True)

    measured = correct / total * 100
    relative_gap = abs(measured - OFFICIAL_LOCAL_ACCURACY_PERCENT) / OFFICIAL_LOCAL_ACCURACY_PERCENT * 100
    result = {
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "status": "passed" if relative_gap <= 5 else "failed",
        "mapping": mapping,
        "samples": total,
        "forward_total_seconds":timing[0],
        "forward_ms_per_sample":timing[0]*1000/total,
        "correct": correct,
        "cvqa_local_accuracy_percent": round(measured, 6),
        "official_reference_percent": OFFICIAL_LOCAL_ACCURACY_PERCENT,
        "relative_gap_percent": round(relative_gap, 6),
        "within_5_percent_relative": relative_gap <= 5,
        "subset_accuracy": {key: values[0] / values[1] for key, values in subset_counts.items()},
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(result, indent=2, ensure_ascii=False))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()
    mapping = preflight()
    if args.run:
        evaluate(mapping, args.batch_size)


if __name__ == "__main__":
    main()
