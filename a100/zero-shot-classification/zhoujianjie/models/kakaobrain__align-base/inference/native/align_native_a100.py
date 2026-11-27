#!/usr/bin/env python3
"""Strict official Flickr30K retrieval evaluation for kakaobrain/align-base."""

from __future__ import annotations
from model_paths import legacy as _legacy, legacy_str as _legacy_str, relative as _relative, revision as _revision, weight as _weight, result as _result

import argparse
import hashlib
import io
import json
import platform
import tarfile
import time
from datetime import datetime
from pathlib import Path


ROOT = _legacy('/data/zhoujianjie')
MODEL_ID = "kakaobrain/align-base"
MODEL_REVISION = "e96a37facc7b1f59090ece82293226b817afd6ba"
MODEL_DIR = _legacy('/data/zhoujianjie/model_inference_for_GPU_adaptation/classification/zhoujianjie/models/kakaobrain__align-base/weights')
DATASET_DIR = _legacy('/data/zhoujianjie/model_inference_for_GPU_adaptation/dataset/raw/flickr30k_1')
DATASET_MANIFEST = _legacy('/data/zhoujianjie/environment_setup/native_sources/evaluation_audit/b04_flickr30k_download_manifest.json')
KARPATHY_JSON = _legacy('/data/zhoujianjie/model_inference_for_GPU_adaptation/dataset/raw/flickr30k_2/flickr30k_test.json')
CONVERSION_EVIDENCE = _legacy('/data/zhoujianjie/environment_setup/native_sources/evaluation_audit/kakao_align_checkpoint_conversion_evidence.json')
OFFICIAL_REPO = _legacy('/data/zhoujianjie/environment_setup/native_sources/evaluation_audit/reference_code/coyo-align')
OFFICIAL_REPO_COMMIT = "8cea82e5381b7c4806967746473c6053b6817905"
PREFLIGHT = _legacy('/data/zhoujianjie/environment_setup/native_sources/evaluation_audit/kakao_align_flickr30k_official_preflight.json')
OUTPUT = _legacy('/data/zhoujianjie/results/flickr30k_official/kakaobrain/align-base/summary.json')
REFERENCE = {"image_to_text_recall_at_1_percent": 88.1, "text_to_image_recall_at_1_percent": 73.2}


def now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_records():
    def undo_webdataset_csv_quoting(caption: str) -> str:
        # The frozen clip-benchmark WebDataset serialized captions through CSV.
        # Undo only that reversible quoting so inputs byte-match Karpathy JSON.
        if len(caption) >= 2 and caption[0] == caption[-1] == '"':
            caption = caption[1:-1]
        return caption.replace('""', '"')

    records = []
    for shard in sorted(DATASET_DIR.glob("*.tar")):
        with tarfile.open(shard) as archive:
            members = {Path(m.name).name: m for m in archive if m.isfile()}
            stems = sorted({Path(name).stem for name in members})
            for stem in stems:
                image_member, text_member = members.get(f"{stem}.jpg"), members.get(f"{stem}.txt")
                if image_member is None or text_member is None:
                    raise RuntimeError(f"unpaired Flickr30K record {stem}")
                image_bytes = archive.extractfile(image_member).read()
                captions = [
                    undo_webdataset_csv_quoting(caption)
                    for caption in archive.extractfile(text_member).read().decode("utf-8").splitlines()
                ]
                records.append((stem, image_bytes, captions))
    records.sort(key=lambda x: x[0])
    return records


def run_preflight(records) -> dict:
    manifest = json.loads(DATASET_MANIFEST.read_text())
    conversion = json.loads(CONVERSION_EVIDENCE.read_text())
    karpathy = json.loads(KARPATHY_JSON.read_text())
    expected_captions = [row["caption"] for row in karpathy]
    actual_captions = [row[2] for row in records]
    checks = {
        "model_revision_fixed": json.loads((_legacy('/data/zhoujianjie/model_inference_for_GPU_adaptation/classification/zhoujianjie/models/kakaobrain__align-base/weights/download_receipt.json')).read_text())["revision"] == MODEL_REVISION,
        "checkpoint_exact_lossless_conversion": conversion.get("verdict") == "exact_lossless_conversion",
        "dataset_revision_fixed": manifest["datasets"][0]["revision"] == "f8af3ceee1b944bf761340682621dfafd23974d2",
        "dataset_shards_verified": all(row["verified"] for row in manifest["datasets"][0]["files"]),
        "karpathy_test_image_count_1000": len(records) == len(karpathy) == 1000,
        "five_captions_each": all(len(row[2]) == 5 for row in records),
        "captions_and_order_exactly_match_frozen_karpathy_test": actual_captions == expected_captions,
        "official_evaluator_present": (_legacy('/data/zhoujianjie/environment_setup/native_sources/evaluation_audit/reference_code/coyo-align/evaluate/eval_retrieval.py')).is_file(),
        "official_reference_values_present": set(REFERENCE) == {
            "image_to_text_recall_at_1_percent", "text_to_image_recall_at_1_percent"
        },
    }
    passed = all(checks.values())
    result = {
        "generated_at": now(),
        "status": "passed" if passed else "hard_failure_before_gpu",
        "mapping": {
            "model": MODEL_ID,
            "target_revision": MODEL_REVISION,
            "original_checkpoint": "kakaobrain/coyo-align-b7-base@248f3b899593e489d73ed6ddf88469ad042ccbe1",
            "checkpoint_equivalence": str(_relative(CONVERSION_EVIDENCE,ROOT)),
            "dataset": "Flickr30K Karpathy test",
            "dataset_split": "test: 1,000 images, 5,000 raw captions",
            "dataset_revision": manifest["datasets"][0]["revision"],
            "protocol": "COYO-ALIGN official raw-caption dual-encoder cosine retrieval; R@1 over the full split; image resize 346 then center crop 289; BERT uncased max length 64",
            "official_reference": REFERENCE,
            "reference_source": "COYO-ALIGN README result table at official repository commit",
            "official_code_commit": OFFICIAL_REPO_COMMIT,
            "metric_code": "evaluate/eval_retrieval.py: cosine similarity and R@1 exact image/caption-id match",
        },
        "checks": checks,
        "frozen_assets": {
            "dataset_manifest_sha256": sha256(DATASET_MANIFEST),
            "karpathy_test_json_sha256": sha256(KARPATHY_JSON),
            "preprocessor_config_sha256": sha256(_legacy('/data/zhoujianjie/model_inference_for_GPU_adaptation/classification/zhoujianjie/models/kakaobrain__align-base/weights/preprocessor_config.json')),
            "official_eval_retrieval_sha256": sha256(_legacy('/data/zhoujianjie/environment_setup/native_sources/evaluation_audit/reference_code/coyo-align/evaluate/eval_retrieval.py')),
        },
    }
    PREFLIGHT.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["mapping"], indent=2), flush=True)
    print("PREFLIGHT", result["status"], flush=True)
    if not passed:
        missing = [key for key, value in checks.items() if not value]
        raise RuntimeError(f"strict preflight failed before GPU: {missing}")
    return result


def recall_metrics(image_features, text_features, targets):
    import torch

    similarity = image_features @ text_features.T
    result = {}
    for k in (1, 5, 10):
        t2i = similarity.T.topk(k, dim=1).indices
        target_tensor = torch.tensor(targets).view(-1, 1)
        result[f"text_to_image_recall_at_{k}_percent"] = round(
            100.0 * (t2i == target_tensor).any(dim=1).float().mean().item(), 4
        )
        i2t = similarity.topk(k, dim=1).indices.tolist()
        hits = sum(any(targets[caption_index] == image_index for caption_index in predictions)
                   for image_index, predictions in enumerate(i2t))
        result[f"image_to_text_recall_at_{k}_percent"] = round(100.0 * hits / len(image_features), 4)
    return result


def evaluate(records, image_batch_size: int, text_batch_size: int) -> dict:
    import torch
    from PIL import Image
    from transformers import AlignModel, AlignProcessor

    result = {
        "status": "failed",
        "purpose": "same_checkpoint_same_protocol_official_metric_acceptance",
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "dataset": "Flickr30K Karpathy test",
        "official_code_commit": OFFICIAL_REPO_COMMIT,
        "official_reference": REFERENCE,
        "precision": "FP32",
        "started_at": now(),
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    try:
        processor = AlignProcessor.from_pretrained(MODEL_DIR, local_files_only=True)
        model = AlignModel.from_pretrained(MODEL_DIR, local_files_only=True).eval().cuda()
        captions = [caption for _, _, group in records for caption in group]
        targets = [index for index, row in enumerate(records) for _ in row[2]]
        image_features, text_features = [], []
        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        forward_seconds = 0.0
        with torch.inference_mode():
            for offset in range(0, len(records), image_batch_size):
                images = []
                for _, image_bytes, _ in records[offset:offset + image_batch_size]:
                    with Image.open(io.BytesIO(image_bytes)) as image:
                        images.append(image.convert("RGB").copy())
                inputs = processor(images=images, return_tensors="pt")
                pixels = inputs["pixel_values"].cuda(non_blocking=True)
                torch.cuda.synchronize(); tick = time.perf_counter()
                features = model.get_image_features(pixel_values=pixels)
                torch.cuda.synchronize(); forward_seconds += time.perf_counter()-tick
                image_features.append(torch.nn.functional.normalize(features.float(), dim=-1).cpu())
                print(f"images {min(offset + image_batch_size, len(records))}/{len(records)}", flush=True)
            for offset in range(0, len(captions), text_batch_size):
                inputs = processor(
                    text=captions[offset:offset + text_batch_size], return_tensors="pt",
                    padding="max_length", truncation=True, max_length=64,
                )
                inputs = {key: value.cuda(non_blocking=True) for key, value in inputs.items()}
                features = model.get_text_features(**inputs)
                text_features.append(torch.nn.functional.normalize(features.float(), dim=-1).cpu())
        image_features, text_features = torch.cat(image_features), torch.cat(text_features)
        torch.save({'image_features':image_features,'text_features':text_features,'targets':targets}, OUTPUT.parent/'embeddings.pt')
        ranks = (text_features @ image_features.T).topk(10,dim=1).indices.tolist()
        with (OUTPUT.parent/'predictions.jsonl').open('w') as stream:
            for i,ranking in enumerate(ranks):stream.write(json.dumps({'query_index':i,'target_image_index':targets[i],'ranked_image_indices':ranking})+'\n')
        metrics = recall_metrics(image_features, text_features, targets)
        elapsed = time.perf_counter() - started
        gaps = {key: round(abs(metrics[key] - value) / value * 100.0, 4) for key, value in REFERENCE.items()}
        accepted = all(value <= 5.0 for value in gaps.values())
        result.update(
            status="passed", image_count=len(records), caption_count=len(captions), **metrics,
            relative_gap_percent=gaps, acceptance_threshold="all published R@1 relative gaps <= 5%",
            official_metric_accepted=accepted,
            elapsed_seconds=round(elapsed, 3),
            forward_total_seconds=forward_seconds, forward_ms_per_sample=forward_seconds*1000/len(records),
            peak_gpu_memory_mb=round(torch.cuda.max_memory_allocated() / 1024**2, 2),
            environment={"python": platform.python_version(), "torch": torch.__version__,
                         "transformers": __import__("transformers").__version__,
                         "cuda_runtime": torch.version.cuda, "gpu": torch.cuda.get_device_name(0)},
        )
    except Exception as error:
        import traceback
        result.update(error_type=type(error).__name__, error=str(error), traceback=traceback.format_exc(limit=30))
    finally:
        result["finished_at"] = now()
        OUTPUT.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--image-batch-size", type=int, default=16)
    parser.add_argument("--text-batch-size", type=int, default=256)
    args = parser.parse_args()
    records = load_records()
    run_preflight(records)
    if args.preflight_only:
        return 0
    result = evaluate(records, args.image_batch_size, args.text_batch_size)
    return 0 if result.get("status") == "passed" and result.get("official_metric_accepted") else 1


if __name__ == "__main__":
    raise SystemExit(main())
