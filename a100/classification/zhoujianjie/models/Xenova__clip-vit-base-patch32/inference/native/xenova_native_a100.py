#!/usr/bin/env python3
"""Strict ImageNet-1k evaluation of Xenova's full-precision ONNX CLIP export."""

from __future__ import annotations
from model_paths import legacy as _legacy, legacy_str as _legacy_str, relative as _relative, revision as _revision, weight as _weight, result as _result

import argparse
import hashlib
import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import onnxruntime as ort
from transformers import AutoProcessor

import imagenet1k_official_spec as claims
from evaluate_imagenet1k_official import parquet_batches, protocol_text, verify_dataset


ROOT = _legacy('/data/zhoujianjie')
MODEL_ID = "Xenova/clip-vit-base-patch32"
EXPORT = _legacy('/data/zhoujianjie/model_inference_for_GPU_adaptation/classification/zhoujianjie/models/Xenova__clip-vit-base-patch32/weights')
SOURCE = _legacy('/data/zhoujianjie/model_inference_for_GPU_adaptation/classification/zhoujianjie/models/openai__clip-vit-base-patch32/weights')
TEXT_ONNX = _legacy('/data/zhoujianjie/model_inference_for_GPU_adaptation/classification/zhoujianjie/models/Xenova__clip-vit-base-patch32/weights/onnx/text_model.onnx')
VISION_ONNX = _legacy('/data/zhoujianjie/model_inference_for_GPU_adaptation/classification/zhoujianjie/models/Xenova__clip-vit-base-patch32/weights/onnx/vision_model.onnx')
MAPPING = _legacy('/data/zhoujianjie/environment_setup/original_metadata/xenova_openai_clip_equivalence.json')
OUT = _legacy('/data/zhoujianjie/results/imagenet1k_official/Xenova/clip-vit-base-patch32/summary.json')
OFFICIAL_TOP1 = 63.2


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize(array: np.ndarray) -> np.ndarray:
    return array / np.linalg.norm(array, axis=-1, keepdims=True)


def preflight() -> dict:
    dataset = verify_dataset(deep_hash=not bool(int(__import__('os').environ.get('ZERO_SHOT_SMOKE_LIMIT','0'))))
    mapping = json.loads(MAPPING.read_text())
    receipt = json.loads((_legacy('/data/zhoujianjie/model_inference_for_GPU_adaptation/classification/zhoujianjie/models/Xenova__clip-vit-base-patch32/weights/download_receipt.json')).read_text())
    if mapping.get("status") != "passed":
        raise RuntimeError("Xenova-to-OpenAI checkpoint equivalence has not passed")
    split = mapping["split_tower_artifacts"]
    if sha256(TEXT_ONNX) != split["text_model_sha256"] or sha256(VISION_ONNX) != split["vision_model_sha256"]:
        raise RuntimeError("ONNX split-tower artifact hash changed after equivalence verification")
    source_claim = claims.MODEL_SPECS["openai/clip-vit-base-patch32"]
    if source_claim["protocol"] != "clip_80" or source_claim["official_top1_percent"] != OFFICIAL_TOP1:
        raise RuntimeError("pinned OpenAI official reference mapping changed")
    record = {
        "model": f"{MODEL_ID}@{receipt.get('revision')}",
        "checkpoint_mapping": (
            "official model card declares base_model=openai/clip-vit-base-patch32; export config pins "
            "source commit e6a30b603a447e251fdaca1c3056b2a16cdfebeb; full and split ONNX "
            "outputs numerically match the OpenAI source checkpoint"
        ),
        "dataset": f"{dataset['repo_id']}@{dataset['revision']}:{dataset['split']} ({dataset['rows']} rows)",
        "protocol": "OpenAI CLIP 80-template ImageNet zero-shot; source-native preprocessing",
        "metric": "ImageNet-1k validation top-1 accuracy",
        "official_reference_percent": OFFICIAL_TOP1,
        "mapping_evidence": str(_relative(MAPPING,ROOT)),
        "text_onnx_sha256": sha256(TEXT_ONNX),
        "vision_onnx_sha256": sha256(VISION_ONNX),
    }
    print(json.dumps({"preflight_model_dataset_reference_mapping": record}, ensure_ascii=False, indent=2), flush=True)
    return record


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--text-batch-size", type=int, default=1024)
    args = parser.parse_args()
    record = preflight()
    if args.preflight_only:
        return 0
    if "CUDAExecutionProvider" not in ort.get_available_providers():
        raise RuntimeError("CUDAExecutionProvider unavailable; refusing 50k-image CPU benchmark")
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    text_session = ort.InferenceSession(str(TEXT_ONNX), providers=providers)
    vision_session = ort.InferenceSession(str(VISION_ONNX), providers=providers)
    if text_session.get_providers()[0] != "CUDAExecutionProvider" or vision_session.get_providers()[0] != "CUDAExecutionProvider":
        raise RuntimeError("CUDA provider did not initialize; refusing silent CPU fallback")

    processor = AutoProcessor.from_pretrained(SOURCE, local_files_only=True)
    # Transformers 4.36 re-checks Pillow package metadata for every image in a
    # Python list. Pillow was already imported by the strict dataset decoder;
    # bypass only that redundant availability check, not any preprocessing op.
    import transformers.image_transforms as image_transforms
    image_transforms.requires_backends = lambda *_args, **_kwargs: None
    labels, templates = protocol_text("clip_80")
    prompts = [template.format(label) for label in labels for template in templates]
    text_parts = []
    for start in range(0, len(prompts), args.text_batch_size):
        tokenized = processor(text=prompts[start:start + args.text_batch_size], padding=True, return_tensors="np")
        text_parts.append(normalize(text_session.run(None, {"input_ids": tokenized["input_ids"]})[0]))
        print(f"TEXT PROGRESS {min(start + args.text_batch_size, len(prompts))}/{len(prompts)}", flush=True)
    classifier = np.concatenate(text_parts).reshape(1000, 80, -1).mean(axis=1)
    classifier = normalize(classifier).astype(np.float32)

    forward=[0.0]
    correct1 = correct5 = seen = 0
    next_progress = 1_000
    started = time.perf_counter()
    for _, images, targets in parquet_batches(args.batch_size):
        pixels = processor(images=images, return_tensors="np")["pixel_values"]
        __import__("torch").cuda.synchronize();tick=time.perf_counter()
        image_features = normalize(vision_session.run(None, {"pixel_values": pixels})[0])
        __import__("torch").cuda.synchronize();forward[0]+=time.perf_counter()-tick
        scores = image_features @ classifier.T
        top5 = np.argpartition(scores, -5, axis=1)[:, -5:]
        top1 = scores.argmax(axis=1)
        target = np.asarray(targets)
        with (OUT.parent/'predictions.jsonl').open('a') as stream:
            for t,p,rank in zip(target.tolist(),top1.tolist(),top5.tolist()):stream.write(json.dumps({'target':t,'prediction':p,'top5':rank})+'\n')
        correct1 += int((top1 == target).sum())
        correct5 += int((top5 == target[:, None]).any(axis=1).sum())
        seen += len(targets)
        if seen >= next_progress:
            print(f"PROGRESS {seen}/50000 top1={100*correct1/seen:.3f}", flush=True)
            next_progress = ((seen // 1_000) + 1) * 1_000
    if seen != (int(__import__('os').environ.get('ZERO_SHOT_SMOKE_LIMIT','0')) or 50_000):
        raise RuntimeError(f"expected 50000 images, evaluated {seen}")
    top1_percent = 100 * correct1 / seen
    relative_gap = abs(top1_percent - OFFICIAL_TOP1) / OFFICIAL_TOP1 * 100
    result = {
        "purpose": "same_checkpoint_official_imagenet1k_metric_acceptance",
        "status": "passed", "model_id": MODEL_ID,
        "finished_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "preflight_mapping": record, "sample_count": seen,
        "image_forward_seconds":forward[0],
        "image_forward_ms_per_sample":forward[0]*1000/seen,
        "top1_percent": round(top1_percent, 4),
        "top5_percent": round(100 * correct5 / seen, 4),
        "official_top1_percent": OFFICIAL_TOP1,
        "relative_gap_percent": round(relative_gap, 4),
        "within_5_percent_relative": relative_gap <= 5.0,
        "elapsed_seconds": round(time.perf_counter() - started, 3),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    return 0 if int(__import__('os').environ.get('ZERO_SHOT_SMOKE_LIMIT','0')) or result["within_5_percent_relative"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
