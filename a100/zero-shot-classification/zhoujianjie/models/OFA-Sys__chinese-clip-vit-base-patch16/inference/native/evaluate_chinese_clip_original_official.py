#!/usr/bin/env python3
"""Evaluate Chinese-CLIP with the original official cn_clip checkpoints/backend."""

from __future__ import annotations
from model_paths import legacy as _legacy, legacy_str as _legacy_str, relative as _relative, revision as _revision, weight as _weight, result as _result

import argparse
import hashlib
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT / "evaluation_audit/reference_code/Chinese-CLIP"
DATA_ROOT = ROOT / "datasets/evaluation/chinese_clip_eval_official"
RESULT_ROOT = ROOT / "results/chinese_clip_original_official"
REFERENCE_REVISION = "31863c707501bf1605d36842f43deb78793dbc5d"
DATA_REVISION = "cc0018605734020987981c1c86c50950d632c94b"

MODELS = {
    "OFA-Sys/chinese-clip-vit-base-patch16": {
        "revision": "36e679e65c2a2fead755ae21162091293ad37834",
        "arch": "ViT-B-16",
        "checkpoint": "clip_cn_vit-b-16.pt",
    },
    "OFA-Sys/chinese-clip-vit-large-patch14": {
        "revision": "660941af70c6ff89ce658a1735404c0f3e536c38",
        "arch": "ViT-L-14",
        "checkpoint": "clip_cn_vit-l-14.pt",
    },
    "OFA-Sys/chinese-clip-vit-large-patch14-336px": {
        "revision": "567e85f859213abde0cd45e95d87f4421ccbc14c",
        "arch": "ViT-L-14-336",
        "checkpoint": "clip_cn_vit-l-14-336.pt",
    },
    "OFA-Sys/chinese-clip-vit-huge-patch14": {
        "revision": "503e16b560aff94c1922f13a86a7693d36957a4f",
        "arch": "ViT-H-14",
        "checkpoint": "clip_cn_vit-h-14.pt",
    },
}

DATASETS = {
    "fer-2013": 3589,
    "fgvc-aircraft-2013b-variants102": 3333,
    "kitti-distance": 711,
    "mnist": 10000,
    "patch-camelyon": 32768,
    "voc-2007-classification": 4952,
}

OFFICIAL_TOP1 = {
    "OFA-Sys/chinese-clip-vit-base-patch16": {
        "fer-2013": 47.2, "fgvc-aircraft-2013b-variants102": 12.8,
        "kitti-distance": 33.5, "mnist": 67.6, "patch-camelyon": 54.0,
        "voc-2007-classification": 83.3,
    },
    "OFA-Sys/chinese-clip-vit-large-patch14": {
        "fer-2013": 54.6, "fgvc-aircraft-2013b-variants102": 16.0,
        "kitti-distance": 49.9, "mnist": 69.8, "patch-camelyon": 63.5,
        "voc-2007-classification": 84.5,
    },
    "OFA-Sys/chinese-clip-vit-large-patch14-336px": {
        "fer-2013": 55.1, "fgvc-aircraft-2013b-variants102": 17.1,
        "kitti-distance": 49.8, "mnist": 65.0, "patch-camelyon": 62.9,
        "voc-2007-classification": 84.9,
    },
    "OFA-Sys/chinese-clip-vit-huge-patch14": {
        "fer-2013": 49.2, "fgvc-aircraft-2013b-variants102": 26.2,
        "kitti-distance": 39.1, "mnist": 79.4, "patch-camelyon": 52.4,
        "voc-2007-classification": 84.9,
    },
}


def now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def preflight(model_id: str, dataset: str) -> tuple[Path, dict]:
    spec = MODELS[model_id]
    model_root = ROOT.joinpath("models", *model_id.split("/"))
    receipt_path = model_root / "download_receipt.json"
    checkpoint = model_root / spec["checkpoint"]
    if not receipt_path.is_file() or not checkpoint.is_file():
        raise RuntimeError(f"official checkpoint/receipt missing: {checkpoint}")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if receipt.get("revision") != spec["revision"]:
        raise RuntimeError(f"checkpoint revision mismatch: {receipt.get('revision')} != {spec['revision']}")
    actual_reference = _revision(str(REFERENCE),status=False).strip()
    if actual_reference != REFERENCE_REVISION:
        raise RuntimeError(f"official evaluator revision mismatch: {actual_reference}")
    dataset_root = DATA_ROOT / dataset
    manifest_path = dataset_root / "official_test_manifest.json"
    if not manifest_path.is_file():
        raise RuntimeError(f"official dataset manifest missing: {dataset}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("test_file_count") != DATASETS[dataset]:
        raise RuntimeError(f"official test count mismatch: {dataset}")
    paths = sorted(path for path in (dataset_root / "test").glob("*/*") if path.is_file())
    if len(paths) != DATASETS[dataset] or (dataset_root / "train").exists():
        raise RuntimeError(f"official test filesystem mismatch: {dataset}")
    template = "aircraft_templates" if dataset == "fgvc-aircraft-2013b-variants102" else "openai_templates"
    checkpoint_hash = sha256(checkpoint)
    print(
        f"PREFLIGHT MAP model={model_id}@{spec['revision']} checkpoint={checkpoint.name} "
        f"sha256={checkpoint_hash} backend=official-cn_clip@{REFERENCE_REVISION} "
        f"dataset={dataset}@{DATA_REVISION}:test protocol={template}/context_length=52/fp32 "
        f"metric=Top-1 reference={OFFICIAL_TOP1[model_id][dataset]:.1f}%",
        flush=True,
    )
    return checkpoint, {"paths": paths, "root": dataset_root, "sha256": checkpoint_hash, "template": template}


def evaluate(model_id: str, dataset: str, batch_size: int) -> Path:
    import torch
    from PIL import Image

    checkpoint, resolved = preflight(model_id, dataset)
    sys.path.insert(0, str(REFERENCE))
    from cn_clip.clip import load_from_name, tokenize
    from cn_clip.eval.cvinw_zeroshot_templates import aircraft_templates, openai_templates

    spec = MODELS[model_id]
    model, preprocess = load_from_name(spec["arch"], device="cuda", download_root=str(checkpoint.parent))
    # The official zeroshot_eval.sh uses the evaluator's default `amp`, whose
    # implementation converts the complete model to FP32 and never enters an
    # autocast context during classifier construction or image inference.
    model.float()
    model.eval()
    labels = (resolved["root"] / "label_cn.txt").read_text(encoding="utf-8").splitlines()
    templates = aircraft_templates if resolved["template"] == "aircraft_templates" else openai_templates
    weights = []
    with torch.no_grad():
        for label in labels:
            prompts = [template(label).lower().replace("“", '"').replace("”", '"') for template in templates]
            texts = tokenize(prompts, context_length=52).cuda()
            features = model(None, texts)
            features /= features.norm(dim=-1, keepdim=True)
            feature = features.mean(dim=0)
            weights.append(feature / feature.norm())
        classifier = torch.stack(weights, dim=1)
        correct1 = correct5 = 0
        started = time.perf_counter()
        for offset in range(0, len(resolved["paths"]), batch_size):
            batch_paths = resolved["paths"][offset:offset + batch_size]
            images = torch.stack([preprocess(Image.open(path).convert("RGB")) for path in batch_paths]).cuda()
            targets = torch.tensor([int(path.parent.name) for path in batch_paths], device="cuda")
            image_features = model(images, None)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            predictions = (image_features @ classifier).topk(min(5, len(labels)), dim=1).indices
            correct1 += int((predictions[:, 0] == targets).sum())
            correct5 += int((predictions == targets[:, None]).any(dim=1).sum())
            print(f"PROGRESS {min(offset + batch_size, len(resolved['paths']))}/{len(resolved['paths'])}", flush=True)
    elapsed = time.perf_counter() - started
    top1 = 100.0 * correct1 / len(resolved["paths"])
    official = OFFICIAL_TOP1[model_id][dataset]
    relative_gap = abs(top1 - official) / official * 100.0
    payload = {
        "status": "passed",
        "acceptance_status": "passed" if relative_gap <= 5.0 else "failed_metric_outside_tolerance",
        "model_id": model_id,
        "model_revision": spec["revision"],
        "checkpoint": str(_relative(checkpoint,ROOT)),
        "checkpoint_sha256": resolved["sha256"],
        "official_backend_revision": REFERENCE_REVISION,
        "dataset": dataset,
        "dataset_revision": DATA_REVISION,
        "split": "test",
        "dataset_count": len(resolved["paths"]),
        "prompt_source": resolved["template"],
        "prompt_template_count": len(templates),
        "precision": "fp32 (official evaluator default amp implementation)",
        "metric": "Top-1 accuracy",
        "top1_percent": round(top1, 4),
        "top5_percent": round(100.0 * correct5 / len(resolved["paths"]), 4),
        "top5_reference_status": "missing_same_checkpoint_official_reference",
        "official_top1_percent": official,
        "delta_percentage_points": round(top1 - official, 4),
        "relative_gap_percent": round(relative_gap, 4),
        "within_5_percent_relative": relative_gap <= 5.0,
        "inference_seconds": round(elapsed, 4),
        "finished_at": now(),
    }
    output = RESULT_ROOT.joinpath(*model_id.split("/"), dataset, "summary.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2), flush=True)
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", required=True, choices=MODELS)
    parser.add_argument("--dataset", required=True, choices=DATASETS)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    if args.preflight_only:
        preflight(args.model_id, args.dataset)
        return 0
    evaluate(args.model_id, args.dataset, args.batch_size)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
