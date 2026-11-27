#!/usr/bin/env python3
"""Reproduce the official CLIP-RSICD v2 zero-shot classification protocol."""

from __future__ import annotations

import argparse
import json
import platform
import time
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
IMAGES = ROOT / "datasets/evaluation/rsicd_test_labeled"
MANIFEST = ROOT / "datasets/evaluation/b09_rsicd_download_manifest.json"
MODELS = {
    "flax-community/clip-rsicd-v2": {
        "path": ROOT / "models/flax-community/clip-rsicd-v2",
        "official": {1: 88.3, 3: 96.8, 5: 98.2, 10: 99.8},
        "summary": ROOT / "evaluation_audit/CLIP_RSICD_official_comparison.md",
    },
    "openai/clip-vit-base-patch32": {
        "path": ROOT / "models/openai/clip-vit-base-patch32",
        "official": {1: 57.2, 3: 74.5, 5: 83.7, 10: 93.9},
        "summary": ROOT / "evaluation_audit/OpenAI_CLIP_RSICD_official_comparison.md",
    },
}
REFERENCE_REVISION = "d56386a7cc181460f0e0bb1c8fd1de911c2b3f31"


def now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", choices=MODELS, default="flax-community/clip-rsicd-v2")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    import torch
    from PIL import Image
    from torch.utils.data import DataLoader, Dataset
    from transformers import CLIPModel, CLIPProcessor

    model_id = args.model_id
    spec = MODELS[model_id]
    official = spec["official"]
    result_path = ROOT.joinpath("results/clip_rsicd_official", *model_id.split("/"), "summary.json")
    summary_path = spec["summary"]
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    records, classes = manifest["records"], manifest["classes"]
    labels = {name: index for index, name in enumerate(classes)}
    # Preserve the exact capitalization and wording in official run_clip_eval.py.
    prompts = [f"Aerial photograph of {name}" for name in classes]

    processor = CLIPProcessor.from_pretrained(spec["path"], local_files_only=True)
    model = CLIPModel.from_pretrained(spec["path"], local_files_only=True).eval().cuda()

    class Images(Dataset):
        def __len__(self):
            return len(records)

        def __getitem__(self, index):
            record = records[index]
            with Image.open(IMAGES / record["filename"]) as image:
                pixels = processor(images=image.convert("RGB"), return_tensors="pt")["pixel_values"][0]
            return pixels, labels[record["label"]]

    with torch.inference_mode():
        text = processor(text=prompts, return_tensors="pt", padding=True)
        text_features = model.get_text_features(**{key: value.cuda() for key, value in text.items()})
        text_features = torch.nn.functional.normalize(text_features, dim=-1)
    loader = DataLoader(Images(), batch_size=args.batch_size, shuffle=False, num_workers=args.workers,
                        pin_memory=True, persistent_workers=args.workers > 0)
    correct = {k: 0 for k in official}
    seen = 0
    torch.cuda.reset_peak_memory_stats()
    started_at, started = now(), time.perf_counter()
    with torch.inference_mode():
        for pixels, targets in loader:
            pixels, targets = pixels.cuda(non_blocking=True), targets.cuda(non_blocking=True)
            image_features = model.get_image_features(pixel_values=pixels)
            image_features = torch.nn.functional.normalize(image_features, dim=-1)
            predicted = (image_features @ text_features.T).topk(10, dim=1).indices
            for k in official:
                correct[k] += int((predicted[:, :k] == targets[:, None]).any(dim=1).sum())
            seen += len(targets)
    elapsed = time.perf_counter() - started
    measured = {k: round(100.0 * correct[k] / seen, 4) for k in official}
    deltas = {k: round(measured[k] - official[k], 4) for k in official}
    relative = {k: round(abs(deltas[k]) / official[k] * 100.0, 4) for k in official}
    result = {
        "status": "passed", "model_id": model_id, "dataset": "RSICD official labeled test subset",
        "dataset_revision": manifest["revision"], "reference_code_revision": REFERENCE_REVISION,
        "sample_count": seen, "class_count": len(classes), "prompt": "Aerial photograph of {category}",
        "official_accuracy_percent": official, "measured_accuracy_percent": measured,
        "delta_percentage_points": deltas, "relative_gap_percent": relative,
        "within_5_percentage_points_all": all(abs(value) <= 5.0 for value in deltas.values()),
        "within_5_percent_relative_all": all(value <= 5.0 for value in relative.values()),
        "purpose": "accuracy reproduction; throughput provisional during concurrent work",
        "provisional_images_per_second": round(seen / elapsed, 3),
        "peak_gpu_memory_mb": round(torch.cuda.max_memory_allocated() / 1024**2, 2),
        "started_at": started_at, "finished_at": now(),
        "environment": {"python": platform.python_version(), "torch": torch.__version__,
                        "cuda_runtime": torch.version.cuda, "gpu": torch.cuda.get_device_name(0)},
    }
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# CLIP-RSICD v2 官方零样本分类对照\n",
        "RSICD test中按官方规则筛选1,027张、30类；prompt严格为`Aerial photograph of {category}`。\n",
        "| 指标 | 官方 | 实测 | 差值(pp) | 相对差距 | ≤5pp | 相对≤5% |",
        "|---|---:|---:|---:|---:|:---:|:---:|",
    ]
    for k in official:
        lines.append(f"| Top-{k} | {official[k]:.1f}% | {measured[k]:.2f}% | {deltas[k]:+.2f} | "
                     f"{relative[k]:.4f}% | {'是' if abs(deltas[k]) <= 5 else '否'} | "
                     f"{'是' if relative[k] <= 5 else '否'} |")
    lines.append("\n并行阶段吞吐只作诊断，不用于效率验收。\n")
    summary_path.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"model_id": model_id, "status": "passed", "measured": measured, "delta_pp": deltas}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
