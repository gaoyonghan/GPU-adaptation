#!/usr/bin/env python3
"""Native inference for dima806/traffic_sign_detection.

Forward only: rebuilds the registered tuanai protocol (merge Unknown -> oversample
-> 60/40 stratified split, random_state 83 / 42), loads weights from this model's
weights/ directory, and exports per-sample predictions. Scoring and timing are not
performed here; they belong to the shared classification/evaluate.py.

The dataset construction is part of this model's native workflow: the author's
test split does not exist on disk, it is produced at run time.
"""
import argparse
import collections
import csv
import io
import json
import re
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from torchvision.transforms import Compose, Normalize, Resize, ToTensor
from transformers import ViTForImageClassification, ViTImageProcessor

import timing_support as ts

MODEL_ROOT = Path(__file__).resolve().parents[2]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", required=True, help="Run directory for all outputs")
    parser.add_argument("--weights", default=str(MODEL_ROOT / "weights"))
    parser.add_argument("--zip", required=True, help="traffic_signs_tuanai/archive.zip")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-images", type=int, default=0, help="0 evaluates the whole test split")
    return parser.parse_args()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def build_test_split(archive_path):
    """Reproduce the author's protocol: merge Unknown*, oversample, then split 60/40."""
    from imblearn.over_sampling import RandomOverSampler

    archive = zipfile.ZipFile(archive_path)
    label_names = {}
    for row in csv.DictReader(io.StringIO(archive.read("labels.csv").decode("utf-8", "replace"))):
        label_names[int(row["ClassId"])] = row["Name"]

    def merge(name):
        return re.sub(r"^Unknown.*$", "Unknown", name.strip())

    paths, names = [], []
    for member in archive.namelist():
        parts = member.split("/")
        if len(parts) == 3 and "." in parts[-1]:
            paths.append(member)
            names.append(merge(label_names[int(parts[-2])]))

    categories = sorted(set(names))
    name_to_id = {name: index for index, name in enumerate(categories)}
    labels = np.array([name_to_id[name] for name in names])

    resampler = RandomOverSampler(random_state=83)
    resampled_paths, resampled_labels = resampler.fit_resample(
        pd.DataFrame({"path": paths}), labels)
    train_paths, test_paths, _, test_labels = train_test_split(
        resampled_paths["path"].tolist(), np.array(resampled_labels),
        test_size=0.4, stratify=resampled_labels, random_state=42)

    train_set = set(train_paths)
    overlap = sum(1 for path in test_paths if path in train_set)
    uniques = set(test_paths)
    unique_overlap = sum(1 for path in uniques if path in train_set)
    print(f"[native] source images={len(paths)} classes={len(categories)}")
    print(f"[native] test={len(test_paths)} train={len(train_paths)}")
    print(f"[native] test rows also present in train={overlap} ({100 * overlap / len(test_paths):.1f}%)")
    print(f"[native] unique test images={len(uniques)}, of which in train={unique_overlap} "
          f"({100 * unique_overlap / max(len(uniques), 1):.1f}%)")
    return archive, test_paths, test_labels.tolist(), categories


def main():
    args = parse_args()
    root = Path(args.output_root).resolve()
    root.mkdir(parents=True, exist_ok=True)

    started_at = time.perf_counter()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA is not available on this host")

    archive, test_paths, test_labels, categories = build_test_split(args.zip)

    processor = ViTImageProcessor.from_pretrained(args.weights, local_files_only=True)
    size = processor.size["height"]
    transform = Compose([Resize((size, size)), ToTensor(),
                         Normalize(mean=processor.image_mean, std=processor.image_std)])
    model = ViTForImageClassification.from_pretrained(args.weights, local_files_only=True)
    model.eval().to(device)
    model_load_seconds = time.perf_counter() - started_at
    timer = ts.Timer(device)

    model_labels = {int(key): value for key, value in model.config.id2label.items()}

    def normalize(name):
        return re.sub(r"[^a-z0-9]+", "", name.lower().replace("km/h", ""))

    if not all(normalize(model_labels[i]) == normalize(categories[i]) for i in range(len(categories))):
        raise SystemExit("category order differs from the model config; refusing to guess")

    if args.max_images > 0:
        test_paths = test_paths[: args.max_images]
        test_labels = test_labels[: args.max_images]

    cache = {}

    def load(member):
        if member not in cache:
            cache[member] = Image.open(io.BytesIO(archive.read(member))).convert("RGB")
        return cache[member]

    rows = []
    seen = collections.Counter()
    batch_size = args.batch_size
    with torch.inference_mode():
        for start in range(0, len(test_paths), batch_size):
            chunk = test_paths[start:start + batch_size]
            pixel_values = torch.stack([transform(load(member)) for member in chunk]).to(device)
            timing_handle = timer.start()
            logits = model(pixel_values=pixel_values).logits
            timer.stop(timing_handle, len(chunk), start < timer.warmup_batches * batch_size)
            probabilities = torch.softmax(logits.float(), dim=-1)
            k = min(5, probabilities.shape[1])
            top_probabilities, top_indices = probabilities.topk(k, dim=1, largest=True, sorted=True)
            top_indices = top_indices.cpu().tolist()
            top_probabilities = top_probabilities.cpu().tolist()
            for offset, member in enumerate(chunk):
                sample_id = f"{member}#{seen[member]}"
                seen[member] += 1
                rows.append(dict(
                    sample_id=sample_id,
                    target=int(test_labels[start + offset]),
                    prediction=int(top_indices[offset][0]),
                    topk=[int(v) for v in top_indices[offset]],
                    scores=[float(v) for v in top_probabilities[offset]],
                    topk_scores=[float(v) for v in top_probabilities[offset]],
                ))
            cache.clear()
            print(f"[native] {min(start + batch_size, len(test_paths))}/{len(test_paths)}", flush=True)

    if len(rows) != len(test_paths):
        raise SystemExit(f"export mismatch: rows={len(rows)} test={len(test_paths)}")
    if len({r['sample_id'] for r in rows}) != len(rows):
        raise SystemExit("duplicate sample_id detected")

    with (root / "predictions.jsonl").open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")

    write_json(root / "classification_annotations.json",
               {"samples": [dict(sample_id=r["sample_id"], label=r["target"]) for r in rows]})
    write_json(root / "classification_capture.json",
               dict(state="complete", capture_stage="native_metric_input", extra_forward_calls=0))
    ts.write_reports(root, dict(
        sequence=217, model_id="dima806/traffic_sign_detection", images=len(rows),
        dataset="traffic_signs_tuanai", dataset_format="zip",
        transform="author val_transforms (224px resize/stretch, ImageNet mean/std)",
        device=str(device), precision="fp32", batch_size=batch_size,
        phase=("smoke" if args.max_images > 0 else "full"),
        model_load_seconds=model_load_seconds,
        extra=dict(weights=str(Path(args.weights).resolve()),
                   protocol="tuanai_oversample_then_split", num_classes=len(categories)),
    ), timer, time.perf_counter() - started_at, root / "predictions.jsonl")
    print(f"[native] exported {len(rows)} predictions -> {root / 'predictions.jsonl'}")


if __name__ == "__main__":
    main()
