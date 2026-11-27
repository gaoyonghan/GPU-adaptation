#!/usr/bin/env python3
"""Native inference for jacoballessio/ai-image-detect-distilled.

Forward only: loads weights from this model's weights/ directory, runs the Defactify
parquet shard, and exports per-sample predictions. Scoring and timing are not
performed here; they belong to the shared classification/evaluate.py.
"""
import argparse
import glob
import io
import json
import os
import time
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForImageClassification, AutoImageProcessor

try:
    from transformers import ViTImageProcessor
except Exception:  # pragma: no cover - depends on installed transformers version
    ViTImageProcessor = None

import timing_support as ts

MODEL_ROOT = Path(__file__).resolve().parents[2]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", required=True, help="Run directory for all outputs")
    parser.add_argument("--weights", default=str(MODEL_ROOT / "weights"))
    parser.add_argument("--parquet", required=True, help="Parquet file or quoted glob")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--max-images", type=int, default=0, help="0 runs the whole shard set")
    return parser.parse_args()


def load_processor(weights):
    """Newer transformers reject the legacy image_processor_type field; fall back explicitly."""
    try:
        return AutoImageProcessor.from_pretrained(weights, local_files_only=True)
    except Exception:
        if ViTImageProcessor is None:
            raise
        return ViTImageProcessor.from_pretrained(weights, local_files_only=True)


class ParquetBinary(Dataset):
    def __init__(self, parquet_paths, processor):
        frames, ids = [], []
        for path in parquet_paths:
            frame = pd.read_parquet(path, columns=["Image", "Label_A"])
            stem = os.path.basename(path)
            frames.append(frame)
            ids.extend(f"{stem}#{index}" for index in range(len(frame)))
        self.frame = pd.concat(frames, ignore_index=True)
        self.ids = ids
        self.processor = processor

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, index):
        value = self.frame.iloc[index]["Image"]
        payload = value["bytes"] if isinstance(value, dict) else value
        image = Image.open(io.BytesIO(payload)).convert("RGB")
        pixel_values = self.processor(images=image, return_tensors="pt")["pixel_values"][0]
        return pixel_values, int(self.frame.iloc[index]["Label_A"]), self.ids[index]


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def main():
    args = parse_args()
    root = Path(args.output_root).resolve()
    root.mkdir(parents=True, exist_ok=True)

    parquet_paths = sorted(glob.glob(args.parquet))
    if not parquet_paths:
        raise SystemExit(f"no parquet matched: {args.parquet}")

    started_at = time.perf_counter()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA is not available on this host")

    processor = load_processor(args.weights)
    model = AutoModelForImageClassification.from_pretrained(args.weights, local_files_only=True)
    model.eval().to(device)
    model_load_seconds = time.perf_counter() - started_at
    timer = ts.Timer(device)

    id2label = {int(key): str(value) for key, value in model.config.id2label.items()}
    name_to_index = {value.lower().strip(): key for key, value in id2label.items()}
    if "real" not in name_to_index or "fake" not in name_to_index:
        raise SystemExit(f"config.id2label has no real/fake pair: {id2label}")
    real_index, fake_index = name_to_index["real"], name_to_index["fake"]

    def map_label(dataset_label):
        # Dataset Label_A: 0 = real, 1 = AI-generated. Model indices come from its own config.
        return real_index if dataset_label == 0 else fake_index

    dataset = ParquetBinary(parquet_paths, processor)
    if args.max_images > 0 and args.max_images < len(dataset):
        dataset.frame = dataset.frame.iloc[: args.max_images].reset_index(drop=True)
        dataset.ids = dataset.ids[: args.max_images]

    def collate(batch):
        pixel_values = torch.stack([row[0] for row in batch])
        labels = torch.tensor([map_label(row[1]) for row in batch], dtype=torch.long)
        return pixel_values, labels, [row[2] for row in batch]

    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, pin_memory=(device.type == "cuda"),
                        collate_fn=collate)

    rows = []
    seen = 0
    with torch.inference_mode():
        for index, (pixel_values, labels, sample_ids) in enumerate(loader):
            pixel_values = pixel_values.to(device, non_blocking=True)
            timing_handle = timer.start()
            logits = model(pixel_values=pixel_values).logits
            timer.stop(timing_handle, int(labels.numel()), index < timer.warmup_batches)
            probabilities = torch.softmax(logits.float(), dim=-1)
            k = min(5, probabilities.shape[1])
            top_probabilities, top_indices = probabilities.topk(k, dim=1, largest=True, sorted=True)
            top_indices = top_indices.cpu().tolist()
            top_probabilities = top_probabilities.cpu().tolist()
            for position, (sample_id, target) in enumerate(zip(sample_ids, labels.tolist())):
                rows.append(dict(
                    sample_id=sample_id,
                    target=int(target),
                    prediction=int(top_indices[position][0]),
                    topk=[int(v) for v in top_indices[position]],
                    scores=[float(v) for v in top_probabilities[position]],
                    topk_scores=[float(v) for v in top_probabilities[position]],
                ))
            seen += labels.numel()
            print(f"[native] batch {index + 1}/{len(loader)} samples={seen}", flush=True)

    if len(rows) != len(dataset):
        raise SystemExit(f"export mismatch: rows={len(rows)} dataset={len(dataset)}")
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
        sequence=221, model_id="jacoballessio/ai-image-detect-distilled", images=len(rows),
        dataset="defactify", dataset_format="parquet",
        transform="processor default (224px, ImageNet mean/std)", device=str(device), precision="fp32",
        batch_size=args.batch_size,
        phase=("smoke" if args.max_images > 0 else "full"),
        model_load_seconds=model_load_seconds,
        extra=dict(weights=str(Path(args.weights).resolve()),
                   parquet=[os.path.basename(p) for p in parquet_paths],
                   label_map=dict(real=real_index, fake=fake_index)),
    ), timer, time.perf_counter() - started_at, root / "predictions.jsonl")
    print(f"[native] exported {len(rows)} predictions -> {root / 'predictions.jsonl'}")


if __name__ == "__main__":
    main()
