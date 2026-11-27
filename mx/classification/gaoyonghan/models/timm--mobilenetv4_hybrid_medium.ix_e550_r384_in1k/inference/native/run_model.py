#!/usr/bin/env python3
"""Native inference for timm/mobilenetv4_hybrid_medium.ix_e550_r384_in1k.

Forward only: loads weights from this model's weights/ directory, runs the full
ImageNet-1k validation split, and exports per-sample predictions. Scoring and
timing are not performed here; they belong to the shared classification/evaluate.py.

Two dataset layouts are accepted under --dataset-path, detected automatically:
  parquet : <root>/data/validation-*.parquet            (self-built shards)
  folder  : <root>/val/*.JPEG + LOC_val_solution.csv + LOC_synset_mapping.txt
            (the shared dataset/ImageNet-1K layout on the MX cluster)
"""
import argparse
import csv
import json
import os
import time
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader
from transformers import AutoImageProcessor, AutoModelForImageClassification

import timing_support as ts

MODEL_ROOT = Path(__file__).resolve().parents[2]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", required=True, help="Run directory for all outputs")
    parser.add_argument("--weights", default=str(MODEL_ROOT / "weights"))
    parser.add_argument("--dataset-path", required=True, help="ImageNet-1k root under shared dataset/")
    parser.add_argument("--split", default="validation")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--max-images", type=int, default=0, help="0 runs the whole split")
    return parser.parse_args()


class ImageNetValidation(torch.utils.data.Dataset):
    """Yields (image, label, sample_id) from either parquet shards or the ILSVRC folder."""

    def __init__(self, root, split, limit):
        self.root = Path(root)
        self.rows = []
        self.hf = None
        data_dir = self.root / "data"
        shards = sorted(data_dir.glob(f"{split}-*.parquet")) if data_dir.is_dir() else []
        if shards:
            from datasets import load_dataset

            self.hf = load_dataset("parquet", data_files={split: [str(p) for p in shards]},
                                   split=split, cache_dir=os.environ.get("HF_DATASETS_CACHE") or None)
            self.kind = "parquet"
            total = len(self.hf)
        else:
            solution = self.root / "LOC_val_solution.csv"
            mapping = self.root / "LOC_synset_mapping.txt"
            val_dir = self.root / "val"
            missing = [str(p) for p in (solution, mapping) if not p.is_file()]
            if missing or not val_dir.is_dir():
                raise SystemExit(
                    f"unrecognised ImageNet layout under {self.root}: "
                    f"expected data/{split}-*.parquet, or LOC_val_solution.csv + "
                    f"LOC_synset_mapping.txt + val/ (missing: {missing or [str(val_dir)]})")
            synsets = [line.split(" ", 1)[0] for line in mapping.read_text(encoding="utf-8").splitlines() if line.strip()]
            index = {name: position for position, name in enumerate(synsets)}
            with solution.open(newline="", encoding="utf-8") as stream:
                for entry in csv.DictReader(stream):
                    synset = entry["PredictionString"].split()[0]
                    self.rows.append((entry["ImageId"], index[synset]))
            self.kind = "folder"
            total = len(self.rows)
        self.count = total if limit <= 0 else min(limit, total)

    def __len__(self):
        return self.count

    def __getitem__(self, position):
        if self.hf is not None:
            item = self.hf[position]
            return item["image"], int(item.get("label", -1)), f"validation#{position}"
        image_id, label = self.rows[position]
        return Image.open(self.root / "val" / f"{image_id}.JPEG"), label, f"validation#{image_id}"


def collate(processor):
    def build(batch):
        images = [row[0].convert("RGB") for row in batch]
        labels = torch.tensor([row[1] for row in batch], dtype=torch.long)
        inputs = processor(images=images, return_tensors="pt")
        return inputs["pixel_values"], labels, [row[2] for row in batch]

    return build


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def main():
    args = parse_args()
    root = Path(args.output_root).resolve()
    root.mkdir(parents=True, exist_ok=True)

    started_at = time.perf_counter()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA is not available on this host")

    dataset = ImageNetValidation(args.dataset_path, args.split, args.max_images)
    print(f"[native] dataset layout={dataset.kind} samples={len(dataset)}", flush=True)

    processor = AutoImageProcessor.from_pretrained(args.weights, local_files_only=True)
    model = AutoModelForImageClassification.from_pretrained(args.weights, local_files_only=True)
    model.eval().to(device)
    model_load_seconds = time.perf_counter() - started_at
    timer = ts.Timer(device)

    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, pin_memory=(device.type == "cuda"),
                        collate_fn=collate(processor))

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
                if target < 0:
                    continue
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
        sequence=220, model_id="timm/mobilenetv4_hybrid_medium.ix_e550_r384_in1k", images=len(rows),
        dataset="ImageNet-1K", dataset_format=dataset.kind,
        transform="processor default (384px, bicubic, center crop)", device=str(device), precision="fp32",
        batch_size=args.batch_size,
        phase=("smoke" if args.max_images > 0 else "full"),
        model_load_seconds=model_load_seconds,
        extra=dict(split=args.split, weights=str(Path(args.weights).resolve())),
    ), timer, time.perf_counter() - started_at, root / "predictions.jsonl")
    print(f"[native] exported {len(rows)} predictions -> {root / 'predictions.jsonl'}")


if __name__ == "__main__":
    main()
