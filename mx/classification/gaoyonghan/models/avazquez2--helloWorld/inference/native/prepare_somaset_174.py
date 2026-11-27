#!/usr/bin/env python3
"""Create the frozen, prediction-only SOMASet sample for model 174."""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import zipfile
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=174)
    args = parser.parse_args()

    with zipfile.ZipFile(args.archive) as archive:
        members = sorted(
            name for name in archive.namelist()
            if name.startswith("somaset/") and name.lower().endswith(".jpg")
        )
    expected = 50 * 8 * 250
    if len(members) != expected:
        raise SystemExit(f"Expected {expected} images, found {len(members)}")

    available = set(members)
    rng = random.Random(args.seed)
    rows = []
    for prototype in range(1, 51):
        prototype_id = f"{prototype:02d}"
        poses = list(range(1, 251))
        rng.shuffle(poses)
        # One image from every garment, plus four additional deterministic views.
        choices = [(garment, poses[i]) for i, garment in enumerate(range(1, 9))]
        choices.extend((garment, poses[8 + i]) for i, garment in enumerate(range(1, 5)))
        for garment, pose in choices:
            member = f"somaset/{prototype_id}/{garment:02d}/{pose:04d}.jpg"
            if member not in available:
                raise SystemExit(f"Missing archive member: {member}")
            rows.append({
                "id": f"somaset-{prototype_id}-{garment:02d}-{pose:04d}",
                "member": member,
                "prototype": prototype_id,
                "garment": f"{garment:02d}",
                "pose": f"{pose:04d}",
                "target": None,
            })

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    metadata = {
        "dataset": "SOMASet",
        "source": "https://www.kaggle.com/datasets/vicolab/somaset",
        "archive": str(args.archive),
        "archive_sha256": sha256(args.archive),
        "sample_seed": args.seed,
        "images": len(rows),
        "sampling": "12 images per prototype: one from each garment 01-08 and four extra views",
        "target_labels": None,
        "status": "prediction_only",
        "reason": "SOMASet documents mixed somatotype prototypes but no auditable per-prototype or per-image three-class labels.",
        "mapping_reference": "https://arxiv.org/html/1701.03153",
    }
    metadata_path = args.output.with_suffix(".metadata.json")
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
