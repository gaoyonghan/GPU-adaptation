import io
import csv

import json

import time

from collections import Counter

from datetime import datetime

from pathlib import Path

import numpy as np

import torch

from datasets import ClassLabel, Dataset as HFDataset

from PIL import Image, ImageFile

from torch.utils.data import DataLoader, Dataset

from torchvision.transforms import Compose, InterpolationMode, Normalize, Resize, ToTensor

from tqdm.auto import tqdm

ImageFile.LOAD_TRUNCATED_IMAGES = True

LABELS = ["sunglasses", "glasses", "other", "mask", "hand", "none"]

NUM_LABELS = 6

OFFICIAL_ACCURACY = 18649 / 20532

OFFICIAL_MACRO_F1 = 0.9083

OVERSAMPLE_SEED = 83

SPLIT_SEED = 42

EXPECTED_COUNTS = {"glasses": 809, "hand": 989, "mask": 6148, "none": 8555,
                   "other": 985, "sunglasses": 1677}

SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

def _collect(root, model):
    root = Path(root).expanduser().resolve()
    base = root / "data"
    label2id = {str(label): int(index) for label, index in model.config.label2id.items()}
    if set(label2id) != set(LABELS):
        raise ValueError("Dataset labels do not match model labels")
    samples = []
    for label in sorted(LABELS):
        paths = sorted(path for path in (base / label).iterdir()
                       if path.is_file() and path.suffix.lower() in SUFFIXES)
        if len(paths) != EXPECTED_COUNTS[label]:
            raise ValueError(f"Unexpected {label} count: {len(paths)}")
        for path in paths:
            samples.append({"source_index": len(samples), "path": path.resolve(),
                            "relative_path": path.relative_to(root).as_posix(),
                            "label_name": label, "target": label2id[label]})
    if len(samples) != 19163:
        raise ValueError(f"Expected 19163 source images, found {len(samples)}")
    return samples, root

def _select(samples):
    """Reproduce RandomOverSampler(83), then HF Dataset stratified split(seed=42)."""
    names = np.asarray([sample["label_name"] for sample in samples])
    rng = np.random.RandomState(OVERSAMPLE_SEED)
    resampled = list(range(len(samples)))
    majority = max(Counter(names).values())
    for label in sorted(set(names)):
        indices = np.flatnonzero(names == label)
        resampled.extend(rng.choice(indices, majority - len(indices), replace=True).tolist())
    dataset = HFDataset.from_dict({
        "resampled_index": list(range(len(resampled))), "source_index": resampled,
        "label": [samples[index]["target"] for index in resampled],
    }).cast_column("label", ClassLabel(num_classes=NUM_LABELS, names=LABELS))
    split = dataset.train_test_split(test_size=0.4, shuffle=True,
                                     stratify_by_column="label", seed=SPLIT_SEED)["test"]
    selected = []
    for row in split:
        source = samples[int(row["source_index"])]
        selected.append(dict(source, resampled_index=int(row["resampled_index"])))
    support = np.bincount([sample["target"] for sample in selected], minlength=NUM_LABELS)
    if len(resampled) != 51330 or len(selected) != 20532 or not np.all(support == 3422):
        raise ValueError(f"Unexpected oversampling/split: {len(resampled)}, {support.tolist()}")
    return selected

class Images:
    def __init__(self,records,raw=None):
        self.records,self.raw=records,raw
        self.samples=[{'sample_id':str(i)+':'+str(r.get('relative_path',r.get('source_id',i))),'label':int(r['target'])} for i,r in enumerate(records)]
    def __len__(self):return len(self.samples)
    def __getitem__(self,i):
        r=self.records[i]
        source=io.BytesIO(self.raw[r['source_index']]['img_bytes']) if self.raw is not None else r['path']
        with Image.open(source) as image:return image.convert('RGB')
def load_data(root,model):
    categories=[{'id':int(i),'name':str(n)} for i,n in sorted(model.config.id2label.items(),key=lambda x:int(x[0]))]
    records=_select(_collect(root,model)[0])
    dataset=Images(records)
    return dataset,categories,{'root':str(root),'protocol':'author-seed42-approx','limitation':'Preserves prior project dataset protocol; approximate author splits are not independent official test sets'}

class ValidationProcessor:
    """此任务作者用torchvision验证transform，不用HF默认resize代替。"""
    def __init__(self, processor):
        h=int(processor.size["height"]); w=int(processor.size.get("width",h))
        self.transform=Compose([Resize((h,w), InterpolationMode.BILINEAR), ToTensor(), Normalize(processor.image_mean,processor.image_std)])
    def __call__(self, images, return_tensors="pt"):
        return {"pixel_values": self.transform(images).unsqueeze(0)}
    def __str__(self): return str(self.transform)
