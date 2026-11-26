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

from PIL import Image

from torch.utils.data import DataLoader, Dataset

from torchvision.transforms import Compose, InterpolationMode, Normalize, Resize, ToTensor

from tqdm.auto import tqdm

LABELS = ["curly", "dreadlocks", "kinky", "straight", "wavy"]

NUM_LABELS = 5

SOURCE_COUNTS = {"curly": 514, "dreadlocks": 443, "kinky": 217, "straight": 488, "wavy": 330}

OFFICIAL_ACCURACY = 954 / 1028

OFFICIAL_MACRO_F1 = 0.9281

OVERSAMPLE_SEED = 83

SPLIT_SEED = 42

SUFFIXES = {".jpg", ".jpeg", ".png", ".gif", ".webp"}

def _collect(root, model):
    root = Path(root).expanduser().resolve()
    paths = sorted(path for path in (root / "data").glob("*/*")
                   if path.is_file() and path.suffix.lower() in SUFFIXES)
    label2id = {str(label): int(index) for label, index in model.config.label2id.items()}
    counts = Counter(path.parent.name.lower() for path in paths)
    if len(paths) != 1992 or dict(sorted(counts.items())) != SOURCE_COUNTS:
        raise ValueError(f"Unexpected source counts: {dict(counts)}")
    if label2id != {label: index for index, label in enumerate(LABELS)}:
        raise ValueError(f"Dataset/model label mapping mismatch: {label2id}")
    samples = [{"source_index": index, "path": path.resolve(),
                "relative_path": path.relative_to(root).as_posix(),
                "label_name": path.parent.name.lower(),
                "target": label2id[path.parent.name.lower()]}
               for index, path in enumerate(paths)]
    return samples, root

def _select(samples):
    """Reproduce author oversampling(seed=83), then local fixed stratified seed 42 split."""
    names = np.asarray([sample["label_name"] for sample in samples])
    rng = np.random.RandomState(OVERSAMPLE_SEED)
    resampled = list(range(len(samples)))
    for label in sorted(set(names)):
        indices = np.flatnonzero(names == label)
        resampled.extend(rng.choice(indices, 514 - len(indices), replace=True).tolist())
    split = HFDataset.from_dict({
        "resampled_index": list(range(len(resampled))), "source_index": resampled,
        "label": [samples[index]["target"] for index in resampled],
    }).cast_column("label", ClassLabel(num_classes=NUM_LABELS, names=LABELS)) \
      .train_test_split(test_size=0.4, shuffle=True, stratify_by_column="label", seed=SPLIT_SEED)["test"]
    selected = [dict(samples[int(row["source_index"])], resampled_index=int(row["resampled_index"]))
                for row in split]
    support = np.bincount([sample["target"] for sample in selected], minlength=NUM_LABELS)
    if len(resampled) != 2570 or len(selected) != 1028 or set(support.tolist()) != {205, 206}:
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
