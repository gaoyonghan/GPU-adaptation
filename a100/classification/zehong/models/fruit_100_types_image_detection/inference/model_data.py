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

NUM_LABELS = 100

OFFICIAL_ACCURACY = 15296 / 18000

OFFICIAL_MACRO_F1 = 0.8417

SPLIT_SEED = 42

SUFFIXES = {".jpg", ".jpeg", ".png"}

def _collect(root, model):
    root = Path(root).expanduser().resolve()
    paths = sorted(path for path in (root / "train").glob("*/*")
                   if path.is_file() and path.suffix.lower() in SUFFIXES)
    paths += sorted(path for path in (root / "val").glob("*/*")
                    if path.is_file() and path.suffix.lower() in SUFFIXES)
    label2id = {str(label): int(index) for label, index in model.config.label2id.items()}
    counts = Counter(path.parent.name for path in paths)
    if len(paths) != 45000 or len(counts) != NUM_LABELS or set(counts.values()) != {450}:
        raise ValueError(f"Expected 45,000 train+val images / 450 per class; got {len(paths)}")
    expected_order = [model.config.id2label[index] for index in range(NUM_LABELS)]
    if sorted(counts) != expected_order or set(label2id) != set(counts):
        raise ValueError("Dataset label order does not match model config")
    samples = [{"source_index": index, "path": path.resolve(),
                "relative_path": path.relative_to(root).as_posix(),
                "label_name": path.parent.name, "target": label2id[path.parent.name]}
               for index, path in enumerate(paths)]
    return samples, root

def _select(samples):
    """Author logic with fixed local seed: balanced pool then stratified 60/40 split."""
    labels = [sample["target"] for sample in samples]
    split = HFDataset.from_dict({"source_index": list(range(len(samples))), "label": labels}) \
        .cast_column("label", ClassLabel(num_classes=NUM_LABELS,
                                          names=[str(index) for index in range(NUM_LABELS)])) \
        .train_test_split(test_size=0.4, shuffle=True, stratify_by_column="label", seed=SPLIT_SEED)["test"]
    selected = [samples[int(row["source_index"])] for row in split]
    support = np.bincount([sample["target"] for sample in selected], minlength=NUM_LABELS)
    if len(selected) != 18000 or not np.all(support == 180):
        raise ValueError(f"Unexpected selection: {len(selected)}, {support.tolist()[:5]}...")
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
