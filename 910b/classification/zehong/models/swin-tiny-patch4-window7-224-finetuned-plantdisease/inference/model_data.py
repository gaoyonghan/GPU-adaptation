import io
import csv

import json

import re

import time

from datetime import datetime

from pathlib import Path

import numpy as np

import torch

from PIL import Image

from sklearn.model_selection import train_test_split

from torch.utils.data import DataLoader, Dataset

from tqdm.auto import tqdm

OFFICIAL_ACCURACY = 0.9689922480620154

NUM_LABELS = 15

SUFFIXES = {".jpg", ".jpeg", ".png"}

def _normalize(label):
    return re.sub(r"[^a-z0-9]", "", str(label).lower())

def _find_imagefolder(root, model):
    root = Path(root).expanduser().resolve()
    label2id = {str(label): int(index) for label, index in model.config.label2id.items()}
    expected = {_normalize(label) for label in label2id}
    if len(expected) != NUM_LABELS:
        raise ValueError(f"Expected {NUM_LABELS} model labels, got {len(expected)}")
    for base in [root, *sorted(path for path in root.rglob("*") if path.is_dir())]:
        children = [path for path in base.iterdir() if path.is_dir()]
        image_children = [path for path in children if any(
            item.is_file() and item.suffix.lower() in SUFFIXES for item in path.iterdir())]
        if {_normalize(path.name) for path in image_children} == expected:
            return base, image_children, label2id
    raise FileNotFoundError(f"A 15-class PlantVillage imagefolder was not found under {root}")

def _select(root, model):
    base, class_dirs, label2id = _find_imagefolder(root, model)
    normalized_model = {_normalize(label): (label, index) for label, index in label2id.items()}
    source = []
    for class_dir in sorted(class_dirs, key=lambda path: path.name):
        label_name, target = normalized_model[_normalize(class_dir.name)]
        for path in sorted(item for item in class_dir.iterdir()
                           if item.is_file() and item.suffix.lower() in SUFFIXES):
            source.append({"source_index": len(source), "path": path.resolve(),
                           "relative_path": path.relative_to(base).as_posix(),
                           "source_label": class_dir.name, "target": target,
                           "target_label": label_name})
    targets = np.asarray([sample["target"] for sample in source])
    _, indices = train_test_split(np.arange(len(source)), test_size=0.2, shuffle=True,
                                  stratify=targets, random_state=42)
    return base, source, [source[int(index)] for index in indices]

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
    records=_select(root,model)[2]
    dataset=Images(records)
    return dataset,categories,{'root':str(root),'protocol':'author-seed42-approx','limitation':'Preserves prior project dataset protocol; approximate author splits are not independent official test sets'}
