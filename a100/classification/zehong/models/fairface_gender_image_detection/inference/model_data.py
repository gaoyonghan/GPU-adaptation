import csv

import io

import json

import pickle

import pickletools

import time

from datetime import datetime

from pathlib import Path

import numpy as np

import torch

from datasets import Dataset as HFDataset

from PIL import Image

from torch.utils.data import DataLoader, Dataset

from tqdm.auto import tqdm

NUM_LABELS = 2

SOURCE_ROWS = 10954

EXPECTED_ROWS = 10000

OFFICIAL_ACCURACY = 0.9342

OFFICIAL_MACRO_F1 = 0.9338

VAL_SIZE = 64980802

VAL_SHA256 = "84872ed6677728e503cef760ae012a49a7c643144313fce2d03fe660a0ebc437"

FORBIDDEN = {"GLOBAL", "STACK_GLOBAL", "REDUCE", "BUILD", "OBJ", "INST", "NEWOBJ",
             "NEWOBJ_EX", "EXT1", "EXT2", "EXT4", "PERSID", "BINPERSID"}

def _sha256(path):
    import hashlib
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

def _load_records(path):
    path = Path(path)
    if not path.is_file() or path.stat().st_size != VAL_SIZE or _sha256(path) != VAL_SHA256:
        raise ValueError("val.pt does not match the fixed FairFace revision")
    with path.open("rb") as stream:
        bad = [op.name for op, _, _ in pickletools.genops(stream) if op.name in FORBIDDEN]
    if bad:
        raise ValueError(f"Unsafe pickle opcodes: {bad[:5]}")
    with path.open("rb") as stream:
        records = pickle.load(stream)
    if not isinstance(records, list) or len(records) != SOURCE_ROWS:
        raise ValueError("Unexpected FairFace validation records")
    return records

def _selection(records, model):
    labels = {str(label): int(index) for label, index in model.config.label2id.items()}
    if labels != {"Female": 0, "Male": 1}:
        raise ValueError(f"Unexpected model labels: {labels}")
    # This is the exact selection statement in the author's Notebook.
    indices = HFDataset.from_dict({"source_index": list(range(len(records)))}) \
        .shuffle(seed=42).select(range(EXPECTED_ROWS))["source_index"]
    rows = []
    for occurrence, source_index in enumerate(indices):
        record = records[int(source_index)]
        if record["_id"] != f"val-{int(source_index) + 1}" or record["gender"] not in labels:
            raise ValueError(f"Unexpected record at source index {source_index}")
        rows.append({"source_index": int(source_index), "source_id": record["_id"],
                     "target": labels[record["gender"]]})
    supports = np.bincount([row["target"] for row in rows], minlength=NUM_LABELS)
    if supports.tolist() != [4720, 5280]:
        raise ValueError(f"Unexpected author selection supports: {supports.tolist()}")
    return rows

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
    raw=_load_records(Path(root)/'repository/val.pt'); records=_selection(raw,model)
    dataset=Images(records,raw)
    return dataset,categories,{'root':str(root),'protocol':'author-seed42-exact-selection','limitation':'Preserves prior project dataset protocol; approximate author splits are not independent official test sets'}
