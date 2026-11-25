"""ImageNet官方validation Parquet → 图片及独立GT，不改变类别编号。"""
import os
from pathlib import Path
from datasets import load_dataset

class Images:
    def __init__(self, source):
        self.source = source
        self.samples = [{"sample_id": f"validation:{i}", "label": int(y)} for i,y in enumerate(source["label"])]
    def __len__(self): return len(self.samples)
    def __getitem__(self, i): return self.source[i]["image"].convert("RGB")

def load_data(root, model):
    files = sorted((Path(root)/"data").glob("validation-*.parquet"))
    if not files: raise FileNotFoundError(f"没有validation Parquet: {root}")
    source = load_dataset("parquet", data_files={"validation": [str(p) for p in files]}, split="validation", cache_dir=os.environ.get("HF_DATASETS_CACHE"))
    categories = [{"id": int(i), "name": str(n)} for i,n in sorted(model.config.id2label.items(), key=lambda x:int(x[0]))]
    return Images(source), categories, {"id":"ILSVRC/imagenet-1k", "revision":"49e2ee26f3810fb5a7536bbf732a7b07389a47b5", "split":"validation", "root":str(root), "protocol":"official-index-order"}
