"""鸟类数据适配：读取原Parquet，保留原seed42划分，返回统一图片/GT接口。

公共推理器要求 dataset.samples = [{sample_id, label}, ...]，
并且 dataset[i] 返回PIL图片。换数据集时只需实现相同接口。
"""
import os
from pathlib import Path
import numpy as np
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset


class BirdImages(Dataset):
    def __init__(self, source, addresses, samples):
        self.source, self.addresses, self.samples = source, addresses, samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        split, row = self.addresses[index]
        return self.source[split][row]["image"].convert("RGB")


def load_birds(dataset_root, model_label2id):
    """GT从原始label读取，再按类别名称映射到模型输出编号，不读取预测。

    路径下可直接有Parquet，或放在data/下；此鸟类协议需要三个split。
    不固定总图片数、类别数和测试图片数；修改划分应同时修改下面的协议说明。
    """
    root = Path(dataset_root).expanduser().resolve()
    base = root / "data" if (root / "data").is_dir() else root
    files = {split: [str(p) for p in sorted(base.glob(f"{split}-*.parquet"))]
             for split in ("test", "train", "validation")}
    if not all(files.values()):
        raise FileNotFoundError(f"鸟类协议需要test/train/validation Parquet：{base}")
    source = load_dataset("parquet", data_files=files,
                          cache_dir=os.environ.get("HF_DATASETS_CACHE"))
    addresses = [(split, i) for split in files for i in range(len(source[split]))]
    _, selected = train_test_split(np.arange(len(addresses)), test_size=0.2, random_state=42)
    selected_addresses, samples = [], []
    # 每个split用自己的类别名称映射；不假定各split编号相同。
    labels = {split: source[split]["label"] for split in files}
    mappings = {split: {i: int(model_label2id[name]) for i, name in enumerate(source[split].features["label"].names)} for split in files}
    for index in selected:
        split, row = addresses[int(index)]
        selected_addresses.append((split, row))
        samples.append({"sample_id": f"{split}:{row}", "label": mappings[split][int(labels[split][row])]})
    categories = [{"id": int(i), "name": str(name)} for name, i in sorted(model_label2id.items(), key=lambda item: int(item[1]))]
    info = {"id": "yashikota/birds-525-species-image-classification", "root": str(root),
            "protocol": "author-seed42-approx", "source_order": list(files),
            "total_rows": len(addresses), "selected_rows": len(samples),
            "limitation": "作者未公开原80/20划分；此处沿用test/train/validation拼接、seed42抽取20%的工程近似。"}
    return BirdImages(source, selected_addresses, samples), categories, info
