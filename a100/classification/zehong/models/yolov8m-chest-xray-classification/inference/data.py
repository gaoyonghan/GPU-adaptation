"""ImageFolder 式划分读取：确定性排序；用 cv2 解码以复现 Ultralytics 原生预处理。"""
from pathlib import Path

import cv2
import numpy as np
from torch.utils.data import Dataset

EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}


class Sample:
    __slots__ = ('sample_id', 'path', 'label')

    def __init__(self, sample_id, path, label):
        self.sample_id, self.path, self.label = sample_id, path, label


class ClassImageDataset(Dataset):
    """返回 BGR ndarray，交给 ultralytics classify_transforms（CenterCrop+ToTensor+Normalize）。"""

    def __init__(self, root, split, names_by_id, transform, limit=None):
        root = Path(root)
        split_dir = root / split
        if not split_dir.is_dir():
            raise FileNotFoundError(f'数据划分不存在: {split_dir}')
        name_to_id = {name: cid for cid, name in names_by_id.items()}
        rows, self.excluded = [], []
        for class_dir in sorted(p for p in split_dir.iterdir() if p.is_dir()):
            if class_dir.name not in name_to_id:
                images = sum(1 for p in class_dir.iterdir() if p.suffix.lower() in EXTS)
                self.excluded.append({'class': class_dir.name, 'images': images})
                continue
            label = name_to_id[class_dir.name]
            for path in sorted(p for p in class_dir.iterdir() if p.suffix.lower() in EXTS):
                rows.append((f'{split}/{class_dir.name}/{path.name}', path, label))
        if not rows:
            raise ValueError(f'未在 {split_dir} 找到与模型类别匹配的图片')
        rows.sort(key=lambda row: row[0])
        if limit is not None:
            rows = rows[:limit]
        self.root, self.split, self.transform = root, split, transform
        self.samples = [Sample(*row) for row in rows]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        image = cv2.imread(str(sample.path), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f'无法解码图片: {sample.path}')
        return self.transform(image), index
