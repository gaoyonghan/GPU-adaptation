"""保留原模型的数据划分和作者验证transform；GT与预测独立。"""
from pathlib import Path
import numpy as np
from PIL import Image, ImageFile
from sklearn.model_selection import train_test_split
from torchvision.transforms import Compose, Resize, ToTensor, Normalize, InterpolationMode
ImageFile.LOAD_TRUNCATED_IMAGES = True

LABELS = ("NORMAL", "PNEUMONIA")

SUFFIXES = {".jpg", ".jpeg", ".png"}

def _find_base(root):
    root = Path(root).expanduser().resolve()
    for base in (root, root / "chest_xray", root / "chest_xray" / "chest_xray"):
        if all((base / split).is_dir() for split in ("train", "test", "val")):
            return base
    raise FileNotFoundError(f"train/test/val directories were not found under {root}")

def _source_samples(base):
    samples = []
    for split in ("train", "test", "val"):
        for target, label in enumerate(LABELS):
            for path in (base / split / label).iterdir():
                if path.is_file() and path.suffix.lower() in SUFFIXES:
                    samples.append({"path": path.resolve(),
                                    "relative_path": path.relative_to(base).as_posix(),
                                    "source_split": split, "target": target})
    return sorted(samples, key=lambda sample: sample["relative_path"])

def _author_seed42_selection(samples):
    labels = np.asarray([sample["target"] for sample in samples])
    counts = np.bincount(labels, minlength=2)
    majority = int(counts.max())
    rng = np.random.RandomState(83)
    resampled = list(range(len(samples)))
    for class_id, count in enumerate(counts):
        indices = np.flatnonzero(labels == class_id)
        resampled.extend(rng.choice(indices, size=majority - int(count), replace=True).tolist())
    resampled = np.asarray(resampled)
    _, positions = train_test_split(np.arange(len(resampled)), test_size=0.2, shuffle=True,
                                    stratify=labels[resampled], random_state=42)
    selected = [dict(samples[int(resampled[position])], resampled_position=int(position))
                for position in positions.tolist()]
    supports = np.bincount([sample["target"] for sample in selected], minlength=2).tolist()
    if len(selected) != 1710 or supports != [855, 855]:
        raise ValueError(f"Expected 1710 balanced occurrences; got {len(selected)}, {supports}")
    return selected

class Images:
    def __init__(self, records):
        self.records = records
        # 胸片过采样可重复同一图片，使用出现序号区分；不得去重改变GT。
        self.samples = [{"sample_id": f"occurrence:{i}:{r['relative_path']}", "label": int(r["target"])} for i,r in enumerate(records)]
    def __len__(self): return len(self.samples)
    def __getitem__(self,i):
        with Image.open(self.records[i]["path"]) as image: return image.convert("RGB")

class ValidationProcessor:
    """此任务作者用torchvision验证transform，不用HF默认resize代替。"""
    def __init__(self, processor):
        h=int(processor.size["height"]); w=int(processor.size.get("width",h))
        self.transform=Compose([Resize((h,w), InterpolationMode.BILINEAR), ToTensor(), Normalize(processor.image_mean,processor.image_std)])
    def __call__(self, images, return_tensors="pt"):
        return {"pixel_values": self.transform(images).unsqueeze(0)}
    def __str__(self): return str(self.transform)

def load_data(root,model):
    records=_author_seed42_selection(_source_samples(_find_base(root)))
    categories=[{"id":int(i),"name":str(n)} for i,n in sorted(model.config.id2label.items(),key=lambda x:int(x[0]))]
    return Images(records),categories,{"id":"paultimothymooney/chest-xray-pneumonia","root":str(root),"split_seed":42,"protocol":"author-seed42-approx","limitation":"作者未公开原划分清单，沿用历史工程近似；不能视为独立官方测试集。"}
