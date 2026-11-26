"""保留原模型的数据划分和作者验证transform；GT与预测独立。"""
from pathlib import Path
import numpy as np
from PIL import Image, ImageFile
from sklearn.model_selection import train_test_split
from torchvision.transforms import Compose, Resize, ToTensor, Normalize, InterpolationMode
ImageFile.LOAD_TRUNCATED_IMAGES = True

NUM_LABELS = 15

SPLIT_SEED = 42

LABEL_TRANSLATION = {
    "Blazer": "Blazer", "Celana_Panjang": "Long Pants", "Celana_Pendek": "Shorts",
    "Gaun": "Dresses", "Hoodie": "Hoodie", "Jaket": "Jacket",
    "Jaket_Denim": "Denim Jacket", "Jaket_Olahraga": "Sports Jacket",
    "Jeans": "Jeans", "Kaos": "T-shirt", "Kemeja": "Shirt", "Mantel": "Coat",
    "Polo": "Polo", "Rok": "Skirt", "Sweter": "Sweater",
}

SUFFIXES = {".jpg", ".jpeg", ".png"}

def _collect_samples(root, model):
    root = Path(root).expanduser().resolve()
    base = root / "Clothes_Dataset"
    if not base.is_dir():
        raise FileNotFoundError(f"Clothes_Dataset not found under {root}")
    label2id = {str(label): int(index) for label, index in model.config.label2id.items()}
    if set(label2id) != set(LABEL_TRANSLATION.values()):
        raise ValueError("Dataset labels do not match model labels")
    samples = []
    for source_label, label_name in LABEL_TRANSLATION.items():
        paths = sorted(path for path in (base / source_label).iterdir()
                       if path.is_file() and path.suffix.lower() in SUFFIXES)
        if len(paths) != 500:
            raise ValueError(f"Expected 500 images in {source_label}, found {len(paths)}")
        for path in paths:
            samples.append({"path": path.resolve(),
                            "relative_path": path.relative_to(root).as_posix(),
                            "source_label": source_label, "target": label2id[label_name]})
    return sorted(samples, key=lambda item: item["relative_path"]), root

def _select(samples):
    targets = np.asarray([sample["target"] for sample in samples])
    _, indices = train_test_split(np.arange(len(samples)), test_size=0.4, shuffle=True,
                                  stratify=targets, random_state=SPLIT_SEED)
    selected = [dict(samples[int(index)], source_index=int(index)) for index in indices]
    support = np.bincount([sample["target"] for sample in selected], minlength=NUM_LABELS)
    if len(selected) != 3000 or not np.all(support == 200):
        raise ValueError(f"Unexpected selection: {len(selected)}, {support.tolist()}")
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
    records=_select(_collect_samples(root,model)[0])
    categories=[{"id":int(i),"name":str(n)} for i,n in sorted(model.config.id2label.items(),key=lambda x:int(x[0]))]
    return Images(records),categories,{"id":"ryanbadai/clothes-dataset","root":str(root),"split_seed":42,"protocol":"author-seed42-approx","limitation":"作者未公开原划分清单，沿用历史工程近似；不能视为独立官方测试集。"}
