
import os as _portable_os
from common import configure_native, select_torch_device as _select_torch_device, tensorflow_gpus as _tensorflow_gpus, onnx_providers as _onnx_providers, verify_onnx_session as _verify_onnx_session
if __name__ == "__main__":
    _runtime_options = configure_native()
from common import PROJECT_ROOT as _PACKAGE_ROOT, WEIGHTS_ROOT as _WEIGHTS_ROOT, DATASET_ROOT as _DATASET_ROOT, OUTPUT_ROOT as _OUTPUT_ROOT
from timing import warmup_steps as _warmup_steps


# 原脚本直接保存逐样本结果，不新增推理入口或模型前向。
if __name__ == "__main__":
    import sys as _records_sys
    from pathlib import Path as _RecordsPath
    _records_sys.path.insert(0, str(_RecordsPath(__file__).resolve().parents[0]))
    from common import start as _start_records
    _test_record = _start_records(102, __file__, globals())

import csv
import json
import os
import time
from datetime import datetime
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import CenterCrop, Compose, InterpolationMode, Normalize, Resize, ToTensor
from transformers import DeiTForImageClassificationWithTeacher


# ===== 1. 设置目录与评测参数 =====
PROJECT_ROOT = _PACKAGE_ROOT
MODEL_CACHE_DIR = str(_WEIGHTS_ROOT)
DATASET_DIR = _DATASET_ROOT/'ImageNet-1K'
VAL_DIR = DATASET_DIR / "val"
VAL_LABELS_PATH = DATASET_DIR / "LOC_val_solution.csv"
SYNSET_MAPPING_PATH = DATASET_DIR / "LOC_synset_mapping.txt"
RESULTS_PATH = _OUTPUT_ROOT/_portable_os.environ.get('MODEL_PHASE','full')/'m102/metrics.json'
if __name__ == "__main__":
    RESULTS_PATH = _test_record.output / "metrics.json"

MODEL_ID = "facebook/deit-tiny-distilled-patch16-224"
BATCH_SIZE = int(_portable_os.environ.get('MODEL_BATCH_SIZE',64))
NUM_WORKERS = int(_portable_os.environ.get('MODEL_NUM_WORKERS',8))
WARMUP_STEPS = _warmup_steps(10)

os.makedirs(MODEL_CACHE_DIR, exist_ok=True)
os.environ["HF_HOME"] = str(_OUTPUT_ROOT / ".runtime/huggingface")
os.environ["HF_HUB_CACHE"] = str(_OUTPUT_ROOT / ".runtime/huggingface")
os.environ["TRANSFORMERS_CACHE"] = str(_OUTPUT_ROOT / ".runtime/huggingface")


class ImageNetValidationDataset(Dataset):
    def __init__(self, transform):
        with SYNSET_MAPPING_PATH.open(encoding="utf-8") as f:
            synsets = [line.split(maxsplit=1)[0] for line in f if line.strip()]
        synset_to_index = {synset: index for index, synset in enumerate(synsets)}

        with VAL_LABELS_PATH.open(newline="", encoding="utf-8") as f:
            rows = csv.DictReader(f)
            labels_by_image_id = {
                row["ImageId"]: synset_to_index[row["PredictionString"].split()[0]]
                for row in rows
            }

        self.image_paths = sorted(VAL_DIR.glob("*.JPEG"))
        self.labels = [labels_by_image_id[path.stem] for path in self.image_paths]
        self.transform = transform

        if len(synsets) != 1000:
            raise ValueError(f"类别映射应包含 1000 类，实际为 {len(synsets)} 类")
        if len(self.image_paths) != 50000:
            raise ValueError(
                f"ImageNet-1K val 应包含 50000 张图片，实际为 {len(self.image_paths)} 张"
            )

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        with Image.open(self.image_paths[index]) as image:
            pixel_values = self.transform(image.convert("RGB"))
        return pixel_values, self.labels[index]


# ===== 2. 加载蒸馏模型和验证集 =====
device = _select_torch_device(torch, 'cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")
print(f"模型: {MODEL_ID}")

transform = Compose(
    [
        Resize(256, interpolation=InterpolationMode.BICUBIC),
        CenterCrop(224),
        ToTensor(),
        Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ]
)
model = DeiTForImageClassificationWithTeacher.from_pretrained(
    str(_WEIGHTS_ROOT), cache_dir=MODEL_CACHE_DIR
).to(device)
model.eval()

dataset = ImageNetValidationDataset(transform)
if model.config.num_labels != 1000:
    raise ValueError(f"模型分类头应输出 1000 类，实际为 {model.config.num_labels} 类")

dataloader = _test_record.loader(DataLoader, 
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=device == "cuda",
    persistent_workers=True,
)
print(f"验证图片数量: {len(dataset)}")
print(f"batch size: {BATCH_SIZE}")
print("推理输出: 分类头与蒸馏头 logits 的平均值")
print("预处理: 短边缩放至 256，中心裁剪至 224")

# ===== 3. 模型预热 =====
with Image.open(dataset.image_paths[0]) as image:
    warmup_pixel_values = transform(image.convert("RGB")).unsqueeze(0)
    warmup_pixel_values = warmup_pixel_values.repeat(BATCH_SIZE, 1, 1, 1).to(device)

if device == "cuda":
    torch.backends.cudnn.benchmark = True

with torch.inference_mode():
    for _ in range(WARMUP_STEPS):
        model(pixel_values=warmup_pixel_values)
if device == "cuda":
    torch.cuda.synchronize()
del warmup_pixel_values

# ===== 4. 完整验证集评测 =====
correct_top1 = torch.zeros((), dtype=torch.long, device=device)
correct_top5 = torch.zeros((), dtype=torch.long, device=device)
gpu_events = []
cpu_forward_seconds = 0.0

wall_start = time.perf_counter()
with torch.inference_mode():
    for batch_index, (pixel_values, labels) in enumerate(dataloader, start=1):
        _test_record.before_batch(locals())
        pixel_values = pixel_values.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        if device == "cuda":
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
            logits = model(pixel_values=pixel_values).logits
            end_event.record()
            gpu_events.append((start_event, end_event))
        else:
            forward_start = time.perf_counter()
            logits = model(pixel_values=pixel_values).logits
            cpu_forward_seconds += time.perf_counter() - forward_start

        top5_indices = logits.topk(k=5, dim=1).indices
        matches = top5_indices.eq(labels.unsqueeze(1))
        correct_top1 += matches[:, :1].sum()
        correct_top5 += matches.any(dim=1).sum()

        if batch_index % 100 == 0:
            processed = min(batch_index * BATCH_SIZE, len(dataset))
            print(f"已处理: {processed}/{len(dataset)}")
        _test_record.capture(locals())

if device == "cuda":
    torch.cuda.synchronize()
wall_seconds = time.perf_counter() - wall_start

if device == "cuda":
    forward_seconds = sum(
        start.elapsed_time(end) for start, end in gpu_events
    ) / 1000
else:
    forward_seconds = cpu_forward_seconds

# ===== 5. 输出指标 =====
image_count = len(dataset)
top1_accuracy = correct_top1.item() / image_count * 100
top5_accuracy = correct_top5.item() / image_count * 100
average_forward_ms = forward_seconds / image_count * 1000
model_fps = image_count / forward_seconds
end_to_end_ms = wall_seconds / image_count * 1000
end_to_end_fps = image_count / wall_seconds

print("\n" + "=" * 60)
print("ImageNet-1K val 上的 DeiT-Tiny Distilled 评测结果")
print("=" * 60)
print(f"Top-1 accuracy: {top1_accuracy:.2f}%")
print(f"Top-5 accuracy: {top5_accuracy:.2f}%")
print(f"纯模型平均推理时间: {average_forward_ms:.3f} ms/image")
print(f"纯模型吞吐量: {model_fps:.2f} images/s")
print(f"端到端平均时间: {end_to_end_ms:.3f} ms/image")
print(f"端到端吞吐量: {end_to_end_fps:.2f} images/s")

# ===== 6. 保存本次评测结果 =====
result = {
    "recorded_at": datetime.now().astimezone().isoformat(timespec="seconds"),
    "model_id": MODEL_ID,
    "model_path": MODEL_CACHE_DIR,
    "script_path": str(Path(__file__).resolve()),
    "dataset": {
        "name": "ImageNet-1K val",
        "path": str(DATASET_DIR),
        "image_count": image_count,
    },
    "environment": {
        "device": (
            torch.cuda.get_device_name(torch.cuda.current_device())
            if device == "cuda"
            else "CPU"
        ),
        "precision": "FP32",
        "batch_size": BATCH_SIZE,
        "warmup_steps": WARMUP_STEPS,
        "num_workers": NUM_WORKERS,
    },
    "preprocessing": "resize shortest edge to 256, then center crop to 224",
    "inference_output": "mean of classification-head and distillation-head logits",
    "accuracy": {
        "image_count": image_count,
        "top1_pct": round(top1_accuracy, 4),
        "top5_pct": round(top5_accuracy, 4),
    },
    "timing": {
        "model_latency_ms_per_image": round(average_forward_ms, 4),
        "model_throughput_images_per_second": round(model_fps, 4),
        "end_to_end_latency_ms_per_image": round(end_to_end_ms, 4),
        "end_to_end_throughput_images_per_second": round(end_to_end_fps, 4),
    },
}

RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
temporary_results_path = RESULTS_PATH.with_suffix(".json.tmp")
with temporary_results_path.open("w", encoding="utf-8") as f:
    json.dump(result, f, ensure_ascii=False, indent=2)
    f.write("\n")
os.replace(temporary_results_path, RESULTS_PATH)
print(f"评测结果已更新: {RESULTS_PATH}")

_test_record.finish(globals())
