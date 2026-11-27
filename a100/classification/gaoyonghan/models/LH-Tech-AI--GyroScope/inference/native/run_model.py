
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
    _test_record = _start_records(152, __file__, globals())

import json
import time
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from transformers import ResNetForImageClassification


# ===== 1. 模型、数据集和评测参数 =====
MODEL_ID = "LH-Tech-AI/GyroScope"
PROJECT_ROOT = _PACKAGE_ROOT
MODEL_DIR = _WEIGHTS_ROOT
DATASET_DIR = _DATASET_ROOT/'ImageNet-1K'
VAL_DIR = DATASET_DIR / "val"
RESULTS_PATH = _OUTPUT_ROOT/_portable_os.environ.get('MODEL_PHASE','full')/'m152/metrics.json'
if __name__ == "__main__":
    RESULTS_PATH = _test_record.output / "metrics.json"

ANGLES = (0, 90, 180, 270)
ANGLE_NAMES = ("0°", "90° CCW", "180°", "270° CCW")
BATCH_SIZE = int(_portable_os.environ.get('MODEL_BATCH_SIZE',128))
NUM_WORKERS = int(_portable_os.environ.get('MODEL_NUM_WORKERS',4))
WARMUP_STEPS = _warmup_steps(10)


class RotatedImageNetDataset(Dataset):
    def __init__(self, transform):
        self.image_paths = sorted(VAL_DIR.glob("*.JPEG"))
        self.transform = transform

        if len(self.image_paths) != 50000:
            raise ValueError(
                f"ImageNet-1K val 应包含 50000 张图片，实际为 {len(self.image_paths)} 张"
            )

    def __len__(self):
        return len(self.image_paths) * len(ANGLES)

    def __getitem__(self, index):
        image_index, angle_index = divmod(index, len(ANGLES))
        with Image.open(self.image_paths[image_index]) as image:
            image = image.convert("RGB")
            if angle_index == 1:
                image = image.transpose(Image.Transpose.ROTATE_90)
            elif angle_index == 2:
                image = image.transpose(Image.Transpose.ROTATE_180)
            elif angle_index == 3:
                image = image.transpose(Image.Transpose.ROTATE_270)
            pixel_values = self.transform(image)
        return pixel_values, angle_index


def forward(model, pixel_values):
    outputs = model(pixel_values=pixel_values)
    if hasattr(outputs, "logits"):
        return outputs.logits
    if isinstance(outputs, dict):
        return outputs["logits"]
    return outputs


# ===== 2. 加载模型和四方向旋转数据集 =====
device = _select_torch_device(torch, 'cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")
print(f"模型: {MODEL_ID}")
print("数据集: ImageNet-1K validation × 4 rotations")

model = ResNetForImageClassification.from_pretrained(
    str(MODEL_DIR), local_files_only=True
).to(device).eval()
if model.config.num_labels != 4:
    raise ValueError(f"GyroScope 应输出 4 类，实际为 {model.config.num_labels} 类")

transform = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
        ),
    ]
)
dataset = RotatedImageNetDataset(transform)
dataloader = _test_record.loader(DataLoader, 
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=device == "cuda",
    persistent_workers=NUM_WORKERS > 0,
)

parameter_count = sum(parameter.numel() for parameter in model.parameters())
print(f"原始图片数量: {len(dataset) // len(ANGLES)}")
print(f"旋转样本数量: {len(dataset)}")
print(f"模型参数量: {parameter_count / 1_000_000:.2f} M")
print(f"batch size: {BATCH_SIZE}")

warmup = dataset[0][0].unsqueeze(0).repeat(BATCH_SIZE, 1, 1, 1).to(device)
if device == "cuda":
    torch.backends.cudnn.benchmark = True
with torch.inference_mode():
    for _ in range(WARMUP_STEPS):
        forward(model, warmup)
if device == "cuda":
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
del warmup

# ===== 3. 评测四类旋转准确率和推理时间 =====
confusion = torch.zeros((4, 4), dtype=torch.int64)
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
            logits = forward(model, pixel_values)
            end_event.record()
            gpu_events.append((start_event, end_event))
        else:
            forward_start = time.perf_counter()
            logits = forward(model, pixel_values)
            cpu_forward_seconds += time.perf_counter() - forward_start

        predictions = logits.argmax(dim=1)
        bins = torch.bincount(
            labels * 4 + predictions, minlength=16
        ).reshape(4, 4).cpu()
        confusion += bins

        if batch_index % 100 == 0:
            processed = min(batch_index * BATCH_SIZE, len(dataset))
            print(f"已处理: {processed}/{len(dataset)}")
        _test_record.capture(locals())

if device == "cuda":
    torch.cuda.synchronize()
    forward_seconds = sum(
        start.elapsed_time(end) for start, end in gpu_events
    ) / 1000
    peak_allocated = torch.cuda.max_memory_allocated()
    peak_reserved = torch.cuda.max_memory_reserved()
else:
    forward_seconds = cpu_forward_seconds
    peak_allocated = peak_reserved = None

wall_seconds = time.perf_counter() - wall_start
sample_count = len(dataset)
overall_accuracy = confusion.diag().sum().item() / sample_count * 100
model_latency_ms = forward_seconds / sample_count * 1000
model_throughput = sample_count / forward_seconds
end_to_end_latency_ms = wall_seconds / sample_count * 1000
end_to_end_throughput = sample_count / wall_seconds

per_angle = {}
for angle_index, angle_name in enumerate(ANGLE_NAMES):
    class_total = confusion[angle_index].sum().item()
    class_correct = confusion[angle_index, angle_index].item()
    per_angle[str(ANGLES[angle_index])] = {
        "name": angle_name,
        "image_count": class_total,
        "correct": class_correct,
        "accuracy_pct": round(class_correct / class_total * 100, 4),
    }

print("\n" + "=" * 68)
print(f"{MODEL_ID} - ImageNet-1K 四方向旋转评测结果")
print("=" * 68)
print(f"Overall accuracy: {overall_accuracy:.2f}%")
for angle_index, angle_name in enumerate(ANGLE_NAMES):
    class_total = confusion[angle_index].sum().item()
    class_correct = confusion[angle_index, angle_index].item()
    print(
        f"{angle_name}: {class_correct}/{class_total} "
        f"({class_correct / class_total * 100:.2f}%)"
    )
print(f"纯模型平均推理时间: {model_latency_ms:.3f} ms/sample")
print(f"纯模型吞吐量: {model_throughput:.2f} samples/s")
print(f"端到端平均时间: {end_to_end_latency_ms:.3f} ms/sample")
print(f"端到端吞吐量: {end_to_end_throughput:.2f} samples/s")
if peak_allocated is not None:
    print(f"峰值已分配显存: {peak_allocated / 1024**3:.3f} GiB")
    print(f"峰值保留显存: {peak_reserved / 1024**3:.3f} GiB")

result = {
    "recorded_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    "model_id": MODEL_ID,
    "model_path": str(MODEL_DIR),
    "script_path": str(Path(__file__).resolve()),
    "dataset": {
        "name": "ImageNet-1K val × 4 rotations",
        "path": str(DATASET_DIR),
        "original_image_count": len(dataset) // len(ANGLES),
        "rotation_sample_count": sample_count,
        "angles": list(ANGLES),
    },
    "environment": {
        "device": (
            torch.cuda.get_device_name(torch.cuda.current_device())
            if device == "cuda"
            else "CPU"
        ),
        "precision": "FP32",
        "batch_size": BATCH_SIZE,
        "num_workers": NUM_WORKERS,
        "warmup_steps": WARMUP_STEPS,
    },
    "accuracy": {
        "overall_pct": round(overall_accuracy, 4),
        "per_angle": per_angle,
        "confusion_matrix": confusion.tolist(),
    },
    "timing": {
        "model_latency_ms_per_sample": round(model_latency_ms, 4),
        "model_throughput_samples_per_second": round(model_throughput, 4),
        "end_to_end_latency_ms_per_sample": round(end_to_end_latency_ms, 4),
        "end_to_end_throughput_samples_per_second": round(
            end_to_end_throughput, 4
        ),
    },
    "memory": {
        "peak_allocated_gib": (
            round(peak_allocated / 1024**3, 4)
            if peak_allocated is not None
            else None
        ),
        "peak_reserved_gib": (
            round(peak_reserved / 1024**3, 4)
            if peak_reserved is not None
            else None
        ),
    },
}

RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
RESULTS_PATH.write_text(
    json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
)
print(f"评测结果已更新: {RESULTS_PATH}")

_test_record.finish(globals())
