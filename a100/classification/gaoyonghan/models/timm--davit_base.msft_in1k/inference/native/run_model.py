
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
    _test_record = _start_records(138, __file__, globals())

import csv
import json
import time
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from transformers import AutoImageProcessor, AutoModelForImageClassification


# ===== 1. 模型、数据集和评测参数 =====
MODEL_ID = "timm/davit_base.msft_in1k"
PROJECT_ROOT = _PACKAGE_ROOT
MODEL_DIR = _WEIGHTS_ROOT
BACKEND = "timm"
TIMM_NAME = "davit_base.msft_in1k"
DATASET_DIR = _DATASET_ROOT/'ImageNet-1K'
VAL_DIR = DATASET_DIR / "val"
VAL_LABELS_PATH = DATASET_DIR / "LOC_val_solution.csv"
SYNSET_MAPPING_PATH = DATASET_DIR / "LOC_synset_mapping.txt"
RESULTS_PATH = _OUTPUT_ROOT/_portable_os.environ.get('MODEL_PHASE','full')/'m138/metrics.json'
if __name__ == "__main__":
    RESULTS_PATH = _test_record.output / "metrics.json"

BATCH_SIZE = int(_portable_os.environ.get('MODEL_BATCH_SIZE',64))
NUM_WORKERS = int(_portable_os.environ.get('MODEL_NUM_WORKERS',8))
WARMUP_STEPS = _warmup_steps(10)


class ImageNetValidationDataset(Dataset):
    def __init__(self, transform):
        with SYNSET_MAPPING_PATH.open(encoding="utf-8") as file:
            synsets = [line.split(maxsplit=1)[0] for line in file if line.strip()]
        synset_to_index = {synset: index for index, synset in enumerate(synsets)}

        with VAL_LABELS_PATH.open(newline="", encoding="utf-8") as file:
            rows = csv.DictReader(file)
            labels_by_image_id = {
                row["ImageId"]: synset_to_index[row["PredictionString"].split()[0]]
                for row in rows
            }

        self.image_paths = sorted(VAL_DIR.glob("*.JPEG"))
        self.labels = [labels_by_image_id[path.stem] for path in self.image_paths]
        self.transform = transform

        if len(synsets) != 1000:
            raise ValueError(f"ImageNet-1K 映射应包含 1000 类，实际为 {len(synsets)} 类")
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


def find_weight_file():
    for filename in ("model.safetensors", "pytorch_model.bin"):
        path = MODEL_DIR / filename
        if path.is_file():
            return path
    raise FileNotFoundError(f"未找到模型权重文件: {MODEL_DIR}")


def load_model_and_transform():
    if BACKEND == "transformers":
        processor = AutoImageProcessor.from_pretrained(
            str(MODEL_DIR),
            local_files_only=True,
            use_fast=False,
            trust_remote_code=True,
        )
        model = AutoModelForImageClassification.from_pretrained(
            str(MODEL_DIR),
            local_files_only=True,
            trust_remote_code=True,
        )

        def transform(image):
            return processor(images=image, return_tensors="pt")["pixel_values"][0]

        return model, transform, None

    if BACKEND == "timm":
        try:
            import timm
            from timm.data import create_transform
            try:
                from timm.data import resolve_model_data_config
            except ImportError:
                from timm.data import resolve_data_config as resolve_model_data_config
        except ImportError as exc:
            raise RuntimeError(
                "该模型需要安装 timm 和匹配的 torchvision"
            ) from exc

        model = timm.create_model(
            TIMM_NAME,
            pretrained=False,
            checkpoint_path=str(find_weight_file()),
            num_classes=1000,
        )
        transform = create_transform(
            **resolve_model_data_config(model), is_training=False
        )
        return model, transform, None

    if BACKEND == "mambavision":
        try:
            from timm.data import create_transform
        except ImportError as exc:
            raise RuntimeError(
                "MambaVision 需要安装 timm、torchvision、mamba-ssm 和 einops"
            ) from exc

        model = AutoModelForImageClassification.from_pretrained(
            str(MODEL_DIR),
            local_files_only=True,
            trust_remote_code=True,
        )
        transform = create_transform(
            input_size=(3, 256, 256),
            is_training=False,
            mean=tuple(model.config.mean),
            std=tuple(model.config.std),
            crop_mode=model.config.crop_mode,
            crop_pct=model.config.crop_pct,
        )
        return model, transform, None

    if BACKEND == "onnx":
        try:
            import numpy as np
            import onnxruntime as ort
        except ImportError as exc:
            raise RuntimeError(
                "qualcomm/MobileNet-v2 需要安装 onnxruntime 和 numpy"
            ) from exc

        onnx_files = sorted(MODEL_DIR.rglob("*.onnx"))
        if not onnx_files:
            raise FileNotFoundError(
                "未找到 ONNX 文件。请先下载并解压 Qualcomm MobileNet-v2 的 ONNX float 资产。"
            )
        providers = ort.get_available_providers()
        preferred = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        providers = [name for name in preferred if name in providers] or providers[:1]
        session = ort.InferenceSession(str(onnx_files[0]), providers=providers)
        input_name = session.get_inputs()[0].name
        output_name = session.get_outputs()[0].name

        def transform(image):
            image = image.resize((256, 256), Image.Resampling.BILINEAR)
            image = image.crop((16, 16, 240, 240))
            array = np.asarray(image).astype("float32") / 255.0
            array = (array - np.array((0.485, 0.456, 0.406), dtype="float32"))
            array = array / np.array((0.229, 0.224, 0.225), dtype="float32")
            return torch.from_numpy(array.transpose(2, 0, 1)).contiguous()

        return session, transform, {
            "input_name": input_name,
            "output_name": output_name,
            "providers": providers,
        }

    raise ValueError(f"未知模型后端: {BACKEND}")


def forward(model, pixel_values, device, extra):
    if BACKEND == "onnx":
        output = extra["session"].run(
            [extra["output_name"]],
            {extra["input_name"]: pixel_values.numpy()},
        )[0]
        return torch.from_numpy(output).to(device)
    output = model(pixel_values) if BACKEND == "timm" else model(pixel_values=pixel_values)
    if hasattr(output, "logits"):
        return output.logits
    if isinstance(output, dict):
        return output["logits"]
    return output


# ===== 2. 加载模型和 ImageNet-1K validation =====
device = _select_torch_device(torch, 'cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")
print(f"模型: {MODEL_ID}")
print("数据集: ImageNet-1K validation")

model, transform, extra = load_model_and_transform()
if BACKEND != "onnx":
    model = model.to(device).eval()
else:
    print(f"ONNX providers: {extra['providers']}")

dataset = ImageNetValidationDataset(transform)
if BACKEND != "onnx":
    num_labels = getattr(getattr(model, "config", None), "num_labels", 1000)
    if num_labels != 1000:
        raise ValueError(f"模型分类头应输出 1000 类，实际为 {num_labels} 类")
dataloader = _test_record.loader(DataLoader, 
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=device == "cuda" and BACKEND != "onnx",
    persistent_workers=NUM_WORKERS > 0,
)

parameter_count = (
    sum(parameter.numel() for parameter in model.parameters())
    if BACKEND != "onnx"
    else 0
)
print(f"验证图片数量: {len(dataset)}")
print(f"模型参数量: {parameter_count / 1_000_000:.2f} M")
print(f"batch size: {BATCH_SIZE}")

warmup = dataset[0][0].unsqueeze(0).repeat(BATCH_SIZE, 1, 1, 1)
if BACKEND != "onnx":
    warmup = warmup.to(device)
if device == "cuda" and BACKEND != "onnx":
    torch.backends.cudnn.benchmark = True
with torch.inference_mode():
    for _ in range(WARMUP_STEPS):
        forward(model, warmup, device, extra)
if device == "cuda" and BACKEND != "onnx":
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
del warmup

# ===== 3. 评测 Top-1、Top-5 和推理时间 =====
correct_top1 = 0
correct_top5 = 0
gpu_events = []
cpu_forward_seconds = 0.0
wall_start = time.perf_counter()

with torch.inference_mode():
    for batch_index, (pixel_values, labels) in enumerate(dataloader, start=1):
        _test_record.before_batch(locals())
        if BACKEND != "onnx":
            pixel_values = pixel_values.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        if device == "cuda" and BACKEND != "onnx":
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
            logits = forward(model, pixel_values, device, extra)
            end_event.record()
            gpu_events.append((start_event, end_event))
        else:
            forward_start = time.perf_counter()
            logits = forward(model, pixel_values, device, extra)
            cpu_forward_seconds += time.perf_counter() - forward_start

        top5_indices = logits.topk(k=5, dim=1).indices
        correct_top1 += top5_indices[:, :1].eq(labels.unsqueeze(1)).sum().item()
        correct_top5 += top5_indices.eq(labels.unsqueeze(1)).any(dim=1).sum().item()

        if batch_index % 100 == 0:
            print(f"已处理: {min(batch_index * BATCH_SIZE, len(dataset))}/{len(dataset)}")
        _test_record.capture(locals())

if device == "cuda" and BACKEND != "onnx":
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
image_count = len(dataset)
top1_accuracy = correct_top1 / image_count * 100
top5_accuracy = correct_top5 / image_count * 100
average_forward_ms = forward_seconds / image_count * 1000
model_fps = image_count / forward_seconds
end_to_end_ms = wall_seconds / image_count * 1000
end_to_end_fps = image_count / wall_seconds

print("\n" + "=" * 68)
print(f"{MODEL_ID} - ImageNet-1K validation 评测结果")
print("=" * 68)
print(f"Top-1 accuracy: {top1_accuracy:.2f}%")
print(f"Top-5 accuracy: {top5_accuracy:.2f}%")
print(f"纯模型平均推理时间: {average_forward_ms:.3f} ms/image")
print(f"纯模型吞吐量: {model_fps:.2f} images/s")
print(f"端到端平均时间: {end_to_end_ms:.3f} ms/image")
print(f"端到端吞吐量: {end_to_end_fps:.2f} images/s")
if peak_allocated is not None:
    print(f"峰值已分配显存: {peak_allocated / 1024**3:.3f} GiB")
    print(f"峰值保留显存: {peak_reserved / 1024**3:.3f} GiB")

result = {
    "recorded_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    "model_id": MODEL_ID,
    "model_path": str(MODEL_DIR),
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
        "num_workers": NUM_WORKERS,
        "warmup_steps": WARMUP_STEPS,
    },
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
if BACKEND == "onnx":
    result["environment"]["onnx_providers"] = extra["providers"]

RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
RESULTS_PATH.write_text(
    json.dumps(result, ensure_ascii=False, indent=2) + "\n",
    encoding="utf-8",
)
print(f"评测结果已更新: {RESULTS_PATH}")


_test_record.finish(globals())
