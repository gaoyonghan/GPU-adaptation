"""共享清单→预处理→设备前向计时→逐图完整NPY；不做相似度比较。"""
import csv
import hashlib
import json
import shutil
import time
from contextlib import nullcontext
from io import BytesIO
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

from .feature_adapter import describe_output, named_inputs
from .protocol import SCHEMA, inside, read_manifest, sha256_file, tensor_fingerprint, write_json

TIMING = {
    "forward": "current-stream device events around forward_fn only; excludes input transfer, output selection, D2H and save",
    "end_to_end": "fresh loader iteration through final NPY/index/timing flush and device sync; includes read/hash/decode, preprocessing/input hash, H2D, forward, D2H, checks, NPY writes/output hash; no fsync",
    "excluded": "model loading, model metadata, warmup, manifest/config/summary writes, worker shutdown and offline comparison",
    "mode": "synchronous_per_batch",
}


def precision_context(device, precision):
    """none保留模型本来的dtype；AMP只在明确要求时开启，不把模型强转FP32。"""
    if precision == "none":
        return nullcontext()
    return torch.autocast(device.type, dtype={"amp-fp16": torch.float16, "amp-bf16": torch.bfloat16}[precision])


def _move_inputs(inputs, device, non_blocking=False):
    # 只搬设备，保留浮点、整数mask等输入原来的dtype。
    if isinstance(inputs, torch.Tensor):
        return inputs.to(device, non_blocking=non_blocking)
    return {key: value.to(device, non_blocking=non_blocking) for key, value in inputs.items()}


def _time_forward(model, inputs, forward_fn, device, precision):
    """参考分类案例：设备事件仅包围真实前向，输出选择/回传在此函数外。"""
    with precision_context(device, precision):
        if device.type in {"cuda", "npu"}:
            # 迁移关键点：使用对应设备的Event和Stream，不能只改model.to(device)。
            api = torch.cuda if device.type == "cuda" else torch.npu
            begin, end = api.Event(enable_timing=True), api.Event(enable_timing=True)
            stream = api.current_stream(device)
            begin.record(stream)
            outputs = forward_fn(model, inputs)
            end.record(stream)
            end.synchronize()
            seconds = begin.elapsed_time(end) / 1000.0
        else:
            # 只供开发验证，start.py不允许生产入口自动转到CPU。
            begin = time.perf_counter()
            outputs = forward_fn(model, inputs)
            seconds = time.perf_counter() - begin
    if seconds <= 0:
        raise RuntimeError("Non-positive forward time; check device events")
    return outputs, seconds


class ManifestImages(Dataset):
    def __init__(self, root, records):
        self.root, self.records = Path(root), records

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        row = self.records[index]
        data = inside(self.root, row["relative_path"]).read_bytes()
        if hashlib.sha256(data).hexdigest() != row["image_sha256"]:
            raise ValueError(f"Image changed: {row['sample_id']}")
        with Image.open(BytesIO(data)) as image:
            image = image.convert("RGB")
        return image, row["sample_id"]


class ImageCollator:
    """在worker内调用模型预处理；返回可由DataLoader固定内存的Tensor/dict。"""
    def __init__(self, prepare_batch):
        self.prepare_batch = prepare_batch

    def __call__(self, rows):
        images, ids = zip(*rows)
        inputs = self.prepare_batch(list(images))
        tensors = named_inputs(inputs)
        if any(v.device.type != "cpu" or v.ndim < 1 or v.shape[0] != len(ids) for v in tensors.values()):
            raise ValueError("prepare_batch must return CPU tensors with image batch first")
        shape = {k: list(v.shape[1:]) for k, v in tensors.items()}
        dtypes = {k: str(v.dtype) for k, v in tensors.items()}
        metadata = [{"sample_id": sid,
                     "input_tensor_sha256": tensor_fingerprint({k: v[i] for k, v in tensors.items()})}
                    for i, sid in enumerate(ids)]
        return inputs, metadata, shape, dtypes


def evaluate_model(model, prepare_batch, forward_fn, select_output, *, bundle_dir, output_dir,
                   model_info, preprocessing_info, output_name, batch_size=8, num_workers=8,
                   warmup_steps=3, precision="none", pin_memory=False, non_blocking=True,
                   multiprocessing_context="spawn", persistent_workers=True, prefetch_factor=2,
                   seed=0):
    """生产使用CUDA/NPU；CPU仅供公共函数开发验证，start.py不提供CPU运行选项。"""
    if batch_size <= 0 or num_workers < 0 or warmup_steps < 0 or prefetch_factor <= 0:
        raise ValueError("Invalid batch/workers/warmup/prefetch")
    if precision not in {"none", "amp-fp16", "amp-bf16"}:
        raise ValueError("Unknown precision")
    if model.training:
        raise ValueError("Call model.eval() first")
    device = next(model.parameters()).device
    if any(p.device != device for p in model.parameters()) or any(b.device != device for b in model.buffers()):
        raise ValueError("Mixed parameter/buffer devices; no implicit CPU offload")
    output_dir, bundle_dir = Path(output_dir), Path(bundle_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for name in ("features", "summary.json", "evaluation_config.json", "input_manifest.jsonl", "feature_index.jsonl", "batch_timings.csv"):
        if (output_dir / name).exists():
            raise FileExistsError(f"Refusing to overwrite {output_dir / name}")
    records = read_manifest(bundle_dir / "input_manifest.jsonl")
    shutil.copyfile(bundle_dir / "input_manifest.jsonl", output_dir / "input_manifest.jsonl")
    (output_dir / "features").mkdir()
    dataset = ManifestImages(bundle_dir, records)
    collator = ImageCollator(prepare_batch)
    worker_options = {}
    if num_workers:
        worker_options = {"multiprocessing_context": multiprocessing_context,
                          "persistent_workers": persistent_workers, "prefetch_factor": prefetch_factor}
    loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers,
                        shuffle=False, drop_last=False, pin_memory=pin_memory, collate_fn=collator,
                        generator=torch.Generator().manual_seed(seed), **worker_options)
    # 记录真实后端开关，不覆盖官方加载/前向的默认策略。
    if device.type == "cuda":
        math = {"matmul_reduced_precision": bool(torch.backends.cuda.matmul.allow_tf32),
                "conv_reduced_precision": bool(torch.backends.cudnn.allow_tf32)}
    elif device.type == "npu":
        math = {"matmul_reduced_precision": bool(torch.npu.matmul.allow_hf32),
                "conv_reduced_precision": bool(torch.npu.conv.allow_hf32)}
    else:
        math = {"matmul_reduced_precision": False, "conv_reduced_precision": False}
    api = torch.cuda if device.type == "cuda" else torch.npu if device.type == "npu" else None
    config = {
        "schema": SCHEMA, "model": model_info, "preprocessing": preprocessing_info,
        "output": {"name": output_name, "selection": "full_tensor_in_original_order"},
        "dataset": {"num_images": len(records), "manifest_sha256": sha256_file(output_dir / "input_manifest.jsonl")},
        "execution": {"device": str(device), "precision": precision,
                      "parameter_dtypes": sorted({str(p.dtype) for p in model.parameters()}),
                      "seed": seed, "batch_size": batch_size, "warmup_steps": warmup_steps,
                      "math": math},
        "loader": {"num_workers": loader.num_workers, "multiprocessing_context": multiprocessing_context if num_workers else None,
                   "pin_memory": pin_memory, "non_blocking": non_blocking,
                   "persistent_workers": loader.persistent_workers, "prefetch_factor": loader.prefetch_factor,
                   "torch_num_threads": torch.get_num_threads()},
        "timing": {**TIMING, "method": "device_events" if api else "cpu_perf_counter_debug"},
    }
    count, invalid, elapsed_batches = 0, 0, []
    try:
        with torch.inference_mode():
            # 预热正式batch尺寸；数据不足时使用全部样本，不虚构额外图片。
            if warmup_steps:
                warm, _, _, _ = collator([dataset[i] for i in range(min(batch_size, len(dataset)))])
                warm = _move_inputs(warm, device)
                for _ in range(warmup_steps):
                    with precision_context(device, precision):
                        outputs = forward_fn(model, warm)
                    describe_output(select_output(outputs), min(batch_size, len(dataset)), device)
                if api:
                    api.synchronize(device)
                del warm, outputs
            if api:
                api.reset_peak_memory_stats(device)
            # 文件准备在计时外；迭代器/worker启动与输出flush在端到端计时内。
            with (output_dir / "feature_index.jsonl").open("x", encoding="utf-8") as index_file, \
                 (output_dir / "batch_timings.csv").open("x", newline="", encoding="utf-8") as timing_file:
                writer = csv.DictWriter(timing_file, ["batch_index", "batch_size", "forward_ms"])
                writer.writeheader()
                if api:
                    api.synchronize(device)
                wall_start = time.perf_counter()
                for batch_index, (inputs, metadata, shapes, dtypes) in enumerate(tqdm(loader, desc="Extract features", unit="batch")):
                    if "input" not in config:
                        config["input"] = {"shapes": shapes, "dtypes": dtypes}
                        config["loader"]["observed_input_pinned"] = {k: v.is_pinned() for k, v in named_inputs(inputs).items()}
                    elif config["input"] != {"shapes": shapes, "dtypes": dtypes}:
                        raise ValueError("Dynamic input shape/dtype: use an explicit batching/index adapter")
                    inputs = _move_inputs(inputs, device, non_blocking)
                    if any(v.device != device for v in named_inputs(inputs).values()):
                        raise RuntimeError("Model inputs are not on selected device")
                    outputs, seconds = _time_forward(model, inputs, forward_fn, device, precision)
                    raw = select_output(outputs)
                    description = describe_output(raw, len(metadata), device)
                    if "shape" not in config["output"]:
                        config["output"].update(description)
                    elif any(config["output"][k] != v for k, v in description.items()):
                        raise ValueError("Dynamic output shape/dtype: adapt per-image index before use")
                    # 仅去掉batch维，保留CLS/register及所有原始数值/排列。
                    stored = raw.detach()
                    if stored.dtype == torch.bfloat16:
                        stored = stored.float()
                    arrays = stored.cpu().numpy()
                    elapsed_batches.append(seconds)
                    writer.writerow({"batch_index": batch_index, "batch_size": len(metadata), "forward_ms": seconds * 1000})
                    for array, row in zip(arrays, metadata):
                        finite, zero = bool(np.isfinite(array).all()), bool(np.all(array == 0))
                        invalid += int(not finite or zero)
                        path = output_dir / "features" / f"{row['sample_id']}.npy"
                        with path.open("xb") as handle:
                            np.save(handle, array.astype(array.dtype.newbyteorder("<"), copy=False), allow_pickle=False)
                        record = {**row, "feature_file": path.relative_to(output_dir).as_posix(),
                                  "feature_sha256": sha256_file(path), "finite": finite, "zero_output": zero}
                        index_file.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
                        count += 1
                    del outputs, raw, stored, arrays, inputs
                index_file.flush()
                timing_file.flush()
                if api:
                    api.synchronize(device)
                wall_seconds = time.perf_counter() - wall_start
        if count != len(records):
            raise RuntimeError("Not all images processed")
        forward_seconds = sum(elapsed_batches)
        summary = {"schema": SCHEMA, "status": "invalid_features" if invalid else "complete",
                   "model_id": model_info["id"], "revision": model_info["revision"],
                   "device": str(device), "precision": precision, "num_images": count, "invalid_images": invalid,
                   "performance": {"forward_seconds": forward_seconds,
                       "forward_images_per_second": count / forward_seconds,
                       "forward_mean_ms_per_image": forward_seconds * 1000 / count,
                       "forward_batch_median_ms": float(np.median(elapsed_batches) * 1000),
                       "forward_batch_p95_ms": float(np.percentile(elapsed_batches, 95) * 1000),
                       "end_to_end_seconds": wall_seconds, "end_to_end_images_per_second": count / wall_seconds},
                   "memory": {"peak_allocated_bytes": api.max_memory_allocated(device) if api else None,
                              "peak_reserved_bytes": api.max_memory_reserved(device) if api else None}}
        write_json(output_dir / "evaluation_config.json", config)
        write_json(output_dir / "summary.json", summary)
        return summary
    except Exception as exc:
        write_json(output_dir / "evaluation_config.json", config)
        write_json(output_dir / "summary.json", {"schema": SCHEMA, "status": "failed", "num_images": count, "error": str(exc)})
        raise
