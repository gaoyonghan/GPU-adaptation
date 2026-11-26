"""只负责前向、计时与原始结果保存，不计算准确率。

数据接口：dataset.samples为sample_id/label清单，dataset[i]返回图片。
默认支持HF分类器。其他接口通过forward_fn替换前向，返回[B,C]浮点logits。
"""
import csv
import json
import time
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm


def hf_forward(model, inputs):
    return model(**inputs).logits


class ImageBatch:
    def __init__(self, dataset, processor):
        self.dataset, self.processor = dataset, processor

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        inputs = self.processor(images=self.dataset[index], return_tensors="pt")
        return {key: value.squeeze(0) for key, value in inputs.items()}, index


def run_inference(model, processor, dataset, categories, output_dir, *, batch_size=64,
                  num_workers=8, warmup_steps=3, model_info=None, dataset_info=None,
                  forward_fn=hf_forward):
    output_dir = Path(output_dir)
    device = next(model.parameters()).device
    if device.type != "cuda":
        raise ValueError("当前计时器用于CUDA；不要用CPU运行冒充GPU计时。")
    # 独立GT在前向之前保存；评分包直接读取它，无需再次加载图片。
    annotation = {"categories": categories, "samples": dataset.samples}
    (output_dir / "annotations.json").write_text(json.dumps(annotation, ensure_ascii=False), encoding="utf-8")
    loader = DataLoader(ImageBatch(dataset, processor), batch_size=batch_size,
                        num_workers=num_workers, shuffle=False, drop_last=False,
                        pin_memory=True, persistent_workers=num_workers > 0, multiprocessing_context="spawn" if num_workers else None)
    # 用真实预处理输入预热，尺寸、输入键和dtype无需人工填写。
    warmup = next(iter(DataLoader(ImageBatch(dataset, processor), batch_size=batch_size, num_workers=0)))[0]
    warmup = {key: value.to(device) for key, value in warmup.items()}
    model.eval()
    with torch.inference_mode():
        for _ in range(warmup_steps):
            forward_fn(model, warmup)
    torch.cuda.synchronize(device)
    del warmup
    torch.cuda.reset_peak_memory_stats(device)
    processed, forward_seconds = 0, 0.0
    logits_file = None
    output_shape, output_dtype = None, None
    start_event, end_event = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    wall_start = time.perf_counter()
    try:
        with (output_dir / "predictions.csv").open("w", newline="", encoding="utf-8") as predictions, (output_dir / "batch_timings.csv").open("w", newline="", encoding="utf-8") as timing:
            writer = csv.DictWriter(predictions, fieldnames=["sample_id", "prediction", "topk", "batch_index", "forward_ms_per_image"])
            batches = csv.DictWriter(timing, fieldnames=["batch_index", "sample_count", "forward_ms"])
            writer.writeheader(); batches.writeheader()
            with torch.inference_mode():
                for batch_index, (inputs, indices) in enumerate(tqdm(loader, desc="Inference")):
                    inputs = {key: value.to(device, non_blocking=True) for key, value in inputs.items()}
                    start_event.record()
                    logits = forward_fn(model, inputs)
                    end_event.record(); end_event.synchronize()
                    elapsed_ms = start_event.elapsed_time(end_event)
                    if logits.ndim != 2 or not logits.is_floating_point() or logits.shape[0] != len(indices) or not torch.isfinite(logits).all():
                        raise ValueError("分类前向须返回有限的[B,C] logits，第一维对应图片。")
                    count, classes = logits.shape
                    # 保留原始数值dtype；BF16由FP32无损承载，因为NumPy不原生支持它。
                    array = (logits.float() if logits.dtype == torch.bfloat16 else logits).cpu().numpy()
                    if logits_file is None:
                        output_shape, output_dtype = classes, str(logits.dtype)
                        if {c["id"] for c in categories} != set(range(classes)):
                            raise ValueError("类别表应覆盖logits的全部列编号；先明确模型/GT映射。")
                        logits_file = np.lib.format.open_memmap(output_dir / "logits.npy", mode="w+", dtype=array.dtype, shape=(len(dataset), classes))
                    logits_file[processed:processed + count] = array
                    # Top1沿用argmax；并列分数时Top5与argmax的首项规则可能不同。
                    top1 = logits.argmax(1).cpu().tolist()
                    ranks = logits.topk(min(5, classes), dim=1).indices.cpu().tolist()
                    for index, pred, ranked in zip(indices.tolist(), top1, ranks):
                        if ranked[0] != pred:
                            ranked = [pred]  # 不编造新的并列Top5规则；只保留明确的Top1。
                        writer.writerow({"sample_id": dataset.samples[index]["sample_id"], "prediction": pred,
                                         "topk": json.dumps(ranked), "batch_index": batch_index,
                                         "forward_ms_per_image": elapsed_ms / count})
                    batches.writerow({"batch_index": batch_index, "sample_count": count, "forward_ms": elapsed_ms})
                    forward_seconds += elapsed_ms / 1000
                    processed += count
    finally:
        if logits_file is not None:
            logits_file.flush()
            del logits_file
    torch.cuda.synchronize(device)
    wall_seconds = time.perf_counter() - wall_start
    if processed != len(dataset):
        raise RuntimeError(f"推理未完整完成：{processed}/{len(dataset)}")
    summary = {"model": model_info, "dataset": dataset_info, "sample_count": processed,
               "preprocessor": str(processor),
               "parameter_dtype": str(next(model.parameters()).dtype), "output_dtype": output_dtype,
               "num_classes": output_shape, "batch_size": batch_size, "num_workers": num_workers,
               "warmup_steps": warmup_steps, "pin_memory": True,
               "allow_tf32_matmul": torch.backends.cuda.matmul.allow_tf32,
               "allow_tf32_cudnn": torch.backends.cudnn.allow_tf32,
               "torch_num_threads": torch.get_num_threads(),
               "forward_seconds": forward_seconds, "forward_ms_per_image": forward_seconds * 1000 / processed,
               "forward_images_per_second": processed / forward_seconds,
               "end_to_end_seconds": wall_seconds, "end_to_end_images_per_second": processed / wall_seconds,
               "peak_cuda_memory_allocated_bytes": torch.cuda.max_memory_allocated(device),
               "peak_cuda_memory_reserved_bytes": torch.cuda.max_memory_reserved(device)}
    (output_dir / "inference_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary
