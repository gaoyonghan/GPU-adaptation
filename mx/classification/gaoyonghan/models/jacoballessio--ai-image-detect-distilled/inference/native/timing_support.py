"""Timing and run-report helpers for this model's native runner.

Timing follows the recorded delivery protocol: forward only, warmup batches
excluded, CUDA events on GPU. The helpers also write the run reports the
delivery layout expects next to predictions.jsonl:

    status.json   run status + provenance fields (state must read "completed")
    timing.json   forward-only latency / throughput / peak memory / wall clock
"""
import hashlib
import json
import time
from pathlib import Path

import torch

WARMUP_BATCHES = 5


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


class Timer:
    """Forward-only stopwatch; warmup batches are timed separately and excluded."""

    def __init__(self, device, warmup_batches=WARMUP_BATCHES):
        self.device = device
        self.warmup_batches = warmup_batches
        self.warmup_seconds = 0.0
        self.forward_seconds = 0.0
        self.timed_images = 0

    def start(self):
        if self.device.type == "cuda":
            begin = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            begin.record()
            return begin, end
        return time.perf_counter(), None

    def stop(self, handle, images, is_warmup):
        begin, end = handle
        if end is not None:
            end.record()
            torch.cuda.synchronize()
            elapsed = begin.elapsed_time(end) / 1000.0
        else:
            elapsed = time.perf_counter() - begin
        if is_warmup:
            self.warmup_seconds += elapsed
        else:
            self.forward_seconds += elapsed
            self.timed_images += images


def _write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def write_reports(root, meta, timer, run_wall_seconds, predictions):
    """Write status.json and timing.json for one completed run."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    predictions = Path(predictions)

    status = {
        "sequence": meta.get("sequence"),
        "model_id": meta.get("model_id"),
        "task": "classification",
        "phase": meta.get("phase", "full"),
        "state": "completed",
        "images": meta.get("images"),
        "dataset": meta.get("dataset"),
        "dataset_format": meta.get("dataset_format"),
        "transform": meta.get("transform"),
        "device": meta.get("device"),
        "precision": meta.get("precision", "fp32"),
        "batch_size": meta.get("batch_size"),
        "torch": torch.__version__,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "predictions": predictions.name,
        "predictions_sha256": sha256_file(predictions),
        "run_wall_seconds": round(run_wall_seconds, 3),
    }
    for key, value in meta.get("extra", {}).items():
        status[key] = value
    _write_json(root / "status.json", status)

    timed = timer.timed_images or 0
    timing = {
        "model_load_seconds_this_segment": round(meta.get("model_load_seconds", 0.0), 3),
        "warmup_batches": timer.warmup_batches,
        "warmup_seconds_this_segment": round(timer.warmup_seconds, 3),
        "timed_images": timed,
        "forward_seconds": round(timer.forward_seconds, 3),
        "forward_amortized_ms_per_image": round(1000.0 * timer.forward_seconds / timed, 4) if timed else None,
        "forward_images_per_second": round(timed / timer.forward_seconds, 2) if timer.forward_seconds else None,
        "forward_method": "CUDA events" if timer.device.type == "cuda" else "perf_counter",
        "peak_allocated_bytes_this_segment": torch.cuda.max_memory_allocated() if torch.cuda.is_available() else None,
        "run_wall_seconds_this_segment": round(run_wall_seconds, 3),
    }
    _write_json(root / "timing.json", timing)
    return status, timing
