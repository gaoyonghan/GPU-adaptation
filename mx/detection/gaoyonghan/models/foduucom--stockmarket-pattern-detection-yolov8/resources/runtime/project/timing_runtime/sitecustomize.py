"""Record model-call inference time without changing the archived evaluator.

Python imports this module at startup through the launcher-managed PYTHONPATH.
Each process writes a separate JSON record, including its measurement method.
"""

from __future__ import annotations

import atexit
import builtins
import io
import json
import os
import runpy
import sys
import threading
import time
from pathlib import Path


_yolov10_source = os.environ.get("METAX_YOLOV10_SOURCE")
if _yolov10_source:
    source = Path(_yolov10_source).resolve(strict=True)
    if not (source / "ultralytics/__init__.py").is_file():
        raise RuntimeError(f"YOLOv10 source is incomplete: {source}")
    sys.path.insert(0, str(source))


# The MetaX compatibility probe must never replace the vendor's PyTorch stack.
# A model's archived runner may try `python -m pip` despite the top-level
# launcher intentionally skipping environment installation.
if os.environ.get("BENCH_PROBE_NO_INSTALL") == "1":
    _run_module_as_main = runpy._run_module_as_main

    def _probe_no_install(module_name, *args, **kwargs):
        if module_name.split(".", 1)[0] in {"pip", "ensurepip", "virtualenv"}:
            raise RuntimeError("Package installation is disabled in the MetaX probe")
        return _run_module_as_main(module_name, *args, **kwargs)

    runpy._run_module_as_main = _probe_no_install


_output = os.environ.get("CODEX_TIMING_OUTPUT_DIR")
_framework = os.environ.get("CODEX_TIMING_FRAMEWORK", "torch")
_device = os.environ.get("CODEX_TIMING_DEVICE", "cuda:0")
_started = time.perf_counter()
_owner_pid = os.getpid()
_local = threading.local()
_stats = {"calls": 0, "forward_seconds": 0.0, "failed_calls": 0}
_calls = []
_errors: list[str] = []
_synchronization = "none"
_protected_name = os.environ.get("CODEX_TIMING_PROTECTED_ROOT")
_protected_root = Path(_protected_name).resolve() if _protected_name else None


def _check_write(path, mode) -> None:
    if _protected_root is None:
        return
    try:
        candidate = Path(path).resolve(strict=False)
    except (TypeError, ValueError):
        return
    if candidate == _protected_root or _protected_root in candidate.parents:
        if isinstance(mode, int):
            writing = bool(mode & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND))
        else:
            writing = any(character in str(mode) for character in "wax+")
        if writing:
            raise PermissionError(f"historical results are read-only during timed reruns: {candidate}")


if _output and _protected_root is not None:
    _builtin_open = builtins.open
    _io_open = io.open
    _os_open = os.open
    _os_mkdir = os.mkdir
    _os_replace = os.replace
    _os_rename = os.rename

    def _safe_builtin_open(file, mode="r", *args, **kwargs):
        _check_write(file, mode)
        return _builtin_open(file, mode, *args, **kwargs)

    def _safe_io_open(file, mode="r", *args, **kwargs):
        _check_write(file, mode)
        return _io_open(file, mode, *args, **kwargs)

    def _safe_os_open(file, flags, *args, **kwargs):
        _check_write(file, flags)
        return _os_open(file, flags, *args, **kwargs)

    def _safe_mkdir(path, *args, **kwargs):
        _check_write(path, "w")
        return _os_mkdir(path, *args, **kwargs)

    def _safe_replace(source, destination, *args, **kwargs):
        _check_write(destination, "w")
        return _os_replace(source, destination, *args, **kwargs)

    def _safe_rename(source, destination, *args, **kwargs):
        _check_write(destination, "w")
        return _os_rename(source, destination, *args, **kwargs)

    builtins.open = _safe_builtin_open
    io.open = _safe_io_open
    os.open = _safe_os_open
    os.mkdir = _safe_mkdir
    os.replace = _safe_replace
    os.rename = _safe_rename


def _record(seconds: float, failed: bool, target: str) -> None:
    _stats["calls"] += 1
    _stats["forward_seconds"] += seconds
    _stats["failed_calls"] += int(failed)
    _calls.append({"call": _stats["calls"], "seconds": seconds,
                   "failed": failed, "target": target})


def _wrap(original, sync):
    def measured(self, *args, **kwargs):
        # Forked DataLoader workers inherit this wrapper, but must never initialize CUDA.
        if os.getpid() != _owner_pid:
            return original(self, *args, **kwargs)
        if _framework == 'torch':
            if type(self).__module__.startswith(('torchvision.transforms', 'timm.data.')):
                return original(self, *args, **kwargs)
            if torch.utils.data.get_worker_info() is not None:
                return original(self, *args, **kwargs)
        depth = getattr(_local, "depth", 0)
        if depth:
            return original(self, *args, **kwargs)
        _local.depth = 1
        failed = False
        try:
            sync()
            begun = time.perf_counter()
            try:
                return original(self, *args, **kwargs)
            except BaseException:
                failed = True
                raise
            finally:
                sync()
                _record(time.perf_counter() - begun, failed,
                        type(self).__module__ + "." + type(self).__qualname__)
        finally:
            _local.depth = 0

    return measured


if _output:
    try:
        if _framework == "torch":
            import torch

            def _torch_sync():
                if _device.startswith("cuda") and torch.cuda.is_available():
                    index = int(_device.split(":", 1)[1]) if ":" in _device else 0
                    torch.cuda.synchronize(index)

            torch.nn.Module.__call__ = _wrap(torch.nn.Module.__call__, _torch_sync)
            _synchronization = "selected_cuda_device" if _device.startswith("cuda") else "cpu_no_cuda_sync"
        elif _framework == "onnx":
            import onnxruntime as ort

            ort.InferenceSession.run = _wrap(ort.InferenceSession.run, lambda: None)
            _synchronization = "onnxruntime_run_blocking"
        elif _framework == "tensorflow":
            import tensorflow as tf

            sync = getattr(tf.experimental, "async_wait", None)
            if sync is None and hasattr(tf.config.experimental, "set_synchronous_execution"):
                tf.config.experimental.set_synchronous_execution(True)
                _synchronization = "tensorflow_synchronous_execution"
            elif sync:
                _synchronization = "tensorflow_async_wait"
            else:
                _synchronization = "host_dispatch_only"
            tf.keras.Model.__call__ = _wrap(tf.keras.Model.__call__, sync or (lambda: None))
        elif _framework == "paddle":
            import paddle

            sync = getattr(paddle.device.cuda, "synchronize", lambda: None)
            paddle.nn.Layer.__call__ = _wrap(paddle.nn.Layer.__call__, sync)
            _synchronization = "paddle_cuda_synchronize"
        else:
            _errors.append(f"unsupported framework: {_framework}")
    except BaseException as exc:
        _errors.append(f"patch failed: {type(exc).__name__}: {exc}")

    def _save() -> None:
        if os.getpid() != _owner_pid:
            return
        try:
            destination = Path(_output)
            destination.mkdir(parents=True, exist_ok=True)
            payload = {
                "schema": "model_call_timing_v2",
                "pid": os.getpid(),
                "framework": _framework,
                "device": _device,
                "synchronization": _synchronization,
                "model_call_seconds": _stats["forward_seconds"],
                "model_calls": _stats["calls"],
                "failed_calls": _stats["failed_calls"],
                "process_seconds_since_sitecustomize": time.perf_counter() - _started,
                "patch_errors": _errors,
                "call_records_file": f"calls-{os.getpid()}.jsonl",
                "scope": "Top-level model calls; warmup and sample coverage require runner-specific reconciliation.",
                "excluded": ["inherited fork workers", "DataLoader workers", "torchvision/timm data transforms"],
            }
            with (destination / f"calls-{os.getpid()}.jsonl").open("w") as stream:
                for row in _calls:
                    stream.write(json.dumps(row) + "\n")
            temporary = destination / f"process-{os.getpid()}.json.tmp"
            temporary.write_text(json.dumps(payload, indent=2) + "\n")
            temporary.replace(destination / f"process-{os.getpid()}.json")
        except BaseException:
            pass

    atexit.register(_save)


# Capture the exact GT seen by native evaluators for the opt-in unified run.
# The hook runs outside model forward timing and leaves native scoring intact.
if os.environ.get("BENCH_UNIFIED_SCORING") == "1" and os.environ.get("BENCH_SCORING_CAPTURE_ROOT"):
    try:
        import importlib.util
        capture_path = Path(__file__).resolve().parents[4] / "inference/output_format/detection.py"
        capture_spec = importlib.util.spec_from_file_location("unified_gt_capture", capture_path)
        capture_module = importlib.util.module_from_spec(capture_spec)
        capture_spec.loader.exec_module(capture_module)
        capture_module.install_scoring_capture()
    except Exception as exc:
        print("Unified GT observation unavailable: " + repr(exc), file=sys.stderr, flush=True)
