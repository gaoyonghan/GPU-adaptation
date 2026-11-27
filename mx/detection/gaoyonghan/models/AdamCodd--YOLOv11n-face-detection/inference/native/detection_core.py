"""Local data and output helpers for this model; copied from the verified runner."""
from __future__ import annotations
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


def append_jsonl(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def runtime_versions() -> dict[str, str | None]:
    import importlib.metadata as metadata

    result: dict[str, str | None] = {"python": sys.version.split()[0]}
    for package in (
        "torch", "torchvision", "transformers", "huggingface-hub", "ultralytics", "yolov5", "pycocotools",
        "Pillow", "numpy", "scipy", "pyarrow", "onnxruntime-gpu", "tensorflow", "rfdetr", "vis4d", "lvis",
    ):
        try:
            result[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            result[package] = None
    return result
