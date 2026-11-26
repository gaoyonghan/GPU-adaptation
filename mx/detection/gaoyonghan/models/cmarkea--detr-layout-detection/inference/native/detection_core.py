"""Local data and output helpers for this model; copied from the verified runner."""
from __future__ import annotations
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any
import zipfile
from dataclasses import dataclass
from typing import Callable
from PIL import Image


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


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"损坏的断点文件 {path}:{line_number}") from exc
    ids = [record["image_id"] for record in records]
    if len(ids) != len(set(ids)):
        raise RuntimeError(f"断点文件包含重复 image_id: {path}")
    return records


def ensure_coco_fields(annotation: dict[str, Any]) -> dict[str, Any]:
    annotation.setdefault("info", {})
    annotation.setdefault("licenses", [])
    annotation.setdefault("annotations", [])
    return annotation


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


@dataclass
class PreparedDataset:
    annotation: dict[str, Any]
    read_image: Callable[[dict[str, Any]], Image.Image]
    close: Callable[[], None]
    extras: dict[int, dict[str, Any]] | None = None


def zip_member_index(archive: zipfile.ZipFile) -> tuple[set[str], dict[str, str]]:
    members = set(archive.namelist())
    by_basename: dict[str, str] = {}
    duplicates: set[str] = set()
    for member in members:
        name = Path(member).name
        if name in by_basename:
            duplicates.add(name)
        else:
            by_basename[name] = member
    for name in duplicates:
        by_basename.pop(name, None)
    return members, by_basename


def image_from_zip(archive: zipfile.ZipFile, member: str) -> Image.Image:
    with archive.open(member) as source, Image.open(source) as opened:
        return opened.convert("RGB")


def prepare_coco_single_zip(dataset: dict[str, Any], root: Path) -> PreparedDataset:
    archive = zipfile.ZipFile(root / dataset["archive"])
    annotation = ensure_coco_fields(json.loads(archive.read(dataset["annotation_member"])))
    members, basenames = zip_member_index(archive)

    def read_image(info: dict[str, Any]) -> Image.Image:
        name = info["file_name"]
        member = name if name in members else basenames.get(Path(name).name)
        if member is None:
            raise FileNotFoundError(f"图像不在 ZIP 中: {name}")
        return image_from_zip(archive, member)

    return PreparedDataset(annotation, read_image, archive.close)
