#!/usr/bin/env python3
"""Offline, resumable inference for hustvl/yolos-small."""

from __future__ import annotations

import argparse
import contextlib
import fcntl
import hashlib
import importlib.util
import io
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
import zipfile
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from PIL import Image
from download_official_source import MARKER, verify_bundle

Image.MAX_IMAGE_PIXELS = None
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL_ROOT = Path("__MODEL_ROOT__/weights")
DEFAULT_DATASET_ROOT = Path("__DATASET_ROOT__")
DEFAULT_OUTPUT_ROOT = Path("__MODEL_ROOT__/resources/runtime/assets/workspace/results/object_detection_official_v2")
DEFAULT_SOURCE_ROOT = Path("__MODEL_ROOT__/resources/runtime/assets/workspace/official_sources/object_detection")
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
COCO80_NAMES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
    "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard",
    "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
    "teddy bear", "hair drier", "toothbrush",
]


def load_manifest() -> dict[str, Any]:
    manifest = json.loads((SCRIPT_DIR / "models.json").read_text(encoding="utf-8"))
    for dataset_id, dataset in manifest["datasets"].items():
        dataset.setdefault("id", dataset_id)
    return manifest


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


def source_directory(spec: dict[str, Any], source_root: Path) -> Path | None:
    repository = (spec.get("official_sources") or {}).get("repository") or ""
    name = {
        "https://github.com/microsoft/table-transformer": "table-transformer",
        "https://github.com/THU-MIG/yolov10": "yolov10",
        "https://github.com/THU-MIG/yoloe": "yoloe",
        "https://github.com/iMoonLab/yolov13": "yolov13",
        "https://github.com/allenai/WildDet3D": "WildDet3D",
    }.get(repository)
    return source_root / name if name else None


def source_revision(spec: dict[str, Any], source_root: Path) -> str | None:
    directory = source_directory(spec, source_root)
    if directory is None:
        return spec.get("official_sources", {}).get("fixed_revision")
    if (directory / MARKER).is_file():
        return verify_bundle(source_root)[0]["commit"]
    if not (directory / ".git").is_dir():
        return None
    result = subprocess.run(["git", "-C", str(directory), "rev-parse", "HEAD"], capture_output=True, text=True)
    return result.stdout.strip() if result.returncode == 0 else None


def source_revision_matches(actual: str | None, expected: str | None) -> bool:
    if expected is None:
        return True
    return actual is not None and (actual == expected or actual.startswith(expected))


def run_fingerprint(spec: dict[str, Any], dtype: str, batch_size: int, versions: dict[str, Any], revision: str | None) -> str:
    payload = {
        "manifest_protocol_fingerprint": spec["protocol_fingerprint"],
        "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "source_helper_sha256": hashlib.sha256((SCRIPT_DIR / "download_official_source.py").read_bytes()).hexdigest(),
        "dtype": dtype,
        "batch_size": batch_size,
        "runtime_versions": versions,
        "official_source_revision": revision,
    }
    canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def resolve_model(manifest: dict[str, Any], selector: str) -> dict[str, Any]:
    matches = [item for item in manifest["models"] if selector in (item["model_key"], item["model_id"])]
    if len(matches) != 1:
        raise KeyError(f"模型选择器必须唯一匹配 model_key 或 model_id: {selector}")
    return matches[0]


def selected_models(manifest: dict[str, Any], value: str) -> list[dict[str, Any]]:
    if value == "all":
        return list(manifest["models"])
    return [resolve_model(manifest, token.strip()) for token in value.split(",") if token.strip()]


def resolved_weight(spec: dict[str, Any], model_root: Path) -> Path | None:
    weight = spec["model"]["weight_file"]
    return model_root / spec["model_id"] / weight if weight else None


def dataset_paths(dataset: dict[str, Any], dataset_root: Path) -> list[Path]:
    if "paths" in dataset:
        return [dataset_root / item for item in dataset["paths"]]
    kind = dataset["kind"]
    if kind == "coco_archives":
        return [dataset_root / dataset["annotation_archive"], dataset_root / dataset["image_archive"]]
    if kind in {"coco_single_zip", "yolo_zip"}:
        return [dataset_root / dataset["archive"]]
    if kind in {"coco_directory", "yolo_directory"}:
        return [dataset_root / dataset["path"]]
    if kind == "delegated":
        key = "archive" if "archive" in dataset else "path"
        return [dataset_root / dataset[key]]
    if kind in {"pascal_tar", "pascal_tar_structure"}:
        keys = ["annotation_archive", "image_archive", "filelist_archive"]
        if kind == "pascal_tar_structure":
            keys.append("words_archive")
        return [dataset_root / dataset[key] for key in keys]
    if "path" in dataset:
        return [dataset_root / dataset["path"]]
    return []


def preflight(manifest: dict[str, Any], model_root: Path, dataset_root: Path, source_root: Path, paths_only: bool) -> dict[str, Any]:
    dependencies: dict[str, Any] = {"checked": not paths_only, "ok": True, "error": None}
    if not paths_only:
        try:
            import pycocotools  # noqa: F401
            import torch  # noqa: F401
            import transformers  # noqa: F401
            import ultralytics  # noqa: F401
            import yaml  # noqa: F401
        except Exception as exc:  # pragma: no cover - environment-specific
            dependencies.update({"ok": False, "error": repr(exc)})

    rows = []
    for spec in manifest["models"]:
        model_dir = model_root / spec["model_id"]
        weight = resolved_weight(spec, model_root)
        required = dataset_paths(manifest["datasets"][spec["dataset"]["id"]], dataset_root)
        required.extend(model_dir / item for item in spec["model"].get("auxiliary_weight_files", []))
        source = source_directory(spec, source_root)
        if spec["execution_state"] == "ready" and source is not None:
            required.append(source)
        expected_revision = (spec.get("official_sources") or {}).get("fixed_revision") if source is not None else None
        actual_revision = source_revision(spec, source_root) if source is not None and source.exists() else None
        missing = [str(path) for path in required if not path.exists()]
        if not model_dir.is_dir():
            missing.insert(0, str(model_dir))
        if weight is not None and not weight.is_file():
            missing.insert(0, str(weight))
        rows.append({
            "sequence": spec["sequence"],
            "model_key": spec["model_key"],
            "execution_state": spec["execution_state"],
            "environment": spec.get("environment"),
            "runner": spec["protocol"]["runner"],
            "paths_ok": not missing,
            "missing": missing,
            "expected_source_revision": expected_revision,
            "actual_source_revision": actual_revision,
            "source_revision_ok": source_revision_matches(actual_revision, expected_revision) if actual_revision else None,
        })
    dataset_checks: dict[str, Any] = {}
    if dependencies["ok"] and not paths_only:
        ready_dataset_ids = sorted({
            spec["dataset"]["id"] for spec in manifest["models"]
            if spec["execution_state"] == "ready" and spec["protocol"]["runner"] != "document_layout_delegate"
        })
        for dataset_id in ready_dataset_ids:
            dataset = manifest["datasets"][dataset_id]
            if dataset["kind"] not in {"coco_archives", "coco_single_zip", "coco_directory", "yolo_directory", "yolo_zip", "parquet_detection"}:
                dataset_checks[dataset_id] = {"ok": True, "check": "paths_only_for_native_or_materialized_dataset"}
                continue
            try:
                prepared = prepare_dataset(dataset, dataset_root)
                try:
                    actual = {
                        "images": len(prepared.annotation["images"]),
                        "categories": len(prepared.annotation["categories"]),
                        "annotations": len(prepared.annotation["annotations"]),
                    }
                    errors = []
                    if dataset.get("expected_images") is not None and actual["images"] != dataset["expected_images"]:
                        errors.append(f"images={actual['images']} expected={dataset['expected_images']}")
                    if dataset.get("expected_categories") is not None and actual["categories"] != dataset["expected_categories"]:
                        errors.append(f"categories={actual['categories']} expected={dataset['expected_categories']}")
                    dataset_checks[dataset_id] = {"ok": not errors, **actual, "errors": errors}
                finally:
                    prepared.close()
            except Exception as exc:
                dataset_checks[dataset_id] = {"ok": False, "error": repr(exc)}
    ready_missing = sum(row["execution_state"] == "ready" and not row["paths_ok"] for row in rows)
    report = {
        "status": "ready" if dependencies["ok"] and ready_missing == 0 else ("missing_files" if dependencies["ok"] else "environment_error"),
        "dependencies": dependencies,
        "summary": {
            "total": len(rows),
            "manifest_ready": sum(row["execution_state"] == "ready" for row in rows),
            "ready_paths_ok": sum(row["execution_state"] == "ready" and row["paths_ok"] for row in rows),
            "ready_paths_missing": ready_missing,
            "pending_or_blocked": sum(row["execution_state"] != "ready" for row in rows),
        },
        "dataset_checks": dataset_checks,
        "models": rows,
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return report


@dataclass
class PreparedDataset:
    annotation: dict[str, Any]
    read_image: Callable[[dict[str, Any]], Image.Image]
    close: Callable[[], None]
    extras: dict[int, dict[str, Any]] | None = None


def ensure_coco_fields(annotation: dict[str, Any]) -> dict[str, Any]:
    annotation.setdefault("info", {})
    annotation.setdefault("licenses", [])
    annotation.setdefault("annotations", [])
    return annotation


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


def prepare_coco_archives(dataset: dict[str, Any], root: Path) -> PreparedDataset:
    annotation_zip = zipfile.ZipFile(root / dataset["annotation_archive"])
    image_zip = zipfile.ZipFile(root / dataset["image_archive"])
    annotation = ensure_coco_fields(json.loads(annotation_zip.read(dataset["annotation_member"])))
    members, basenames = zip_member_index(image_zip)

    def read_image(info: dict[str, Any]) -> Image.Image:
        name = info["file_name"]
        candidates = (name, f"{dataset.get('image_prefix', '')}{name}")
        member = next((candidate for candidate in candidates if candidate in members), basenames.get(Path(name).name))
        if member is None:
            raise FileNotFoundError(f"图像不在 ZIP 中: {name}")
        return image_from_zip(image_zip, member)

    return PreparedDataset(annotation, read_image, lambda: (annotation_zip.close(), image_zip.close()))


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


def prepare_coco_directory(dataset: dict[str, Any], root: Path) -> PreparedDataset:
    base = root / dataset["path"]
    annotation = ensure_coco_fields(json.loads((base / dataset["annotation"]).read_text(encoding="utf-8")))
    image_root = base / dataset.get("images", ".")

    def read_image(info: dict[str, Any]) -> Image.Image:
        path = image_root / info["file_name"]
        if not path.is_file():
            path = image_root / Path(info["file_name"]).name
        with Image.open(path) as opened:
            return opened.convert("RGB")

    return PreparedDataset(annotation, read_image, lambda: None)


def yaml_names(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value]
    if isinstance(value, dict):
        return [str(value[key]) for key in sorted(value, key=lambda key: int(key))]
    raise RuntimeError("data.yaml 缺少 names")


def yolo_annotation(image_entries: list[tuple[str, bytes, bytes | None]], names: list[str]) -> dict[str, Any]:
    images: list[dict[str, Any]] = []
    annotations: list[dict[str, Any]] = []
    annotation_id = 1
    for image_id, (name, image_bytes, label_bytes) in enumerate(image_entries, 1):
        with Image.open(io.BytesIO(image_bytes)) as opened:
            width, height = opened.size
        images.append({"id": image_id, "file_name": name, "width": width, "height": height})
        if label_bytes:
            for line in label_bytes.decode("utf-8").splitlines():
                fields = line.split()
                if len(fields) < 5:
                    continue
                category, cx, cy, bw, bh = map(float, fields[:5])
                box_width, box_height = bw * width, bh * height
                annotations.append({
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": int(category) + 1,
                    "bbox": [cx * width - box_width / 2, cy * height - box_height / 2, box_width, box_height],
                    "area": box_width * box_height,
                    "iscrowd": 0,
                })
                annotation_id += 1
    return ensure_coco_fields({
        "images": images,
        "annotations": annotations,
        "categories": [{"id": index + 1, "name": name} for index, name in enumerate(names)],
    })


def prepare_yolo_directory(dataset: dict[str, Any], root: Path) -> PreparedDataset:
    import yaml

    base = root / dataset["path"]
    config = yaml.safe_load((base / "data.yaml").read_text(encoding="utf-8"))
    image_root = base / dataset["split_dir"] / "images"
    label_root = base / dataset["split_dir"] / "labels"
    paths = sorted(path for path in image_root.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES)
    entries = []
    for path in paths:
        label = label_root / f"{path.stem}.txt"
        entries.append((path.name, path.read_bytes(), label.read_bytes() if label.is_file() else None))
    annotation = yolo_annotation(entries, yaml_names(config["names"]))
    by_name = {path.name: path for path in paths}

    def read_image(info: dict[str, Any]) -> Image.Image:
        with Image.open(by_name[info["file_name"]]) as opened:
            return opened.convert("RGB")

    return PreparedDataset(annotation, read_image, lambda: None)


def prepare_yolo_zip(dataset: dict[str, Any], root: Path) -> PreparedDataset:
    import yaml

    archive = zipfile.ZipFile(root / dataset["archive"])
    prefix = dataset["root"].rstrip("/")
    prefix_part = f"{prefix}/" if prefix else ""
    if "names" in dataset:
        names = yaml_names(dataset["names"])
    else:
        config = yaml.safe_load(archive.read(f"{prefix_part}data.yaml"))
        names = yaml_names(config["names"])
    image_prefix = f"{prefix_part}{dataset['split_dir']}/images/"
    label_prefix = f"{prefix_part}{dataset['split_dir']}/labels/"
    image_members = sorted(member for member in archive.namelist() if member.startswith(image_prefix) and Path(member).suffix.lower() in IMAGE_SUFFIXES)
    entries = []
    member_by_name = {}
    archive_members = set(archive.namelist())
    for member in image_members:
        label_member = f"{label_prefix}{Path(member).stem}.txt"
        entries.append((Path(member).name, archive.read(member), archive.read(label_member) if label_member in archive_members else None))
        member_by_name[Path(member).name] = member
    annotation = yolo_annotation(entries, names)

    def read_image(info: dict[str, Any]) -> Image.Image:
        return image_from_zip(archive, member_by_name[info["file_name"]])

    return PreparedDataset(annotation, read_image, archive.close)


def parquet_class_names(dataset: dict[str, Any]) -> list[str]:
    if "class_names" in dataset:
        return [str(value) for value in dataset["class_names"]]
    config = json.loads(Path(dataset["class_names_from_model_config"]).read_text(encoding="utf-8"))
    labels = config["id2label"]
    return [str(labels[str(index)]) for index in range(len(labels))]


def prepare_parquet_detection(dataset: dict[str, Any], root: Path) -> PreparedDataset:
    import pyarrow.parquet as parquet

    rows: list[dict[str, Any]] = []
    for relative in dataset["paths"]:
        rows.extend(parquet.read_table(root / relative).to_pylist())
    names = parquet_class_names(dataset)
    images: list[dict[str, Any]] = []
    annotations: list[dict[str, Any]] = []
    image_bytes: dict[int, bytes] = {}
    annotation_id = 1
    seen_ids: set[int] = set()
    for row_number, row in enumerate(rows, 1):
        image_id = int(row.get("image_id", row_number))
        while image_id in seen_ids:
            image_id += 10_000_000
        seen_ids.add(image_id)
        payload = row["image"]
        data = payload["bytes"] if isinstance(payload, dict) else payload
        if not data:
            raise RuntimeError(f"Parquet 第 {row_number} 行没有内嵌图像")
        image_bytes[image_id] = data
        with Image.open(io.BytesIO(data)) as opened:
            width, height = opened.size
        images.append({"id": image_id, "file_name": (payload.get("path") if isinstance(payload, dict) else None) or f"{image_id}.jpg", "width": width, "height": height})
        objects = row.get("objects") or {}
        boxes = objects.get("bbox") or []
        categories = objects.get("category") or []
        areas = objects.get("area") or []
        for offset, (box, category) in enumerate(zip(boxes, categories)):
            values = [float(value) for value in box]
            if dataset["bbox_format"] == "xyxy":
                x1, y1, x2, y2 = values
                coco_box = [x1, y1, max(0.0, x2 - x1), max(0.0, y2 - y1)]
            elif dataset["bbox_format"] == "xywh":
                coco_box = values
            else:
                raise RuntimeError(f"未知 Parquet bbox 格式: {dataset['bbox_format']}")
            area = float(areas[offset]) if offset < len(areas) else coco_box[2] * coco_box[3]
            annotations.append({
                "id": annotation_id,
                "image_id": image_id,
                "category_id": int(category) - int(dataset.get("category_index_base", 0)) + 1,
                "bbox": coco_box,
                "area": area,
                "iscrowd": 0,
            })
            annotation_id += 1
    annotation = ensure_coco_fields({
        "images": images,
        "annotations": annotations,
        "categories": [{"id": index + 1, "name": name} for index, name in enumerate(names)],
    })

    def read_image(info: dict[str, Any]) -> Image.Image:
        with Image.open(io.BytesIO(image_bytes[int(info["id"])])) as opened:
            return opened.convert("RGB")

    return PreparedDataset(annotation, read_image, lambda: None)


def safe_extract_tar(archive_path: Path, target: Path) -> None:
    target_resolved = target.resolve()
    with tarfile.open(archive_path, "r:gz") as archive:
        for member in archive.getmembers():
            destination = (target / member.name).resolve()
            if target_resolved != destination and target_resolved not in destination.parents:
                raise RuntimeError(f"拒绝解压越界 TAR 成员: {member.name}")
        archive.extractall(target)


def materialize_pascal_tar(dataset: dict[str, Any], root: Path, cache_root: Path) -> Path:
    target = cache_root / dataset["id"]
    target.mkdir(parents=True, exist_ok=True)
    marker = target / ".complete.json"
    archive_keys = ["annotation_archive", "image_archive", "filelist_archive"]
    if dataset["kind"] == "pascal_tar_structure":
        archive_keys.append("words_archive")
    snapshot = {
        key: {"path": dataset[key], "size": (root / dataset[key]).stat().st_size}
        for key in archive_keys
    }
    with (target / ".prepare.lock").open("w", encoding="utf-8") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if marker.is_file():
            previous = json.loads(marker.read_text(encoding="utf-8"))
            if previous != snapshot:
                raise RuntimeError(f"PubTables 缓存输入已变化；请改用新的 OUTPUT_ROOT: {target}")
            return target
        destinations = {
            "annotation_archive": target / "annotations",
            "image_archive": target / "images",
            "filelist_archive": target / "filelists",
            "words_archive": target / "words",
        }
        for key in archive_keys:
            destination = destinations[key]
            destination.mkdir(parents=True, exist_ok=True)
            print(f"[dataset:{dataset['id']}] 首次解压 {Path(dataset[key]).name} -> {destination}", flush=True)
            safe_extract_tar(root / dataset[key], destination)
        atomic_json(marker, snapshot)
    return target


def prepare_pascal_tar(dataset: dict[str, Any], root: Path, cache_root: Path) -> PreparedDataset:
    materialized = materialize_pascal_tar(dataset, root, cache_root)
    filelist = materialized / "filelists" / dataset["filelist_member"]
    lines = [line.strip() for line in filelist.read_text(encoding="utf-8").splitlines() if line.strip()]
    class_names = [str(value) for value in dataset["class_names"]]
    category_ids = {name: index + 1 for index, name in enumerate(class_names)}
    images: list[dict[str, Any]] = []
    annotations: list[dict[str, Any]] = []
    extras: dict[int, dict[str, Any]] = {}
    annotation_id = 1
    image_by_id: dict[int, Path] = {}
    for image_id, listed in enumerate(lines, 1):
        stem = Path(listed).stem
        xml_path = materialized / "annotations" / f"{stem}.xml"
        image_path = materialized / "images" / f"{stem}.jpg"
        if not xml_path.is_file() or not image_path.is_file():
            raise FileNotFoundError(f"PubTables 解压后缺文件: {xml_path} / {image_path}")
        tree = ET.parse(xml_path).getroot()
        width = int(float(tree.findtext("size/width", default="0")))
        height = int(float(tree.findtext("size/height", default="0")))
        images.append({"id": image_id, "file_name": image_path.name, "width": width, "height": height})
        image_by_id[image_id] = image_path
        extras[image_id] = {"xml_path": str(xml_path)}
        if dataset["kind"] == "pascal_tar_structure":
            words_path = materialized / "words" / f"{stem}_words.json"
            if not words_path.is_file():
                raise FileNotFoundError(words_path)
            extras[image_id]["words_path"] = str(words_path)
        for obj in tree.findall("object"):
            name = obj.findtext("name")
            if name not in category_ids:
                continue
            box = obj.find("bndbox")
            assert box is not None
            x1 = float(box.findtext("xmin", default="0"))
            y1 = float(box.findtext("ymin", default="0"))
            x2 = float(box.findtext("xmax", default="0"))
            y2 = float(box.findtext("ymax", default="0"))
            width_box, height_box = max(0.0, x2 - x1), max(0.0, y2 - y1)
            annotations.append({
                "id": annotation_id, "image_id": image_id, "category_id": category_ids[name],
                "bbox": [x1, y1, width_box, height_box], "area": width_box * height_box, "iscrowd": 0,
            })
            annotation_id += 1
    annotation = ensure_coco_fields({
        "images": images, "annotations": annotations,
        "categories": [{"id": index + 1, "name": name} for index, name in enumerate(class_names)],
    })

    def read_image(info: dict[str, Any]) -> Image.Image:
        with Image.open(image_by_id[int(info["id"])]) as opened:
            return opened.convert("RGB")

    return PreparedDataset(annotation, read_image, lambda: None, extras)


def collapse_categories(annotation: dict[str, Any], mapping: dict[str, str]) -> dict[str, Any]:
    target_names: list[str] = []
    source_to_target_name: dict[int, str] = {}
    for category in annotation["categories"]:
        source_name = str(category["name"])
        target_name = mapping.get(source_name, source_name)
        source_to_target_name[int(category["id"])] = target_name
        if target_name not in target_names:
            target_names.append(target_name)
    target_ids = {name: index + 1 for index, name in enumerate(target_names)}
    for item in annotation["annotations"]:
        item["category_id"] = target_ids[source_to_target_name[int(item["category_id"])]]
    annotation["categories"] = [{"id": target_ids[name], "name": name} for name in target_names]
    return annotation


def prepare_dataset(dataset: dict[str, Any], root: Path, cache_root: Path | None = None) -> PreparedDataset:
    functions = {
        "coco_archives": prepare_coco_archives,
        "coco_single_zip": prepare_coco_single_zip,
        "coco_directory": prepare_coco_directory,
        "yolo_directory": prepare_yolo_directory,
        "yolo_zip": prepare_yolo_zip,
        "parquet_detection": prepare_parquet_detection,
    }
    if dataset["kind"] in {"pascal_tar", "pascal_tar_structure"}:
        if cache_root is None:
            raise RuntimeError("PubTables 数据准备需要 --output-root 对应缓存目录")
        return prepare_pascal_tar(dataset, root, cache_root)
    function = functions.get(dataset["kind"])
    if function is None:
        raise RuntimeError(f"尚未实现的数据适配器: {dataset['kind']}")
    prepared = function(dataset, root)
    if "category_collapse" in dataset:
        prepared.annotation = collapse_categories(prepared.annotation, dataset["category_collapse"])
    return prepared


def normalize_label(value: str) -> str:
    aliases = {
        "aeroplane": "airplane", "motorbike": "motorcycle", "sofa": "couch",
        "tvmonitor": "tv", "diningtable": "dining table", "pottedplant": "potted plant",
        "licenseplate": "license plate", "licence plate": "license plate", "number plate": "license plate",
        "down": "down", "up": "up",
    }
    normalized = " ".join(value.lower().replace("_", " ").replace("-", " ").split())
    compact = normalized.replace(" ", "")
    return aliases.get(normalized, aliases.get(compact, normalized))


def label_mapping(model_names: dict[int, str], categories: list[dict[str, Any]]) -> dict[int, int]:
    categories_by_name = {normalize_label(str(item["name"])): int(item["id"]) for item in categories}
    mapping = {int(index): categories_by_name[normalize_label(name)] for index, name in model_names.items() if normalize_label(name) in categories_by_name}
    if len(model_names) == 1 and len(categories) == 1:
        mapping[next(iter(model_names))] = int(categories[0]["id"])
    if not mapping:
        raise RuntimeError(f"模型标签与数据集类别没有交集: model={model_names}, dataset={categories}")
    return mapping


def resolve_device(requested: str, runner: str) -> str:
    if runner == "tensorflow_retinanet":
        import tensorflow as tf
        available = bool(tf.config.list_physical_devices("GPU"))
    elif runner == "onnx_detr":
        import onnxruntime as ort
        available = "CUDAExecutionProvider" in ort.get_available_providers()
    else:
        import torch
        available = torch.cuda.is_available()
    if requested == "auto":
        return "cuda:0" if available else "cpu"
    if requested.startswith("cuda") and not available:
        raise RuntimeError(f"请求了 CUDA，但 {runner} 对应运行时未发现 GPU")
    return requested


def load_hf_predictor(spec: dict[str, Any], weight_dir: Path, device: str, dtype: str):
    import torch
    from transformers import AutoConfig, AutoImageProcessor, AutoModelForObjectDetection

    try:
        processor = AutoImageProcessor.from_pretrained(weight_dir, local_files_only=True, use_fast=False)
    except TypeError:
        processor = AutoImageProcessor.from_pretrained(weight_dir, local_files_only=True)
    config = AutoConfig.from_pretrained(weight_dir, local_files_only=True)
    if hasattr(config, "use_pretrained_backbone"):
        config.use_pretrained_backbone = False
    model = AutoModelForObjectDetection.from_pretrained(weight_dir, config=config, local_files_only=True)
    torch_dtype = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}[dtype]
    if device == "cpu" and torch_dtype == torch.float16:
        raise RuntimeError("CPU 不支持本协议的 float16 推理")
    model.to(device=torch.device(device), dtype=torch_dtype)
    model.eval()
    names = {int(key): str(value) for key, value in model.config.id2label.items()}

    def predict(image: Image.Image) -> list[tuple[list[float], float, int]]:
        width, height = image.size
        inputs = processor(images=image, return_tensors="pt")
        inputs = {
            key: value.to(device=device, dtype=torch_dtype if value.is_floating_point() else value.dtype)
            for key, value in inputs.items()
        }
        with torch.inference_mode():
            outputs = model(**inputs)
        target_sizes = torch.tensor([[height, width]], device=device)
        result = processor.post_process_object_detection(outputs, threshold=0.0, target_sizes=target_sizes)[0]
        ranked = sorted(
            zip(result["boxes"].detach().cpu().tolist(), result["scores"].detach().cpu().tolist(), result["labels"].detach().cpu().tolist()),
            key=lambda item: item[1], reverse=True,
        )
        return [(box, float(score), int(label)) for box, score, label in ranked[:300]]

    return names, predict




def detector_names(model: Any, model_dir: Path) -> dict[int, str]:
    raw_names = model.names
    names = {int(key): str(value) for key, value in (raw_names.items() if isinstance(raw_names, dict) else enumerate(raw_names))}
    if names and all(value.strip() == str(key) for key, value in names.items()):
        config_path = model_dir / "config.json"
        if config_path.is_file():
            config = json.loads(config_path.read_text(encoding="utf-8"))
            configured = config.get("names")
            if isinstance(configured, dict):
                names = {int(key): str(value) for key, value in configured.items()}
        elif len(names) == 80:
            names = dict(enumerate(COCO80_NAMES))
    return names










def coco_metrics(annotation: dict[str, Any], detections: list[dict[str, Any]]) -> dict[str, float]:
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    coco_gt = COCO()
    coco_gt.dataset = annotation
    coco_gt.createIndex()
    if detections:
        coco_dt = coco_gt.loadRes(detections)
    else:
        coco_dt = COCO()
        coco_dt.dataset = {"images": annotation["images"], "categories": annotation["categories"], "annotations": []}
        coco_dt.createIndex()
    evaluator = COCOeval(coco_gt, coco_dt, "bbox")
    evaluator.params.imgIds = [item["id"] for item in annotation["images"]]
    evaluator.evaluate()
    evaluator.accumulate()
    evaluator.summarize()
    names = ["AP", "AP50", "AP75", "AP_small", "AP_medium", "AP_large", "AR1", "AR10", "AR100", "AR_small", "AR_medium", "AR_large"]
    return {name: float(value) for name, value in zip(names, evaluator.stats)}


def reference_result(spec: dict[str, Any], metrics: dict[str, float]) -> dict[str, Any]:
    reference = spec["protocol"]["official_reference"]
    if spec["dataset"]["relation"] != "original_exact" or spec["fidelity"] == "reference_baseline":
        return {"classification": "not_officially_comparable", "reason": "dataset/protocol is not an exact official comparison chain"}
    if reference is None:
        return {"classification": "manual_reference_required"}
    if reference.get("comparable") is False:
        return {"classification": "not_officially_comparable", "reason": reference.get("note")}
    metric = reference["metric"]
    if metric not in metrics:
        return {"classification": "manual_reference_required", "reason": f"本地结果没有同名指标: {metric}"}
    actual = metrics[metric] * 100.0
    expected = float(reference["value_percent"])
    difference = actual - expected
    tolerance = float(spec["protocol"]["comparison_tolerance_percent"])
    return {
        "classification": "matches_reference" if abs(difference) <= tolerance else "does_not_match_reference",
        "metric": metric, "actual_percent": actual, "reference_percent": expected,
        "difference_percent_points": difference, "tolerance_percent_points": tolerance,
    }


def run_generic(spec: dict[str, Any], dataset: dict[str, Any], model_root: Path, dataset_root: Path, output_dir: Path, max_images: int, device: str, dtype: str) -> dict[str, Any]:
    prepared = prepare_dataset(dataset, dataset_root, output_dir.parent / "_prepared_datasets")
    try:
        images = list(prepared.annotation["images"])
        expected = dataset.get("expected_images")
        if expected is not None and len(images) != expected:
            raise RuntimeError(f"数据集图像数不符: actual={len(images)}, expected={expected}")
        selected = images if max_images == 0 else images[:max_images]
        predictions_path = output_dir / "predictions.jsonl"
        records = read_jsonl(predictions_path)
        existing = {record["image_id"] for record in records}
        valid_ids = {item["id"] for item in images}
        if not existing <= valid_ids:
            raise RuntimeError("断点文件含有不属于当前数据集的 image_id")

        model_dir = model_root / spec["model_id"]
        runner = spec["protocol"]["runner"]
        weight = resolved_weight(spec, model_root)
        model_names, predictor = load_hf_predictor(spec, model_dir, device, dtype)
        mapping = label_mapping(model_names, prepared.annotation["categories"])

        for position, info in enumerate(selected, 1):
            if info["id"] in existing:
                continue
            image = prepared.read_image(info)
            raw = predictor(image)
            detections = []
            for box, score, label in raw:
                if label not in mapping:
                    continue
                x1, y1, x2, y2 = box
                detections.append({
                    "image_id": info["id"], "category_id": mapping[label],
                    "bbox": [x1, y1, max(0.0, x2 - x1), max(0.0, y2 - y1)], "score": score,
                })
            append_jsonl(predictions_path, {"image_id": info["id"], "detections": detections})
            if position == 1 or position % 25 == 0 or position == len(selected):
                print(f"[{spec['model_key']}] {position}/{len(selected)}", flush=True)

        records = read_jsonl(predictions_path)
        selected_ids = {item["id"] for item in selected}
        completed_selected = sum(record["image_id"] in selected_ids for record in records)
        result: dict[str, Any] = {
            "state": "partial_complete", "completed_images": completed_selected,
            "total_images": len(images), "class_mapping": {str(key): value for key, value in mapping.items()},
            "dataset_snapshot": {
                "images": len(images),
                "annotations": len(prepared.annotation["annotations"]),
                "categories": len(prepared.annotation["categories"]),
                "annotation_sha256": hashlib.sha256(json.dumps(prepared.annotation, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest(),
            },
        }
        if max_images == 0:
            if len(records) != len(images):
                raise RuntimeError(f"全量预测覆盖不完整: {len(records)}/{len(images)}")
            detections = [item for record in records for item in record["detections"]]
            atomic_json(output_dir / "predictions.json", detections)
            metrics = coco_metrics(prepared.annotation, detections)
            metrics_document = {
                "metrics": metrics,
                "metrics_percent": {key: value * 100.0 for key, value in metrics.items()},
                "official_reference": spec["protocol"]["official_reference"],
                "reference_check": reference_result(spec, metrics),
            }
            atomic_json(output_dir / "metrics.json", metrics_document)
            result.update({"state": "complete", "completed_images": len(images), "metrics": metrics_document})
        return result
    finally:
        prepared.close()




def load_table_official_eval(source_root: Path):
    repository = source_root / "table-transformer"
    source = repository / "src"
    detr = repository / "detr"
    if not (source / "eval.py").is_file():
        raise FileNotFoundError(f"缺少微软官方 Table Transformer 源码；请先运行 setup_env.sh: {repository}")
    sys.path.insert(0, str(detr))
    sys.path.insert(0, str(source))
    module_spec = importlib.util.spec_from_file_location("official_table_transformer_eval", source / "eval.py")
    if module_spec is None or module_spec.loader is None:
        raise RuntimeError("无法加载 Table Transformer 官方 eval.py")
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    return module




def box_iou_xyxy(first: list[float], second: list[float]) -> float:
    x1, y1 = max(first[0], second[0]), max(first[1], second[1])
    x2, y2 = min(first[2], second[2]), min(first[3], second[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_first = max(0.0, first[2] - first[0]) * max(0.0, first[3] - first[1])
    area_second = max(0.0, second[2] - second[0]) * max(0.0, second[3] - second[1])
    union = area_first + area_second - intersection
    return intersection / union if union > 0 else 0.0




def safe_extract_zip(archive_path: Path, target: Path) -> None:
    target_resolved = target.resolve()
    with zipfile.ZipFile(archive_path) as archive:
        for member in archive.infolist():
            destination = (target / member.filename).resolve()
            if target_resolved != destination and target_resolved not in destination.parents:
                raise RuntimeError(f"拒绝解压越界 ZIP 成员: {member.filename}")
        archive.extractall(target)


def prepare_yoloe_dataset(dataset: dict[str, Any], dataset_root: Path, cache_root: Path, max_images: int) -> tuple[Path, Path, int]:
    target = cache_root / dataset["id"]
    target.mkdir(parents=True, exist_ok=True)
    marker = target / ".complete.json"
    snapshot = {
        key: {"path": dataset[key], "size": (dataset_root / dataset[key]).stat().st_size}
        for key in ("labels_zip", "coco_images_zip", "minival", "lvis_annotation_zip")
    }
    with (target / ".prepare.lock").open("w", encoding="utf-8") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if marker.is_file():
            if json.loads(marker.read_text(encoding="utf-8")) != snapshot:
                raise RuntimeError(f"YOLOE 数据缓存输入已变化；请改用新 OUTPUT_ROOT: {target}")
        else:
            print("[dataset:yoloe_lvis_minival] 首次解压 LVIS 标签与 COCO val2017 图像", flush=True)
            safe_extract_zip(dataset_root / dataset["labels_zip"], target)
            image_target = target / "lvis" / "images"
            image_target.mkdir(parents=True, exist_ok=True)
            safe_extract_zip(dataset_root / dataset["coco_images_zip"], image_target)
            shutil.copy2(dataset_root / dataset["minival"], target / "lvis" / "minival.txt")
            atomic_json(marker, snapshot)
    lvis_root = target / "lvis"
    lines = [line.strip() for line in (lvis_root / "minival.txt").read_text(encoding="utf-8").splitlines() if line.strip()]
    selected = lines if max_images == 0 else lines[:max_images]
    selected_file = lvis_root / ("minival.txt" if max_images == 0 else f"minival.smoke_{max_images}.txt")
    if max_images != 0:
        selected_file.write_text("\n".join(selected) + "\n", encoding="utf-8")
    return lvis_root, selected_file, len(lines)


@contextlib.contextmanager
def working_directory(path: Path):
    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


def parse_metric_log(text: str) -> dict[str, float]:
    metrics: dict[str, float] = {}
    patterns = {
        "AP": r"Average Precision\s+\(AP\).*?area=\s*all.*?=\s*([0-9.]+)",
        "AP_r": r"\bAP[_ ]?r\b\s*[:=]\s*([0-9.]+)",
        "AP_c": r"\bAP[_ ]?c\b\s*[:=]\s*([0-9.]+)",
        "AP_f": r"\bAP[_ ]?f\b\s*[:=]\s*([0-9.]+)",
    }
    for key, pattern in patterns.items():
        matches = re.findall(pattern, text, flags=re.IGNORECASE | re.DOTALL)
        if matches:
            metrics[key] = float(matches[-1])
    return metrics




def to_jsonable(value: Any) -> Any:
    if hasattr(value, "detach"):
        return value.detach().cpu().tolist()
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    return value






def blocked_status(spec: dict[str, Any], output_root: Path, reason: str | None = None, blocker_type: str | None = None) -> dict[str, Any]:
    output_dir = output_root / spec["model_key"]
    path = output_dir / "status.json"
    if path.is_file():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing.get("state") in {"complete", "verified_existing", "running", "partial_complete"}:
            return existing
    status = {
        "model_key": spec["model_key"], "model_id": spec["model_id"], "sequence": spec["sequence"],
        "state": "not_selected" if spec["execution_state"] == "not_selected_non_green" else "blocked",
        "blocker_type": blocker_type or spec["execution_state"],
        "reason": reason or spec["next_action"], "next_action": spec["next_action"],
        "protocol_fingerprint": spec["protocol_fingerprint"], "manifest_snapshot": spec,
        "updated_at_unix": time.time(),
    }
    atomic_json(path, status)
    return status


def run_one(manifest: dict[str, Any], spec: dict[str, Any], args: argparse.Namespace) -> bool:
    output_dir = args.output_root / spec["model_key"]
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / ".lock").open("w", encoding="utf-8") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        status_path = output_dir / "status.json"
        old = json.loads(status_path.read_text(encoding="utf-8")) if status_path.is_file() else None
        versions = runtime_versions()
        revision = source_revision(spec, args.source_root)
        fingerprint = run_fingerprint(spec, args.dtype, args.batch_size, versions, revision)
        if old and old.get("state") != "blocked" and old.get("run_fingerprint") != fingerprint:
            raise RuntimeError(f"协议或 dtype 已变化，拒绝混用旧结果目录: {output_dir}")
        if old and old.get("state") in {"complete", "verified_existing"}:
            print(f"[{spec['model_key']}] 已完整完成，跳过。")
            return True
        if spec["execution_state"] != "ready":
            blocked_status(spec, args.output_root)
            print(f"[{spec['model_key']}] 阻塞，已记录: {spec['next_action']}")
            return True

        source = source_directory(spec, args.source_root)
        expected_revision = (spec.get("official_sources") or {}).get("fixed_revision") if source is not None else None
        if source is not None and (revision is None or not source_revision_matches(revision, expected_revision)):
            reason = f"官方源码版本不符: path={source}, expected={expected_revision}, actual={revision}; 请先运行 setup_env.sh"
            blocked_status(spec, args.output_root, reason, "official_source_revision_mismatch")
            print(f"[{spec['model_key']}] {reason}", file=sys.stderr)
            return False

        missing = []
        model_dir = args.model_root / spec["model_id"]
        if not model_dir.is_dir():
            missing.append(str(model_dir))
        weight = resolved_weight(spec, args.model_root)
        if weight is not None and not weight.is_file():
            missing.append(str(weight))
        for auxiliary in spec["model"].get("auxiliary_weight_files", []):
            auxiliary_path = model_dir / auxiliary
            if not auxiliary_path.is_file():
                missing.append(str(auxiliary_path))
        missing.extend(str(path) for path in dataset_paths(manifest["datasets"][spec["dataset"]["id"]], args.dataset_root) if not path.exists())
        if missing:
            blocked_status(spec, args.output_root, f"运行时缺少本地文件: {missing}", "runtime_missing_file")
            print(f"[{spec['model_key']}] 缺文件，跳过: {missing}")
            return False

        device = resolve_device(args.device, spec["protocol"]["runner"])
        started = time.time()
        status = {
            "model_key": spec["model_key"], "model_id": spec["model_id"], "sequence": spec["sequence"],
            "state": "running", "protocol_fingerprint": spec["protocol_fingerprint"],
            "run_fingerprint": fingerprint,
            "manifest_snapshot": spec, "runtime_versions": versions,
            "official_source_revision": revision, "environment_profile": spec.get("environment"),
            "official_source_packages": verify_bundle(args.source_root) if spec.get("environment") == "wilddet3d" else None,
            "device": device, "dtype": args.dtype, "batch_size": args.batch_size,
            "max_images": args.max_images, "started_at_unix": started,
        }
        atomic_json(status_path, status)
        try:
            runner = spec["protocol"]["runner"]
            dataset = manifest["datasets"][spec["dataset"]["id"]]
            result = run_generic(spec, dataset, args.model_root, args.dataset_root, output_dir, args.max_images, device, args.dtype)
            status.update(result)
            status["elapsed_seconds"] = time.time() - started
            atomic_json(status_path, status)
            print(f"[{spec['model_key']}] {status['state']} -> {output_dir}")
            return True
        except BaseException as exc:
            status.update({"state": "failed", "error": repr(exc), "elapsed_seconds": time.time() - started})
            atomic_json(status_path, status)
            print(f"[{spec['model_key']}] 失败: {exc}", file=sys.stderr)
            return False


def summarize(manifest: dict[str, Any], output_root: Path) -> dict[str, Any]:
    rows = []
    for spec in manifest["models"]:
        status_path = output_root / spec["model_key"] / "status.json"
        if status_path.is_file():
            status = json.loads(status_path.read_text(encoding="utf-8"))
            state = status.get("state", "unknown")
        else:
            state = "not_started" if spec["execution_state"] == "ready" else "not_selected"
            status = None
        metric = None
        metrics_path = output_root / spec["model_key"] / "metrics.json"
        if metrics_path.is_file():
            payload = json.loads(metrics_path.read_text(encoding="utf-8"))
            metric = payload.get("metrics_percent", {}).get("AP")
        rows.append({
            "sequence": spec["sequence"], "model_key": spec["model_key"], "model_id": spec["model_id"],
            "state": state, "technical_complete": state in {"complete", "verified_existing"},
            "fidelity": spec["fidelity"], "dataset_relation": spec["dataset"]["relation"],
            "case_type": spec["case_type"], "special_tags": spec["special_tags"], "AP_percent": metric,
            "reference_check": (status or {}).get("metrics", {}).get("reference_check"),
        })
    counts: dict[str, int] = {}
    for row in rows:
        counts[row["state"]] = counts.get(row["state"], 0) + 1
    document = {
        "suite": manifest["suite"], "generated_at_unix": time.time(), "total_models": len(rows),
        "state_counts": counts, "technical_complete": sum(row["technical_complete"] for row in rows), "models": rows,
    }
    atomic_json(output_root / "audit_summary.json", document)
    print(json.dumps(document, ensure_ascii=False, indent=2))
    return document


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-root", type=Path, default=DEFAULT_MODEL_ROOT)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", choices=("float32", "float16", "bfloat16"), default="float32")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-images", type=int, default=10, help="10=冒烟；0=全量")
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--paths-only", action="store_true")
    parser.add_argument("--list-models", action="store_true")
    parser.add_argument("--list-runnable", action="store_true")
    parser.add_argument("--list-runnable-with-env", action="store_true")
    parser.add_argument("--shard", default="0/1", help="供 run_all.sh 使用，例如 0/2")
    parser.add_argument("--describe")
    parser.add_argument("--materialize-blocked", action="store_true")
    parser.add_argument("--summarize", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.max_images < 0:
        raise ValueError("--max-images 必须为 0 或正整数")
    if args.batch_size != 1:
        raise ValueError("当前冻结协议 batch_size=1；不要用参数隐式改变数值协议")
    manifest = load_manifest()
    if args.describe:
        print(json.dumps(resolve_model(manifest, args.describe), ensure_ascii=False, indent=2))
        return 0
    if args.list_models:
        for spec in manifest["models"]:
            print(f"{spec['sequence']:02d}\t{spec['model_key']}\t{spec['execution_state']}\t{spec['model_id']}")
        return 0
    if args.list_runnable or args.list_runnable_with_env:
        shard_index, shard_count = (int(value) for value in args.shard.split("/", 1))
        if shard_count < 1 or not 0 <= shard_index < shard_count:
            raise ValueError("--shard 必须满足 0 <= index < count")
        ready = [spec for spec in manifest["models"] if spec["execution_state"] == "ready"]
        for index, spec in enumerate(ready):
            if index % shard_count == shard_index:
                if args.list_runnable_with_env:
                    print(f"{spec['model_key']}\t{spec['environment']}")
                else:
                    print(spec["model_key"])
        return 0
    if args.preflight:
        report = preflight(manifest, args.model_root, args.dataset_root, args.source_root, args.paths_only)
        return 0 if report["status"] == "ready" else 1
    if args.materialize_blocked:
        for spec in manifest["models"]:
            if spec["execution_state"] != "ready":
                blocked_status(spec, args.output_root)
        return 0
    if args.summarize:
        summarize(manifest, args.output_root)
        return 0

    failed = []
    if len(manifest["models"]) != 1:
        raise ValueError("Expected one model in this folder")
    for spec in manifest["models"]:
        if not run_one(manifest, spec, args):
            failed.append(spec["model_key"])
    if failed:
        print(f"失败模型: {', '.join(failed)}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
