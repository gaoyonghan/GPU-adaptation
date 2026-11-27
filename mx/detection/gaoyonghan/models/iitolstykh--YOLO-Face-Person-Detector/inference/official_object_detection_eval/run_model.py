#!/usr/bin/env python3
"""Offline, resumable dispatcher for the first 62 object-detection models."""

from __future__ import annotations

import argparse
import contextlib
import fcntl
import hashlib
import importlib.util
import io
import json
import os
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
import traceback
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
DEFAULT_MODEL_ROOT = Path("/data2/gaoyonghan/gpu_model_standalone_20261110/release/model/检测/iitolstykh--YOLO-Face-Person-Detector/resources/model_index")
DEFAULT_DATASET_ROOT = Path("/home/ubuntu/文档/ChatGPT/idea 2/outputs/launch_audit_20261005/migration_inputs/datasets")
DEFAULT_OUTPUT_ROOT = Path("/data2/gaoyonghan/gpu_model_standalone_20261110/release/model/检测/iitolstykh--YOLO-Face-Person-Detector/resources/runtime/assets/workspace/results/object_detection_official_v2")
DEFAULT_SOURCE_ROOT = Path("/data2/gaoyonghan/gpu_model_standalone_20261110/release/model/检测/iitolstykh--YOLO-Face-Person-Detector/resources/runtime/assets/workspace/official_sources/object_detection")
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
    if spec["protocol"]["runner"] == "tensorflow_retinanet":
        payload["official_decoder_sha256"] = hashlib.sha256((SCRIPT_DIR / "retinanet_official.py").read_bytes()).hexdigest()
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


def load_ultralytics_predictor(spec: dict[str, Any], weight: Path, device: str, dtype: str):
    from ultralytics import YOLO

    model = YOLO(str(weight))
    names = detector_names(model, weight.parent)
    ultra_device = device.split(":", 1)[1] if device.startswith("cuda:") else device

    def predict(image: Image.Image) -> list[tuple[list[float], float, int]]:
        result = model.predict(
            source=image, imgsz=int(spec["protocol"]["input_size"]), conf=0.001, iou=0.7,
            max_det=300, device=ultra_device, half=dtype == "float16", verbose=False,
        )[0]
        if result.boxes is None:
            return []
        return [
            (box, float(score), int(label))
            for box, score, label in zip(
                result.boxes.xyxy.detach().cpu().tolist(),
                result.boxes.conf.detach().cpu().tolist(),
                result.boxes.cls.detach().cpu().tolist(),
            )
        ]

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


def load_yolov5_predictor(spec: dict[str, Any], weight: Path, device: str, dtype: str):
    if dtype == "bfloat16":
        raise RuntimeError("YOLOv5 隔离环境不支持本协议的 bfloat16")
    from yolov5 import load

    model = load(str(weight), device=device)
    model.conf = float(spec["protocol"]["score_threshold"])
    model.iou = float(spec["protocol"].get("nms_iou") or 0.7)
    model.max_det = int(spec["protocol"]["max_detections"])
    if dtype == "float16" and device.startswith("cuda"):
        model.model.half()
    names = detector_names(model, weight.parent)

    def predict(image: Image.Image) -> list[tuple[list[float], float, int]]:
        result = model(image, size=int(spec["protocol"]["input_size"]), augment=False)
        tensor = result.pred[0].detach().cpu().tolist()
        return [(row[:4], float(row[4]), int(row[5])) for row in tensor]

    return names, predict


def load_yolov10_predictor(spec: dict[str, Any], weight: Path, device: str, dtype: str):
    from ultralytics import YOLOv10

    model = YOLOv10.from_pretrained(str(weight.parent)) if weight.suffix == ".safetensors" else YOLOv10(str(weight))
    names = detector_names(model, weight.parent)
    ultra_device = device.split(":", 1)[1] if device.startswith("cuda:") else device

    def predict(image: Image.Image) -> list[tuple[list[float], float, int]]:
        result = model.predict(
            source=image, imgsz=int(spec["protocol"]["input_size"]),
            conf=float(spec["protocol"]["score_threshold"]), max_det=int(spec["protocol"]["max_detections"]),
            device=ultra_device, half=dtype == "float16", verbose=False,
        )[0]
        if result.boxes is None:
            return []
        return [
            (box, float(score), int(label))
            for box, score, label in zip(
                result.boxes.xyxy.detach().cpu().tolist(), result.boxes.conf.detach().cpu().tolist(),
                result.boxes.cls.detach().cpu().tolist(),
            )
        ]

    return names, predict


def load_onnx_detr_predictor(spec: dict[str, Any], model_dir: Path, weight: Path, device: str, dtype: str):
    if dtype != "float32":
        raise RuntimeError("Xenova FP32 ONNX 权重冻结为 float32；如需 FP16 必须改权重并使用新结果目录")
    import numpy as np
    import onnxruntime as ort
    from transformers import AutoImageProcessor

    processor = AutoImageProcessor.from_pretrained(model_dir, local_files_only=True, use_fast=False)
    available = ort.get_available_providers()
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if device.startswith("cuda") and "CUDAExecutionProvider" in available else ["CPUExecutionProvider"]
    session = ort.InferenceSession(str(weight), providers=providers)
    if device.startswith("cuda") and "CUDAExecutionProvider" not in session.get_providers():
        raise RuntimeError("ONNX CUDA provider 初始化失败，拒绝静默退回 CPU；请检查本环境的 CUDA/cuDNN 运行库")
    input_names = {item.name for item in session.get_inputs()}
    mask_input = next((item for item in session.get_inputs() if item.name == "pixel_mask"), None)
    if mask_input is not None and (mask_input.shape[-2:] != [64, 64] or mask_input.type != "tensor(int64)"):
        raise RuntimeError(f"ONNX mask differs from Transformers.js 2.17.2: {mask_input.shape}, {mask_input.type}")
    config = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
    names = {int(key): str(value) for key, value in config["id2label"].items()}

    def predict(image: Image.Image) -> list[tuple[list[float], float, int]]:
        width, height = image.size
        encoded = processor(images=image, return_tensors="np")
        feeds = {key: np.asarray(value) for key, value in encoded.items() if key in input_names}
        if mask_input is not None:
            # Publisher's Transformers.js DetrFeatureExtractor uses a 64x64 all-one mask.
            # This is not the image resolution; pixel_values keeps its configured size.
            feeds["pixel_mask"] = np.ones((1, 64, 64), dtype=np.int64)
        output_names = [item.name for item in session.get_outputs()]
        values = session.run(None, feeds)
        outputs = dict(zip(output_names, values))
        logits = next((value for key, value in outputs.items() if "logits" in key), None)
        boxes = next((value for key, value in outputs.items() if "box" in key and value.shape[-1] == 4), None)
        if logits is None:
            logits = next(value for value in values if value.ndim == 3 and value.shape[-1] != 4)
        if boxes is None:
            boxes = next(value for value in values if value.ndim == 3 and value.shape[-1] == 4)
        shifted = logits[0] - logits[0].max(axis=-1, keepdims=True)
        probabilities = np.exp(shifted) / np.exp(shifted).sum(axis=-1, keepdims=True)
        scores = probabilities[:, :-1].max(axis=-1)
        labels = probabilities[:, :-1].argmax(axis=-1)
        results = []
        for box, score, label in zip(boxes[0], scores, labels):
            cx, cy, bw, bh = [float(value) for value in box]
            results.append(([(cx - bw / 2) * width, (cy - bh / 2) * height, (cx + bw / 2) * width, (cy + bh / 2) * height], float(score), int(label)))
        return sorted(results, key=lambda item: item[1], reverse=True)[: int(spec["protocol"]["max_detections"])]

    return names, predict


def load_tensorflow_retinanet_predictor(spec: dict[str, Any], model_dir: Path, device: str, dtype: str):
    if dtype != "float32":
        raise RuntimeError("Keras SavedModel 官方链路冻结为 float32")
    import numpy as np
    import tensorflow as tf
    from retinanet_official import DecodePredictions, prepare_image

    model = tf.saved_model.load(str(model_dir))
    infer = model.signatures["serving_default"]
    if len(infer.structured_input_signature[1]) != 1:
        raise RuntimeError(f"Unexpected RetinaNet inputs: {infer.structured_input_signature}")
    input_name = next(iter(infer.structured_input_signature[1]))
    decoder = DecodePredictions(
        confidence_threshold=float(spec["protocol"]["score_threshold"]),
        nms_iou_threshold=float(spec["protocol"]["nms_iou"]),
        max_detections=int(spec["protocol"]["max_detections"]),
    )

    def predict(image: Image.Image) -> list[tuple[list[float], float, int]]:
        tensor, ratio = prepare_image(np.asarray(image.convert("RGB"), dtype=np.float32))
        outputs = infer(**{input_name: tensor})
        if len(outputs) != 1:
            raise RuntimeError(f"Unexpected RetinaNet outputs: {list(outputs)}")
        raw = next(iter(outputs.values()))
        if raw.shape.rank != 3 or raw.shape[-1] != 84:
            raise RuntimeError(f"Expected RetinaNet box deltas + 80 logits, got {raw.shape}")
        detections = decoder(tensor, raw)
        count = int(detections.valid_detections[0])
        # Official convert_to_corners returns XYXY (despite TensorFlow NMS naming).
        boxes = (detections.nmsed_boxes[0][:count] / ratio).numpy().tolist()
        return [
            (box, float(score), int(label))
            for box, score, label in zip(
                boxes, detections.nmsed_scores[0][:count].numpy(),
                detections.nmsed_classes[0][:count].numpy(),
            )
        ]

    return dict(enumerate(COCO80_NAMES)), predict


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
        if runner == "hf_object_detection":
            model_names, predictor = load_hf_predictor(spec, model_dir, device, dtype)
        elif runner == "ultralytics_detection":
            assert weight is not None
            model_names, predictor = load_ultralytics_predictor(spec, weight, device, dtype)
        elif runner == "yolov5_detection":
            assert weight is not None
            model_names, predictor = load_yolov5_predictor(spec, weight, device, dtype)
        elif runner == "yolov10_detection":
            assert weight is not None
            model_names, predictor = load_yolov10_predictor(spec, weight, device, dtype)
        elif runner == "onnx_detr":
            assert weight is not None
            model_names, predictor = load_onnx_detr_predictor(spec, model_dir, weight, device, dtype)
        elif runner == "tensorflow_retinanet":
            model_names, predictor = load_tensorflow_retinanet_predictor(spec, model_dir, device, dtype)
        else:
            raise RuntimeError(f"通用推理器不支持 runner={runner}")
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


def load_table_structure_predictor(model_dir: Path, device: str, dtype: str):
    import torch
    from transformers import AutoConfig, AutoModelForObjectDetection
    from table_datasets import get_structure_transform

    transform = get_structure_transform("val")
    config = AutoConfig.from_pretrained(model_dir, local_files_only=True)
    if hasattr(config, "use_pretrained_backbone"):
        config.use_pretrained_backbone = False
    model = AutoModelForObjectDetection.from_pretrained(model_dir, config=config, local_files_only=True)
    torch_dtype = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}[dtype]
    model.to(device=torch.device(device), dtype=torch_dtype).eval()

    def predict(image: Image.Image) -> list[tuple[list[float], float, int]]:
        width, height = image.size
        tensor, _ = transform(image, {"boxes": torch.empty((0, 4))})
        encoded = {"pixel_values": tensor.unsqueeze(0).to(device=device, dtype=torch_dtype)}
        with torch.inference_mode():
            outputs = model(**encoded)
        probabilities = outputs.logits[0].softmax(-1)
        scores, labels = probabilities.max(-1)
        boxes = outputs.pred_boxes[0].detach().cpu().tolist()
        results = []
        for box, score, label in zip(boxes, scores.detach().cpu().tolist(), labels.detach().cpu().tolist()):
            cx, cy, bw, bh = [float(value) for value in box]
            results.append(([(cx - bw / 2) * width, (cy - bh / 2) * height, (cx + bw / 2) * width, (cy + bh / 2) * height], float(score), int(label)))
        return results

    return predict


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


def run_table_structure(
    spec: dict[str, Any], dataset: dict[str, Any], model_root: Path, dataset_root: Path,
    source_root: Path, output_dir: Path, max_images: int, device: str, dtype: str,
) -> dict[str, Any]:
    official = load_table_official_eval(source_root)
    import fitz
    if fitz.VersionBind != "1.21.1":
        raise RuntimeError("GriTS requires author's PyMuPDF==1.21.1; run setup_repairs.sh")
    prepared = prepare_dataset(dataset, dataset_root, output_dir.parent / "_prepared_datasets")
    try:
        images = list(prepared.annotation["images"])
        selected = images if max_images == 0 else images[:max_images]
        expected = dataset.get("expected_images")
        if expected is not None and len(images) != expected:
            raise RuntimeError(f"PubTables 图像数不符: {len(images)} != {expected}")
        extras = prepared.extras or {}
        annotations_by_image: dict[int, list[dict[str, Any]]] = {}
        for annotation in prepared.annotation["annotations"]:
            annotations_by_image.setdefault(int(annotation["image_id"]), []).append(annotation)
        predictions_path = output_dir / "predictions.jsonl"
        records = read_jsonl(predictions_path)
        existing = {int(record["image_id"]) for record in records}
        predictor = load_table_structure_predictor(model_root / spec["model_id"], device, dtype)
        names = {index: name for index, name in enumerate(dataset["class_names"])}
        mapping = {index: index + 1 for index in names}
        class_names = official.structure_class_names
        thresholds = official.structure_class_thresholds
        class_map = official.structure_class_map

        for position, info in enumerate(selected, 1):
            image_id = int(info["id"])
            if image_id in existing:
                continue
            image = prepared.read_image(info)
            predicted = predictor(image)
            pred_boxes = [box for box, _, _ in predicted]
            pred_scores = [score for _, score, _ in predicted]
            pred_labels = [label for _, _, label in predicted]
            true_boxes = []
            true_labels = []
            for annotation in annotations_by_image.get(image_id, []):
                x, y, width, height = annotation["bbox"]
                true_boxes.append([x, y, x + width, y + height])
                true_labels.append(int(annotation["category_id"]) - 1)
            true_scores = [1.0] * len(true_labels)
            tokens = json.loads(Path(extras[image_id]["words_path"]).read_text(encoding="utf-8"))
            true_structure, true_cells, _ = official.objects_to_cells(
                [box.copy() for box in true_boxes], true_labels, true_scores, tokens, class_names, thresholds, class_map,
            )
            _, pred_cells, _ = official.objects_to_cells(
                [box.copy() for box in pred_boxes], pred_labels, pred_scores, tokens, class_names, thresholds, class_map,
            )
            # Official standard metrics; grits-all additionally computes diagnostic RawLoc/DAR.
            # objects_to_cells mutates boxes, so pass copies and preserve raw detections above.
            sample_metrics = official.compute_metrics(
                "grits", true_boxes, true_labels, true_scores, true_cells,
                pred_boxes, pred_labels, pred_scores, pred_cells,
            )
            sample_metrics.update(official.compute_statistics(true_structure, true_cells))
            sample_metrics = {key: float(value) if hasattr(value, "item") or isinstance(value, float) else value for key, value in sample_metrics.items()}
            detections = []
            for box, score, label in predicted:
                if label not in mapping:
                    continue
                x1, y1, x2, y2 = box
                detections.append({"image_id": image_id, "category_id": mapping[label], "bbox": [x1, y1, max(0.0, x2 - x1), max(0.0, y2 - y1)], "score": score})
            append_jsonl(predictions_path, {"image_id": image_id, "detections": detections, "grits": sample_metrics})
            if position == 1 or position % 25 == 0 or position == len(selected):
                print(f"[{spec['model_key']}] {position}/{len(selected)}", flush=True)

        records = read_jsonl(predictions_path)
        selected_ids = {int(info["id"]) for info in selected}
        completed = sum(int(record["image_id"]) in selected_ids for record in records)
        result = {
            "state": "partial_complete", "completed_images": completed, "total_images": len(images),
            "class_mapping": {str(key): value for key, value in mapping.items()},
            "dataset_snapshot": {
                "images": len(images), "annotations": len(prepared.annotation["annotations"]),
                "categories": len(prepared.annotation["categories"]),
                "annotation_sha256": hashlib.sha256(json.dumps(prepared.annotation, sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
            },
        }
        if max_images == 0:
            if len(records) != len(images):
                raise RuntimeError(f"全量 GriTS 覆盖不完整: {len(records)}/{len(images)}")
            detections = [item for record in records for item in record["detections"]]
            atomic_json(output_dir / "predictions.json", detections)
            summary = official.compute_metrics_summary([record["grits"] for record in records], "grits")
            serializable = {group: {key: float(value) if hasattr(value, "item") else value for key, value in values.items()} for group, values in summary.items()}
            metrics_document = {
                "metrics": serializable,
                "official_evaluator": "microsoft/table-transformer src/eval.py",
                "official_evaluation_mode": "grits",
                "excluded_diagnostic_metrics": ["grits_rawloc", "dar_con_original", "dar_con"],
                "official_reference": spec["protocol"]["official_reference"],
                "reference_check": {"classification": "manual_reference_required"},
            }
            atomic_json(output_dir / "metrics.json", metrics_document)
            result.update({"state": "complete", "completed_images": len(images), "metrics": metrics_document})
        return result
    finally:
        prepared.close()


def box_iou_xyxy(first: list[float], second: list[float]) -> float:
    x1, y1 = max(first[0], second[0]), max(first[1], second[1])
    x2, y2 = min(first[2], second[2]), min(first[3], second[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_first = max(0.0, first[2] - first[0]) * max(0.0, first[3] - first[1])
    area_second = max(0.0, second[2] - second[0]) * max(0.0, second[3] - second[1])
    union = area_first + area_second - intersection
    return intersection / union if union > 0 else 0.0


def run_rfdetr_webclick(
    spec: dict[str, Any], dataset: dict[str, Any], model_root: Path, dataset_root: Path,
    output_dir: Path, max_images: int, dtype: str,
) -> dict[str, Any]:
    if dtype != "float32":
        raise RuntimeError("UI-DETR-1 模型卡链路冻结为 float32")
    import numpy as np
    import pyarrow.parquet as parquet
    from rfdetr.detr import RFDETRMedium

    rows: list[dict[str, Any]] = []
    for relative in dataset["paths"]:
        rows.extend(parquet.read_table(dataset_root / relative).to_pylist())
    expected = dataset.get("expected_images")
    if expected is not None and len(rows) != expected:
        raise RuntimeError(f"WebClick 样本数不符: {len(rows)} != {expected}")
    selected = rows if max_images == 0 else rows[:max_images]
    predictions_path = output_dir / "predictions.jsonl"
    records = read_jsonl(predictions_path)
    existing = {int(record["image_id"]) for record in records}
    model = RFDETRMedium(
        pretrain_weights=str(model_root / spec["model_id"] / spec["model"]["weight_file"]),
        resolution=int(spec["protocol"]["input_size"]),
    )
    threshold = float(spec["protocol"]["score_threshold"])
    for index, row in enumerate(selected):
        image_id = index + 1
        if image_id in existing:
            continue
        with Image.open(io.BytesIO(row["image"])) as opened:
            image = opened.convert("RGB")
        width, height = image.size
        detections = model.predict(np.asarray(image), threshold=threshold)
        boxes = [[float(value) for value in box] for box in detections.xyxy]
        confidence = getattr(detections, "confidence", None)
        scores = [float(value) for value in confidence] if confidence is not None else [1.0] * len(boxes)
        relative = [float(value) for value in row["bbox"]]
        target = [relative[0] * width, relative[1] * height, relative[2] * width, relative[3] * height]
        center_x, center_y = (target[0] + target[2]) / 2, (target[1] + target[3]) / 2
        center_covered = any(box[0] <= center_x <= box[2] and box[1] <= center_y <= box[3] for box in boxes)
        maximum_iou = max((box_iou_xyxy(box, target) for box in boxes), default=0.0)
        append_jsonl(predictions_path, {
            "image_id": image_id, "instruction": row["instruction"], "bucket": row["bucket"],
            "target_bbox_relative_xyxy": relative, "target_bbox_pixel_xyxy": target,
            "detections": [{"bbox_xyxy": box, "score": score} for box, score in zip(boxes, scores)],
            "oracle_target_center_covered": center_covered, "oracle_max_iou": maximum_iou,
        })
        position = index + 1
        if position == 1 or position % 25 == 0 or position == len(selected):
            print(f"[{spec['model_key']}] {position}/{len(selected)}", flush=True)
    records = read_jsonl(predictions_path)
    selected_ids = set(range(1, len(selected) + 1))
    completed = sum(int(record["image_id"]) in selected_ids for record in records)
    result = {"state": "partial_complete", "completed_images": completed, "total_images": len(rows)}
    if max_images == 0:
        if len(records) != len(rows):
            raise RuntimeError(f"WebClick 全量覆盖不完整: {len(records)}/{len(rows)}")
        overall = sum(bool(record["oracle_target_center_covered"]) for record in records) / len(records)
        iou50 = sum(float(record["oracle_max_iou"]) >= 0.5 for record in records) / len(records)
        buckets = {}
        for bucket in sorted({str(record["bucket"]) for record in records}):
            subset = [record for record in records if str(record["bucket"]) == bucket]
            buckets[bucket] = {
                "samples": len(subset),
                "oracle_target_center_coverage": sum(bool(record["oracle_target_center_covered"]) for record in subset) / len(subset),
            }
        metrics_document = {
            "metrics": {"oracle_target_center_coverage": overall, "oracle_iou50_coverage": iou50},
            "by_bucket": buckets,
            "official_reference": spec["protocol"]["official_reference"],
            "reference_check": {
                "classification": "not_officially_comparable",
                "reason": "模型卡 70.8% 还包含 Gemini 2.5 Pro 根据编号框和文本指令选择点击目标；本地只评估检测框覆盖能力。",
            },
        }
        atomic_json(output_dir / "metrics.json", metrics_document)
        result.update({"state": "complete", "completed_images": len(rows), "metrics": metrics_document})
    return result


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
    # YOLOE's official Fixed-AP script publishes named, percentage-valued rows.
    copied = re.findall(r"copypaste:\s*([^\n\r]+)", text)
    for names, values in zip(copied, copied[1:]):
        keys = [key.strip() for key in names.split(",")]
        if keys and keys[0] == "AP":
            numbers = [float(value.strip()) / 100.0 for value in values.split(",")]
            if len(keys) != len(numbers):
                raise RuntimeError("Official Fixed-AP metric columns do not match")
            return dict(zip(keys, numbers))
    metrics: dict[str, float] = {}
    patterns = {
        "AP": r"^AP:\s*([0-9.]+)\s*$",
        "AP_r": r"\bAP[_ ]?r\b\s*[:=]\s*([0-9.]+)",
        "AP_c": r"\bAP[_ ]?c\b\s*[:=]\s*([0-9.]+)",
        "AP_f": r"\bAP[_ ]?f\b\s*[:=]\s*([0-9.]+)",
    }
    for key, pattern in patterns.items():
        matches = re.findall(pattern, text, flags=re.IGNORECASE | re.MULTILINE)
        if matches:
            metrics[key] = float(matches[-1])
    return metrics


def run_yoloe_official(
    spec: dict[str, Any], dataset: dict[str, Any], model_root: Path, dataset_root: Path,
    source_root: Path, output_dir: Path, max_images: int, device: str, dtype: str,
) -> dict[str, Any]:
    import yaml
    from ultralytics import YOLOE
    from ultralytics.models.yolo.yoloe.val_pe_free import YOLOEPEFreeDetectValidator
    from ultralytics.nn.modules.head import YOLOEDetect

    repository = source_root / "yoloe"
    if not (repository / "val_pe_free.py").is_file():
        raise FileNotFoundError(f"缺少 YOLOE 官方源码；请先运行 setup_env.sh: {repository}")
    lvis_root, selected_file, total_images = prepare_yoloe_dataset(
        dataset, dataset_root, output_dir.parent / "_prepared_datasets", max_images,
    )
    data = yaml.safe_load((repository / "ultralytics/cfg/datasets/lvis.yaml").read_text(encoding="utf-8"))
    data["path"] = str(lvis_root)
    data["minival"] = str(selected_file)
    run_name = "full" if max_images == 0 else f"smoke_{max_images}"
    data_yaml = output_dir / f"lvis.{run_name}.yaml"
    data_yaml.write_text(yaml.safe_dump(data, allow_unicode=True, sort_keys=False), encoding="utf-8")
    model_dir = model_root / spec["model_id"]
    prompt_free_weight = model_dir / spec["model"]["weight_file"]
    unfused_weight = model_dir / spec["model"]["auxiliary_weight_files"][0]
    ultra_device = device.split(":", 1)[1] if device.startswith("cuda:") else device
    if ultra_device == "cpu":
        raise RuntimeError("YOLOE 官方 prompt-free vocab/Fixed AP 链路需要 CUDA")
    native_root = output_dir / "native"
    with working_directory(repository):
        unfused = YOLOE(str(unfused_weight))
        unfused.eval()
        unfused.to(device)
        names = [line.strip() for line in (repository / "tools/ram_tag_list.txt").read_text(encoding="utf-8").splitlines() if line.strip()]
        vocab = unfused.get_vocab(names)
        model = YOLOE(str(prompt_free_weight)).to(device)
        model.set_vocab(vocab, names=names)
        head = model.model.model[-1]
        head.is_fused = True
        head.conf = float(spec["protocol"]["score_threshold"])
        head.max_det = int(spec["protocol"]["max_detections"])
        head.__class__ = YOLOEDetect
        validation = model.val(
            data=str(data_yaml), batch=1, split="minival", rect=False,
            max_det=int(spec["protocol"]["max_detections"]), single_cls=False,
            validator=YOLOEPEFreeDetectValidator, project=str(native_root), name=run_name,
            exist_ok=True, save_json=True, device=ultra_device, half=dtype == "float16",
        )
    save_dir = Path(getattr(validation, "save_dir", native_root / run_name))
    prediction_candidates = [save_dir / "predictions.json", native_root / run_name / "predictions.json"]
    predictions_path = next((path for path in prediction_candidates if path.is_file()), None)
    if predictions_path is None:
        found = list(native_root.rglob("predictions.json"))
        predictions_path = found[-1] if found else None
    if predictions_path is None:
        raise FileNotFoundError(f"YOLOE 官方 validator 未生成 predictions.json: {native_root}")
    completed = total_images if max_images == 0 else min(max_images, total_images)
    result: dict[str, Any] = {
        "state": "partial_complete", "completed_images": completed, "total_images": total_images,
        "native_predictions": str(predictions_path),
    }
    if max_images == 0:
        annotation = lvis_root / "annotations/lvis_v1_minival.json"
        command = [
            sys.executable, str(repository / "tools/eval_open_ended.py"), "--json", str(annotation),
            "--pred", str(predictions_path), "--names", str(repository / "tools/ram_tag_list.txt"),
            "--devices", "0", "--fixed",
        ]
        environment = dict(os.environ)
        environment["PYTHONPATH"] = os.pathsep.join([str(repository / "third_party/lvis-api"), str(repository), environment.get("PYTHONPATH", "")])
        # The official open-ended script starts eval_fixed_ap.py via `python`.
        environment["PATH"] = str(Path(sys.executable).parent) + os.pathsep + environment.get("PATH", "")
        with working_directory(repository):
            evaluated = subprocess.run(command, capture_output=True, text=True, env=environment)
        log = evaluated.stdout + ("\nSTDERR:\n" + evaluated.stderr if evaluated.stderr else "")
        (output_dir / "official_evaluator.log").write_text(log, encoding="utf-8")
        print(log, flush=True)
        if evaluated.returncode != 0:
            raise RuntimeError(f"YOLOE 官方 LVIS evaluator 失败，退出码 {evaluated.returncode}")
        metrics = parse_metric_log(log)
        if "AP" not in metrics:
            raise RuntimeError("YOLOE evaluator returned without an AP; inspect official_evaluator.log (nested evaluator may have failed)")
        metrics_document = {
            "metrics": metrics,
            "metrics_percent": {key: value * 100.0 for key, value in metrics.items()},
            "official_reference": spec["protocol"]["official_reference"],
            "reference_check": reference_result(spec, metrics),
            "official_command": command,
        }
        atomic_json(output_dir / "metrics.json", metrics_document)
        result.update({"state": "complete", "metrics": metrics_document})
    return result


def to_jsonable(value: Any) -> Any:
    if hasattr(value, "detach"):
        return value.detach().cpu().tolist()
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    return value


def run_wilddet3d_official(
    spec: dict[str, Any], dataset: dict[str, Any], model_root: Path, dataset_root: Path,
    source_root: Path, output_dir: Path, max_images: int, device: str,
) -> dict[str, Any]:
    repository = source_root / "WildDet3D"
    if not (repository / "configs/eval/in_the_wild/text.py").is_file():
        raise FileNotFoundError(f"缺少 WildDet3D 官方源码；请先运行 setup_env.sh: {repository}")
    dataset_dir = dataset_root / dataset["path"]
    annotation_path = dataset_dir / dataset["annotation"]
    annotation = json.loads(annotation_path.read_text(encoding="utf-8"))
    if len(annotation["images"]) != int(dataset["expected_images"]):
        raise RuntimeError(f"WildDet3D 图像数不符: {len(annotation['images'])} != {dataset['expected_images']}")
    checkpoint = model_root / spec["model_id"] / spec["model"]["weight_file"]
    if max_images == 0:
        required = [repository / "pretrained/sam3/sam3_detector.pt", repository / "pretrained/lingbot-depth/postrain-dc-vitl14/model.pt"]
        missing = [str(path) for path in required if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"Official WildDet3D auxiliary weights missing: {missing}; run prepare_wilddet_weights.py after obtaining SAM3 access")
    if max_images != 0:
        import numpy as np
        import torch

        sys.path.insert(0, str(repository))
        with working_directory(repository):
            from wilddet3d import build_model, preprocess
            model = build_model(checkpoint=str(checkpoint), score_threshold=float(spec["protocol"]["score_threshold"]), skip_pretrained=True)
            model.eval()
            model.to(device)
            annotations_by_image: dict[int, list[str]] = {}
            for item in annotation["annotations"]:
                annotations_by_image.setdefault(int(item["image_id"]), []).append(str(item["category_name"]))
            selected = annotation["images"][:max_images]
            predictions_path = output_dir / "predictions.jsonl"
            records = read_jsonl(predictions_path)
            existing = {int(record["image_id"]) for record in records}
            for position, info in enumerate(selected, 1):
                image_id = int(info["id"])
                if image_id in existing:
                    continue
                image_path = dataset_dir / info["file_path"]
                with Image.open(image_path) as opened:
                    image = np.asarray(opened.convert("RGB"), dtype=np.float32)
                prepared = preprocess(image, np.asarray(info["K"], dtype=np.float32))
                names = list(dict.fromkeys(annotations_by_image.get(image_id, [])))[:50]
                with torch.inference_mode():
                    results = model(
                        images=prepared["images"].to(device),
                        intrinsics=prepared["intrinsics"].to(device)[None],
                        input_hw=[prepared["input_hw"]], original_hw=[prepared["original_hw"]],
                        padding=[prepared["padding"]], input_texts=names,
                    )
                append_jsonl(predictions_path, {"image_id": image_id, "input_texts": names, "outputs": to_jsonable(results)})
                print(f"[{spec['model_key']}] {position}/{len(selected)}", flush=True)
        return {"state": "partial_complete", "completed_images": len(selected), "total_images": len(annotation["images"])}

    data_parent = repository / "data"
    data_parent.mkdir(parents=True, exist_ok=True)
    link = data_parent / "in_the_wild"
    if link.is_symlink():
        if link.resolve() != dataset_dir.resolve():
            raise RuntimeError(f"WildDet3D 数据链接指向其他目录: {link} -> {link.resolve()}")
    elif link.exists():
        if link.resolve() != dataset_dir.resolve():
            raise RuntimeError(f"WildDet3D 官方源码的 data/in_the_wild 已存在且不是本数据集: {link}")
    else:
        link.symlink_to(dataset_dir, target_is_directory=True)
    vis4d = Path(sys.executable).parent / "vis4d"
    command = [str(vis4d), "test", "--config", "configs/eval/in_the_wild/text.py", "--gpus", "1", "--ckpt", str(checkpoint)]
    environment = dict(os.environ)
    environment["VIS4D_WORKSPACE"] = str(output_dir / "native")
    environment["PYTHONPATH"] = os.pathsep.join([str(repository), environment.get("PYTHONPATH", "")])
    with working_directory(repository):
        evaluated = subprocess.run(command, capture_output=True, text=True, env=environment)
    log = evaluated.stdout + ("\nSTDERR:\n" + evaluated.stderr if evaluated.stderr else "")
    (output_dir / "official_evaluator.log").write_text(log, encoding="utf-8")
    print(log, flush=True)
    if evaluated.returncode != 0:
        raise RuntimeError(f"WildDet3D 官方 Vis4D evaluator 失败，退出码 {evaluated.returncode}")
    metrics = parse_metric_log(log)
    if "AP" not in metrics:
        raise RuntimeError("WildDet3D evaluator returned without an AP; inspect official_evaluator.log")
    metrics_document = {
        "metrics": metrics,
        "metrics_percent": {key: value * 100.0 if value <= 1.0 else value for key, value in metrics.items()},
        "official_reference": spec["protocol"]["official_reference"],
        "reference_check": reference_result(spec, {key: value if value <= 1.0 else value / 100.0 for key, value in metrics.items()}),
        "official_command": command,
    }
    atomic_json(output_dir / "metrics.json", metrics_document)
    return {"state": "complete", "completed_images": len(annotation["images"]), "total_images": len(annotation["images"]), "metrics": metrics_document}


def run_delegate(spec: dict[str, Any], output_dir: Path, max_images: int, device: str) -> dict[str, Any]:
    package = Path("/data2/gaoyonghan/gpu_model_standalone_20261110/release/model/检测/iitolstykh--YOLO-Face-Person-Detector/resources/runtime/assets/workspace/document_layout_table_eval")
    if not (package / "run_model.py").is_file():
        raise FileNotFoundError(package / "run_model.py")
    keys = ["taha-ctdar-tracka"] if spec["sequence"] == 17 else ["aryn-doclaynet-val", "aryn-doclaynet-test"]
    statuses = {}
    for key in keys:
        command = [
            sys.executable, str(package / "run_model.py"), "--model", key,
            "--max-images", str(max_images), "--device", device,
            "--output-root", str(output_dir / "delegate"),
        ]
        subprocess.run(command, check=True)
        status = json.loads((output_dir / "delegate" / key / "status.json").read_text(encoding="utf-8"))
        statuses[key] = status
    states = {status["state"] for status in statuses.values()}
    state = "complete" if max_images == 0 and states == {"complete"} else "partial_complete"
    metrics = {}
    for key in keys:
        path = output_dir / "delegate" / key / "metrics.json"
        if path.is_file():
            metrics[key] = json.loads(path.read_text(encoding="utf-8"))
    if metrics:
        atomic_json(output_dir / "metrics.json", {"splits": metrics, "reference_check": {"classification": "manual_reference_required"}})
    return {
        "state": state,
        "completed_images": sum(int(status.get("completed_images", 0)) for status in statuses.values()),
        "total_images": sum(int(status.get("total_images", 0)) for status in statuses.values()),
        "delegated_statuses": statuses,
    }


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

        if spec["protocol"]["runner"] == "wilddet3d_official" and args.max_images == 0:
            auxiliary = ["pretrained/sam3/sam3_detector.pt", "pretrained/lingbot-depth/postrain-dc-vitl14/model.pt"]
            absent = [name for name in auxiliary if not (args.source_root / "WildDet3D" / name).is_file()]
            if absent:
                reason = f"官方评估依赖缺失: {absent}。SAM3 需要发布者授权；获准后运行 prepare_wilddet_weights.py。"
                blocked_status(spec, args.output_root, reason, "official_auxiliary_weights_required")
                print(f"[{spec['model_key']}] {reason}", file=sys.stderr)
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
            if runner == "document_layout_delegate":
                result = run_delegate(spec, output_dir, args.max_images, device)
            elif runner == "table_structure_grits":
                result = run_table_structure(
                    spec, dataset, args.model_root, args.dataset_root, args.source_root,
                    output_dir, args.max_images, device, args.dtype,
                )
            elif runner == "rfdetr_webclick":
                result = run_rfdetr_webclick(
                    spec, dataset, args.model_root, args.dataset_root, output_dir, args.max_images, args.dtype,
                )
            elif runner == "yoloe_official":
                result = run_yoloe_official(
                    spec, dataset, args.model_root, args.dataset_root, args.source_root,
                    output_dir, args.max_images, device, args.dtype,
                )
            elif runner == "wilddet3d_official":
                result = run_wilddet3d_official(
                    spec, dataset, args.model_root, args.dataset_root, args.source_root,
                    output_dir, args.max_images, device,
                )
            else:
                result = run_generic(spec, dataset, args.model_root, args.dataset_root, output_dir, args.max_images, device, args.dtype)
            status.update(result)
            status["elapsed_seconds"] = time.time() - started
            atomic_json(status_path, status)
            print(f"[{spec['model_key']}] {status['state']} -> {output_dir}")
            return True
        except BaseException as exc:
            status.update({"state": "failed", "error": repr(exc), "traceback": traceback.format_exc(), "elapsed_seconds": time.time() - started})
            atomic_json(status_path, status)
            traceback.print_exc()
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
    parser.add_argument("--models", default="all", help="all，或逗号分隔的 model_key/model_id")
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
    for spec in selected_models(manifest, args.models):
        if not run_one(manifest, spec, args):
            failed.append(spec["model_key"])
    if failed:
        print(f"失败模型: {', '.join(failed)}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
