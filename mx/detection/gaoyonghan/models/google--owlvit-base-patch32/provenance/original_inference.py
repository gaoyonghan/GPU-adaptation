#!/usr/bin/env python3
"""Auditable, offline evaluation for the eight downloaded zero-shot detectors."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import os
import platform
import re
import socket
import sys
import tempfile
import time
import traceback
import zipfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence


os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

SCRIPT_DIR = Path(__file__).resolve().parent
MANIFEST_PATH = SCRIPT_DIR / "models.json"
PROMPT_TEMPLATES = (
    "itap of a {}.",
    "a bad photo of the {}.",
    "a origami {}.",
    "a photo of the large {}.",
    "a {} in a video game.",
    "art of the {}.",
    "a photo of the small {}.",
)
COCO_STATS = (
    "AP", "AP50", "AP75", "AP_small", "AP_medium", "AP_large",
    "AR_1", "AR_10", "AR_100", "AR_small", "AR_medium", "AR_large",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def protocol_fingerprint(spec: dict[str, Any]) -> str:
    value = json.dumps(spec, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def manifest() -> dict[str, Any]:
    return read_json(MANIFEST_PATH)


def model_index(data: dict[str, Any]) -> dict[str, dict[str, Any]]:
    result = {}
    for spec in data["models"]:
        result[spec["key"]] = spec
        result[spec["repo_id"]] = spec
    return result


def read_json_member(archive_path: Path, member: str) -> dict[str, Any]:
    with zipfile.ZipFile(archive_path) as archive:
        with archive.open(member) as stream:
            return json.load(io.TextIOWrapper(stream, encoding="utf-8"))


class EvaluationDataset:
    def __init__(self, dataset_root: Path, dataset_key: str, dataset_spec: dict[str, Any]):
        self.dataset_root = dataset_root
        self.key = dataset_key
        self.spec = dataset_spec
        annotation_path = dataset_root / dataset_spec["annotation_archive"]
        self.annotation_path = annotation_path
        self.annotation = read_json_member(annotation_path, dataset_spec["annotation_member"])
        self.images = sorted(self.annotation["images"], key=lambda row: int(row["id"]))
        self.categories = sorted(self.annotation["categories"], key=lambda row: int(row["id"]))
        self.archives: dict[str, zipfile.ZipFile] = {}
        for relative_path in dataset_spec["image_archives"]:
            path = dataset_root / relative_path
            self.archives[path.stem] = zipfile.ZipFile(path)

    def close(self) -> None:
        for archive in self.archives.values():
            archive.close()

    def load_image(self, record: dict[str, Any]):
        from PIL import Image

        if self.key == "coco2017_val":
            split = "val2017"
            filename = Path(record["file_name"]).name
        else:
            parts = record["coco_url"].split("/")
            split, filename = parts[-2], parts[-1]
        member = f"{split}/{filename}"
        data = self.archives[split].read(member)
        return Image.open(io.BytesIO(data)).convert("RGB")


def batches(items: Sequence[Any], batch_size: int) -> Iterable[Sequence[Any]]:
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def canonicalize_label(value: str) -> str:
    value = value.lower()
    value = re.sub(r"[^a-z0-9- ]", " ", value)
    value = re.sub(r"\s+", " ", value)
    value = re.sub(r"-+", "-", value)
    return value.strip()


def dino_label(value: str) -> str:
    return " ".join(value.replace("_", " ").replace("(", " ").replace(")", " ").split()).lower()


def build_grounding_caption(categories: Sequence[dict[str, Any]]):
    caption = ""
    all_spans = []
    for category in categories:
        spans = []
        for word in dino_label(category["name"]).split():
            if caption:
                caption += " "
            start = len(caption)
            caption += word
            spans.append((start, len(caption)))
        caption += " ."
        all_spans.append(spans)
    return caption, all_spans


def grounding_positive_map(processor, caption, spans, width, device):
    import torch

    encoded = processor.tokenizer(
        caption,
        add_special_tokens=True,
        truncation=True,
        max_length=width,
        return_offsets_mapping=True,
    )
    positive_map = torch.zeros((len(spans), width), dtype=torch.float32, device=device)
    for category_index, category_spans in enumerate(spans):
        for token_index, (token_start, token_end) in enumerate(encoded["offset_mapping"][:width]):
            if token_end > token_start and any(
                token_start < end and token_end > start for start, end in category_spans
            ):
                positive_map[category_index, token_index] = 1.0
    sums = positive_map.sum(dim=1, keepdim=True)
    if (sums == 0).any():
        raise RuntimeError("Grounding DINO 类别提示词被截断，无法构造官方 positive-map")
    return positive_map / sums


def move_inputs(inputs, device, dtype):
    import torch

    moved = {}
    for key, value in inputs.items():
        if not isinstance(value, torch.Tensor):
            moved[key] = value
        elif key == "pixel_values":
            moved[key] = value.to(device=device, dtype=dtype)
        else:
            moved[key] = value.to(device=device)
    return moved


def xyxy_predictions(image_id, boxes, scores, labels, category_ids, width, height):
    predictions = []
    for box, score, label in zip(boxes.tolist(), scores.tolist(), labels.tolist()):
        x1 = min(max(float(box[0]), 0.0), float(width))
        y1 = min(max(float(box[1]), 0.0), float(height))
        x2 = min(max(float(box[2]), 0.0), float(width))
        y2 = min(max(float(box[3]), 0.0), float(height))
        box_width, box_height = x2 - x1, y2 - y1
        if box_width <= 0 or box_height <= 0 or not math.isfinite(float(score)):
            continue
        predictions.append({
            "image_id": int(image_id),
            "category_id": int(category_ids[int(label)]),
            "bbox": [x1, y1, box_width, box_height],
            "score": float(score),
        })
    return predictions


def topk_xyxy(probabilities, pred_boxes, records, category_ids, max_detections):
    import torch

    result = []
    probabilities = probabilities.float()
    for batch_index, record in enumerate(records):
        scores, indexes = torch.topk(
            probabilities[batch_index].flatten(),
            min(max_detections, probabilities[batch_index].numel()),
        )
        class_count = probabilities.shape[-1]
        box_indexes = torch.div(indexes, class_count, rounding_mode="floor")
        labels = indexes.remainder(class_count)
        boxes = pred_boxes[batch_index][box_indexes].float()
        cx, cy, width, height = boxes.unbind(-1)
        boxes = torch.stack((cx - width / 2, cy - height / 2, cx + width / 2, cy + height / 2), -1)
        scale = boxes.new_tensor([record["width"], record["height"], record["width"], record["height"]])
        result.append(xyxy_predictions(
            record["id"], boxes.mul(scale).cpu(), scores.cpu(), labels.cpu(),
            category_ids, record["width"], record["height"],
        ))
    return result


def prepare_owl_queries(processor, model, categories, device, chunk_size=256):
    import torch

    backbone = model.owlv2 if hasattr(model, "owlv2") else model.owlvit
    ensemble = []
    for template in PROMPT_TEMPLATES:
        prompts = [canonicalize_label(template.format(category["name"])) for category in categories]
        embeddings = []
        for prompt_batch in batches(prompts, chunk_size):
            tokens = processor.tokenizer(
                list(prompt_batch),
                padding="max_length",
                max_length=16,
                truncation=True,
                return_tensors="pt",
            )
            tokens = {key: value.to(device) for key, value in tokens.items()}
            with torch.inference_mode():
                embeddings.append(backbone.get_text_features(**tokens))
        ensemble.append(torch.cat(embeddings, dim=0))
    return ensemble


def scenic_resize_with_pad(images, target_size):
    """PyTorch equivalent of Scenic ResizeWithPad(size, pad_value=0.0)."""
    import numpy as np
    import torch
    import torch.nn.functional as functional

    padded = []
    for image in images:
        width, height = image.size
        ratio = max(height / target_size, width / target_size)
        fit_height = int(min(height / ratio, target_size))
        fit_width = int(min(width / ratio, target_size))
        pixels = torch.from_numpy(np.asarray(image).copy()).permute(2, 0, 1).float() / 255.0
        resized = functional.interpolate(
            pixels.unsqueeze(0),
            size=(fit_height, fit_width),
            mode="bilinear",
            align_corners=False,
            antialias=False,
        )[0]
        canvas = torch.zeros((3, target_size, target_size), dtype=torch.float32)
        canvas[:, :fit_height, :fit_width] = resized
        padded.append(canvas.permute(1, 2, 0).numpy())
    return padded


def owl_topk_predictions(probabilities, pred_boxes, records, category_ids, max_detections):
    """Match Scenic: global top-k, unpad cxcywh, then convert to LVIS xywh."""
    import torch

    result = []
    probabilities = probabilities.float()
    for batch_index, record in enumerate(records):
        scores, indexes = torch.topk(
            probabilities[batch_index].flatten(),
            min(max_detections, probabilities[batch_index].numel()),
        )
        class_count = probabilities.shape[-1]
        box_indexes = torch.div(indexes, class_count, rounding_mode="floor")
        labels = indexes.remainder(class_count)
        boxes = pred_boxes[batch_index][box_indexes].float()
        width, height = int(record["width"]), int(record["height"])
        padded_size = max(width, height)
        fractions = boxes.new_tensor([
            width / padded_size, height / padded_size,
            width / padded_size, height / padded_size,
        ]) + 1e-6
        boxes = torch.clamp(boxes / fractions, 0.0, 1.0)
        rows = []
        for box, score, label in zip(boxes.cpu().tolist(), scores.cpu().tolist(), labels.cpu().tolist()):
            # The pinned Scenic evaluator explicitly skips label index 0 and
            # shifts all remaining zero-based labels by one.
            if int(label) == 0:
                continue
            cx, cy, box_width, box_height = [float(value) for value in box]
            if box_width <= 0 or box_height <= 0 or not math.isfinite(float(score)):
                continue
            rows.append({
                "image_id": int(record["id"]),
                "category_id": int(category_ids[int(label)]),
                "bbox": [
                    (cx - box_width / 2) * width,
                    (cy - box_height / 2) * height,
                    box_width * width,
                    box_height * height,
                ],
                "score": float(score),
            })
        result.append(rows)
    return result


def infer_owl_batch(processor, model, query_ensemble, images, records, category_ids, dtype, device, max_detections):
    import torch

    size = processor.image_processor.size
    if isinstance(size, (list, tuple)):
        target_size = int(size[0])
    else:
        target_size = int(size.get("height", size.get("shortest_edge")))
    padded_images = scenic_resize_with_pad(images, target_size)
    image_kwargs = {
        "images": padded_images,
        "return_tensors": "pt",
        "do_resize": False,
        "do_rescale": False,
    }
    if hasattr(model, "owlv2"):
        image_kwargs["do_pad"] = False
    inputs = processor.image_processor(**image_kwargs)
    pixel_values = inputs["pixel_values"].to(device=device, dtype=dtype)
    with torch.inference_mode():
        feature_map, _ = model.image_embedder(pixel_values)
        batch_size, grid_height, grid_width, hidden_dim = feature_map.shape
        image_features = feature_map.reshape(batch_size, grid_height * grid_width, hidden_dim)
        pred_boxes = model.box_predictor(image_features, feature_map)
        logits_sum = None
        for query_embeddings in query_ensemble:
            queries = query_embeddings.unsqueeze(0).expand(batch_size, -1, -1)
            logits, _ = model.class_predictor(image_features, queries, query_mask=None)
            logits_sum = logits.float() if logits_sum is None else logits_sum + logits.float()
        probabilities = torch.sigmoid(logits_sum / len(query_ensemble))
    return owl_topk_predictions(
        probabilities, pred_boxes, records, category_ids, max_detections
    )


def infer_dino_batch(processor, model, images, records, categories, category_ids, dtype, device, max_detections):
    import torch

    caption, spans = build_grounding_caption(categories)
    inputs = move_inputs(
        processor(images=images, text=[caption] * len(images), return_tensors="pt"),
        device,
        dtype,
    )
    with torch.inference_mode():
        outputs = model(**inputs)
    positive_map = grounding_positive_map(
        processor, caption, spans, outputs.logits.shape[-1], outputs.logits.device
    )
    probabilities = outputs.logits.float().sigmoid() @ positive_map.T
    return topk_xyxy(probabilities, outputs.pred_boxes, records, category_ids, max_detections)


def infer_omdet_batch(processor, model, images, records, categories, category_ids, dtype, device, max_detections):
    import torch

    labels = [dino_label(category["name"]) for category in categories]
    if len(images) == 1:
        inputs = processor(images[0], text=labels, return_tensors="pt")
        postprocess_labels: Any = labels
    else:
        postprocess_labels = [labels for _ in images]
        inputs = processor(images=images, text=postprocess_labels, return_tensors="pt")
    inputs = move_inputs(inputs, device, dtype)
    with torch.inference_mode():
        outputs = model(**inputs)
    target_sizes = [(record["height"], record["width"]) for record in records]
    results = processor.post_process_grounded_object_detection(
        outputs,
        text_labels=postprocess_labels,
        threshold=0.0,
        nms_threshold=0.5,
        target_sizes=target_sizes,
        max_num_det=max_detections,
    )
    label_to_index = {label: index for index, label in enumerate(labels)}
    predictions = []
    for record, result in zip(records, results):
        raw_labels = result.get("labels", result.get("classes"))
        if raw_labels is None:
            raise RuntimeError("OmDet 后处理未返回 labels/classes")
        if isinstance(raw_labels, torch.Tensor):
            label_indexes = raw_labels.cpu()
        else:
            label_indexes = torch.tensor([label_to_index[str(label)] for label in raw_labels])
        predictions.append(xyxy_predictions(
            record["id"], result["boxes"].cpu(), result["scores"].cpu(), label_indexes,
            category_ids, record["width"], record["height"],
        ))
    return predictions


def load_model(spec, model_path, dtype, device):
    from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

    processor_kwargs = {}
    if spec["family"] == "grounding_dino":
        processor_kwargs["use_fast"] = True
    elif spec["family"] == "omdet":
        processor_kwargs["use_fast"] = False
    processor = AutoProcessor.from_pretrained(model_path, local_files_only=True, **processor_kwargs)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(
        model_path,
        local_files_only=True,
        torch_dtype=dtype,
        trust_remote_code=False,
    )
    model.to(device).eval()
    return processor, model


def completed_image_ids(path: Path) -> set[int]:
    completed = set()
    if not path.exists():
        return completed
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            try:
                completed.add(int(json.loads(line)["image_id"]))
            except Exception as exc:
                raise RuntimeError(f"断点文件损坏: {path}:{line_number}") from exc
    return completed


def flatten_predictions(jsonl_path: Path, output_path: Path):
    temporary = output_path.with_suffix(".json.tmp")
    prediction_count = 0
    image_ids = set()
    duplicates = 0
    first = True
    with jsonl_path.open("r", encoding="utf-8") as source, temporary.open("w", encoding="utf-8") as target:
        target.write("[")
        for line in source:
            row = json.loads(line)
            image_id = int(row["image_id"])
            if image_id in image_ids:
                duplicates += 1
                continue
            image_ids.add(image_id)
            for prediction in row["predictions"]:
                if not first:
                    target.write(",")
                target.write(json.dumps(prediction, ensure_ascii=False, separators=(",", ":")))
                first = False
                prediction_count += 1
        target.write("]\n")
    temporary.replace(output_path)
    return prediction_count, len(image_ids), duplicates


def evaluate(dataset: EvaluationDataset, prediction_path: Path) -> dict[str, Any]:
    if dataset.key == "coco2017_val":
        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval

        ground_truth = COCO()
        ground_truth.dataset = dataset.annotation
        ground_truth.createIndex()
        detections = ground_truth.loadRes(str(prediction_path))
        evaluator = COCOeval(ground_truth, detections, "bbox")
        evaluator.evaluate()
        evaluator.accumulate()
        evaluator.summarize()
        return {
            "metric_implementation": "pycocotools.COCOeval",
            "metrics": dict(zip(COCO_STATS, [float(value) for value in evaluator.stats])),
        }

    import numpy as np
    if "float" not in np.__dict__:
        setattr(np, "float", float)
    from lvis import LVIS, LVISEval, LVISResults

    with tempfile.TemporaryDirectory(prefix="official_lvis_eval_") as temporary_dir:
        with zipfile.ZipFile(dataset.annotation_path) as archive:
            annotation_path = Path(archive.extract(dataset.spec["annotation_member"], temporary_dir))
        ground_truth = LVIS(str(annotation_path))
        detections = LVISResults(ground_truth, str(prediction_path), max_dets=300)
        evaluator = LVISEval(ground_truth, detections, "bbox")
        evaluator.run()
        evaluator.print_results()
        metrics = {key: float(value) for key, value in evaluator.get_results().items()}
    return {"metric_implementation": "lvis-api LVISEval", "metrics": metrics}


def package_versions() -> dict[str, str]:
    import PIL
    import numpy
    import torch
    import transformers

    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "numpy": numpy.__version__,
        "Pillow": PIL.__version__,
    }


def local_artifact(model_path: Path) -> dict[str, Any]:
    metadata_root = model_path / ".cache" / "huggingface" / "download"
    config_metadata = metadata_root / "config.json.metadata"
    revision = None
    if config_metadata.is_file():
        lines = config_metadata.read_text(encoding="utf-8").splitlines()
        revision = lines[0] if lines else None
    for filename in ("model.safetensors", "pytorch_model.bin"):
        weight_path = model_path / filename
        if weight_path.is_file():
            weight_metadata = metadata_root / f"{filename}.metadata"
            etag = None
            if weight_metadata.is_file():
                lines = weight_metadata.read_text(encoding="utf-8").splitlines()
                etag = lines[1] if len(lines) > 1 else None
            return {
                "model_path": str(model_path),
                "hf_revision_detected": revision,
                "weight_file": filename,
                "weight_bytes": weight_path.stat().st_size,
                "weight_etag": etag,
            }
    raise FileNotFoundError(f"模型权重不存在: {model_path}")


def compare_reference(spec, metrics):
    if spec["fidelity"] == "reference_baseline":
        return {"classification": "not_officially_comparable", "reason": "官方未发布完整评估入口"}
    expected = spec.get("official_reference", {}).get("metrics_percent", {})
    if not expected:
        return {"classification": "manual_reference_required"}
    observed = metrics["metrics"]
    deltas = {}
    for name, reference in expected.items():
        if name in observed:
            deltas[name] = observed[name] * 100.0 - float(reference)
    if not deltas:
        return {"classification": "metric_name_not_found"}
    maximum = max(abs(value) for value in deltas.values())
    return {
        "classification": "matches_reference" if maximum <= 1.0 else "does_not_match_reference",
        "tolerance_absolute_points": 1.0,
        "delta_points": deltas,
    }


def run_one(spec, data, args):
    import torch

    model_path = args.model_root / spec["repo_id"]
    output_dir = args.output_root / spec["key"]
    output_dir.mkdir(parents=True, exist_ok=True)
    status_path = output_dir / "status.json"
    jsonl_path = output_dir / "predictions.jsonl"
    prediction_path = output_dir / "predictions.json"
    metrics_path = output_dir / "metrics.json"
    fingerprint = protocol_fingerprint(spec)

    if status_path.is_file():
        prior = read_json(status_path)
        if prior.get("state") == "complete" and not args.max_images and metrics_path.is_file():
            print(f"[{spec['key']}] 已按同一协议完整完成，跳过。", flush=True)
            return
        if jsonl_path.is_file() and prior.get("protocol_fingerprint") != fingerprint:
            raise RuntimeError("已有断点的协议清单已改变；请人工移动该模型输出目录后重新运行")
        expected_runtime = {
            "dtype": args.dtype,
            "batch_size": args.batch_size,
            "max_detections": args.max_detections,
        }
        mismatches = {
            key: (prior.get(key), value)
            for key, value in expected_runtime.items()
            if jsonl_path.is_file() and prior.get(key) is not None and prior.get(key) != value
        }
        if mismatches:
            raise RuntimeError(f"已有断点的运行参数不一致: {mismatches}")

    dataset_spec = data["datasets"][spec["dataset"]]
    dataset = EvaluationDataset(args.dataset_root, spec["dataset"], dataset_spec)
    try:
        records = dataset.images[: args.max_images or None]
        done = completed_image_ids(jsonl_path)
        selected_ids = {int(record["id"]) for record in records}
        completed = done & selected_ids
        pending = [record for record in records if int(record["id"]) not in completed]
        artifact = local_artifact(model_path)
        if artifact["hf_revision_detected"] != spec["hf_revision"]:
            raise RuntimeError(
                f"本地 HF revision {artifact['hf_revision_detected']} 与清单 {spec['hf_revision']} 不一致"
            )
        category_ids = [int(category["id"]) for category in dataset.categories]
        dtype = torch.float32 if args.dtype == "float32" else torch.float16
        status = {
            "state": "running",
            "model": spec,
            "protocol_fingerprint": fingerprint,
            "artifact": artifact,
            "dataset": {
                "key": spec["dataset"],
                "spec": dataset_spec,
                "annotation_sha256": sha256_file(dataset.annotation_path),
                "detected_images": len(dataset.images),
                "detected_categories": len(dataset.categories),
            },
            "selected_image_count": len(records),
            "already_completed": len(completed),
            "started_at": utc_now(),
            "host": socket.gethostname(),
            "device": args.device,
            "dtype": args.dtype,
            "batch_size": args.batch_size,
            "max_detections": args.max_detections,
            "versions": package_versions(),
        }
        atomic_json(status_path, status)
        print(f"[{spec['key']}] {spec['fidelity']}，加载 {model_path}", flush=True)
        processor, model = load_model(spec, model_path, dtype, args.device)
        owl_queries = None
        if spec["runner"] == "scenic_owl_lvis":
            print(f"[{spec['key']}] 预计算 Scenic 七模板 × {len(dataset.categories)} 类文本嵌入", flush=True)
            owl_queries = prepare_owl_queries(processor, model, dataset.categories, args.device)
        if args.device.startswith("cuda"):
            torch.cuda.reset_peak_memory_stats()

        started = time.monotonic()
        processed = len(completed)
        with jsonl_path.open("a", encoding="utf-8", buffering=1) as stream:
            for record_batch in batches(pending, args.batch_size):
                images = [dataset.load_image(record) for record in record_batch]
                if spec["runner"] == "scenic_owl_lvis":
                    prediction_batches = infer_owl_batch(
                        processor, model, owl_queries, images, record_batch, category_ids,
                        dtype, args.device, args.max_detections,
                    )
                elif spec["runner"] == "grounding_dino_coco":
                    prediction_batches = infer_dino_batch(
                        processor, model, images, record_batch, dataset.categories, category_ids,
                        dtype, args.device, args.max_detections,
                    )
                elif spec["runner"] == "omdet_coco_reference":
                    prediction_batches = infer_omdet_batch(
                        processor, model, images, record_batch, dataset.categories, category_ids,
                        dtype, args.device, args.max_detections,
                    )
                else:
                    raise RuntimeError(f"未实现 runner: {spec['runner']}")
                for record, predictions in zip(record_batch, prediction_batches):
                    stream.write(json.dumps({
                        "image_id": int(record["id"]), "predictions": predictions,
                    }, ensure_ascii=False, separators=(",", ":")) + "\n")
                    processed += 1
                if processed % args.progress_every < len(record_batch):
                    elapsed = time.monotonic() - started
                    rate = max(processed - len(completed), 1) / max(elapsed, 1e-6)
                    remaining = (len(records) - processed) / max(rate, 1e-6)
                    print(
                        f"[{spec['key']}] {processed}/{len(records)}, {rate:.2f} images/s, "
                        f"ETA {remaining / 3600:.2f} h",
                        flush=True,
                    )

        prediction_count, image_count, duplicates = flatten_predictions(jsonl_path, prediction_path)
        partial = bool(args.max_images and args.max_images < len(dataset.images))
        metrics = None
        if not partial:
            metrics = evaluate(dataset, prediction_path)
            metrics.update({
                "model_key": spec["key"],
                "fidelity": spec["fidelity"],
                "protocol_fingerprint": fingerprint,
            })
            metrics["official_reference_comparison"] = compare_reference(spec, metrics)
            atomic_json(metrics_path, metrics)
        status.update({
            "state": "partial_complete" if partial else "complete",
            "completed_at": utc_now(),
            "processed_image_count": processed,
            "unique_prediction_image_count": image_count,
            "duplicate_rows_ignored": duplicates,
            "prediction_count": prediction_count,
            "elapsed_seconds_this_invocation": time.monotonic() - started,
            "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated()) if args.device.startswith("cuda") else 0,
            "metrics_written": metrics is not None,
        })
        atomic_json(status_path, status)
        del model, processor, owl_queries
        if args.device.startswith("cuda"):
            torch.cuda.empty_cache()
        print(f"[{spec['key']}] 完成: {output_dir}", flush=True)
    finally:
        dataset.close()


def validate_inputs(data, model_root, dataset_root):
    detected = {}
    for dataset_key, spec in data["datasets"].items():
        annotation_path = dataset_root / spec["annotation_archive"]
        if not annotation_path.is_file():
            raise FileNotFoundError(annotation_path)
        annotation = read_json_member(annotation_path, spec["annotation_member"])
        if len(annotation["images"]) != spec["images"] or len(annotation["categories"]) != spec["categories"]:
            raise RuntimeError(f"{dataset_key} 数量与清单不符")
        for relative_path in spec["image_archives"]:
            if not (dataset_root / relative_path).is_file():
                raise FileNotFoundError(dataset_root / relative_path)
        detected[dataset_key] = {"images": len(annotation["images"]), "categories": len(annotation["categories"])}
    for spec in data["models"]:
        model_path = model_root / spec["repo_id"]
        for filename in ("config.json", "preprocessor_config.json"):
            if not (model_path / filename).is_file():
                raise FileNotFoundError(model_path / filename)
        artifact = local_artifact(model_path)
        if artifact["hf_revision_detected"] != spec["hf_revision"]:
            raise RuntimeError(f"{spec['key']} revision 与 models.json 不一致")
    return detected


def summarize(data, output_root):
    rows = []
    for spec in data["models"]:
        status_path = output_root / spec["key"] / "status.json"
        metrics_path = output_root / spec["key"] / "metrics.json"
        if status_path.is_file():
            state = read_json(status_path).get("state", "unknown")
        elif spec["key"].startswith("grounding-dino-") and spec.get("legacy_result", {}).get("classification") == "matches_reference":
            state = "verified_existing"
        else:
            state = "not_started"
        row = {
            "sequence": spec["sequence"],
            "key": spec["key"],
            "repo_id": spec["repo_id"],
            "dataset": spec["dataset"],
            "fidelity": spec["fidelity"],
            "state": state,
            "official_reference": spec.get("official_reference"),
            "known_differences": spec.get("known_differences", []),
        }
        if metrics_path.is_file():
            metrics = read_json(metrics_path)
            row["metrics"] = metrics.get("metrics")
            row["comparison"] = metrics.get("official_reference_comparison")
        elif state == "verified_existing":
            row["legacy_result"] = spec["legacy_result"]
        rows.append(row)
    report = {
        "generated_at": utc_now(),
        "suite": data["suite"],
        "counts_by_fidelity": dict(Counter(row["fidelity"] for row in rows)),
        "counts_by_state": dict(Counter(row["state"] for row in rows)),
        "models": rows,
    }
    atomic_json(output_root / "audit_summary.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


def selected_models(value, data):
    index = model_index(data)
    if value == "all":
        return data["models"]
    selected = []
    for name in value.split(","):
        name = name.strip()
        if name not in index:
            raise ValueError(f"未知模型: {name}")
        if index[name] not in selected:
            selected.append(index[name])
    return selected


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", default="all")
    parser.add_argument("--model-root", type=Path, default=Path("/data2/gaoyonghan/gpu/model"))
    parser.add_argument("--dataset-root", type=Path, default=Path("/data2/gaoyonghan/gpu/dataset"))
    parser.add_argument("--output-root", type=Path, default=Path("/home/ubuntu/gaoyonghan/gpu/results/zero_shot_official"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=("float32", "float16"), default="float32")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-images", type=int, default=0)
    parser.add_argument("--max-detections", type=int, default=300)
    parser.add_argument("--progress-every", type=int, default=25)
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--summarize", action="store_true")
    parser.add_argument("--list-models", action="store_true")
    parser.add_argument("--describe")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    data = manifest()
    if args.list_models:
        for spec in data["models"]:
            print(f"{spec['sequence']}\t{spec['key']}\t{spec['fidelity']}\t{spec['dataset']}")
        return 0
    if args.describe:
        spec = model_index(data).get(args.describe)
        if spec is None:
            raise ValueError(f"未知模型: {args.describe}")
        print(json.dumps(spec, ensure_ascii=False, indent=2))
        return 0
    if args.preflight:
        import torch
        import transformers
        if transformers.__version__ != "4.50.0":
            raise RuntimeError(f"需要 transformers 4.50.0，当前 {transformers.__version__}")
        if not torch.cuda.is_available():
            raise RuntimeError("PyTorch 未检测到 CUDA")
        counts = validate_inputs(data, args.model_root, args.dataset_root)
        print(json.dumps({"status": "ready", "datasets": counts}, ensure_ascii=False, indent=2))
        return 0
    if args.summarize:
        summarize(data, args.output_root)
        return 0
    if args.batch_size < 1 or args.max_images < 0 or args.max_detections < 1:
        raise ValueError("batch-size/max-detections 必须大于 0，max-images 不能为负数")
    if args.max_detections != 300:
        raise ValueError("本评估包固定官方 max_detections=300，不允许改写")

    failures = 0
    for spec in selected_models(args.models, data):
        try:
            run_one(spec, data, args)
        except Exception as exc:
            failures += 1
            output_dir = args.output_root / spec["key"]
            output_dir.mkdir(parents=True, exist_ok=True)
            status_path = output_dir / "status.json"
            status = read_json(status_path) if status_path.is_file() else {}
            status.update({
                "state": "failed",
                "model": spec,
                "failed_at": utc_now(),
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
            })
            atomic_json(status_path, status)
            print(f"[{spec['key']}] 失败: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
