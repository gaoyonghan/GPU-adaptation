#!/usr/bin/env python3
"""Deterministic B01 zero-shot evaluation on locally pinned datasets.

The coordinator launches one isolated worker per model. The B01 run is an
engineering baseline; formal NVIDIA/domestic-chip comparisons must reuse the
same sample manifest, prompts, preprocessing revision and precision.
"""

from __future__ import annotations
from model_paths import legacy as _legacy, legacy_str as _legacy_str, relative as _relative, revision as _revision, weight as _weight, result as _result

import argparse
import gc
import hashlib
import io
import json
import platform
import random
import subprocess
import sys
import tempfile
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
MODEL_ROOT = ROOT / "models"
RESULT_ROOT = ROOT / "results" / "b01_zero_shot"
MANIFEST_ROOT = ROOT / "evaluation_audit" / "manifests" / "b01"
CACHE_ROOT = ROOT / "evaluation_audit" / "cache" / "b01"
LEDGER = ROOT / "负责的模型.xlsx"
DEFAULT_DATASETS = ("cifar10", "cifar100", "dtd", "eurosat")
SEED = 20260828
DATASET_LENGTHS = {"cifar10": 10000, "cifar100": 10000, "dtd": 1880, "eurosat": 27000}
WORKER_PYTHON_OVERRIDES = {
    "OFA-Sys/chinese-clip-vit-base-patch16": _legacy('/home/dell/anaconda3/envs/hengxiang_stage2_clean/bin/python'),
    "OFA-Sys/chinese-clip-vit-large-patch14": _legacy('/home/dell/anaconda3/envs/hengxiang_stage2_clean/bin/python'),
    "OFA-Sys/chinese-clip-vit-large-patch14-336px": _legacy('/home/dell/anaconda3/envs/hengxiang_stage2_clean/bin/python'),
    "OFA-Sys/chinese-clip-vit-huge-patch14": _legacy('/home/dell/anaconda3/envs/hengxiang_stage2_clean/bin/python'),
}
DEFAULT_WORKER_PYTHON = _legacy('/home/dell/anaconda3/envs/hengxiang_stage2_clean/bin/python')

CIFAR10_LABELS = (
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
)
CIFAR100_LABELS = (
    "apple", "aquarium fish", "baby", "bear", "beaver", "bed", "bee", "beetle",
    "bicycle", "bottle", "bowl", "boy", "bridge", "bus", "butterfly", "camel",
    "can", "castle", "caterpillar", "cattle", "chair", "chimpanzee", "clock", "cloud",
    "cockroach", "couch", "crab", "crocodile", "cup", "dinosaur", "dolphin", "elephant",
    "flatfish", "forest", "fox", "girl", "hamster", "house", "kangaroo", "keyboard",
    "lamp", "lawn mower", "leopard", "lion", "lizard", "lobster", "man", "maple tree",
    "motorcycle", "mountain", "mouse", "mushroom", "oak tree", "orange", "orchid", "otter",
    "palm tree", "pear", "pickup truck", "pine tree", "plain", "plate", "poppy", "porcupine",
    "possum", "rabbit", "raccoon", "ray", "road", "rocket", "rose", "sea", "seal", "shark",
    "shrew", "skunk", "skyscraper", "snail", "snake", "spider", "squirrel", "streetcar",
    "sunflower", "sweet pepper", "table", "tank", "telephone", "television", "tiger", "tractor",
    "train", "trout", "tulip", "turtle", "wardrobe", "whale", "willow tree", "wolf", "woman", "worm",
)
DTD_LABELS = (
    "banded", "blotchy", "braided", "bubbly", "bumpy", "chequered", "cobwebbed", "cracked",
    "crosshatched", "crystalline", "dotted", "fibrous", "flecked", "freckled", "frilly", "gauzy",
    "grid", "grooved", "honeycombed", "interlaced", "knitted", "lacelike", "lined", "marbled",
    "matted", "meshed", "paisley", "perforated", "pitted", "pleated", "polka-dotted", "porous",
    "potholed", "scaly", "smeared", "spiralled", "sprinkled", "stained", "stratified", "striped",
    "studded", "swirly", "veined", "waffled", "woven", "wrinkled", "zigzagged",
)
EUROSAT_LABELS = (
    "annual crop land", "forest", "herbaceous vegetation land", "highway", "industrial buildings",
    "pasture land", "permanent crop land", "residential buildings", "river", "sea or lake",
)

DATASET_SPECS = {
    "cifar10": {
        "labels": CIFAR10_LABELS,
        "templates": ("a photo of a {}.", "a blurry photo of a {}.", "a black and white photo of a {}.", "a close-up photo of a {}.", "a photo of the {}."),
        "source": "torchvision CIFAR-10 test", "revision": "local-python-batches",
    },
    "cifar100": {
        "labels": CIFAR100_LABELS,
        "templates": ("a photo of a {}.", "a blurry photo of a {}.", "a black and white photo of a {}.", "a close-up photo of a {}.", "a photo of the {}."),
        "source": "uoft-cs/cifar100 test", "revision": "aadb3af77e9048adbea6b47c21a81e47dd092ae5",
    },
    "dtd": {
        "labels": DTD_LABELS,
        "templates": ("a photo of a {} texture.", "a photo of a {} pattern.", "a {} texture.", "a close-up photo of a {} surface."),
        "source": "tanganke/dtd test", "revision": "d2afa97d9f335b1a6b3b09c637aef667f98f966e",
    },
    "eurosat": {
        "labels": EUROSAT_LABELS,
        "templates": ("a centered satellite photo of {}.", "a satellite photo of {}.", "an aerial photo of {}."),
        "source": "yasimed/EuroSAT single split", "revision": "7139c055788f6974202a4eeb2a177a4356fa225b",
    },
}


def now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", action="append", help="Repeat for multiple models")
    parser.add_argument("--batch", action="append", help="Ledger batch prefix, e.g. P0")
    parser.add_argument("--datasets", nargs="+", choices=DEFAULT_DATASETS, default=list(DEFAULT_DATASETS))
    parser.add_argument("--max-samples", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--timeout-seconds", type=int, default=3600)
    parser.add_argument("--worker-model-id", help=argparse.SUPPRESS)
    parser.add_argument("--worker-backend", help=argparse.SUPPRESS)
    parser.add_argument("--worker-output", type=Path, help=argparse.SUPPRESS)
    return parser.parse_args()


def ledger_rows() -> list[dict[str, Any]]:
    from openpyxl import load_workbook

    workbook = load_workbook(LEDGER, read_only=True, data_only=True)
    sheet = workbook["模型台账"]
    rows = sheet.iter_rows(values_only=True)
    headers = [str(value or "") for value in next(rows)]
    return [dict(zip(headers, values)) for values in rows]


def backend_for(row: dict[str, Any]) -> str:
    overrides = {
        "Marqo/marqo-fashionCLIP": "OpenCLIP",
        "Marqo/marqo-fashionSigLIP": "OpenCLIP",
        # The frozen stage-2 audit loaded and inferred this standard CLIP export
        # successfully; the spreadsheet predates that backend verification.
        "zer0int/LongCLIP-SAE-ViT-L-14": "Transformers AutoModel/AutoProcessor",
    }
    return overrides.get(str(row["模型名称"]), str(row["建议加载后端"]))


def model_dir(model_id: str) -> Path:
    return MODEL_ROOT.joinpath(*model_id.split("/"))


def run_tag(args: argparse.Namespace) -> str:
    return f"{'-'.join(args.datasets)}_fp16_seed{args.seed}_n{args.max_samples}"


def output_path(model_id: str, args: argparse.Namespace) -> Path:
    return RESULT_ROOT.joinpath(*model_id.split("/"), run_tag(args), "summary.json")


def selected_indices(name: str, length: int, maximum: int, seed: int) -> list[int]:
    count = length if maximum == -1 else min(length, maximum)
    values = list(range(length))
    random.Random(f"{seed}:{name}").shuffle(values)
    values = sorted(values[:count])
    MANIFEST_ROOT.mkdir(parents=True, exist_ok=True)
    path = MANIFEST_ROOT / f"{name}_seed{seed}_n{count}.json"
    payload = {
        "dataset": name, "dataset_length": length, "sample_count": count, "seed": seed,
        "selection": "sort(first N of deterministic shuffled indices)", "indices": values,
    }
    encoded = json.dumps(payload, ensure_ascii=False, indent=2).encode()
    if path.exists() and path.read_bytes() != encoded:
        raise RuntimeError(f"样本manifest冲突: {path}")
    path.write_bytes(encoded)
    return values


class B01Dataset:
    def __init__(self, name: str):
        self.name = name
        self.table = None
        self.dataset = None
        if name == "cifar10":
            from torchvision.datasets import CIFAR10

            self.dataset = CIFAR10(root=ROOT / "datasets" / "cifar10", train=False, download=False)
        else:
            import pyarrow.parquet as pq

            paths = {
                "cifar100": ROOT / "datasets/evaluation/cifar100_hf/cifar100/test-00000-of-00001.parquet",
                "dtd": ROOT / "datasets/evaluation/dtd_hf/data/test-00000-of-00001.parquet",
                "eurosat": ROOT / "datasets/evaluation/eurosat_hf/data/train-00000-of-00001-8924b0cc377e5bcb.parquet",
            }
            self.image_column = "img" if name == "cifar100" else "image"
            self.label_column = "fine_label" if name == "cifar100" else "label"
            self.table = pq.read_table(paths[name], columns=[self.image_column, self.label_column], memory_map=True)

    def __len__(self) -> int:
        return len(self.dataset) if self.dataset is not None else len(self.table)

    def __getitem__(self, index: int):
        from PIL import Image

        if self.dataset is not None:
            return self.dataset[index]
        image_value = self.table[self.image_column][index].as_py()
        if isinstance(image_value, dict):
            image = (Image.open(io.BytesIO(image_value["bytes"])) if image_value.get("bytes") is not None
                     else Image.open(image_value["path"])).convert("RGB")
        elif isinstance(image_value, bytes):
            image = Image.open(io.BytesIO(image_value)).convert("RGB")
        else:
            raise TypeError(f"未知图像字段类型: {type(image_value).__name__}")
        return image, int(self.table[self.label_column][index].as_py())


class CachedB01Dataset:
    def __init__(self, manifest: Path):
        self.records = json.loads(manifest.read_text(encoding="utf-8"))["samples"]

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int):
        from PIL import Image

        record = self.records[index]
        return Image.open(ROOT / record["path"]).convert("RGB"), int(record["label"]), int(record["dataset_index"])


def cache_manifest_path(name: str, count: int, seed: int) -> Path:
    return CACHE_ROOT / name / f"seed{seed}_n{count}.json"


def ensure_sample_cache(name: str, maximum: int, seed: int) -> Path:
    length = DATASET_LENGTHS[name]
    count = length if maximum == -1 else min(length, maximum)
    manifest = cache_manifest_path(name, count, seed)
    if manifest.is_file():
        payload = json.loads(manifest.read_text(encoding="utf-8"))
        if len(payload.get("samples", [])) == count and all((ROOT / item["path"]).is_file() for item in payload["samples"]):
            return manifest
    dataset = B01Dataset(name)
    indices = selected_indices(name, len(dataset), maximum, seed)
    directory = manifest.parent / f"seed{seed}_n{count}_images"
    directory.mkdir(parents=True, exist_ok=True)
    records = []
    for index in indices:
        image, label = dataset[index]
        path = directory / f"{index:06d}.png"
        if not path.is_file():
            image.save(path, format="PNG")
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        records.append({
            "dataset_index": index, "label": int(label),
            "path": str(_relative(path,ROOT)), "sha256": digest,
        })
    manifest.write_text(json.dumps({
        "dataset": name, "source_length": len(dataset), "sample_count": count,
        "seed": seed, "encoding": "lossless PNG decoded from pinned source",
        "samples": records,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def batches(dataset: B01Dataset, indices: list[int], size: int):
    for start in range(0, len(indices), size):
        current = indices[start:start + size]
        samples = [dataset[index] for index in current]
        source_indices = [sample[2] if len(sample) == 3 else index for sample, index in zip(samples, current)]
        yield source_indices, [sample[0] for sample in samples], [sample[1] for sample in samples]


def tensor_from_feature(value: Any):
    import torch

    if isinstance(value, torch.Tensor):
        return value
    for key in ("pooler_output", "text_embeds", "image_embeds", "last_hidden_state"):
        candidate = getattr(value, key, None)
        if isinstance(candidate, torch.Tensor):
            return candidate[:, 0] if key == "last_hidden_state" else candidate
    if isinstance(value, (tuple, list)) and value and isinstance(value[0], torch.Tensor):
        return value[0]
    raise TypeError(f"无法提取特征张量: {type(value).__name__}")


class TransformersAdapter:
    name = "Transformers AutoModel/AutoProcessor"

    def __init__(self, model_id: str):
        import torch
        from transformers import AutoModel, AutoProcessor

        directory = model_dir(model_id)
        self.processor = AutoProcessor.from_pretrained(directory, local_files_only=True, trust_remote_code=False)
        tokenizer = getattr(self.processor, "tokenizer", None)
        model_max_length = getattr(tokenizer, "model_max_length", None)
        # Variable-length encoders (notably AltCLIP/XLM-R) should not be padded
        # to their 512-token ceiling for short class prompts.
        self.text_padding = True
        self.model = AutoModel.from_pretrained(
            directory, local_files_only=True, trust_remote_code=False,
            torch_dtype=torch.float16, low_cpu_mem_usage=True,
        ).eval().cuda()
        self.model_type = getattr(self.model.config, "model_type", "")
        if "siglip" in self.model_type:
            self.text_padding = "max_length"
            self.text_max_length = int(getattr(self.model.config.text_config, "max_position_embeddings", 64))
        else:
            self.text_max_length = None

    def text_features(self, prompts: list[str]):
        import torch

        text_kwargs = {"return_tensors": "pt", "padding": self.text_padding, "truncation": True}
        if self.text_max_length is not None:
            text_kwargs["max_length"] = self.text_max_length
        inputs = dict(self.processor(text=prompts, **text_kwargs))
        inputs = {key: value.cuda(non_blocking=True) for key, value in inputs.items() if hasattr(value, "cuda")}
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
            if self.model_type == "chinese_clip":
                # Transformers 4.50+ constructs ChineseCLIPTextModel without a
                # pooler while get_text_features still reads pooler_output.
                # The model documentation defines the embedding as projected
                # final [CLS], so compute that path explicitly.
                output = self.model.text_model(**inputs)
                features = self.model.text_projection(output.last_hidden_state[:, 0])
            else:
                features = tensor_from_feature(self.model.get_text_features(**inputs))
        return torch.nn.functional.normalize(features.float(), dim=-1)

    def image_features(self, images: list[Any]):
        import torch

        inputs = dict(self.processor(images=images, return_tensors="pt"))
        inputs = {key: value.cuda(non_blocking=True) for key, value in inputs.items() if hasattr(value, "cuda")}
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
            features = tensor_from_feature(self.model.get_image_features(**inputs))
        return torch.nn.functional.normalize(features.float(), dim=-1)


class OpenCLIPAdapter:
    name = "OpenCLIP local config/checkpoint"

    def __init__(self, model_id: str):
        import open_clip
        if str(ROOT) not in sys.path:
            sys.path.insert(0, str(ROOT))
        from validate_stage2_models import choose_openclip_checkpoint, patch_openclip_config

        directory = model_dir(model_id)
        config_path = directory / "open_clip_config.json"
        if not config_path.is_file():
            raise RuntimeError("缺少 open_clip_config.json")
        self.temporary = tempfile.TemporaryDirectory(prefix="b01_openclip_")
        model_name, preprocess_cfg = patch_openclip_config(config_path, model_id, Path(self.temporary.name))
        open_clip.add_model_config(Path(self.temporary.name))
        checkpoint = choose_openclip_checkpoint(directory)
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=str(checkpoint), precision="fp16", device="cuda", pretrained_hf=False,
            image_mean=tuple(preprocess_cfg["mean"]) if preprocess_cfg.get("mean") else None,
            image_std=tuple(preprocess_cfg["std"]) if preprocess_cfg.get("std") else None,
            image_interpolation=preprocess_cfg.get("interpolation"),
            image_resize_mode=preprocess_cfg.get("resize_mode"),
        )
        self.model.eval()
        self.tokenizer = open_clip.get_tokenizer(model_name)

    def text_features(self, prompts: list[str]):
        import torch

        tokens = self.tokenizer(prompts).cuda(non_blocking=True)
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
            features = self.model.encode_text(tokens)
        return torch.nn.functional.normalize(features.float(), dim=-1)

    def image_features(self, images: list[Any]):
        import torch

        pixels = torch.stack([self.preprocess(image) for image in images]).cuda(non_blocking=True)
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
            features = self.model.encode_image(pixels)
        return torch.nn.functional.normalize(features.float(), dim=-1)


def load_adapter(model_id: str, backend: str):
    if backend == "Transformers AutoModel/AutoProcessor":
        return TransformersAdapter(model_id)
    if backend == "OpenCLIP":
        return OpenCLIPAdapter(model_id)
    raise RuntimeError(f"B01尚未适配此后端: {backend}")


def class_features(adapter: Any, labels: tuple[str, ...], templates: tuple[str, ...]):
    import torch

    prompts = [template.format(label) for label in labels for template in templates]
    features = adapter.text_features(prompts)
    features = features.reshape(len(labels), len(templates), -1).mean(dim=1)
    return torch.nn.functional.normalize(features, dim=-1), prompts


def evaluate_dataset(adapter: Any, name: str, args: argparse.Namespace) -> dict[str, Any]:
    import torch

    spec = DATASET_SPECS[name]
    length = DATASET_LENGTHS[name]
    count = length if args.max_samples == -1 else min(length, args.max_samples)
    cache_manifest = cache_manifest_path(name, count, args.seed)
    if cache_manifest.is_file():
        dataset = CachedB01Dataset(cache_manifest)
        indices = list(range(len(dataset)))
        source_indices = [record["dataset_index"] for record in dataset.records]
    else:
        dataset = B01Dataset(name)
        indices = selected_indices(name, len(dataset), args.max_samples, args.seed)
        source_indices = indices
    text_features, prompts = class_features(adapter, spec["labels"], spec["templates"])
    correct1 = correct5 = 0
    predictions = []
    elapsed = 0.0
    torch.cuda.reset_peak_memory_stats()
    for batch_indices, images, targets in batches(dataset, indices, args.batch_size):
        torch.cuda.synchronize()
        started = time.perf_counter()
        image_features = adapter.image_features(images)
        logits = image_features @ text_features.T
        torch.cuda.synchronize()
        elapsed += time.perf_counter() - started
        topk = logits.topk(min(5, len(spec["labels"])), dim=1).indices.cpu()
        target_tensor = torch.tensor(targets)
        correct1 += int((topk[:, 0] == target_tensor).sum())
        correct5 += int((topk == target_tensor[:, None]).any(dim=1).sum())
        for index, target, guesses in zip(batch_indices, targets, topk.tolist()):
            predictions.append({"dataset_index": index, "target": target, "prediction": guesses[0], "top5": guesses})
    count = len(indices)
    index_hash = hashlib.sha256(json.dumps(source_indices, separators=(",", ":")).encode()).hexdigest()
    return {
        "dataset": name, "source": spec["source"], "revision": spec["revision"],
        "dataset_length": length, "sample_count": count,
        "sample_indices_sha256": index_hash, "labels": list(spec["labels"]),
        "sample_cache_manifest": str(_relative(cache_manifest,ROOT)) if cache_manifest.is_file() else None,
        "prompt_templates": list(spec["templates"]), "prompt_count": len(prompts),
        "top1_accuracy_percent": round(100 * correct1 / count, 4),
        "top5_accuracy_percent": round(100 * correct5 / count, 4),
        "timed_inference_seconds": round(elapsed, 4),
        "images_per_second": round(count / elapsed, 3),
        "peak_gpu_memory_mb": round(torch.cuda.max_memory_allocated() / 1024**2, 2),
        "predictions": predictions,
    }


def worker(args: argparse.Namespace) -> int:
    import torch
    import torchvision
    import transformers

    output = args.worker_output
    assert output is not None and args.worker_model_id and args.worker_backend
    result: dict[str, Any] = {
        "purpose": "B01_engineering_baseline_not_final_5_percent_acceptance",
        "model_id": args.worker_model_id, "backend": args.worker_backend,
        "status": "failed", "started_at": now(), "datasets": [],
    }
    adapter = None
    try:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA不可用")
        torch.manual_seed(args.seed)
        adapter = load_adapter(args.worker_model_id, args.worker_backend)
        result["actual_backend"] = adapter.name
        receipt = model_dir(args.worker_model_id) / "download_receipt.json"
        if receipt.is_file():
            receipt_data = json.loads(receipt.read_text(encoding="utf-8"))
            result["model_revision"] = receipt_data.get("revision") or receipt_data.get("resolved_revision")
        for name in args.datasets:
            print(f"DATASET {args.worker_model_id} {name}", flush=True)
            result["datasets"].append(evaluate_dataset(adapter, name, args))
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        result["status"] = "passed"
        result["environment"] = {
            "python": platform.python_version(), "torch": torch.__version__,
            "torchvision": torchvision.__version__, "transformers": transformers.__version__,
            "cuda_runtime": torch.version.cuda, "gpu": torch.cuda.get_device_name(0),
        }
    except Exception as error:
        result["error_type"] = type(error).__name__
        result["error"] = str(error)
        result["traceback"] = traceback.format_exc(limit=16)
    finally:
        result["finished_at"] = now()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        del adapter
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    print(json.dumps({"model_id": args.worker_model_id, "status": result["status"], "output": str(output)}), flush=True)
    return 0 if result["status"] == "passed" else 1


def coordinator(args: argparse.Namespace) -> int:
    rows = ledger_rows()
    requested_ids = set(args.model_id or [])
    requested_batches = tuple(args.batch or ())
    selected = [row for row in rows
                if (not requested_ids or str(row["模型名称"]) in requested_ids)
                and (not requested_batches or any(str(row["测试批次"]).startswith(value) for value in requested_batches))]
    if not selected:
        raise SystemExit("没有匹配的模型")
    for name in args.datasets:
        manifest = ensure_sample_cache(name, args.max_samples, args.seed)
        print(f"CACHE {name} {_relative(manifest,ROOT)}", flush=True)
    failures = 0
    for position, row in enumerate(selected, 1):
        model_id = str(row["模型名称"])
        backend = backend_for(row)
        output = output_path(model_id, args)
        if output.exists() and not args.force:
            old = json.loads(output.read_text(encoding="utf-8"))
            if old.get("status") == "passed":
                print(f"[{position}/{len(selected)}] SKIP {model_id}", flush=True)
                continue
        if backend not in {"Transformers AutoModel/AutoProcessor", "OpenCLIP"}:
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(json.dumps({
                "purpose": "B01_engineering_baseline_not_final_5_percent_acceptance",
                "model_id": model_id, "backend": backend, "status": "unsupported_backend",
                "error": "需要专用零样本特征适配器", "finished_at": now(),
            }, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"[{position}/{len(selected)}] UNSUPPORTED {model_id} backend={backend}", flush=True)
            failures += 1
            continue
        # Keep the NVIDIA reference environment fixed across model families.
        # It also includes Torch 2.6's secure .bin loader and OpenCLIP's timm.
        worker_python = WORKER_PYTHON_OVERRIDES.get(model_id, DEFAULT_WORKER_PYTHON)
        command = [
            str(worker_python), str(Path(__file__).resolve()),
            "--worker-model-id", model_id, "--worker-backend", backend,
            "--worker-output", str(output), "--datasets", *args.datasets,
            "--max-samples", str(args.max_samples), "--batch-size", str(args.batch_size),
            "--seed", str(args.seed),
        ]
        print(f"[{position}/{len(selected)}] RUN {model_id} backend={backend}", flush=True)
        try:
            completed = subprocess.run(command, cwd=ROOT, timeout=args.timeout_seconds)
            failures += int(completed.returncode != 0)
        except subprocess.TimeoutExpired:
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(json.dumps({
                "model_id": model_id, "backend": backend, "status": "timeout",
                "timeout_seconds": args.timeout_seconds, "finished_at": now(),
            }, ensure_ascii=False, indent=2), encoding="utf-8")
            failures += 1
    print(json.dumps({"models": len(selected), "failures": failures, "result_root": str(RESULT_ROOT)}, ensure_ascii=False), flush=True)
    return 1 if failures else 0


def main() -> int:
    args = parse_args()
    if args.max_samples == 0 or args.max_samples < -1:
        raise SystemExit("--max-samples必须为正数，或-1表示完整数据集")
    if args.batch_size <= 0:
        raise SystemExit("--batch-size必须为正数")
    return worker(args) if args.worker_model_id else coordinator(args)


if __name__ == "__main__":
    raise SystemExit(main())
