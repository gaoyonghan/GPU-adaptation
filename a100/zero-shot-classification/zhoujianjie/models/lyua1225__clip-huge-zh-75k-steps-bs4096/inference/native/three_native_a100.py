#!/usr/bin/env python3
"""Run explicitly non-official proxy diagnostics for the remaining 11 models.

Every worker result is isolated, pinned to a local checkpoint receipt, and
marked diagnostic-only.  These results must not change the strict official
acceptance count.
"""

from __future__ import annotations
from model_paths import legacy as _legacy, legacy_str as _legacy_str, relative as _relative, revision as _revision, weight as _weight, result as _result

import argparse
import gc
import hashlib
import json
import platform
import subprocess
import sys
import time
import traceback
from collections import OrderedDict
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "evaluation_audit"))
import evaluate_b01 as b01

SEED = 20260909
WORKER_PYTHON = _legacy_str('/home/dell/anaconda3/envs/UMind_agent/bin/python')
COLIPRI_PYTHON = _legacy_str('/home/dell/anaconda3/envs/hengxiang_stage2_clean/bin/python')
RESULT_ROOT = ROOT / "results/proxy_diagnostics/remaining_11"
PREFLIGHT_PATH = ROOT / "evaluation_audit/remaining_11_proxy_preflight.json"
AGGREGATE_PATH = ROOT / "evaluation_audit/remaining_11_proxy_results.json"
ANIME_ROOT = _legacy('/tmp/anime_dbrating_proxy')
ANIME_MANIFEST = _legacy('/tmp/anime_dbrating_proxy/manifest.json')
ANIME_REVISION = "de982f3a388fc78f6b7dd6429f0f8ee0be62e2f9"
CT_PATH = _legacy('/tmp/colipri_CT_chest.nrrd')
CT_SHA256 = "4507b664690840abb6cb9af2d919377ffc4ef75b167cb6fd0f747befdb12e38e"
PCAM_ROOT = ROOT / "datasets/evaluation/biomedclip_pcam_official_original"
COUNTRY_PARQUET = ROOT / "datasets/evaluation/mmeb_eval_hf/Country211/test-00000-of-00001.parquet"
COUNTRY_IMAGES = ROOT / "datasets/evaluation/mmeb_eval_images/Country211"
COUNTRY_REVISION = "2f069730be515ea60778413777816b53e2d2a697"
IMAGENETV2_REVISION = "d626240be2538720e83103a0e1178d24aca8b12c"
XTD_ROOT = ROOT / "datasets/evaluation/xtd10_hf"
XTD_REVISION = "63227af6ee1d13951d244b979254154d7babe463"

MODEL_SPECS = OrderedDict([
    ("geolocal/StreetCLIP", {
        "task": "country211", "dataset": "TIGER-Lab/MMEB-eval Country211 test",
        "split": "test, 1,000 images", "metric": "zero-shot country top-1/top-5 accuracy",
    }),
    ("google/siglip-base-patch16-256-multilingual", {
        "task": "xtd10", "languages": ["es", "it", "ko", "pl", "ru", "tr", "zh"],
        "dataset": "Haon-Chen/XTD-10", "split": "test, 999 pairs per language",
        "metric": "text-to-image Recall@1/5/10 by language and macro mean",
    }),
    ("google/siglip-so400m-patch16-256-i18n", {
        "task": "xtd10", "languages": ["es", "it", "ko", "pl", "ru", "tr", "zh"],
        "dataset": "Haon-Chen/XTD-10", "split": "test, 999 pairs per language",
        "metric": "text-to-image Recall@1/5/10 by language and macro mean",
    }),
    ("lyua1225/clip-huge-zh-75k-steps-bs4096", {
        "task": "xtd10", "languages": ["zh"], "dataset": "Haon-Chen/XTD-10",
        "split": "Chinese test, 999 pairs", "metric": "text-to-image Recall@1/5/10",
    }),
    ("Bingsu/clip-vit-large-patch14-ko", {
        "task": "xtd10", "languages": ["ko"], "dataset": "Haon-Chen/XTD-10",
        "split": "Korean test, 999 pairs", "metric": "text-to-image Recall@1/5/10",
    }),
    ("google/siglip-so400m-patch14-224", {
        "task": "imagenetv2", "dataset": "ImageNetV2 matched-frequency",
        "split": "10,000-image validation-style robustness set", "metric": "top-1/top-5 accuracy",
    }),
    ("HuggingFaceM4/siglip-so400m-14-980-flash-attn2-navit", {
        "task": "imagenetv2", "dataset": "ImageNetV2 matched-frequency",
        "split": "10,000-image validation-style robustness set", "metric": "top-1/top-5 accuracy",
    }),
    ("woweenie/open-clip-vit-h-nsfw-finetune", {
        "task": "anime_binary", "dataset": "deepghs/anime_dbrating proxy subset",
        "split": "deterministic train-archive sample, 32 images per rating",
        "metric": "SFW/NSFW zero-shot accuracy, balanced accuracy, macro-F1",
    }),
    ("OysterQAQ/DanbooruCLIP", {
        "task": "anime_rating", "dataset": "deepghs/anime_dbrating proxy subset",
        "split": "deterministic train-archive sample, 32 images per rating",
        "metric": "four-rating zero-shot top-1 accuracy and macro-F1",
    }),
    ("google/medsiglip-448", {
        "task": "pcam", "dataset": "PatchCamelyon original test split",
        "split": "test, 32,768 pathology patches", "metric": "zero-shot accuracy, balanced accuracy, AUROC",
    }),
    ("microsoft/colipri", {
        "task": "ct_prompt", "dataset": "3D Slicer CTChest sample",
        "split": "one unlabeled public example CT", "metric": "frozen clinical prompt similarities/ranking",
    }),
])


def now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def result_path(model_id: str) -> Path:
    return RESULT_ROOT.joinpath(*model_id.split("/"), "summary.json")


class LyuaChineseAdapter:
    name = "Transformers CLIPModel + audited local Chinese tokenizer"

    def __init__(self):
        import torch
        from transformers import AutoTokenizer, CLIPImageProcessor, CLIPModel

        directory = b01.model_dir("lyua1225/clip-huge-zh-75k-steps-bs4096")
        self.tokenizer = AutoTokenizer.from_pretrained(
            directory, local_files_only=True, trust_remote_code=True,
        )
        self.processor = CLIPImageProcessor.from_pretrained(directory, local_files_only=True)
        self.model = CLIPModel.from_pretrained(
            directory, local_files_only=True, torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
        ).eval().cuda()

    def text_features(self, prompts: list[str]):
        import torch

        inputs = self.tokenizer(
            prompts, padding=True, truncation=True, max_length=77, return_tensors="pt",
        )
        inputs.pop("token_type_ids", None)
        inputs = {key: value.cuda(non_blocking=True) for key, value in inputs.items()}
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
            features = self.model.get_text_features(**inputs)
        return torch.nn.functional.normalize(features.float(), dim=-1)

    def image_features(self, images: list[Any]):
        import torch

        values = self.processor(images=images, return_tensors="pt")["pixel_values"].cuda(non_blocking=True)
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
            features = self.model.get_image_features(pixel_values=values)
        return torch.nn.functional.normalize(features.float(), dim=-1)


class Navit384Adapter:
    name = "audited local NaViT model with publisher-documented 384px compatibility preprocessing"

    def __init__(self):
        import torch
        from transformers import AutoModel, AutoProcessor

        model_dir = b01.model_dir("HuggingFaceM4/siglip-so400m-14-980-flash-attn2-navit")
        processor_dir = b01.model_dir("google/siglip-so400m-patch14-384")
        self.processor = AutoProcessor.from_pretrained(processor_dir, local_files_only=True)
        self.model = AutoModel.from_pretrained(
            model_dir, local_files_only=True, trust_remote_code=True,
            torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
        ).eval().cuda()

    def text_features(self, prompts: list[str]):
        import torch

        values = self.processor(
            text=prompts, padding="max_length", truncation=True,
            max_length=64, return_tensors="pt",
        )
        values = {key: value.cuda(non_blocking=True) for key, value in values.items()}
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            features = self.model.get_text_features(**values)
        return torch.nn.functional.normalize(features.float(), dim=-1)

    def image_features(self, images: list[Any]):
        import torch

        pixels = self.processor(images=images, return_tensors="pt")["pixel_values"].cuda(non_blocking=True)
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            features = self.model.get_image_features(pixel_values=pixels)
        return torch.nn.functional.normalize(features.float(), dim=-1)


def load_adapter(model_id: str, backend: str):
    if model_id == "lyua1225/clip-huge-zh-75k-steps-bs4096":
        return LyuaChineseAdapter()
    if model_id == "HuggingFaceM4/siglip-so400m-14-980-flash-attn2-navit":
        return Navit384Adapter()
    return b01.load_adapter(model_id, backend)


def class_features(adapter: Any, prompt_groups: list[list[str]]):
    import torch

    outputs = []
    for prompts in prompt_groups:
        values = adapter.text_features(prompts).mean(dim=0, keepdim=True)
        outputs.append(torch.nn.functional.normalize(values, dim=-1))
    return torch.cat(outputs)


def confusion_metrics(targets: list[int], predictions: list[int], class_count: int) -> dict:
    confusion = [[0 for _ in range(class_count)] for _ in range(class_count)]
    for target, prediction in zip(targets, predictions):
        confusion[target][prediction] += 1
    recalls, f1s = [], []
    for label in range(class_count):
        tp = confusion[label][label]
        fn = sum(confusion[label]) - tp
        fp = sum(row[label] for row in confusion) - tp
        recall = tp / (tp + fn) if tp + fn else 0.0
        precision = tp / (tp + fp) if tp + fp else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        recalls.append(recall)
        f1s.append(f1)
    return {
        "accuracy_percent": round(100 * sum(t == p for t, p in zip(targets, predictions)) / len(targets), 4),
        "balanced_accuracy_percent": round(100 * sum(recalls) / class_count, 4),
        "macro_f1_percent": round(100 * sum(f1s) / class_count, 4),
        "confusion_matrix_rows_target_columns_prediction": confusion,
    }


def binary_auroc(targets: list[int], scores: list[float]) -> float:
    ordered = sorted(zip(scores, targets), key=lambda item: item[0])
    ranks = [0.0] * len(ordered)
    index = 0
    while index < len(ordered):
        end = index + 1
        while end < len(ordered) and ordered[end][0] == ordered[index][0]:
            end += 1
        average_rank = (index + 1 + end) / 2
        for position in range(index, end):
            ranks[position] = average_rank
        index = end
    positive = sum(targets)
    negative = len(targets) - positive
    positive_rank_sum = sum(rank for rank, (_, target) in zip(ranks, ordered) if target == 1)
    value = (positive_rank_sum - positive * (positive + 1) / 2) / (positive * negative)
    return round(100 * value, 4)


def evaluate_pcam(adapter: Any, batch_size: int) -> dict:
    import h5py
    import torch
    from PIL import Image

    prompt_groups = [
        ["a histopathology image without metastatic tumor tissue", "benign lymph node tissue"],
        ["a histopathology image containing metastatic tumor tissue", "metastatic tumor tissue"],
    ]
    weights = class_features(adapter, prompt_groups)
    targets, predictions, scores = [], [], []
    x_path = PCAM_ROOT / "camelyonpatch_level_2_split_test_x.h5"
    y_path = PCAM_ROOT / "camelyonpatch_level_2_split_test_y.h5"
    started = time.perf_counter()
    with h5py.File(x_path) as x_file, h5py.File(y_path) as y_file:
        x_values, y_values = x_file["x"], y_file["y"]
        for offset in range(0, len(x_values), batch_size):
            images = [Image.fromarray(value, mode="RGB") for value in x_values[offset:offset + batch_size]]
            target = y_values[offset:offset + batch_size].reshape(-1).astype(int).tolist()
            logits = adapter.image_features(images) @ weights.T
            targets.extend(target)
            predictions.extend(logits.argmax(dim=1).cpu().tolist())
            scores.extend((logits[:, 1] - logits[:, 0]).cpu().tolist())
            seen = min(offset + batch_size, len(x_values))
            if seen % 4096 < batch_size:
                print(f"PCAM {seen:,}/32,768", flush=True)
    metrics = confusion_metrics(targets, predictions, 2)
    metrics["auroc_percent"] = binary_auroc(targets, scores)
    return {
        "sample_count": len(targets), "labels": ["negative", "positive"],
        "prompt_groups": prompt_groups, "metrics": metrics,
        "dataset_files": {"x": str(x_path), "y": str(y_path)},
        "elapsed_seconds": round(time.perf_counter() - started, 4),
        "peak_gpu_memory_mb": round(torch.cuda.max_memory_allocated() / 1024**2, 2),
    }


def evaluate_country211(adapter: Any, batch_size: int) -> dict:
    import pyarrow.parquet as pq
    import torch
    from PIL import Image

    rows = pq.read_table(COUNTRY_PARQUET).to_pylist()
    labels = rows[0]["tgt_text"]
    if len(labels) != 211 or any(set(row["tgt_text"]) != set(labels) for row in rows):
        raise RuntimeError("Country211 candidate set mismatch")
    prompt_templates = ["{}", "a street-level photo in {}", "a photo taken in {}"]
    weights = class_features(adapter, [[template.format(label) for template in prompt_templates] for label in labels])
    label_index = {label: index for index, label in enumerate(labels)}
    targets, predictions, top5_predictions = [], [], []
    started = time.perf_counter()
    for offset in range(0, len(rows), batch_size):
        batch = rows[offset:offset + batch_size]
        images = []
        for row in batch:
            name = Path(row["qry_img_path"]).name
            with Image.open(COUNTRY_IMAGES / name) as source:
                images.append(source.convert("RGB").copy())
        logits = adapter.image_features(images) @ weights.T
        top5 = logits.topk(5, dim=1).indices.cpu().tolist()
        targets.extend(label_index[row["tgt_text"][0]] for row in batch)
        predictions.extend(item[0] for item in top5)
        top5_predictions.extend(top5)
    (PREDICTION_ROOT/'country_predictions.json').write_text(json.dumps({'targets':targets,'predictions':predictions,'top5_predictions':top5_predictions}))
    top5_correct = sum(target in guesses for target, guesses in zip(targets, top5_predictions))
    return {
        "sample_count": len(rows), "class_count": len(labels), "labels": labels,
        "prompt_templates": prompt_templates,
        "top1_accuracy_percent": round(100 * sum(t == p for t, p in zip(targets, predictions)) / len(rows), 4),
        "top5_accuracy_percent": round(100 * top5_correct / len(rows), 4),
        "elapsed_seconds": round(time.perf_counter() - started, 4),
        "peak_gpu_memory_mb": round(torch.cuda.max_memory_allocated() / 1024**2, 2),
    }


def evaluate_imagenetv2(adapter: Any, batch_size: int) -> dict:
    import torch
    from evaluate_downloaded_imagenet_robustness import classifier, protocol, sample_batches

    classnames, templates, wnids, label_by_wnid = protocol("imagenetv2")
    weights = classifier(adapter, classnames, templates)
    correct1 = correct5 = seen = 0
    started = time.perf_counter()
    for images, targets in sample_batches("imagenetv2", label_by_wnid, batch_size):
        predictions = (adapter.image_features(images) @ weights.T).topk(5, dim=1).indices.cpu()
        target = torch.tensor(targets)
        correct1 += int((predictions[:, 0] == target).sum())
        correct5 += int((predictions == target[:, None]).any(dim=1).sum())
        seen += len(targets)
        if seen % 2000 < len(targets):
            print(f"IMAGENETV2 {seen:,}/10,000", flush=True)
    if seen != 10000:
        raise RuntimeError(f"expected 10,000 ImageNetV2 samples, found {seen}")
    return {
        "sample_count": seen, "class_count": len(classnames), "class_wnids": wnids,
        "prompt_source": "PE bundled CLIP Benchmark ImageNet classnames and 80-template ensemble",
        "prompt_templates_per_class": len(templates),
        "top1_accuracy_percent": round(100 * correct1 / seen, 4),
        "top5_accuracy_percent": round(100 * correct5 / seen, 4),
        "elapsed_seconds": round(time.perf_counter() - started, 4),
        "peak_gpu_memory_mb": round(torch.cuda.max_memory_allocated() / 1024**2, 2),
    }


def evaluate_xtd10(adapter: Any, languages: list[str], image_batch_size: int, text_batch_size: int) -> dict:
    import torch
    from PIL import Image
    from evaluate_xtd10_clip import records

    frozen = {language: records(language) for language in languages}
    _limit=int(__import__('os').environ.get('ZERO_SHOT_SMOKE_LIMIT','0'))
    if _limit:
        if _limit<10:raise ValueError('Original XTD Top10 smoke requires --max-images >=10')
        frozen={lang:([q for q,t in zip(qs,ts) if t<_limit][:_limit],cs[:_limit],[t for t in ts if t<_limit][:_limit]) for lang,(qs,cs,ts) in frozen.items()}
    first_candidates = frozen[languages[0]][1]
    if any(candidates != first_candidates for _, candidates, _ in frozen.values()):
        raise RuntimeError("XTD-10 candidate image order differs across requested languages")
    image_features = []
    started = time.perf_counter()
    for offset in range(0, len(first_candidates), image_batch_size):
        images = []
        for name in first_candidates[offset:offset + image_batch_size]:
            with Image.open(XTD_ROOT / "XTD10_dataset" / name) as source:
                images.append(source.convert("RGB").copy())
        image_features.append(adapter.image_features(images).cpu())
    image_features = torch.cat(image_features)
    per_language = {}
    for language in languages:
        queries, _, targets = frozen[language]
        hits = {1: 0, 5: 0, 10: 0}
        for offset in range(0, len(queries), text_batch_size):
            values = adapter.text_features(queries[offset:offset + text_batch_size]).cpu()
            ranked = (values @ image_features.T).topk(10, dim=1).indices.tolist()
            with (PREDICTION_ROOT/(language+'_predictions.jsonl')).open('a') as stream:
                for target,ranking in zip(targets[offset:offset + text_batch_size],ranked):stream.write(json.dumps({'target':target,'ranking':ranking})+'\n')
            for target, prediction in zip(targets[offset:offset + text_batch_size], ranked):
                for k in hits:
                    hits[k] += target in prediction[:k]
        per_language[language] = {
            f"text_to_image_recall_at_{k}_percent": round(100 * hits[k] / len(queries), 4)
            for k in hits
        }
        print(json.dumps({"language": language, **per_language[language]}), flush=True)
    macro = {
        f"text_to_image_recall_at_{k}_percent": round(
            sum(per_language[language][f"text_to_image_recall_at_{k}_percent"] for language in languages)
            / len(languages), 4,
        )
        for k in (1, 5, 10)
    }
    return {
        "languages": languages, "query_count_per_language": len(frozen[languages[0]][0]), "candidate_count": len(first_candidates),
        "instruction_handling": "MMEB retrieval prefix removed; raw XTD-10 captions encoded",
        "per_language": per_language, "macro_average": macro,
        "elapsed_seconds": round(time.perf_counter() - started, 4),
        "peak_gpu_memory_mb": round(torch.cuda.max_memory_allocated() / 1024**2, 2),
    }


def evaluate_anime(adapter: Any, binary: bool, batch_size: int) -> dict:
    import torch
    from PIL import Image

    manifest = json.loads(ANIME_MANIFEST.read_text(encoding="utf-8"))
    if binary:
        labels = ["sfw", "nsfw"]
        prompt_groups = [
            ["safe for work", "non-explicit illustration", "general audience illustration"],
            ["not safe for work", "explicit adult illustration", "questionable adult illustration"],
        ]
        target_for = {"general": 0, "sensitive": 0, "questionable": 1, "explicit": 1}
    else:
        labels = ["general", "sensitive", "questionable", "explicit"]
        prompt_groups = [
            ["general", "rating:general"], ["sensitive", "rating:sensitive"],
            ["questionable", "rating:questionable"], ["explicit", "rating:explicit"],
        ]
        target_for = {label: index for index, label in enumerate(labels)}
    weights = class_features(adapter, prompt_groups)
    samples = manifest["samples"]
    targets, predictions = [], []
    started = time.perf_counter()
    for offset in range(0, len(samples), batch_size):
        batch = samples[offset:offset + batch_size]
        images = []
        for record in batch:
            with Image.open(ANIME_ROOT / record["relative_path"]) as source:
                images.append(source.convert("RGB").copy())
        predictions.extend((adapter.image_features(images) @ weights.T).argmax(dim=1).cpu().tolist())
        targets.extend(target_for[record["label"]] for record in batch)
    return {
        "sample_count": len(samples), "labels": labels, "prompt_groups": prompt_groups,
        "sample_manifest": str(ANIME_MANIFEST), "metrics": confusion_metrics(targets, predictions, len(labels)),
        "elapsed_seconds": round(time.perf_counter() - started, 4),
        "peak_gpu_memory_mb": round(torch.cuda.max_memory_allocated() / 1024**2, 2),
    }


def evaluate_colipri() -> dict:
    import torch
    from colipri import get_model, get_processor
    from safetensors.torch import load_file

    directory = b01.model_dir("microsoft/colipri")
    dependency = ROOT / "models/_dependencies/microsoft/BiomedVLP-CXR-BERT-specialized"
    model = get_model(
        pretrained=False,
        **{
            "model.text_encoder.backbone.pretrained_model_name_or_path": str(dependency),
            "model.text_encoder.backbone.local_files_only": True,
        },
    )
    incompatible = model.load_state_dict(load_file(directory / "model.safetensors"), strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(f"COLIPRI checkpoint keys mismatch: {incompatible}")
    model = model.eval().half().cuda()
    processor = get_processor(**{
        "processor.tokenizer.pretrained_model_name_or_path": str(dependency),
        "processor.tokenizer.local_files_only": True,
    })
    prompt_pairs = OrderedDict([
        ("lung_nodule", ["No lung nodules are present.", "Lung nodules are present."]),
        ("pleural_effusion", ["No pleural effusion is present.", "Pleural effusion is present."]),
        ("emphysema", ["No emphysema is present.", "Emphysema is present."]),
        ("atelectasis", ["No atelectasis is present.", "Atelectasis is present."]),
        ("cardiomegaly", ["No cardiomegaly is present.", "Cardiomegaly is present."]),
    ])
    prompts = [prompt for pair in prompt_pairs.values() for prompt in pair]
    started = time.perf_counter()
    processed = processor.process_images(CT_PATH)
    volume = processor.to_images_batch(processed).half().cuda()
    token_ids, attention_mask = processor.process_text(prompts)
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
        image_features = model.encode_image(volume, pool=True, project=True, normalize=True)
        text_features = model.encode_text(
            token_ids.cuda(), attention_mask.cuda(), pool=True, normalize=True,
        )
        similarities = (image_features @ text_features.T).float().cpu().squeeze(0)
    pair_results = {}
    for pair_index, (name, pair) in enumerate(prompt_pairs.items()):
        pair_scores = similarities[2 * pair_index:2 * pair_index + 2]
        probabilities = torch.softmax(pair_scores, dim=0)
        winner = int(pair_scores.argmax())
        pair_results[name] = {
            "prompts": pair, "cosine_similarities": [round(float(value), 6) for value in pair_scores],
            "two_prompt_softmax_percent": [round(100 * float(value), 4) for value in probabilities],
            "higher_similarity_prompt": pair[winner],
        }
    return {
        "sample_count": 1, "input_path": str(CT_PATH), "input_sha256": sha256_file(CT_PATH),
        "processed_tensor_shape": list(volume.shape), "unlabeled_input": True,
        "interpretation_warning": "prompt rankings are diagnostics, not clinical predictions or accuracy",
        "prompt_pair_results": pair_results,
        "elapsed_seconds": round(time.perf_counter() - started, 4),
        "peak_gpu_memory_mb": round(torch.cuda.max_memory_allocated() / 1024**2, 2),
    }


def worker(args: argparse.Namespace) -> int:
    import torch

    model_id = args.worker_model_id
    spec = MODEL_SPECS[model_id]
    output = Path(args.worker_output)
    result = {
        "status": "failed", "purpose": "user_authorized_proxy_diagnostic_only",
        "strict_official_acceptance": False, "changes_strict_acceptance_count": False,
        "user_relaxed_acceptance": None, "same_checkpoint_official_reference": None,
        "model_id": model_id, "task": spec["task"], "dataset": spec["dataset"],
        "dataset_split": spec["split"], "metric": spec["metric"], "seed": SEED,
        "started_at": now(),
    }
    adapter = None
    try:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable")
        torch.manual_seed(SEED)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        receipt = json.loads((b01.model_dir(model_id) / "download_receipt.json").read_text(encoding="utf-8"))
        result["model_revision"] = receipt.get("revision") or receipt.get("resolved_revision")
        if spec["task"] == "ct_prompt":
            result["backend"] = "COLIPRI 0.1.3 official package/local checkpoint"
            metrics = evaluate_colipri()
        else:
            adapter = load_adapter(model_id, args.worker_backend)
            result["backend"] = adapter.name
            task = spec["task"]
            if task == "pcam":
                metrics = evaluate_pcam(adapter, args.batch_size)
            elif task == "country211":
                metrics = evaluate_country211(adapter, args.batch_size)
            elif task == "imagenetv2":
                metrics = evaluate_imagenetv2(adapter, args.batch_size)
            elif task == "xtd10":
                metrics = evaluate_xtd10(adapter, spec["languages"], args.batch_size, args.text_batch_size)
            elif task == "anime_binary":
                metrics = evaluate_anime(adapter, True, args.batch_size)
            elif task == "anime_rating":
                metrics = evaluate_anime(adapter, False, args.batch_size)
            else:
                raise RuntimeError(f"unknown task: {task}")
        result.update(
            status="completed_diagnostic_only", result=metrics,
            environment={
                "python": platform.python_version(), "torch": torch.__version__,
                "cuda_runtime": torch.version.cuda, "gpu": torch.cuda.get_device_name(0),
            },
        )
    except Exception as error:
        result.update(
            error_type=type(error).__name__, error=str(error),
            traceback=traceback.format_exc(limit=24),
        )
    finally:
        result["finished_at"] = now()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        del adapter
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    print(json.dumps({"model_id": model_id, "status": result["status"], "output": str(output)}), flush=True)
    return 0 if result["status"] == "completed_diagnostic_only" else 1


def verify_proxy_assets(selected_specs: list[dict]) -> dict:
    import h5py
    import pyarrow.parquet as pq

    tasks = {item["task"] for item in selected_specs}
    evidence = {}
    if "pcam" in tasks:
        x_path = PCAM_ROOT / "camelyonpatch_level_2_split_test_x.h5"
        y_path = PCAM_ROOT / "camelyonpatch_level_2_split_test_y.h5"
        with h5py.File(x_path) as x_file, h5py.File(y_path) as y_file:
            if x_file["x"].shape != (32768, 96, 96, 3) or y_file["y"].shape != (32768, 1, 1, 1):
                raise RuntimeError("PatchCamelyon test shape mismatch")
        evidence["pcam"] = {"x": str(x_path), "y": str(y_path), "rows": 32768}
    if "country211" in tasks:
        metadata = ROOT / "datasets/evaluation/mmeb_eval_hf/.cache/huggingface/download/Country211/test-00000-of-00001.parquet.metadata"
        revision, expected_hash, *_ = metadata.read_text(encoding="utf-8").splitlines()
        if revision != COUNTRY_REVISION or sha256_file(COUNTRY_PARQUET) != expected_hash:
            raise RuntimeError("Country211 revision/hash mismatch")
        if pq.ParquetFile(COUNTRY_PARQUET).metadata.num_rows != 1000:
            raise RuntimeError("Country211 row count mismatch")
        images = [path for path in COUNTRY_IMAGES.iterdir() if path.is_file()]
        if len(images) != 1000:
            raise RuntimeError("Country211 extracted image count mismatch")
        evidence["country211"] = {"revision": revision, "sha256": expected_hash, "rows": 1000}
    if "imagenetv2" in tasks:
        from evaluate_downloaded_imagenet_robustness import dataset_preflight
        evidence["imagenetv2"] = dataset_preflight("imagenetv2")
    if "xtd10" in tasks:
        languages = sorted({language for spec in selected_specs for language in spec.get("languages", [])})
        for language in languages:
            metadata = XTD_ROOT / f".cache/huggingface/download/{language}/{language}.parquet.metadata"
            revision, expected_hash, *_ = metadata.read_text(encoding="utf-8").splitlines()
            parquet = XTD_ROOT / language / f"{language}.parquet"
            if revision != XTD_REVISION or sha256_file(parquet) != expected_hash:
                raise RuntimeError(f"XTD-10 {language} revision/hash mismatch")
            if pq.ParquetFile(parquet).metadata.num_rows != 999:
                raise RuntimeError(f"XTD-10 {language} row count mismatch")
        image_count = sum(1 for path in (XTD_ROOT / "XTD10_dataset").iterdir() if path.is_file())
        if image_count != 1000:
            raise RuntimeError(f"XTD-10 extracted image count mismatch: {image_count}")
        evidence["xtd10"] = {"revision": XTD_REVISION, "languages": languages, "pairs_per_language": 999}
    if tasks & {"anime_binary", "anime_rating"}:
        manifest = json.loads(ANIME_MANIFEST.read_text(encoding="utf-8"))
        if manifest.get("dataset_revision") != ANIME_REVISION or manifest.get("seed") != SEED:
            raise RuntimeError("anime proxy manifest revision/seed mismatch")
        for sample in manifest["samples"]:
            path = ANIME_ROOT / sample["relative_path"]
            if not path.is_file() or sha256_file(path) != sample["sha256"]:
                raise RuntimeError(f"anime proxy sample hash mismatch: {path}")
        evidence["anime_dbrating"] = {
            "revision": ANIME_REVISION, "seed": SEED, "sample_count": len(manifest["samples"]),
            "manifest": str(ANIME_MANIFEST),
        }
    if "ct_prompt" in tasks:
        if not CT_PATH.is_file() or sha256_file(CT_PATH) != CT_SHA256:
            raise RuntimeError("COLIPRI sample CT is missing or has the wrong hash")
        evidence["ct_chest"] = {"path": str(CT_PATH), "sha256": CT_SHA256, "unlabeled": True}
    return evidence


def preflight(model_ids: list[str], rows: dict[str, dict]) -> dict:
    selected_specs = [MODEL_SPECS[model_id] for model_id in model_ids]
    datasets = verify_proxy_assets(selected_specs)
    mapping = []
    for model_id in model_ids:
        receipt_path = b01.model_dir(model_id) / "download_receipt.json"
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        if receipt.get("model_id") != model_id:
            raise RuntimeError(f"checkpoint receipt model ID mismatch: {model_id}")
        revision = receipt.get("revision") or receipt.get("resolved_revision")
        if not revision:
            raise RuntimeError(f"checkpoint revision missing: {model_id}")
        spec = MODEL_SPECS[model_id]
        mapping.append({
            "model_id": model_id, "checkpoint_revision": revision,
            "backend": "COLIPRI official package" if spec["task"] == "ct_prompt" else b01.backend_for(rows[model_id]),
            "proxy_dataset": spec["dataset"], "proxy_split": spec["split"],
            "protocol": spec["task"], "metric": spec["metric"],
            "same_checkpoint_official_reference": None,
            "strict_official_acceptance": False,
            "acceptance_effect": "none_pending_user_review",
            "user_decision": "run a frozen substitute dataset/protocol first when official evidence is unavailable",
        })
    payload = {
        "generated_at": now(), "status": "PASS_DIAGNOSTIC_PREFLIGHT",
        "purpose": "multi-model proxy diagnostic; explicitly not official metric acceptance",
        "strict_acceptance_count_before_run": "60/77",
        "user_relaxed_acceptance_count_before_run": "66/77",
        "seed": SEED, "datasets": datasets,
        "model_to_dataset_to_reference": mapping,
    }
    PREFLIGHT_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("DIAGNOSTIC PREFLIGHT MODEL->DATASET->REFERENCE", flush=True)
    print(json.dumps(payload, ensure_ascii=False, indent=2), flush=True)
    return payload


def write_aggregate(model_ids: list[str], preflight_payload: dict) -> dict:
    records = []
    for model_id in model_ids:
        output = result_path(model_id)
        record = json.loads(output.read_text(encoding="utf-8")) if output.is_file() else {
            "model_id": model_id, "status": "not_started",
        }
        records.append(record)
    completed = sum(record.get("status") == "completed_diagnostic_only" for record in records)
    aggregate = {
        "generated_at": now(), "purpose": "remaining 11 proxy diagnostic aggregate",
        "preflight": str(_relative(PREFLIGHT_PATH,ROOT)), "selected_model_count": len(model_ids),
        "completed_diagnostic_count": completed,
        "failed_or_incomplete_count": len(model_ids) - completed,
        "strict_acceptance_count_unchanged": "60/77",
        "user_relaxed_acceptance_count_unchanged_pending_review": "66/77",
        "results": records,
    }
    AGGREGATE_PATH.write_text(json.dumps(aggregate, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return aggregate


def coordinator(args: argparse.Namespace) -> int:
    rows = {str(row["模型名称"]): row for row in b01.ledger_rows()}
    model_ids = list(dict.fromkeys(args.model_id or MODEL_SPECS.keys()))
    unknown = [model_id for model_id in model_ids if model_id not in MODEL_SPECS]
    missing = [model_id for model_id in model_ids if model_id not in rows]
    if unknown or missing:
        raise SystemExit(f"unknown proxy models={unknown}; absent from ledger={missing}")
    preflight_payload = preflight(model_ids, rows)
    if args.preflight_only:
        return 0
    failures = 0
    for position, model_id in enumerate(model_ids, 1):
        output = result_path(model_id)
        if output.is_file() and not args.force:
            previous = json.loads(output.read_text(encoding="utf-8"))
            if previous.get("status") == "completed_diagnostic_only":
                print(f"[{position}/{len(model_ids)}] SKIP COMPLETED {model_id}", flush=True)
                write_aggregate(model_ids, preflight_payload)
                continue
        backend = b01.backend_for(rows[model_id])
        python = COLIPRI_PYTHON if MODEL_SPECS[model_id]["task"] == "ct_prompt" else WORKER_PYTHON
        command = [
            python, str(Path(__file__).resolve()),
            "--worker-model-id", model_id, "--worker-backend", backend,
            "--worker-output", str(output), "--batch-size", str(args.batch_size),
            "--text-batch-size", str(args.text_batch_size),
        ]
        print(f"[{position}/{len(model_ids)}] RUN {model_id} -> {MODEL_SPECS[model_id]['task']}", flush=True)
        try:
            completed = subprocess.run(command, cwd=ROOT, check=False, timeout=args.timeout_seconds)
            failures += int(completed.returncode != 0)
        except subprocess.TimeoutExpired:
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(json.dumps({
                "status": "timeout", "model_id": model_id,
                "timeout_seconds": args.timeout_seconds, "finished_at": now(),
                "strict_official_acceptance": False,
            }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            failures += 1
        write_aggregate(model_ids, preflight_payload)
    aggregate = write_aggregate(model_ids, preflight_payload)
    print(json.dumps({
        "completed_diagnostic_count": aggregate["completed_diagnostic_count"],
        "failed_or_incomplete_count": aggregate["failed_or_incomplete_count"],
        "aggregate": str(AGGREGATE_PATH),
    }, ensure_ascii=False), flush=True)
    return 1 if failures else 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", action="append")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--text-batch-size", type=int, default=128)
    parser.add_argument("--timeout-seconds", type=int, default=7200)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--worker-model-id", help=argparse.SUPPRESS)
    parser.add_argument("--worker-backend", help=argparse.SUPPRESS)
    parser.add_argument("--worker-output", help=argparse.SUPPRESS)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.worker_model_id:
        if args.worker_model_id not in MODEL_SPECS or not args.worker_output:
            raise SystemExit("invalid worker arguments")
        return worker(args)
    return coordinator(args)


if __name__ == "__main__":
    raise SystemExit(main())
