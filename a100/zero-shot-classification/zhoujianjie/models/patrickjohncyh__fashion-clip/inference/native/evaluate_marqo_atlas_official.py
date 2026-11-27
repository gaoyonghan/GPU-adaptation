#!/usr/bin/env python3
"""Reproduce Marqo's fixed-query Atlas retrieval benchmark with local models."""

from __future__ import annotations
from model_paths import legacy as _legacy, legacy_str as _legacy_str, relative as _relative, revision as _revision, weight as _weight, result as _result

import argparse
import gc
import json
import platform
import sys
import time
from datetime import datetime
from pathlib import Path


ROOT = _legacy('/data/zhoujianjie/environment_setup/native_workspace')
DATASET = _legacy('/data/zhoujianjie/environment_setup/native_workspace/datasets/evaluation/marqo_atlas_hf/data')
REFERENCE = _legacy('/data/zhoujianjie/environment_setup/native_workspace/evaluation_audit/reference_code/marqo-FashionCLIP')
GT = _legacy('/data/zhoujianjie/environment_setup/native_workspace/evaluation_audit/reference_code/marqo-FashionCLIP/data/Atlas/gt_query_doc')
REFERENCE_REVISION = "d0b3bdfb62fb4964537cb582e76d6ddab0be8450"
DATASET_REVISION = "54924db5b6b2f7f10dfce2053222d20bc614ba0c"
MODELS = {
    "patrickjohncyh/fashion-clip": {
        "path": _legacy('/data/zhoujianjie/environment_setup/native_workspace/models/patrickjohncyh/fashion-clip'),
        "batch_size": 256,
        "backend": "transformers",
        "official": {
            "text_to_image": {"recall_at_1": 0.055, "recall_at_10": 0.214, "mrr": 0.140},
            "subcategory_to_product": {"precision_at_1": 0.618, "precision_at_10": 0.624, "mrr": 0.678},
        },
    },
    "Marqo/marqo-fashionCLIP": {
        "path": _legacy('/data/zhoujianjie/environment_setup/native_workspace/models/Marqo/marqo-fashionCLIP'),
        "batch_size": 256,
        "official": {
            "text_to_image": {"recall_at_1": 0.074, "recall_at_10": 0.236, "mrr": 0.172},
            "subcategory_to_product": {"precision_at_1": 0.706, "precision_at_10": 0.629, "mrr": 0.725},
        },
    },
    "Marqo/marqo-fashionSigLIP": {
        "path": _legacy('/data/zhoujianjie/environment_setup/native_workspace/models/Marqo/marqo-fashionSigLIP'),
        "batch_size": 192,
        "official": {
            "text_to_image": {"recall_at_1": 0.109, "recall_at_10": 0.313, "mrr": 0.244},
            "subcategory_to_product": {"precision_at_1": 0.706, "precision_at_10": 0.668, "mrr": 0.744},
        },
    },
}


def now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def load_local_open_clip(model_id: str):
    import open_clip
    import open_clip.factory as factory

    spec = MODELS[model_id]
    config = json.loads((spec["path"] / "open_clip_config.json").read_text(encoding="utf-8"))
    model_cfg = config["model_cfg"]
    if "hf_tokenizer_name" in model_cfg.get("text_cfg", {}):
        model_cfg["text_cfg"]["hf_tokenizer_name"] = str(spec["path"])
    local_name = "audit_" + model_id.rsplit("/", 1)[-1]
    factory._MODEL_CONFIGS[local_name] = model_cfg
    preprocess_cfg = config["preprocess_cfg"]
    model, _, preprocess = open_clip.create_model_and_transforms(
        local_name,
        pretrained=str(spec["path"] / "open_clip_model.safetensors"),
        image_mean=tuple(preprocess_cfg["mean"]), image_std=tuple(preprocess_cfg["std"]),
        image_interpolation=preprocess_cfg["interpolation"],
        image_resize_mode=preprocess_cfg["resize_mode"],
    )
    tokenizer = open_clip.get_tokenizer(local_name)
    return model.eval().cuda(), preprocess, tokenizer


def load_local_transformers_clip(model_id: str):
    import torch
    from transformers import AutoModel, AutoProcessor

    spec = MODELS[model_id]
    processor = AutoProcessor.from_pretrained(spec["path"], local_files_only=True, use_fast=False)
    hf_model = AutoModel.from_pretrained(
        spec["path"], local_files_only=True, torch_dtype=torch.float16, low_cpu_mem_usage=True
    ).eval().cuda()
    text_max_length = int(getattr(hf_model.config.text_config, "max_position_embeddings", 77))

    class CudaInputs(dict):
        def cuda(self, non_blocking=True):
            return CudaInputs({key: value.cuda(non_blocking=non_blocking) for key, value in self.items()})

    class Wrapper:
        def eval(self):
            return self

        def cuda(self):
            return self

        def encode_image(self, pixels, normalize=False):
            features = hf_model.get_image_features(pixel_values=pixels)
            return torch.nn.functional.normalize(features, dim=-1) if normalize else features

        def encode_text(self, inputs, normalize=False):
            features = hf_model.get_text_features(**inputs)
            return torch.nn.functional.normalize(features, dim=-1) if normalize else features

    def preprocess(image):
        return processor(images=image, return_tensors="pt").pixel_values[0]

    def tokenizer(texts):
        values = processor(text=texts, return_tensors="pt", padding=True, truncation=True,
                           max_length=text_max_length)
        return CudaInputs(values)

    wrapper = Wrapper()
    wrapper.hf_model = hf_model
    return wrapper, preprocess, tokenizer


def load_local_model(model_id: str):
    if MODELS[model_id].get("backend") == "transformers":
        return load_local_transformers_clip(model_id)
    return load_local_open_clip(model_id)


def encode_documents(model, preprocess, tokenizer, batch_size: int, limit: int | None,
                     dataset_label: str = "Atlas"):
    from concurrent.futures import ThreadPoolExecutor
    import io
    import pyarrow.parquet as pq
    import torch
    from PIL import Image

    image_features, text_features, item_ids = [], [], []
    encoded = 0
    started = time.perf_counter()
    def prepare_image(row):
        with Image.open(io.BytesIO(row["image"]["bytes"])) as image:
            return preprocess(image.convert("RGB"))

    decode_pool = ThreadPoolExecutor(max_workers=8, thread_name_prefix="fashion_decode")
    with torch.inference_mode():
        for parquet in sorted(DATASET.glob("*.parquet")):
            source = pq.ParquetFile(parquet)
            for batch in source.iter_batches(batch_size=batch_size, columns=["image", "text", "item_ID"]):
                rows = batch.to_pylist()
                if limit is not None:
                    rows = rows[:max(0, limit - encoded)]
                if not rows:
                    break
                images = list(decode_pool.map(prepare_image, rows))
                pixels = torch.stack(images).cuda(non_blocking=True)
                tokens = tokenizer([row["text"] for row in rows]).cuda(non_blocking=True)
                with torch.autocast("cuda"):
                    torch.cuda.synchronize()
                    image_features.append(model.encode_image(pixels, normalize=True).float().cpu())
                    torch.cuda.synchronize()
                    text_features.append(model.encode_text(tokens, normalize=True).float().cpu())
                item_ids.extend(str(row["item_ID"]) for row in rows)
                encoded += len(rows)
                if encoded % 5000 < len(rows):
                    print(f"encoded {encoded:,} {dataset_label} documents", flush=True)
                if limit is not None and encoded >= limit:
                    break
            if limit is not None and encoded >= limit:
                break
    decode_pool.shutdown(wait=True)
    return torch.cat(image_features), torch.cat(text_features), item_ids, time.perf_counter() - started


def retrieve(query_strings, document_features, item_ids, model, tokenizer, batch_size=256):
    import torch

    results = {}
    with torch.inference_mode():
        for offset in range(0, len(query_strings), batch_size):
            queries = query_strings[offset:offset + batch_size]
            tokens = tokenizer(queries).cuda(non_blocking=True)
            with torch.autocast("cuda"):
                query_features = model.encode_text(tokens, normalize=True).float().cpu()
            scores = query_features @ document_features.T
            values, indices = scores.topk(10, dim=1)
            for query, row_indices, row_values in zip(queries, indices.tolist(), values.tolist()):
                results[query] = {item_ids[index]: score for index, score in zip(row_indices, row_values)}
    return results


def metrics(qrels, results, task: str) -> dict[str, float]:
    recall, precision, reciprocal = {1: 0.0, 10: 0.0}, {1: 0.0, 10: 0.0}, 0.0
    for query, relevant_scores in qrels.items():
        relevant = {doc_id for doc_id, score in relevant_scores.items() if score > 0}
        ranked = list(results[query])
        for k in (1, 10):
            hits = len(relevant.intersection(ranked[:k]))
            recall[k] += hits / len(relevant)
            precision[k] += hits / k
        for rank, doc_id in enumerate(ranked, 1):
            if doc_id in relevant:
                reciprocal += 1.0 / rank
                break
    count = len(qrels)
    if task == "text_to_image":
        return {"recall_at_1": recall[1] / count, "recall_at_10": recall[10] / count,
                "mrr": reciprocal / count}
    return {"precision_at_1": precision[1] / count, "precision_at_10": precision[10] / count,
            "mrr": reciprocal / count}


def evaluate(model_id: str, limit: int | None) -> dict:
    import torch
    import torch.nn.functional as F

    spec = MODELS[model_id]
    output = ROOT.joinpath("results/marqo_atlas_official", *model_id.split("/"), "summary.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    result = {"status": "failed", "model_id": model_id, "dataset": "Marqo Atlas",
              "dataset_revision": DATASET_REVISION, "reference_code_revision": REFERENCE_REVISION,
              "started_at": now(), "smoke_limit": limit}
    model = None
    try:
        model, preprocess, tokenizer = load_local_model(model_id)
        torch.cuda.reset_peak_memory_stats()
        images, texts, item_ids, elapsed = encode_documents(
            model, preprocess, tokenizer, spec["batch_size"], limit, "Atlas"
        )
        result["document_count"] = len(item_ids)
        if limit is not None:
            result.update(status="smoke_passed", embedding_dimensions=images.shape[1])
        else:
            tasks = {
                "text_to_image": (
                    _legacy('/data/zhoujianjie/environment_setup/native_workspace/evaluation_audit/reference_code/marqo-FashionCLIP/data/Atlas/gt_query_doc/ground_truth_text-image.json'), images,
                ),
                "subcategory_to_product": (
                    _legacy('/data/zhoujianjie/environment_setup/native_workspace/evaluation_audit/reference_code/marqo-FashionCLIP/data/Atlas/gt_query_doc/ground_truth_sub-category-image+text.json'),
                    F.normalize(0.9 * images + 0.1 * texts, dim=-1),
                ),
            }
            measured = {}
            for task, (path, documents) in tasks.items():
                qrels = json.loads(path.read_text(encoding="utf-8"))
                retrieved = retrieve(list(qrels), documents, item_ids, model, tokenizer)
                measured[task] = metrics(qrels, retrieved, task)
            comparisons = {}
            for task in measured:
                comparisons[task] = {}
                for metric, value in measured[task].items():
                    official = spec["official"][task][metric]
                    delta_pp = 100.0 * (value - official)
                    comparisons[task][metric] = {
                        "official": official, "measured": value, "delta_percentage_points": delta_pp,
                        "relative_gap_percent": abs(value - official) / official * 100.0,
                        "within_5_percentage_points": abs(delta_pp) <= 5.0,
                        "within_5_percent_relative": abs(value - official) / official <= 0.05,
                    }
            result.update(status="passed", query_counts={"text_to_image": 2000, "subcategory_to_product": 34},
                          measured=measured, comparisons=comparisons)
        result.update(
            purpose="accuracy reproduction; throughput provisional during concurrent work",
            provisional_documents_per_second=round(len(item_ids) / elapsed, 3),
            peak_gpu_memory_mb=round(torch.cuda.max_memory_allocated() / 1024**2, 2),
            environment={"python": platform.python_version(), "torch": torch.__version__,
                         "cuda_runtime": torch.version.cuda, "gpu": torch.cuda.get_device_name(0)},
        )
    except Exception as error:
        import traceback
        result.update(error_type=type(error).__name__, error=str(error), traceback=traceback.format_exc(limit=30))
    finally:
        result["finished_at"] = now()
        output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    print(json.dumps({"model_id": model_id, "status": result["status"],
                      "documents": result.get("document_count"), "measured": result.get("measured")}), flush=True)
    return result


def write_summary(results: list[dict]) -> None:
    rows = [row for row in results if row.get("status") == "passed"]
    lines = [
        "# Marqo Atlas 官方固定查询对照\n",
        "使用官方仓库固定的2,000个text query、34个subcategory query和完整78,370个候选商品。\n",
        "| 模型 | T2I R@1 官方/实测 | R@10 官方/实测 | MRR 官方/实测 | Subcat P@1 官方/实测 | P@10 官方/实测 | MRR 官方/实测 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    keys = [("text_to_image", "recall_at_1"), ("text_to_image", "recall_at_10"),
            ("text_to_image", "mrr"), ("subcategory_to_product", "precision_at_1"),
            ("subcategory_to_product", "precision_at_10"), ("subcategory_to_product", "mrr")]
    for row in rows:
        values = []
        for task, metric in keys:
            cell = row["comparisons"][task][metric]
            values.append(f"{100*cell['official']:.1f}%/{100*cell['measured']:.2f}%")
        lines.append(f"| {row['model_id']} | " + " | ".join(values) + " |")
    lines.append("\n官方表格仅保留3位小数；差值判断使用读取到的官方舍入值。并行吞吐不用于效率验收。\n")
    (_legacy('/data/zhoujianjie/environment_setup/native_workspace/evaluation_audit/Marqo_Atlas_official_comparison.md')).write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", action="append", choices=MODELS)
    parser.add_argument("--smoke-docs", type=int)
    args = parser.parse_args()
    selected = args.model_id or list(MODELS)
    results = []
    for index, model_id in enumerate(selected, 1):
        print(f"[{index}/{len(selected)}] RUN {model_id}", flush=True)
        results.append(evaluate(model_id, args.smoke_docs))
        if args.smoke_docs is None:
            write_summary(results)
    good = {"passed", "smoke_passed"}
    return 1 if any(row.get("status") not in good for row in results) else 0


if __name__ == "__main__":
    raise SystemExit(main())
