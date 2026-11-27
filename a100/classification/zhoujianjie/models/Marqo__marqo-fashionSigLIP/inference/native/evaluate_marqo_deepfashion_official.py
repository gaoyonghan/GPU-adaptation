#!/usr/bin/env python3
"""Reproduce Marqo fixed-query DeepFashion In-Shop and Multimodal tasks."""

from __future__ import annotations

import argparse
import gc
import json
import platform
from datetime import datetime
from pathlib import Path

import evaluate_marqo_atlas_official as common


ROOT = Path(__file__).resolve().parents[1]
REFERENCE_REVISION = "d0b3bdfb62fb4964537cb582e76d6ddab0be8450"
DATASETS = {
    "inshop": {
        "name": "DeepFashion-InShop", "slug": "deepfashion_inshop",
        "path": ROOT / "datasets/evaluation/marqo_deepfashion_inshop_hf/data",
        "revision": "886fbe26ceac0b2bc8195d606e3fde641e7157f8", "documents": 52591,
        "gt": ROOT / "evaluation_audit/reference_code/marqo-FashionCLIP/data/DeepFashion-InShop/gt_query_doc",
        "tasks": {"text": 2000, "category2": 16, "category3": 8, "color": 804},
    },
    "multimodal": {
        "name": "DeepFashion-Multimodal", "slug": "deepfashion_multimodal",
        "path": ROOT / "datasets/evaluation/marqo_deepfashion_multimodal_hf/data",
        "revision": "21601c432a281c8a7deaa2afdda952b4b9ef3fd7", "documents": 42537,
        "gt": ROOT / "evaluation_audit/reference_code/marqo-FashionCLIP/data/DeepFashion-Multimodal/gt_query_doc",
        "tasks": {"text": 2000, "category2": 16},
    },
}
MODELS = {
    "patrickjohncyh/fashion-clip": {
        "path": ROOT / "models/patrickjohncyh/fashion-clip", "batch_size": 256,
        "backend": "transformers",
        "official": {
            "inshop": {"text": (.039, .175, .259), "category2": (.750, .738, .781),
                       "category3": (.875, .700, .875), "color": (.021, .024, .041)},
            "multimodal": {"text": (.003, .026, .008), "category2": (.688, .662, .729)},
        },
    },
    "Marqo/marqo-fashionCLIP": {
        "path": ROOT / "models/Marqo/marqo-fashionCLIP", "batch_size": 256,
        "official": {
            "inshop": {"text": (.048, .239, .331), "category2": (.625, .700, .722),
                       "category3": (.875, .762, .875), "color": (.035, .029, .055)},
            "multimodal": {"text": (.004, .024, .009), "category2": (.812, .681, .812)},
        },
    },
    "Marqo/marqo-fashionSigLIP": {
        "path": ROOT / "models/Marqo/marqo-fashionSigLIP", "batch_size": 192,
        "official": {
            "inshop": {"text": (.050, .248, .330), "category2": (.812, .787, .865),
                       "category3": (.875, .738, .875), "color": (.041, .034, .066)},
            "multimodal": {"text": (.007, .031, .015), "category2": (.812, .713, .859)},
        },
    },
}
common.MODELS = MODELS


def now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def evaluate(dataset_key: str, model_id: str) -> dict:
    import torch
    import torch.nn.functional as F

    data = DATASETS[dataset_key]
    spec = MODELS[model_id]
    common.DATASET = data["path"]
    output = ROOT.joinpath(f"results/marqo_{data['slug']}_official", *model_id.split("/"), "summary.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    result = {"status": "failed", "model_id": model_id, "dataset": data["name"],
              "dataset_revision": data["revision"], "reference_code_revision": REFERENCE_REVISION,
              "started_at": now()}
    model = None
    try:
        model, preprocess, tokenizer = common.load_local_model(model_id)
        torch.cuda.reset_peak_memory_stats()
        images, texts, item_ids, elapsed = common.encode_documents(
            model, preprocess, tokenizer, spec["batch_size"], None, data["name"]
        )
        if len(item_ids) != data["documents"]:
            raise RuntimeError(f"expected {data['documents']} documents, found {len(item_ids)}")
        products = F.normalize(.9 * images + .1 * texts, dim=-1)
        measured = {}
        for task, count in data["tasks"].items():
            suffix = "image" if task == "text" else "image+text"
            path = data["gt"] / f"ground_truth_{task}-{suffix}.json"
            qrels = json.loads(path.read_text(encoding="utf-8"))
            if len(qrels) != count:
                raise RuntimeError(f"{task}: expected {count} queries, found {len(qrels)}")
            retrieved = common.retrieve(list(qrels), images if task == "text" else products,
                                        item_ids, model, tokenizer)
            measured[task] = common.metrics(qrels, retrieved, "text_to_image" if task == "text" else task)
        comparisons = {}
        for task, values in measured.items():
            metric_names = ("recall_at_1", "recall_at_10", "mrr") if task == "text" else (
                "precision_at_1", "precision_at_10", "mrr")
            comparisons[task] = {}
            for metric, official in zip(metric_names, spec["official"][dataset_key][task]):
                value = values[metric]
                delta_pp = 100 * (value - official)
                comparisons[task][metric] = {
                    "official": official, "measured": value, "delta_percentage_points": delta_pp,
                    "relative_gap_percent": abs(value - official) / official * 100 if official else None,
                    "within_5_percentage_points": abs(delta_pp) <= 5,
                    "within_5_percent_relative": abs(value - official) / official <= .05 if official else None,
                }
        result.update(
            status="passed", document_count=len(item_ids), query_counts=data["tasks"],
            measured=measured, comparisons=comparisons,
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
    print(json.dumps({"dataset": dataset_key, "model_id": model_id, "status": result["status"],
                      "measured": result.get("measured")}), flush=True)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", action="append", choices=DATASETS)
    parser.add_argument("--model-id", action="append", choices=MODELS)
    args = parser.parse_args()
    selected_datasets = args.dataset or list(DATASETS)
    selected_models = args.model_id or list(MODELS)
    results = [evaluate(dataset, model) for dataset in selected_datasets for model in selected_models]
    return 0 if all(row["status"] == "passed" for row in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
