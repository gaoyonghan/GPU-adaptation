#!/usr/bin/env python3
"""Run the exact published MTEB Flickr30k T2I route for three SigLIP models."""

from __future__ import annotations
from model_paths import legacy as _legacy, legacy_str as _legacy_str, relative as _relative, revision as _revision, weight as _weight, result as _result

import argparse
import gc
import io
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq
from PIL import Image

import prepare_siglip_mteb_flickr30k_official as route


ROOT = Path(__file__).resolve().parents[1]
RESULT_ROOT = ROOT / "results/official/siglip_mteb_flickr30k_t2i"
PREFLIGHT_PATH = ROOT / "evaluation_audit/siglip_mteb_flickr30k_official_preflight.json"


def now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", action="append", choices=sorted(route.MODELS))
    parser.add_argument("--image-batch-size", type=int, default=16)
    parser.add_argument("--text-batch-size", type=int, default=256)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    return parser.parse_args()


def result_path(model_id: str) -> Path:
    return RESULT_ROOT.joinpath(*model_id.split("/"), "summary.json")


def print_mapping(preflight: dict[str, Any], model_ids: list[str]) -> None:
    print("SIGLIP MTEB FLICKR30K OFFICIAL MAPPING", flush=True)
    mapping = preflight["official_mapping"]
    dataset = mapping["dataset"]
    protocol = mapping["protocol"]
    for model_id in model_ids:
        model = mapping["models"][model_id]
        reference = model["official_metrics"]
        print(f"checkpoint: {model_id}@{model['revision']}", flush=True)
        print(f"dataset: {dataset['repo_id']}@{dataset['revision']} split={dataset['split']}", flush=True)
        print(f"protocol: {protocol['evaluator']}; RGB/exact processor; cosine", flush=True)
        print(
            "reference: "
            f"nDCG@10={reference['ndcg_at_10'] * 100:.3f}% "
            f"R@1/5/10={reference['recall_at_1'] * 100:.2f}%/"
            f"{reference['recall_at_5'] * 100:.2f}%/{reference['recall_at_10'] * 100:.2f}%",
            flush=True,
        )


def load_dataset() -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, str]]:
    corpus = pq.read_table(route.DATASET_ROOT / "corpus/corpus-00000-of-00001.parquet").to_pylist()
    queries = pq.read_table(route.DATASET_ROOT / "query/test-00000-of-00001.parquet").to_pylist()
    qrel_rows = pq.read_table(route.DATASET_ROOT / "qrels/test-00000-of-00001.parquet").to_pylist()
    relevant = {row["query-id"]: row["corpus-id"] for row in qrel_rows if row["score"] > 0}
    if len(relevant) != len(queries):
        raise RuntimeError("official qrels do not contain exactly one positive per query")
    return corpus, queries, relevant


def pooled(output: Any):
    return output.pooler_output if hasattr(output, "pooler_output") else output


def encode_images(model: Any, processor: Any, rows: list[dict[str, Any]], batch_size: int, device: str):
    import torch

    batches = []
    with torch.inference_mode():
        for start in range(0, len(rows), batch_size):
            images = []
            for row in rows[start:start + batch_size]:
                image_value = row["image"]
                image_bytes = image_value["bytes"] if isinstance(image_value, dict) else image_value.as_py()["bytes"]
                with Image.open(io.BytesIO(image_bytes)) as image:
                    images.append(image.convert("RGB"))
            inputs = processor(images=images, return_tensors="pt", padding=True)
            inputs = {key: value.to(device) for key, value in inputs.items()}
            batches.append(pooled(model.get_image_features(**inputs)).float().cpu())
            print(f"  images {min(start + batch_size, len(rows))}/{len(rows)}", flush=True)
    return torch.cat(batches)


def encode_text(model: Any, processor: Any, rows: list[dict[str, Any]], batch_size: int, device: str):
    import torch

    batches = []
    with torch.inference_mode():
        for start in range(0, len(rows), batch_size):
            texts = [row["text"] for row in rows[start:start + batch_size]]
            inputs = processor(
                text=texts,
                return_tensors="pt",
                padding="max_length",
                truncation=True,
            )
            inputs = {key: value.to(device) for key, value in inputs.items()}
            batches.append(pooled(model.get_text_features(**inputs)).float().cpu())
            print(f"  texts {min(start + batch_size, len(rows))}/{len(rows)}", flush=True)
    return torch.cat(batches)


def calculate_metrics(
    text_embeddings: Any,
    image_embeddings: Any,
    queries: list[dict[str, Any]],
    corpus: list[dict[str, Any]],
    relevant: dict[str, str],
) -> dict[str, float]:
    import torch
    import torch.nn.functional as functional

    text_embeddings = functional.normalize(text_embeddings.float(), dim=-1)
    image_embeddings = functional.normalize(image_embeddings.float(), dim=-1)
    scores = text_embeddings @ image_embeddings.T
    corpus_index = {row["id"]: index for index, row in enumerate(corpus)}
    relevant_indices = torch.tensor(
        [corpus_index[relevant[row["id"]]] for row in queries], dtype=torch.long
    )
    relevant_scores = scores[torch.arange(len(queries)), relevant_indices]
    # Equal cosine scores are practically absent. Counting strictly greater scores
    # gives the same rank as pytrec_eval for this one-positive-per-query task.
    ranks = (scores > relevant_scores[:, None]).sum(dim=1) + 1
    ranks_np = ranks.numpy()
    return {
        "ndcg_at_10": round(float(np.mean(np.where(ranks_np <= 10, 1.0 / np.log2(ranks_np + 1), 0.0))), 5),
        "recall_at_1": round(float(np.mean(ranks_np <= 1)), 5),
        "recall_at_5": round(float(np.mean(ranks_np <= 5)), 5),
        "recall_at_10": round(float(np.mean(ranks_np <= 10)), 5),
    }


def compare(measured: dict[str, float], reference: dict[str, float]) -> dict[str, Any]:
    rows: dict[str, Any] = {}
    for metric, official in reference.items():
        value = measured[metric]
        gap = value - official
        relative_gap = abs(gap) / official * 100
        rows[metric] = {
            "measured": value,
            "official": official,
            "absolute_gap": round(gap, 5),
            "relative_gap_percent": round(relative_gap, 4),
            "within_5_percent_relative": relative_gap <= 5.0,
        }
    return {
        "metrics": rows,
        "within_5_percent_relative_all_official_metrics": all(
            row["within_5_percent_relative"] for row in rows.values()
        ),
    }


def evaluate_one(
    model_id: str,
    corpus: list[dict[str, Any]],
    queries: list[dict[str, Any]],
    relevant: dict[str, str],
    image_batch_size: int,
    text_batch_size: int,
) -> dict[str, Any]:
    import torch
    from transformers import AutoModel, AutoProcessor

    started = time.monotonic()
    device = "cuda"
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the official SigLIP evaluation")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    spec = route.MODELS[model_id]
    local_model = _weight(model_id)
    print(f"loading {model_id}@{spec['revision']} from {local_model}", flush=True)
    model = AutoModel.from_pretrained(local_model, local_files_only=True).to(device).eval()
    processor = AutoProcessor.from_pretrained(local_model, local_files_only=True)
    torch.cuda.reset_peak_memory_stats()
    timings={"image_forward_seconds":0.0,"text_forward_seconds":0.0}
    for method,key in [("get_image_features","image_forward_seconds"),("get_text_features","text_forward_seconds")]:
        original=getattr(model,method)
        def timed(*a,_original=original,_key=key,**kw):
            torch.cuda.synchronize(); t=time.perf_counter()
            result=_original(*a,**kw)
            torch.cuda.synchronize(); timings[_key]+=time.perf_counter()-t
            return result
        setattr(model,method,timed)
    image_embeddings = encode_images(model, processor, corpus, image_batch_size, device)
    text_embeddings = encode_text(model, processor, queries, text_batch_size, device)
    output = RESULT_ROOT
    output.mkdir(parents=True,exist_ok=True)
    torch.save({"image_embeddings":image_embeddings,"text_embeddings":text_embeddings,"corpus_ids":[r["id"] for r in corpus],"query_ids":[r["id"] for r in queries],"qrels":relevant},output/"embeddings_and_targets.pt")
    with (output/"predictions.jsonl").open("w") as pred:
        im=torch.nn.functional.normalize(image_embeddings.float(),dim=-1)
        tx=torch.nn.functional.normalize(text_embeddings.float(),dim=-1)
        for start in range(0,len(queries),256):
            rankings=(tx[start:start+256]@im.T).topk(min(10,len(corpus)),dim=1).indices.tolist()
            for q,ids in zip(queries[start:start+256],rankings):
                pred.write(json.dumps({"query_id":q["id"],"target":relevant[q["id"]],"ranked_corpus_ids":[corpus[i]["id"] for i in ids]})+"\n")
    measured = calculate_metrics(text_embeddings, image_embeddings, queries, corpus, relevant)
    comparison = compare(measured, spec["reference"])
    accepted = comparison["within_5_percent_relative_all_official_metrics"]
    return {
        "status": "completed", "inference_timing": timings, "forward_ms_per_image": timings["image_forward_seconds"]*1000/len(corpus),
        "strict_official_acceptance": accepted,
        "model_id": model_id,
        "model_revision": spec["revision"],
        "dataset_id": route.DATASET_ID,
        "dataset_revision": route.DATASET_REVISION,
        "split": "test",
        "task": route.TASK_NAME,
        "protocol": "MTEB SiglipModelWrapper behavior; exact AutoProcessor; RGB; no instruction; cosine; FP32",
        "measured": measured,
        "official_reference": spec["reference"],
        "comparison": comparison,
        "samples": {"queries": len(queries), "corpus": len(corpus), "qrels": len(relevant)},
        "image_batch_size": image_batch_size,
        "text_batch_size": text_batch_size,
        "peak_gpu_memory_mib": round(torch.cuda.max_memory_allocated() / 1024**2, 3),
        "elapsed_seconds": round(time.monotonic() - started, 3),
        "completed_at": now(),
    }


def main() -> None:
    args = parse_args()
    model_ids = list(args.model_id or route.MODELS)
    preflight = route.build_preflight()
    atomic_json(PREFLIGHT_PATH, preflight)
    if preflight["preflight_decision"] != "pass" or preflight["hard_missing_items"]:
        raise RuntimeError(f"official preflight failed: {preflight['hard_missing_items']}")
    # This mapping is intentionally printed before importing/loading torch models.
    print_mapping(preflight, model_ids)
    if args.preflight_only:
        return

    corpus, queries, relevant = load_dataset()
    summaries = []
    for model_id in model_ids:
        path = result_path(model_id)
        if path.is_file() and not args.force:
            existing = json.loads(path.read_text(encoding="utf-8"))
            if existing.get("status") == "completed":
                print(f"skip completed: {model_id}", flush=True)
                summaries.append(existing)
                continue
        summary = evaluate_one(
            model_id,
            corpus,
            queries,
            relevant,
            args.image_batch_size,
            args.text_batch_size,
        )
        atomic_json(path, summary)
        summaries.append(summary)
        print(
            f"completed {model_id}: nDCG@10={summary['measured']['ndcg_at_10'] * 100:.3f}% "
            f"accepted={summary['strict_official_acceptance']}",
            flush=True,
        )
        del summary
        gc.collect()
        import torch

        torch.cuda.empty_cache()

    combined = {
        "status": "completed",
        "completed_at": now(),
        "models": summaries,
        "accepted": sum(item.get("strict_official_acceptance") is True for item in summaries),
        "total": len(summaries),
    }
    atomic_json(RESULT_ROOT / "summary.json", combined)


if __name__ == "__main__":
    main()
