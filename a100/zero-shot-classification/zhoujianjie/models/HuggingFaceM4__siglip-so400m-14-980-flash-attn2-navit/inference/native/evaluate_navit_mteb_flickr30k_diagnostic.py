#!/usr/bin/env python3
"""Compare the NaViT conversion with its released 384px parent on Flickr30k.

This is a user-authorized diagnostic.  The dataset and retrieval protocol are
official MTEB assets, but the NaViT checkpoint has no same-revision reference,
so this result can never change the strict official acceptance count.
"""

from __future__ import annotations

import gc
import json
import time
from datetime import datetime
from pathlib import Path

import evaluate_siglip_mteb_flickr30k_official as flickr
import prepare_siglip_mteb_flickr30k_official as route


ROOT = Path(__file__).resolve().parents[1]
MODEL_ID = "HuggingFaceM4/siglip-so400m-14-980-flash-attn2-navit"
MODEL_REVISION = "d66538faeba44480d0bfaa42145eef26f9423199"
PARENT_ID = "google/siglip-so400m-patch14-384"
PARENT_REVISION = "9fdffc58afc957d1a03a25b10dba0329ab15c2a3"
OUT = ROOT / "results/proxy_diagnostics/enhanced/navit_mteb_flickr30k/summary.json"
PREFLIGHT = ROOT / "evaluation_audit/navit_mteb_flickr30k_diagnostic_preflight.json"


def now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def receipt_revision(model_id: str) -> str | None:
    path = ROOT / "models" / model_id / "download_receipt.json"
    value = json.loads(path.read_text(encoding="utf-8"))
    return value.get("revision") or value.get("resolved_revision")


def build_preflight() -> dict:
    dataset_preflight = route.build_preflight()
    errors = []
    if dataset_preflight.get("preflight_decision") != "pass":
        errors.append("MTEB Flickr30k official data preflight failed")
    if receipt_revision(MODEL_ID) != MODEL_REVISION:
        errors.append("NaViT checkpoint revision mismatch")
    if receipt_revision(PARENT_ID) != PARENT_REVISION:
        errors.append("parent checkpoint revision mismatch")
    payload = {
        "generated_at": now(),
        "status": "PASS_DIAGNOSTIC_PREFLIGHT" if not errors else "HARD_FAIL",
        "errors": errors,
        "model_to_dataset_to_reference": {
            "checkpoint": f"{MODEL_ID}@{MODEL_REVISION}",
            "comparison_checkpoint": f"{PARENT_ID}@{PARENT_REVISION}",
            "dataset": f"{route.DATASET_ID}@{route.DATASET_REVISION}",
            "split": "test: 5,000 caption queries against 1,000 images",
            "protocol": "MTEB Flickr30kT2IRetrieval; exact 384px processor; cosine",
            "metrics": "nDCG@10 and Recall@1/5/10",
            "same_checkpoint_official_reference": None,
        },
        "strict_official_acceptance": False,
        "user_decision": "run a frozen substitute/variant comparison despite the missing same-checkpoint reference",
    }
    flickr.atomic_json(PREFLIGHT, payload)
    print("NAVIT DIAGNOSTIC PREFLIGHT MODEL->DATASET->REFERENCE", flush=True)
    print(json.dumps(payload, ensure_ascii=False, indent=2), flush=True)
    if errors:
        raise RuntimeError("; ".join(errors))
    return payload


def evaluate(model_id: str, trust_remote_code: bool, corpus, queries, relevant) -> dict:
    import torch
    from transformers import AutoModel, AutoProcessor

    local_model = ROOT / "models" / model_id
    processor_dir = ROOT / "models" / PARENT_ID
    started = time.monotonic()
    model = AutoModel.from_pretrained(
        local_model,
        local_files_only=True,
        trust_remote_code=trust_remote_code,
    ).cuda().eval()
    processor = AutoProcessor.from_pretrained(processor_dir, local_files_only=True)
    torch.cuda.reset_peak_memory_stats()
    images = flickr.encode_images(model, processor, corpus, 16, "cuda")
    texts = flickr.encode_text(model, processor, queries, 256, "cuda")
    measured = flickr.calculate_metrics(texts, images, queries, corpus, relevant)
    result = {
        "model_id": model_id,
        "model_revision": receipt_revision(model_id),
        "measured": measured,
        "elapsed_seconds": round(time.monotonic() - started, 3),
        "peak_gpu_memory_mib": round(torch.cuda.max_memory_allocated() / 1024**2, 3),
    }
    del texts, images, processor, model
    gc.collect()
    torch.cuda.empty_cache()
    return result


def main() -> None:
    preflight = build_preflight()
    corpus, queries, relevant = flickr.load_dataset()
    parent = evaluate(PARENT_ID, False, corpus, queries, relevant)
    navit = evaluate(MODEL_ID, True, corpus, queries, relevant)
    comparison = {}
    for metric, parent_value in parent["measured"].items():
        value = navit["measured"][metric]
        comparison[metric] = {
            "navit": value,
            "parent_384": parent_value,
            "absolute_gap": round(value - parent_value, 5),
            "relative_gap_percent": round(abs(value - parent_value) / parent_value * 100, 4),
        }
    within_five = all(row["relative_gap_percent"] <= 5 for row in comparison.values())
    result = {
        "status": "completed_diagnostic_only",
        "completed_at": now(),
        "purpose": "user-authorized variant comparison",
        "strict_official_acceptance": False,
        "changes_strict_acceptance_count": False,
        "suggested_user_relaxed_acceptance": within_five,
        "reason_not_strict": "the NaViT checkpoint has no same-revision official reference value",
        "preflight": preflight,
        "dataset": {
            "id": route.DATASET_ID,
            "revision": route.DATASET_REVISION,
            "split": "test",
            "queries": len(queries),
            "images": len(corpus),
        },
        "parent": parent,
        "navit": navit,
        "comparison": comparison,
    }
    flickr.atomic_json(OUT, result)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
