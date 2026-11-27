#!/usr/bin/env python3
"""Verify the exact MTEB Flickr30k route for three downloaded SigLIP models."""

from __future__ import annotations
from model_paths import legacy as _legacy, legacy_str as _legacy_str, relative as _relative, revision as _revision, weight as _weight, result as _result

import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = ROOT / "datasets/evaluation/mteb_flickr30kt2i_official"
OUTPUT = ROOT / "evaluation_audit/siglip_mteb_flickr30k_official_preflight.json"
DATASET_ID = "mteb/flickr30kt2i"
DATASET_REVISION = "e819702b287bfbe084e129a61f308a802b7c108e"
MTEB_RESULT_VERSION = "1.14.15"
MTEB_RESULTS_MIRROR_REVISION = "7e68306607cab0538a7fd7e9e0f2e5708feb1491"
TASK_NAME = "Flickr30kT2IRetrieval"

DATA_FILES = {
    "corpus/corpus-00000-of-00001.parquet": {
        "rows": 1000,
        "sha256": "ae65cca5561d68c305d6d8e2ee0980964f555a73ad24bd5a504c4e342113c24b",
        "columns": ["id", "modality", "image"],
    },
    "query/test-00000-of-00001.parquet": {
        "rows": 5000,
        "sha256": "b4fec243cfae9154b3cea274f780a09041ac50dbdbb5dca3d6802d7743437643",
        "columns": ["id", "modality", "text"],
    },
    "qrels/test-00000-of-00001.parquet": {
        "rows": 5000,
        "sha256": "9a52f5390672770d6d9a506ac8d4b850df21e2a16805162baee0a5c440826beb",
        "columns": ["query-id", "corpus-id", "score"],
    },
}

MODELS = {
    "google/siglip-base-patch16-256-multilingual": {
        "revision": "8952a4eafcde3cb7ab46b1dd629b33f8784ca9c6",
        "reference_file": "siglip-base-patch16-256-multilingual__Flickr30kT2IRetrieval.json",
        "reference_url": "https://github.com/embeddings-benchmark/results/blob/main/results/google__siglip-base-patch16-256-multilingual/8952a4eafcde3cb7ab46b1dd629b33f8784ca9c6/Flickr30kT2IRetrieval.json",
        "reference_sha256": "92e884c88237beda5202d5a7e58acec603e82c2b99a0bc0b634da7f60e331f3b",
        "reference": {"ndcg_at_10": 0.83383, "recall_at_1": 0.7192, "recall_at_5": 0.9016, "recall_at_10": 0.9468},
    },
    "google/siglip-so400m-patch14-224": {
        "revision": "d04cf29fca7b6374f74d8bea1969314492266b5e",
        "reference_file": "siglip-so400m-patch14-224__Flickr30kT2IRetrieval.json",
        "reference_url": "https://github.com/embeddings-benchmark/results/blob/main/results/google__siglip-so400m-patch14-224/d04cf29fca7b6374f74d8bea1969314492266b5e/Flickr30kT2IRetrieval.json",
        "reference_sha256": "a3673bde4bae6734762c2acfef62093307c6f6cf33785d8b3e7bc7d25e3963da",
        "reference": {"ndcg_at_10": 0.85811, "recall_at_1": 0.7522, "recall_at_5": 0.9246, "recall_at_10": 0.9544},
    },
    "google/siglip-so400m-patch16-256-i18n": {
        "revision": "365d321c0cfdea96bc28e3a29787a11a062681a1",
        "reference_file": "siglip-so400m-patch16-256-i18n__Flickr30kT2IRetrieval.json",
        "reference_url": "https://github.com/embeddings-benchmark/results/blob/main/results/google__siglip-so400m-patch16-256-i18n/365d321c0cfdea96bc28e3a29787a11a062681a1/Flickr30kT2IRetrieval.json",
        "reference_sha256": "c566ae969b31c67edec9a124fcdc05100ad365322b08d883dff7f9f156d4e7a7",
        "reference": {"ndcg_at_10": 0.87146, "recall_at_1": 0.7732, "recall_at_5": 0.934, "recall_at_10": 0.9608},
    },
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_data() -> dict[str, Any]:
    records: dict[str, Any] = {}
    for relative_path, expected in DATA_FILES.items():
        path = DATASET_ROOT / relative_path
        if not path.is_file():
            raise FileNotFoundError(f"official MTEB data file missing: {path}")
        parquet = pq.ParquetFile(path)
        columns = parquet.schema_arrow.names
        actual_sha256 = sha256(path)
        if parquet.metadata.num_rows != expected["rows"]:
            raise RuntimeError(f"row count mismatch for {relative_path}")
        if columns != expected["columns"]:
            raise RuntimeError(f"schema mismatch for {relative_path}: {columns}")
        if actual_sha256 != expected["sha256"]:
            raise RuntimeError(f"SHA-256 mismatch for {relative_path}")
        records[relative_path] = {
            "rows": parquet.metadata.num_rows,
            "columns": columns,
            "bytes": path.stat().st_size,
            "sha256": actual_sha256,
        }

    corpus = pq.read_table(DATASET_ROOT / "corpus/corpus-00000-of-00001.parquet", columns=["id", "modality"])
    queries = pq.read_table(DATASET_ROOT / "query/test-00000-of-00001.parquet", columns=["id", "modality"])
    qrels = pq.read_table(DATASET_ROOT / "qrels/test-00000-of-00001.parquet").to_pydict()
    corpus_ids = set(corpus.column("id").to_pylist())
    query_ids = set(queries.column("id").to_pylist())
    if set(corpus.column("modality").to_pylist()) != {"image"}:
        raise RuntimeError("official corpus contains a non-image modality")
    if set(queries.column("modality").to_pylist()) != {"text"}:
        raise RuntimeError("official queries contain a non-text modality")
    if set(qrels["query-id"]) != query_ids or not set(qrels["corpus-id"]) <= corpus_ids:
        raise RuntimeError("qrels IDs do not match the official corpus/query tables")
    if set(qrels["score"]) != {1}:
        raise RuntimeError("unexpected qrels relevance values")
    if len(set(zip(qrels["query-id"], qrels["corpus-id"]))) != len(qrels["query-id"]):
        raise RuntimeError("duplicate query/corpus rows in qrels")
    return records


def verify_models() -> dict[str, Any]:
    records: dict[str, Any] = {}
    reference_root = ROOT / "evaluation_audit/reference_results/mteb"
    for model_id, spec in MODELS.items():
        model_root = ROOT / "models" / model_id
        receipt_path = model_root / "download_receipt.json"
        if not receipt_path.is_file():
            raise FileNotFoundError(f"model receipt missing: {receipt_path}")
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        if receipt.get("model_id") != model_id or receipt.get("revision") != spec["revision"]:
            raise RuntimeError(f"model/revision mismatch for {model_id}")
        for required in ("config.json", "model.safetensors", "preprocessor_config.json", "tokenizer_config.json"):
            if not (model_root / required).is_file():
                raise FileNotFoundError(f"checkpoint file missing: {model_root / required}")

        reference_path = reference_root / spec["reference_file"]
        if sha256(reference_path) != spec["reference_sha256"]:
            raise RuntimeError(f"official result JSON hash mismatch for {model_id}")
        official_result = json.loads(reference_path.read_text(encoding="utf-8"))
        score = official_result["scores"]["test"][0]
        if official_result.get("dataset_revision") != DATASET_REVISION:
            raise RuntimeError(f"official result dataset revision mismatch for {model_id}")
        if official_result.get("mteb_version") != MTEB_RESULT_VERSION:
            raise RuntimeError(f"official result MTEB version mismatch for {model_id}")
        if official_result.get("task_name") != TASK_NAME:
            raise RuntimeError(f"official result task mismatch for {model_id}")
        for metric, expected in spec["reference"].items():
            if float(score[metric]) != expected:
                raise RuntimeError(f"official result metric mismatch for {model_id}/{metric}")
        if float(score["main_score"]) != spec["reference"]["ndcg_at_10"]:
            raise RuntimeError(f"official main_score mismatch for {model_id}")
        records[model_id] = {
            "revision": spec["revision"],
            "local_dir": str(_relative(model_root,ROOT)),
            "reference_result": str(_relative(reference_path,ROOT)),
            "reference_url_with_model_revision": spec["reference_url"],
            "reference_result_sha256": spec["reference_sha256"],
            "official_metrics": spec["reference"],
        }
    return records


def build_preflight() -> dict[str, Any]:
    data = verify_data()
    models = verify_models()
    return {
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "status": "ready_for_gpu_evaluation",
        "preflight_decision": "pass",
        "hard_missing_items": [],
        "official_mapping": {
            "models": models,
            "official_results_repository": {
                "github": "https://github.com/embeddings-benchmark/results",
                "huggingface_mirror": "mteb/results",
                "huggingface_mirror_revision": MTEB_RESULTS_MIRROR_REVISION,
                "path_contract": "results/{model_id with / replaced by __}/{model_revision}/Flickr30kT2IRetrieval.json",
            },
            "dataset": {
                "repo_id": DATASET_ID,
                "revision": DATASET_REVISION,
                "split": "test",
                "local_dir": str(_relative(DATASET_ROOT,ROOT)),
                "files": data,
            },
            "protocol": {
                "evaluator": f"MTEB {MTEB_RESULT_VERSION} {TASK_NAME}",
                "task": "caption-to-image retrieval; 5,000 caption queries against 1,000 images",
                "text": "AutoProcessor(text=..., padding='max_length', truncation=True); no instruction because SigLIP ModelMeta use_instructions=False",
                "image": "PIL image converted to RGB, then the exact checkpoint AutoProcessor",
                "embedding": "AutoModel.get_text_features / get_image_features in eval mode and FP32",
                "similarity": "cosine",
                "metric": "nDCG@10 (main_score), plus Recall@1/5/10 diagnostics with official references",
            },
        },
        "acceptance_rule": "All four published metrics must be within 5% relative error on the exact official route.",
    }


def main() -> None:
    output = build_preflight()
    OUTPUT.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("SIGLIP MTEB FLICKR30K OFFICIAL PREFLIGHT")
    for model_id, model in output["official_mapping"]["models"].items():
        ref = model["official_metrics"]
        print(
            f"{model_id}@{model['revision']} -> {DATASET_ID}@{DATASET_REVISION} test -> "
            f"{TASK_NAME}/cosine/nDCG@10 -> {ref['ndcg_at_10'] * 100:.3f}%"
        )
    print("status: pass")


if __name__ == "__main__":
    main()
