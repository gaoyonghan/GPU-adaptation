#!/usr/bin/env python3
"""Strict OpenCLIP zero-shot validation on downloaded official WDS test splits.

This wrapper verifies the checkpoint, dataset assets, protocol, and published
reference before delegating inference to the pinned upstream CLIP Benchmark.
Only acc1 has a same-checkpoint published reference in OpenCLIP's official CSV;
acc5 and mean_per_class_recall are retained as diagnostics.
"""

from __future__ import annotations
from model_paths import legacy as _legacy, legacy_str as _legacy_str, relative as _relative, revision as _revision, weight as _weight, result as _result

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "evaluation_audit/openclip_suite_test_only_manifest.json"
MANIFEST_BATCH4 = ROOT / "evaluation_audit/openclip_suite_test_only_batch4_manifest.json"
UPSTREAM = ROOT / "evaluation_audit/reference_code/CLIP_benchmark"
RESULT_DIR = ROOT / "evaluation_audit/results/openclip_new_suite_official"

MODELS = {
    "laion/CLIP-ViT-L-14-DataComp.XL-s13B-b90K": {
        "arch": "ViT-L-14",
        "pretrained": "datacomp_xl_s13b_b90k",
        "checkpoint": ROOT / "models/laion/CLIP-ViT-L-14-DataComp.XL-s13B-b90K/open_clip_pytorch_model.bin",
        "revision": "84c9828e63dc9a9351d1fe637c346d4c1c4db341",
        "checkpoint_sha256": "6509f07e6fc0da68f8e1ee881bf90803f0b053d2f7ed2013cc7c3a49ac4dd3db",
        "image_size": 224,
        "mean": [0.48145466, 0.4578275, 0.40821073],
        "std": [0.26862954, 0.26130258, 0.27577711],
    },
    "laion/CLIP-ViT-L-14-laion2B-s32B-b82K": {
        "arch": "ViT-L-14",
        "pretrained": "laion2b_s32b_b82k",
        "checkpoint": ROOT / "models/laion/CLIP-ViT-L-14-laion2B-s32B-b82K/open_clip_pytorch_model.safetensors",
        "revision": "1627032197142fbe2a7cfec626f4ced3ae60d07a",
        "checkpoint_sha256": "7d129ed747e0ed53e82dfcc140382b51be66b56e6a9bdc3258afd2846e3bb019",
        "image_size": 224,
        "mean": [0.5, 0.5, 0.5],
        "std": [0.5, 0.5, 0.5],
    },
    "laion/CLIP-ViT-H-14-laion2B-s32B-b79K": {
        "arch": "ViT-H-14",
        "pretrained": "laion2b_s32b_b79k",
        "checkpoint": ROOT / "models/laion/CLIP-ViT-H-14-laion2B-s32B-b79K/open_clip_model.safetensors",
        "revision": "1c2b8495b28150b8a4922ee1c8edee224c284c0c",
        "checkpoint_sha256": "0084e75319a50ad85ef45377bad5bc38f2f58824459eb690048d51c9f8863be5",
        "image_size": 224,
        "mean": [0.48145466, 0.4578275, 0.40821073],
        "std": [0.26862954, 0.26130258, 0.27577711],
    },
    "laion/CLIP-ViT-g-14-laion2B-s34B-b88K": {
        "arch": "ViT-g-14",
        "pretrained": "laion2b_s34b_b88k",
        "checkpoint": ROOT / "models/laion/CLIP-ViT-g-14-laion2B-s34B-b88K/open_clip_model.safetensors",
        "revision": "a20bc450485fc2dd335317234582df0e69d21b5a",
        "checkpoint_sha256": "0b890e533656dbadbf30b0f4f62a053b51185622ecb7e2c7e2c6232cd95d438a",
        "image_size": 224,
        "mean": [0.48145466, 0.4578275, 0.40821073],
        "std": [0.26862954, 0.26130258, 0.27577711],
    },
    "laion/CLIP-ViT-bigG-14-laion2B-39B-b160k": {
        "arch": "ViT-bigG-14",
        "pretrained": "laion2b_s39b_b160k",
        "checkpoint": ROOT / "models/laion/CLIP-ViT-bigG-14-laion2B-39B-b160k/open_clip_model.safetensors",
        "revision": "743c27bd53dfe508a0ade0f50698f99b39d03bec",
        "checkpoint_sha256": "db71bd27925917762c506c1b8a08e191d236e5586b5cd99f77d11aced578ec34",
        "image_size": 224,
        "mean": [0.48145466, 0.4578275, 0.40821073],
        "std": [0.26862954, 0.26130258, 0.27577711],
    },
    "laion/CLIP-convnext_xxlarge-laion2B-s34B-b82K-augreg-soup": {
        "arch": "convnext_xxlarge",
        "pretrained": "laion2b_s34b_b82k_augreg_soup",
        "checkpoint": ROOT / "models/laion/CLIP-convnext_xxlarge-laion2B-s34B-b82K-augreg-soup/open_clip_pytorch_model.bin",
        "revision": "9f3e8ee3f383c672388d9178afe70af9e63ac9df",
        "checkpoint_sha256": "d571d992c3fc1d9d6f771ad4eb0ae4cb63745d3bd0cf3363ecde5ff249e2f155",
        "image_size": 256,
        "mean": [0.48145466, 0.4578275, 0.40821073],
        "std": [0.26862954, 0.26130258, 0.27577711],
    },
    "laion/CLIP-ViT-B-32-laion2B-s34B-b79K": {
        "arch": "ViT-B-32",
        "pretrained": "laion2b_s34b_b79k",
        "checkpoint": ROOT / "models/laion/CLIP-ViT-B-32-laion2B-s34B-b79K/open_clip_model.safetensors",
        "revision": "1a25a446712ba5ee05982a381eed697ef9b435cf",
        "checkpoint_sha256": "ac4f8c4b88af6d963118cbf40ad93176d092abbedfcb752601ae1866352656e6",
        "image_size": 224,
        "mean": [0.48145466, 0.4578275, 0.40821073],
        "std": [0.26862954, 0.26130258, 0.27577711],
    },
    "laion/CLIP-ViT-B-16-laion2B-s34B-b88K": {
        "arch": "ViT-B-16",
        "pretrained": "laion2b_s34b_b88k",
        "checkpoint": ROOT / "models/laion/CLIP-ViT-B-16-laion2B-s34B-b88K/open_clip_model.safetensors",
        "revision": "7288da5a0d6f0b51c4a2b27c624837a9236d0112",
        "checkpoint_sha256": "3f25d29d3cc74e1d25d47e0593b4dd0864ced1e9b4d8e486a247ca4502f227f1",
        "image_size": 224,
        "mean": [0.48145466, 0.4578275, 0.40821073],
        "std": [0.26862954, 0.26130258, 0.27577711],
    },
    "timm/ViT-B-16-SigLIP": {
        "arch": "ViT-B-16-SigLIP", "pretrained": "webli",
        "checkpoint": ROOT / "models/timm/ViT-B-16-SigLIP/open_clip_model.safetensors",
        "revision": "41f575766f40e752fdd1383e9565b7f02388c1c4",
        "checkpoint_sha256": "81942ea8c09b9e41963357cca5d2682118bf7eb7491c2ae51a28bc4f4f95b194",
        "image_size": 224, "mean": [0.5, 0.5, 0.5], "std": [0.5, 0.5, 0.5],
    },
    "timm/ViT-L-16-SigLIP-384": {
        "arch": "ViT-L-16-SigLIP-384", "pretrained": "webli",
        "checkpoint": ROOT / "models/timm/ViT-L-16-SigLIP-384/open_clip_model.safetensors",
        "revision": "d8837d1669e859c65bc949b26c6050fb0fb82981",
        "checkpoint_sha256": "cb93064778814979fb20377f38dda9d451e4b5a50cd6e83bdecb9438c563ad88",
        "image_size": 384, "mean": [0.5, 0.5, 0.5], "std": [0.5, 0.5, 0.5],
    },
    "timm/ViT-SO400M-14-SigLIP": {
        "arch": "ViT-SO400M-14-SigLIP", "pretrained": "webli",
        "checkpoint": ROOT / "models/timm/ViT-SO400M-14-SigLIP/open_clip_model.safetensors",
        "revision": "9179d15177ece40964c50492136eda2f3e0c9f61",
        "checkpoint_sha256": "0a04a48b797e187a568335ab67e57a8958c32cee5a707a3feb6d4c97149dcd9c",
        "image_size": 224, "mean": [0.5, 0.5, 0.5], "std": [0.5, 0.5, 0.5],
    },
    "timm/ViT-SO400M-14-SigLIP-384": {
        "arch": "ViT-SO400M-14-SigLIP-384", "pretrained": "webli",
        "checkpoint": ROOT / "models/timm/ViT-SO400M-14-SigLIP-384/open_clip_model.safetensors",
        "revision": "ac16108d567c4389e6cd2b11c9b8585f7474435b",
        "checkpoint_sha256": "97faeafa0576c1ba4c92bae739ca06ef5f5d0c2db92f86a6086653d82a262043",
        "image_size": 384, "mean": [0.5, 0.5, 0.5], "std": [0.5, 0.5, 0.5],
    },
    "openai/clip-vit-large-patch14": {
        "arch": "ViT-L-14", "pretrained": "openai",
        "loader_arch": "ViT-L-14-quickgelu",
        "checkpoint": ROOT / "models/openai/clip-vit-large-patch14/ViT-L-14.pt",
        "revision": "b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836",
        "checkpoint_sha256": "b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836",
        "official_registry_url": "https://openaipublic.azureedge.net/clip/models/b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836/ViT-L-14.pt",
        "result_variant": "official_original",
        "image_size": 224,
        "mean": [0.48145466, 0.4578275, 0.40821073],
        "std": [0.26862954, 0.26130258, 0.27577711],
    },
}

DATASETS = {
    "caltech101": ("Caltech-101", "caltech101"),
    "gtsrb": ("GTSRB", "gtsrb"),
    "renderedsst2": ("Rendered SST2", "renderedsst2"),
    "stl10": ("STL-10", "stl10"),
    "svhn": ("SVHN", "svhn"),
    "flowers": ("Oxford Flowers-102", "flowers102"),
    "pets": ("Oxford-IIIT Pet", "oxford_pets"),
    "resisc45": ("RESISC45", "resisc45"),
    "cars": ("Stanford Cars", "stanford_cars"),
    "mnist": ("MNIST", "mnist"),
    "fgvc_aircraft": ("FGVC Aircraft", "fgvc_aircraft"),
    "imagenet_o": ("ImageNet-O", "imagenet_o"),
    "country211": ("Country211", "country211"),
    "clevr_count": ("CLEVR Counts", "clevr_count_all"),
    "clevr_distance": ("CLEVR Distance", "clevr_closest_object_distance"),
    "food101": ("Food-101", "food101"),
}

BATCH3_KEYS = [
    "caltech101", "gtsrb", "renderedsst2", "stl10", "svhn",
    "flowers", "pets", "resisc45", "cars", "mnist", "fgvc_aircraft",
]
BATCH4_KEYS = ["imagenet_o", "country211", "clevr_count", "clevr_distance", "food101"]

# Frozen from the official DataComp tasklist. Unlisted classification tasks use
# acc1; these four explicitly override the suite's main metric.
OFFICIAL_METRICS = {
    "caltech101": "mean_per_class_recall",
    "flowers": "mean_per_class_recall",
    "pets": "mean_per_class_recall",
    "fgvc_aircraft": "mean_per_class_recall",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def aggregate_sha256(path: Path) -> str:
    lines = []
    files = sorted(
        p for p in path.rglob("*")
        if p.is_file() and ".cache" not in p.parts and p.suffix != ".aria2"
    )
    for item in files:
        lines.append(f"{sha256_file(item)}  {_relative(item,ROOT)}\n")
    return hashlib.sha256("".join(lines).encode()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-key", required=True, choices=sorted(MODELS))
    parser.add_argument("--suite", choices=("batch3", "batch4"), default="batch3")
    parser.add_argument("--datasets", nargs="+", choices=sorted(DATASETS))
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()

    manifest_path = MANIFEST_BATCH4 if args.suite == "batch4" else MANIFEST
    data_root = ROOT / (
        "datasets/evaluation/openclip_suite_test_only_batch4"
        if args.suite == "batch4" else "datasets/evaluation/openclip_suite_test_only"
    )
    suite_keys = BATCH4_KEYS if args.suite == "batch4" else BATCH3_KEYS
    selected_keys = args.datasets or suite_keys
    outside = sorted(set(selected_keys) - set(suite_keys))
    if outside:
        raise SystemExit(f"datasets outside {args.suite}: {outside}")
    manifest = json.loads(manifest_path.read_text())
    manifest_ds = {
        entry.get("name", entry.get("official_name")): entry
        for entry in manifest["datasets"]
    }
    references = manifest["official_percent_references"].get(args.model_key)
    manifest_metrics = manifest.get("official_main_metric_by_dataset", {})
    model = MODELS[args.model_key]
    failures = []

    if not model["checkpoint"].is_file():
        failures.append(f"missing checkpoint: {model['checkpoint']}")
    elif sha256_file(model["checkpoint"]) != model["checkpoint_sha256"]:
        failures.append("checkpoint SHA256 mismatch")
    is_hf_clip = model.get("model_type") == "hf_clip"
    is_registry_checkpoint = "official_registry_url" in model
    config_path = model["checkpoint"].parent / (
        "preprocessor_config.json" if is_hf_clip else "open_clip_config.json"
    )
    local_config = None
    if is_registry_checkpoint:
        import open_clip

        local_config = open_clip.get_pretrained_cfg(model["arch"], model["pretrained"])
        if local_config.get("url") != model["official_registry_url"]:
            failures.append("official OpenCLIP registry URL mismatch")
        if local_config.get("mean") != tuple(model["mean"]):
            failures.append(f"official registry mean mismatch: {local_config.get('mean')} != {model['mean']}")
        if local_config.get("std") != tuple(model["std"]):
            failures.append(f"official registry std mismatch: {local_config.get('std')} != {model['std']}")
        config_path = Path(open_clip.__file__).resolve().parent / "pretrained.py"
    elif not config_path.is_file():
        failures.append(f"missing checkpoint-native preprocessing config: {config_path}")
    else:
        local_config = json.loads(config_path.read_text())
        preprocess_cfg = local_config if is_hf_clip else local_config.get("preprocess_cfg", {})
        vision_cfg = (
            {"image_size": preprocess_cfg.get("crop_size")}
            if is_hf_clip else local_config.get("model_cfg", {}).get("vision_cfg", {})
        )
        if is_hf_clip:
            preprocess_cfg = {
                "mean": preprocess_cfg.get("image_mean"),
                "std": preprocess_cfg.get("image_std"),
            }
        if preprocess_cfg.get("mean") != model["mean"]:
            failures.append(
                f"checkpoint-native mean mismatch: {preprocess_cfg.get('mean')} != {model['mean']}"
            )
        if preprocess_cfg.get("std") != model["std"]:
            failures.append(
                f"checkpoint-native std mismatch: {preprocess_cfg.get('std')} != {model['std']}"
            )
        if vision_cfg.get("image_size") != model["image_size"]:
            failures.append(
                f"checkpoint-native image size mismatch: {vision_cfg.get('image_size')} != {model['image_size']}"
            )
    if references is None:
        failures.append("missing same-checkpoint official reference row")

    mappings = []
    for key in selected_keys:
        reference_name, local_name = DATASETS[key]
        local_path = data_root / local_name
        entry = manifest_ds.get(reference_name)
        ref = None if references is None else references.get(reference_name)
        official_metric = OFFICIAL_METRICS.get(key, "acc1")
        missing = []
        if entry is None:
            missing.append("dataset manifest entry")
        if ref is None:
            missing.append("published acc1 reference")
        if args.suite == "batch3" and manifest_metrics.get(reference_name) != official_metric:
            missing.append("official main-metric mapping")
        for required in (local_path / "test/nshards.txt", local_path / "classnames.txt", local_path / "zeroshot_classification_templates.txt"):
            if not required.is_file():
                missing.append(str(required))
        if list(local_path.glob("train/*")):
            missing.append("unexpected train files")
        if entry is not None and not missing:
            if "aggregate_sha256" in entry:
                observed = aggregate_sha256(local_path)
                if observed != entry["aggregate_sha256"]:
                    missing.append(f"aggregate SHA256 mismatch ({observed})")
            else:
                for frozen in entry.get("files", []):
                    shard = local_path / "test" / frozen["name"]
                    if not shard.is_file() or sha256_file(shard) != frozen["sha256"]:
                        missing.append(f"test shard SHA256 mismatch: {frozen['name']}")
        mappings.append({
            "model": args.model_key,
            "checkpoint_revision": model["revision"],
            "checkpoint_sha256": model["checkpoint_sha256"],
            "checkpoint_config": str(config_path),
            "loader_architecture": model.get("loader_arch", model["arch"]),
            "checkpoint_native_preprocessing": {
                "image_size": model["image_size"],
                "mean": model["mean"],
                "std": model["std"],
                "quick_gelu": bool(model.get("loader_arch", "").endswith("-quickgelu")),
            },
            "dataset": reference_name,
            "dataset_revision": None if entry is None else entry["revision"],
            "dataset_path": str(local_path),
            "protocol": "pinned CLIP Benchmark with local-checkpoint metadata adapter, WDS test, embedded classnames/prompts, checkpoint-native preprocessing, AMP",
            "published_metric": f"{official_metric}_percent",
            "reference": ref,
            "diagnostics_only": [
                metric for metric in ("acc5", "mean_per_class_recall")
                if metric != official_metric
            ],
            "status": "RUNNABLE" if not missing else "BLOCKED",
            "missing": missing,
        })
        failures.extend(f"{reference_name}: {item}" for item in missing)

    print(json.dumps({"preflight": mappings}, ensure_ascii=False, indent=2), flush=True)
    if failures:
        print(json.dumps({"hard_failure": failures}, ensure_ascii=False, indent=2), file=sys.stderr)
        return 2
    if args.preflight_only:
        return 0

    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    result_name = f"{model['arch']}__{model['pretrained']}"
    if model.get("result_variant"):
        result_name += f"__{model['result_variant']}"
    result_prefix = re.sub(r"[^a-z0-9]+", "_", result_name.lower()).strip("_")
    env = os.environ.copy()
    env["PYTHONPATH"] = str(UPSTREAM) + os.pathsep + env.get("PYTHONPATH", "")
    results = []
    for mapping, key in zip(mappings, selected_keys):
        reference_name, local_name = DATASETS[key]
        dataset_root = data_root / local_name
        nshards = int((dataset_root / "test/nshards.txt").read_text().strip())
        effective_workers = min(args.num_workers, nshards)
        output = RESULT_DIR / f"{result_prefix}__{key}.json"
        if not (args.skip_existing and output.exists()):
            command = [
                sys.executable, "-m", "clip_benchmark.cli", "eval",
                "--model_type", model.get("model_type", "open_clip"),
                "--model", model.get("loader_arch", model["arch"]),
                "--pretrained", str(model.get("checkpoint_dir", model["checkpoint"])),
                "--dataset", f"wds/{key}",
                "--dataset_root", str(dataset_root),
                "--split", "test",
                "--task", "zeroshot_classification",
                "--batch_size", str(args.batch_size),
                "--num_workers", str(effective_workers),
                "--dump_classnames", "--dump_templates",
                "--output", str(output),
            ]
            print("RUN", " ".join(command), flush=True)
            subprocess.run(command, cwd=UPSTREAM, env=env, check=True)
        payload = json.loads(output.read_text())
        metric_name = mapping["published_metric"].removesuffix("_percent")
        measured = payload["metrics"][metric_name] * 100.0
        reference = mapping["reference"]
        relative_error = abs(measured - reference) / reference * 100.0
        results.append({
            **mapping,
            "measured_official_metric_percent": measured,
            "relative_error_percent": relative_error,
            "strict_pass": relative_error <= 5.0,
            "diagnostic_metrics": {
                name + "_percent": value * 100.0
                for name, value in payload["metrics"].items() if name != metric_name
            },
            "raw_result": str(output),
        })
        print(json.dumps(results[-1], ensure_ascii=False, indent=2), flush=True)

    summary_path = RESULT_DIR / f"{result_prefix}__summary.json"
    merged_by_dataset = {}
    for candidate in RESULT_DIR.glob("*__summary.json"):
        previous = json.loads(candidate.read_text())
        previous_results = previous.get("results", [])
        if previous.get("model") != args.model_key or not previous_results:
            continue
        if any(item.get("checkpoint_sha256") != model["checkpoint_sha256"] for item in previous_results):
            continue
        for item in previous_results:
            dataset_name = item["dataset"]
            dataset_key = next((k for k, value in DATASETS.items() if value[0] == dataset_name), None)
            metric_name = OFFICIAL_METRICS.get(dataset_key, "acc1")
            if metric_name == "acc1":
                measured = item.get("measured_official_metric_percent", item.get("measured_acc1_percent"))
            else:
                measured = item.get("measured_official_metric_percent")
                if item.get("published_metric") != f"{metric_name}_percent":
                    measured = item.get("diagnostic_metrics", {}).get(f"{metric_name}_percent")
            if measured is None:
                raise RuntimeError(f"cannot migrate official metric for {dataset_name} in {candidate}")
            item["published_metric"] = f"{metric_name}_percent"
            item["measured_official_metric_percent"] = measured
            item.pop("measured_acc1_percent", None)
            item["relative_error_percent"] = abs(measured - item["reference"]) / item["reference"] * 100.0
            item["strict_pass"] = item["relative_error_percent"] <= 5.0
            merged_by_dataset[dataset_name] = item
    merged_by_dataset.update({item["dataset"]: item for item in results})
    results = list(merged_by_dataset.values())
    all_pass = all(item["strict_pass"] for item in results)
    summary = {
        "generated_at": datetime.now().astimezone().isoformat(),
        "acceptance_rule": "same checkpoint/dataset/protocol/metric; relative error <= 5%; every published metric must pass",
        "model": args.model_key,
        "scope": "all downloaded official datasets evaluated so far; not the model's complete published benchmark suite",
        "all_evaluated_published_metrics_pass": all_pass,
        "model_overall_status": (
            "partial_official_suite" if all_pass else "partial_official_suite_failed"
        ),
        "results": results,
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    print(f"SUMMARY {summary_path}", flush=True)
    return 0 if all_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
