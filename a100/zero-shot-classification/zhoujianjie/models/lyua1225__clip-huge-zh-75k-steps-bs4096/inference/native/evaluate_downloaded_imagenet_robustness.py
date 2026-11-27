#!/usr/bin/env python3
"""Evaluate mapped dual encoders on locally verified ImageNet robustness sets."""

from __future__ import annotations
from model_paths import legacy as _legacy, legacy_str as _legacy_str, relative as _relative, revision as _revision, weight as _weight, result as _result

import argparse
import csv
import gc
import hashlib
import io
import json
import platform
import subprocess
import sys
import time
import traceback
import zipfile
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "evaluation_audit"))
import evaluate_b01 as b01
from evaluate_pe_core_imagenet_a import imagenet_wnids

DATASET_KEYS = ("imagenetv2", "imagenet_a", "imagenet_r", "imagenet_sketch")
DATASET_INFO = {
    "imagenetv2": {"length": 10000, "revision": "d626240be2538720e83103a0e1178d24aca8b12c"},
    "imagenet_a": {"length": 7500, "revision": "1d04812197b88e02740e919e975bf113d6af0831"},
    "imagenet_r": {"length": 30000, "revision": "762a23f169e080bef0be7c9d96bf73a126cbb660"},
    "imagenet_sketch": {"length": 50889, "revision": "0d8b8b5380efe50baae075842505918ad43c88fe"},
}
RESULT_ROOT = ROOT / "results/imagenet_robustness"
REFERENCE_DATA = ROOT / "evaluation_audit/reference_code/perception_models/apps/pe/clip_benchmark/datasets"
WORKER_PYTHON = _legacy_str('/home/dell/anaconda3/envs/hengxiang_stage2_clean/bin/python')
MODEL_PYTHONS = {
    "HuggingFaceM4/siglip-so400m-14-980-flash-attn2-navit":
        _legacy_str('/home/dell/anaconda3/envs/UMind_agent/bin/python'),
}
NAVIT_MODEL_ID = "HuggingFaceM4/siglip-so400m-14-980-flash-attn2-navit"


def now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def protocol(dataset: str):
    all_names = json.loads((REFERENCE_DATA / "en_classnames.json").read_text())["imagenet1k"]
    templates = json.loads((REFERENCE_DATA / "en_zeroshot_classification_templates.json").read_text())["imagenet1k"]
    wnids = imagenet_wnids()
    name_by_wnid = dict(zip(wnids, all_names))
    if dataset == "imagenet_a":
        info = json.loads((ROOT / "datasets/evaluation/imagenet_a_hf/dataset_infos.json").read_text())
        subset = next(iter(info.values()))["features"]["label"]["names"]
    elif dataset == "imagenet_r":
        import pyarrow.parquet as pq
        present = set()
        for path in sorted((ROOT / "datasets/evaluation/imagenet_r_hf/test").glob("*.parquet")):
            present.update(pq.read_table(path, columns=["wnid"])["wnid"].to_pylist())
        subset = [wnid for wnid in wnids if wnid in present]
    else:
        subset = wnids
    return [name_by_wnid[wnid] for wnid in subset], templates, subset, {wnid: i for i, wnid in enumerate(subset)}


def classifier(adapter, classnames, templates, classes_per_chunk=4):
    import torch

    outputs = []
    for offset in range(0, len(classnames), classes_per_chunk):
        names = classnames[offset:offset + classes_per_chunk]
        prompts = [template.format(c=name) for name in names for template in templates]
        features = adapter.text_features(prompts).reshape(len(names), len(templates), -1).mean(dim=1)
        outputs.append(torch.nn.functional.normalize(features, dim=-1))
    return torch.cat(outputs)


class Navit384DiagnosticAdapter:
    """Use the publisher-documented 384px compatibility path for NaViT."""

    name = "audited local NaViT code with Google SigLIP 384px processor"

    def __init__(self):
        import torch
        from transformers import AutoModel, AutoProcessor

        model_dir = ROOT / "models/HuggingFaceM4/siglip-so400m-14-980-flash-attn2-navit"
        processor_dir = ROOT / "models/google/siglip-so400m-patch14-384"
        self.processor = AutoProcessor.from_pretrained(processor_dir, local_files_only=True)
        self.model = AutoModel.from_pretrained(
            model_dir, local_files_only=True, trust_remote_code=True,
            torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
        ).eval().cuda()

    def text_features(self, prompts):
        import torch

        values = self.processor(
            text=prompts, padding="max_length", truncation=True,
            max_length=64, return_tensors="pt",
        )
        values = {key: value.cuda(non_blocking=True) for key, value in values.items()}
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            features = self.model.get_text_features(**values)
        return torch.nn.functional.normalize(features.float(), dim=-1)

    def image_features(self, images):
        import torch

        values = self.processor(images=images, return_tensors="pt")
        pixels = values["pixel_values"].cuda(non_blocking=True)
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            features = self.model.get_image_features(pixel_values=pixels)
        return torch.nn.functional.normalize(features.float(), dim=-1)


def load_adapter(model_id: str, backend: str):
    if model_id == NAVIT_MODEL_ID:
        return Navit384DiagnosticAdapter()
    return b01.load_adapter(model_id, backend)


def sample_batches(dataset: str, label_by_wnid: dict[str, int], batch_size: int):
    from PIL import Image

    images, targets = [], []

    def emit(image, target):
        images.append(image); targets.append(target)
        if len(images) == batch_size:
            batch = (images[:], targets[:]); images.clear(); targets.clear(); return batch
        return None

    if dataset == "imagenetv2":
        root = ROOT / "datasets/evaluation/imagenetv2_hf/imagenetv2-matched-frequency-format-val"
        paths = sorted(root.glob("*/*.jpeg"), key=lambda p: (int(p.parent.name), p.name))
        for path in paths:
            with Image.open(path) as source: image = source.convert("RGB").copy()
            batch = emit(image, int(path.parent.name))
            if batch: yield batch
    elif dataset == "imagenet_a":
        paths = sorted((ROOT / "datasets/evaluation/imagenet_a_extracted").glob("*/*.jpg"))
        for path in paths:
            with Image.open(path) as source: image = source.convert("RGB").copy()
            batch = emit(image, label_by_wnid[path.parent.name])
            if batch: yield batch
    elif dataset == "imagenet_r":
        import pyarrow.parquet as pq
        for path in sorted((ROOT / "datasets/evaluation/imagenet_r_hf/test").glob("*.parquet")):
            for record_batch in pq.ParquetFile(path).iter_batches(batch_size=batch_size, columns=["image", "wnid"]):
                for row in record_batch.to_pylist():
                    value = row["image"]
                    with Image.open(io.BytesIO(value["bytes"])) as source: image = source.convert("RGB").copy()
                    batch = emit(image, label_by_wnid[row["wnid"]])
                    if batch: yield batch
    else:
        archive = ROOT / "datasets/evaluation/imagenet_sketch_hf/data/ImageNet-Sketch.zip"
        with zipfile.ZipFile(archive) as source:
            entries = sorted(name for name in source.namelist() if not name.endswith("/"))
            for name in entries:
                wnid = Path(name).parent.name
                with Image.open(io.BytesIO(source.read(name))) as raw: image = raw.convert("RGB").copy()
                batch = emit(image, label_by_wnid[wnid])
                if batch: yield batch
    if images:
        yield images, targets


def worker(args) -> int:
    import torch

    output = Path(args.worker_output)
    result = {"status": "failed", "model_id": args.worker_model_id, "dataset": args.dataset,
              "dataset_revision": DATASET_INFO[args.dataset]["revision"], "started_at": now(),
              "purpose": "uniform_diagnostic_not_official_metric_acceptance",
              "protocol": "PE bundled CLIP Benchmark ImageNet classnames + 80-template ensemble",
              "strict_official_acceptance": False,
              "same_checkpoint_official_reference": None}
    adapter = None
    try:
        adapter = load_adapter(args.worker_model_id, args.worker_backend)
        receipt = json.loads(
            (b01.model_dir(args.worker_model_id) / "download_receipt.json").read_text(encoding="utf-8")
        )
        result["model_revision"] = receipt.get("revision") or receipt.get("resolved_revision")
        result["backend"] = adapter.name
        classnames, templates, wnids, label_by_wnid = protocol(args.dataset)
        weights = classifier(adapter, classnames, templates)
        correct1 = correct5 = seen = 0
        started = time.perf_counter()
        torch.cuda.reset_peak_memory_stats()
        for images, targets in sample_batches(args.dataset, label_by_wnid, args.batch_size):
            features = adapter.image_features(images)
            predictions = (features @ weights.T).topk(5, dim=1).indices.cpu()
            target = torch.tensor(targets)
            correct1 += int((predictions[:, 0] == target).sum())
            correct5 += int((predictions == target[:, None]).any(dim=1).sum())
            seen += len(targets)
            if seen % 5000 < len(targets): print(f"encoded {seen:,} {args.dataset} images", flush=True)
        elapsed = time.perf_counter() - started
        if seen != DATASET_INFO[args.dataset]["length"]:
            raise RuntimeError(f"expected {DATASET_INFO[args.dataset]['length']} samples, found {seen}")
        result.update(status="passed", sample_count=seen, class_count=len(classnames), class_wnids=wnids,
                      prompt_templates_per_class=len(templates), top1_percent=round(100*correct1/seen, 4),
                      top5_percent=round(100*correct5/seen, 4), images_per_second=round(seen/elapsed, 3),
                      peak_gpu_memory_mb=round(torch.cuda.max_memory_allocated()/1024**2, 2),
                      environment={"python": platform.python_version(), "torch": torch.__version__,
                                   "cuda_runtime": torch.version.cuda, "gpu": torch.cuda.get_device_name(0)})
    except Exception as error:
        result.update(error_type=type(error).__name__, error=str(error), traceback=traceback.format_exc(limit=20))
    finally:
        result["finished_at"] = now(); output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
        del adapter; gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()
    print(json.dumps({"model_id": args.worker_model_id, "dataset": args.dataset,
                      "status": result["status"], "top1": result.get("top1_percent")}), flush=True)
    return 0 if result["status"] == "passed" else 1


def mapped_models(dataset: str):
    needle = {"imagenetv2": "ImageNetV2", "imagenet_a": "ImageNet-A",
              "imagenet_r": "ImageNet-R", "imagenet_sketch": "ImageNet-Sketch"}[dataset]
    rows = list(csv.DictReader((ROOT / "evaluation_audit/model_evaluation_mapping.csv").open(encoding="utf-8-sig")))
    wanted = {row["模型名称"] for row in rows if needle in row["补充测试集"]}
    return [row for row in b01.ledger_rows() if str(row["模型名称"]) in wanted]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def dataset_preflight(dataset: str) -> dict:
    manifest_path = ROOT / "datasets/evaluation/b03_download_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    names = {
        "imagenetv2": "ImageNetV2 matched-frequency",
        "imagenet_a": "ImageNet-A",
        "imagenet_r": "ImageNet-R",
        "imagenet_sketch": "ImageNet-Sketch",
    }
    matches = [item for item in manifest["datasets"] if item["name"] == names[dataset]]
    if len(matches) != 1:
        raise RuntimeError(f"missing unique frozen dataset manifest for {dataset}")
    record = matches[0]
    if record["revision"] != DATASET_INFO[dataset]["revision"]:
        raise RuntimeError(f"dataset revision mismatch for {dataset}")
    if record["rows"] != DATASET_INFO[dataset]["length"]:
        raise RuntimeError(f"dataset row-count mismatch for {dataset}")
    if not all(item.get("sha256_verified") is True for item in record["files"]):
        raise RuntimeError(f"dataset hash evidence is not verified for {dataset}")
    if dataset == "imagenetv2":
        root = ROOT / "datasets/evaluation/imagenetv2_hf/imagenetv2-matched-frequency-format-val"
        class_dirs = [path for path in root.iterdir() if path.is_dir()]
        images = list(root.glob("*/*.jpeg"))
        if len(class_dirs) != 1000 or len(images) != 10000:
            raise RuntimeError(
                f"extracted ImageNetV2 mismatch: {len(class_dirs)} classes, {len(images)} images"
            )
    return record


def checkpoint_mapping(model_id: str, backend: str, dataset: str) -> dict:
    receipt_path = b01.model_dir(model_id) / "download_receipt.json"
    if not receipt_path.is_file():
        raise RuntimeError(f"missing pinned download receipt: {model_id}")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    recorded_id = receipt.get("model_id")
    if recorded_id != model_id:
        raise RuntimeError(f"checkpoint ID mismatch for {model_id}: {recorded_id}")
    revision = receipt.get("revision") or receipt.get("resolved_revision")
    if not revision:
        raise RuntimeError(f"checkpoint revision missing for {model_id}")
    return {
        "model_id": model_id,
        "checkpoint_revision": revision,
        "backend": "audited_local_navit" if model_id == NAVIT_MODEL_ID else backend,
        "dataset": dataset,
        "dataset_revision": DATASET_INFO[dataset]["revision"],
        "protocol": "PE bundled CLIP Benchmark ImageNet classnames + 80-template ensemble",
        "metric": "top-1 and top-5 accuracy",
        "same_checkpoint_official_reference": None,
        "scope": "diagnostic_only_not_official_acceptance",
    }


def coordinator(args) -> int:
    requested_order = list(dict.fromkeys(args.model_id or []))
    requested = set(requested_order)
    all_rows = {str(row["模型名称"]): row for row in b01.ledger_rows()}
    missing = sorted(requested - set(all_rows))
    if missing:
        raise SystemExit(f"models absent from frozen ledger: {missing}")
    datasets = args.dataset or list(DATASET_KEYS)
    preflight_rows = []
    dataset_records = {}
    selected_by_dataset = {}
    for dataset in datasets:
        dataset_records[dataset] = dataset_preflight(dataset)
        rows = ([all_rows[model_id] for model_id in requested_order]
                if requested_order else mapped_models(dataset))
        selected_by_dataset[dataset] = rows
        preflight_rows.extend(
            checkpoint_mapping(str(row["模型名称"]), b01.backend_for(row), dataset)
            for row in rows
        )
    preflight = {
        "generated_at": now(),
        "purpose": "multi_model_diagnostic_preflight",
        "strict_official_acceptance": False,
        "datasets": dataset_records,
        "model_to_dataset_to_reference": preflight_rows,
        "status": "PASS_DIAGNOSTIC",
    }
    preflight_path = ROOT / "evaluation_audit/imagenet_robustness_diagnostic_preflight.json"
    preflight_path.write_text(json.dumps(preflight, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(preflight, ensure_ascii=False, indent=2), flush=True)
    if args.preflight_only:
        return 0
    failures = 0
    for dataset in datasets:
        rows = selected_by_dataset[dataset]
        for index, row in enumerate(rows, 1):
            model_id = str(row["模型名称"]); backend = b01.backend_for(row)
            output = RESULT_ROOT.joinpath(dataset, *model_id.split("/"), "summary.json")
            if output.is_file() and not args.force:
                old = json.loads(output.read_text(encoding="utf-8"))
                if old.get("status") == "passed":
                    print(f"[{dataset} {index}/{len(rows)}] SKIP {model_id}", flush=True); continue
            if backend not in {"Transformers AutoModel/AutoProcessor", "OpenCLIP"}:
                output.parent.mkdir(parents=True, exist_ok=True)
                output.write_text(json.dumps({"status":"unsupported_backend","model_id":model_id,
                    "dataset":dataset,"backend":backend,"finished_at":now()},ensure_ascii=False,indent=2)+"\n")
                failures += 1; continue
            worker_python = MODEL_PYTHONS.get(model_id, WORKER_PYTHON)
            command = [worker_python, str(Path(__file__).resolve()), "--dataset", dataset,
                       "--worker-model-id", model_id, "--worker-backend", backend,
                       "--worker-output", str(output), "--batch-size", str(args.batch_size)]
            print(f"[{dataset} {index}/{len(rows)}] RUN {model_id}", flush=True)
            try:
                completed = subprocess.run(command, cwd=ROOT, check=False, timeout=args.timeout_seconds)
                failures += int(completed.returncode != 0)
            except subprocess.TimeoutExpired:
                output.parent.mkdir(parents=True, exist_ok=True)
                output.write_text(json.dumps({"status":"timeout","model_id":model_id,
                    "dataset":dataset,"timeout_seconds":args.timeout_seconds,"finished_at":now()},
                    ensure_ascii=False,indent=2)+"\n")
                failures += 1
    return 1 if failures else 0


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", action="append", choices=DATASET_KEYS)
    parser.add_argument("--model-id", action="append")
    parser.add_argument("--batch-size", type=int, default=16)
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
        if not args.dataset or len(args.dataset) != 1:
            raise SystemExit("worker requires exactly one --dataset")
        args.dataset = args.dataset[0]
        return worker(args)
    return coordinator(args)


if __name__ == "__main__":
    raise SystemExit(main())
