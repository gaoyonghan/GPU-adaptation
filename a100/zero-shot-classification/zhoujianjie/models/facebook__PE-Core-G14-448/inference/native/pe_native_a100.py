#!/usr/bin/env python3
"""Evaluate local PE-Core checkpoints on the official ImageNet-A test set."""

from __future__ import annotations
from model_paths import legacy as _legacy, legacy_str as _legacy_str, relative as _relative, revision as _revision, weight as _weight, result as _result

import argparse
import ast
import gc
import io
import json
import platform
import sys
import time
from datetime import datetime
from pathlib import Path


ROOT = _legacy('/data/zhoujianjie/model_inference_for_GPU_adaptation')
REFERENCE = _legacy('/data/zhoujianjie/environment_setup/native_sources/evaluation_audit/reference_code/perception_models')
DATASET_ROOT = _legacy('/data/zhoujianjie/model_inference_for_GPU_adaptation/dataset/raw/imagenet_a_1')
EXTRACTED_ROOT = _legacy('/data/zhoujianjie/model_inference_for_GPU_adaptation/dataset/raw/imagenet_a_2')
RESULT_ROOT = _legacy('/data/zhoujianjie/model_inference_for_GPU_adaptation/results/pe_core_imagenet_a')
SUMMARY = _legacy('/data/zhoujianjie/model_inference_for_GPU_adaptation/evaluation_audit/PE_Core_ImageNetA_official_comparison.md')
REFERENCE_REVISION = "3e352cca660658d4b5c90f42a7808b11469e4c66"
MODELS = {
    "facebook/PE-Core-L14-336": {
        "config": "PE-Core-L14-336",
        "checkpoint": _legacy('/data/zhoujianjie/model_inference_for_GPU_adaptation/classification/zhoujianjie/models/facebook__PE-Core-L14-336/weights/PE-Core-L14-336.pt'),
        "official_top1": 89.0,
        "batch_size": 64,
    },
    "facebook/PE-Core-G14-448": {
        "config": "PE-Core-G14-448",
        "checkpoint": _legacy('/data/zhoujianjie/model_inference_for_GPU_adaptation/classification/zhoujianjie/models/facebook__PE-Core-G14-448/weights/PE-Core-G14-448.pt'),
        "official_top1": 92.6,
        "batch_size": 8,
    },
}


def now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def imagenet_wnids() -> list[str]:
    source = _legacy('/data/zhoujianjie/environment_setup/native_sources/evaluation_audit/reference_code/perception_models/apps/pe/clip_benchmark/datasets/builder.py')
    module = ast.parse(source.read_text(encoding="utf-8"))
    for node in module.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "all_imagenet_wordnet_ids"
            for target in node.targets
        ):
            values = ast.literal_eval(node.value)
            if len(values) != 1000:
                raise RuntimeError(f"expected 1,000 ImageNet WNIDs, found {len(values)}")
            return values
    raise RuntimeError("all_imagenet_wordnet_ids not found in official builder")


def load_protocol() -> tuple[list[str], list[str], list[str]]:
    metadata = _legacy('/data/zhoujianjie/environment_setup/native_sources/evaluation_audit/reference_code/perception_models/apps/pe/clip_benchmark/datasets')
    all_names = json.loads((metadata / "en_classnames.json").read_text())["imagenet1k"]
    templates = json.loads((metadata / "en_zeroshot_classification_templates.json").read_text())["imagenet1k"]
    info = json.loads((_legacy('/data/zhoujianjie/model_inference_for_GPU_adaptation/dataset/raw/imagenet_a_1/dataset_infos.json')).read_text())
    dataset_info = next(iter(info.values()))
    test_wnids = dataset_info["features"]["label"]["names"]
    name_by_wnid = dict(zip(imagenet_wnids(), all_names))
    classnames = [name_by_wnid[wnid] for wnid in test_wnids]
    return classnames, templates, test_wnids


def evaluate_one(model_id: str, workers: int) -> dict:
    import torch
    from PIL import Image
    from torch.utils.data import DataLoader, Dataset
    import core.vision_encoder.pe as pe
    import core.vision_encoder.transforms as transforms

    class ParquetImages(Dataset):
        def __init__(self):
            self.paths = sorted(EXTRACTED_ROOT.glob("*/*.jpg"))
            self.label_by_wnid = {wnid: index for index, wnid in enumerate(test_wnids)}
            if len(self.paths) != 7500:
                raise RuntimeError(f"expected 7,500 materialized images, found {len(self.paths)}")

            if __import__("os").environ.get("NATIVE_SMALL")=="1": self.paths=self.paths[:8]

        def __len__(self):
            return len(self.paths)

        def __getitem__(self, index):
            path = self.paths[index]
            with Image.open(path) as image:
                sample = preprocess(image.convert("RGB"))
            return sample, self.label_by_wnid[path.parent.name]

    spec = MODELS[model_id]
    classnames, templates, test_wnids = load_protocol()
    small=__import__('os').environ.get('NATIVE_SMALL')=='1'
    output=_result(model_id)/(("small_validation_" if small else "native_")+datetime.now().strftime('%Y%m%dT%H%M%S'))/'metrics.json'
    forward_seconds=0.0
    output.parent.mkdir(parents=True, exist_ok=True)
    result = {
        "status": "failed", "purpose": "accuracy reproduction; throughput provisional",
        "model_id": model_id, "dataset": "ImageNet-A official test", "sample_count": 7500,
        "dataset_revision": "1d04812197b88e02740e919e975bf113d6af0831",
        "official_top1_percent": spec["official_top1"],
        "reference_code_revision": REFERENCE_REVISION,
        "class_count": len(classnames), "class_wnids": test_wnids,
        "prompt_templates_per_class": len(templates), "batch_size": spec["batch_size"],
        "precision": "official FP32 weights with CUDA autocast", "started_at": now(),
    }
    model = loader = classifier = dataset = None
    prediction_file=None
    try:
        model = pe.CLIP.from_config(
            spec["config"], pretrained=True, checkpoint_path=str(spec["checkpoint"])
        ).eval().cuda()
        preprocess = transforms.get_image_transform(model.image_size)
        tokenizer = transforms.get_text_tokenizer(model.context_length)
        weights = []
        with torch.no_grad(), torch.autocast("cuda"):
            for classname in classnames:
                tokens = tokenizer([template.format(c=classname) for template in templates]).cuda()
                embeddings = torch.nn.functional.normalize(model.encode_text(tokens), dim=-1)
                embedding = embeddings.mean(dim=0)
                weights.append(embedding / embedding.norm())
        classifier = torch.stack(weights, dim=1).cuda()
        dataset = ParquetImages()
        loader = DataLoader(
            dataset, batch_size=spec["batch_size"], shuffle=False, num_workers=workers,
            pin_memory=True, persistent_workers=workers > 0,
        )
        correct1 = correct5 = seen = 0
        prediction_file=(output.parent/"predictions.jsonl").open("w")
        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        with torch.inference_mode():
            for images, targets in loader:
                images, targets = images.cuda(non_blocking=True), targets.cuda(non_blocking=True)
                with torch.autocast("cuda"):
                    torch.cuda.synchronize(); t=time.perf_counter()
                    image_features=model.encode_image(images, normalize=True)
                    torch.cuda.synchronize(); forward_seconds+=time.perf_counter()-t
                    predictions = (image_features @ classifier).topk(5, dim=1).indices
                correct1 += int((predictions[:, 0] == targets).sum())
                correct5 += int((predictions == targets[:, None]).any(dim=1).sum())
                for i,(target,rank) in enumerate(zip(targets.cpu().tolist(),predictions.cpu().tolist())):
                    prediction_file.write(json.dumps({'sample_id':str(dataset.paths[seen+i].relative_to(EXTRACTED_ROOT)),'target':target,'ranked_predictions':rank})+'\n')
                seen += len(targets)
                print('PROGRESS',seen,len(dataset),flush=True)
        elapsed = time.perf_counter() - started
        measured = 100.0 * correct1 / seen
        official = spec["official_top1"]
        result.update(
            status="passed", sample_count=seen, top1_percent=round(measured, 4), num_samples=seen, small_validation=small,
            forward_total_seconds=forward_seconds, forward_ms_per_sample=forward_seconds*1000/seen,
            purpose="A100_P800_same_4090_protocol",
            top5_percent=round(100.0 * correct5 / seen, 4),
            delta_percentage_points=round(measured - official, 4),
            relative_gap_percent=round(abs(measured - official) / official * 100.0, 4),
            within_5_percentage_points=abs(measured - official) <= 5.0,
            within_5_percent_relative=abs(measured - official) / official <= 0.05,
            provisional_images_per_second=round(seen / elapsed, 3),
            peak_gpu_memory_mb=round(torch.cuda.max_memory_allocated() / 1024**2, 2),
            environment={"python": platform.python_version(), "torch": torch.__version__,
                         "cuda_runtime": torch.version.cuda, "gpu": torch.cuda.get_device_name(0)},
        )
    except Exception as error:
        import traceback
        result.update(error_type=type(error).__name__, error=str(error), traceback=traceback.format_exc(limit=20))
    finally:
        if prediction_file: prediction_file.close()
        result["finished_at"] = now()
        output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
        del model, loader, classifier, dataset
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    print(json.dumps({"model_id": model_id, "status": result["status"], "top1": result.get("top1_percent")}), flush=True)
    return result


def write_summary(results: list[dict]) -> None:
    rows = [row for row in results if row.get("status") == "passed"]
    lines = [
        "# PE-Core ImageNet-A 官方协议对照\n",
        "ImageNet-A official test（7,500张）；类别映射和每类80个prompt来自PE官方CLIP Benchmark。\n",
        "| 模型 | 官方 Top-1 | 实测 Top-1 | Top-5 | 差值(pp) | 相对差距 | ≤5pp | 相对≤5% |",
        "|---|---:|---:|---:|---:|---:|:---:|:---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['model_id']} | {row['official_top1_percent']:.1f}% | {row['top1_percent']:.2f}% | "
            f"{row['top5_percent']:.2f}% | {row['delta_percentage_points']:+.2f} | "
            f"{row['relative_gap_percent']:.4f}% | {'是' if row['within_5_percentage_points'] else '否'} | "
            f"{'是' if row['within_5_percent_relative'] else '否'} |"
        )
    lines.append("\n并行阶段吞吐只作诊断，不用于效率验收。\n")
    SUMMARY.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", action="append", choices=MODELS)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    sys.path.insert(0, str(REFERENCE))
    sys.path.insert(0, str(_legacy('/data/zhoujianjie/environment_setup/native_sources/evaluation_audit/reference_code/perception_models/apps/pe')))
    selected = args.model_id or list(MODELS)
    results = []
    for index, model_id in enumerate(selected, 1):
        print(f"[{index}/{len(selected)}] RUN {model_id}", flush=True)
        results.append(evaluate_one(model_id, args.workers))
        write_summary(results)
    return 1 if any(row.get("status") != "passed" for row in results) else 0


if __name__ == "__main__":
    raise SystemExit(main())
