#!/usr/bin/env python3
"""Eight frozen classification protocols; no implicit weight/data downloads at inference."""
import argparse
import collections
import contextlib
import csv
import fcntl
import hashlib
import importlib.metadata
import importlib.util
import io
import json
import math
import os
from pathlib import Path
import re
import shutil
import sys
import tempfile
import time
import traceback
import urllib.request
import zipfile

HERE = Path(__file__).resolve().parent
MANIFEST = json.loads((HERE / "models.json").read_text())
RUNTIME = (HERE / ".runtime").resolve()


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=path.parent, prefix=".writing-")
    try:
        with os.fdopen(fd, "w") as handle:
            json.dump(value, handle, indent=2, ensure_ascii=False, allow_nan=False)
            handle.write("\n")
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


@contextlib.contextmanager
def locked(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        yield


def check_environment():
    versions = {name: importlib.metadata.version(name) for name in MANIFEST["versions"]}
    if versions != MANIFEST["versions"]:
        raise RuntimeError(f"Environment differs from frozen versions: {versions}; run setup_env.sh")
    return versions


def setup_sources():
    path = RUNTIME / "timm/validate.py"
    source = MANIFEST["timm_source"]
    if path.exists():
        if digest(path) != source["sha256"]:
            raise RuntimeError(f"Refusing changed official source: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(source["url"], timeout=60) as response:
        data = response.read()
    if hashlib.sha256(data).hexdigest() != source["sha256"]:
        raise RuntimeError("Downloaded timm source hash mismatch")
    path.write_bytes(data)


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def check_model(spec):
    root = Path(MANIFEST["model_root"]) / spec["model_id"]
    for name, expected in spec["files"].items():
        path = root / name
        if not path.is_file() or path.stat().st_size != expected["size"] or digest(path) != expected["sha256"]:
            raise RuntimeError(f"Missing/changed frozen model asset: {path}")
    return root


def normal_label(label):
    return " ".join(label.replace("_", " ").split())


def dataset_sources(kind):
    root = Path(MANIFEST["dataset_root"])
    return {"imagenet": [Path(MANIFEST["imagenet_zip"])],
            "plantvillage": [root / "plantvillage_eval/data.zip", root / "plantvillage_eval/splits/color_test.txt"],
            "synthraise": [root / "synthbuster_eval/synthbuster.zip", root / "raise_tiff_x300_kaggle/raise_tiff_x300_kaggle.zip"]}[kind]


def prepare_dataset(kind):
    """Extract only ImageNet val (~6GB), shared by native evaluators; other images stay in ZIPs."""
    sources = dataset_sources(kind)
    identity = [{"path": str(p), "size": p.stat().st_size, "mtime_ns": p.stat().st_mtime_ns} for p in sources]
    if kind == "plantvillage":
        identity[-1]["sha256"] = digest(sources[-1])
    cache = RUNTIME / "data" / kind
    marker = cache / "index.json"
    with locked(cache / ".prepare.lock"):
        if marker.exists():
            result = json.loads(marker.read_text())
            if result["sources"] != identity:
                raise RuntimeError(f"Dataset sources changed; preserve existing cache and choose a new package: {cache}")
            # A source snapshot symlinks to the shared runtime: never depend on its output directory.
            if kind == "imagenet" and result["folder"] != str(cache):
                previous = Path(result["folder"])
                for row in result["rows"]:
                    row["path"] = str(cache / Path(row["path"]).relative_to(previous))
                result["folder"] = str(cache)
                result.pop("fingerprint")
                result["fingerprint"] = canonical(result)
                save(marker, result)
            return result
        rows, classes = [], []
        if kind == "imagenet":
            from timm.data import ImageNetInfo
            info = ImageNetInfo("imagenet-1k")
            classes = info.label_names()
            assert classes == sorted(classes) and len(classes) == 1000
            indices = {name: index for index, name in enumerate(classes)}
            with zipfile.ZipFile(sources[0]) as archive:
                mapping_bytes = archive.read("LOC_val_solution.csv")
                mapping = list(csv.DictReader(io.StringIO(mapping_bytes.decode())))
                annotations = {}
                for entry in mapping:
                    labels = set(entry["PredictionString"].split()[::5])
                    if len(labels) != 1 or entry["ImageId"] in annotations:
                        raise RuntimeError("Ambiguous/duplicate ImageNet classification ground truth")
                    annotations[entry["ImageId"]] = indices[labels.pop()]
                names = sorted(n for n in archive.namelist() if "/Data/CLS-LOC/val/" in n and n.endswith(".JPEG"))
                assert {Path(n).stem for n in names} == set(annotations) and len(names) == 50000
                for index, member in enumerate(names, 1):
                    image_id = Path(member).stem
                    target = annotations[image_id]
                    image_path = cache / "val" / classes[target] / Path(member).name
                    entry = archive.getinfo(member)
                    if not image_path.exists() or image_path.stat().st_size != entry.file_size:
                        image_path.parent.mkdir(parents=True, exist_ok=True)
                        with archive.open(member) as source, image_path.open("wb") as destination:
                            shutil.copyfileobj(source, destination)
                    rows.append({"id": image_id, "path": str(image_path), "label": target, "crc32": entry.CRC})
                    if index % 5000 == 0:
                        print(f"[prepare ImageNet val] {index}/50000", flush=True)
            extra = {"label_descriptions": info.label_descriptions(), "annotation_sha256": hashlib.sha256(mapping_bytes).hexdigest(), "folder": str(cache)}
        elif kind == "plantvillage":
            names = sources[1].read_text().splitlines()
            assert len(names) == len(set(names)) == 10709
            classes = sorted({Path(n).parent.name for n in names})
            with zipfile.ZipFile(sources[0]) as archive:
                for member in names:
                    entry = archive.getinfo(member)
                    rows.append({"id": member, "archive": str(sources[0]), "member": member, "label": classes.index(Path(member).parent.name), "crc32": entry.CRC})
            extra = {}
        else:
            groups = {}
            for source in sources:
                with zipfile.ZipFile(source) as archive:
                    for entry in archive.infolist():
                        member = entry.filename
                        if not member.lower().endswith((".jpg", ".jpeg", ".png", ".tif", ".tiff")):
                            continue
                        group = Path(member).parent.name if source == sources[0] else "real"
                        groups.setdefault(group, []).append({"id": f"{group}/{member}", "archive": str(source), "member": member, "label": int(group == "real"), "group": group, "crc32": entry.CRC})
            assert len(groups) == 10 and len(groups["real"]) == 300
            for group, entries in groups.items():
                assert len(entries) == (300 if group == "real" else 1000), group
                groups[group] = sorted(entries, key=lambda x: hashlib.sha256(x["member"].encode()).hexdigest())[:300]
            # Round robin ensures even the first 10-image check includes every generator and real.
            rows = [groups[g][i] for i in range(300) for g in sorted(groups)]
            classes, extra = ["artificial", "real"], {}
        expected = MANIFEST["datasets"][kind]
        assert len(rows) == expected["images"] and len(classes) == expected["classes"]
        result = {"sources": identity, "classes": classes, "rows": rows, **extra}
        result["fingerprint"] = canonical(result)
        save(marker, result)
        return result


def select_rows(data, limit):
    return data["rows"] if limit == 0 else data["rows"][:limit]


def read_image(row, archives):
    from PIL import Image
    if "path" in row:
        with Image.open(row["path"]) as image:
            return image.convert("RGB")
    path = row["archive"]
    if path not in archives:
        archives[path] = zipfile.ZipFile(path)
    with archives[path].open(row["member"]) as handle, Image.open(handle) as image:
        return image.convert("RGB")


def label_mapping(spec, data, model_root):
    config = json.loads((model_root / "config.json").read_text())
    if spec["runner"] != "hf":
        return list(range(len(data["classes"])))
    labels = [config["id2label"][str(i)] for i in range(len(config["id2label"]))]
    if spec["dataset"] == "imagenet":
        # HF canonical lemmas must match the author-supplied ImageNet output order exactly.
        if labels != data["label_descriptions"]:
            raise RuntimeError("HF ImageNet labels differ from canonical index order; explicit mapping required")
        return list(range(1000))
    targets = [normal_label(x) for x in labels]
    source = [normal_label(x) for x in data["classes"]]
    assert len(set(targets)) == len(targets) and set(source) == set(targets)
    return [targets.index(x) for x in source]


def read_records(path):
    if not path.exists():
        return []
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def hf_inference(spec, data, model_root, output, limit, device, mapping):
    import torch
    from transformers import AutoImageProcessor, AutoModelForImageClassification
    processor = AutoImageProcessor.from_pretrained(model_root, local_files_only=True, use_fast=False)
    model, loading = AutoModelForImageClassification.from_pretrained(model_root, local_files_only=True, output_loading_info=True)
    if loading.get("missing_keys") or loading.get("mismatched_keys") or loading.get("error_msgs"):
        raise RuntimeError(f"Incomplete model load: {loading}")
    model.to(device).eval()
    selected = select_rows(data, limit)
    predictions = output / "predictions.jsonl"
    records = read_records(predictions)
    by_id = {r["id"]: r for r in records}
    if len(by_id) != len(records) or not set(by_id) <= {r["id"] for r in selected}:
        raise RuntimeError("Duplicate/out-of-split resume records")
    pending = [r for r in selected if r["id"] not in by_id]
    archives = {}
    try:
        with predictions.open("a") as handle:
            for start in range(0, len(pending), spec["batch_size"]):
                batch = pending[start:start + spec["batch_size"]]
                images = [read_image(row, archives) for row in batch]
                inputs = processor(images=images, return_tensors="pt").to(device)
                with torch.inference_mode():
                    logits = model(**inputs).logits.float()
                assert logits.shape == (len(batch), len(mapping)) and torch.isfinite(logits).all()
                top = logits.topk(min(5, logits.shape[1]), dim=1).indices.cpu().tolist()
                probabilities = logits.softmax(-1).cpu().tolist() if spec["dataset"] == "synthraise" else None
                for index, row in enumerate(batch):
                    record = {"id": row["id"], "target": mapping[row["label"]], "prediction": top[index][0], "top5": top[index]}
                    if probabilities:
                        record.update(group=row["group"], p_artificial=probabilities[index][0])
                    handle.write(json.dumps(record, allow_nan=False) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
                completed = len(records) + start + len(batch)
                if start == 0 or completed % 200 < spec["batch_size"] or completed == len(selected):
                    print(f"[{spec['sequence']}] {completed}/{len(selected)}", flush=True)
    finally:
        for archive in archives.values():
            archive.close()
    records = read_records(predictions)
    assert len(records) == len(selected) and {r["id"] for r in records} == {r["id"] for r in selected}
    return records


class Tee:
    def __init__(self, *streams):
        self.streams = streams
    def write(self, text):
        for stream in self.streams:
            stream.write(text)
        return len(text)
    def flush(self):
        for stream in self.streams:
            stream.flush()


def native_evaluate(spec, data, model_root, output, limit, device):
    import torch
    from torch.utils.data import Subset, DataLoader
    count = len(select_rows(data, limit))
    capture = io.StringIO()
    with (output / "official_evaluator.log").open("a") as log, contextlib.redirect_stdout(Tee(sys.stdout, log, capture)):
        if spec["runner"] == "timm":
            path = RUNTIME / "timm/validate.py"
            assert digest(path) == MANIFEST["timm_source"]["sha256"]
            official = load_module("classification8_official_timm", path)
            official.setup_default_logging()
            cfg = spec["config"]["pretrained_cfg"]
            size = cfg.get("test_input_size", cfg["input_size"])
            args = official.parser.parse_args([
                "--data-dir", data["folder"], "--split", "val", "--model", spec["model_id"].split("/")[1],
                "--checkpoint", str(model_root / spec["weight_file"]), "--device", device,
                "--batch-size", str(spec["batch_size"]), "--workers", "2", "--no-prefetcher",
                "--input-size", *map(str, size), "--crop-pct", str(cfg.get("test_crop_pct", cfg["crop_pct"])),
                "--interpolation", cfg["interpolation"], "--num-classes", "1000"])
            create = official.create_dataset
            def checked_dataset(*a, **kw):
                dataset = create(*a, **kw)
                assert dataset.reader.class_to_idx == {c: i for i, c in enumerate(data["classes"])}
                assert {(p, y) for p, y in dataset.reader.samples} == {(r["path"], r["label"]) for r in data["rows"]}
                if limit:
                    dataset.reader.samples = dataset.reader.samples[:count]
                return dataset
            official.create_dataset = checked_dataset
            metrics = official.validate(args)
            print(json.dumps(metrics), flush=True)
            result = {"top1_percent": float(metrics["top1"]), "top5_percent": float(metrics["top5"])}
        else:
            # Keep the publisher code untouched; call its own loader and evaluator.
            sys.path.insert(0, str(model_root))
            official = load_module("classification8_official_enot", model_root / "test.py")
            dataset, _ = official.load_data(str(Path(data["folder"]) / "val"))
            assert dataset.class_to_idx == {c: i for i, c in enumerate(data["classes"])}
            assert set(dataset.samples) == {(r["path"], r["label"]) for r in data["rows"]}
            if limit:
                dataset = Subset(dataset, list(range(count)))
            loader = DataLoader(dataset, batch_size=spec["batch_size"], shuffle=False, num_workers=0)
            checkpoint = torch.load(model_root / spec["weight_file"], map_location="cpu", weights_only=False)
            model = checkpoint["model_ckpt"]
            if "model_ema" in checkpoint:
                model.load_state_dict({k.replace("module.", ""): v for k, v in checkpoint["model_ema"].items() if "module." in k})
            model.to(device).eval()
            accuracy = official.evaluate(torch.nn.CrossEntropyLoss(), loader, torch.device(device), model=model)
            matches = re.findall(r"Acc@1 ([\d.]+) Acc@5 ([\d.]+)", capture.getvalue())
            assert matches, "Publisher evaluator did not report final Top-5"
            result = {"top1_percent": float(accuracy), "top5_percent": float(matches[-1][1])}
    assert all(math.isfinite(v) and 0 <= v <= 100 for v in result.values())
    return result


def classification_metrics(records, spec):
    from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, roc_auc_score
    y = [r["target"] for r in records]
    predicted = [r["prediction"] for r in records]
    result = {"top1_percent": 100 * accuracy_score(y, predicted),
              "macro_f1": f1_score(y, predicted, average="macro", zero_division=0)}
    if spec["dataset"] == "imagenet":
        result["top5_percent"] = 100 * accuracy_score([True] * len(y), [r["target"] in r["top5"] for r in records])
    else:
        result["confusion_matrix"] = confusion_matrix(y, predicted, labels=list(range(38 if spec["dataset"] == "plantvillage" else 2))).tolist()
    if spec["dataset"] == "synthraise":
        result["auroc_artificial_positive"] = roc_auc_score([int(v == 0) for v in y], [r["p_artificial"] for r in records])
        groups = sorted({r["group"] for r in records} - {"real"})
        per_group = {}
        for group in groups:
            rows = [r for r in records if r["group"] in (group, "real")]
            assert len(rows) == 600
            per_group[group] = {"images": len(rows), "accuracy_percent": 100 * accuracy_score([r["target"] for r in rows], [r["prediction"] for r in rows]),
                                "auroc": roc_auc_score([int(r["target"] == 0) for r in rows], [r["p_artificial"] for r in rows])}
        result["per_generator_balanced"] = per_group
        result["generator_macro_accuracy_percent"] = sum(v["accuracy_percent"] for v in per_group.values()) / 9
    return result


def run(spec, args):
    output = args.output_root / spec["model_key"]
    with locked(output / ".lock"):
        status_path = output / "status.json"
        old = json.loads(status_path.read_text()) if status_path.exists() else None
        status = {"state": "preflight_ready", "sequence": spec["sequence"], "model_id": spec["model_id"], "manifest_snapshot": spec}
        started = time.time()
        try:
            versions = check_environment()
            model_root = check_model(spec)
            data = prepare_dataset(spec["dataset"])
            mapping = label_mapping(spec, data, model_root)
            fingerprint = canonical({"spec": spec, "versions": versions, "dataset": data["fingerprint"], "mapping": mapping,
                                     "runner_sha256": digest(__file__), "timm_source": MANIFEST["timm_source"], "device": args.device})
            if old and old.get("run_fingerprint") and old["run_fingerprint"] != fingerprint:
                raise RuntimeError("Frozen protocol changed; use a NEW OUTPUT_ROOT. Old results were not overwritten.")
            if old and (old["state"] == "complete" or old["state"] == "partial_complete" and old.get("requested_images") == args.max_images):
                if old["state"] == "complete":
                    saved_metrics = json.loads((output / "metrics.json").read_text())
                    assert saved_metrics["run_fingerprint"] == fingerprint and saved_metrics["evaluation_images"] == len(data["rows"])
                if spec["runner"] == "hf":
                    saved_records = read_records(output / "predictions.jsonl")
                    expected_rows = select_rows(data, old["requested_images"])
                    assert len(saved_records) == len(expected_rows) and {r["id"] for r in saved_records} == {r["id"] for r in expected_rows}
                print(f"[{spec['sequence']}] 已完成当前范围，跳过", flush=True)
                return True
            status.update(run_fingerprint=fingerprint, runtime_versions=versions, dataset_fingerprint=data["fingerprint"], label_mapping=mapping,
                          total_images=len(data["rows"]), requested_images=args.max_images, device=args.device, started_at=time.time())
            if args.preflight:
                print(json.dumps({"sequence": spec["sequence"], "state": "preflight_ready", "images": len(data["rows"])}))
                return True
            import torch
            if args.device.startswith("cuda") and not torch.cuda.is_available():
                raise RuntimeError("CUDA unavailable; refusing silent CPU fallback")
            torch.set_num_threads(2)
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = spec["runner"] != "timm"
            if args.device.startswith("cuda"):
                torch.cuda.reset_peak_memory_stats()
                status["gpu_name"] = torch.cuda.get_device_name()
            status["state"] = "running"
            save(status_path, status)
            if spec["runner"] == "hf":
                records = hf_inference(spec, data, model_root, output, args.max_images, args.device, mapping)
                metrics = classification_metrics(records, spec) if args.max_images == 0 else None
            else:
                native_metrics = native_evaluate(spec, data, model_root, output, args.max_images, args.device)
                metrics = native_metrics if args.max_images == 0 else None
            if metrics is not None:
                save(output / "metrics.json", {"metrics": metrics, "evaluation_images": len(data["rows"]), "dataset_relation": spec["dataset_relation"],
                                               "fidelity": spec["fidelity"], "official_reference": spec["official_reference"],
                                               "reference_check": spec["official_reference"]["comparison"], "run_fingerprint": fingerprint})
            status.update(state="complete" if args.max_images == 0 else "partial_complete", completed_images=len(select_rows(data, args.max_images)),
                          elapsed_seconds=time.time() - started, finished_at=time.time(),
                          backend_flags={"matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
                                         "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
                                         "cudnn_benchmark": torch.backends.cudnn.benchmark,
                                         "cudnn_deterministic": torch.backends.cudnn.deterministic})
            if args.device.startswith("cuda"):
                status["peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
            save(status_path, status)
            print(f"[{spec['sequence']}] {status['state']} -> {output}", flush=True)
            return True
        except Exception as error:
            # Do not replace evidence when a user tries to mix a changed protocol into old results.
            if old and old.get("run_fingerprint") and status.get("run_fingerprint") != old["run_fingerprint"]:
                traceback.print_exc()
                return False
            status.update(state="failed", error=str(error), traceback=traceback.format_exc(), elapsed_seconds=time.time() - started)
            save(status_path, status)
            traceback.print_exc()
            return False


def summarize(root):
    rows = []
    for spec in MANIFEST["models"]:
        path = root / spec["model_key"] / "status.json"
        state = json.loads(path.read_text()) if path.exists() else {"state": "not_started"}
        metric_path = path.parent / "metrics.json"
        metric_data = json.loads(metric_path.read_text()) if metric_path.exists() else {}
        rows.append({"sequence": spec["sequence"], "model_id": spec["model_id"], "state": state["state"],
                     "dataset": spec["dataset"], "dataset_relation": spec["dataset_relation"], "fidelity": spec["fidelity"],
                     "case_type": spec["case_type"], "special_tags": spec["special_tags"],
                     "completed_images": state.get("completed_images", 0), "metrics": metric_data.get("metrics"),
                     "official_reference": spec["official_reference"], "reference_check": metric_data.get("reference_check", "not_evaluated"),
                     "elapsed_seconds": state.get("elapsed_seconds"), "peak_allocated_bytes": state.get("peak_allocated_bytes"),
                     "error": state.get("error")})
    result = {"scope": MANIFEST["scope"], "models": rows, "counts": dict(collections.Counter(r["state"] for r in rows)),
              "dataset_relation_counts": dict(collections.Counter(r["dataset_relation"] for r in rows)),
              "fidelity_counts": dict(collections.Counter(r["fidelity"] for r in rows)),
              "special_tag_counts": dict(collections.Counter(t for r in rows for t in r["special_tags"])),
              "note": "仅统计本批 8 个唯一编号，不把冒烟或本批统计冒充 100 模型全项目完成数。"}
    save(root / "audit_summary.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", help="sequence or exact model_id")
    parser.add_argument("--describe", action="store_true")
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--setup-sources", action="store_true")
    parser.add_argument("--check-environment", action="store_true")
    parser.add_argument("--summarize", action="store_true")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-images", type=int, default=10)
    parser.add_argument("--output-root", type=Path, default=Path(MANIFEST["result_root"]))
    args = parser.parse_args()
    if args.max_images < 0:
        parser.error("max-images must be >=0")
    if args.setup_sources:
        setup_sources()
        return 0
    if args.check_environment:
        print(check_environment())
        return 0
    if args.summarize:
        summarize(args.output_root)
        return 0
    selected = [s for s in MANIFEST["models"] if args.model in (str(s["sequence"]), s["model_id"])]
    if len(selected) != 1:
        parser.error("Choose --model from 71,74,84,87,88,89,91,95")
    if args.describe:
        print(json.dumps(selected[0], ensure_ascii=False, indent=2))
        return 0
    return 0 if run(selected[0], args) else 1


if __name__ == "__main__":
    raise SystemExit(main())
