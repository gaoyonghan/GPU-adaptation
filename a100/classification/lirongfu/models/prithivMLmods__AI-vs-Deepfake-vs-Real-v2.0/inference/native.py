#!/usr/bin/env python3
"""Offline model construction, preprocessing, inference, and dataset evaluation.

Use ``python inference.py --image IMAGE`` for one image, or omit --image to
infer the packaged evaluation dataset and then calculate metrics. All model
assets and bundled architecture sources are unpacked into an external temporary
directory. Installed framework dependencies are still required.
"""
import sys
sys.dont_write_bytecode = True

import argparse
import hashlib
import importlib
import importlib.util
import json
import os
from pathlib import Path, PurePosixPath
import tempfile
import uuid

MODEL_DIR = Path(__file__).resolve().parent.parent
PROJECT_ROOT = MODEL_DIR.parents[3]
_path_spec = importlib.util.spec_from_file_location('_native_paths', Path(__file__).with_name('paths.py'))
PATHS = importlib.util.module_from_spec(_path_spec); _path_spec.loader.exec_module(PATHS)

MODEL_SPEC = {'model_key': 'prithivMLmods__AI-vs-Deepfake-vs-Real-v2.0'}
EMBEDDED_SOURCES = {'_shared_test_resources/code/dima_kaggle_evaluator.py': '#!/usr/bin/env python3\n"""Reproduce the evaluation protocol published in dima806 Kaggle notebooks.\n\nThis script performs inference only.  It deliberately mirrors the notebook\'s\nsorted file enumeration, RandomOverSampler(random_state=83), stratified HF\nDataset split (seed 42), square resize, normalization, and sklearn metrics.\n"""\n\nfrom __future__ import annotations\n\nimport argparse\nimport json\nimport time\nfrom pathlib import Path\n\nimport numpy as np\nimport pandas as pd\nimport torch\nfrom datasets import ClassLabel, Dataset, Image\nfrom imblearn.over_sampling import RandomOverSampler\nfrom PIL import Image as PILImage\nfrom sklearn.metrics import (\n    accuracy_score,\n    classification_report,\n    confusion_matrix,\n    f1_score,\n    precision_score,\n    recall_score,\n)\nfrom torch.utils.data import DataLoader, Dataset as TorchDataset\nfrom torchvision.transforms import Compose, Normalize, Resize, ToTensor\nfrom transformers import ViTForImageClassification, ViTImageProcessor\n\n\ndef parse_args() -> argparse.Namespace:\n    p = argparse.ArgumentParser()\n    p.add_argument("--model-dir", type=Path, required=True)\n    p.add_argument("--dataset-root", type=Path, required=True)\n    p.add_argument("--glob", action="append", required=True, help="Notebook glob relative to dataset root; repeatable")\n    p.add_argument("--label-mode", choices=("parent", "filename_prefix", "yoga"), required=True)\n    p.add_argument("--test-size", type=float, required=True)\n    p.add_argument("--expected-support", type=int, required=True)\n    p.add_argument("--official-accuracy", type=float, required=True)\n    p.add_argument("--official-macro-f1", type=float, required=True)\n    p.add_argument("--official-weighted-f1", type=float, required=True)\n    p.add_argument("--official-macro-precision", type=float)\n    p.add_argument("--official-macro-recall", type=float)\n    p.add_argument("--official-weighted-precision", type=float)\n    p.add_argument("--official-weighted-recall", type=float)\n    p.add_argument("--batch-size", type=int, default=64)\n    p.add_argument("--max-samples", type=int)\n    p.add_argument("--result-dir", type=Path)\n    return p.parse_args()\n\n\ndef get_label(path: Path, mode: str) -> str:\n    if mode == "parent":\n        return path.parent.name\n    if mode == "filename_prefix":\n        return path.name.split("_")[0]\n    # Keep the notebook\'s exact Python rstrip semantics (not suffix removal).\n    return path.parent.name.rstrip("Pose").rstrip("-")\n\n\nclass EvalImages(TorchDataset):\n    def __init__(self, paths: list[str], labels: list[int], transform):\n        self.paths, self.labels, self.transform = paths, labels, transform\n\n    def __len__(self):\n        return len(self.paths)\n\n    def __getitem__(self, idx):\n        with PILImage.open(self.paths[idx]) as im:\n            pixels = self.transform(im.convert("RGB"))\n        return pixels, self.labels[idx], self.paths[idx]\n\n\ndef main() -> None:\n    a = parse_args()\n    started = time.time()\n    result_dir = a.result_dir or a.model_dir / "result"\n    result_dir.mkdir(parents=True, exist_ok=True)\n\n    paths = []\n    for pattern in a.glob:\n        paths.extend(sorted(a.dataset_root.glob(pattern)))\n    paths = [p for p in paths if p.is_file()]\n    if not paths:\n        raise FileNotFoundError(f"No files matched {a.glob} below {a.dataset_root}")\n    labels = [get_label(p, a.label_mode) for p in paths]\n\n    config = json.loads((a.model_dir / "config.json").read_text())\n    id2label = {int(k): v for k, v in config["id2label"].items()}\n    label_names = [id2label[i] for i in range(len(id2label))]\n    unknown = sorted(set(labels) - set(label_names))\n    if unknown:\n        raise ValueError(f"Dataset labels absent from official model config: {unknown}")\n\n    frame = pd.DataFrame({"image": [str(p) for p in paths], "label": labels})\n    ros = RandomOverSampler(random_state=83)\n    x_resampled, y_resampled = ros.fit_resample(frame[["image"]], frame[["label"]])\n    frame = x_resampled.copy()\n    frame["label"] = np.asarray(y_resampled).reshape(-1)\n\n    # Use HF Dataset for the same stratified split implementation as the notebook.\n    ds = Dataset.from_pandas(frame, preserve_index=False)\n    class_labels = ClassLabel(num_classes=len(label_names), names=label_names)\n    ds = ds.map(lambda batch: {"label": class_labels.str2int(batch["label"])}, batched=True)\n    ds = ds.cast_column("label", class_labels)\n    split = ds.train_test_split(\n        test_size=a.test_size,\n        shuffle=True,\n        stratify_by_column="label",\n        seed=42,\n    )["test"]\n    if a.max_samples:\n        split = split.select(range(min(a.max_samples, len(split))))\n    elif len(split) != a.expected_support:\n        raise RuntimeError(\n            f"Official support mismatch: rebuilt {len(split)}, expected {a.expected_support}. "\n            "The source dataset version may have changed."\n        )\n\n    processor = ViTImageProcessor.from_pretrained(a.model_dir, local_files_only=True)\n    size = processor.size["height"]\n    transform = Compose([\n        Resize((size, size)),\n        ToTensor(),\n        Normalize(mean=processor.image_mean, std=processor.image_std),\n    ])\n    data = EvalImages(split["image"], split["label"], transform)\n    loader = DataLoader(data, batch_size=a.batch_size, shuffle=False, num_workers=4, pin_memory=True)\n\n    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")\n    model = ViTForImageClassification.from_pretrained(a.model_dir, local_files_only=True)\n    model.eval().to(device)\n    y_true, y_pred, used_paths = [], [], []\n    with torch.inference_mode():\n        for pixels, target, batch_paths in loader:\n            logits = model(pixel_values=pixels.to(device, non_blocking=True)).logits\n            y_pred.extend(logits.argmax(1).cpu().tolist())\n            y_true.extend(target.tolist())\n            used_paths.extend(batch_paths)\n\n    accuracy = accuracy_score(y_true, y_pred)\n    macro_precision = precision_score(y_true, y_pred, average="macro", zero_division=0)\n    macro_recall = recall_score(y_true, y_pred, average="macro", zero_division=0)\n    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)\n    weighted_f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)\n    weighted_precision = precision_score(y_true, y_pred, average="weighted", zero_division=0)\n    weighted_recall = recall_score(y_true, y_pred, average="weighted", zero_division=0)\n    report = classification_report(\n        y_true, y_pred, labels=list(range(len(label_names))), target_names=label_names,\n        digits=4, zero_division=0,\n    )\n    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(label_names))))\n    summary = {\n        "model_dir": str(a.model_dir),\n        "dataset_root": str(a.dataset_root),\n        "source_images": len(paths),\n        "oversampled_images": len(frame),\n        "evaluated_predictions": len(y_true),\n        "accuracy": accuracy,\n        "macro_precision": macro_precision,\n        "macro_recall": macro_recall,\n        "macro_f1": macro_f1,\n        "weighted_f1": weighted_f1,\n        "weighted_precision": weighted_precision,\n        "weighted_recall": weighted_recall,\n        "official_accuracy": a.official_accuracy,\n        "official_macro_f1": a.official_macro_f1,\n        "official_weighted_f1": a.official_weighted_f1,\n        "delta_accuracy": accuracy - a.official_accuracy,\n        "delta_macro_f1": macro_f1 - a.official_macro_f1,\n        "delta_weighted_f1": weighted_f1 - a.official_weighted_f1,\n        "device": str(device),\n        "elapsed_seconds": round(time.time() - started, 3),\n        "protocol": "Official Kaggle notebook: sorted glob; ROS seed=83; stratified HF split seed=42; inference-only.",\n    }\n    for metric in ("macro_precision", "macro_recall", "weighted_precision", "weighted_recall"):\n        official = getattr(a, f"official_{metric}")\n        if official is not None:\n            summary[f"official_{metric}"] = official\n            summary[f"delta_{metric}"] = summary[metric] - official\n    (result_dir / "summary.txt").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\\n")\n    (result_dir / "classification_report.txt").write_text(report + "\\n")\n    np.savetxt(result_dir / "confusion_matrix.txt", cm, fmt="%d")\n    with (result_dir / "predictions.txt").open("w") as f:\n        f.write("path\\ttrue_id\\ttrue_label\\tpred_id\\tpred_label\\n")\n        for path, true, pred in zip(used_paths, y_true, y_pred):\n            f.write(f"{path}\\t{true}\\t{label_names[true]}\\t{pred}\\t{label_names[pred]}\\n")\n    pd.DataFrame([summary]).to_excel(result_dir / "metrics.xlsx", index=False)\n    print(json.dumps(summary, ensure_ascii=False, indent=2))\n\n\nif __name__ == "__main__":\n    main()\n', '_shared_test_resources/code/download_remaining_classification_data.py': '#!/usr/bin/env python3\n"""Download one selected public benchmark for every non-strict classifier.\n\nDatasets shared by several models are downloaded once.  Each model receives a\nrelative ``test_data/benchmark`` symlink, which keeps the suite portable when\nthe whole image-classification directory is moved to another machine.\n"""\nfrom __future__ import annotations\n\nimport argparse\nimport io\nimport json\nimport os\nimport shutil\nimport time\nimport zipfile\nfrom pathlib import Path\nfrom urllib.parse import quote\n\nimport requests\nfrom datasets import Dataset, Image as DatasetImage, IterableDataset, get_dataset_split_names, load_dataset\nfrom huggingface_hub import HfApi, hf_hub_download, snapshot_download\nfrom tqdm.auto import tqdm\n\n\nROOT = Path(__file__).resolve().parents[2]\nDEFAULT_PLAN = ROOT / "remaining_classification_plan.json"\n\n\ndef arguments() -> argparse.Namespace:\n    parser = argparse.ArgumentParser()\n    parser.add_argument("--plan", type=Path, default=DEFAULT_PLAN)\n    parser.add_argument("--retries", type=int, default=5)\n    parser.add_argument("--only", action="append", default=[], help="Only download this model key")\n    parser.add_argument("--exclude", action="append", default=[], help="Skip this model key")\n    parser.add_argument("--list-only", action="store_true", help="Print the plan without network access")\n    return parser.parse_args()\n\n\ndef load_plan(path: Path, only: set[str], exclude: set[str]) -> list[dict]:\n    entries = json.loads(path.read_text(encoding="utf-8"))["models"]\n    entries = [item for item in entries if "data_key" in item]\n    entries = [item for item in entries if item["model"] not in exclude]\n    if only:\n        unknown = only - {item["model"] for item in entries}\n        if unknown:\n            raise ValueError(f"Unknown --only model(s): {sorted(unknown)}")\n        entries = [item for item in entries if item["model"] in only]\n    return entries\n\n\ndef retry(label: str, attempts: int, operation):\n    for attempt in range(1, attempts + 1):\n        try:\n            return operation()\n        except Exception as exc:\n            if attempt == attempts:\n                raise\n            delay = min(60, 5 * 2 ** (attempt - 1))\n            detail = str(exc)\n            if len(detail) > 800:\n                detail = detail[:800] + " ... [truncated]"\n            print(f"{label}: attempt {attempt}/{attempts} failed: {detail}", flush=True)\n            print(f"Retrying in {delay}s ...", flush=True)\n            time.sleep(delay)\n\n\ndef dataset_revision(repo_id: str) -> str:\n    token = os.environ.get("HF_TOKEN") or None\n    return HfApi(token=token).dataset_info(repo_id).sha\n\n\ndef choose_split(repo_id: str, config: str | None, requested: str, revision: str) -> str:\n    token = os.environ.get("HF_TOKEN") or None\n    splits = get_dataset_split_names(\n        repo_id,\n        config_name=config,\n        revision=revision,\n        token=token,\n    )\n    for candidate in (requested, "test", "validation", "valid", "train"):\n        if candidate in splits:\n            return candidate\n    raise RuntimeError(f"{repo_id} has no usable split; available={splits}")\n\n\ndef parquet_export(\n    repo_id: str,\n    config: str | None,\n    requested_split: str,\n    streaming: bool,\n    max_samples: int = 0,\n    stream_capped: bool = False,\n):\n    """Load the Hub dataset-server Parquet export to avoid thousands of file API calls."""\n    token = os.environ.get("HF_TOKEN") or None\n    headers = {"Authorization": f"Bearer {token}"} if token else {}\n    response = requests.get(\n        f"https://datasets-server.huggingface.co/parquet?dataset={quote(repo_id, safe=\'\')}",\n        headers=headers,\n        timeout=120,\n    )\n    response.raise_for_status()\n    files = response.json().get("parquet_files", [])\n    if not files:\n        raise RuntimeError(f"No dataset-server Parquet export for {repo_id}")\n    configs = sorted({item["config"] for item in files})\n    selected_config = config if config in configs else ("default" if "default" in configs else configs[0])\n    selected = [item for item in files if item["config"] == selected_config]\n    splits = sorted({item["split"] for item in selected})\n    split = next(\n        (name for name in (requested_split, "test", "validation", "valid", "train") if name in splits),\n        None,\n    )\n    if split is None:\n        raise RuntimeError(f"No usable Parquet split for {repo_id}; available={splits}")\n    urls = [item["url"] for item in selected if item["split"] == split]\n    # For capped datasets, downloading one complete Parquet shard is much\n    # faster and more reliable than thousands of remote range requests.\n    selected_urls = urls[:1] if max_samples else urls\n    use_streaming = bool(streaming and stream_capped)\n    load_split = split if use_streaming else (f"{split}[:{max_samples}]" if max_samples else split)\n    dataset = load_dataset(\n        "parquet",\n        data_files={split: selected_urls},\n        split=load_split,\n        streaming=use_streaming if max_samples else streaming,\n        token=token,\n    )\n    return dataset, selected_config, split\n\n\ndef materialize_remote_images(dataset, data_key: str):\n    """Embed Image features whose Parquet rows only contain remote references."""\n    for column, feature in dataset.features.items():\n        if not isinstance(feature, DatasetImage):\n            continue\n        raw = dataset.cast_column(column, DatasetImage(decode=False))\n        referenced = any(\n            (value := raw[index][column])\n            and value.get("bytes") is None\n            and value.get("path")\n            for index in range(min(len(raw), 32))\n        )\n        if not referenced:\n            continue\n\n        def embed(row, index):\n            image = row[column]\n            if not hasattr(image, "save"):\n                raise RuntimeError(f"Cannot materialize {data_key}.{column} row {index}")\n            buffer = io.BytesIO()\n            image.convert("RGB").save(buffer, format="PNG")\n            return {column: {"bytes": buffer.getvalue(), "path": f"{index:08d}.png"}}\n\n        print(f"Materializing remote image bytes: {data_key}.{column}", flush=True)\n        workers = min(8, os.cpu_count() or 1)\n        dataset = dataset.map(\n            embed,\n            with_indices=True,\n            num_proc=workers,\n            desc=f"{data_key} images",\n        )\n    return dataset\n\n\ndef save_hf_dataset(entry: dict, target: Path, retries: int) -> dict:\n    requested_repo = entry["dataset_id"]\n    if entry.get("local_existing_glob"):\n        import glob\n        files = sorted(glob.glob(str(ROOT / entry["local_existing_glob"])))\n        if files:\n            dataset = load_dataset("parquet", data_files=files, split="train")\n            dataset.save_to_disk(str(target))\n            return {\n                "requested_dataset_id": requested_repo,\n                "resolved_dataset_id": requested_repo,\n                "revision": "existing-local-official-release",\n                "split": entry.get("split", "train"),\n                "config": None,\n                "rows": len(dataset),\n                "source_level": entry["source_level"],\n                "fallback_used": False,\n                "parquet_export": False,\n                "reused_local_files": len(files),\n            }\n    if entry.get("snapshot_subdir"):\n        token = os.environ.get("HF_TOKEN") or None\n        revision = retry(\n            f"resolve {requested_repo}", retries,\n            lambda: dataset_revision(requested_repo),\n        )\n        subdir = entry["snapshot_subdir"].strip("/")\n        snapshot = retry(\n            f"download {requested_repo}/{subdir}", retries,\n            lambda: snapshot_download(\n                repo_id=requested_repo,\n                repo_type="dataset",\n                revision=revision,\n                allow_patterns=[f"{subdir}/**"],\n                token=token,\n                max_workers=4,\n            ),\n        )\n        dataset = load_dataset(\n            "imagefolder", data_dir=str(Path(snapshot) / subdir), split="train"\n        )\n        dataset.save_to_disk(str(target))\n        return {\n            "requested_dataset_id": requested_repo,\n            "resolved_dataset_id": requested_repo,\n            "revision": revision,\n            "split": entry.get("split", "test"),\n            "config": "imagefolder",\n            "rows": len(dataset),\n            "source_level": entry["source_level"],\n            "fallback_used": False,\n            "parquet_export": False,\n            "snapshot_subdir": subdir,\n        }\n    candidates = [requested_repo]\n    if entry.get("fallback_dataset_id"):\n        candidates.append(entry["fallback_dataset_id"])\n    last_error: Exception | None = None\n    for candidate_index, repo_id in enumerate(candidates):\n        try:\n            revision = retry(f"resolve {repo_id}", retries, lambda: dataset_revision(repo_id))\n            config = entry.get("dataset_config")\n            token = os.environ.get("HF_TOKEN") or None\n            requested_split = entry.get("split", "test")\n            try:\n                dataset, config, split = retry(\n                    f"Parquet export {repo_id}",\n                    retries,\n                    lambda: parquet_export(\n                        repo_id,\n                        config,\n                        requested_split,\n                        bool(entry.get("streaming")),\n                        int(entry.get("max_download_samples", 0)),\n                        bool(entry.get("stream_capped_parquet")),\n                    ),\n                )\n                used_parquet_export = True\n            except Exception as export_error:\n                print(f"Parquet export unavailable for {repo_id}: {export_error}; using native loader")\n                split = retry(\n                    f"inspect {repo_id}", retries,\n                    lambda: choose_split(repo_id, config, requested_split, revision),\n                )\n                kwargs = dict(\n                    path=repo_id, name=config, split=split, revision=revision,\n                    token=token,\n                )\n                dataset = retry(\n                    f"download {repo_id}", retries,\n                    lambda: load_dataset(streaming=bool(entry.get("streaming")), **kwargs),\n                )\n                used_parquet_export = False\n            if entry.get("streaming"):\n                limit = int(entry["max_download_samples"])\n                if isinstance(dataset, IterableDataset):\n                    rows = []\n                    for row in tqdm(dataset.take(limit), total=limit, desc=entry["data_key"], unit="sample"):\n                        rows.append(row)\n                    if not rows:\n                        raise RuntimeError(f"Streaming dataset {repo_id} returned no rows")\n                    dataset = Dataset.from_list(rows)\n                elif len(dataset) > limit:\n                    dataset = dataset.select(range(limit))\n            # A text-only dataset can be accessible while still being unusable\n            # for an image classifier. Force the declared fallback in that case.\n            columns = set(dataset.column_names)\n            image_columns = {\n                "image", "front", "img", "photo", "pixel_values", "image_path", "path", "file"\n            }\n            if not columns.intersection(image_columns):\n                raise RuntimeError(\n                    f"{repo_id} split {split} has no image column; available={sorted(columns)}"\n                )\n            dataset = materialize_remote_images(dataset, entry["data_key"])\n            dataset.save_to_disk(str(target))\n            source_level = entry.get("source_level", "substitute")\n            if candidate_index:\n                source_level = entry.get("fallback_source_level", "official-derived")\n            return {\n                "requested_dataset_id": requested_repo,\n                "resolved_dataset_id": repo_id,\n                "revision": revision,\n                "split": split,\n                "config": config,\n                "rows": len(dataset),\n                "source_level": source_level,\n                "fallback_used": bool(candidate_index),\n                "parquet_export": used_parquet_export,\n            }\n        except Exception as exc:\n            last_error = exc\n            if repo_id == requested_repo and len(candidates) > 1:\n                print(\n                    f"Official dataset {repo_id} is unavailable ({exc}). "\n                    f"Using declared public derivative {candidates[1]}.",\n                    flush=True,\n                )\n                continue\n            break\n    assert last_error is not None\n    raise last_error\n\n\ndef save_archive(entry: dict, target: Path, retries: int) -> dict:\n    repo_id = entry["dataset_id"]\n    revision = retry(f"resolve {repo_id}", retries, lambda: dataset_revision(repo_id))\n    token = os.environ.get("HF_TOKEN") or None\n    archive = retry(\n        f"download {repo_id}/{entry[\'archive_file\']}",\n        retries,\n        lambda: hf_hub_download(\n            repo_id=repo_id,\n            repo_type="dataset",\n            filename=entry["archive_file"],\n            revision=revision,\n            token=token,\n        ),\n    )\n    target.mkdir(parents=True, exist_ok=True)\n    with zipfile.ZipFile(archive) as bundle:\n        members = bundle.infolist()\n        for member in tqdm(members, desc=entry["data_key"], unit="file"):\n            resolved = (target / member.filename).resolve()\n            if target.resolve() not in resolved.parents and resolved != target.resolve():\n                raise RuntimeError(f"Unsafe archive member: {member.filename}")\n            bundle.extract(member, target)\n    return {\n        "requested_dataset_id": repo_id,\n        "resolved_dataset_id": repo_id,\n        "revision": revision,\n        "archive_file": entry["archive_file"],\n        "source_level": entry["source_level"],\n        "fallback_used": False,\n    }\n\n\ndef install_link(entry: dict, shared_target: Path, metadata: dict) -> None:\n    model_dir = ROOT / entry["model"]\n    test_data = model_dir / "test_data"\n    test_data.mkdir(parents=True, exist_ok=True)\n    link = test_data / "benchmark"\n    if link.is_symlink():\n        link.unlink()\n    elif link.exists():\n        raise RuntimeError(f"Refusing to replace non-symlink dataset path: {link}")\n    relative = os.path.relpath(shared_target, test_data)\n    link.symlink_to(relative, target_is_directory=True)\n    effective_level = entry["source_level"]\n    if metadata.get("fallback_used") and entry.get("fallback_source_level"):\n        effective_level = entry["fallback_source_level"]\n    source = {\n        **metadata,\n        "model": entry["model"],\n        "adapter": entry["adapter"],\n        "backend": entry["backend"],\n        "evidence": entry["evidence"],\n        "source_note": entry.get("source_note", ""),\n        "source_level": effective_level,\n    }\n    (test_data / "source.json").write_text(\n        json.dumps(source, ensure_ascii=False, indent=2) + "\\n", encoding="utf-8"\n    )\n\n\ndef prepare_auxiliary_model(entry: dict, retries: int) -> None:\n    if entry["backend"] != "mammoscreen":\n        return\n    repo_id = "ianpan/mammo-crop"\n    target = ROOT / entry["model"] / "test_code" / "upstream" / "mammo-crop"\n    if (target / "config.json").is_file():\n        print(f"Keeping auxiliary model: {target}")\n        return\n    revision = retry(f"resolve {repo_id}", retries, lambda: HfApi().model_info(repo_id).sha)\n    retry(\n        f"download auxiliary model {repo_id}", retries,\n        lambda: snapshot_download(\n            repo_id=repo_id,\n            revision=revision,\n            local_dir=target,\n            token=os.environ.get("HF_TOKEN") or None,\n        ),\n    )\n    (target / "SOURCE.json").write_text(\n        json.dumps({"repo_id": repo_id, "revision": revision}, indent=2) + "\\n",\n        encoding="utf-8",\n    )\n\n\ndef main() -> None:\n    args = arguments()\n    entries = load_plan(args.plan, set(args.only), set(args.exclude))\n    shared_root = ROOT / "_shared_test_resources" / "data" / "remaining"\n    shared_root.mkdir(parents=True, exist_ok=True)\n    by_key: dict[str, list[dict]] = {}\n    for entry in entries:\n        by_key.setdefault(entry["data_key"], []).append(entry)\n\n    if args.list_only:\n        for key, consumers in by_key.items():\n            first = consumers[0]\n            print(f"{key}\\t{first[\'dataset_id\']}\\t{\',\'.join(item[\'model\'] for item in consumers)}")\n        print(f"Planned {len(by_key)} shared dataset(s) for {len(entries)} model(s).")\n        return\n\n    progress = tqdm(by_key.items(), desc="Datasets", unit="dataset")\n    failures: list[tuple[str, str]] = []\n    status_path = shared_root / "download_status.tsv"\n    status_path.write_text("data_key\\tstatus\\tdetail\\n", encoding="utf-8")\n    for key, consumers in progress:\n        progress.set_postfix_str(key)\n        canonical = consumers[0]\n        bundle_dir = shared_root / key\n        target = bundle_dir / ("extracted" if canonical.get("archive_file") else "dataset")\n        source_path = bundle_dir / "source.json"\n        if target.exists() and source_path.is_file():\n            metadata = json.loads(source_path.read_text(encoding="utf-8"))\n            print(f"Keeping completed shared dataset: {key} ({metadata.get(\'rows\', \'archive\')})")\n        else:\n            partial = bundle_dir.with_name(bundle_dir.name + ".partial")\n            if bundle_dir.exists():\n                raise RuntimeError(\n                    f"Incomplete target exists and was not overwritten: {bundle_dir}. "\n                    "Inspect it, then remove only that incomplete directory before retrying."\n                )\n            if partial.exists():\n                shutil.rmtree(partial)\n            partial_target = partial / target.name\n            partial.mkdir(parents=True, exist_ok=True)\n            try:\n                if canonical.get("archive_file"):\n                    metadata = save_archive(canonical, partial_target, args.retries)\n                else:\n                    metadata = save_hf_dataset(canonical, partial_target, args.retries)\n                (partial / "source.json").write_text(\n                    json.dumps(metadata, ensure_ascii=False, indent=2) + "\\n", encoding="utf-8"\n                )\n                partial.rename(bundle_dir)\n            except Exception as exc:\n                print(f"Partial files retained for diagnosis: {partial}", flush=True)\n                failures.append((key, repr(exc)))\n                with status_path.open("a", encoding="utf-8") as handle:\n                    handle.write(f"{key}\\tFAILED\\t{exc!r}\\n")\n                print(f"Continuing with remaining datasets after failure: {key}", flush=True)\n                continue\n        for entry in consumers:\n            install_link(entry, target, metadata)\n            prepare_auxiliary_model(entry, args.retries)\n        with status_path.open("a", encoding="utf-8") as handle:\n            handle.write(f"{key}\\tOK\\t{metadata.get(\'resolved_dataset_id\', \'\')}\\n")\n    print(f"Downloaded {len(by_key) - len(failures)}/{len(by_key)} shared dataset(s) for {len(entries)} model(s).")\n    if failures:\n        print(f"Failed datasets: {len(failures)}. Re-run this command after the rate limit/network issue clears.")\n        print(f"Failure status: {status_path}")\n        raise SystemExit(1)\n\n\nif __name__ == "__main__":\n    main()\n', '_shared_test_resources/code/finalize_imagenet_result.py': '#!/usr/bin/env python3\n"""Normalize official ImageNet evaluator output into summary.txt and metrics.xlsx."""\nfrom __future__ import annotations\n\nimport argparse\nimport csv\nimport json\nimport re\nfrom pathlib import Path\n\nimport pandas as pd\n\n\ndef ratio(value: str | float) -> float:\n    number = float(value)\n    return number / 100.0 if number > 1.0 else number\n\n\ndef main() -> None:\n    parser = argparse.ArgumentParser()\n    parser.add_argument("--result-dir", type=Path, required=True)\n    parser.add_argument("--model", required=True)\n    parser.add_argument("--reference-top1", type=float, required=True)\n    parser.add_argument("--reference-top5", type=float)\n    parser.add_argument("--csv", type=Path)\n    parser.add_argument("--log", type=Path)\n    args = parser.parse_args()\n\n    summary: dict[str, object] = {\n        "model": args.model,\n        "dataset": "ImageNet-1K validation",\n        "evaluated_predictions": 50_000,\n        "official_top1_accuracy": args.reference_top1,\n    }\n    if args.reference_top5 is not None:\n        summary["official_top5_accuracy"] = args.reference_top5\n\n    if args.csv and args.csv.exists():\n        with args.csv.open(newline="", encoding="utf-8") as handle:\n            rows = list(csv.DictReader(handle))\n        if rows:\n            row = rows[-1]\n            summary["official_evaluator_row"] = row\n            for key, value in row.items():\n                normalized = key.lower().replace("@", "").replace("_", "")\n                if value in (None, ""):\n                    continue\n                if normalized in {"top1", "acc1", "accuracy", "top1accuracy"}:\n                    summary["top1_accuracy"] = ratio(value)\n                    summary["accuracy"] = ratio(value)\n                elif normalized in {"top5", "acc5", "top5accuracy"}:\n                    summary["top5_accuracy"] = ratio(value)\n\n    if args.log and args.log.exists():\n        text = args.log.read_text(encoding="utf-8", errors="replace")\n        matches = re.findall(\n            r"(?:Acc@1|acc1|top-?1)\\s*[:=]?\\s*([0-9.]+).*?(?:Acc@5|acc5|top-?5)\\s*[:=]?\\s*([0-9.]+)",\n            text,\n            flags=re.IGNORECASE,\n        )\n        if matches:\n            top1, top5 = matches[-1]\n            summary["top1_accuracy"] = ratio(top1)\n            summary["accuracy"] = ratio(top1)\n            summary["top5_accuracy"] = ratio(top5)\n\n    if "top1_accuracy" not in summary:\n        raise RuntimeError("Could not parse top-1 accuracy from the official evaluator output")\n    summary["delta_top1_accuracy"] = float(summary["top1_accuracy"]) - args.reference_top1\n    if args.reference_top5 is not None and "top5_accuracy" in summary:\n        summary["delta_top5_accuracy"] = float(summary["top5_accuracy"]) - args.reference_top5\n\n    args.result_dir.mkdir(parents=True, exist_ok=True)\n    (args.result_dir / "summary.txt").write_text(\n        json.dumps(summary, ensure_ascii=False, indent=2) + "\\n", encoding="utf-8"\n    )\n    pd.DataFrame([summary]).to_excel(args.result_dir / "metrics.xlsx", index=False)\n    print(json.dumps(summary, ensure_ascii=False, indent=2))\n\n\nif __name__ == "__main__":\n    main()\n', '_shared_test_resources/code/hf_classification_evaluator.py': '#!/usr/bin/env python3\n"""Evaluate a local Hugging Face image classifier on local Parquet image data.\n\nThe evaluator never trains or changes model weights.  It loads the downloaded\ncheckpoint, applies the checkpoint\'s official AutoImageProcessor, performs\nbatched inference, and writes every prediction plus aggregate classification\nmetrics to TXT and XLSX files.\n"""\n\nfrom __future__ import annotations\n\nimport argparse\nimport io\nimport json\nimport time\nfrom pathlib import Path\n\nimport numpy as np\nimport pyarrow.parquet as pq\nimport torch\nfrom openpyxl import Workbook\nfrom PIL import Image\nfrom sklearn.metrics import (\n    accuracy_score,\n    classification_report,\n    confusion_matrix,\n    precision_recall_fscore_support,\n)\nfrom tqdm.auto import tqdm\nfrom transformers import AutoImageProcessor, AutoModelForImageClassification\n\n\ndef parse_args() -> argparse.Namespace:\n    parser = argparse.ArgumentParser()\n    parser.add_argument("--model-dir", type=Path, required=True)\n    parser.add_argument("--data-glob", type=str, required=True)\n    parser.add_argument("--result-dir", type=Path, required=True)\n    parser.add_argument("--batch-size", type=int, default=32)\n    parser.add_argument("--device", default="cuda")\n    parser.add_argument("--max-images", "--max-samples", dest="max_samples", type=int, default=0)\n    parser.add_argument(\n        "--expected-samples",\n        type=int,\n        default=0,\n        help="Fail before inference unless the downloaded benchmark has this many rows.",\n    )\n    parser.add_argument(\n        "--ignore-pred-label",\n        type=int,\n        default=None,\n        help="Exclude rows receiving this prediction (used by an official benchmark protocol).",\n    )\n    parser.add_argument(\n        "--pred-label-map",\n        default="",\n        help=\'JSON mapping from model prediction ids to dataset label ids, e.g. "{\\\\"0\\\\":0,\\\\"1\\\\":1}".\',\n    )\n    parser.add_argument(\n        "--reference-metrics",\n        default="",\n        help="JSON object with model-card reference metrics for delta reporting.",\n    )\n    parser.add_argument(\n        "--positive-label",\n        type=int,\n        default=None,\n        help="Dataset label id whose precision/recall/F1 must be reported separately.",\n    )\n    parser.add_argument("--positive-name", default="positive")\n    parser.add_argument("--protocol-note", default="")\n    return parser.parse_args()\n\n\ndef find_parquet_files(pattern: str) -> list[Path]:\n    import glob\n\n    files = [Path(item) for item in sorted(glob.glob(pattern, recursive=True))]\n    if not files:\n        raise FileNotFoundError(f"No Parquet files matched: {pattern}")\n    return files\n\n\ndef decode_image(value: object) -> Image.Image:\n    if isinstance(value, dict):\n        raw = value.get("bytes")\n        path = value.get("path")\n    else:\n        raw = value["bytes"] if value["bytes"] is not None else None\n        path = value["path"] if value["path"] is not None else None\n    if raw is not None:\n        return Image.open(io.BytesIO(raw)).convert("RGB")\n    if path:\n        return Image.open(path).convert("RGB")\n    raise ValueError("Image record contains neither bytes nor path")\n\n\ndef iter_examples(files: list[Path], max_samples: int):\n    emitted = 0\n    for parquet_path in files:\n        parquet = pq.ParquetFile(parquet_path)\n        columns = set(parquet.schema_arrow.names)\n        if "image" not in columns or "label" not in columns:\n            raise ValueError(\n                f"Expected image and label columns in {parquet_path}; got {sorted(columns)}"\n            )\n        for record_batch in parquet.iter_batches(\n            batch_size=256, columns=["image", "label"]\n        ):\n            data = record_batch.to_pydict()\n            for image_value, label in zip(data["image"], data["label"]):\n                yield decode_image(image_value), int(label), str(parquet_path)\n                emitted += 1\n                if max_samples and emitted >= max_samples:\n                    return\n\n\ndef batches(iterator, size: int):\n    batch = []\n    for item in iterator:\n        batch.append(item)\n        if len(batch) == size:\n            yield batch\n            batch = []\n    if batch:\n        yield batch\n\n\ndef safe_json(raw: str) -> dict:\n    return json.loads(raw) if raw else {}\n\n\ndef write_xlsx(\n    path: Path,\n    summary: dict[str, object],\n    report: dict[str, object],\n    matrix: np.ndarray,\n    predictions: list[tuple[int, int, int, float, str]],\n) -> None:\n    workbook = Workbook()\n    ws = workbook.active\n    ws.title = "summary"\n    ws.append(["metric", "value"])\n    for key, value in summary.items():\n        ws.append([key, value])\n\n    metrics = workbook.create_sheet("classification_report")\n    metrics.append(["class", "precision", "recall", "f1-score", "support"])\n    for name, values in report.items():\n        if isinstance(values, dict):\n            metrics.append(\n                [\n                    name,\n                    values.get("precision"),\n                    values.get("recall"),\n                    values.get("f1-score"),\n                    values.get("support"),\n                ]\n            )\n        else:\n            metrics.append([name, values])\n\n    cm_sheet = workbook.create_sheet("confusion_matrix")\n    cm_sheet.append(["true\\\\pred"] + list(range(matrix.shape[1])))\n    for index, row in enumerate(matrix.tolist()):\n        cm_sheet.append([index] + row)\n\n    pred_sheet = workbook.create_sheet("predictions")\n    pred_sheet.append(["index", "true_label", "pred_label", "confidence", "source"])\n    for item in predictions:\n        pred_sheet.append(list(item))\n    workbook.save(path)\n\n\ndef main() -> None:\n    args = parse_args()\n    args.result_dir.mkdir(parents=True, exist_ok=True)\n    files = find_parquet_files(args.data_glob)\n    available_rows = sum(pq.ParquetFile(path).metadata.num_rows for path in files)\n    if args.expected_samples and available_rows != args.expected_samples:\n        raise RuntimeError(\n            f"Official support mismatch: downloaded {available_rows} rows, "\n            f"expected {args.expected_samples}. The dataset revision or file selection differs."\n        )\n    device = torch.device(\n        args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu"\n    )\n\n    processor = AutoImageProcessor.from_pretrained(\n        args.model_dir, local_files_only=True, trust_remote_code=True\n    )\n    model = AutoModelForImageClassification.from_pretrained(\n        args.model_dir, local_files_only=True, trust_remote_code=True\n    ).to(device)\n    model.eval()\n\n    pred_map = {int(k): int(v) for k, v in safe_json(args.pred_label_map).items()}\n    predictions: list[tuple[int, int, int, float, str]] = []\n    started = time.time()\n\n    target_rows = min(available_rows, args.max_samples) if args.max_samples else available_rows\n    iterator = iter_examples(files, args.max_samples)\n    with torch.inference_mode():\n        progress = tqdm(total=target_rows, desc=args.model_dir.name, unit="image")\n        for batch in batches(iterator, args.batch_size):\n            images = [item[0] for item in batch]\n            inputs = processor(images=images, return_tensors="pt")\n            inputs = {key: value.to(device) for key, value in inputs.items()}\n            logits = model(**inputs).logits\n            probabilities = logits.softmax(dim=-1)\n            confidence, predicted = probabilities.max(dim=-1)\n            for image_item, pred, score in zip(\n                batch, predicted.cpu().tolist(), confidence.cpu().tolist()\n            ):\n                true_label = image_item[1]\n                mapped_pred = pred_map.get(pred, pred)\n                predictions.append(\n                    (len(predictions), true_label, mapped_pred, float(score), image_item[2])\n                )\n            progress.update(len(batch))\n        progress.close()\n\n    evaluated = [\n        item\n        for item in predictions\n        if args.ignore_pred_label is None or item[2] != args.ignore_pred_label\n    ]\n    y_true = np.asarray([item[1] for item in evaluated], dtype=np.int64)\n    y_pred = np.asarray([item[2] for item in evaluated], dtype=np.int64)\n    if not len(y_true):\n        raise RuntimeError("No predictions remained after applying the benchmark protocol")\n\n    labels = sorted(set(y_true.tolist()) | set(y_pred.tolist()))\n    report = classification_report(\n        y_true, y_pred, labels=labels, output_dict=True, zero_division=0\n    )\n    matrix = confusion_matrix(y_true, y_pred, labels=labels)\n    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(\n        y_true, y_pred, average="macro", zero_division=0\n    )\n    precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(\n        y_true, y_pred, average="weighted", zero_division=0\n    )\n    summary: dict[str, object] = {\n        "model_dir": str(args.model_dir.resolve()),\n        "data_glob": args.data_glob,\n        "data_files": len(files),\n        "available_dataset_rows": available_rows,\n        "total_predictions": len(predictions),\n        "evaluated_predictions": len(evaluated),\n        "excluded_predictions": len(predictions) - len(evaluated),\n        "accuracy": float(accuracy_score(y_true, y_pred)),\n        "macro_precision": float(precision_macro),\n        "macro_recall": float(recall_macro),\n        "macro_f1": float(f1_macro),\n        "weighted_precision": float(precision_weighted),\n        "weighted_recall": float(recall_weighted),\n        "weighted_f1": float(f1_weighted),\n        "elapsed_seconds": round(time.time() - started, 3),\n        "device": str(device),\n        "protocol_note": args.protocol_note,\n    }\n    if args.positive_label is not None:\n        positive_precision, positive_recall, positive_f1, positive_support = (\n            precision_recall_fscore_support(\n                y_true,\n                y_pred,\n                labels=[args.positive_label],\n                average=None,\n                zero_division=0,\n            )\n        )\n        prefix = args.positive_name.strip().lower().replace(" ", "_")\n        summary[f"{prefix}_precision"] = float(positive_precision[0])\n        summary[f"{prefix}_recall"] = float(positive_recall[0])\n        summary[f"{prefix}_f1"] = float(positive_f1[0])\n        summary[f"{prefix}_support"] = int(positive_support[0])\n    for metric, reference in safe_json(args.reference_metrics).items():\n        summary[f"official_{metric}"] = reference\n        if metric in summary:\n            summary[f"delta_{metric}"] = float(summary[metric]) - float(reference)\n\n    with (args.result_dir / "predictions.txt").open("w", encoding="utf-8") as handle:\n        handle.write("index\\ttrue_label\\tpred_label\\tconfidence\\tsource\\n")\n        for item in predictions:\n            handle.write("\\t".join(map(str, item)) + "\\n")\n    with (args.result_dir / "summary.txt").open("w", encoding="utf-8") as handle:\n        for key, value in summary.items():\n            handle.write(f"{key}\\t{value}\\n")\n    with (args.result_dir / "classification_report.txt").open(\n        "w", encoding="utf-8"\n    ) as handle:\n        handle.write(json.dumps(report, ensure_ascii=False, indent=2))\n        handle.write("\\n")\n    with (args.result_dir / "confusion_matrix.txt").open(\n        "w", encoding="utf-8"\n    ) as handle:\n        handle.write("labels\\t" + "\\t".join(map(str, labels)) + "\\n")\n        for label, row in zip(labels, matrix.tolist()):\n            handle.write(str(label) + "\\t" + "\\t".join(map(str, row)) + "\\n")\n    write_xlsx(args.result_dir / "metrics.xlsx", summary, report, matrix, predictions)\n    print(json.dumps(summary, ensure_ascii=False, indent=2))\n\n\nif __name__ == "__main__":\n    main()\n', '_shared_test_resources/code/imagefolder_evaluator.py': '#!/usr/bin/env python3\n"""Evaluate a local HF/timm image classifier on an ImageFolder test set.\n\nThe program is deliberately inference-only.  It writes per-image predictions,\naggregate metrics, a confusion matrix and the published reference metrics into\nthe model\'s ``result`` directory.  Dataset paths are supplied by each model\'s\n``test_code/run_test.sh`` and are always relative to that model directory.\n"""\nfrom __future__ import annotations\n\nimport argparse\nimport json\nimport sys\nimport time\nfrom pathlib import Path\n\nimport numpy as np\nimport pandas as pd\nimport torch\nimport torch.nn.functional as F\nfrom PIL import Image\nfrom sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_recall_fscore_support\nfrom torch.utils.data import DataLoader, Dataset\nfrom tqdm.auto import tqdm\n\n\nIMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}\n\n\ndef arguments() -> argparse.Namespace:\n    p = argparse.ArgumentParser()\n    p.add_argument("--model-dir", type=Path, required=True)\n    p.add_argument("--data-dir", type=Path, required=True)\n    p.add_argument("--backend", choices=("hf", "timm", "mambavision", "nepa"), default="hf")\n    p.add_argument("--official-source-dir", type=Path, help="Local official source tree for custom model code")\n    p.add_argument("--hf-input-key", default="pixel_values", help="Forward argument for custom HF code")\n    p.add_argument("--fallback-image-size", type=int, default=224)\n    p.add_argument("--fallback-mean", default="0.5,0.5,0.5")\n    p.add_argument("--fallback-std", default="0.5,0.5,0.5")\n    p.add_argument("--grayscale", action="store_true")\n    p.add_argument("--report-topk", default="1,5", help="Comma-separated top-k accuracies to save")\n    p.add_argument("--class-map", default="", help="JSON: folder-name -> model label id")\n    p.add_argument("--reference-metrics", default="{}")\n    p.add_argument("--protocol-note", default="")\n    p.add_argument("--result-dir", type=Path, required=True)\n    p.add_argument("--batch-size", type=int, default=32)\n    p.add_argument("--device", default="cuda")\n    p.add_argument("--max-samples", type=int, default=0)\n    p.add_argument("--expected-samples", type=int, default=0)\n    return p.parse_args()\n\n\nclass FolderDataset(Dataset):\n    def __init__(self, root: Path, transform, class_map: dict[str, int], limit: int):\n        self.transform = transform\n        self.rows: list[tuple[Path, int, str]] = []\n        for folder in sorted(p for p in root.iterdir() if p.is_dir()):\n            if folder.name not in class_map:\n                continue\n            for path in sorted(p for p in folder.rglob("*") if p.suffix.lower() in IMAGE_SUFFIXES):\n                self.rows.append((path, class_map[folder.name], folder.name))\n        if limit:\n            self.rows = self.rows[:limit]\n        if not self.rows:\n            raise FileNotFoundError(\n                f"No labelled images under {root}. Expected class folders: {sorted(class_map)}"\n            )\n\n    def __len__(self) -> int:\n        return len(self.rows)\n\n    def __getitem__(self, index: int):\n        path, label, label_name = self.rows[index]\n        with Image.open(path) as image:\n            pixels = self.transform(image.convert("RGB"))\n        return pixels, label, label_name, str(path)\n\n\ndef normalize(text: str) -> str:\n    return "".join(ch for ch in text.lower() if ch.isalnum())\n\n\ndef hf_backend(model_dir: Path, fallback_image_size: int, grayscale: bool, mean: str, std: str):\n    from transformers import AutoImageProcessor, AutoModelForImageClassification\n    try:\n        processor = AutoImageProcessor.from_pretrained(model_dir, local_files_only=True, trust_remote_code=True)\n    except OSError:\n        processor = None\n    model = AutoModelForImageClassification.from_pretrained(model_dir, local_files_only=True, trust_remote_code=True)\n    raw = model.config.id2label or {}\n    class_count = getattr(model.config, "num_labels", None) or getattr(model.config, "num_classes", 0)\n    labels = {int(k): str(v) for k, v in raw.items()} or {i: str(i) for i in range(class_count)}\n    if getattr(model.config, "model_type", "") == "timm_wrapper":\n        from torchvision.transforms import Compose, Normalize, Resize, ToTensor\n\n        pretrained = model.config.pretrained_cfg\n        image_size = int(pretrained["input_size"][-1])\n        transform = Compose(\n            [\n                Resize((image_size, image_size)),\n                ToTensor(),\n                Normalize(mean=pretrained["mean"], std=pretrained["std"]),\n            ]\n        )\n        return model, lambda images: torch.stack([transform(image) for image in images]), labels\n    if processor is not None:\n        return model, lambda images: processor(images=images, return_tensors="pt")["pixel_values"], labels\n    from torchvision.transforms import Compose, Grayscale, Normalize, Resize, ToTensor\n    steps = [Grayscale(num_output_channels=1)] if grayscale else []\n    if grayscale:\n        mean_values, std_values = [0.0], [1.0]\n    else:\n        mean_values = [float(item) for item in mean.split(",")]\n        std_values = [float(item) for item in std.split(",")]\n    transform = Compose(steps + [Resize((fallback_image_size, fallback_image_size)), ToTensor(), Normalize(mean_values, std_values)])\n    return model, lambda images: torch.stack([transform(image) for image in images]), labels\n\n\ndef timm_backend(model_dir: Path):\n    import timm\n    from timm.data import create_transform, resolve_data_config\n    model = timm.create_model(str(model_dir), pretrained=True)\n    config = resolve_data_config({}, model=model)\n    transform = create_transform(**config, is_training=False)\n    labels = {index: str(index) for index in range(model.num_classes)}\n    return model, lambda images: torch.stack([transform(image) for image in images]), labels\n\n\ndef mambavision_backend(model_dir: Path):\n    """Use the exact transform published in NVIDIA\'s MambaVision model card."""\n    from timm.data.transforms_factory import create_transform\n    from transformers import AutoModelForImageClassification\n\n    model = AutoModelForImageClassification.from_pretrained(\n        model_dir, local_files_only=True, trust_remote_code=True\n    )\n    transform = create_transform(\n        input_size=(3, 512, 512),\n        is_training=False,\n        mean=model.config.mean,\n        std=model.config.std,\n        crop_mode=model.config.crop_mode,\n        crop_pct=model.config.crop_pct,\n    )\n    raw = model.config.id2label or {}\n    labels = {int(k): str(v) for k, v in raw.items()} or {\n        index: str(index) for index in range(model.config.num_labels)\n    }\n    return model, lambda images: torch.stack([transform(image) for image in images]), labels\n\n\ndef nepa_backend(model_dir: Path, source_dir: Path | None):\n    """Load NEPA through its official class and reproduce its published eval transform."""\n    if source_dir is None:\n        raise ValueError("--official-source-dir is required for the NEPA backend")\n    sys.path.insert(0, str(source_dir.resolve()))\n    from models.vit_nepa import ViTNepaConfig, ViTNepaForImageClassification\n    from torchvision.transforms import CenterCrop, Compose, InterpolationMode, Normalize, Resize, ToTensor\n\n    config = ViTNepaConfig.from_pretrained(model_dir, local_files_only=True)\n    model = ViTNepaForImageClassification.from_pretrained(\n        model_dir,\n        config=config,\n        local_files_only=True,\n    )\n    raw = model.config.id2label or {}\n    labels = {int(k): str(v) for k, v in raw.items()} or {\n        index: str(index) for index in range(model.config.num_labels)\n    }\n    transform = Compose(\n        [\n            Resize(256, interpolation=InterpolationMode.BICUBIC),\n            CenterCrop((224, 224)),\n            ToTensor(),\n            Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),\n        ]\n    )\n    return model, lambda images: torch.stack([transform(image) for image in images]), labels\n\n\ndef main() -> None:\n    a = arguments()\n    a.result_dir.mkdir(parents=True, exist_ok=True)\n    if a.backend == "hf":\n        model, preprocess, id2label = hf_backend(a.model_dir, a.fallback_image_size, a.grayscale, a.fallback_mean, a.fallback_std)\n    elif a.backend == "timm":\n        model, preprocess, id2label = timm_backend(a.model_dir)\n    elif a.backend == "mambavision":\n        model, preprocess, id2label = mambavision_backend(a.model_dir)\n    else:\n        model, preprocess, id2label = nepa_backend(a.model_dir, a.official_source_dir)\n    class_map = {str(k): int(v) for k, v in json.loads(a.class_map).items()} if a.class_map else {}\n    if not class_map:\n        by_name = {normalize(name): index for index, name in id2label.items()}\n        class_map = {folder.name: by_name[normalize(folder.name)] for folder in a.data_dir.iterdir() if folder.is_dir() and normalize(folder.name) in by_name}\n        if not class_map:\n            folders = sorted(folder for folder in a.data_dir.iterdir() if folder.is_dir())\n            class_map = {folder.name: int(folder.name) if folder.name.isdigit() else index for index, folder in enumerate(folders)}\n    dataset = FolderDataset(a.data_dir, lambda image: image, class_map, 0)\n    if a.expected_samples and len(dataset) != a.expected_samples:\n        raise RuntimeError(\n            f"Official support mismatch: found {len(dataset)} images, expected {a.expected_samples}"\n        )\n    if a.max_samples:\n        dataset = torch.utils.data.Subset(\n            dataset, range(min(a.max_samples, len(dataset)))\n        )\n    device = torch.device(a.device if a.device != "cuda" or torch.cuda.is_available() else "cpu")\n    model.to(device).eval()\n    loader = DataLoader(dataset, batch_size=a.batch_size, shuffle=False, num_workers=4, collate_fn=lambda batch: batch)\n    truth: list[int] = []\n    predicted: list[int] = []\n    rows: list[dict[str, object]] = []\n    topk = sorted({int(item) for item in a.report_topk.split(",") if int(item) > 0})\n    topk_hits = {k: 0 for k in topk}\n    loss_sum = 0.0\n    started = time.time()\n    with torch.inference_mode():\n        for batch in tqdm(loader, desc=a.model_dir.name, unit="batch"):\n            images = [item[0] for item in batch]\n            pixels = preprocess(images).to(device)\n            if a.backend == "mambavision":\n                # NVIDIA\'s official remote-code model exposes forward(x), not\n                # the Transformers-style forward(pixel_values=...).\n                output = model(pixels)\n                logits = output.logits if hasattr(output, "logits") else output["logits"]\n            elif a.backend in {"hf", "nepa"}:\n                output = model(**{a.hf_input_key: pixels})\n                logits = output.logits if hasattr(output, "logits") else output["logits"]\n            else:\n                logits = model(pixels)\n            targets = torch.tensor([item[1] for item in batch], device=device)\n            loss_sum += float(F.cross_entropy(logits, targets, reduction="sum").item())\n            probabilities = logits.softmax(dim=-1)\n            confidence, labels = probabilities.max(dim=-1)\n            ranked = probabilities.topk(min(max(topk), probabilities.shape[1]), dim=-1).indices.cpu().tolist()\n            for item, label, score, top_ids in zip(batch, labels.cpu().tolist(), confidence.cpu().tolist(), ranked):\n                truth.append(item[1]); predicted.append(label)\n                for k in topk:\n                    topk_hits[k] += int(item[1] in top_ids[:k])\n                rows.append({"path": item[3], "true_id": item[1], "true_label": item[2], "pred_id": label, "pred_label": id2label.get(label, str(label)), "confidence": score})\n    all_labels = sorted(set(truth) | set(predicted))\n    macro = precision_recall_fscore_support(truth, predicted, average="macro", zero_division=0)\n    weighted = precision_recall_fscore_support(truth, predicted, average="weighted", zero_division=0)\n    summary: dict[str, object] = {\n        "evaluated_predictions": len(truth), "accuracy": accuracy_score(truth, predicted),\n        "loss": loss_sum / len(truth),\n        "macro_precision": macro[0], "macro_recall": macro[1], "macro_f1": macro[2],\n        "weighted_precision": weighted[0], "weighted_recall": weighted[1], "weighted_f1": weighted[2],\n        "backend": a.backend, "model_dir": str(a.model_dir), "data_dir": str(a.data_dir),\n        "protocol_note": a.protocol_note, "elapsed_seconds": round(time.time() - started, 3), "device": str(device),\n    }\n    for k, hits in topk_hits.items():\n        summary[f"top{k}_accuracy"] = hits / len(truth)\n    for name, value in json.loads(a.reference_metrics).items():\n        summary[f"official_{name}"] = value\n        if name in summary and isinstance(value, (int, float)):\n            summary[f"delta_{name}"] = summary[name] - value\n    pd.DataFrame(rows).to_csv(a.result_dir / "predictions.tsv", sep="\\t", index=False)\n    (a.result_dir / "summary.txt").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\\n", encoding="utf-8")\n    (a.result_dir / "classification_report.txt").write_text(classification_report(truth, predicted, labels=all_labels, zero_division=0) + "\\n", encoding="utf-8")\n    np.savetxt(a.result_dir / "confusion_matrix.txt", confusion_matrix(truth, predicted, labels=all_labels), fmt="%d")\n    pd.DataFrame([summary]).to_excel(a.result_dir / "metrics.xlsx", index=False)\n\n\nif __name__ == "__main__":\n    main()\n', '_shared_test_resources/code/prepare_imagenet_val.py': '#!/usr/bin/env python3\n"""Convert the official HF ImageNet-1K validation Parquet shards to ImageFolder."""\nfrom __future__ import annotations\n\nimport argparse\nimport shutil\nfrom pathlib import Path\n\nimport pyarrow.parquet as pq\nfrom tqdm import tqdm\n\n\ndef main() -> None:\n    parser = argparse.ArgumentParser()\n    parser.add_argument("--source", type=Path, required=True)\n    parser.add_argument("--destination", type=Path, required=True)\n    args = parser.parse_args()\n\n    shards = sorted(args.source.glob("validation-*.parquet"))\n    if len(shards) != 14:\n        raise RuntimeError(f"Expected 14 ImageNet validation shards, found {len(shards)}")\n    expected = sum(pq.ParquetFile(path).metadata.num_rows for path in shards)\n    if expected != 50_000:\n        raise RuntimeError(f"Expected 50000 validation rows, found {expected}")\n\n    args.destination.mkdir(parents=True, exist_ok=True)\n    written = 0\n    with tqdm(total=expected, desc="Preparing ImageNet val", unit="image") as progress:\n        for shard_index, shard in enumerate(shards):\n            parquet = pq.ParquetFile(shard)\n            row_index = 0\n            for batch in parquet.iter_batches(batch_size=128, columns=["image", "label"]):\n                values = batch.to_pydict()\n                for image_value, label in zip(values["image"], values["label"]):\n                    class_dir = args.destination / f"{int(label):04d}"\n                    class_dir.mkdir(exist_ok=True)\n                    output = class_dir / f"val_{shard_index:02d}_{row_index:05d}.JPEG"\n                    row_index += 1\n                    if not output.exists():\n                        raw = image_value.get("bytes") if isinstance(image_value, dict) else image_value["bytes"]\n                        source_path = image_value.get("path") if isinstance(image_value, dict) else image_value["path"]\n                        if raw is not None:\n                            # Preserve the official encoded bytes exactly; re-encoding\n                            # JPEGs can change a few ImageNet predictions.\n                            output.write_bytes(raw)\n                        elif source_path:\n                            shutil.copyfile(source_path, output)\n                        else:\n                            raise RuntimeError(f"Missing image bytes/path in {shard}:{row_index - 1}")\n                    written += 1\n                    progress.update(1)\n\n    files = sum(1 for path in args.destination.rglob("*.JPEG") if path.is_file())\n    classes = sum(1 for path in args.destination.iterdir() if path.is_dir())\n    if files != 50_000 or classes != 1_000:\n        raise RuntimeError(f"Prepared validation set is incomplete: files={files}, classes={classes}")\n    (args.destination.parent / ".validation_complete").write_text(\n        "images=50000\\nclasses=1000\\n", encoding="utf-8"\n    )\n    print(f"ImageNet validation ready: {files} images, {classes} classes")\n\n\nif __name__ == "__main__":\n    main()\n', '_shared_test_resources/code/remaining_classification_evaluator.py': '#!/usr/bin/env python3\n"""Unified inference-only evaluator for non-strict classification models."""\nfrom __future__ import annotations\n\nimport argparse\nimport io\nimport json\nimport math\nimport os\nimport re\nimport sys\nimport time\nfrom pathlib import Path\nfrom typing import Any, Iterable\n\nimport numpy as np\nimport pandas as pd\nimport torch\nfrom PIL import Image\nfrom sklearn.metrics import (\n    accuracy_score,\n    classification_report,\n    confusion_matrix,\n    mean_absolute_error,\n    precision_recall_fscore_support,\n    roc_auc_score,\n)\nfrom tqdm.auto import tqdm\n\n\nROOT = Path(__file__).resolve().parents[2]\nDEFAULT_PLAN = ROOT / "remaining_classification_plan.json"\nIMAGE_KEYS = ("image", "front", "img", "pixel_values", "photo", "file", "image_path", "path")\nLABEL_KEYS = ("label", "labels", "class", "category", "target", "class_name", "folder")\n\n\ndef arguments() -> argparse.Namespace:\n    parser = argparse.ArgumentParser()\n    parser.add_argument("--model-key", required=True)\n    parser.add_argument("--plan", type=Path, default=DEFAULT_PLAN)\n    parser.add_argument("--device", default="cuda")\n    parser.add_argument("--batch-size", type=int, default=0)\n    parser.add_argument("--max-images", "--max-samples", dest="max_samples", type=int, default=0)\n    parser.add_argument("--max-predictions", type=int, default=0)\n    parser.add_argument("--result-dir", type=Path)\n    return parser.parse_args()\n\n\ndef normalize(value: Any) -> str:\n    return re.sub(r"[^a-z0-9]+", "", str(value).lower())\n\n\ndef first_value(row: dict, names: Iterable[str], default: Any = None) -> Any:\n    normalized = {normalize(key): key for key in row}\n    for name in names:\n        key = normalized.get(normalize(name))\n        if key is not None and row[key] is not None:\n            return row[key]\n    return default\n\n\ndef image_from_value(value: Any, data_root: Path) -> Image.Image:\n    if isinstance(value, Image.Image):\n        return value.convert("RGB")\n    if isinstance(value, np.ndarray):\n        return Image.fromarray(value).convert("RGB")\n    if isinstance(value, (bytes, bytearray, memoryview)):\n        return Image.open(io.BytesIO(bytes(value))).convert("RGB")\n    if isinstance(value, dict):\n        if value.get("bytes") is not None:\n            return Image.open(io.BytesIO(value["bytes"])).convert("RGB")\n        for key in ("path", "image_path", "file"):\n            if value.get(key):\n                return image_from_value(value[key], data_root)\n    if isinstance(value, (str, Path)):\n        path = Path(value)\n        candidates = [path] if path.is_absolute() else [data_root / path, path]\n        for candidate in candidates:\n            if candidate.is_file():\n                return Image.open(candidate).convert("RGB")\n    raise ValueError(f"Cannot decode image value of type {type(value).__name__}")\n\n\ndef image_from_row(row: dict, data_root: Path) -> Image.Image:\n    value = first_value(row, IMAGE_KEYS)\n    if value is None:\n        for candidate in row.values():\n            if isinstance(candidate, (Image.Image, bytes, bytearray)):\n                value = candidate\n                break\n            if isinstance(candidate, dict) and ("bytes" in candidate or "path" in candidate):\n                value = candidate\n                break\n    if value is None:\n        raise ValueError(f"No image column among {sorted(row)}")\n    return image_from_value(value, data_root)\n\n\ndef class_name(dataset, row: dict, key_candidates=LABEL_KEYS) -> str:\n    normalized = {normalize(key): key for key in row}\n    key = next((normalized[normalize(candidate)] for candidate in key_candidates if normalize(candidate) in normalized), None)\n    if key is None:\n        raise ValueError(f"No label column among {sorted(row)}")\n    value = row[key]\n    feature = getattr(dataset, "features", {}).get(key)\n    if isinstance(value, (int, np.integer)) and hasattr(feature, "int2str"):\n        return str(feature.int2str(int(value)))\n    return str(value)\n\n\ndef model_labels(model) -> dict[int, str]:\n    raw = getattr(model.config, "id2label", None) or {}\n    count = int(getattr(model.config, "num_labels", 0) or 0)\n    return {int(key): str(value) for key, value in raw.items()} or {\n        index: str(index) for index in range(count)\n    }\n\n\ndef match_label(name: str, labels: dict[int, str]) -> int | None:\n    wanted = normalize(name)\n    aliases = {\n        "male": "man", "men": "man", "female": "woman", "women": "woman", "notsafe": "nosafe",\n        "fake": "fake", "artificial": "artificial", "human": "real",\n        "livingroom": "livingroom", "diningroom": "diningroom",\n        "housefacade": "housefacade", "no": "0", "yes": "1",\n        "radish": "raddish", "jalapeno": "jalepeno",\n    }\n    wanted = aliases.get(wanted, wanted)\n    for index, label in labels.items():\n        candidate = aliases.get(normalize(label), normalize(label))\n        if wanted == candidate:\n            return index\n    return None\n\n\ndef raw_gender(row: dict) -> str:\n    value = first_value(row, ("gender", "sex"))\n    if isinstance(value, (int, np.integer, float)):\n        return "male" if int(value) == 0 else "female"\n    text = normalize(value)\n    return "female" if text in {"1", "f", "female", "woman"} else "male"\n\n\ndef truth_name(entry: dict, dataset, row: dict) -> str | None:\n    adapter = entry["adapter"]\n    if adapter in {"direct", "deepfake_binary", "ai_deepfake_real", "bird_species"}:\n        return class_name(dataset, row)\n    if adapter == "dog_emotion":\n        return class_name(dataset, row, ("emotion", "label", "class"))\n    if adapter == "skin_condition":\n        label = normalize(class_name(dataset, row, ("type", "condition", "skin_condition", "label", "class")))\n        return {\n            "acne": "acne",\n            "skinredness": "skin redness",\n            "redness": "skin redness",\n            "bags": "puffy eyes",\n            "bagsundereyes": "puffy eyes",\n            "undereyebags": "puffy eyes",\n        }.get(label)\n    if adapter == "cifake":\n        return class_name(dataset, row)\n    if adapter == "all_unwatermarked":\n        return "no watermark detected"\n    if adapter == "gender":\n        return "woman" if raw_gender(row) == "female" else "man"\n    if adapter == "gender_age":\n        age = float(first_value(row, ("age",)))\n        return f"model-{raw_gender(row)}-{\'child\' if age < 18 else \'adult\'}"\n    if adapter == "beauty_binary":\n        score = float(first_value(row, ("beauty", "beauty_score", "score", "rating", "label")))\n        return "pos" if score >= 3.0 else "neg"\n    if adapter == "elpv_severity":\n        label = normalize(class_name(dataset, row, ("label4", "label", "class")))\n        mapping = {\n            "nodefect": "no defect", "certainlydefective": "serious defect",\n            "likelydefective": "medium defect", "possiblydefective": "minor defect",\n            "functional": "no defect",\n            "0": "no defect", "1": "minor defect", "2": "medium defect", "3": "serious defect",\n        }\n        return mapping.get(label)\n    if adapter == "room9":\n        label = normalize(class_name(dataset, row))\n        mapping = {\n            "bathroom": "bathroom", "bedroom": "bedroom", "closet": "closets",\n            "diningroom": "dining_room", "kitchen": "kitchen", "livingroom": "living_room",\n            "insidebus": "Interior", "insidesubway": "Interior",\n        }\n        return mapping.get(label, "others")\n    if adapter == "room5":\n        label = normalize(class_name(dataset, row))\n        return {"bathroom": "bathroom", "bedroom": "bedroom", "kitchen": "kitchen", "livingroom": "living room"}.get(label)\n    if adapter == "nsfw":\n        label = normalize(class_name(dataset, row))\n        return "safe" if label in {"safe", "sfw", "normal", "neutral", "general", "sensitive", "drawing", "0", "l1", "l2"} else "no_safe"\n    if adapter == "chest_ct":\n        label = normalize(class_name(dataset, row))\n        for candidate in ("adenocarcinoma", "largecellcarcinoma", "squamouscellcarcinoma", "normal"):\n            if candidate in label:\n                return candidate\n        return None\n    if adapter == "brain_tumor_binary":\n        label = normalize(class_name(dataset, row))\n        return "1" if label in {"1", "yes", "tumor", "positive", "glioma", "meningioma", "pituitary"} else "0"\n    if adapter == "body_type":\n        value = first_value(row, ("body_type", "label", "class"))\n        if value is None:\n            prompt = str(first_value(row, ("prompt", "text", "caption"), ""))\n            groups = (\n                ("hulk", r"\\b(?:heavily\\s+muscular|muscular|athletic|toned|fit)\\s+build\\b"),\n                ("overweight", r"\\b(?:plus-size|heavy|large-framed|broad-shouldered|broad|curvy|full-figured|stocky)\\s+build\\b"),\n                ("skinny", r"\\b(?:very\\s+slender|thin|lean|petite|slender|small-framed|compact)\\s+build\\b"),\n                ("ordinary", r"\\b(?:average|medium)\\s+build\\b"),\n            )\n            value = next((target for target, pattern in groups if re.search(pattern, prompt, re.IGNORECASE)), None)\n        label = normalize(value)\n        return {\n            "thin": "skinny", "skinny": "skinny",\n            "normal": "ordinary", "average": "ordinary", "ordinary": "ordinary",\n            "obese": "overweight", "fat": "overweight", "overweight": "overweight",\n            "muscular": "hulk", "hulk": "hulk",\n        }.get(label)\n    if adapter == "animal_species":\n        label = normalize(class_name(dataset, row))\n        taxonomy = {\n            "canislatrans": "coyote", "coyote": "coyote", "cervuselaphus": "deer",\n            "capreoluscapreolus": "deer", "deer": "deer", "canislupus": "wolf",\n            "wolf": "wolf", "vulpesvulpes": "fox", "fox": "fox", "ursus": "bear",\n            "bear": "bear", "melesmeles": "badger", "badger": "badger",\n            "feliscatus": "feline", "cat": "feline", "equus": "horse", "horse": "horse",\n            "ovis": "sheep", "sheep": "sheep", "capra": "goat", "goat": "goat",\n            "lepus": "rabbit", "rabbit": "rabbit", "hare": "rabbit",\n            "bird": "bird", "dolphin": "dolphin", "giraffe": "giraffe",\n            "kangaroo": "kangaroo", "lion": "lion", "snake": "snake",\n            "tiger": "tiger", "cow": "cattle",\n        }\n        for token, target in taxonomy.items():\n            if token in label:\n                return target\n        return None\n    if adapter == "trapper_species":\n        label = normalize(class_name(dataset, row))\n        mapping = {\n            "canislupus": "gray wolf", "vulpesvulpes": "red fox", "melesmeles": "european badger",\n            "capreoluscapreolus": "roe deer", "cervuselaphus": "red deer", "dama": "fallow deer",\n            "alces": "moose", "rangifer": "reindeer", "sus": "wild boar", "lynx": "eurasian lynx",\n            "sciurus": "eurasian red squirrel", "lepus": "hare", "martes": "marten",\n            "rupicapra": "chamois", "ovis": "european mouflon", "cat": "cat", "dog": "dog", "bird": "bird",\n        }\n        for token, target in mapping.items():\n            if token in label:\n                return target\n        return None\n    if adapter == "dog_breed":\n        label = class_name(dataset, row)\n        label = re.sub(r"^n\\d+-", "", label, flags=re.IGNORECASE)\n        return label.replace("_", " ")\n    if adapter == "aesthetic_tier":\n        value = first_value(row, ("aesthetic_tier", "tier", "label", "score"))\n        if isinstance(value, str) and not value.isdigit():\n            token = normalize(value)\n            return {"c": "C_tier", "b": "B_tier", "a": "A_tier", "s": "S_tier", "ss": "SS_tier"}.get(token)\n        return {0: "C_tier", 1: "B_tier", 2: "A_tier", 3: "S_tier", 4: "SS_tier"}.get(int(value))\n    if adapter == "mri_bodypart":\n        modality = normalize(first_value(row, ("modality", "image_modality"), ""))\n        if modality and "mri" not in modality and "magneticresonance" not in modality:\n            return None\n        body = normalize(first_value(row, ("body_part", "bodypart", "anatomy", "organ"), ""))\n        if "brain" in body or "head" in body: return "Brain"\n        if "cervical" in body or "neck" in body: return "Cervical Spine"\n        if "spine" in body: return "Whole Spine"\n        if "pelv" in body: return "Pelvic"\n        if "shoulder" in body: return "Right Shoulder"\n        return None\n    raise ValueError(f"Unsupported label adapter: {adapter}")\n\n\ndef batches(items: Iterable, size: int):\n    batch = []\n    for item in items:\n        batch.append(item)\n        if len(batch) >= size:\n            yield batch\n            batch = []\n    if batch:\n        yield batch\n\n\ndef load_saved_dataset(path: Path):\n    from datasets import DatasetDict, load_from_disk\n    data = load_from_disk(str(path))\n    if isinstance(data, DatasetDict):\n        for split in ("test", "validation", "valid", "train"):\n            if split in data:\n                return data[split]\n        return data[next(iter(data))]\n    return data\n\n\ndef standard_hf_backend(entry: dict, model_dir: Path, device: torch.device):\n    os.environ.setdefault("USE_TF", "0")\n    from transformers import AutoImageProcessor, AutoModelForImageClassification, SiglipImageProcessor\n    model = AutoModelForImageClassification.from_pretrained(\n        model_dir, local_files_only=True, trust_remote_code=True\n    ).to(device).eval()\n    if entry["model"].startswith("kawaimasa__"):\n        from torchvision.transforms import Compose, Normalize, ToTensor\n        size = 768\n        tensorize = Compose([ToTensor(), Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])\n        def preprocess(images):\n            output = []\n            for image in images:\n                image = image.convert("RGB")\n                width, height = image.size\n                scale = size / max(width, height)\n                resized = image.resize((max(1, int(width * scale)), max(1, int(height * scale))), Image.Resampling.LANCZOS)\n                canvas = Image.new("RGB", (size, size), (0, 0, 0))\n                canvas.paste(resized, ((size - resized.width) // 2, (size - resized.height) // 2))\n                output.append(tensorize(canvas))\n            return {"pixel_values": torch.stack(output).to(device)}\n    else:\n        try:\n            processor = AutoImageProcessor.from_pretrained(model_dir, local_files_only=True, trust_remote_code=True)\n        except OSError:\n            size = int(getattr(getattr(model.config, "vision_config", None), "image_size", 224))\n            processor = SiglipImageProcessor(size={"height": size, "width": size})\n        def preprocess(images):\n            return {key: value.to(device) for key, value in processor(images=images, return_tensors="pt").items()}\n    return model, preprocess, model_labels(model)\n\n\ndef tf_backend(model_dir: Path):\n    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")\n    os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")\n    import tensorflow as tf\n    from transformers import AutoImageProcessor, TFAutoModelForImageClassification\n    processor = AutoImageProcessor.from_pretrained(model_dir, local_files_only=True)\n    model = TFAutoModelForImageClassification.from_pretrained(model_dir, local_files_only=True)\n    labels = {int(key): str(value) for key, value in model.config.id2label.items()}\n    def predict(images):\n        values = processor(images=images, return_tensors="np")["pixel_values"]\n        return np.asarray(model(pixel_values=tf.convert_to_tensor(values), training=False).logits)\n    return predict, labels\n\n\ndef dataset_examples(entry: dict, dataset, data_root: Path, labels: dict[int, str], maximum: int):\n    count = min(len(dataset), maximum) if maximum else len(dataset)\n    for index in range(count):\n        row = dataset[index]\n        try:\n            target_name = truth_name(entry, dataset, row)\n            target = match_label(target_name, labels) if target_name is not None else None\n            if target is None:\n                continue\n            yield index, image_from_row(row, data_root), target, target_name\n        except Exception as exc:\n            print(f"Skipping row {index}: {exc}", file=sys.stderr)\n\n\ndef evaluate_classification(entry: dict, model_dir: Path, dataset, data_root: Path, device: torch.device, batch_size: int, maximum: int, prediction_limit: int = 0):\n    backend = entry["backend"]\n    if backend in {"hf", "imatag", "scalar"}:\n        model, preprocess, labels = standard_hf_backend(entry, model_dir, device)\n    elif backend == "tf_hf":\n        predict_tf, labels = tf_backend(model_dir)\n        model = preprocess = None\n    elif backend == "itsnotai":\n        os.environ.setdefault("USE_TF", "0")\n        from transformers import AutoImageProcessor, AutoModelForImageClassification\n        model = AutoModelForImageClassification.from_pretrained(model_dir, local_files_only=True).to(device).eval()\n        processor = AutoImageProcessor.from_pretrained(model_dir, local_files_only=True)\n        meta = json.loads((model_dir / "source_meta.json").read_text())\n        head = torch.nn.Sequential(torch.nn.Dropout(0.1), torch.nn.Linear(int(meta["hidden_size"]), 2))\n        head.load_state_dict(torch.load(model_dir / "binary_head.pt", map_location="cpu", weights_only=True))\n        head.to(device).eval()\n        labels = {0: "REAL", 1: "FAKE"}\n        def preprocess(images): return {key: value.to(device) for key, value in processor(images=images, return_tensors="pt").items()}\n    elif backend == "infobay":\n        from torchvision.models import resnet18\n        from torchvision.transforms import Compose, Grayscale, Normalize, Resize, ToTensor\n        model = resnet18(weights=None)\n        model.conv1 = torch.nn.Conv2d(1, 64, 7, 2, 3, bias=False)\n        model.fc = torch.nn.Linear(512, 5)\n        model.load_state_dict(torch.load(model_dir / "pytorch_model.bin", map_location="cpu", weights_only=True))\n        model.to(device).eval()\n        names = json.loads((model_dir / "classes.json").read_text())\n        labels = dict(enumerate(names))\n        transform = Compose([Grayscale(1), Resize((224, 224)), ToTensor(), Normalize([0.5], [0.5])])\n        def preprocess(images): return {"pixel_values": torch.stack([transform(image) for image in images]).to(device)}\n    elif backend.startswith("yolo_"):\n        from ultralytics import YOLO\n        weights = next(model_dir.glob("*.pt"))\n        model = YOLO(str(weights))\n        labels = {int(key): str(value) for key, value in model.names.items()}\n        preprocess = None\n    else:\n        raise ValueError(f"Unsupported classification backend: {backend}")\n\n    truth: list[int] = []\n    predicted: list[int] = []\n    rows: list[dict] = []\n    if backend == "scalar":\n        scores, targets = [], []\n        raw_examples = dataset_examples({**entry, "adapter": "aesthetic_tier"}, dataset, data_root, {0:"C_tier",1:"B_tier",2:"A_tier",3:"S_tier",4:"SS_tier"}, maximum)\n        for batch in tqdm(batches(raw_examples, batch_size), total=math.ceil((maximum or len(dataset))/batch_size), desc=entry["model"], unit="batch"):\n            images = [item[1] for item in batch]\n            with torch.inference_mode():\n                output = model(**preprocess(images)).logits.reshape(-1).float().cpu().numpy()\n            for item, score in zip(batch, output):\n                tier_score = {0:0.0, 1:1.0, 2:2.0, 3:3.0, 4:4.0}[item[2]]\n                scores.append(float(score)); targets.append(tier_score)\n                rows.append({"index":item[0], "true_score":tier_score, "pred_score":float(score)})\n            if prediction_limit and len(rows) >= prediction_limit:\n                rows = rows[:prediction_limit]; scores = scores[:prediction_limit]; targets = targets[:prediction_limit]\n                break\n        if not scores: raise RuntimeError("No compatible labelled examples were found")\n        from scipy.stats import spearmanr\n        correlation = float(spearmanr(targets, scores).statistic)\n        return {"evaluated_predictions":len(scores), "mae":float(mean_absolute_error(targets,scores)), "spearman":correlation}, rows, None, None, None\n\n    examples = dataset_examples(entry, dataset, data_root, labels, maximum)\n    total = math.ceil((maximum or len(dataset)) / batch_size)\n    for batch in tqdm(batches(examples, batch_size), total=total, desc=entry["model"], unit="batch"):\n        images = [item[1] for item in batch]\n        if backend == "tf_hf":\n            logits = predict_tf(images)\n            probabilities = torch.from_numpy(np.array(logits, copy=True)).softmax(-1)\n            confidence, output_ids = probabilities.max(-1)\n        elif backend == "yolo_classify":\n            outputs = model.predict(images, device=str(device), verbose=False)\n            output_ids = torch.tensor([int(item.probs.top1) for item in outputs])\n            confidence = torch.tensor([float(item.probs.top1conf) for item in outputs])\n        elif backend == "yolo_detect":\n            outputs = model.predict(images, imgsz=1024, device=str(device), verbose=False)\n            pred, conf = [], []\n            for output in outputs:\n                if output.boxes is None or len(output.boxes) == 0:\n                    pred.append(-1); conf.append(0.0)\n                else:\n                    best = int(output.boxes.conf.argmax().item())\n                    pred.append(int(output.boxes.cls[best].item())); conf.append(float(output.boxes.conf[best].item()))\n            output_ids, confidence = torch.tensor(pred), torch.tensor(conf)\n        else:\n            inputs = preprocess(images)\n            with torch.inference_mode():\n                if backend == "itsnotai":\n                    if hasattr(model, "beit"):\n                        features = model.beit(inputs["pixel_values"]).last_hidden_state[:, 0]\n                    elif hasattr(model, "vit"):\n                        features = model.vit(inputs["pixel_values"]).last_hidden_state[:, 0]\n                    else:\n                        raise RuntimeError("ItsNotAI backbone exposes neither beit nor vit")\n                    logits = head(features)\n                elif backend == "infobay":\n                    logits = model(inputs["pixel_values"])\n                else:\n                    logits = model(**inputs).logits\n            if backend == "imatag":\n                # The author-defined decision rule uses the sign of logit 0,\n                # not argmax over the two configured labels.\n                raw_score = logits[:, 0].float().cpu()\n                output_ids = (raw_score < 0).long()\n                confidence = torch.sigmoid(raw_score.abs())\n            else:\n                probabilities = logits.float().softmax(-1).cpu()\n                confidence, output_ids = probabilities.max(-1)\n        for item, pred, conf in zip(batch, output_ids.tolist(), confidence.tolist()):\n            truth.append(item[2]); predicted.append(int(pred))\n            rows.append({"index":item[0], "true_id":item[2], "true_label":item[3], "pred_id":int(pred), "pred_label":labels.get(int(pred), "NO_DETECTION"), "confidence":float(conf)})\n        if prediction_limit and len(truth) >= prediction_limit:\n            truth = truth[:prediction_limit]; predicted = predicted[:prediction_limit]; rows = rows[:prediction_limit]\n            break\n    if not truth:\n        raise RuntimeError("No compatible labelled examples were found in the selected dataset")\n    ids = sorted(set(truth) | set(predicted))\n    macro = precision_recall_fscore_support(truth, predicted, average="macro", zero_division=0)\n    weighted = precision_recall_fscore_support(truth, predicted, average="weighted", zero_division=0)\n    summary = {\n        "evaluated_predictions":len(truth), "accuracy":float(accuracy_score(truth,predicted)),\n        "macro_precision":float(macro[0]), "macro_recall":float(macro[1]), "macro_f1":float(macro[2]),\n        "weighted_precision":float(weighted[0]), "weighted_recall":float(weighted[1]), "weighted_f1":float(weighted[2]),\n    }\n    report = classification_report(truth, predicted, labels=ids, target_names=[labels.get(i,str(i)) for i in ids], output_dict=True, zero_division=0)\n    return summary, rows, report, confusion_matrix(truth,predicted,labels=ids), ids\n\n\ndef evaluate_onnx(entry: dict, model_dir: Path, dataset, data_root: Path, batch_size: int, maximum: int, prediction_limit: int = 0):\n    os.environ.setdefault("USE_TF", "0")\n    import onnxruntime as ort\n    from transformers import AutoImageProcessor\n    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]\n    session = ort.InferenceSession(str(model_dir / "onnx" / "model.onnx"), providers=providers)\n    processor = AutoImageProcessor.from_pretrained(model_dir, local_files_only=True)\n    count = min(len(dataset), maximum) if maximum else len(dataset)\n    rows, ages, age_pred, genders, gender_pred = [], [], [], [], []\n    def prepared_rows():\n        for index in range(count):\n            row = dataset[index]\n            try:\n                yield index, image_from_row(row,data_root), float(first_value(row,("age",))), raw_gender(row)\n            except Exception as exc:\n                print(f"Skipping row {index}: {exc}", file=sys.stderr)\n    for batch in tqdm(batches(prepared_rows(),batch_size), total=math.ceil(count/batch_size), desc=entry["model"], unit="batch"):\n        values = processor(images=[item[1] for item in batch], return_tensors="np")["pixel_values"]\n        logits = session.run(None, {session.get_inputs()[0].name: values})[0]\n        for item, output in zip(batch, logits):\n            pa = float(np.clip(np.rint(output[0]),0,100)); pg = "female" if float(output[1]) >= .5 else "male"\n            ages.append(item[2]); age_pred.append(pa); genders.append(item[3]); gender_pred.append(pg)\n            rows.append({"index":item[0],"true_age":item[2],"pred_age":pa,"true_gender":item[3],"pred_gender":pg,"gender_score":float(output[1])})\n        if prediction_limit and len(rows) >= prediction_limit:\n            rows = rows[:prediction_limit]; ages = ages[:prediction_limit]; age_pred = age_pred[:prediction_limit]\n            genders = genders[:prediction_limit]; gender_pred = gender_pred[:prediction_limit]\n            break\n    return {"evaluated_predictions":len(rows),"age_mae":float(mean_absolute_error(ages,age_pred)),"gender_accuracy":float(accuracy_score(genders,gender_pred)),"providers":session.get_providers()}, rows, None, None, None\n\n\ndef evaluate_mammoscreen(entry: dict, model_dir: Path, data_root: Path, device: torch.device, maximum: int, prediction_limit: int = 0):\n    os.environ.setdefault("USE_TF", "0")\n    from transformers import AutoModel\n    csv_files = list(data_root.rglob("*.csv"))\n    if not csv_files: raise FileNotFoundError(f"No label CSV found under {data_root}")\n    frame = pd.concat([pd.read_csv(path) for path in csv_files], ignore_index=True)\n    cancer_col = next((c for c in frame if normalize(c) in {"cancer","label","target","malignant"}), None)\n    image_col = next((c for c in frame if normalize(c) in {"imageid","image","filepath","path","filename"}), None)\n    if cancer_col is None: raise ValueError(f"No cancer label in columns {list(frame)}")\n    if prediction_limit:\n        frame = frame.groupby(cancer_col, group_keys=False).head(max(1, prediction_limit // 2))\n    files = [path for path in data_root.rglob("*") if path.suffix.lower() in {".png",".jpg",".jpeg",".dcm"}]\n    by_stem = {path.stem:path for path in files}\n    sys.path.insert(0,str(model_dir))\n    model = AutoModel.from_pretrained(model_dir,local_files_only=True,trust_remote_code=True).to(device).eval()\n    crop_dir = model_dir / "test_code" / "upstream" / "mammo-crop"\n    if not (crop_dir / "config.json").is_file():\n        raise FileNotFoundError("Missing ianpan/mammo-crop; rerun the dataset download script")\n    crop = AutoModel.from_pretrained(crop_dir,local_files_only=True,trust_remote_code=True).to(device).eval()\n    truth,scores,rows=[],[],[]\n    for index,row in tqdm(frame.iterrows(),total=min(len(frame),maximum) if maximum else len(frame),desc=entry["model"],unit="image"):\n        if maximum and len(rows)>=maximum: break\n        key = str(row[image_col]) if image_col else ""\n        path = Path(key)\n        if not path.is_file(): path = by_stem.get(path.stem or key, Path())\n        if not path.is_file(): continue\n        try:\n            if path.suffix.lower()==".dcm": image=model.load_image_from_dicom(str(path))\n            else: image=np.asarray(Image.open(path).convert("L"))\n            shape=torch.tensor([image.shape[:2]],device=device)\n            crop_input=torch.from_numpy(crop.preprocess(image)).expand(1,1,-1,-1).float().to(device)\n            with torch.inference_mode():\n                x,y,w,h=crop(crop_input,shape)[0].detach().cpu().numpy().astype(int)\n            image=image[y:y+h,x:x+w]\n            with torch.inference_mode(): score=float(model([{"view":image}],device=str(device))["cancer"].item())\n            label=int(row[cancer_col]); truth.append(label); scores.append(score); rows.append({"path":str(path),"cancer":label,"score":score})\n        except Exception as exc: print(f"Skipping {path}: {exc}",file=sys.stderr)\n    if len(set(truth))<2: raise RuntimeError("Mammography subset must contain both cancer classes for AUROC")\n    return {"evaluated_predictions":len(rows),"auroc":float(roc_auc_score(truth,scores))},rows,None,None,None\n\n\ndef write_results(result_dir: Path, summary: dict, rows: list[dict], report, matrix, matrix_labels) -> None:\n    result_dir.mkdir(parents=True, exist_ok=True)\n    pd.DataFrame(rows).to_csv(result_dir/"predictions.tsv",sep="\\t",index=False)\n    if report is not None:\n        (result_dir/"classification_report.txt").write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\\n",encoding="utf-8")\n    if matrix is not None:\n        pd.DataFrame(matrix,index=matrix_labels,columns=matrix_labels).to_csv(result_dir/"confusion_matrix.txt",sep="\\t")\n    with pd.ExcelWriter(result_dir/"metrics.xlsx",engine="openpyxl") as writer:\n        pd.DataFrame(list(summary.items()),columns=["metric","value"]).to_excel(writer,index=False,sheet_name="summary")\n        pd.DataFrame(rows).to_excel(writer,index=False,sheet_name="predictions")\n        if report is not None:\n            pd.DataFrame(report).transpose().to_excel(writer, sheet_name="classification_report")\n        if matrix is not None:\n            pd.DataFrame(matrix,index=matrix_labels,columns=matrix_labels).to_excel(writer,sheet_name="confusion_matrix")\n    temporary = result_dir / "summary.txt.tmp"\n    temporary.write_text(json.dumps(summary,ensure_ascii=False,indent=2)+"\\n",encoding="utf-8")\n    temporary.replace(result_dir / "summary.txt")\n\n\ndef main() -> None:\n    args=arguments()\n    plan=json.loads(args.plan.read_text(encoding="utf-8"))["models"]\n    entry=next((item for item in plan if item["model"]==args.model_key),None)\n    if entry is None: raise ValueError(f"Model not present in plan: {args.model_key}")\n    if entry.get("existing_result"):\n        print(f"Existing result retained: {ROOT/entry[\'model\']/entry[\'existing_result\']}"); return\n    model_dir=ROOT/entry["model"]\n    data_root=model_dir/"test_data"/"benchmark"\n    source_path=model_dir/"test_data"/"source.json"\n    if not data_root.exists() or not source_path.is_file():\n        raise FileNotFoundError(f"Dataset is not prepared. Run download_all_non_strict_classification_data.sh first: {data_root}")\n    source=json.loads(source_path.read_text(encoding="utf-8"))\n    result_dir=args.result_dir.resolve() if args.result_dir else model_dir/"result"\n    device=torch.device(args.device if args.device!="cuda" or torch.cuda.is_available() else "cpu")\n    batch_size=args.batch_size or int(entry.get("batch_size",32))\n    started=time.time()\n    if entry["backend"]=="mammoscreen":\n        output=evaluate_mammoscreen(entry,model_dir,data_root,device,args.max_samples,args.max_predictions)\n    else:\n        dataset=load_saved_dataset(data_root)\n        if entry["backend"]=="onnx_age_gender": output=evaluate_onnx(entry,model_dir,dataset,data_root,batch_size,args.max_samples,args.max_predictions)\n        else: output=evaluate_classification(entry,model_dir,dataset,data_root,device,batch_size,args.max_samples,args.max_predictions)\n    metrics,rows,report,matrix,matrix_labels=output\n    if entry["adapter"] == "all_unwatermarked" and "accuracy" in metrics:\n        metrics["specificity"] = metrics["accuracy"]\n        metrics["false_positive_rate"] = 1.0 - metrics["accuracy"]\n    summary={\n        "model":entry["model"], "dataset":source.get("resolved_dataset_id",entry["dataset_id"]),\n        "dataset_revision":source.get("revision",""), "dataset_split":source.get("split",entry.get("split","")),\n        "downloaded_dataset_rows":source.get("rows","archive"),\n        "source_level":source.get("source_level",entry["source_level"]), "official_preferred":True,\n        "backend":entry["backend"], "label_adapter":entry["adapter"], "batch_size":batch_size,\n        "device":str(device), "elapsed_seconds":round(time.time()-started,3),\n        "source_note":entry.get("source_note",""), **metrics,\n    }\n    for name, value in entry.get("reference_metrics", {}).items():\n        summary[f"published_reference_{name}"] = value\n    # Every entry in this suite lacks at least one strict-reproduction element\n    # (exact split and/or author evaluator), so published values are references,\n    # never silently treated as an apples-to-apples comparison.\n    summary["published_reference_directly_comparable"] = False\n    write_results(result_dir,summary,rows,report,matrix,matrix_labels)\n    print(json.dumps(summary,ensure_ascii=False,indent=2))\n\n\nif __name__=="__main__":\n    main()\n', '_shared_test_resources/code/run_timm_official.sh': '#!/usr/bin/env bash\nset -euo pipefail\nMODEL_DIR="$1"; MODEL_NAME="$2"; REFERENCE="$3"; REF_TOP1="$4"; REF_TOP5="$5"; shift 5\nROOT="$(cd "$(dirname "$0")/../.." && pwd)"\nVALIDATE="$ROOT/_shared_test_resources/official_sources/timm/validate.py"\nRESULT="$MODEL_DIR/result"; mkdir -p "$RESULT"\nprintf \'%s\\n\' "$REFERENCE" > "$RESULT/official_reference.txt"\nif [ ! -f "$VALIDATE" ]; then\n  echo "Missing official timm source. Run: bash image-classification/download_official_sources.sh" >&2; exit 1\nfi\nif [ ! -f "$MODEL_DIR/model.safetensors" ]; then\n  echo "Missing local open weight: $MODEL_DIR/model.safetensors" >&2; exit 1\nfi\npython3 "$VALIDATE" "$MODEL_DIR/test_data/imagenet-1k/val" --model "$MODEL_NAME" \\\n  --checkpoint "$MODEL_DIR/model.safetensors" --results-file "$RESULT/official_validate.csv" "$@" \\\n  2>&1 | tee "$RESULT/official_validate.log"\npython3 "$ROOT/_shared_test_resources/code/finalize_imagenet_result.py" \\\n  --result-dir "$RESULT" --model "$MODEL_NAME" \\\n  --reference-top1 "$REF_TOP1" --reference-top5 "$REF_TOP5" \\\n  --csv "$RESULT/official_validate.csv" --log "$RESULT/official_validate.log"\n', 'prithivMLmods__AI-vs-Deepfake-vs-Real-v2.0/test_code/run_test.sh': '#!/usr/bin/env bash\nset -euo pipefail\nHERE="$(cd "$(dirname "$0")" && pwd)"\nMODEL_DIR="$(cd "$HERE/.." && pwd)"\nROOT="$(cd "$MODEL_DIR/.." && pwd)"\nexec python3 "$ROOT/_shared_test_resources/code/remaining_classification_evaluator.py" \\\n  --model-key "$(basename "$MODEL_DIR")" "$@"\n'}


def _virtual_path(root, name):
    if not isinstance(name, str) or not name or "\\" in name:
        raise ValueError(f"Invalid bundle member: {name!r}")
    relative = PurePosixPath(name)
    if relative.is_absolute() or any(part in ("", ".", "..") for part in name.split("/")):
        raise ValueError(f"Unsafe bundle member: {name!r}")
    return root.joinpath(*relative.parts)


"""Embedded by each inference.py; no dependency on the maintenance directory."""
import functools as _ft_functools
import json as _ft_json
import math as _ft_math
import os as _ft_os
import sys as _ft_sys
import tempfile as _ft_tempfile
import time as _ft_time
from contextlib import contextmanager as _ft_contextmanager
from pathlib import Path as _FTPath


class ForwardTiming:
    """Synchronized forward calls only; nested calls are counted once."""

    def __init__(self, device="cpu", sync=None, clock=None):
        self.device = str(device)
        self.seconds = 0.0
        self.depth = 0
        self.active = True
        self._sync = sync
        self._clock = clock or _ft_time.perf_counter
        self._wrapped = []

    def synchronize(self):
        if self._sync is not None:
            self._sync()
        elif self.device.startswith("cuda"):
            torch = _ft_sys.modules.get("torch")
            if torch is not None and torch.cuda.is_available():
                torch.cuda.synchronize(self.device)

    @_ft_contextmanager
    def forward(self):
        if not self.active or self.depth:
            yield
            return
        self.synchronize()
        start = self._clock()
        self.depth += 1
        try:
            yield
            self.synchronize()
        except BaseException:
            # Failed/OOM attempts do not become successful-sample latency.
            raise
        else:
            self.seconds += self._clock() - start
        finally:
            self.depth -= 1

    def call(self, function, *args, **kwargs):
        with self.forward():
            return function(*args, **kwargs)

    def watch(self, owner, name="forward"):
        original = getattr(owner, name)
        @_ft_functools.wraps(original)
        def measured(*args, **kwargs):
            return self.call(original, *args, **kwargs)
        self._wrapped.append((owner, name, original))
        setattr(owner, name, measured)

    def close(self):
        for owner, name, original in reversed(self._wrapped):
            setattr(owner, name, original)
        self._wrapped.clear()


def _ft_validate(seconds, count):
    if not isinstance(count, int) or isinstance(count, bool) or count < 1:
        raise ValueError("Timing requires a positive integer sample count")
    seconds = float(seconds)
    if not _ft_math.isfinite(seconds) or seconds < 0:
        raise ValueError("Invalid forward latency")
    return {"total_latency_s": seconds, "sample_count": count,
            "avg_latency_ms": seconds * 1000.0 / count}


def _ft_atomic(path, value):
    path = _FTPath(path)
    fd, temporary = _ft_tempfile.mkstemp(prefix="." + path.name + ".", dir=path.parent)
    try:
        with _ft_os.fdopen(fd, "w", encoding="utf-8") as stream:
            _ft_json.dump(value, stream, ensure_ascii=False, allow_nan=False, indent=2)
            stream.write("\n")
            stream.flush()
            _ft_os.fsync(stream.fileno())
        _ft_os.replace(temporary, path)
    finally:
        if _ft_os.path.exists(temporary):
            _ft_os.unlink(temporary)


def save_forward_timing(model_dir, seconds, sample_count):
    """Merge one model atomically. Lock the existing Model directory: no lock file."""
    import fcntl
    model_dir = _FTPath(model_dir).resolve()
    value = _ft_validate(seconds, sample_count)
    target = PATHS.RESULT_DIR / "inference_time.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor = _ft_os.open(PATHS.RESULT_DIR, _ft_os.O_RDONLY)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        contents = _ft_json.loads(target.read_text(encoding="utf-8")) if target.exists() else {}
        if not isinstance(contents, dict):
            raise ValueError("inference_time.json must be a model-keyed object")
        contents[model_dir.name] = value
        _ft_atomic(target, contents)
        run_id = _ft_os.environ.get('CLASSIFICATION_RUN_ID')
        if run_id and _FTPath(run_id).name == run_id and run_id not in ('.', '..'):
            run_dir = PATHS.RESULT_DIR/'runs'/run_id
            run_dir.mkdir(parents=True, exist_ok=True)
            _ft_atomic(run_dir/'inference_time.json', {model_dir.name:value})
    finally:
        _ft_os.close(descriptor)
    print(f"Forward timing saved: {target} [{model_dir.name}]", flush=True)
    return value


class ForwardJournal:
    """Per-completed-unit times stay with the external resumable work files."""

    def __init__(self, path):
        self.path = _FTPath(path)
        self.units = _ft_json.loads(self.path.read_text()) if self.path.exists() else {}

    def put(self, key, seconds, count):
        self.units[str(key)] = _ft_validate(seconds, int(count))
        _ft_atomic(self.path, self.units)

    def publish(self, model_dir, *, expected=None):
        count = sum(x["sample_count"] for x in self.units.values())
        if count == 0 or (expected is not None and count != expected):
            print("Forward timing not published: cached predictions have no complete matching timing record.", flush=True)
            return None
        return save_forward_timing(model_dir, sum(x["total_latency_s"] for x in self.units.values()), count)


class RuntimeBundle:
    """Safely materialize a portable checkpoint for one inference lifetime."""

    def __init__(self, checkpoint=None):
        self.checkpoint = PATHS.project_path(checkpoint) if checkpoint is not None else PATHS.WEIGHTS_PATH
        self.spec = dict(MODEL_SPEC)
        self.metadata = self.spec
        self.root = self.model_dir = None
        self._temporary = None
        self._environment = {}
        self._saved_modules = {}
        self._module_attributes = []
        self._paths = []
        self._loaded_modules = set()
        self.files = ()
        self.sha256 = {}
        self.model_key = self.spec.get("model_key")

    def __enter__(self):
        if self._temporary is not None:
            return self
        self._temporary = tempfile.TemporaryDirectory(prefix="classification-runtime-")
        self.root = Path(self._temporary.name)
        try:
            # Configure caches before importing any framework.
            cache = self.root / "_runtime_cache"
            environment = {
                "PYTHONDONTWRITEBYTECODE": "1", "HF_HUB_OFFLINE": "1",
                "TRANSFORMERS_OFFLINE": "1", "HF_DATASETS_OFFLINE": "1",
                "HF_HOME": str(cache / "huggingface"),
                "HF_HUB_CACHE": str(cache / "huggingface" / "hub"),
                "HF_MODULES_CACHE": str(cache / "huggingface" / "modules"),
                "HF_DATASETS_CACHE": str(cache / "datasets"),
                "TRANSFORMERS_CACHE": str(cache / "huggingface" / "hub"),
                "TORCH_HOME": str(cache / "torch"),
                "XDG_CACHE_HOME": str(cache), "MPLCONFIGDIR": str(cache / "matplotlib"),
                "YOLO_CONFIG_DIR": str(cache / "ultralytics"),
                "NUMBA_CACHE_DIR": str(cache / "numba"),
                "TRITON_CACHE_DIR": str(cache / "triton"),
                "TF_USE_LEGACY_KERAS": "1", "TF_CPP_MIN_LOG_LEVEL": "2",
            }
            for name, value in environment.items():
                self._environment[name] = os.environ.get(name)
                os.environ[name] = value
            for value in environment.values():
                if value.startswith(str(cache)):
                    Path(value).mkdir(parents=True, exist_ok=True)
            self._refresh_framework_caches()
            import torch
            archive = torch.load(self.checkpoint, map_location="cpu", weights_only=True, mmap=True)
            if not isinstance(archive, dict) or archive.get("format") != "classification-bundle-v1":
                raise ValueError("Expected a classification-bundle-v1 checkpoint")
            key = archive.get("model_key")
            if not isinstance(key, str) or len(PurePosixPath(key).parts) != 1:
                raise ValueError("Invalid checkpoint model_key")
            if self.model_key and self.model_key != key:
                raise ValueError(f"Checkpoint belongs to {key}, expected {self.model_key}")
            metadata = archive.get("metadata", {})
            files = archive.get("files")
            hashes = archive.get("sha256")
            if not isinstance(metadata, dict) or not isinstance(files, dict) or not isinstance(hashes, dict):
                raise ValueError("Checkpoint requires metadata, files, and sha256 dictionaries")
            self.spec = {**metadata, **self.spec, "model_key": key}
            self.metadata = self.spec
            self.model_key = key
            self.model_dir = _virtual_path(self.root, key)
            if not isinstance(self.spec.get("backend"), str):
                raise ValueError("Missing model backend")
            if set(files) != set(hashes):
                raise ValueError("The asset hash manifest must cover every checkpoint file")
            self.files = tuple(files)
            self.sha256 = dict(hashes)
            for name, payload in files.items():
                path = _virtual_path(self.root, name)
                path.parent.mkdir(parents=True, exist_ok=True)
                if isinstance(payload, bytes):
                    path.write_bytes(payload)
                elif isinstance(payload, torch.Tensor) and payload.dtype == torch.uint8 and payload.ndim == 1:
                    payload.numpy().tofile(path)
                elif isinstance(payload, (list, tuple)):
                    # Large native files are stored as independently allocated
                    # uint8 tensor chunks, so no ZIP record needs a multi-GB
                    # write. Preserve the original byte sequence and SHA256.
                    with path.open("wb") as handle:
                        for index, chunk in enumerate(payload):
                            if not isinstance(chunk, torch.Tensor) or chunk.dtype != torch.uint8 or chunk.ndim != 1:
                                raise ValueError(f"Invalid uint8 chunk {index} in bundle asset: {name}")
                            if not chunk.is_contiguous():
                                raise ValueError(f"Noncontiguous chunk {index} in bundle asset: {name}")
                            chunk.numpy().tofile(handle)
                else:
                    raise ValueError(f"Bundle asset is not raw bytes, uint8 storage, or uint8 chunks: {name}")
            del archive, files
            for name, source in EMBEDDED_SOURCES.items():
                path = _virtual_path(self.root, name)
                if path.exists():
                    raise ValueError(f"Source collides with checkpoint asset: {name}")
                if not isinstance(source, str):
                    raise ValueError(f"Embedded source is not text: {name}")
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(source, encoding="utf-8")
            self.dataset_paths = {}
            for virtual, relative in self.spec.get("dataset_links", {}).items():
                if not isinstance(relative, str) or not relative.startswith("../../Dataset/"):
                    raise ValueError(f"Unexpected packaged dataset reference: {relative!r}")
                tail = relative[len("../../Dataset/"):]
                target = _virtual_path(PATHS.DATASET_ROOT, tail).resolve(strict=True)
                if not target.is_relative_to(PATHS.DATASET_ROOT):
                    raise ValueError("Dataset reference escapes shared dataset")
                self.dataset_paths[virtual] = target
            if not self.model_dir.is_dir():
                raise ValueError("Checkpoint does not contain its model directory")
            return self
        except BaseException:
            self.close()
            raise

    def _refresh_framework_caches(self):
        # Hugging Face caches constants on first import. Refresh existing modules
        # too, so a second predictor cannot reuse the first predictor's sources.
        updates = {
            "huggingface_hub.constants": {"HF_HOME": os.environ["HF_HOME"], "HF_HUB_CACHE": os.environ["HF_HUB_CACHE"], "HF_HUB_OFFLINE": True},
            "transformers.utils.hub": {"HF_MODULES_CACHE": os.environ["HF_MODULES_CACHE"], "TRANSFORMERS_CACHE": os.environ["TRANSFORMERS_CACHE"], "_is_offline_mode": True},
            "transformers.dynamic_module_utils": {"HF_MODULES_CACHE": os.environ["HF_MODULES_CACHE"]},
        }
        for module_name, values in updates.items():
            module = sys.modules.get(module_name)
            if module is not None:
                for name, value in values.items():
                    if hasattr(module, name):
                        self._module_attributes.append((module, name, getattr(module, name)))
                        setattr(module, name, value)
        # Dynamic custom-code package paths also belong to one lifetime.
        for name in list(sys.modules):
            if name == "transformers_modules" or name.startswith("transformers_modules."):
                self._saved_modules.setdefault(name, sys.modules.pop(name))

    def add_import_path(self, relative):
        source = _virtual_path(self.root, relative)
        if not source.is_dir():
            raise FileNotFoundError(f"Missing bundled architecture sources: {relative}")
        names = {p.stem for p in source.glob("*.py") if p.stem != "__init__"}
        names.update(p.name for p in source.iterdir() if p.is_dir() and any(p.rglob("*.py")))
        for name in list(sys.modules):
            if name.split(".")[0] in names:
                module = sys.modules[name]
                module_path = getattr(module, "__file__", "") or ""
                if not module_path.startswith(str(self.root)):
                    self._saved_modules.setdefault(name, sys.modules.pop(name))
        value = str(source)
        sys.path.insert(0, value)
        self._paths.append(value)
        importlib.invalidate_caches()
        return source

    def import_source(self, relative):
        path = _virtual_path(self.root, relative)
        name = "_classification_source_" + uuid.uuid4().hex
        spec = importlib.util.spec_from_file_location(name, path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot import bundled source: {relative}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        self._loaded_modules.add(name)
        spec.loader.exec_module(module)
        return module

    def resolve_data(self, value):
        parts = PurePosixPath(value)
        if parts.is_absolute() or ".." in parts.parts:
            raise ValueError("Unsafe virtual dataset path")
        for prefix, target in sorted(self.dataset_paths.items(), key=lambda item:len(item[0]), reverse=True):
            if value == prefix: return target
            if value.startswith(prefix + "/"):
                resolved = target.joinpath(*PurePosixPath(value[len(prefix)+1:]).parts).resolve(strict=True)
                if target.is_dir() and not resolved.is_relative_to(target):
                    raise ValueError("Dataset member escapes its root")
                return resolved
        raise FileNotFoundError(f"Unknown packaged dataset reference: {value}")

    def verify(self):
        if self.root is None or self._temporary is None:
            raise RuntimeError("Enter RuntimeBundle before verification")
        for name, expected in self.sha256.items():
            digest = hashlib.sha256()
            with _virtual_path(self.root, name).open("rb") as handle:
                for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                    digest.update(block)
            if digest.hexdigest() != expected:
                raise ValueError(f"Checkpoint asset SHA256 mismatch: {name}")
        for name, target in self.dataset_paths.items():
            if not target.exists(): raise FileNotFoundError(target)
        return {"model_key": self.model_key, "backend": self.spec["backend"], "asset_files": len(self.files), "source_files": len(EMBEDDED_SOURCES), "dataset_links": len(self.spec.get("dataset_links", {})), "sha256_verified": True}

    def close(self):
        if self.root is not None:
            prefix = str(self.root) + os.sep
            for name, module in list(sys.modules.items()):
                attributes = vars(module) if module is not None else {}
                path = attributes.get("__file__", "") or ""
                package_paths = attributes.get("__path__", ()) or ()
                try:
                    package_is_local = any(isinstance(p, str) and p.startswith(prefix) for p in package_paths)
                except TypeError:
                    package_is_local = False
                if name in self._loaded_modules or (isinstance(path, str) and path.startswith(prefix)) or package_is_local:
                    sys.modules.pop(name, None)
            sys.path[:] = [p for p in sys.path if not str(p).startswith(prefix)]
        sys.modules.update(self._saved_modules)
        self._saved_modules.clear()
        for module, name, value in reversed(self._module_attributes):
            setattr(module, name, value)
        self._module_attributes.clear()
        for name, value in self._environment.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
        self._environment.clear()
        if self._temporary is not None:
            self._temporary.cleanup()
            self._temporary = None

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()


def _labels(model):
    config = getattr(model, "config", None)
    raw = getattr(config, "id2label", None) or {}
    if raw:
        return {int(key): str(value) for key, value in raw.items()}
    count = getattr(config, "num_labels", None) or getattr(config, "num_classes", None) or getattr(model, "num_classes", 0)
    return {index: str(index) for index in range(count)}


def _logits(output):
    if hasattr(output, "logits"):
        return output.logits
    if isinstance(output, dict):
        return output["logits"]
    return output


class Predictor:
    """Context-managed classifier. ``predict_batch`` retains model label IDs."""

    def __init__(self, device="cuda", checkpoint=None):
        self.runtime = RuntimeBundle(checkpoint)
        self.runtime.__enter__()
        self.model = self.processor = self.transform = self.head = self.crop_model = None
        self.session = None
        try:
            import torch
            self.torch = torch
            self.device = torch.device(device if device != "cuda" or torch.cuda.is_available() else "cpu")
            self.spec = self.runtime.spec
            self.options = self.spec.get("options", {})
            backend = self.spec["backend"]
            aliases = {
                "tf_hf": "remaining_tf", "imatag": "remaining_imatag",
                "itsnotai": "remaining_itsnotai", "infobay": "remaining_infobay",
                "yolo_classify": "remaining_yolo_classify", "yolo_detect": "remaining_yolo_detect",
                "mammoscreen": "remaining_mammoscreen", "onnx_age_gender": "remaining_onnx_age_gender",
                "arabichar": "ahcd",
            }
            if backend == "hf":
                backend = "remaining_hf" if self.spec.get("entry") else (
                    "hf_folder" if self.options.get("data_format") == "imagefolder" else "hf_parquet"
                )
            self.backend = aliases.get(backend, backend)
            self.model_dir = self.runtime.model_dir
            self.labels = {}
            self._build()
            if self.model is not None and hasattr(self.model, "eval") and not self.backend.startswith("remaining_yolo"):
                self.model.to(self.device).eval()
            self._install_forward_timing()
        except BaseException:
            self.close()
            raise

    def _install_forward_timing(self):
        self.forward_timing = ForwardTiming(self.device)
        if self.backend in {"remaining_tf", "remaining_tf_hf"}:
            from tensorflow.python.eager import context
            self.forward_timing._sync = context.context().sync_executors
            self.forward_timing.watch(self.tf_model, "call")
        elif self.backend == "remaining_onnx_age_gender":
            # ORT run is synchronous and returns host outputs.
            self.forward_timing = ForwardTiming("cpu")
            self.forward_timing.watch(self.session, "run")
        elif self.backend.startswith("remaining_yolo"):
            # Ultralytics times inference separately from preprocess/postprocess,
            # excluding its initial dummy warmup. Consume that native profile.
            pass
        elif self.backend == "remaining_mammoscreen":
            self.forward_timing.watch(self.crop_model)
            # The ensemble's outer forward includes NumPy preprocessing.
            for index in range(self.model.num_models):
                self.forward_timing.watch(getattr(self.model, f"net{index}"))
        else:
            self.forward_timing.watch(self.model)
            if self.backend == "remaining_itsnotai":
                self.forward_timing.watch(self.model.beit if hasattr(self.model, "beit") else self.model.vit)
                self.forward_timing.watch(self.head)

    def _option(self, name, default=None):
        return self.options.get(name, self.spec.get(name, default))

    def _build(self):
        torch, directory, backend = self.torch, self.model_dir, self.backend
        if backend in {"remaining_tf", "remaining_tf_hf"}:
            import tensorflow as tf
            from transformers import AutoImageProcessor, TFAutoModelForImageClassification
            self.tf = tf
            self.processor = AutoImageProcessor.from_pretrained(directory, local_files_only=True)
            self.tf_model = TFAutoModelForImageClassification.from_pretrained(directory, local_files_only=True)
            self.labels = _labels(self.tf_model)
            return
        if backend.startswith("remaining_yolo"):
            from ultralytics import YOLO
            weights = sorted(directory.glob("*.pt"))
            if len(weights) != 1:
                raise ValueError(f"Expected exactly one YOLO checkpoint, got {len(weights)}")
            self.model = YOLO(str(weights[0]))
            self.labels = {int(k): str(v) for k, v in self.model.names.items()}
            return
        if backend == "remaining_onnx_age_gender":
            import onnxruntime as ort
            from transformers import AutoImageProcessor
            providers = ["CPUExecutionProvider"]
            if self.device.type == "cuda":
                providers.insert(0, "CUDAExecutionProvider")
            self.session = ort.InferenceSession(str(directory / "onnx" / "model.onnx"), providers=providers)
            self.processor = AutoImageProcessor.from_pretrained(directory, local_files_only=True)
            self.labels = {0: "male", 1: "female"}
            return
        if backend == "remaining_infobay":
            from torchvision.models import resnet18
            from torchvision.transforms import Compose, Grayscale, Normalize, Resize, ToTensor
            self.model = resnet18(weights=None)
            self.model.conv1 = torch.nn.Conv2d(1, 64, 7, 2, 3, bias=False)
            self.model.fc = torch.nn.Linear(512, 5)
            self.model.load_state_dict(torch.load(directory / "pytorch_model.bin", map_location="cpu", weights_only=True))
            self.labels = dict(enumerate(json.loads((directory / "classes.json").read_text())))
            self.transform = Compose([Grayscale(1), Resize((224, 224)), ToTensor(), Normalize([.5], [.5])])
            return
        if backend == "dima":
            from transformers import ViTImageProcessor, ViTForImageClassification
            from torchvision.transforms import Compose, Resize, ToTensor, Normalize
            processor = ViTImageProcessor.from_pretrained(directory, local_files_only=True)
            self.model = ViTForImageClassification.from_pretrained(directory, local_files_only=True)
            size = processor.size["height"]
            self.transform = Compose([Resize((size, size)), ToTensor(), Normalize(processor.image_mean, processor.image_std)])
            self.labels = _labels(self.model)
            return
        if backend == "sumsub":
            module = self.runtime.import_source(self.runtime.model_key + "/diffusions5M.py")
            from torchvision.transforms import Compose, Resize, ToTensor, Normalize
            self.model = module.Model5M()
            self.model.load_state_dict(torch.load(directory / "diffusions5M.pt", map_location="cpu", weights_only=True))
            self.transform = Compose([Resize((224, 224)), ToTensor(), Normalize([.485, .456, .406], [.229, .224, .225])])
            self.labels = {0: "human", 1: "AI"}
            return
        if backend == "aim":
            self.runtime.add_import_path("_shared_test_resources/code/ml-aim/aim-v1")
            from aim.v1.torch.data import val_transforms
            from aim.v1.torch.models import AIMForImageClassification
            if (directory / "model.safetensors").is_file():
                self.model = AIMForImageClassification.from_pretrained(str(directory), local_files_only=True)
            else:
                config = json.loads((directory / "config.json").read_text())
                with torch.device("meta"):
                    self.model = AIMForImageClassification(config=config)
                state = torch.load(directory / "pytorch_model.bin", map_location="cpu", weights_only=True, mmap=True)
                mismatch = self.model.load_state_dict(state, strict=False, assign=True)
                if mismatch.missing_keys:
                    raise RuntimeError(f"Missing AIM parameters: {mismatch.missing_keys}")
            self.transform = val_transforms(224)
            self.labels = {i: str(i) for i in range(1000)}
            return
        if backend == "lsnet":
            self.runtime.add_import_path(self._option("source_dir", "_shared_test_resources/official_sources/lsnet"))
            import model.build
            from timm.models import create_model
            from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
            self.model = create_model("lsnet_b", num_classes=1000, distillation=False)
            state = torch.load(directory / "pytorch_model.bin", map_location="cpu", weights_only=True)
            state = state.get("model", state.get("state_dict", state))
            mismatch = self.model.load_state_dict(state, strict=False)
            if mismatch.missing_keys:
                raise RuntimeError(f"Missing LSNet parameters: {mismatch.missing_keys}")
            self.transform = Compose([Resize(256, interpolation=3), CenterCrop(224), ToTensor(), Normalize([.485, .456, .406], [.229, .224, .225])])
            self.labels = {i: str(i) for i in range(1000)}
            return
        if backend == "timm":
            source = "_shared_test_resources/official_sources/timm"
            if (self.runtime.root / source / "timm" / "__init__.py").is_file():
                self.runtime.add_import_path(source)
            from timm.models import create_model, load_checkpoint
            from timm.data import create_transform, resolve_data_config
            name = self._option("model_name", self.runtime.model_key.split("__", 1)[1])
            self.model = create_model(name, pretrained=False)
            load_checkpoint(self.model, str(directory / "model.safetensors"))
            self.data_config = resolve_data_config({}, model=self.model, use_test_size=True)
            self.naflex = "naflex" in name.lower()
            if not self.naflex:
                self.transform = create_transform(**self.data_config, is_training=False)
            self.labels = _labels(self.model)
            return
        if backend == "nepa":
            self.runtime.add_import_path(self._option("official_source_dir", "_shared_test_resources/official_sources/nepa"))
            from models.vit_nepa import ViTNepaConfig, ViTNepaForImageClassification
            from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize, InterpolationMode
            config = ViTNepaConfig.from_pretrained(directory, local_files_only=True)
            self.model = ViTNepaForImageClassification.from_pretrained(directory, config=config, local_files_only=True)
            self.transform = Compose([Resize(256, interpolation=InterpolationMode.BICUBIC), CenterCrop((224, 224)), ToTensor(), Normalize([.5] * 3, [.5] * 3)])
            self.labels = _labels(self.model)
            return
        if backend == "remaining_mammoscreen":
            from transformers import AutoModel
            self.model = AutoModel.from_pretrained(directory, local_files_only=True, trust_remote_code=True)
            self.crop_model = AutoModel.from_pretrained(directory / "test_code/upstream/mammo-crop", local_files_only=True, trust_remote_code=True).to(self.device).eval()
            self.labels = {0: "no cancer", 1: "cancer"}
            return
        known = {"hf_folder", "hf_parquet", "mambavision", "ahcd", "remaining_hf", "remaining_imatag", "remaining_itsnotai"}
        if backend not in known:
            raise ValueError(f"Unsupported model backend: {backend}")
        from transformers import AutoImageProcessor, AutoModelForImageClassification
        self.model = AutoModelForImageClassification.from_pretrained(directory, local_files_only=True, trust_remote_code=True)
        self.labels = _labels(self.model)
        if backend == "ahcd":
            from torchvision.transforms import Compose, Grayscale, Resize, ToTensor, Normalize
            self.transform = Compose([Grayscale(1), Resize((32, 32)), ToTensor(), Normalize([.5], [.5])])
            return
        if backend == "mambavision":
            from timm.data.transforms_factory import create_transform
            self.transform = create_transform(input_size=(3, 512, 512), is_training=False, mean=self.model.config.mean, std=self.model.config.std, crop_mode=self.model.config.crop_mode, crop_pct=self.model.config.crop_pct)
            return
        if self.runtime.model_key.startswith("kawaimasa__"):
            from torchvision.transforms import Compose, Normalize, ToTensor
            self.transform = Compose([ToTensor(), Normalize([.485, .456, .406], [.229, .224, .225])])
            self.letterbox = True
        elif backend == "hf_folder" and getattr(self.model.config, "model_type", "") == "timm_wrapper":
            from torchvision.transforms import Compose, Normalize, Resize, ToTensor
            config = self.model.config.pretrained_cfg
            size = int(config["input_size"][-1])
            self.transform = Compose([Resize((size, size)), ToTensor(), Normalize(config["mean"], config["std"])])
        else:
            try:
                self.processor = AutoImageProcessor.from_pretrained(directory, local_files_only=True, trust_remote_code=True)
            except OSError:
                if backend == "hf_folder":
                    from torchvision.transforms import Compose, Grayscale, Normalize, Resize, ToTensor
                    grayscale = self._option("grayscale", False)
                    means = [0.] if grayscale else [float(x) for x in self._option("fallback_mean", "0.5,0.5,0.5").split(",")]
                    stds = [1.] if grayscale else [float(x) for x in self._option("fallback_std", "0.5,0.5,0.5").split(",")]
                    steps = [Grayscale(1)] if grayscale else []
                    size = int(self._option("fallback_image_size", 224))
                    self.transform = Compose(steps + [Resize((size, size)), ToTensor(), Normalize(means, stds)])
                elif backend in {"remaining_hf", "remaining_imatag"}:
                    from transformers import SiglipImageProcessor
                    size = int(getattr(getattr(self.model.config, "vision_config", None), "image_size", 224))
                    self.processor = SiglipImageProcessor(size={"height": size, "width": size})
                else:
                    raise
        if backend == "remaining_itsnotai":
            meta = json.loads((directory / "source_meta.json").read_text())
            self.head = torch.nn.Sequential(torch.nn.Dropout(.1), torch.nn.Linear(int(meta["hidden_size"]), 2))
            self.head.load_state_dict(torch.load(directory / "binary_head.pt", map_location="cpu", weights_only=True))
            self.head.to(self.device).eval()
            self.labels = {0: "REAL", 1: "FAKE"}

    def _result(self, label_id, score, **extra):
        label_id = int(label_id)
        return {"pred_id": label_id, "top1_id": label_id, "label": self.labels.get(label_id, "NO_DETECTION" if label_id == -1 else str(label_id)), "score": float(score), **extra}

    def _probability_results(self, logits):
        probabilities = logits.float().softmax(-1)
        confidence, indices = probabilities.max(-1)
        count = probabilities.shape[-1]
        for index in range(count):
            self.labels.setdefault(index, str(index))
        # Keep the existing max() top-1 exactly, including tied maxima. The
        # remaining order is PyTorch's native topk over the same probabilities.
        scores, ranks = probabilities.topk(min(5, count), dim=-1)
        results = []
        for winner, score, ranking, values in zip(indices.cpu().tolist(), confidence.cpu().tolist(), ranks.cpu().tolist(), scores.cpu().tolist()):
            rest = [(i, v) for i, v in zip(ranking, values) if i != winner]
            pairs = [(winner, score), *rest][:min(5, count)]
            results.append(self._result(winner, score, topk_ids=[i for i, _ in pairs], topk_scores=[v for _, v in pairs]))
        return results

    def _naflex_logits(self, images):
        from timm.data import create_naflex_loader
        torch = self.torch
        class ImageBatch(torch.utils.data.Dataset):
            transform = None
            def __len__(self):
                return len(images)
            def __getitem__(self, index):
                image = images[index]
                return self.transform(image) if self.transform is not None else image, 0
        config = self.data_config
        patch_size = getattr(getattr(self.model, "embeds", None), "patch_size", (16, 16))
        loader = create_naflex_loader(ImageBatch(), batch_size=len(images), use_prefetcher=True, interpolation=config["interpolation"], mean=config["mean"], std=config["std"], num_workers=0, crop_pct=config["crop_pct"], crop_mode=config["crop_mode"], pin_memory=self.device.type == "cuda", device=self.device, patch_size=patch_size, max_seq_len=int(self._option("naflex_max_seq_len", 576)))
        return torch.cat([self.model(inputs) for inputs, _ in loader], dim=0)

    def _mammogram_result(self, image):
        import numpy as np
        torch = self.torch
        if isinstance(image, (str, os.PathLike)) and Path(image).suffix.lower() == ".dcm":
            pixels = self.model.load_image_from_dicom(str(image))
            if pixels is None:
                raise ValueError(f"Could not decode DICOM image: {image}")
        else:
            pixels = np.asarray(_image(image).convert("L"))
        shape = torch.tensor([pixels.shape[:2]], device=self.device)
        crop_input = torch.from_numpy(self.crop_model.preprocess(pixels)).expand(1, 1, -1, -1).float().to(self.device)
        x, y, width, height = self.crop_model(crop_input, shape)[0].detach().cpu().numpy().astype(int)
        pixels = pixels[y:y + height, x:x + width]
        if pixels.size == 0:
            raise ValueError("MammoScreen crop returned an empty image")
        # The architecture applies sigmoid and averages views internally.
        probability = float(self.model([{"view": pixels}], device=str(self.device))["cancer"].item())
        predicted = int(probability >= .5)
        return self._result(predicted, probability if predicted else 1. - probability, cancer_probability=probability)

    def predict_batch(self, images):
        if self.runtime._temporary is None:
            raise RuntimeError("This predictor is closed")
        if not images:
            return []
        torch, backend = self.torch, self.backend
        with torch.inference_mode():
            if backend == "remaining_mammoscreen":
                return [self._mammogram_result(image) for image in images]
            images = [_image(image).convert("RGB") for image in images]
            if backend.startswith("remaining_yolo"):
                kwargs = {"device": str(self.device), "verbose": False}
                if backend == "remaining_yolo_detect":
                    kwargs["imgsz"] = 1024
                outputs = self.model.predict(images, **kwargs)
                durations = [item.speed.get("inference") for item in outputs]
                if len(outputs) != len(images) or any(x is None or x < 0 for x in durations):
                    raise RuntimeError("Missing Ultralytics native forward timing")
                self.forward_timing.seconds += sum(durations) / 1000.0
                results = []
                for output in outputs:
                    if backend == "remaining_yolo_classify":
                        ranks = [int(i) for i in output.probs.top5]
                        scores = [float(output.probs.data[i]) for i in ranks]
                        # Ultralytics top1 and top5 may select different tied
                        # maxima; preserve the native existing top1 decision.
                        first = int(output.probs.top1)
                        pairs = [(first, float(output.probs.top1conf))] + [(i, v) for i, v in zip(ranks, scores) if i != first]
                        pairs = pairs[:min(5, len(self.labels))]
                        results.append(self._result(first, output.probs.top1conf,
                                                   topk_ids=[i for i, _ in pairs], topk_scores=[v for _, v in pairs]))
                    elif output.boxes is None or len(output.boxes) == 0:
                        results.append(self._result(-1, 0.))
                    else:
                        best = int(output.boxes.conf.argmax().item())
                        results.append(self._result(output.boxes.cls[best].item(), output.boxes.conf[best].item()))
                return results
            if backend in {"remaining_tf", "remaining_tf_hf"}:
                import numpy as np
                values = self.processor(images=images, return_tensors="np")["pixel_values"]
                logits = self.tf_model(pixel_values=self.tf.convert_to_tensor(values), training=False).logits
                return self._probability_results(torch.from_numpy(np.asarray(logits)))
            if backend == "remaining_onnx_age_gender":
                import numpy as np
                values = self.processor(images=images, return_tensors="np")["pixel_values"]
                outputs = self.session.run(None, {self.session.get_inputs()[0].name: values})[0]
                results = []
                for output in outputs:
                    female_probability = float(output[1])
                    predicted = int(female_probability >= .5)
                    results.append(self._result(predicted, female_probability if predicted else 1. - female_probability, age=float(np.clip(np.rint(output[0]), 0, 100)), gender_score=female_probability))
                return results
            if backend == "timm" and self.naflex:
                return self._probability_results(self._naflex_logits(images))
            if getattr(self, "letterbox", False):
                from PIL import Image
                padded = []
                for image in images:
                    scale = 768 / max(image.size)
                    resized = image.resize((max(1, int(image.width * scale)), max(1, int(image.height * scale))), Image.Resampling.LANCZOS)
                    canvas = Image.new("RGB", (768, 768), (0, 0, 0))
                    canvas.paste(resized, ((768 - resized.width) // 2, (768 - resized.height) // 2))
                    padded.append(canvas)
                images = padded
            if self.transform is not None:
                pixels = torch.stack([self.transform(image) for image in images]).to(self.device)
                if backend == "ahcd":
                    output = self.model(tensor=pixels)
                elif backend in {"hf_folder", "nepa", "dima"}:
                    output = self.model(**{self._option("hf_input_key", "pixel_values"): pixels})
                elif backend.startswith("remaining_hf"):
                    output = self.model(pixel_values=pixels)
                else:
                    output = self.model(pixels)
            else:
                inputs = {key: value.to(self.device) for key, value in self.processor(images=images, return_tensors="pt").items()}
                if backend == "remaining_itsnotai":
                    if hasattr(self.model, "beit"):
                        features = self.model.beit(inputs["pixel_values"]).last_hidden_state[:, 0]
                    elif hasattr(self.model, "vit"):
                        features = self.model.vit(inputs["pixel_values"]).last_hidden_state[:, 0]
                    else:
                        raise RuntimeError("ItsNotAI backbone exposes neither beit nor vit")
                    output = self.head(features)
                else:
                    output = self.model(**inputs)
            logits = _logits(output)
            if backend == "remaining_imatag":
                values = logits[:, 0].float().cpu()
                return [self._result(int(v < 0), torch.sigmoid(v.abs()).item(), watermark_logit=float(v)) for v in values]
            return self._probability_results(logits)

    def predict(self, image):
        return self.predict_batch([image])[0]

    def close(self):
        if hasattr(self, "forward_timing"):
            self.forward_timing.close()
        self.model = self.processor = self.transform = self.head = self.crop_model = self.session = None
        if hasattr(self, "tf_model"):
            self.tf_model = None
        self.runtime.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()


def _image(value):
    from PIL import Image
    if isinstance(value, Image.Image):
        return value
    if isinstance(value, (str, os.PathLike)):
        with Image.open(value) as image:
            return image.copy()
    raise TypeError("Expected a PIL image or image-file path")


def load_model(device="cuda", checkpoint=None):
    return Predictor(device=device, checkpoint=checkpoint)


def predict(image, device="cuda", checkpoint=None, model=None):
    if model is not None:
        return model.predict(image)
    with load_model(device=device, checkpoint=checkpoint) as predictor:
        result = predictor.predict(image)
        save_forward_timing(MODEL_DIR, predictor.forward_timing.seconds, 1)
        return result


"""Dataset readers and inference-to-metric orchestration embedded in inference.py.

The frozen manifest contains ground-truth labels and sample identifiers only.
No previous predicted labels are used by a fresh inference run.
"""


def iter_evaluation_samples(bundle, maximum=0):
    import io
    from pathlib import Path
    from zipfile import ZipFile
    from PIL import Image

    spec = bundle.spec['data_spec']
    kind = spec['kind']
    rows = spec.get('rows', [])
    if maximum:
        rows = rows[:maximum]

    def path(value):
        candidate = bundle.resolve_data(value)
        if not candidate.exists():
            raise FileNotFoundError(f'Dataset member is missing: {value}')
        return candidate

    if kind == 'hf_disk':
        from datasets import DatasetDict, load_from_disk
        dataset = load_from_disk(str(path(spec['root'])))
        if isinstance(dataset, DatasetDict):
            dataset = dataset[spec.get('split', 'train')]
        for i, record in enumerate(rows):
            raw = dataset[record['index']]
            column = spec.get('image_column', 'image')
            value = raw[column]
            if isinstance(value, Image.Image):
                image = value.convert('RGB')
            elif isinstance(value, (bytes, bytearray)):
                with Image.open(io.BytesIO(value)) as opened:
                    image = opened.convert('RGB')
            elif isinstance(value, dict) and value.get('bytes') is not None:
                with Image.open(io.BytesIO(value['bytes'])) as opened:
                    image = opened.convert('RGB')
            elif isinstance(value, dict) and value.get('path'):
                with Image.open(path(spec['root']) / value['path']) as opened:
                    image = opened.convert('RGB')
            else:
                raise ValueError(f'No offline image for dataset index {record["index"]}')
            yield i, image, record['true_id']
    elif kind == 'image_paths':
        for i, record in enumerate(rows):
            with Image.open(path(record['path'])) as opened:
                # Preserve grayscale mammograms before official preprocessing.
                image = opened.convert('L' if spec.get('grayscale') else 'RGB')
            image.info['_coco_source_path'] = str(path(record['path']).resolve())
            yield i, image, record['true_id']
    elif kind == 'imagefolder':
        # Mirror torchvision's alphabetically sorted folder/file enumeration.
        import os
        extensions = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp')
        dataset_root = path(spec['root'])
        classes = sorted(p.name for p in dataset_root.iterdir() if p.is_dir())
        index = 0
        for class_id, class_name in enumerate(classes):
            for current, _, files in sorted(os.walk(dataset_root / class_name, followlinks=True)):
                for name in sorted(files):
                    if not name.lower().endswith(extensions):
                        continue
                    with Image.open(Path(current) / name) as opened:
                        image = opened.convert('RGB')
                    image.info['_coco_source_path'] = str((Path(current) / name).resolve())
                    yield index, image, class_id
                    index += 1
                    if maximum and index >= maximum:
                        return
        if index != spec['count']:
            raise RuntimeError(f'ImageFolder sample count changed: {index} != {spec["count"]}')
    elif kind == 'parquet':
        import pyarrow.parquet as pq
        index = 0
        for filename in spec['files']:
            reader = pq.ParquetFile(path(filename))
            for batch in reader.iter_batches(batch_size=128, columns=['image', 'label']):
                data = batch.to_pydict()
                for value, label in zip(data['image'], data['label']):
                    if isinstance(value, dict) and value.get('bytes') is not None:
                        with Image.open(io.BytesIO(value['bytes'])) as opened:
                            image = opened.convert('RGB')
                    else:
                        raise ValueError(f'Parquet record {index} is not self-contained')
                    yield index, image, int(label)
                    index += 1
                    if maximum and index >= maximum:
                        return
        if index != spec['count']:
            raise RuntimeError(f'Parquet sample count changed: {index} != {spec["count"]}')
    elif kind == 'ahcd':
        import numpy as np
        import pandas as pd
        images = pd.read_csv(path(spec['images_csv']), header=None).to_numpy(dtype=np.uint8)
        labels = pd.read_csv(path(spec['labels_csv']), header=None).to_numpy(dtype=np.int64).reshape(-1)
        if (int(labels.min()), int(labels.max())) == (1, 28):
            labels = labels - 1
        elif (int(labels.min()), int(labels.max())) != (0, 27):
            raise ValueError('AHCD labels must be 0..27 or 1..28')
        if len(images) != spec['count'] or len(labels) != spec['count']:
            raise ValueError('AHCD test sample count changed')
        count = min(maximum, len(labels)) if maximum else len(labels)
        for i in range(count):
            # The model preprocessing applies [-1,1] normalization later.
            image = Image.fromarray(images[i].reshape(32, 32).T)
            yield i, image, int(labels[i])
    elif kind == 'trashnet':
        from contextlib import ExitStack
        with ExitStack() as stack:
            archives = {name: stack.enter_context(ZipFile(path(virtual))) for name, virtual in spec['archives'].items()}
            for i, record in enumerate(rows):
                with Image.open(io.BytesIO(archives[record['archive']].read(record['member']))) as opened:
                    image = opened.convert('RGB')
                yield i, image, record['true_id']
    else:
        raise ValueError(f'Unsupported evaluation data kind: {kind}')


def run_dataset_inference(*, device='cuda', batch_size=None, maximum=0, output=None, checkpoint=None):
    """Infer the configured split, export ImageFolder data, call shared evaluate.py."""
    import importlib.util
    from pathlib import Path
    from tqdm.auto import tqdm

    model_dir = MODEL_DIR
    if output is not None: output = PATHS.project_path(output)
    if maximum and output is None: output = PATHS.RESULT_DIR/'smoke_results.json'
    if maximum and Path(output).resolve() == PATHS.RESULT_DIR/'results.json':
        raise ValueError('Limited inference must not overwrite the full result')
    helper_path = model_dir/'inference'/'bundle.py'
    spec = importlib.util.spec_from_file_location('_classification_imagenet_bundle', helper_path)
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    with load_model(device=device, checkpoint=checkpoint) as predictor:
        runtime = predictor.runtime
        size = int(runtime.spec.get('batch_size', 16)) if batch_size is None else batch_size
        if size <= 0: raise ValueError('batch_size must be positive')
        total = min(maximum, runtime.spec['data_spec']['count']) if maximum else runtime.spec['data_spec']['count']
        writer = helper.BundleWriter(model_dir, runtime.spec, predictor.labels, sample_limit=maximum)
        processed = 0

        def infer_batch(items):
            outputs = predictor.predict_batch([item[1] for item in items])
            if len(outputs) != len(items):
                raise RuntimeError('Model returned a different number of predictions')
            for (sample_id, image, truth), prediction in zip(items, outputs):
                if prediction.get('pred_id') is None:
                    raise RuntimeError('This model did not produce a classification label')
                writer.add(sample_id, prediction, truth, image)

        batch = []
        with tqdm(total=total, desc=runtime.spec['model_key'], unit='image') as progress:
            for sample in iter_evaluation_samples(runtime, maximum):
                batch.append(sample)
                if len(batch) >= size:
                    infer_batch(batch)
                    processed += len(batch); progress.update(len(batch)); batch.clear()
            if batch:
                infer_batch(batch)
                processed += len(batch); progress.update(len(batch))
        if processed != total:
            raise RuntimeError(f'Incomplete inference: {processed}/{total}')
        bundle = writer.finish(total)
        # Only forward calls are timed; export, ranking and metrics are excluded.
        save_forward_timing(model_dir, predictor.forward_timing.seconds, processed)
    return helper.score_bundle(model_dir, bundle, output, publish=True)



def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, help="Predict one image; omit for the complete evaluation dataset")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--check", action="store_true", help="Validate checkpoint hashes and dataset links without loading the model")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--max-images", "--max-samples", dest="max_samples", type=int, default=0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.image is not None: args.image = PATHS.project_path(args.image)
    if args.max_samples < 0:
        parser.error("--max-samples must be nonnegative")
    if args.check:
        with RuntimeBundle(args.checkpoint) as runtime:
            result = runtime.verify()
    elif args.image is not None:
        result = predict(args.image, device=args.device, checkpoint=args.checkpoint)
    else:
        result = run_dataset_inference(device=args.device, batch_size=args.batch_size, maximum=args.max_samples, output=args.output, checkpoint=args.checkpoint)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
