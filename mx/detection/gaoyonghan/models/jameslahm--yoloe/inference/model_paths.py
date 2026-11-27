"""Resolve paths for one model's preserved native files without creating links."""

from pathlib import Path
import hashlib
import json
import os
import sys


MODEL = Path(__file__).resolve().parents[1]


def digest(path):
    hasher = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def values():
    config = json.loads((MODEL / "paths.json").read_text())
    dataset = Path(os.environ.get("MODEL_DATASET_ROOT", config["dataset_root"])).expanduser()
    output = Path(os.environ.get("MODEL_OUTPUT_ROOT", "result/run_001")).expanduser()
    if not dataset.is_absolute():
        dataset = MODEL / dataset
    if not output.is_absolute():
        output = MODEL / output
    return {
        "__MODEL_ROOT__": str(MODEL),
        "__DATASET_ROOT__": str(dataset.resolve()),
        "__OUTPUT_ROOT__": str(output.resolve()),
        "__PYTHON__": sys.executable,
        "__SCORE_PYTHON__": os.environ.get("MODEL_SCORE_PYTHON", sys.executable),
    }


def expand(value, mapping):
    for old, new in mapping.items():
        value = value.replace(old, new)
    return value


def write_atomic(path, value):
    temporary = path.with_name(path.name + ".relocating")
    temporary.parent.mkdir(parents=True, exist_ok=True)
    temporary.write_text(value)
    if path.exists():
        temporary.chmod(path.stat().st_mode)
    temporary.replace(path)


def prepare():
    import fcntl

    mapping = values()
    plan = json.loads((MODEL / "inference/local_paths.json").read_text())
    if plan.get("links"):
        raise ValueError("This model package must not require symbolic links")
    identity = {"mapping": mapping, "plan_sha256": digest(MODEL / "inference/local_paths.json")}
    marker = MODEL / "inference/.local_paths_applied.json"
    if marker.is_file() and json.loads(marker.read_text()) == identity:
        return mapping
    with (MODEL / "inference/.local_paths.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        for item in plan["text_templates"]:
            source = MODEL / item["source"]
            target = MODEL / item["destination"]
            write_atomic(target, expand(source.read_text(), mapping))
        import shutil
        for relative in plan.get("classification_manifests", []):
            manifest_path = MODEL / relative
            if not manifest_path.is_file(): continue
            manifest_data = json.loads(manifest_path.read_text())
            for spec in manifest_data.get("models", []):
                target = Path(spec.get("index", ""))
                local = MODEL / "inference/native/indices" / target.name
                if target.is_absolute() and target.is_relative_to(MODEL) and not target.exists() and local.is_file():
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(local, target)
        for relative in plan.get("classification_manifests", []):
            path = MODEL / relative
            if not path.is_file():
                continue
            data = json.loads(path.read_text())
            for spec in data.get("models", []):
                index = spec.get("index")
                if index and Path(index).is_file():
                    spec["index_sha256"] = digest(index)
                for name, info in spec.get("data_containers", {}).items():
                    source = Path(name)
                    if source.is_file():
                        stat = source.stat()
                        if stat.st_size != info["size"]:
                            raise ValueError("Dataset size differs: " + name)
                        info["mtime_ns"] = stat.st_mtime_ns
                for key in ("class_map", "corrected_label_config"):
                    if key not in spec:
                        continue
                    name = spec[key] if key == "class_map" else spec[key]["path"]
                    if Path(name).is_file():
                        if key == "class_map":
                            spec["class_map_sha256"] = digest(name)
                        else:
                            spec[key]["sha256"] = digest(name)
                if "protocol_fingerprint" in spec:
                    payload = {key: value for key, value in spec.items() if key != "protocol_fingerprint"}
                    spec["protocol_fingerprint"] = hashlib.sha256(
                        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
                    ).hexdigest()
            write_atomic(path, json.dumps(data, ensure_ascii=False, indent=2) + "\n")
        marker.write_text(json.dumps(identity, ensure_ascii=False, indent=2) + "\n")
    return mapping
