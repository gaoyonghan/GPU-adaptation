"""离线比较：第一个结果目录为baseline（A100），第二个为candidate（910B）。"""
import argparse
import csv
import json
from contextlib import ExitStack
from pathlib import Path
import numpy as np

import hashlib
import re

SCHEMA = "raw-tensor-features-v4"
DYNAMIC_SCHEMA = "raw-tensor-features-v4-dynamic-single-image-v1"

def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))

def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")

def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

def inside(root, relative):
    root = Path(root).resolve()
    path = (root / relative).resolve()
    if Path(relative).is_absolute() or not path.is_relative_to(root):
        raise ValueError(f"Path outside root: {relative}")
    return path

def read_manifest(path):
    rows, seen = [], set()
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        sid = row["sample_id"]
        if not isinstance(sid, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", sid):
            raise ValueError(f"Unsafe sample_id: {sid!r}")
        if sid in seen:
            raise ValueError(f"Duplicate sample_id: {sid}")
        if not isinstance(row["relative_path"], str):
            raise ValueError(f"Invalid image path: {sid}")
        inside(Path(path).parent, row["relative_path"])
        if not re.fullmatch(r"[0-9a-f]{64}", row["image_sha256"]):
            raise ValueError(f"Invalid image hash: {sid}")
        seen.add(sid)
        rows.append(row)
    if not rows:
        raise ValueError("Empty input manifest")
    return rows

# 复用旧项目的数值实现；CSV保留明细，summary聚焦前三个核心指标。
METRICS = ("cosine", "cosine_distance", "l2_error", "relative_l2", "norm_ratio", "mean_abs_error", "max_abs_error")
CORE_METRICS = ("cosine", "l2_error", "relative_l2")


def measure(x, y):
    """评分用 Float64；零范数影响各指标不同，分别记录有效值。"""
    x, y = x.astype(np.float64, copy=False).reshape(-1), y.astype(np.float64, copy=False).reshape(-1)
    row = {key: None for key in METRICS}
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        return {**row, "status": "nonfinite"}
    nx, ny = float(np.linalg.norm(x)), float(np.linalg.norm(y))
    delta = y-x
    absolute = float(np.linalg.norm(delta))
    row.update(l2_error=absolute, mean_abs_error=float(np.abs(delta).mean()), max_abs_error=float(np.abs(delta).max()))
    if nx > 0:
        row.update(relative_l2=absolute/nx, norm_ratio=ny/nx)
    if nx > 0 and ny > 0:
        cosine = float(np.clip(np.dot(x,y)/(nx*ny), -1, 1))
        row.update(cosine=cosine, cosine_distance=1-cosine)
    return {**row, "status": "valid" if nx > 0 and ny > 0 else "zero_norm"}


def aggregate(rows):
    result = {}
    for key in METRICS:
        values = np.asarray([r[key] for r in rows if r[key] is not None], dtype=np.float64)
        result[key] = {"valid_count": len(values), "mean": float(values.mean()) if len(values) else None,
                       "min": float(values.min()) if len(values) else None,
                       "max": float(values.max()) if len(values) else None,
                       "p50": float(np.percentile(values,50)) if len(values) else None,
                       "p95": float(np.percentile(values,95)) if len(values) else None,
                       "p99": float(np.percentile(values,99)) if len(values) else None}
    return result


def read_run(root):
    """只读取当前v4；原始dtype指纹与旧结果不同，两端需使用相同版本。"""
    root = Path(root)
    cfg, summary = read_json(root / "evaluation_config.json"), read_json(root / "summary.json")
    schema = cfg.get("schema")
    if schema not in {SCHEMA, DYNAMIC_SCHEMA} or summary.get("schema") != schema:
        raise ValueError("Expected matching v4 or v4 dynamic-single-image result schemas")
    dynamic = schema == DYNAMIC_SCHEMA
    if dynamic and (cfg["input"]["shapes"] != "per_image_index" or
                    cfg["output"]["shape"] != "per_image_index"):
        raise ValueError("Dynamic result requires per-image input/output shapes")
    if summary["status"] not in {"complete", "invalid_features"}:
        raise ValueError("Extraction is incomplete")
    manifest_path = root / "input_manifest.jsonl"
    manifest = read_manifest(manifest_path)
    if sha256_file(manifest_path) != cfg["dataset"]["manifest_sha256"]:
        raise ValueError("Manifest hash differs")
    if len(manifest) != summary["num_images"] or len(manifest) != cfg["dataset"]["num_images"]:
        raise ValueError("Image counts disagree")
    if not cfg["model"]["checkpoint_files"]:
        raise ValueError("Missing checkpoint identity")
    images = {row["sample_id"]: row for row in manifest}
    index = {}
    for line in (root / "feature_index.jsonl").read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        sid = row["sample_id"]
        if sid in index or sid not in images:
            raise ValueError(f"Duplicate/unexpected feature: {sid}")
        if dynamic:
            shape = row.get("feature_shape")
            if (not isinstance(shape, list) or not shape or
                    any(type(n) is not int or n < 1 for n in shape) or
                    shape[1:] != cfg["output"]["fixed_tail_shape"] or
                    not isinstance(row.get("input_shapes"), dict) or not row["input_shapes"] or
                    not isinstance(row.get("input_dtypes"), dict) or not row["input_dtypes"]):
                raise ValueError(f"Missing/invalid dynamic per-image shape or dtype: {sid}")
        index[sid] = {
            **row,
            "input_shapes": row.get("input_shapes", cfg["input"]["shapes"]),
            "input_dtypes": row.get("input_dtypes", cfg["input"]["dtypes"]),
        }
    if set(index) != set(images):
        raise ValueError("Missing features; refusing intersection-only comparison")
    execution = cfg["execution"]
    env_path = root / "model_environment.json"
    environment = read_json(env_path) if env_path.is_file() else {}
    environment = {k: environment[k] for k in ("python", "packages", "versions", "software", "torch", "torch_npu", "transformers", "timm", "numpy", "pillow") if k in environment}
    environment["device"] = execution.get("device", summary.get("device"))
    return {"root": root, "schema": schema, "identity": cfg["model"], "preprocessing": cfg["preprocessing"],
            "numerical": {key: execution[key] for key in ("precision", "parameter_dtypes", "seed", "math")},
            "feature": cfg["output"], "images": images, "index": index, "environment": environment,
            "performance": summary["performance"], "loader": cfg["loader"], "timing": cfg["timing"],
            "batch_size": execution["batch_size"], "warmup_steps": execution["warmup_steps"]}


def load_feature(run, row):
    path = inside(run["root"], row["feature_file"])
    if sha256_file(path) != row["feature_sha256"]:
        raise ValueError(f"Feature checksum differs: {row['sample_id']}")
    array = np.load(path, allow_pickle=False)
    expected_dtype = np.dtype(run["feature"]["saved_dtype"]).newbyteorder("<")
    expected_shape = row.get("feature_shape", run["feature"]["shape"])
    if ((expected_shape is not None and list(array.shape) != expected_shape) or
            array.dtype != expected_dtype or array.dtype.kind != "f" or not array.size):
        raise ValueError(
            f"Feature shape/dtype differs: {row['sample_id']}; "
            f"root={run['root']}; "
            f"actual_shape={list(array.shape)}, expected_shape={expected_shape}; "
            f"actual_dtype={array.dtype.str}, expected_dtype={expected_dtype.str}; "
            f"size={array.size}"
        )
    return array


def compare_results(
    baseline, candidate, output_dir,
    min_cosine=None, vector_axis=None, strict_inputs=False,
    allow_numerical_differences=False, strict_implementation=False,
):
    """默认比较同一原图经过两端完整系统的输出；strict_inputs额外要求预处理配置一致，文件和张量哈希差异只记录。"""
    if min_cosine is not None and not -1 <= min_cosine <= 1:
        raise ValueError("min_cosine must be in [-1,1]")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    report = {"baseline": str(Path(baseline).resolve()), "candidate": str(Path(candidate).resolve()),
              "scope": "image_flattened", "comparison_mode": "strict_inputs" if strict_inputs else "whole_system",
              "status": "failed",
              "relative_l2_definition": "norm(candidate-baseline)/norm(baseline), per image"}
    try:
        a, b = read_run(baseline), read_run(candidate)
        if a["schema"] != b["schema"]:
            raise ValueError("Comparison result schemas differ")
        # processor文件属于预处理条件；其差异由input_comparison记录，不冒充权重差异。
        processor_files = {"preprocessor_config.json", "processor_config.json"}
        # Separate model/weight identity from platform implementation and execution policy.
        def model_identity(run):
            raw = run["identity"]
            weights = {k: v for k, v in raw["checkpoint_files"].items()
                       if Path(k).suffix.lower() not in {".json", ".py", ".pyc", ".md", ".txt", ".yaml", ".yml", ".toml"}}
            if not weights:
                raise ValueError("Missing weight content fingerprints; metadata alone cannot identify a checkpoint")
            if not raw.get("id"):
                raise ValueError("Missing model id")
            return {"model": {k: raw[k] for k in ("id", "variant", "network", "architecture", "arch") if k in raw},
                    "weight_hashes": sorted(weights.values())}
        if model_identity(a) != model_identity(b):
            raise ValueError("Comparison protocol differs: model or weight contents")
        feature_keys = {"name", "selection", "shape", "fixed_tail_shape"}
        if ({k: v for k, v in a["feature"].items() if k in feature_keys} !=
                {k: v for k, v in b["feature"].items() if k in feature_keys}):
            raise ValueError("Comparison protocol differs: output semantics or shape")
        implementation_equal = a["identity"] == b["identity"]
        report["admission_policy"] = "same-model-weights-images-output-v2"
        report["implementation_comparison"] = {
            "matches": implementation_equal,
            "different_fields": [k for k in sorted(set(a["identity"]) | set(b["identity"]))
                                 if a["identity"].get(k) != b["identity"].get(k)],
            "baseline": a["identity"], "candidate": b["identity"],
            "strict": strict_implementation,
        }
        if strict_implementation and not implementation_equal:
            raise ValueError("Strict implementation comparison requires identical recorded model metadata")
        report["output_comparison"] = {"baseline": a["feature"], "candidate": b["feature"]}
        report["numerical_comparison"] = {
            "matches": a["numerical"] == b["numerical"],
            "baseline": a["numerical"], "candidate": b["numerical"],
            "differences_explicitly_allowed": allow_numerical_differences,
            "policy": "record differences; whole-system comparison allows different execution policies",
        }
        report["environment_comparison"] = {"baseline": a["environment"], "candidate": b["environment"]}
        if set(a["images"]) != set(b["images"]):
            raise ValueError("Image sets differ; refusing intersection-only comparison")
        input_matches = {}
        input_differences = {key: 0 for key in ("input_tensor_sha256", "input_shapes", "input_dtypes")}
        for sid in a["images"]:
            if a["images"][sid]["image_sha256"] != b["images"][sid]["image_sha256"]:
                raise ValueError(f"Source image differs: {sid}")
            mismatches = [key for key in input_differences if a["index"][sid][key] != b["index"][sid][key]]
            input_matches[sid] = not mismatches
            for key in mismatches:
                input_differences[key] += 1
        config_equal = a["preprocessing"] == b["preprocessing"]
        files_equal = ({k: v for k, v in a["identity"]["checkpoint_files"].items() if k in processor_files} ==
                       {k: v for k, v in b["identity"]["checkpoint_files"].items() if k in processor_files})
        report["input_comparison"] = {
            "preprocessing_matches": config_equal, "processor_files_match": files_equal,
            "matched_tensor_images": sum(input_matches.values()),
            "different_tensor_images": sum(not value for value in input_matches.values()),
            "difference_counts": input_differences,
            "hash_policy": "processor and input tensor hashes are diagnostic only",
            "interpretation": "Output differences may include preprocessing numerical differences",
        }
        if strict_inputs and not config_equal:
            raise ValueError("Strict input comparison requires identical preprocessing configuration")
        metric_rows = []
        vector_stats = {key: {"valid_count": 0, "sum": 0.0, "min": None, "max": None} for key in CORE_METRICS}
        vector_count = vector_invalid = 0
        with ExitStack() as stack:
            image_file = stack.enter_context((output_dir / "image_similarity.csv").open("x", newline="", encoding="utf-8"))
            iw = csv.DictWriter(image_file, ["sample_id", "input_matches", *METRICS, "status"])
            iw.writeheader()
            vw = None
            if vector_axis is not None:
                vector_file = stack.enter_context((output_dir / "vector_similarity.csv").open("x", newline="", encoding="utf-8"))
                vw = csv.DictWriter(vector_file, ["sample_id", "vector_index", *METRICS, "status"])
                vw.writeheader()
            for sid in a["images"]:
                x, y = load_feature(a, a["index"][sid]), load_feature(b, b["index"][sid])
                if x.shape != y.shape:
                    raise ValueError(
                        f"Feature shapes differ between runs: {sid}; "
                        f"baseline={x.shape}, candidate={y.shape}"
                    )
                row = measure(x, y)
                metric_rows.append(row)
                iw.writerow({"sample_id": sid, "input_matches": input_matches[sid], **row})
                if vw is not None:
                    if not -x.ndim <= vector_axis < x.ndim:
                        raise ValueError(f"vector_axis={vector_axis} invalid for {x.shape}")
                    axis = vector_axis % x.ndim
                    xv = np.moveaxis(x, axis, -1).reshape(-1, x.shape[axis])
                    yv = np.moveaxis(y, axis, -1).reshape(-1, y.shape[axis])
                    for i, (xx, yy) in enumerate(zip(xv, yv)):
                        value = measure(xx, yy)
                        vw.writerow({"sample_id": sid, "vector_index": i, **value})
                        vector_count += 1
                        vector_invalid += int(value["status"] != "valid")
                        # 逐向量只累计mean/min/max，不把数百万条统计常驻内存。
                        for key, stats in vector_stats.items():
                            v = value[key]
                            if v is not None:
                                stats["valid_count"] += 1
                                stats["sum"] += v
                                stats["min"] = v if stats["min"] is None else min(stats["min"], v)
                                stats["max"] = v if stats["max"] is None else max(stats["max"], v)
        invalid = sum(row["status"] != "valid" for row in metric_rows)
        failures = sum(row["cosine"] is not None and min_cosine is not None and row["cosine"] < min_cosine for row in metric_rows)
        report.update(status="invalid_features" if invalid else "threshold_failed" if failures else "passed" if min_cosine is not None else "measured",
                      matched_images=len(metric_rows), invalid_images=invalid)
        all_stats = aggregate(metric_rows)
        report["metrics"] = {key: all_stats[key] for key in CORE_METRICS}
        if min_cosine is not None:
            report["threshold"] = {"min_cosine": min_cosine, "failed_images": failures}
        if vw is not None:
            report["vector_detail"] = {"vector_axis": vector_axis, "num_vectors": vector_count, "invalid_vectors": vector_invalid}
            for key, stats in vector_stats.items():
                total = stats.pop("sum")
                stats["mean"] = total / stats["valid_count"] if stats["valid_count"] else None
                report["vector_detail"][key] = stats
        differences = [key for key in ("batch_size", "warmup_steps", "loader", "timing") if a[key] != b[key]]
        if list(a["images"]) != list(b["images"]):
            differences.append("manifest_order")
        keys = ("forward_images_per_second", "forward_mean_ms_per_image", "end_to_end_images_per_second")
        report["performance"] = {"baseline": {key: a["performance"][key] for key in keys},
                                 "candidate": {key: b["performance"][key] for key in keys},
                                 "recorded_condition_differences": differences}
        for key in ("forward_images_per_second", "end_to_end_images_per_second"):
            if key == "forward_images_per_second" and a["timing"] != b["timing"]:
                report["performance"]["forward_ratio_note"] = (
                    "Not calculated because timing definitions differ"
                )
                continue
            denominator = a["performance"][key]
            if denominator and b["performance"][key] is not None:
                report["performance"][key + "_ratio"] = b["performance"][key] / denominator
    except (ValueError, KeyError, TypeError, OSError, AttributeError) as exc:
        report["status"] = "failed"
        report["error"] = str(exc)
    write_json(output_dir / "comparison_summary.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", nargs="?", type=Path, help="A100结果根目录，包含features/及元数据")
    parser.add_argument("candidate", nargs="?", type=Path, help="910B结果根目录")
    parser.add_argument("--output-root", "--output-dir", dest="output_dir", type=Path, required=True, help="尚不存在的新目录")
    parser.add_argument("--vector-axis", type=int, help="可选逐向量：[N,D]用-1，[C,H,W]用0")
    parser.add_argument("--min-cosine", type=float, help="可选阈值；未设置则只报告实测值")
    parser.add_argument("--strict-inputs", action="store_true", help="可选：要求两端记录的预处理配置一致；processor文件及输入张量哈希差异仅记录")
    parser.add_argument(
        "--allow-numerical-differences",
        action="store_true",
        help="兼容旧命令；默认已允许数值策略不同，并在报告中保留两端策略",
    )
    parser.add_argument("--strict-implementation", action="store_true", help="可选：要求模型实现元数据及源码指纹完全一致")
    parser.add_argument("--baseline", dest="baseline_named", type=Path)
    parser.add_argument("--candidate", dest="candidate_named", type=Path)
    args = parser.parse_args()
    if (args.baseline is not None and args.baseline_named is not None) or (args.candidate is not None and args.candidate_named is not None):
        parser.error("Do not supply the same input both positionally and by name")
    args.baseline = args.baseline_named or args.baseline
    args.candidate = args.candidate_named or args.candidate
    if args.baseline is None or args.candidate is None:
        parser.error("Both --baseline and --candidate are required (or two legacy positional inputs)")
    task_root = Path(__file__).resolve().parent
    for key in ("baseline", "candidate", "output_dir"):
        value = getattr(args, key)
        if not value.is_absolute(): setattr(args, key, task_root / value)
    report = compare_results(
        args.baseline,
        args.candidate,
        args.output_dir,
        args.min_cosine,
        args.vector_axis,
        args.strict_inputs,
        args.allow_numerical_differences,
        args.strict_implementation,
    )
    print("Comparison:", report["status"], args.output_dir)
    if report["status"] == "failed":
        print(report["error"])
        raise SystemExit(2)
    if report["status"] in {"invalid_features", "threshold_failed"}:
        raise SystemExit(3)


if __name__ == "__main__":
    main()
