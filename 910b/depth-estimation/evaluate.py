# 统一深度评分器，指标算法复用原evaluate_metrics.py。
import argparse
import csv
import json
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

# ==============================================================================
# ★ 默认评测目录（支持相对路径，可直接修改此处后运行 python evaluate_metrics.py）
# ==============================================================================
DEFAULT_RESULT_DIR = str(Path(__file__).resolve().parent / "zehong" / "models" / "glpn-nyu" / "result" / "REPLACE_WITH_RUN_DIRECTORY")

DATASET_PROTOCOLS = {
    "KITTI": {
        "split": "Eigen test",
        "min_depth": 1e-5,
        "max_depth": 80.0,
        "crop": "KITTI benchmark crop followed by Eigen evaluation crop",
    },
    "NYUv2": {
        "split": "test",
        "min_depth": 1e-3,
        "max_depth": 10.0,
        "crop": "Eigen evaluation crop [45:471, 41:601]",
    },
    "DIODE": {
        "split": "val/indoors and val/outdoor",
        "min_depth": 0.6,
        "max_depth": 350.0,
        "crop": None,
    },
    "Sintel": {
        "split": "training/final",
        "min_depth": 1e-5,
        "max_depth": 72.0,
        "crop": None,
    },
}

METRIC_NAMES = ("abs_rel", "rmse", "log10", "delta1", "delta2", "delta3")



def _readable_timing_fields(value):
    """Add readable aliases; preserve measured timing boundaries and legacy keys."""
    if not isinstance(value, dict):
        return value
    for child in value.values():
        if isinstance(child, dict):
            _readable_timing_fields(child)
    if 'mean_forward_ms' in value:
        value['forward_ms_per_image'] = value['mean_forward_ms']
    if 'forward_fps' in value:
        value['forward_images_per_second'] = value['forward_fps']
    if 'total_forward_seconds' in value:
        value['forward_seconds'] = value['total_forward_seconds']
    datasets = value.get('datasets')
    if isinstance(datasets, dict) and datasets:
        rows = list(datasets.values())
        rows = [r.get('evaluation', r) for r in rows]
        counts = [r.get('forward_recorded_images', r.get('stage1_completed_images', r.get('num_samples', r.get('sample_count')))) for r in rows]
        seconds = [r.get('total_forward_seconds', r.get('forward_seconds')) for r in rows]
        if all(n is not None for n in counts) and all(s is not None for s in seconds):
            n, total = sum(counts), sum(seconds)
            overall = value.setdefault('overall', {})
            info = overall.get('evaluation', overall)
            info['forward_seconds'] = total
            info['forward_ms_per_image'] = total * 1000 / n if n else None
            info['forward_images_per_second'] = n / total if total > 0 else None
    return value

def _read_sintel_depth(path):
    with Path(path).open("rb") as depth_file:
        tag = np.fromfile(depth_file, dtype="<f4", count=1)
        width = np.fromfile(depth_file, dtype="<i4", count=1)
        height = np.fromfile(depth_file, dtype="<i4", count=1)
        if len(tag) != 1 or tag[0] != 202021.25 or len(width) != 1 or len(height) != 1:
            raise ValueError(f"Invalid Sintel depth header: {path}")
        pixel_count = int(width[0]) * int(height[0])
        depth = np.fromfile(depth_file, dtype="<f4", count=pixel_count)
    if depth.size != pixel_count:
        raise ValueError(f"Truncated Sintel depth file: {path}")
    return depth.reshape((int(height[0]), int(width[0])))


def _kitti_benchmark_crop(array):
    height, width = array.shape[:2]
    crop_height, crop_width = 352, 1216
    if height < crop_height or width < crop_width:
        raise ValueError(f"KITTI image is smaller than benchmark crop: {array.shape}")
    top = height - crop_height
    left = (width - crop_width) // 2
    return array[top : top + crop_height, left : left + crop_width]


def _read_depth_and_mask(dataset_name, depth_path, mask_path=""):
    """读取金标准深度图 (Ground Truth) 并按数据集协议生成有效像素掩码 valid_mask"""
    protocol = DATASET_PROTOCOLS[dataset_name]
    if dataset_name == "KITTI":
        depth = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
        if depth is None:
            raise OSError(f"Unable to read KITTI depth: {depth_path}")
        depth = _kitti_benchmark_crop(depth).astype(np.float32) / 256.0
        valid_mask = np.zeros(depth.shape, dtype=bool)
        height, width = depth.shape
        valid_mask[
            int(0.3324324 * height) : int(0.91351351 * height),
            int(0.0359477 * width) : int(0.96405229 * width),
        ] = True
    elif dataset_name == "NYUv2":
        depth = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
        if depth is None:
            raise OSError(f"Unable to read NYUv2 depth: {depth_path}")
        depth = depth.astype(np.float32) / 1000.0
        valid_mask = np.zeros(depth.shape, dtype=bool)
        valid_mask[45:471, 41:601] = True
    elif dataset_name == "DIODE":
        depth = np.load(depth_path, allow_pickle=False).squeeze().astype(np.float32)
        if not mask_path:
            mask_path = str(Path(depth_path).with_name(f"{Path(depth_path).stem}_mask.npy"))
        valid_mask = np.load(mask_path, allow_pickle=False).squeeze() > 0.5
    elif dataset_name == "Sintel":
        depth = _read_sintel_depth(depth_path).astype(np.float32)
        valid_mask = depth < 1e10
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    valid_mask &= np.isfinite(depth)
    valid_mask &= depth > protocol["min_depth"]
    valid_mask &= depth < protocol["max_depth"]
    return depth, valid_mask


def _align_relative_depth(raw_prediction, target_depth, valid_mask, min_depth, max_depth):
    """在正深度空间进行尺度 (scale) 和偏移 (shift) 最小二乘对齐（适用于 DA3 系列）"""
    prediction = raw_prediction[valid_mask].astype(np.float64)
    target = target_depth[valid_mask].astype(np.float64)
    count = prediction.size
    if count < 10:
        raise ValueError(f"Only {count} valid pixels")

    sum_prediction = prediction.sum()
    sum_target = target.sum()
    denominator = count * np.dot(prediction, prediction) - sum_prediction**2
    if not np.isfinite(denominator) or abs(denominator) < np.finfo(np.float64).eps:
        raise ValueError("Prediction is constant on valid pixels")

    scale = (
        count * np.dot(prediction, target)
        - sum_prediction * sum_target
    ) / denominator
    shift = (sum_target - scale * sum_prediction) / count

    aligned_depth = raw_prediction.astype(np.float64) * scale + shift
    aligned_depth = np.clip(aligned_depth, min_depth, max_depth)
    return aligned_depth, float(scale), float(shift)


def _align_inverse_depth(raw_prediction, target_depth, valid_mask, min_depth, max_depth):
    """在逆深度空间进行尺度 (scale) 和偏移 (shift) 最小二乘对齐（适用于 DepthCrafter / Depth-Anything-V2）"""
    prediction = raw_prediction[valid_mask].astype(np.float64)
    target_disparity = 1.0 / target_depth[valid_mask].astype(np.float64)
    count = prediction.size
    if count < 10:
        raise ValueError(f"Only {count} valid pixels")

    sum_prediction = prediction.sum()
    sum_target = target_disparity.sum()
    denominator = count * np.dot(prediction, prediction) - sum_prediction**2
    if not np.isfinite(denominator) or abs(denominator) < np.finfo(np.float64).eps:
        raise ValueError("Prediction is constant on valid pixels")

    scale = (
        count * np.dot(prediction, target_disparity)
        - sum_prediction * sum_target
    ) / denominator
    shift = (sum_target - scale * sum_prediction) / count
    aligned_disparity = raw_prediction.astype(np.float64) * scale + shift
    aligned_disparity = np.clip(aligned_disparity, 1.0 / max_depth, 1.0 / min_depth)
    aligned_depth = 1.0 / aligned_disparity
    return aligned_depth, float(scale), float(shift)


def _compute_metrics(prediction, target, valid_mask):
    """在有效像素上计算 6 项标准深度估计指标"""
    prediction = prediction[valid_mask]
    target = target[valid_mask].astype(np.float64)
    difference = prediction - target
    ratio = np.maximum(target / prediction, prediction / target)
    return {
        "abs_rel": float(np.mean(np.abs(difference) / target)),
        "rmse": float(np.sqrt(np.mean(difference**2))),
        "log10": float(np.mean(np.abs(np.log10(prediction) - np.log10(target)))),
        "delta1": float(np.mean(ratio < 1.25)),
        "delta2": float(np.mean(ratio < 1.25**2)),
        "delta3": float(np.mean(ratio < 1.25**3)),
    }


def _mean_metrics(rows):
    successful = [row for row in rows if row["status"] == "ok"]
    if not successful:
        return {name: None for name in METRIC_NAMES}
    return {
        name: float(np.mean([float(row[name]) for row in successful]))
        for name in METRIC_NAMES
    }


def _write_csv(path, rows):
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def evaluate_from_predictions(result_dir, keep_raw_predictions=True, dataset_root=None):
    """阶段3：读取阶段1保存的预测图与双计时记录，与金标准对比计算指标并汇总落盘。"""
    result_dir = Path(result_dir)
    if not result_dir.is_dir():
        raise FileNotFoundError(f"Result directory not found: {result_dir}")

    config_path = result_dir / "evaluation_config.json"
    inference_summary_path = result_dir / "inference_summary.json"
    if not config_path.is_file() or not inference_summary_path.is_file():
        raise FileNotFoundError(
            f"Missing evaluation_config.json or inference_summary.json in {result_dir}"
        )

    config = json.loads(config_path.read_text(encoding="utf-8"))
    inference_summary = json.loads(inference_summary_path.read_text(encoding="utf-8"))

    dataset_names = config["datasets"]
    prediction_type = config.get("prediction_type", "metric depth")
    max_samples = config.get("max_samples", None)

    # 新包使用相对GT路径；旧包的绝对GT路径仍可直接读取。
    root = Path(dataset_root or config.get("dataset_root", "."))
    if dataset_root is None and not root.is_absolute():
        root = result_dir / root
    all_rows = []
    all_forward_timings_ms = []
    dataset_summaries = {}
    stage3_global_start = time.perf_counter()

    for dataset_name in dataset_names:
        stage3_dataset_start = time.perf_counter()

        protocol = DATASET_PROTOCOLS[dataset_name]
        dataset_dir = result_dir / dataset_name
        records_csv_path = dataset_dir / "inference_records.csv"
        if not records_csv_path.is_file():
            raise FileNotFoundError(f"Missing inference_records.csv: {records_csv_path}")

        with records_csv_path.open("r", encoding="utf-8", newline="") as csv_file:
            records = list(csv.DictReader(csv_file))

        print(f"\n[Evaluation] Computing metrics for {dataset_name}: {len(records)} images (mode: {prediction_type})")
        rows = []

        for idx, rec in enumerate(records, 1):
            raw_rel_path = Path(rec["raw_prediction_path"])
            raw_abs_path = result_dir / raw_rel_path if not raw_rel_path.is_absolute() else raw_rel_path
            if not raw_abs_path.is_file():
                raise FileNotFoundError(f"Missing raw prediction file: {raw_abs_path}")

            for key in ("depth_path", "mask_path"):
                if rec.get(key) and not Path(rec[key]).is_absolute():
                    rec[key] = str(root / rec[key])
            raw_prediction = np.load(raw_abs_path, allow_pickle=False)
            target_depth, valid_mask = _read_depth_and_mask(
                dataset_name, rec["depth_path"], rec.get("mask_path", "")
            )

            if raw_prediction.shape != target_depth.shape:
                raise ValueError(
                    f"Prediction/GT shape mismatch for {rec['sample_id']}: "
                    f"{raw_prediction.shape} vs {target_depth.shape}"
                )

            valid_pixels = int(valid_mask.sum())
            forward_ms_val = float(rec["forward_ms"])
            all_forward_timings_ms.append(forward_ms_val)

            row = {
                "dataset": dataset_name,
                "sample_id": rec["sample_id"],
                "image_path": rec["image_path"],
                "depth_path": rec["depth_path"],
                "raw_prediction_path": str(raw_rel_path) if keep_raw_predictions else "",
                "height": int(rec["height"]),
                "width": int(rec["width"]),
                "input_height": int(rec["input_height"]) if "input_height" in rec else "",
                "input_width": int(rec["input_width"]) if "input_width" in rec else "",
                "valid_pixels": valid_pixels,
                "forward_ms": forward_ms_val,
                "forward_timing_kind": rec.get("forward_timing_kind", "single_image"),
                "alignment_scale": "",
                "alignment_shift": "",
                **{name: "" for name in METRIC_NAMES},
                "status": "ok",
                "error": "",
            }

            if valid_pixels < 10:
                row["status"] = "skipped"
                row["error"] = f"Only {valid_pixels} valid pixels"
            else:
                try:
                    if prediction_type == "metric depth":
                        # 1. 度量深度（如 GLPN-NYU）：直接截断至协议允许的最小/最大距离
                        metric_depth = np.clip(
                            raw_prediction.astype(np.float64),
                            protocol["min_depth"],
                            protocol["max_depth"],
                        )
                        row["alignment_scale"] = 1.0
                        row["alignment_shift"] = 0.0
                        row.update(_compute_metrics(metric_depth, target_depth, valid_mask))
                    elif prediction_type == "scale-and-shift-invariant depth":
                        # 2. 正深度空间尺度与偏移不变深度（如 DA3 系列）：最小二乘 scale + shift 对齐
                        aligned_depth, scale, shift = _align_relative_depth(
                            raw_prediction,
                            target_depth,
                            valid_mask,
                            protocol["min_depth"],
                            protocol["max_depth"],
                        )
                        row["alignment_scale"] = scale
                        row["alignment_shift"] = shift
                        row.update(_compute_metrics(aligned_depth, target_depth, valid_mask))
                    elif prediction_type == "affine-invariant inverse depth":
                        # 3. 逆深度空间仿射不变深度（如 DepthCrafter / Depth-Anything-V2）：逆深度最小二乘对齐
                        aligned_depth, scale, shift = _align_inverse_depth(
                            raw_prediction,
                            target_depth,
                            valid_mask,
                            protocol["min_depth"],
                            protocol["max_depth"],
                        )
                        row["alignment_scale"] = scale
                        row["alignment_shift"] = shift
                        row.update(_compute_metrics(aligned_depth, target_depth, valid_mask))
                    else:
                        raise ValueError(
                            f"Unsupported prediction_type: '{prediction_type}'. "
                            f"Please add corresponding alignment logic in evaluate_metrics.py."
                        )
                except (FloatingPointError, ValueError) as error:
                    row["status"] = "error"
                    row["error"] = str(error)

            rows.append(row)
            all_rows.append(row)

            if not keep_raw_predictions:
                raw_abs_path.unlink(missing_ok=True)

            if idx % 25 == 0 or idx == len(records):
                print(f"  {idx}/{len(records)}")

        # 保存逐图指标 CSV 与数据集精度汇总日志（位于阶段3计时内）
        _write_csv(dataset_dir / "per_image_metrics.csv", rows)
        metrics_mean = _mean_metrics(rows)
        print(json.dumps(metrics_mean, ensure_ascii=False, indent=2))

        stage3_dataset_seconds = time.perf_counter() - stage3_dataset_start

        num_samples = len(rows)
        successful_count = sum(row["status"] == "ok" for row in rows)
        skipped_count = sum(row["status"] == "skipped" for row in rows)
        failed_count = num_samples - successful_count - skipped_count

        # 读取阶段1计时字典，并补入阶段3精度评测状态与合成后的两阶段总耗时
        eval_info = dict(inference_summary["datasets"][dataset_name])
        stage1_sec = eval_info.get("stage1_dataset_seconds")
        combined_e2e_sec = (
            (stage1_sec + stage3_dataset_seconds) if stage1_sec is not None else None
        )
        eval_info["accuracy_status"] = "completed"
        eval_info["stage3_accuracy_seconds"] = stage3_dataset_seconds
        eval_info["dataset_e2e_seconds"] = combined_e2e_sec
        eval_info["e2e_completed_images"] = num_samples
        eval_info["dataset_e2e_fps"] = (
            num_samples / combined_e2e_sec
            if combined_e2e_sec and combined_e2e_sec > 0 else None
        )

        dataset_summary = {
            "complete": True,
            "evaluation_scope": config.get("evaluation_scope", "full_validation" if max_samples is None else f"subset_{max_samples}"),
            "completed_at": datetime.now().astimezone().isoformat(),
            "dataset": dataset_name,
            "sample_count": num_samples,
            "successful_count": successful_count,
            "skipped_count": skipped_count,
            "failed_count": failed_count,
            "protocol": protocol,
            "metrics": metrics_mean,
            "evaluation": eval_info,
        }
        dataset_summaries[dataset_name] = dataset_summary

        (dataset_dir / "summary.json").write_text(
            json.dumps(_readable_timing_fields(dataset_summary), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    stage3_global_seconds = time.perf_counter() - stage3_global_start

    total_samples = len(all_rows)
    total_successful = sum(row["status"] == "ok" for row in all_rows)
    total_skipped = sum(row["status"] == "skipped" for row in all_rows)
    total_failed = sum(row["status"] == "error" for row in all_rows)

    # 遵照 v2.0 标准 6.3 节：若合并全局前向统计，从全部逐图耗时重算，不直接平均各数据集 P95 或 FPS
    overall_eval = dict(inference_summary["overall"])
    overall_eval["stage3_run_wall_seconds"] = stage3_global_seconds
    if all_forward_timings_ms:
        overall_mean_fw = float(np.mean(all_forward_timings_ms))
        overall_eval["total_forward_seconds"] = float(np.sum(all_forward_timings_ms) / 1000.0)
        overall_eval["mean_forward_ms"] = overall_mean_fw
        overall_eval["median_forward_ms"] = float(np.median(all_forward_timings_ms))
        overall_eval["p95_forward_ms"] = float(np.percentile(all_forward_timings_ms, 95))
        overall_eval["forward_fps"] = 1000.0 / overall_mean_fw if overall_mean_fw > 0 else None

    final_summary = {
        "complete": True,
        "completed_at": datetime.now().astimezone().isoformat(),
        "datasets": dataset_summaries,
        "overall": {
            "sample_count": total_samples,
            "successful_count": total_successful,
            "skipped_count": total_skipped,
            "failed_count": total_failed,
            "metrics": _mean_metrics(all_rows),
            "evaluation": overall_eval,
        },
    }

    # 保存 summary.csv（字段完全对齐 v2.0 双计时标准）
    summary_rows = []
    for name, summary in dataset_summaries.items():
        ev = summary["evaluation"]
        summary_rows.append(
            {
                "dataset": name,
                "sample_count": summary["sample_count"],
                "successful_count": summary["successful_count"],
                "skipped_count": summary["skipped_count"],
                "failed_count": summary["failed_count"],
                **summary["metrics"],
                "mean_forward_ms": ev["mean_forward_ms"],
                "forward_ms_per_image": ev["mean_forward_ms"],
                "forward_images_per_second": ev["forward_fps"],
                "median_forward_ms": ev["median_forward_ms"],
                "p95_forward_ms": ev["p95_forward_ms"],
                "forward_fps": ev["forward_fps"],
                "stage1_dataset_seconds": ev["stage1_dataset_seconds"],
                "stage1_dataset_fps": ev["stage1_dataset_fps"],
                "dataset_e2e_seconds": ev["dataset_e2e_seconds"],
                "dataset_e2e_fps": ev["dataset_e2e_fps"],
            }
        )
    _write_csv(result_dir / "summary.csv", summary_rows)

    (result_dir / "summary.json").write_text(
        json.dumps(_readable_timing_fields(final_summary), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"\n[Stage 3 Complete] Final evaluation summary saved to: {result_dir.resolve()}")
    return final_summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Unified Depth Estimation Metrics Evaluator (v2.0 Timing Compatible)")
    parser.add_argument(
        "--result-dir", "--run-dir", "--result_dir", dest="result_dir",
        type=str,
        required=True,
        help="Path (relative or absolute) to the timestamped result directory generated by run_inference.py",
    )
    parser.add_argument(
        "--delete_raw",
        action="store_true",
        help="Delete raw .npy prediction files after computing metrics to save disk space",
    )
    parser.add_argument("--dataset-root", help="可选：重新指定相对GT路径的根目录")
    args = parser.parse_args()
    if not Path(args.result_dir).is_absolute():
        args.result_dir = str(Path(__file__).resolve().parent / args.result_dir)
    if args.dataset_root and not Path(args.dataset_root).is_absolute():
        args.dataset_root = str(Path(__file__).resolve().parent / args.dataset_root)
    evaluate_from_predictions(args.result_dir, keep_raw_predictions=not args.delete_raw, dataset_root=args.dataset_root)