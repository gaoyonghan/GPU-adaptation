TIMER=None
# 文件路径: /data/lijianxu/ascend/JinPeng/depth-estimation/glpn-nyu/utils/inference_model.py
import csv
import json
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

# 尝试导入 torch_npu 注册昇腾后端
try:
    import torch_npu
except ImportError:
    pass

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

def _device_synchronize(device):
    """统一同步逻辑，确保异步硬件指令执行完毕"""
    if device.type == "npu":
        torch.npu.synchronize(device)
    elif device.type == "cuda":
        torch.cuda.synchronize(device)


def _discover_samples(name, root):
    if name == "KITTI":
        samples = []
        for depth_path in sorted(root.glob("*_drive_*_sync/proj_depth/groundtruth/image_02/*.png")):
            drive = depth_path.parents[3].name
            date = drive.split("_drive_", 1)[0]
            image_path = root / date / drive / "image_02" / "data" / depth_path.name
            if not image_path.is_file():
                raise FileNotFoundError(f"Missing KITTI RGB image: {image_path}")
            samples.append(
                {
                    "id": f"{drive}/{depth_path.stem}",
                    "image": image_path,
                    "depth": depth_path,
                }
            )
        return samples

    if name == "NYUv2":
        list_path = root / "filename_list_test.txt"
        if not list_path.is_file():
            raise FileNotFoundError(f"Missing NYUv2 test list: {list_path}")
        samples = []
        for line in list_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            image_rel, depth_rel, _ = line.split()
            image_path = root / image_rel
            depth_path = root / depth_rel
            if not image_path.is_file() or not depth_path.is_file():
                raise FileNotFoundError(f"Missing NYUv2 pair: {image_path}, {depth_path}")
            samples.append(
                {
                    "id": str(Path(image_rel).with_suffix("")),
                    "image": image_path,
                    "depth": depth_path,
                }
            )
        return samples

    if name == "DIODE":
        samples = []
        for image_path in sorted((root / "val").glob("**/*.png")):
            depth_path = image_path.with_name(f"{image_path.stem}_depth.npy")
            mask_path = image_path.with_name(f"{image_path.stem}_depth_mask.npy")
            if not depth_path.is_file() or not mask_path.is_file():
                raise FileNotFoundError(
                    f"Missing DIODE depth or mask: {depth_path}, {mask_path}"
                )
            samples.append(
                {
                    "id": str(image_path.relative_to(root).with_suffix("")),
                    "image": image_path,
                    "depth": depth_path,
                    "mask": mask_path,
                }
            )
        return samples

    if name == "Sintel":
        samples = []
        image_root = root / "training" / "final"
        depth_root = root / "training" / "depth"
        for image_path in sorted(image_root.glob("**/*.png")):
            relative_path = image_path.relative_to(image_root)
            depth_path = (depth_root / relative_path).with_suffix(".dpt")
            if not depth_path.is_file():
                raise FileNotFoundError(f"Missing Sintel depth: {depth_path}")
            samples.append(
                {
                    "id": str(relative_path.with_suffix("")),
                    "image": image_path,
                    "depth": depth_path,
                }
            )
        return samples

    raise ValueError(f"Unsupported dataset: {name}")


def _kitti_benchmark_crop(array):
    height, width = array.shape[:2]
    crop_height, crop_width = 352, 1216
    if height < crop_height or width < crop_width:
        raise ValueError(f"KITTI image is smaller than benchmark crop: {array.shape}")
    top = height - crop_height
    left = (width - crop_width) // 2
    return array[top : top + crop_height, left : left + crop_width]


class _DepthInferenceDataset(Dataset):
    """纯推理数据集：只负责读取 RGB 图像并做 GLPN 专属预处理，只获取目标尺寸 output_shape，不加载真值掩码"""
    def __init__(self, name, root, input_size, max_samples=None):
        self.name = name
        self.root = Path(root)
        if not self.root.is_dir():
            raise FileNotFoundError(f"Dataset directory does not exist: {self.root}")
        self.samples = _discover_samples(name, self.root)
        if max_samples is not None:
            self.samples = self.samples[:max_samples]
        if not self.samples:
            raise RuntimeError(f"No evaluation samples found for {name} under {self.root}")
        
        # 保证 input_size 为 32 的倍数以适配 GLPN 的分层 Transformer 结构
        self.input_size = int(round(input_size / 32) * 32)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        image = cv2.imread(str(sample["image"]), cv2.IMREAD_COLOR)
        if image is None:
            raise OSError(f"Unable to read image: {sample['image']}")
        if self.name == "KITTI":
            image = _kitti_benchmark_crop(image)

        output_shape = image.shape[:2]  # (H, W)，直接以裁剪后原图尺寸作为插值目标尺寸

        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        
        # 保持比例缩放，并保证高宽为 32 的整数倍
        h, w = rgb.shape[:2]
        scale = self.input_size / min(h, w)
        new_h = int(round(h * scale / 32) * 32)
        new_w = int(round(w * scale / 32) * 32)
        resized_rgb = cv2.resize(rgb, (new_w, new_h), interpolation=cv2.INTER_CUBIC)

        # ImageNet 标准归一化
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        normalized = (resized_rgb - mean) / std
        model_input = np.transpose(normalized, (2, 0, 1))

        return {
            "id": sample["id"],
            "image_path": str(sample["image"]),
            "depth_path": str(sample["depth"]),
            "mask_path": str(sample["mask"]) if "mask" in sample else "",
            "model_input": torch.from_numpy(model_input),
            "output_shape": output_shape,
        }


def _collate_as_list(samples):
    return samples


def _write_csv(path, rows):
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _max_samples_for_dataset(max_samples, dataset_name):
    if max_samples is None or isinstance(max_samples, int):
        return max_samples
    return max_samples.get(dataset_name)


def run_inference(
    model,
    datasets=("KITTI", "NYUv2", "DIODE", "Sintel"),
    # dataset_root="/data/lijianxu/datasets",
    # 修改点
    dataset_root=str(Path(__file__).resolve().parents[5] / 'dataset'),
    output_dir="results",
    batch_size=1,
    num_workers=4,
    input_size=480,
    warmup_runs=3,
    max_samples=None,
):
    """阶段1：仅执行模型前向推理、统计延迟/显存指标，并将原始预测深度图保存为中间文件。"""
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")
    if num_workers < 0:
        raise ValueError("num_workers cannot be negative")
    if warmup_runs < 0:
        raise ValueError("warmup_runs cannot be negative")

    dataset_names = [datasets] if isinstance(datasets, str) else list(datasets)
    unknown = sorted(set(dataset_names) - set(DATASET_PROTOCOLS))
    if unknown:
        raise ValueError(f"Unsupported datasets: {', '.join(unknown)}")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_root = Path(dataset_root)
    
    device = next(model.parameters()).device
    is_accelerator = device.type in {"cuda", "npu"}
    model.eval()

    if device.type == "npu":
        timing_backend = "npu_events_network_sum"
    elif device.type == "cuda":
        timing_backend = "cuda_events_network_sum"
    else:
        timing_backend = "perf_counter"

    config = {
        "datasets": dataset_names,
        "dataset_root": str(dataset_root.resolve()),
        "output_dir": str(output_dir.resolve()),
        "batch_size": batch_size,
        "num_workers": num_workers,
        "input_size": input_size,
        "warmup_runs": warmup_runs,
        "max_samples": max_samples,
        "save_raw_predictions": True,
        "device": str(device),
        "prediction_type": "metric depth",
        "alignment": "none (direct metric evaluation)",
        "timing": {
            "full_standard_compliance": "pending_verification",
            "forward_entry": "model(inputs)",
            "forward_scope": (
                "host wall time of model(inputs), including end synchronization; "
                "excluding context entry/exit, pre-sync, preprocessing, H2D, "
                "warmup, independent output processing and D2H"
            ),
            "dataset_scope": (
                "stage1 only: dataset discovery/build through raw prediction "
                "and per-image CSV saving and normal completion logging"
            ),
            "full_dataset_e2e_status": "not_measured",
            "warmup_policy": "once_per_run_on_first_group",
            "forward_ms_semantics": (
                "group forward wall time divided by actual group size"
            ),
            "p95_algorithm": "np.percentile(forward_timings_ms, 95)",
            "numpy_version": np.__version__,
            "torch_version": torch.__version__,
            "torch_npu_version": (
                getattr(torch_npu, "__version__", "unknown")
                if device.type == "npu" else None
            ),
            "synchronization_verification": "pending_on_target_runtime",
        },
        "aggregation": "arithmetic mean of per-image metrics",
        "protocols": {name: DATASET_PROTOCOLS[name] for name in dataset_names},
    }
    (output_dir / "evaluation_config.json").write_text(
        json.dumps(_readable_timing_fields(config), ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    dataset_evaluations = {}
    total_samples = 0

    global_wall_start = time.perf_counter()
    global_peak_npu_memory = 0
    global_peak_npu_reserved = 0
    global_peak_cuda_memory = 0

    warmed_up = False

    for dataset_name in dataset_names:
        _device_synchronize(device)
        dataset_start = time.perf_counter()
        warmup_elapsed=0.0

        dataset = _DepthInferenceDataset(
            dataset_name,
            dataset_root / dataset_name,
            input_size,
            _max_samples_for_dataset(max_samples, dataset_name),
        )
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=is_accelerator,
            persistent_workers=num_workers > 0,
            collate_fn=_collate_as_list,
        )
        dataset_output_dir = output_dir / dataset_name
        raw_output_dir = dataset_output_dir / "raw_predictions"
        raw_output_dir.mkdir(parents=True, exist_ok=True)
            
        records = []
        processed = 0
        forward_timings_ms = []
        total_model_inference_seconds = 0.0

        if device.type == "npu":
            torch.npu.reset_peak_memory_stats(device)
        elif device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)

        print(f"\n[Inference] Running {dataset_name}: {len(dataset)} images")

        for samples in loader:
            shape_groups = defaultdict(list)
            for sample in samples:
                key = (tuple(sample["model_input"].shape), tuple(sample["output_shape"]))
                shape_groups[key].append(sample)

            for (_, output_shape), group in shape_groups.items():
                inputs = torch.stack([sample["model_input"] for sample in group]).to(
                    device, non_blocking=is_accelerator
                )
                
                if not warmed_up and warmup_runs:
                    warmup_start=time.perf_counter()
                    with torch.inference_mode():
                        for _ in range(warmup_runs):
                            warmup_prediction = model(inputs)
                            if warmup_prediction.dim() == 3:
                                warmup_prediction = warmup_prediction[:, None]
                            F.interpolate(
                                warmup_prediction,
                                output_shape,
                                mode="bilinear",
                                align_corners=False,
                            )
                    _device_synchronize(device)
                    warmed_up = True
                    warmup_elapsed += time.perf_counter()-warmup_start

                with torch.inference_mode():
                    _device_synchronize(device)
                    TIMER.reset()
                    forward_start = time.perf_counter()
                    predictions = model(inputs)
                    _device_synchronize(device)
                    forward_end = time.perf_counter()

                    # 在输出处理或保存可能失败之前，保留正式前向记录。
                    forward_seconds = TIMER.seconds()
                    total_model_inference_seconds += forward_seconds
                    forward_ms_per_image = (
                        forward_seconds * 1000.0 / len(group)
                    )
                    forward_timings_ms.extend(
                        [forward_ms_per_image] * len(group)
                    )

                    # 独立输出处理：在前向计时之外。
                    if predictions.dim() == 3:
                        predictions = predictions[:, None]
                    predictions = F.interpolate(
                        predictions,
                        output_shape,
                        mode="bilinear",
                        align_corners=False,
                    )[:, 0]
                    predictions = predictions.float().cpu().numpy()

                for sample, raw_prediction in zip(group, predictions):
                    rel_raw_path = Path(dataset_name) / "raw_predictions" / f"{sample['id']}.npy"
                    abs_raw_path = output_dir / rel_raw_path
                    abs_raw_path.parent.mkdir(parents=True, exist_ok=True)
                    np.save(abs_raw_path, raw_prediction.astype(np.float32, copy=False))

                    records.append({
                        "dataset": dataset_name,
                        "sample_id": sample["id"],
                        "image_path": sample["image_path"],
                        "depth_path": sample["depth_path"],
                        "mask_path": sample["mask_path"],
                        "raw_prediction_path": str(rel_raw_path),
                        "height": sample["output_shape"][0],
                        "width": sample["output_shape"][1],
                        "input_height": int(inputs.shape[-2]),
                        "input_width": int(inputs.shape[-1]),
                        "forward_ms": forward_ms_per_image,
                        "forward_timing_kind": (
                            "single_image"
                            if len(group) == 1
                            else "batch_amortized"
                        ),
                    })
                    processed += 1

            if processed % 25 < len(samples) or processed == len(dataset):
                print(f"  {processed}/{len(dataset)}")

        # 普通逐图结果保存和正常完成日志必须在停止计时之前。
        _write_csv(dataset_output_dir / "inference_records.csv", records)

        num_samples = len(records)
        total_samples += num_samples

        print(
            f"[Inference] {dataset_name} complete: "
            f"{num_samples}/{len(dataset)} images; records saved"
        )

        # 当前 np.save / CSV 写入均直接执行，没有应用异步保存任务。
        # 若以后引入异步保存，必须在这里等待全部任务完成。
        _device_synchronize(device)
        dataset_end = time.perf_counter()

        # 当前范围仅覆盖阶段 1，不能冒充包含精度评测的完整 E2E。
        stage1_dataset_seconds = dataset_end - dataset_start - warmup_elapsed

        # 最终计时统计在停止计时后计算。
        if forward_timings_ms:
            mean_forward_ms = float(np.mean(forward_timings_ms))
            median_forward_ms = float(np.median(forward_timings_ms))
            p95_forward_ms = float(
                np.percentile(forward_timings_ms, 95)
            )
            forward_fps = (
                1000.0 / mean_forward_ms
                if mean_forward_ms > 0 else None
            )
        else:
            mean_forward_ms = None
            median_forward_ms = None
            p95_forward_ms = None
            forward_fps = None

        peak_npu = (
            torch.npu.max_memory_allocated(device)
            if device.type == "npu" else None
        )
        peak_npu_res = (
            torch.npu.max_memory_reserved(device)
            if device.type == "npu" else None
        )
        peak_cuda = (
            torch.cuda.max_memory_allocated(device)
            if device.type == "cuda" else None
        )

        if peak_npu is not None:
            global_peak_npu_memory = max(
                global_peak_npu_memory, peak_npu
            )
        if peak_npu_res is not None:
            global_peak_npu_reserved = max(
                global_peak_npu_reserved, peak_npu_res
            )
        if peak_cuda is not None:
            global_peak_cuda_memory = max(
                global_peak_cuda_memory, peak_cuda
            )

        evaluation_data = {
            "batch_size": batch_size,
            "input_size": input_size,
            "warmup_runs": warmup_runs,
            "warmup_policy": "once_per_run_on_first_group",
            "device": str(device),
            "num_samples": len(dataset),
            "forward_recorded_images": len(forward_timings_ms),
            "stage1_completed_images": num_samples,
            "stage1_status": "completed",
            "accuracy_status": "not_run",
            "total_forward_seconds": total_model_inference_seconds,
            "mean_forward_ms": mean_forward_ms,
            "median_forward_ms": median_forward_ms,
            "p95_forward_ms": p95_forward_ms,
            "forward_fps": forward_fps,
            "stage1_dataset_seconds": stage1_dataset_seconds,
            "stage1_dataset_fps": (
                num_samples / stage1_dataset_seconds
                if stage1_dataset_seconds > 0 else None
            ),
            # 等完整评测流程纳入计时后才能填入。
            "dataset_e2e_seconds": None,
            "e2e_completed_images": None,
            "dataset_e2e_fps": None,
            "timing_backend": timing_backend,
            "peak_npu_memory_bytes": peak_npu,
            "peak_npu_reserved_bytes": peak_npu_res,
            "peak_cuda_memory_bytes": peak_cuda,
        }
        dataset_evaluations[dataset_name] = evaluation_data

        # 最终计时报告输出位于计时之外。
        print(json.dumps(_readable_timing_fields(evaluation_data), ensure_ascii=False, indent=2))

    global_wall_seconds = time.perf_counter() - global_wall_start
    overall_evaluation = {
        "stage1_completed_images": total_samples,
        "stage1_run_wall_seconds_before_final_report": global_wall_seconds,
        "timing_backend": timing_backend,
        "peak_npu_memory_bytes": global_peak_npu_memory or None,
        "peak_npu_reserved_bytes": global_peak_npu_reserved or None,
        "peak_cuda_memory_bytes": global_peak_cuda_memory or None,
    }

    inference_summary = {
        "completed_at": datetime.now().astimezone().isoformat(),
        "datasets": dataset_evaluations,
        "overall": overall_evaluation,
    }
    (output_dir / "inference_summary.json").write_text(
        json.dumps(_readable_timing_fields(inference_summary), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"\n[Stage 1 Complete] Raw predictions and timing saved to: {output_dir.resolve()}")
    return inference_summary