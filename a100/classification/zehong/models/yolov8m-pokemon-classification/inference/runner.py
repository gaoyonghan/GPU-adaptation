"""前向、设备计时与结果写出；指标由任务 evaluate.py 计算。"""
import csv
import json
import platform
import sys
import time
from importlib import metadata
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from .data import ClassImageDataset
from .protocol import write_bundle, write_json

PACKAGES = ('torch', 'torchvision', 'ultralytics', 'numpy', 'Pillow', 'tqdm')


def _version(name):
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def _device_info(device):
    info = {'device': str(device)}
    if device.type == 'cuda':
        props = torch.cuda.get_device_properties(device)
        info.update(name=props.name, total_memory_bytes=props.total_memory,
                    capability='%d.%d' % (props.major, props.minor), runtime=torch.version.cuda)
    elif device.type == 'npu':
        info.update(name=torch.npu.get_device_name(device),
                    ASCEND_HOME_PATH=__import__('os').environ.get('ASCEND_HOME_PATH'))
    return info


def _safe_peak(api, device, reserved):
    try:
        return api.max_memory_reserved(device) if reserved else api.max_memory_allocated(device)
    except Exception:
        return None


def run(net, ckpt, transform, imgsz, device, output_dir, *, dataset_root, split,
        dataset_name, batch_size, num_workers, warmup_steps, max_samples, model_info):
    names = {int(k): str(v) for k, v in ckpt['model'].names.items()}
    dataset = ClassImageDataset(dataset_root, split, names, transform, limit=max_samples)
    categories = [{'id': index, 'name': names[index]} for index in sorted(names)]
    annotation = {'categories': categories,
                  'samples': [{'sample_id': s.sample_id, 'label': s.label} for s in dataset.samples]}
    scope = 'debug_subset' if max_samples is not None else 'full_split'
    dataset_info = {'name': dataset_name, 'root': str(dataset_root), 'split': split,
                    'class_count': len(names), 'image_count': len(dataset.samples),
                    'selection': 'sorted_relative_path',
                    'excluded_classes': dataset.excluded,
                    'excluded_image_count': sum(row['images'] for row in dataset.excluded)}

    loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False,
                        drop_last=False, pin_memory=False,
                        persistent_workers=num_workers > 0,
                        multiprocessing_context='spawn' if num_workers else None)
    warm_images = next(iter(DataLoader(dataset, batch_size=min(batch_size, len(dataset)),
                                       num_workers=0)))[0].to(device)
    api = torch.cuda if device.type == 'cuda' else torch.npu
    net.eval()
    with torch.inference_mode():
        for _ in range(warmup_steps):
            net(warm_images)
    api.synchronize(device)
    del warm_images
    api.reset_peak_memory_stats(device)

    rows, batch_rows = [], []
    forward_seconds = 0.0
    start_event, end_event = api.Event(enable_timing=True), api.Event(enable_timing=True)
    wall_start = time.perf_counter()
    with torch.inference_mode(), (output_dir / 'batch_timings.csv').open('w', newline='', encoding='utf-8') as timing:
        writer = csv.DictWriter(timing, ['batch_index', 'batch_size', 'forward_ms'])
        writer.writeheader()
        for batch_index, (images, indices) in enumerate(tqdm(loader, desc='Inference', unit='batch')):
            images = images.to(device, non_blocking=False)
            start_event.record()
            logits = net(images)
            end_event.record()
            end_event.synchronize()
            forward_ms = start_event.elapsed_time(end_event)
            if logits.ndim != 2 or logits.shape[0] != len(indices) or logits.shape[1] != len(names):
                raise ValueError('分类前向须返回 [B, num_classes] logits')
            if not torch.isfinite(logits).all():
                raise ValueError('前向输出存在非有限值')
            probs = torch.softmax(logits.float(), dim=1)
            ranked = torch.argsort(probs, dim=1, descending=True, stable=True)[:, :min(5, len(names))]
            ranked_scores = probs.gather(1, ranked)
            for row, index in enumerate(indices.tolist()):
                rows.append({'sample_id': dataset.samples[index].sample_id,
                             'prediction': int(ranked[row, 0]),
                             'topk': [int(v) for v in ranked[row].tolist()],
                             'scores': [float(v) for v in ranked_scores[row].tolist()]})
            batch_rows.append({'batch_index': batch_index, 'batch_size': len(indices),
                               'forward_ms': forward_ms})
            writer.writerow(batch_rows[-1])
            forward_seconds += forward_ms / 1000.0
    api.synchronize(device)
    wall_seconds = time.perf_counter() - wall_start

    rows.sort(key=lambda row: row['sample_id'])
    if len(rows) != len(dataset.samples):
        raise RuntimeError(f'推理未完整完成：{len(rows)}/{len(dataset.samples)}')
    completed = {row['sample_id'] for row in rows}
    if completed != {s['sample_id'] for s in annotation['samples']}:
        raise RuntimeError('预测与标注样本集合不一致')

    native_execution = [{'device': str(device), 'dtype': 'torch.float32', 'imgsz': imgsz,
                         'batch_size': batch_size, 'num_workers': num_workers,
                         'warmup_steps': warmup_steps}]
    limitations = ['单卡同步推理，未启用AMP/量化/编译优化',
                   '本run仅用于跨设备数值一致性对比，不代表官方测试集验收']
    if dataset.excluded:
        limitations.append('数据集中%d个类别不在模型类别空间，已排除%d张图片'
                           % (len(dataset.excluded), dataset_info['excluded_image_count']))
    if max_samples is not None:
        limitations.append('调试子集，非完整划分')
    bundle = write_bundle(output_dir, annotation, rows, model_info=model_info,
                          dataset_info=dataset_info, scope=scope,
                          native_execution=native_execution, limitations=limitations)

    forward_ms = [row['forward_ms'] for row in batch_rows]
    summary = {
        'status': 'completed',
        'model': model_info,
        'dataset': dataset_info,
        'scope': scope,
        'sample_count': len(rows),
        'class_count': len(names),
        'imgsz': imgsz,
        'batch_size': batch_size,
        'num_workers': num_workers,
        'warmup_steps': warmup_steps,
        'device': str(device),
        'parameter_dtype': str(next(net.parameters()).dtype),
        'transform': repr(transform),
        'performance': {
            'forward_seconds': forward_seconds,
            'forward_images_per_second': len(rows) / forward_seconds,
            'forward_mean_ms_per_image': forward_seconds * 1000 / len(rows),
            'forward_ms_per_image': forward_seconds * 1000 / len(rows),
            'forward_batch_median_ms': float(np.median(forward_ms)),
            'forward_batch_p95_ms': float(np.percentile(forward_ms, 95)),
            'end_to_end_seconds': wall_seconds,
            'end_to_end_images_per_second': len(rows) / wall_seconds,
        },
        'memory': {'peak_allocated_bytes': _safe_peak(api, device, False),
                   'peak_reserved_bytes': _safe_peak(api, device, True)},
        'environment': {'python': sys.version.split()[0],
                        'packages': {name: _version(name) for name in PACKAGES},
                        'platform': platform.platform(), 'hostname': platform.node(),
                        'torch_num_threads': torch.get_num_threads()},
        'device_info': _device_info(device),
        'bundle': bundle.name,
    }
    # Top-level timing aliases match other classification inference summaries.
    summary.update(summary['performance'])
    summary.update(summary['memory'])
    write_json(output_dir / 'inference_summary.json', summary)
    (output_dir / 'model_structure.txt').write_text(
        'Parameters: %d\n\n%s' % (sum(p.numel() for p in net.parameters()), net), encoding='utf-8')
    write_json(output_dir / 'model_environment.json',
               {'device': summary['device_info'], 'environment': summary['environment'], 'model': model_info})
    write_json(output_dir / 'inference_output.json',
               {'state': 'completed', 'task': 'classification', 'bundles': [bundle.name],
                'scoring_status': 'pending'})
    return summary
