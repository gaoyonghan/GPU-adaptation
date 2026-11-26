"""从本地文件评测 cafeai/cafe_style，记录完整五类风格分类指标。

数据清单没有独立 test split；评测全部本地图片，不创建随机划分。
运行：python code/cafe-style-cafe-style-classsfication-test.py
可通过 BATCH_SIZE、DEVICE 环境变量调整运行参数。
"""
if __name__ == '__main__':
    import sys as _records_sys
    from pathlib import Path as _RecordsPath
    _records_sys.path.insert(0, str(_RecordsPath(__file__).resolve().parent))
    _records_sys.path.insert(1, str(_RecordsPath(__file__).resolve().parents[1]))
    from common import start as _start_records
    _test_record = _start_records(137, __file__, globals())
import os
import sys
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['PYTHONUNBUFFERED'] = '1'
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(line_buffering=True, write_through=True)
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(line_buffering=True, write_through=True)
import csv
import hashlib
import json
from dataclasses import asdict, is_dataclass
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo
import torch
import torch.nn.functional as F
import transformers
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from transformers import AutoConfig, AutoImageProcessor, AutoModelForImageClassification
ROOT = __import__('model_paths').resolve_paths()['project']
MODEL_ID = 'cafeai/cafe_style'
MODEL_DIR = __import__('model_paths').resolve_paths()['weights']
DATASET_DIR = __import__('model_paths').resolve_paths()['dataset'] / 'cafe_style_classsfication'
MANIFEST = DATASET_DIR / 'manifest.csv'
RESULTS_PATH = _test_record.output / 'metrics.json'
if __name__ == '__main__':
    RESULTS_PATH = _test_record.output / 'metrics.json'
BATCH_SIZE = int(os.environ.get('MODEL_BATCH_SIZE', '16'))
NUM_WORKERS = int(os.environ.get('MODEL_WORKERS', '0'))
WARMUP_STEPS = int(os.environ.get('MODEL_WARMUP', '3'))
EXPECTED_LABELS = ['anime', 'real_life', '3d', 'manga_like', 'other']
IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.webp', '.bmp', '.tif', '.tiff'}

def json_metadata_default(value):
    """Serialize library metadata without changing inference or preprocessing."""
    if isinstance(value, (set, frozenset)):
        return sorted(value)
    if is_dataclass(value) and (not isinstance(value, type)):
        return {key: item for (key, item) in asdict(value).items() if item is not None}
    raise TypeError(f'Object of type {type(value).__name__} is not JSON serializable')

def collect_records(label_to_id):
    (records, seen) = ([], set())
    with MANIFEST.open(encoding='utf-8-sig', newline='') as handle:
        reader = csv.DictReader(handle)
        if not {'file_path', 'label'}.issubset(reader.fieldnames or []):
            raise ValueError('manifest.csv 必须包含 file_path 和 label')
        for row in reader:
            label = row['label']
            if label not in label_to_id:
                raise ValueError(f'数据集类别不在模型中：{label}')
            path = (DATASET_DIR / row['file_path']).resolve()
            if not path.is_relative_to(DATASET_DIR.resolve()):
                raise ValueError(f'清单路径超出数据集：{path}')
            if path in seen or not path.is_file():
                raise ValueError(f'清单图片重复或缺失：{path}')
            if path.parent.name != label or path.suffix.lower() not in IMAGE_EXTENSIONS:
                raise ValueError(f'图片目录或扩展名与清单不一致：{path}')
            seen.add(path)
            records.append((path, label_to_id[label]))
    available = {path.resolve() for label in label_to_id for path in (DATASET_DIR / label).rglob('*') if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS}
    if available != seen:
        raise ValueError('数据集图片与 manifest.csv 不完全对应')
    counts = Counter((label for (_, label) in records))
    if any((counts[index] == 0 for index in label_to_id.values())):
        raise ValueError('数据集为空或有类别缺少样本')
    return (records, {name: counts[index] for (name, index) in label_to_id.items()})

class CafeStyleDataset(Dataset):

    def __init__(self, records, processor):
        self.records = records
        self.processor = processor

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        (path, label) = self.records[index]
        with Image.open(path) as image:
            pixels = self.processor(images=image.convert('RGB'), return_tensors='pt')
        return (pixels['pixel_values'].squeeze(0), label)

def calculate_metrics(confusion, top3_correct, loss_sum):
    support = confusion.sum(dim=1).double()
    predicted = confusion.sum(dim=0).double()
    tp = confusion.diag().double()
    precision = tp / predicted.clamp_min(1)
    recall = tp / support.clamp_min(1)
    f1 = 2 * tp / (support + predicted).clamp_min(1)
    count = int(support.sum().item())
    per_class = {name: {'support': int(support[index].item()), 'precision_pct': round(100 * precision[index].item(), 4), 'recall_pct': round(100 * recall[index].item(), 4), 'f1_pct': round(100 * f1[index].item(), 4)} for (index, name) in enumerate(EXPECTED_LABELS)}
    return {'image_count': count, 'top1_correct': int(tp.sum().item()), 'top1_accuracy_pct': round(100 * tp.sum().item() / count, 4), 'top3_accuracy_pct': round(100 * top3_correct / count, 4), 'macro_precision_pct': round(100 * precision.mean().item(), 4), 'macro_recall_pct': round(100 * recall.mean().item(), 4), 'balanced_accuracy_pct': round(100 * recall.mean().item(), 4), 'macro_f1_pct': round(100 * f1.mean().item(), 4), 'weighted_f1_pct': round(100 * (f1 * support).sum().item() / count, 4), 'cross_entropy_loss': round(loss_sum / count, 6), 'per_class': per_class, 'confusion_matrix': confusion.tolist(), 'confusion_matrix_labels': EXPECTED_LABELS, 'confusion_matrix_axes': {'rows': 'true', 'columns': 'predicted'}}

def main():
    if BATCH_SIZE < 1:
        raise ValueError('BATCH_SIZE 必须大于 0')
    torch.manual_seed(0)
    with _test_record.timer.loading():
        config = AutoConfig.from_pretrained(str(MODEL_DIR), local_files_only=True)
    labels = [config.id2label[index] for index in range(config.num_labels)]
    if labels != EXPECTED_LABELS or config.problem_type != 'single_label_classification':
        raise ValueError(f'模型任务或类别不匹配：{labels}, {config.problem_type}')
    label_to_id = {label: index for (index, label) in enumerate(labels)}
    original_label2id_count = len(config.label2id or {})
    label2id_corrected = config.label2id != label_to_id
    config.label2id = label_to_id
    (records, class_counts) = collect_records(label_to_id)
    with _test_record.timer.loading():
        processor = AutoImageProcessor.from_pretrained(str(MODEL_DIR), local_files_only=True, use_fast=False)
    with _test_record.timer.loading():
        (model, loading_info) = AutoModelForImageClassification.from_pretrained(str(MODEL_DIR), config=config, local_files_only=True, output_loading_info=True)
    if any((loading_info.get(key) for key in ('missing_keys', 'mismatched_keys', 'error_msgs'))):
        raise ValueError(f'模型权重加载不完整：{loading_info}')
    device = torch.device(os.environ.get('DEVICE', 'cuda' if torch.cuda.is_available() else 'cpu'))
    device_name = torch.cuda.get_device_name(device) if device.type == 'cuda' else 'CPU'
    with _test_record.timer.loading():
        model = model.to(device).eval()
    loader = _test_record.loader(DataLoader, CafeStyleDataset(records, processor), batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=device.type == 'cuda')
    print(f'模型: {MODEL_ID}; 设备: {device} ({device_name}); FP32; batch={BATCH_SIZE}')
    print(f'本地完整数据集: {len(records)} 张; 类别分布: {class_counts}')
    print('使用全部五类 logits；本地清单未标明独立 test split。')
    (warmup_pixels, _) = next(iter(loader))
    warmup_pixels = warmup_pixels.to(device)
    with torch.inference_mode():
        with _test_record.timer.warmup():
            for _ in range(WARMUP_STEPS):
                _test_record.timer.invoke(model, pixel_values=warmup_pixels)
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
    del warmup_pixels
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)
    class_count = len(labels)
    confusion = torch.zeros((class_count, class_count), dtype=torch.long)
    (top3_correct, loss_sum, forward_seconds) = (0, 0.0, 0.0)
    wall_start = time.perf_counter()
    with torch.inference_mode():
        for (batch_index, (pixels, targets)) in enumerate(loader, 1):
            _test_record.before_batch(locals())
            pixels = pixels.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            if device.type == 'cuda':
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record(torch.cuda.current_stream(device))
                logits = _test_record.timer.invoke(model, pixel_values=pixels).logits
                end.record(torch.cuda.current_stream(device))
                torch.cuda.synchronize(device)
                forward_seconds += start.elapsed_time(end) / 1000
            else:
                start_time = time.perf_counter()
                logits = _test_record.timer.invoke(model, pixel_values=pixels).logits
                forward_seconds += time.perf_counter() - start_time
            if logits.shape != (len(targets), class_count) or not torch.isfinite(logits).all():
                raise ValueError('模型输出维度异常或包含非有限数值')
            predictions = logits.argmax(dim=1)
            _test_record.prediction_ready(locals())
            top3_correct += logits.topk(3, dim=1).indices.eq(targets[:, None]).any(dim=1).sum().item()
            loss_sum += F.cross_entropy(logits, targets, reduction='sum').item()
            confusion += torch.bincount((targets * class_count + predictions).cpu(), minlength=class_count ** 2).reshape(class_count, class_count)
            print(f'已处理: {int(confusion.sum().item())}/{len(records)} ({batch_index}/{len(loader)})')
            _test_record.capture(locals())
    wall_seconds = time.perf_counter() - wall_start
    metrics = calculate_metrics(confusion, top3_correct, loss_sum)
    if metrics['image_count'] != len(records):
        raise ValueError('评测样本计数不一致')
    result = {'recorded_at': datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(timespec='seconds'), 'model_id': MODEL_ID, 'model_path': str(MODEL_DIR), 'script_path': str(Path(__file__).resolve()), 'model': {'id': MODEL_ID, 'architecture': model.__class__.__name__, 'parameter_count_millions': round(sum((p.numel() for p in model.parameters())) / 1000000.0, 4), 'num_labels': class_count, 'class_names': labels, 'label2id_corrected_in_memory': label2id_corrected, 'original_label2id_count': original_label2id_count, 'label_mapping_source': 'local config.json id2label', 'weight_loading_info': loading_info}, 'dataset': {'name': 'cafe_style_classsfication', 'source': 'local image directories and manifest.csv', 'path': str(DATASET_DIR), 'manifest': str(MANIFEST), 'manifest_sha256': hashlib.sha256(MANIFEST.read_bytes()).hexdigest(), 'split': 'full_local_dataset', 'image_count': len(records), 'class_count': class_count, 'class_counts': class_counts, 'label_mapping': label_to_id, 'split_note': '本地清单没有独立 test split；使用全部本地图片，训练数据重叠未知，不能视为独立测试集泛化指标。'}, 'evaluation': {'task': 'five-class image style classification', 'logit_scope': 'all five model classes', 'primary_metrics': ['top1_accuracy_pct', 'macro_f1_pct', 'balanced_accuracy_pct']}, 'environment': {'device': str(device), 'device_name': device_name, 'precision': 'FP32', 'batch_size': BATCH_SIZE, 'num_workers': NUM_WORKERS, 'warmup_steps': WARMUP_STEPS, 'python_executable': sys.executable, 'python_version': sys.version.split()[0], 'torch_version': torch.__version__, 'transformers_version': transformers.__version__, 'cuda_version': torch.version.cuda, 'offline': True}, 'preprocessing': {'processor': processor.__class__.__name__, 'resize': processor.size, 'do_center_crop': processor.do_center_crop, 'do_rescale': processor.do_rescale, 'rescale_factor': processor.rescale_factor, 'resample': int(processor.resample), 'image_mean': processor.image_mean, 'image_std': processor.image_std, 'image_mode': 'RGB'}, 'metrics': metrics, 'timing': {'model_forward_total_seconds': round(forward_seconds, 6), 'model_latency_ms_per_image': round(1000 * forward_seconds / len(records), 6), 'model_throughput_images_per_second': round(len(records) / forward_seconds, 4), 'end_to_end_total_seconds': round(wall_seconds, 6), 'end_to_end_latency_ms_per_image': round(1000 * wall_seconds / len(records), 6), 'end_to_end_throughput_images_per_second': round(len(records) / wall_seconds, 4), 'note': '预热和模型加载不计时；前向计时不含预处理及数据传输；端到端包含读图、预处理、数据传输和指标累计。'}}
    if device.type == 'cuda':
        result['memory'] = {'peak_allocated_mib': round(torch.cuda.max_memory_allocated(device) / 1024 ** 2, 4), 'peak_reserved_mib': round(torch.cuda.max_memory_reserved(device) / 1024 ** 2, 4)}
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    temporary = RESULTS_PATH.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(result, default=json_metadata_default, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    os.replace(temporary, RESULTS_PATH)
    print(json.dumps({key: value for (key, value) in metrics.items() if key not in ['per_class', 'confusion_matrix', 'confusion_matrix_labels', 'confusion_matrix_axes']}, indent=2))
    print(f'结果已保存: {RESULTS_PATH}')
    _test_record.finish(locals())
if __name__ == '__main__':
    main()
