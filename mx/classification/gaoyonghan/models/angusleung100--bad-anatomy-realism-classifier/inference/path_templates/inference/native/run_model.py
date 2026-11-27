"""在本地 anatomy-classifier 数据集上评测 bad-anatomy-realism-classifier。"""
if __name__ == '__main__':
    import sys as _records_sys
    from pathlib import Path as _RecordsPath
    _records_sys.path.insert(0, str(_RecordsPath(__file__).resolve().parent))
    _records_sys.path.insert(1, str(_RecordsPath(__file__).resolve().parents[1]))
    from common import start as _start_records
    _test_record = _start_records(144, __file__, globals())
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
from transformers import AutoModelForImageClassification, ViTImageProcessor
ROOT = __import__('model_paths').resolve_paths()['project']
MODEL_ID = 'angusleung100/bad-anatomy-realism-classifier'
MODEL_DIR = __import__('model_paths').resolve_paths()['weights']
DATASET_DIR = __import__('model_paths').resolve_paths()['dataset'] / 'anatomy-classifier'
MANIFEST_PATH = DATASET_DIR / 'manifest.csv'
RESULTS_PATH = _test_record.output / 'metrics.json'
if __name__ == '__main__':
    RESULTS_PATH = _test_record.output / 'metrics.json'
BATCH_SIZE = int(os.environ.get('MODEL_BATCH_SIZE', '16'))
NUM_WORKERS = int(os.environ.get('MODEL_WORKERS', '0'))
WARMUP_STEPS = int(os.environ.get('MODEL_WARMUP', '3'))
CLASS_NAMES = ['Realistic_Bad_Anatomy', 'Realistic_Good_Anatomy', 'Unrealistic_Bad_Anatomy', 'Unrealistic_Good_Anatomy']
IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.webp', '.bmp', '.tif', '.tiff'}

def json_metadata_default(value):
    """Serialize library metadata without changing inference or preprocessing."""
    if isinstance(value, (set, frozenset)):
        return sorted(value)
    if is_dataclass(value) and (not isinstance(value, type)):
        return {key: item for (key, item) in asdict(value).items() if item is not None}
    raise TypeError(f'Object of type {type(value).__name__} is not JSON serializable')

def collect_records():
    records = []
    seen = set()
    with MANIFEST_PATH.open(encoding='utf-8-sig', newline='') as handle:
        reader = csv.DictReader(handle)
        expected = {'path', 'label_id', 'label_name'}
        if not expected.issubset(reader.fieldnames or []):
            raise ValueError(f'manifest.csv 缺少字段：{sorted(expected - set(reader.fieldnames or []))}')
        for row in reader:
            label_id = int(row['label_id'])
            label_name = row['label_name']
            if not 0 <= label_id < len(CLASS_NAMES) or CLASS_NAMES[label_id] != label_name:
                raise ValueError(f'清单标签与模型不一致：{row}')
            path = (DATASET_DIR / row['path']).resolve()
            if not path.is_relative_to(DATASET_DIR.resolve()) or not path.is_file():
                raise ValueError(f'图片缺失或路径越界：{path}')
            if path in seen or path.parent.name != label_name:
                raise ValueError(f'清单图片重复或目录标签不一致：{path}')
            if path.suffix.lower() not in IMAGE_EXTENSIONS:
                raise ValueError(f'不支持的图片扩展名：{path}')
            seen.add(path)
            records.append((path, label_id))
    available = {path.resolve() for label in CLASS_NAMES for path in (DATASET_DIR / label).iterdir() if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS}
    if available != seen:
        raise ValueError('目录图片与 manifest.csv 不完全对应')
    counts = Counter((label for (_, label) in records))
    expected_counts = {name: counts[index] for (index, name) in enumerate(CLASS_NAMES)}
    if len(records) == 0 or any((value == 0 for value in expected_counts.values())):
        raise ValueError('数据集为空或存在缺失类别')
    return (records, expected_counts)

class AnatomyDataset(Dataset):

    def __init__(self, records, processor):
        self.records = records
        self.processor = processor

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        (path, label) = self.records[index]
        with Image.open(path) as image:
            values = self.processor(images=image.convert('RGB'), return_tensors='pt')['pixel_values']
        return (values.squeeze(0), label)

def metrics(confusion, top3_correct, loss_sum):
    support = confusion.sum(1).double()
    predicted = confusion.sum(0).double()
    true_positive = confusion.diag().double()
    precision = true_positive / predicted.clamp_min(1)
    recall = true_positive / support.clamp_min(1)
    f1 = 2 * true_positive / (support + predicted).clamp_min(1)
    count = int(support.sum().item())
    per_class = {name: {'support': int(support[index].item()), 'precision_pct': round(100 * precision[index].item(), 4), 'recall_pct': round(100 * recall[index].item(), 4), 'f1_pct': round(100 * f1[index].item(), 4)} for (index, name) in enumerate(CLASS_NAMES)}
    return {'image_count': count, 'top1_correct': int(true_positive.sum().item()), 'top1_accuracy_pct': round(100 * true_positive.sum().item() / count, 4), 'top3_accuracy_pct': round(100 * top3_correct / count, 4), 'macro_precision_pct': round(100 * precision.mean().item(), 4), 'macro_recall_pct': round(100 * recall.mean().item(), 4), 'balanced_accuracy_pct': round(100 * recall.mean().item(), 4), 'macro_f1_pct': round(100 * f1.mean().item(), 4), 'weighted_f1_pct': round(100 * (f1 * support).sum().item() / count, 4), 'cross_entropy_loss': round(loss_sum / count, 6), 'per_class': per_class, 'confusion_matrix': confusion.tolist(), 'confusion_matrix_labels': CLASS_NAMES, 'confusion_matrix_axes': {'rows': 'true', 'columns': 'predicted'}}

def main():
    if BATCH_SIZE < 1:
        raise ValueError('BATCH_SIZE 必须大于 0')
    (records, class_counts) = collect_records()
    with _test_record.timer.loading():
        processor = ViTImageProcessor(do_resize=True, size={'height': 224, 'width': 224}, resample=Image.Resampling.BILINEAR, do_rescale=True, rescale_factor=1 / 255, do_normalize=True, image_mean=[0.5, 0.5, 0.5], image_std=[0.5, 0.5, 0.5])
    with _test_record.timer.loading():
        (model, loading_info) = AutoModelForImageClassification.from_pretrained(str(MODEL_DIR), local_files_only=True, output_loading_info=True)
    if any((loading_info.get(key) for key in ('missing_keys', 'mismatched_keys', 'error_msgs'))):
        raise ValueError(f'模型权重加载不完整：{loading_info}')
    model_labels = [model.config.id2label[index] for index in range(model.config.num_labels)]
    if model_labels != CLASS_NAMES:
        raise ValueError(f'模型类别顺序异常：{model_labels}')
    if model.config.problem_type != 'single_label_classification':
        raise ValueError(f'模型任务类型异常：{model.config.problem_type}')
    device = torch.device(os.environ.get('DEVICE', 'cuda' if torch.cuda.is_available() else 'cpu'))
    device_name = torch.cuda.get_device_name(device) if device.type == 'cuda' else 'CPU'
    with _test_record.timer.loading():
        model = model.to(device).eval()
    loader = _test_record.loader(DataLoader, AnatomyDataset(records, processor), batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=device.type == 'cuda')
    print(f'模型: {MODEL_ID}; 设备: {device} ({device_name}); FP32; batch={BATCH_SIZE}')
    print(f'本地 anatomy-classifier: {len(records)} 张; 类别分布: {class_counts}')
    print('四个模型类别与数据集标签完全对齐；本地清单未区分独立 test split。')
    (warmup_pixels, _) = next(iter(loader))
    warmup_pixels = warmup_pixels.to(device)
    with torch.inference_mode():
        with _test_record.timer.warmup():
            for _ in range(WARMUP_STEPS):
                _test_record.timer.invoke(model, pixel_values=warmup_pixels)
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    del warmup_pixels
    class_count = len(CLASS_NAMES)
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
            predictions = logits.argmax(1)
            _test_record.prediction_ready(locals())
            top3_correct += logits.topk(3, dim=1).indices.eq(targets[:, None]).any(1).sum().item()
            loss_sum += F.cross_entropy(logits, targets, reduction='sum').item()
            confusion += torch.bincount((targets * class_count + predictions).cpu(), minlength=class_count ** 2).reshape(class_count, class_count)
            print(f'已处理: {int(confusion.sum().item())}/{len(records)} ({batch_index}/{len(loader)})')
            _test_record.capture(locals())
    wall_seconds = time.perf_counter() - wall_start
    result_metrics = metrics(confusion, top3_correct, loss_sum)
    if result_metrics['image_count'] != len(records):
        raise ValueError('评测图片计数不一致')
    result = {'recorded_at': datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(timespec='seconds'), 'model_id': MODEL_ID, 'model_path': str(MODEL_DIR), 'script_path': str(Path(__file__).resolve()), 'model': {'id': MODEL_ID, 'architecture': model.__class__.__name__, 'parameter_count_millions': round(sum((p.numel() for p in model.parameters())) / 1000000.0, 4), 'num_labels': class_count, 'class_names': CLASS_NAMES, 'label_mapping_source': 'local model config.json id2label', 'weight_loading_info': loading_info}, 'dataset': {'name': 'anatomy-classifier', 'source': 'local image directories and manifest.csv', 'path': str(DATASET_DIR), 'manifest': str(MANIFEST_PATH), 'manifest_sha256': hashlib.sha256(MANIFEST_PATH.read_bytes()).hexdigest(), 'split': 'full_local_dataset', 'image_count': len(records), 'class_count': class_count, 'class_counts': class_counts, 'class_names': CLASS_NAMES, 'split_note': '本地清单没有独立 test split；使用全部 200 张图片，训练数据重叠未知，结果不等价于独立泛化性能。'}, 'evaluation': {'task': 'four-class anatomy and realism image classification', 'primary_metrics': ['top1_accuracy_pct', 'macro_f1_pct', 'balanced_accuracy_pct'], 'model_card_limitations': '模型卡报告训练图像仅 134 张且类别严重不平衡；本数据集为每类 50 张，分布与训练集不同。'}, 'environment': {'device': str(device), 'device_name': device_name, 'precision': 'FP32', 'batch_size': BATCH_SIZE, 'num_workers': NUM_WORKERS, 'warmup_steps': WARMUP_STEPS, 'python_executable': sys.executable, 'python_version': sys.version.split()[0], 'torch_version': torch.__version__, 'transformers_version': transformers.__version__, 'cuda_version': torch.version.cuda, 'offline': True}, 'preprocessing': {'processor': 'ViTImageProcessor (explicit library defaults; model directory has no preprocessor_config.json)', 'resize': {'height': 224, 'width': 224}, 'center_crop': False, 'resample': 'bilinear', 'rescale_factor': 1 / 255, 'image_mean': [0.5, 0.5, 0.5], 'image_std': [0.5, 0.5, 0.5], 'image_mode': 'RGB', 'assumption_note': '缺少原模型预处理配置，采用已安装 ViTImageProcessor 默认值，并与本地同基座 ViT 配置核对；无法保证与作者训练预处理完全一致。'}, 'metrics': result_metrics, 'timing': {'model_forward_total_seconds': round(forward_seconds, 6), 'model_latency_ms_per_image': round(1000 * forward_seconds / len(records), 6), 'model_throughput_images_per_second': round(len(records) / forward_seconds, 4), 'end_to_end_total_seconds': round(wall_seconds, 6), 'end_to_end_latency_ms_per_image': round(1000 * wall_seconds / len(records), 6), 'end_to_end_throughput_images_per_second': round(len(records) / wall_seconds, 4), 'note': '预热和模型加载不计时；前向计时不含预处理及数据传输；端到端包含读图、预处理、数据传输和指标累计。'}}
    if device.type == 'cuda':
        result['memory'] = {'peak_allocated_mib': round(torch.cuda.max_memory_allocated(device) / 1024 ** 2, 4), 'peak_reserved_mib': round(torch.cuda.max_memory_reserved(device) / 1024 ** 2, 4)}
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    temp = RESULTS_PATH.with_suffix('.json.tmp')
    temp.write_text(json.dumps(result, default=json_metadata_default, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    os.replace(temp, RESULTS_PATH)
    print(json.dumps({k: v for (k, v) in result_metrics.items() if k not in {'per_class', 'confusion_matrix', 'confusion_matrix_labels', 'confusion_matrix_axes'}}, indent=2))
    print(f'结果已保存: {RESULTS_PATH}')
    _test_record.finish(locals())
if __name__ == '__main__':
    main()
