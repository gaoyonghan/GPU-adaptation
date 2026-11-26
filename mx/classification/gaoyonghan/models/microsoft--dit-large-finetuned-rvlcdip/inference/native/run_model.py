if __name__ == '__main__':
    import sys as _records_sys
    from pathlib import Path as _RecordsPath
    _records_sys.path.insert(0, str(_RecordsPath(__file__).resolve().parent))
    _records_sys.path.insert(1, str(_RecordsPath(__file__).resolve().parents[1]))
    from common import start as _start_records
    _test_record = _start_records(104, __file__, globals())
import os
import sys
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['PYTHONUNBUFFERED'] = '1'
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(line_buffering=True, write_through=True)
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(line_buffering=True, write_through=True)
import json
from dataclasses import asdict, is_dataclass
import time
from datetime import datetime
from pathlib import Path
import torch
import torch.nn.functional as F
from datasets import Dataset as HFDataset, concatenate_datasets
from torch.utils.data import DataLoader, Dataset
from transformers import AutoImageProcessor, AutoModelForImageClassification
ROOT = __import__('model_paths').resolve_paths()['project']
MODEL_CACHE_DIR = str(__import__('model_paths').resolve_paths()['weights'])
DATASET_DIR = __import__('model_paths').resolve_paths()['dataset'] / 'RVL-CDIP'
DATASET_CACHE_DIR = DATASET_DIR / 'cache'
RESULTS_PATH = _test_record.output / 'metrics.json'
if __name__ == '__main__':
    RESULTS_PATH = _test_record.output / 'metrics.json'
MODEL_ID = 'microsoft/dit-large-finetuned-rvlcdip'
DATASET_ID = 'aharley/rvl_cdip'
DATASET_SPLIT = 'test'
BATCH_SIZE = int(os.environ.get('MODEL_BATCH_SIZE', '64'))
NUM_WORKERS = int(os.environ.get('MODEL_WORKERS', '8'))
WARMUP_STEPS = int(os.environ.get('MODEL_WARMUP', '10'))
os.makedirs(MODEL_CACHE_DIR, exist_ok=True)
os.makedirs(DATASET_CACHE_DIR, exist_ok=True)
os.environ['HF_HOME'] = MODEL_CACHE_DIR
os.environ['HF_HUB_CACHE'] = MODEL_CACHE_DIR
os.environ['TRANSFORMERS_CACHE'] = MODEL_CACHE_DIR

def json_metadata_default(value):
    """Serialize library metadata without changing inference or preprocessing."""
    if isinstance(value, (set, frozenset)):
        return sorted(value)
    if is_dataclass(value) and (not isinstance(value, type)):
        return {key: item for (key, item) in asdict(value).items() if item is not None}
    raise TypeError(f'Object of type {type(value).__name__} is not JSON serializable')

class RVLCDIPTestDataset(Dataset):

    def __init__(self, dataset, processor):
        self.dataset = dataset
        self.processor = processor

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        try:
            sample = self.dataset[index]
            pixel_values = self.processor(images=sample['image'].convert('RGB'), return_tensors='pt')['pixel_values'][0]
            return (pixel_values, sample['label'])
        except Exception:
            return None

def collate_skip_corrupt(batch):
    valid_batch = [item for item in batch if item is not None]
    if not valid_batch:
        return None
    (pixel_values, labels) = zip(*valid_batch)
    return (torch.stack(pixel_values), torch.tensor(labels, dtype=torch.long))

def normalized_label(label):
    return label.lower().replace('_', ' ').replace('-', ' ').strip()

def safe_divide(numerator, denominator):
    return torch.where(denominator > 0, numerator / denominator, torch.zeros_like(numerator))

def load_local_test_split():
    test_shard_paths = sorted(DATASET_CACHE_DIR.glob('aharley___rvl_cdip/default/*/*/rvl_cdip-test-*.arrow'))
    if not test_shard_paths:
        raise FileNotFoundError(f'未找到 RVL-CDIP test 分片: {DATASET_CACHE_DIR}')
    test_shards = [HFDataset.from_file(str(path)) for path in test_shard_paths]
    return concatenate_datasets(test_shards)

def find_local_model_snapshot():
    model_repo_dir = Path(MODEL_CACHE_DIR) / ('models--' + MODEL_ID.replace('/', '--'))
    candidates = []
    for snapshot_dir in sorted((model_repo_dir / 'snapshots').glob('*')):
        has_config = (snapshot_dir / 'config.json').is_file()
        has_processor = (snapshot_dir / 'preprocessor_config.json').is_file()
        has_weights = any(((snapshot_dir / filename).is_file() for filename in ('pytorch_model.bin', 'model.safetensors')))
        if has_config and has_processor and has_weights:
            candidates.append(snapshot_dir)
    if not candidates:
        raise FileNotFoundError(f'未找到完整的本地模型 snapshot: {model_repo_dir}')
    return candidates[-1]
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'使用设备: {device}')
print(f'模型: {MODEL_ID}')
print(f'数据集: {DATASET_ID}, split={DATASET_SPLIT}')
rvlcdip_test = load_local_test_split()
model_snapshot_dir = find_local_model_snapshot()
if len(rvlcdip_test) != 40000:
    raise ValueError(f'RVL-CDIP 官方 test split 应包含 40000 张图片，实际为 {len(rvlcdip_test)} 张')
class_names = rvlcdip_test.features['label'].names
num_classes = len(class_names)
if num_classes != 16:
    raise ValueError(f'RVL-CDIP 应包含 16 类，实际为 {num_classes} 类')
with _test_record.timer.loading():
    processor = AutoImageProcessor.from_pretrained(str(model_snapshot_dir), use_fast=False, local_files_only=True)
with _test_record.timer.loading():
    model = AutoModelForImageClassification.from_pretrained(str(model_snapshot_dir), local_files_only=True).to(device)
with _test_record.timer.loading():
    model.eval()
if model.config.num_labels != num_classes:
    raise ValueError(f'模型分类头应输出 {num_classes} 类，实际为 {model.config.num_labels} 类')
model_class_names = [model.config.id2label[index] for index in range(num_classes)]
dataset_label_lookup = {normalized_label(label): index for (index, label) in enumerate(class_names)}
if set(map(normalized_label, model_class_names)) != set(dataset_label_lookup):
    raise ValueError(f'模型标签与 RVL-CDIP 数据集标签不一致:\n模型标签: {model_class_names}\n数据集标签: {class_names}')
dataset_to_model_indices = torch.empty(num_classes, dtype=torch.long, device=device)
for (model_index, label) in enumerate(model_class_names):
    dataset_index = dataset_label_lookup[normalized_label(label)]
    dataset_to_model_indices[dataset_index] = model_index
dataset = RVLCDIPTestDataset(rvlcdip_test, processor)
dataloader = _test_record.loader(DataLoader, dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=device == 'cuda', persistent_workers=NUM_WORKERS > 0, collate_fn=collate_skip_corrupt)
total_parameters = sum((parameter.numel() for parameter in model.parameters()))
parameter_bytes = sum((parameter.numel() * parameter.element_size() for parameter in model.parameters()))
print(f'测试图片数量: {len(dataset)}')
print(f'类别数量: {num_classes}')
print(f'batch size: {BATCH_SIZE}')
print(f'模型参数量: {total_parameters / 1000000:.2f} M')
print(f'图像处理器: {type(processor).__name__}')
first_image = rvlcdip_test[0]['image'].convert('RGB')
warmup_inputs = processor(images=[first_image] * BATCH_SIZE, return_tensors='pt').to(device)
if device == 'cuda':
    torch.backends.cudnn.benchmark = True
with torch.inference_mode():
    with _test_record.timer.warmup():
        for _ in range(WARMUP_STEPS):
            _test_record.timer.invoke(model, **warmup_inputs)
if device == 'cuda':
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
del warmup_inputs, first_image
confusion_matrix = torch.zeros((num_classes, num_classes), dtype=torch.long, device=device)
negative_log_likelihood = torch.zeros((), dtype=torch.float64, device=device)
gpu_events = []
cpu_forward_seconds = 0.0
wall_start = time.perf_counter()
evaluated_count = 0
skipped_count = 0
with torch.inference_mode():
    for (batch_index, batch) in enumerate(dataloader, start=1):
        _test_record.before_batch(locals())
        if batch is None:
            skipped_count += BATCH_SIZE
            continue
        (pixel_values, labels) = batch
        skipped_count += BATCH_SIZE - labels.shape[0]
        evaluated_count += labels.shape[0]
        pixel_values = pixel_values.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        if device == 'cuda':
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
            model_logits = _test_record.timer.invoke(model, pixel_values=pixel_values).logits
            end_event.record()
            gpu_events.append((start_event, end_event))
        else:
            forward_start = time.perf_counter()
            model_logits = _test_record.timer.invoke(model, pixel_values=pixel_values).logits
            cpu_forward_seconds += time.perf_counter() - forward_start
        logits = model_logits.index_select(1, dataset_to_model_indices)
        predictions = logits.argmax(dim=1)
        _test_record.prediction_ready(locals())
        confusion_matrix += torch.bincount(labels * num_classes + predictions, minlength=num_classes * num_classes).reshape(num_classes, num_classes)
        negative_log_likelihood += F.cross_entropy(logits, labels, reduction='sum').double()
        if batch_index % 100 == 0:
            print(f'已处理: {evaluated_count}/{len(dataset)}，跳过损坏图片: {skipped_count}')
        _test_record.capture(locals())
if device == 'cuda':
    torch.cuda.synchronize()
wall_seconds = time.perf_counter() - wall_start
if device == 'cuda':
    forward_seconds = sum((start.elapsed_time(end) for (start, end) in gpu_events)) / 1000
    peak_allocated_bytes = torch.cuda.max_memory_allocated()
    peak_reserved_bytes = torch.cuda.max_memory_reserved()
else:
    forward_seconds = cpu_forward_seconds
    peak_allocated_bytes = None
    peak_reserved_bytes = None
confusion_cpu = confusion_matrix.cpu()
true_positive = confusion_cpu.diag().double()
support = confusion_cpu.sum(dim=1).double()
predicted_count = confusion_cpu.sum(dim=0).double()
precision = safe_divide(true_positive, predicted_count)
recall = safe_divide(true_positive, support)
f1 = safe_divide(2 * precision * recall, precision + recall)
image_count = int(support.sum().item())
if image_count == 0:
    raise RuntimeError('没有可用于评测的有效图片')
print(f'有效评测图片: {image_count}/{len(rvlcdip_test)}，跳过损坏图片: {skipped_count}')
accuracy = true_positive.sum().item() / image_count
macro_precision = precision.mean().item()
macro_recall = recall.mean().item()
macro_f1 = f1.mean().item()
class_weights = support / support.sum()
weighted_precision = (precision * class_weights).sum().item()
weighted_recall = (recall * class_weights).sum().item()
weighted_f1 = (f1 * class_weights).sum().item()
mean_nll = negative_log_likelihood.item() / image_count
average_forward_ms = forward_seconds / image_count * 1000
model_fps = image_count / forward_seconds
end_to_end_ms = wall_seconds / image_count * 1000
end_to_end_fps = image_count / wall_seconds
per_class_metrics = []
for (index, class_name) in enumerate(class_names):
    per_class_metrics.append({'class_id': index, 'class_name': class_name, 'support': int(support[index].item()), 'precision_pct': round(precision[index].item() * 100, 4), 'recall_pct': round(recall[index].item() * 100, 4), 'f1_pct': round(f1[index].item() * 100, 4)})
print('\n' + '=' * 72)
print('RVL-CDIP 官方 test split 上的 DiT-Large 文档分类评测结果')
print('=' * 72)
print(f'Accuracy: {accuracy * 100:.2f}%')
print(f'Balanced accuracy / Macro recall: {macro_recall * 100:.2f}%')
print(f'Macro precision / recall / F1: {macro_precision * 100:.2f}% / {macro_recall * 100:.2f}% / {macro_f1 * 100:.2f}%')
print(f'Weighted precision / recall / F1: {weighted_precision * 100:.2f}% / {weighted_recall * 100:.2f}% / {weighted_f1 * 100:.2f}%')
print(f'平均负对数似然（NLL）: {mean_nll:.4f}')
print(f'纯模型平均推理时间: {average_forward_ms:.3f} ms/image')
print(f'纯模型吞吐量: {model_fps:.2f} images/s')
print(f'端到端平均时间: {end_to_end_ms:.3f} ms/image')
print(f'端到端吞吐量: {end_to_end_fps:.2f} images/s')
if peak_allocated_bytes is not None:
    print(f'峰值已分配显存: {peak_allocated_bytes / 1024 ** 3:.3f} GiB')
    print(f'峰值保留显存: {peak_reserved_bytes / 1024 ** 3:.3f} GiB')
print('\n各类别 Recall / F1:')
for metric in per_class_metrics:
    print(f"  {metric['class_id']:2d} {metric['class_name']:<24} Recall={metric['recall_pct']:6.2f}%  F1={metric['f1_pct']:6.2f}%  n={metric['support']}")
result = {'recorded_at': datetime.now().astimezone().isoformat(timespec='seconds'), 'task_type': 'document_image_classification', 'model': {'id': MODEL_ID, 'path': MODEL_CACHE_DIR, 'parameter_count': total_parameters, 'parameter_count_millions': round(total_parameters / 1000000, 4), 'parameter_memory_fp32_mib': round(parameter_bytes / 1024 ** 2, 4)}, 'script_path': str(Path(__file__).resolve()), 'dataset': {'id': DATASET_ID, 'name': 'RVL-CDIP', 'split': DATASET_SPLIT, 'path': str(DATASET_DIR), 'image_count': image_count, 'skipped_corrupt_images': skipped_count, 'expected_image_count': len(rvlcdip_test), 'class_count': num_classes, 'class_names': class_names}, 'environment': {'device': torch.cuda.get_device_name(torch.cuda.current_device()) if device == 'cuda' else 'CPU', 'precision': 'FP32', 'batch_size': BATCH_SIZE, 'warmup_steps': WARMUP_STEPS, 'num_workers': NUM_WORKERS, 'peak_allocated_memory_gib': round(peak_allocated_bytes / 1024 ** 3, 4) if peak_allocated_bytes is not None else None, 'peak_reserved_memory_gib': round(peak_reserved_bytes / 1024 ** 3, 4) if peak_reserved_bytes is not None else None}, 'preprocessing': processor.to_dict(), 'classification_metrics': {'accuracy_pct': round(accuracy * 100, 4), 'balanced_accuracy_pct': round(macro_recall * 100, 4), 'negative_log_likelihood': round(mean_nll, 6), 'macro_average': {'precision_pct': round(macro_precision * 100, 4), 'recall_pct': round(macro_recall * 100, 4), 'f1_pct': round(macro_f1 * 100, 4)}, 'weighted_average': {'precision_pct': round(weighted_precision * 100, 4), 'recall_pct': round(weighted_recall * 100, 4), 'f1_pct': round(weighted_f1 * 100, 4)}, 'per_class': per_class_metrics, 'confusion_matrix': {'rows': 'ground_truth', 'columns': 'prediction', 'class_names': class_names, 'counts': confusion_cpu.tolist()}}, 'timing': {'model_latency_ms_per_image': round(average_forward_ms, 4), 'model_throughput_images_per_second': round(model_fps, 4), 'end_to_end_latency_ms_per_image': round(end_to_end_ms, 4), 'end_to_end_throughput_images_per_second': round(end_to_end_fps, 4)}}
RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
temporary_results_path = RESULTS_PATH.with_suffix('.json.tmp')
with temporary_results_path.open('w', encoding='utf-8') as f:
    json.dump(result, f, default=json_metadata_default, ensure_ascii=False, indent=2)
    f.write('\n')
os.replace(temporary_results_path, RESULTS_PATH)
print(f'评测结果已更新: {RESULTS_PATH}')
_test_record.finish(globals())
