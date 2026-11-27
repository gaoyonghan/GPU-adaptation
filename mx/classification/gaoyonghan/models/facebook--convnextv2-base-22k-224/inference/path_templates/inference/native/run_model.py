if __name__ == '__main__':
    import sys as _records_sys
    from pathlib import Path as _RecordsPath
    _records_sys.path.insert(0, str(_RecordsPath(__file__).resolve().parent))
    _records_sys.path.insert(1, str(_RecordsPath(__file__).resolve().parents[1]))
    from common import start as _start_records
    _test_record = _start_records(109, __file__, globals())
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
import json
from dataclasses import asdict, is_dataclass
import time
from datetime import datetime
from pathlib import Path
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from transformers import AutoImageProcessor, ConvNextV2ForImageClassification
ROOT = __import__('model_paths').resolve_paths()['project']
MODEL_CACHE_DIR = str(__import__('model_paths').resolve_paths()['weights'])
DATASET_DIR = __import__('model_paths').resolve_paths()['dataset'] / 'ImageNet-1K'
VAL_DIR = DATASET_DIR / 'val'
VAL_LABELS_PATH = DATASET_DIR / 'LOC_val_solution.csv'
SYNSET_MAPPING_PATH = DATASET_DIR / 'LOC_synset_mapping.txt'
RESULTS_PATH = _test_record.output / 'metrics.json'
if __name__ == '__main__':
    RESULTS_PATH = _test_record.output / 'metrics.json'
MODEL_ID = 'facebook/convnextv2-base-22k-224'
BATCH_SIZE = int(os.environ.get('MODEL_BATCH_SIZE', '64'))
NUM_WORKERS = int(os.environ.get('MODEL_WORKERS', '8'))
WARMUP_STEPS = int(os.environ.get('MODEL_WARMUP', '10'))
ECE_BIN_COUNT = 15
os.makedirs(MODEL_CACHE_DIR, exist_ok=True)
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

class ImageNetValidationDataset(Dataset):

    def __init__(self, processor):
        with SYNSET_MAPPING_PATH.open(encoding='utf-8') as f:
            mapping_rows = [line.strip().split(maxsplit=1) for line in f if line.strip()]
        self.synsets = [row[0] for row in mapping_rows]
        self.class_names = [row[1] for row in mapping_rows]
        synset_to_index = {synset: index for (index, synset) in enumerate(self.synsets)}
        with VAL_LABELS_PATH.open(newline='', encoding='utf-8') as f:
            rows = csv.DictReader(f)
            labels_by_image_id = {row['ImageId']: synset_to_index[row['PredictionString'].split()[0]] for row in rows}
        self.image_paths = sorted(VAL_DIR.glob('*.JPEG'))
        self.labels = [labels_by_image_id[path.stem] for path in self.image_paths]
        self.processor = processor
        if len(self.synsets) != 1000:
            raise ValueError(f'类别映射应包含 1000 类，实际为 {len(self.synsets)} 类')
        if len(self.image_paths) != 50000:
            raise ValueError(f'ImageNet-1K val 应包含 50000 张图片，实际为 {len(self.image_paths)} 张')

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        with Image.open(self.image_paths[index]) as image:
            pixel_values = self.processor(images=image.convert('RGB'), return_tensors='pt')['pixel_values'][0]
        return (pixel_values, self.labels[index])

def safe_divide(numerator, denominator):
    return torch.where(denominator > 0, numerator / denominator, torch.zeros_like(numerator))
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'使用设备: {device}')
print(f'模型: {MODEL_ID}')
with _test_record.timer.loading():
    processor = AutoImageProcessor.from_pretrained(MODEL_ID, cache_dir=MODEL_CACHE_DIR, use_fast=False)
with _test_record.timer.loading():
    model = ConvNextV2ForImageClassification.from_pretrained(MODEL_ID, cache_dir=MODEL_CACHE_DIR, use_safetensors=True).to(device)
with _test_record.timer.loading():
    model.eval()
dataset = ImageNetValidationDataset(processor)
num_classes = len(dataset.class_names)
if model.config.num_labels != num_classes:
    raise ValueError(f'该检查点应输出 {num_classes} 个 ImageNet-1K 类别，实际为 {model.config.num_labels} 类')
model_class_names = [model.config.id2label[index] for index in range(num_classes)]
if model_class_names != dataset.class_names:
    raise ValueError('模型的 1000 类标签顺序与 ImageNet-1K 官方映射不一致')
dataloader = _test_record.loader(DataLoader, dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=device == 'cuda', persistent_workers=NUM_WORKERS > 0)
total_parameters = sum((parameter.numel() for parameter in model.parameters()))
parameter_bytes = sum((parameter.numel() * parameter.element_size() for parameter in model.parameters()))
print(f'验证图片数量: {len(dataset)}')
print(f'模型输出类别: {model.config.num_labels}（ImageNet-1K）')
print(f'batch size: {BATCH_SIZE}')
print(f'模型参数量: {total_parameters / 1000000:.2f} M')
print(f'图像处理器: {type(processor).__name__}')
with Image.open(dataset.image_paths[0]) as image:
    warmup_inputs = processor(images=[image.convert('RGB')] * BATCH_SIZE, return_tensors='pt').to(device)
if device == 'cuda':
    torch.backends.cudnn.benchmark = True
with torch.inference_mode():
    with _test_record.timer.warmup():
        for _ in range(WARMUP_STEPS):
            _test_record.timer.invoke(model, **warmup_inputs)
if device == 'cuda':
    torch.cuda.synchronize()
del warmup_inputs
if device == 'cuda':
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
confusion_matrix = torch.zeros((num_classes, num_classes), dtype=torch.long, device=device)
correct_top5 = torch.zeros((), dtype=torch.long, device=device)
negative_log_likelihood = torch.zeros((), dtype=torch.float64, device=device)
ece_counts = torch.zeros(ECE_BIN_COUNT, dtype=torch.float64, device=device)
ece_confidence_sums = torch.zeros(ECE_BIN_COUNT, dtype=torch.float64, device=device)
ece_correct_sums = torch.zeros(ECE_BIN_COUNT, dtype=torch.float64, device=device)
gpu_events = []
cpu_forward_seconds = 0.0
wall_start = time.perf_counter()
with torch.inference_mode():
    for (batch_index, (pixel_values, labels)) in enumerate(dataloader, start=1):
        _test_record.before_batch(locals())
        pixel_values = pixel_values.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        if device == 'cuda':
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
            logits = _test_record.timer.invoke(model, pixel_values=pixel_values).logits
            end_event.record()
            gpu_events.append((start_event, end_event))
        else:
            forward_start = time.perf_counter()
            logits = _test_record.timer.invoke(model, pixel_values=pixel_values).logits
            cpu_forward_seconds += time.perf_counter() - forward_start
        top5_indices = logits.topk(k=5, dim=1).indices
        predictions = top5_indices[:, 0]
        _test_record.prediction_ready(locals())
        correct_top5 += top5_indices.eq(labels.unsqueeze(1)).any(dim=1).sum()
        confusion_matrix += torch.bincount(labels * num_classes + predictions, minlength=num_classes * num_classes).reshape(num_classes, num_classes)
        negative_log_likelihood += F.cross_entropy(logits, labels, reduction='sum').double()
        confidences = logits.softmax(dim=1).amax(dim=1)
        bin_indices = torch.clamp((confidences * ECE_BIN_COUNT).long(), max=ECE_BIN_COUNT - 1)
        ece_counts += torch.bincount(bin_indices, minlength=ECE_BIN_COUNT).double()
        ece_confidence_sums.scatter_add_(0, bin_indices, confidences.double())
        ece_correct_sums.scatter_add_(0, bin_indices, predictions.eq(labels).double())
        if batch_index % 100 == 0:
            processed = min(batch_index * BATCH_SIZE, len(dataset))
            print(f'已处理: {processed}/{len(dataset)}')
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
top1_accuracy = true_positive.sum().item() / image_count
top5_accuracy = correct_top5.item() / image_count
macro_precision = precision.mean().item()
macro_recall = recall.mean().item()
macro_f1 = f1.mean().item()
mean_nll = negative_log_likelihood.item() / image_count
nonempty_bins = ece_counts > 0
ece_bin_accuracy = safe_divide(ece_correct_sums, ece_counts)
ece_bin_confidence = safe_divide(ece_confidence_sums, ece_counts)
ece = ((ece_bin_accuracy[nonempty_bins] - ece_bin_confidence[nonempty_bins]).abs() * ece_counts[nonempty_bins]).sum().item() / image_count
average_forward_ms = forward_seconds / image_count * 1000
model_fps = image_count / forward_seconds
end_to_end_ms = wall_seconds / image_count * 1000
end_to_end_fps = image_count / wall_seconds
per_class_metrics = []
for (index, (synset, class_name)) in enumerate(zip(dataset.synsets, dataset.class_names)):
    per_class_metrics.append({'class_id': index, 'synset': synset, 'class_name': class_name, 'support': int(support[index].item()), 'precision_pct': round(precision[index].item() * 100, 4), 'recall_pct': round(recall[index].item() * 100, 4), 'f1_pct': round(f1[index].item() * 100, 4)})
lowest_recall_indices = torch.argsort(recall)[:10].tolist()
print('\n' + '=' * 72)
print('ImageNet-1K val 上的 ConvNeXt V2 Base 评测结果')
print('=' * 72)
print(f'Top-1 accuracy: {top1_accuracy * 100:.2f}%')
print(f'Top-5 accuracy: {top5_accuracy * 100:.2f}%')
print(f'Macro precision / recall / F1: {macro_precision * 100:.2f}% / {macro_recall * 100:.2f}% / {macro_f1 * 100:.2f}%')
print(f'平均负对数似然（NLL）: {mean_nll:.4f}')
print(f'期望校准误差（ECE, {ECE_BIN_COUNT} bins）: {ece * 100:.2f}%')
print(f'纯模型平均推理时间: {average_forward_ms:.3f} ms/image')
print(f'纯模型吞吐量: {model_fps:.2f} images/s')
print(f'端到端平均时间: {end_to_end_ms:.3f} ms/image')
print(f'端到端吞吐量: {end_to_end_fps:.2f} images/s')
if peak_allocated_bytes is not None:
    print(f'峰值已分配显存: {peak_allocated_bytes / 1024 ** 3:.3f} GiB')
    print(f'峰值保留显存: {peak_reserved_bytes / 1024 ** 3:.3f} GiB')
print('\nRecall 最低的 10 个类别:')
for index in lowest_recall_indices:
    metric = per_class_metrics[index]
    print(f"  {metric['synset']} {metric['class_name']:<32} Recall={metric['recall_pct']:6.2f}%  F1={metric['f1_pct']:6.2f}%")
result = {'recorded_at': datetime.now().astimezone().isoformat(timespec='seconds'), 'task_type': 'image_classification', 'model': {'id': MODEL_ID, 'path': MODEL_CACHE_DIR, 'pretraining': 'ImageNet-22K', 'output_label_space': 'ImageNet-1K (1000 classes)', 'parameter_count': total_parameters, 'parameter_count_millions': round(total_parameters / 1000000, 4), 'parameter_memory_fp32_mib': round(parameter_bytes / 1024 ** 2, 4)}, 'script_path': str(Path(__file__).resolve()), 'dataset': {'name': 'ImageNet-1K val', 'path': str(DATASET_DIR), 'image_count': image_count, 'class_count': num_classes}, 'environment': {'device': torch.cuda.get_device_name(torch.cuda.current_device()) if device == 'cuda' else 'CPU', 'precision': 'FP32', 'batch_size': BATCH_SIZE, 'warmup_steps': WARMUP_STEPS, 'num_workers': NUM_WORKERS, 'peak_allocated_memory_gib': round(peak_allocated_bytes / 1024 ** 3, 4) if peak_allocated_bytes is not None else None, 'peak_reserved_memory_gib': round(peak_reserved_bytes / 1024 ** 3, 4) if peak_reserved_bytes is not None else None}, 'preprocessing': processor.to_dict(), 'classification_metrics': {'top1_accuracy_pct': round(top1_accuracy * 100, 4), 'top5_accuracy_pct': round(top5_accuracy * 100, 4), 'macro_average': {'precision_pct': round(macro_precision * 100, 4), 'recall_pct': round(macro_recall * 100, 4), 'f1_pct': round(macro_f1 * 100, 4)}, 'negative_log_likelihood': round(mean_nll, 6), 'expected_calibration_error_pct': round(ece * 100, 4), 'ece_bin_count': ECE_BIN_COUNT, 'per_class': per_class_metrics}, 'timing': {'model_latency_ms_per_image': round(average_forward_ms, 4), 'model_throughput_images_per_second': round(model_fps, 4), 'end_to_end_latency_ms_per_image': round(end_to_end_ms, 4), 'end_to_end_throughput_images_per_second': round(end_to_end_fps, 4)}}
RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
temporary_results_path = RESULTS_PATH.with_suffix('.json.tmp')
with temporary_results_path.open('w', encoding='utf-8') as f:
    json.dump(result, f, default=json_metadata_default, ensure_ascii=False, indent=2)
    f.write('\n')
os.replace(temporary_results_path, RESULTS_PATH)
print(f'评测结果已更新: {RESULTS_PATH}')
_test_record.finish(globals())
