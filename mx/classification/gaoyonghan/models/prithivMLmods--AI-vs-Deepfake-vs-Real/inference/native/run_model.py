"""在模型同名的本地数据集上评测 AI-vs-Deepfake-vs-Real。

该 Hugging Face 数据集只有一个完整 split，没有独立的官方 test split；结果因此
标记为 full_local_dataset，不能直接视为独立泛化性能。模型和数据集的三类标签
顺序都必须是 Artificial、Deepfake、Real。
"""
if __name__ == '__main__':
    import sys as _records_sys
    from pathlib import Path as _RecordsPath
    _records_sys.path.insert(0, str(_RecordsPath(__file__).resolve().parent))
    _records_sys.path.insert(1, str(_RecordsPath(__file__).resolve().parents[1]))
    from common import start as _start_records
    _test_record = _start_records(124, __file__, globals())
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
import time
from datetime import datetime
from pathlib import Path
import torch
import torch.nn.functional as F
from datasets import Dataset as HFDataset
from torch.utils.data import DataLoader, Dataset
from transformers import AutoImageProcessor, AutoModelForImageClassification
MODEL_ID = 'prithivMLmods/AI-vs-Deepfake-vs-Real'
ROOT = __import__('model_paths').resolve_paths()['project']
MODEL_CACHE_DIR = __import__('model_paths').resolve_paths()['weights']
DATASET_DIR = __import__('model_paths').resolve_paths()['dataset'] / 'AI-vs-Deepfake-vs-Real'
RESULTS_PATH = _test_record.output / 'metrics.json'
if __name__ == '__main__':
    RESULTS_PATH = _test_record.output / 'metrics.json'
BATCH_SIZE = int(os.environ.get('MODEL_BATCH_SIZE', '128'))
NUM_WORKERS = int(os.environ.get('MODEL_WORKERS', '0'))
WARMUP_STEPS = int(os.environ.get('MODEL_WARMUP', '3'))
EXPECTED_LABELS = ['Artificial', 'Deepfake', 'Real']

def safe_divide(numerator, denominator):
    return float(numerator) / float(denominator) if denominator else 0.0

def model_labels_in_order(config):
    labels = []
    for index in range(config.num_labels):
        if index in config.id2label:
            labels.append(config.id2label[index])
        elif str(index) in config.id2label:
            labels.append(config.id2label[str(index)])
        else:
            raise ValueError(f'模型缺少 id2label[{index}]')
    return labels

class AIvsDeepfakeDataset(Dataset):

    def __init__(self, dataset, processor):
        self.dataset = dataset
        self.processor = processor

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        example = self.dataset[index]
        pixel_values = self.processor(images=example['image'].convert('RGB'), return_tensors='pt')['pixel_values'].squeeze(0)
        return (pixel_values, int(example['label']))

def classification_metrics(labels, predictions, top3_predictions, losses, confidences, ranks, class_names):
    class_count = len(class_names)
    confusion = torch.zeros((class_count, class_count), dtype=torch.long)
    for (label, prediction) in zip(labels, predictions):
        confusion[label, prediction] += 1
    true_positive = confusion.diag().to(torch.float64)
    support = confusion.sum(dim=1).to(torch.float64)
    predicted = confusion.sum(dim=0).to(torch.float64)
    precision = torch.where(predicted > 0, true_positive / predicted, torch.zeros_like(predicted))
    recall = torch.where(support > 0, true_positive / support, torch.zeros_like(support))
    f1 = torch.where(precision + recall > 0, 2 * precision * recall / (precision + recall), torch.zeros_like(precision))
    per_class = {class_name: {'support': int(support[index].item()), 'precision_pct': round(100 * precision[index].item(), 4), 'recall_pct': round(100 * recall[index].item(), 4), 'f1_pct': round(100 * f1[index].item(), 4)} for (index, class_name) in enumerate(class_names)}
    ece = 0.0
    ece_bins = 15
    for bin_index in range(ece_bins):
        lower = bin_index / ece_bins
        upper = (bin_index + 1) / ece_bins
        indices = [index for (index, confidence) in enumerate(confidences) if lower <= confidence < upper or (bin_index == ece_bins - 1 and confidence == upper)]
        if indices:
            accuracy = sum((labels[index] == predictions[index] for index in indices)) / len(indices)
            mean_confidence = sum((confidences[index] for index in indices)) / len(indices)
            ece += len(indices) / len(labels) * abs(accuracy - mean_confidence)
    top1_correct = sum((label == prediction for (label, prediction) in zip(labels, predictions)))
    top3_correct = sum((label in candidates for (label, candidates) in zip(labels, top3_predictions)))
    off_diagonal = confusion.clone()
    off_diagonal.fill_diagonal_(0)
    top_confusions = []
    for flat_index in torch.argsort(off_diagonal.flatten(), descending=True).tolist():
        count = int(off_diagonal.flatten()[flat_index].item())
        if count == 0 or len(top_confusions) >= 20:
            break
        true_index = flat_index // class_count
        predicted_index = flat_index % class_count
        top_confusions.append({'true': class_names[true_index], 'predicted': class_names[predicted_index], 'count': count})
    image_count = len(labels)
    return {'image_count': image_count, 'top1_accuracy_pct': round(100 * safe_divide(top1_correct, image_count), 4), 'top3_accuracy_pct': round(100 * safe_divide(top3_correct, image_count), 4), 'mean_reciprocal_rank': round(safe_divide(sum((1.0 / rank for rank in ranks)), image_count), 6), 'macro_precision_pct': round(100 * precision.mean().item(), 4), 'macro_recall_pct': round(100 * recall.mean().item(), 4), 'balanced_accuracy_pct': round(100 * recall.mean().item(), 4), 'macro_f1_pct': round(100 * f1.mean().item(), 4), 'weighted_f1_pct': round(100 * safe_divide(sum((f1[index].item() * support[index].item() for index in range(class_count))), image_count), 4), 'cross_entropy_loss': round(safe_divide(sum(losses), image_count), 6), 'expected_calibration_error': round(ece, 6), 'per_class': per_class, 'top_confusion_pairs': top_confusions, 'confusion_matrix': confusion.tolist(), 'confusion_matrix_labels': class_names}

def save_result(result):
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = RESULTS_PATH.with_suffix('.json.tmp')
    temporary_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    os.replace(temporary_path, RESULTS_PATH)

def main():
    if not MODEL_CACHE_DIR.is_dir():
        raise FileNotFoundError(f'未找到模型缓存：{MODEL_CACHE_DIR}')
    parquet_files = sorted(DATASET_DIR.glob('*.parquet'))
    if not parquet_files:
        raise FileNotFoundError(f'未找到数据集 parquet：{DATASET_DIR}')
    with _test_record.timer.loading():
        processor = AutoImageProcessor.from_pretrained(str(MODEL_CACHE_DIR), local_files_only=True)
    with _test_record.timer.loading():
        model = AutoModelForImageClassification.from_pretrained(str(MODEL_CACHE_DIR), local_files_only=True)
    model_class_names = model_labels_in_order(model.config)
    if model_class_names != EXPECTED_LABELS:
        raise ValueError(f'模型类别顺序应为 {EXPECTED_LABELS}，实际为 {model_class_names}')
    if model.config.problem_type != 'single_label_classification':
        raise ValueError(f'模型 problem_type 异常：{model.config.problem_type}')
    dataset = HFDataset.from_parquet([str(path) for path in parquet_files], cache_dir=str(_test_record.output / '.dataset_cache'))
    if set(dataset.column_names) != {'image', 'label'}:
        raise ValueError(f'数据集列应为 image、label，实际为：{dataset.column_names}')
    if list(dataset.features['label'].names) != EXPECTED_LABELS:
        raise ValueError(f"数据集类别顺序应为 {EXPECTED_LABELS}，实际为 {dataset.features['label'].names}")
    wrapped_dataset = AIvsDeepfakeDataset(dataset, processor)
    dataloader = _test_record.loader(DataLoader, wrapped_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=torch.cuda.is_available())
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device_name = torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'
    with _test_record.timer.loading():
        model = model.to(device).eval()
    label_values = dataset['label']
    label_counts = {name: sum((label == index for label in label_values)) for (index, name) in enumerate(EXPECTED_LABELS)}
    print(f'使用设备: {device_name}')
    print(f'模型: {MODEL_ID}')
    print(f'数据集: {DATASET_DIR}（本地完整 parquet，未提供独立 test split）')
    print(f'测试图片数量: {len(wrapped_dataset)}')
    print(f'类别数量: {len(model_class_names)}')
    print(f'类别分布: {label_counts}')
    print(f'模型参数量: {sum((parameter.numel() for parameter in model.parameters())) / 1000000.0:.4f} M')
    print(f'batch size: {BATCH_SIZE}')
    (warmup_pixel_values, _) = next(iter(dataloader))
    warmup_pixel_values = warmup_pixel_values.to(device, non_blocking=True)
    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = True
        torch.cuda.reset_peak_memory_stats(device)
    with torch.inference_mode():
        with _test_record.timer.warmup():
            for _ in range(WARMUP_STEPS):
                _test_record.timer.invoke(model, pixel_values=warmup_pixel_values)
    if device.type == 'cuda':
        torch.cuda.synchronize()
    del warmup_pixel_values
    (labels, predictions, top3_predictions) = ([], [], [])
    (losses, confidences, ranks) = ([], [], [])
    forward_seconds = 0.0
    wall_start = time.perf_counter()
    with torch.inference_mode():
        for (batch_index, (pixel_values, batch_labels)) in enumerate(dataloader, start=1):
            _test_record.before_batch(locals())
            pixel_values = pixel_values.to(device, non_blocking=True)
            batch_labels = batch_labels.to(device, non_blocking=True)
            if device.type == 'cuda':
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)
                start_event.record()
                logits = _test_record.timer.invoke(model, pixel_values=pixel_values).logits
                end_event.record()
                torch.cuda.synchronize()
                forward_seconds += start_event.elapsed_time(end_event) / 1000.0
            else:
                forward_start = time.perf_counter()
                logits = _test_record.timer.invoke(model, pixel_values=pixel_values).logits
                forward_seconds += time.perf_counter() - forward_start
            probabilities = F.softmax(logits, dim=1)
            (confidence, batch_predictions) = probabilities.max(dim=1)
            batch_top3 = logits.topk(3, dim=1).indices
            _test_record.prediction_ready(locals())
            batch_loss = F.cross_entropy(logits, batch_labels, reduction='none')
            true_logits = logits.gather(1, batch_labels.unsqueeze(1))
            batch_ranks = logits.gt(true_logits).sum(dim=1).add(1)
            labels.extend(batch_labels.cpu().tolist())
            predictions.extend(batch_predictions.cpu().tolist())
            top3_predictions.extend(batch_top3.cpu().tolist())
            losses.extend(batch_loss.cpu().tolist())
            confidences.extend(confidence.cpu().tolist())
            ranks.extend(batch_ranks.cpu().tolist())
            if batch_index % 10 == 0 or batch_index == len(dataloader):
                print(f'已处理: {min(batch_index * BATCH_SIZE, len(wrapped_dataset))}/{len(wrapped_dataset)}')
            _test_record.capture(locals())
    if device.type == 'cuda':
        torch.cuda.synchronize()
    wall_seconds = time.perf_counter() - wall_start
    metrics = classification_metrics(labels, predictions, top3_predictions, losses, confidences, ranks, model_class_names)
    image_count = len(wrapped_dataset)
    result = {'recorded_at': datetime.now().astimezone().isoformat(timespec='seconds'), 'model_id': MODEL_ID, 'model_path': str(MODEL_CACHE_DIR), 'script_path': str(Path(__file__).resolve()), 'model': {'id': MODEL_ID, 'architecture': model.__class__.__name__, 'parameter_count_millions': round(sum((parameter.numel() for parameter in model.parameters())) / 1000000.0, 4), 'num_labels': model.config.num_labels, 'id2label': {str(index): label for (index, label) in enumerate(model_class_names)}}, 'dataset': {'name': 'AI-vs-Deepfake-vs-Real', 'source': 'prithivMLmods/AI-vs-Deepfake-vs-Real', 'path': str(DATASET_DIR), 'parquet_files': [str(path) for path in parquet_files], 'split': 'full_local_dataset', 'split_note': '本地数据集仅提供完整数据，没有独立官方 test split；结果可能与模型训练数据重叠，不应视为独立泛化性能。', 'image_count': image_count, 'class_count': len(model_class_names), 'class_names': model_class_names, 'class_counts': label_counts}, 'evaluation': {'task': 'three-class image classification', 'primary_metrics': ['top1_accuracy_pct', 'macro_f1_pct', 'balanced_accuracy_pct', 'cross_entropy_loss'], 'independence_warning': True}, 'environment': {'device': device_name, 'precision': 'FP32', 'batch_size': BATCH_SIZE, 'warmup_steps': WARMUP_STEPS, 'num_workers': NUM_WORKERS}, 'preprocessing': {'processor': processor.__class__.__name__}, 'official_reference': {'model_card_accuracy_pct': 97.5, 'model_card_image_count': 4000, 'comparability': '模型卡评测划分未知，且与本地完整数据集不保证独立或相同。'}, 'metrics': metrics, 'accuracy': {'top1_pct': metrics['top1_accuracy_pct'], 'top3_pct': metrics['top3_accuracy_pct']}, 'timing': {'model_forward_total_seconds': round(forward_seconds, 6), 'model_latency_ms_per_image': round(forward_seconds / image_count * 1000.0, 6), 'model_throughput_images_per_second': round(image_count / forward_seconds, 6), 'end_to_end_total_seconds': round(wall_seconds, 6), 'end_to_end_latency_ms_per_image': round(wall_seconds / image_count * 1000.0, 6), 'end_to_end_throughput_images_per_second': round(image_count / wall_seconds, 6)}}
    if device.type == 'cuda':
        result['memory'] = {'peak_allocated_mib': round(torch.cuda.max_memory_allocated(device) / 1024 ** 2, 4), 'peak_reserved_mib': round(torch.cuda.max_memory_reserved(device) / 1024 ** 2, 4)}
    save_result(result)
    print('=' * 64)
    print(f"Top-1 accuracy: {metrics['top1_accuracy_pct']:.2f}%")
    print(f"Top-3 accuracy: {metrics['top3_accuracy_pct']:.2f}%")
    print(f"Macro-F1: {metrics['macro_f1_pct']:.2f}%")
    print(f"Balanced accuracy: {metrics['balanced_accuracy_pct']:.2f}%")
    print(f"模型平均推理时间: {result['timing']['model_latency_ms_per_image']:.4f} ms/image")
    print(f"模型吞吐量: {result['timing']['model_throughput_images_per_second']:.2f} images/s")
    print(f'评测结果已更新: {RESULTS_PATH}')
    _test_record.finish(locals())
if __name__ == '__main__':
    main()
