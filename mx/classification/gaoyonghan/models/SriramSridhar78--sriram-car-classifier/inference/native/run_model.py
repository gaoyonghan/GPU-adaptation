"""在 Stanford Cars 官方 test split 上评测 SriramSridhar78/sriram-car-classifier。"""
if __name__ == '__main__':
    import sys as _records_sys
    from pathlib import Path as _RecordsPath
    _records_sys.path.insert(0, str(_RecordsPath(__file__).resolve().parent))
    _records_sys.path.insert(1, str(_RecordsPath(__file__).resolve().parents[1]))
    from common import start as _start_records
    _test_record = _start_records(153, __file__, globals())
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
import re
import time
from datetime import datetime
from pathlib import Path
import torch
import torch.nn.functional as F
from datasets import Dataset as HFDataset
from torch.utils.data import DataLoader, Dataset
from transformers import AutoImageProcessor, AutoModelForImageClassification
MODEL_ID = 'SriramSridhar78/sriram-car-classifier'
ROOT = __import__('model_paths').resolve_paths()['project']
MODEL_CACHE_DIR = __import__('model_paths').resolve_paths()['weights']
DATASET_DIR = __import__('model_paths').resolve_paths()['dataset'] / 'StanfordCars'
RESULTS_PATH = _test_record.output / 'metrics.json'
if __name__ == '__main__':
    RESULTS_PATH = _test_record.output / 'metrics.json'
BATCH_SIZE = int(os.environ.get('MODEL_BATCH_SIZE', '64'))
NUM_WORKERS = int(os.environ.get('MODEL_WORKERS', '0'))
WARMUP_STEPS = int(os.environ.get('MODEL_WARMUP', '3'))
ECE_BINS = 15

def normalize_class_name(name):
    return re.sub('\\s+', ' ', re.sub('[/_]+', ' ', str(name).lower())).strip()

class StanfordCarsTestDataset(Dataset):

    def __init__(self, dataset, processor, dataset_to_model_label):
        self.dataset = dataset
        self.processor = processor
        self.dataset_to_model_label = dataset_to_model_label

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        example = self.dataset[index]
        dataset_label = int(example['label'])
        model_label = self.dataset_to_model_label[dataset_label]
        pixel_values = self.processor(images=example['image'].convert('RGB'), return_tensors='pt')['pixel_values'].squeeze(0)
        return (pixel_values, model_label)

def safe_divide(numerator, denominator):
    return float(numerator) / float(denominator) if denominator else 0.0

def classification_metrics(labels, predictions, top5_predictions, losses, confidences, class_names):
    class_count = len(class_names)
    confusion = [[0 for _ in range(class_count)] for _ in range(class_count)]
    for (label, prediction) in zip(labels, predictions):
        confusion[label][prediction] += 1
    per_class = {}
    recalls = []
    f1_scores = []
    for (class_index, class_name) in enumerate(class_names):
        true_positive = confusion[class_index][class_index]
        false_positive = sum((row[class_index] for row in confusion)) - true_positive
        false_negative = sum(confusion[class_index]) - true_positive
        precision = safe_divide(true_positive, true_positive + false_positive)
        recall = safe_divide(true_positive, true_positive + false_negative)
        f1 = safe_divide(2 * precision * recall, precision + recall)
        recalls.append(recall)
        f1_scores.append(f1)
        per_class[class_name] = {'support': sum(confusion[class_index]), 'precision_pct': round(100 * precision, 4), 'recall_pct': round(100 * recall, 4), 'f1_pct': round(100 * f1, 4)}
    ece = 0.0
    for bin_index in range(ECE_BINS):
        lower = bin_index / ECE_BINS
        upper = (bin_index + 1) / ECE_BINS
        indices = [index for (index, confidence) in enumerate(confidences) if lower <= confidence < upper or (bin_index == ECE_BINS - 1 and confidence == upper)]
        if indices:
            accuracy = sum((labels[index] == predictions[index] for index in indices)) / len(indices)
            average_confidence = sum((confidences[index] for index in indices)) / len(indices)
            ece += len(indices) / len(labels) * abs(accuracy - average_confidence)
    top1_correct = sum((label == prediction for (label, prediction) in zip(labels, predictions)))
    top5_correct = sum((label in candidates for (label, candidates) in zip(labels, top5_predictions)))
    top_confusions = []
    for (true_index, row) in enumerate(confusion):
        for (predicted_index, count) in enumerate(row):
            if true_index != predicted_index and count:
                top_confusions.append({'true': class_names[true_index], 'predicted': class_names[predicted_index], 'count': count})
    top_confusions.sort(key=lambda item: item['count'], reverse=True)
    return {'image_count': len(labels), 'top1_accuracy_pct': round(100 * safe_divide(top1_correct, len(labels)), 4), 'top5_accuracy_pct': round(100 * safe_divide(top5_correct, len(labels)), 4), 'macro_f1_pct': round(100 * sum(f1_scores) / class_count, 4), 'balanced_accuracy_pct': round(100 * sum(recalls) / class_count, 4), 'cross_entropy_loss': round(safe_divide(sum(losses), len(labels)), 6), 'expected_calibration_error': round(ece, 6), 'per_class': per_class, 'confusion_matrix_rows_true_cols_pred': confusion, 'top_confusion_pairs': top_confusions[:30]}

def find_test_shards():
    shards = sorted(DATASET_DIR.glob('data/test-*.parquet'))
    if not shards:
        raise FileNotFoundError(f'未找到 Stanford Cars test parquet：{DATASET_DIR}')
    return shards

def save_result(result):
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = RESULTS_PATH.with_suffix('.json.tmp')
    temporary_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    os.replace(temporary_path, RESULTS_PATH)

def main():
    if not MODEL_CACHE_DIR.is_dir():
        raise FileNotFoundError(f'未找到模型缓存：{MODEL_CACHE_DIR}')
    shards = find_test_shards()
    with _test_record.timer.loading():
        processor = AutoImageProcessor.from_pretrained(str(MODEL_CACHE_DIR), local_files_only=True)
    with _test_record.timer.loading():
        model = AutoModelForImageClassification.from_pretrained(str(MODEL_CACHE_DIR), local_files_only=True)
    model_label_count = int(model.config.num_labels)
    model_id2label = {int(key): value for (key, value) in model.config.id2label.items()}
    model_class_names = [model_id2label[index] for index in range(model_label_count)]
    if model_label_count != 196:
        raise ValueError(f'模型应为 196 类，实际为 {model_label_count} 类')
    normalized_model_names = [normalize_class_name(name) for name in model_class_names]
    if len(set(normalized_model_names)) != model_label_count:
        raise ValueError('模型类别名称存在重复，无法建立名称映射')
    model_index_by_name = {name: index for (index, name) in enumerate(normalized_model_names)}
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device_name = torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'
    with _test_record.timer.loading():
        model = model.to(device)
    with _test_record.timer.loading():
        model.eval()
    print(f'使用设备: {device_name}')
    print(f'模型: {MODEL_ID}')
    print('数据集: Stanford Cars 官方 test split')
    dataset = HFDataset.from_parquet([str(path) for path in shards], cache_dir=str(_test_record.output / '.dataset_cache'))
    if not {'image', 'label'}.issubset(set(dataset.column_names)):
        raise ValueError(f'Stanford Cars 列不完整，实际列：{dataset.column_names}')
    dataset_class_names = list(dataset.features['label'].names)
    if len(dataset_class_names) != model_label_count:
        raise ValueError(f'数据集类别数 {len(dataset_class_names)} 与模型类别数 {model_label_count} 不一致')
    normalized_dataset_names = [normalize_class_name(name) for name in dataset_class_names]
    missing = [name for name in normalized_dataset_names if name not in model_index_by_name]
    if missing:
        raise ValueError(f'模型缺少数据集类别：{missing}')
    dataset_to_model_label = [model_index_by_name[name] for name in normalized_dataset_names]
    wrapped_dataset = StanfordCarsTestDataset(dataset, processor, dataset_to_model_label)
    dataloader = _test_record.loader(DataLoader, wrapped_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=device.type == 'cuda')
    print(f'测试图片数量: {len(wrapped_dataset)}')
    print(f'类别数量: {model_label_count}')
    print(f'模型参数量: {sum((parameter.numel() for parameter in model.parameters())) / 1000000.0:.2f} M')
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
    (labels, predictions, top5_predictions) = ([], [], [])
    (losses, confidences) = ([], [])
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
            batch_top5 = logits.topk(5, dim=1).indices
            _test_record.prediction_ready(locals())
            batch_loss = F.cross_entropy(logits, batch_labels, reduction='none')
            labels.extend(batch_labels.cpu().tolist())
            predictions.extend(batch_predictions.cpu().tolist())
            top5_predictions.extend(batch_top5.cpu().tolist())
            losses.extend(batch_loss.cpu().tolist())
            confidences.extend(confidence.cpu().tolist())
            if batch_index % 25 == 0 or batch_index == len(dataloader):
                print(f'已处理: {min(batch_index * BATCH_SIZE, len(wrapped_dataset))}/{len(wrapped_dataset)}')
            _test_record.capture(locals())
    wall_seconds = time.perf_counter() - wall_start
    metrics = classification_metrics(labels, predictions, top5_predictions, losses, confidences, model_class_names)
    result = {'recorded_at': datetime.now().astimezone().isoformat(timespec='seconds'), 'model_id': MODEL_ID, 'model_path': str(MODEL_CACHE_DIR), 'script_path': str(Path(__file__).resolve()), 'model': {'id': MODEL_ID, 'architecture': model.__class__.__name__, 'parameter_count_millions': round(sum((parameter.numel() for parameter in model.parameters())) / 1000000.0, 4), 'num_labels': model_label_count, 'id2label': {str(key): value for (key, value) in model_id2label.items()}}, 'dataset': {'name': 'Stanford Cars', 'source': 'tanganke/stanford_cars', 'path': str(DATASET_DIR), 'split': 'test', 'parquet_shards': [str(path) for path in shards], 'image_count': len(wrapped_dataset), 'class_count': len(dataset_class_names), 'label_alignment': 'dataset labels mapped to model indices by normalized class name', 'dataset_class_names': dataset_class_names, 'dataset_to_model_label': dataset_to_model_label, 'split_note': 'Official Stanford Cars test split: 8,041 images across 196 car classes.'}, 'preprocessing': {'processor': 'AutoImageProcessor loaded from local model cache', 'image_size': 224, 'batch_size': BATCH_SIZE}, 'environment': {'device': str(device), 'device_name': device_name, 'torch_version': torch.__version__, 'cuda_available': torch.cuda.is_available()}, 'metrics': metrics, 'timing': {'warmup_steps': WARMUP_STEPS, 'forward_seconds': round(forward_seconds, 6), 'forward_ms_per_image': round(1000 * forward_seconds / len(wrapped_dataset), 6), 'forward_images_per_second': round(safe_divide(len(wrapped_dataset), forward_seconds), 4), 'wall_seconds_including_dataloader': round(wall_seconds, 6), 'wall_ms_per_image': round(1000 * wall_seconds / len(wrapped_dataset), 6), 'peak_cuda_memory_mib': round(torch.cuda.max_memory_allocated(device) / 1024 ** 2, 2) if device.type == 'cuda' else None}}
    save_result(result)
    print('\n评测完成')
    print(f"Top-1 accuracy: {metrics['top1_accuracy_pct']:.4f}%")
    print(f"Top-5 accuracy: {metrics['top5_accuracy_pct']:.4f}%")
    print(f"Macro-F1: {metrics['macro_f1_pct']:.4f}%")
    print(f"Balanced accuracy: {metrics['balanced_accuracy_pct']:.4f}%")
    print(f"Cross-entropy loss: {metrics['cross_entropy_loss']:.6f}")
    print(f'Forward time: {forward_seconds:.4f}s ({1000 * forward_seconds / len(wrapped_dataset):.4f} ms/image)')
    print(f'结果已保存: {RESULTS_PATH}')
    _test_record.finish(locals())
if __name__ == '__main__':
    main()
