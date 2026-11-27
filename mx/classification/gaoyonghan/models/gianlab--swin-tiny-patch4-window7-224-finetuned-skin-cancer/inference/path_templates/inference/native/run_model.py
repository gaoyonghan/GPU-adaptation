"""Evaluate gianlab's Swin skin-cancer classifier on a local HAM10000 holdout.

The cached HAM10000 repository contains one labelled train split (10,015
images), not an official test split.  This script therefore creates a fixed,
stratified 20% holdout and records that fact in the result JSON.
"""
if __name__ == '__main__':
    import sys as _records_sys
    from pathlib import Path as _RecordsPath
    _records_sys.path.insert(0, str(_RecordsPath(__file__).resolve().parent))
    _records_sys.path.insert(1, str(_RecordsPath(__file__).resolve().parents[1]))
    from common import start as _start_records
    _test_record = _start_records(135, __file__, globals())
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
MODEL_ID = 'gianlab/swin-tiny-patch4-window7-224-finetuned-skin-cancer'
ROOT = __import__('model_paths').resolve_paths()['project']
MODEL_CACHE_DIR = __import__('model_paths').resolve_paths()['weights']
DATASET_DIR = __import__('model_paths').resolve_paths()['dataset'] / 'HAM10000'
RESULTS_PATH = _test_record.output / 'metrics.json'
if __name__ == '__main__':
    RESULTS_PATH = _test_record.output / 'metrics.json'
BATCH_SIZE = int(os.environ.get('MODEL_BATCH_SIZE', '64'))
NUM_WORKERS = int(os.environ.get('MODEL_WORKERS', '0'))
WARMUP_STEPS = int(os.environ.get('MODEL_WARMUP', '3'))
HOLDOUT_FRACTION = 0.2
SPLIT_SEED = 42
ECE_BINS = 15
HAM_LABEL_NAMES = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']
MODEL_LABEL_NAMES = ['Actinic-keratoses', 'Basal-cell-carcinoma', 'Benign-keratosis-like-lesions', 'Dermatofibroma', 'Melanocytic-nevi', 'Melanoma', 'Vascular-lesions']
HAM_TO_MODEL_LABEL = {'akiec': 0, 'bcc': 1, 'bkl': 2, 'df': 3, 'nv': 4, 'mel': 5, 'vasc': 6}

def safe_divide(numerator, denominator):
    return float(numerator) / float(denominator) if denominator else 0.0

def find_train_shards():
    shards = sorted(DATASET_DIR.glob('data/train-*.parquet'))
    if not shards:
        raise FileNotFoundError(f'未找到 HAM10000 parquet：{DATASET_DIR}')
    return shards

class HAM10000Dataset(Dataset):

    def __init__(self, dataset, processor, dataset_to_model_label):
        self.dataset = dataset
        self.processor = processor
        self.dataset_to_model_label = dataset_to_model_label

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        example = self.dataset[index]
        dataset_label = int(example['label'])
        image = example['image'].convert('RGB')
        pixel_values = self.processor(images=image, return_tensors='pt')['pixel_values'].squeeze(0)
        return (pixel_values, self.dataset_to_model_label[dataset_label])

def classification_metrics(labels, predictions, top5_predictions, losses, confidences, class_names):
    class_count = len(class_names)
    confusion = [[0 for _ in range(class_count)] for _ in range(class_count)]
    for (label, prediction) in zip(labels, predictions):
        confusion[int(label)][int(prediction)] += 1
    per_class = {}
    recalls = []
    precisions = []
    f1_scores = []
    for (class_index, class_name) in enumerate(class_names):
        true_positive = confusion[class_index][class_index]
        false_positive = sum((row[class_index] for row in confusion)) - true_positive
        false_negative = sum(confusion[class_index]) - true_positive
        precision = safe_divide(true_positive, true_positive + false_positive)
        recall = safe_divide(true_positive, true_positive + false_negative)
        f1 = safe_divide(2 * precision * recall, precision + recall)
        precisions.append(precision)
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
    image_count = len(labels)
    return {'image_count': image_count, 'top1_accuracy_pct': round(100 * safe_divide(top1_correct, image_count), 4), 'top5_accuracy_pct': round(100 * safe_divide(top5_correct, image_count), 4), 'macro_precision_pct': round(100 * sum(precisions) / class_count, 4), 'macro_recall_pct': round(100 * sum(recalls) / class_count, 4), 'balanced_accuracy_pct': round(100 * sum(recalls) / class_count, 4), 'macro_f1_pct': round(100 * sum(f1_scores) / class_count, 4), 'cross_entropy_loss': round(safe_divide(sum(losses), image_count), 6), 'expected_calibration_error': round(ece, 6), 'per_class': per_class, 'confusion_matrix_rows_true_cols_pred': confusion, 'top_confusion_pairs': top_confusions[:30]}

def save_result(result):
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = RESULTS_PATH.with_suffix('.json.tmp')
    temporary_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    os.replace(temporary_path, RESULTS_PATH)

def main():
    if not MODEL_CACHE_DIR.is_dir():
        raise FileNotFoundError(f'未找到模型缓存：{MODEL_CACHE_DIR}')
    shards = find_train_shards()
    with _test_record.timer.loading():
        processor = AutoImageProcessor.from_pretrained(str(MODEL_CACHE_DIR), local_files_only=True)
    with _test_record.timer.loading():
        model = AutoModelForImageClassification.from_pretrained(str(MODEL_CACHE_DIR), local_files_only=True)
    id2label = {int(key): value for (key, value) in model.config.id2label.items()}
    model_class_names = [id2label[index] for index in range(model.config.num_labels)]
    if model_class_names != MODEL_LABEL_NAMES:
        raise ValueError(f'模型标签顺序变化，期望 {MODEL_LABEL_NAMES}，实际 {model_class_names}')
    if model.config.num_labels != len(MODEL_LABEL_NAMES):
        raise ValueError(f'模型应为 7 类，实际为 {model.config.num_labels} 类')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device_name = torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'
    with _test_record.timer.loading():
        model = model.to(device)
    with _test_record.timer.loading():
        model.eval()
    print(f'使用设备: {device_name}')
    print(f'模型: {MODEL_ID}')
    print('数据集: 本地 HAM10000 的固定分层 20% holdout（无官方 test split）')
    dataset = HFDataset.from_parquet([str(path) for path in shards], cache_dir=str(_test_record.output / '.dataset_cache'))
    if set(dataset.column_names) != {'image', 'label'}:
        raise ValueError(f'HAM10000 列应为 image、label，实际为：{dataset.column_names}')
    dataset_label_names = list(dataset.features['label'].names)
    if dataset_label_names != HAM_LABEL_NAMES:
        raise ValueError(f'HAM10000 标签顺序变化，期望 {HAM_LABEL_NAMES}，实际 {dataset_label_names}')
    dataset_to_model_label = [HAM_TO_MODEL_LABEL[name] for name in dataset_label_names]
    split = dataset.train_test_split(test_size=HOLDOUT_FRACTION, seed=SPLIT_SEED, stratify_by_column='label')
    test_dataset = split['test']
    wrapped_dataset = HAM10000Dataset(test_dataset, processor, dataset_to_model_label)
    dataloader = _test_record.loader(DataLoader, wrapped_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=device.type == 'cuda')
    print(f'原始图片数量: {len(dataset)}')
    print(f'测试图片数量: {len(wrapped_dataset)}')
    print(f'类别数量: {len(MODEL_LABEL_NAMES)}')
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
    labels = []
    predictions = []
    top5_predictions = []
    losses = []
    confidences = []
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
            if batch_index % 10 == 0 or batch_index == len(dataloader):
                print(f'已处理: {min(batch_index * BATCH_SIZE, len(wrapped_dataset))}/{len(wrapped_dataset)}')
            _test_record.capture(locals())
    if device.type == 'cuda':
        torch.cuda.synchronize()
    wall_seconds = time.perf_counter() - wall_start
    metrics = classification_metrics(labels, predictions, top5_predictions, losses, confidences, MODEL_LABEL_NAMES)
    image_count = len(wrapped_dataset)
    result = {'recorded_at': datetime.now().astimezone().isoformat(timespec='seconds'), 'model_id': MODEL_ID, 'model_path': str(MODEL_CACHE_DIR), 'script_path': str(Path(__file__).resolve()), 'model': {'id': MODEL_ID, 'architecture': model.__class__.__name__, 'parameter_count_millions': round(sum((parameter.numel() for parameter in model.parameters())) / 1000000.0, 4), 'num_labels': len(MODEL_LABEL_NAMES), 'id2label': {str(index): name for (index, name) in id2label.items()}}, 'dataset': {'name': 'HAM10000', 'source': 'kuchikihater/HAM10000 (local parquet cache)', 'path': str(DATASET_DIR), 'source_split': 'train', 'evaluated_split': 'stratified_holdout', 'split_note': 'HAM10000 原始数据及本地仓库只有一个带标签 train split；按 seed=42 分层抽取 20% 作为测试 holdout，非官方 test split。', 'split_seed': SPLIT_SEED, 'holdout_fraction': HOLDOUT_FRACTION, 'image_count': image_count, 'source_image_count': len(dataset), 'class_count': len(MODEL_LABEL_NAMES), 'class_names': MODEL_LABEL_NAMES}, 'environment': {'device': device_name, 'precision': 'FP32', 'batch_size': BATCH_SIZE, 'warmup_steps': WARMUP_STEPS, 'num_workers': NUM_WORKERS}, 'preprocessing': {'processor': processor.__class__.__name__, 'image_size': [224, 224], 'normalization_mean': [0.485, 0.456, 0.406], 'normalization_std': [0.229, 0.224, 0.225], 'label_mapping': HAM_TO_MODEL_LABEL}, 'official_reference': {'model_card_eval_accuracy_pct': 72.75, 'model_card_eval_loss': 0.7695, 'comparability': '仅作参考；模型卡的 imagefolder eval split 与本 holdout 不保证相同。'}, 'accuracy': metrics, 'timing': {'model_forward_total_seconds': round(forward_seconds, 6), 'model_latency_ms_per_image': round(forward_seconds / image_count * 1000.0, 6), 'model_throughput_images_per_second': round(image_count / forward_seconds, 6), 'end_to_end_total_seconds': round(wall_seconds, 6), 'end_to_end_latency_ms_per_image': round(wall_seconds / image_count * 1000.0, 6), 'end_to_end_throughput_images_per_second': round(image_count / wall_seconds, 6)}}
    if device.type == 'cuda':
        result['environment']['peak_gpu_memory_mib'] = round(torch.cuda.max_memory_allocated(device) / 1024 ** 2, 4)
    save_result(result)
    print('=' * 60)
    print(f"Top-1 accuracy: {metrics['top1_accuracy_pct']:.2f}%")
    print(f"Top-5 accuracy: {metrics['top5_accuracy_pct']:.2f}%")
    print(f"Macro-F1: {metrics['macro_f1_pct']:.2f}%")
    print(f"Balanced accuracy: {metrics['balanced_accuracy_pct']:.2f}%")
    print(f"Cross-entropy loss: {metrics['cross_entropy_loss']:.6f}")
    print(f"模型平均推理时间: {result['timing']['model_latency_ms_per_image']:.4f} ms/image")
    print(f"模型吞吐量: {result['timing']['model_throughput_images_per_second']:.2f} images/s")
    print(f'评测结果已更新: {RESULTS_PATH}')
    _test_record.finish(locals())
if __name__ == '__main__':
    main()
