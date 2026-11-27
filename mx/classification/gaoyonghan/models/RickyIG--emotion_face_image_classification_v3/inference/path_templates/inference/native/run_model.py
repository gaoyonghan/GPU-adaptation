"""Evaluate RickyIG/emotion_face_image_classification_v3 on FER+ privateTest.

FER+ supplies crowd-sourced vote annotations for the FER2013 images. This script
uses Microsoft's majority-vote rule to construct valid eight-class hard labels
for the FER2013 PrivateTest images already stored locally.
"""
if __name__ == '__main__':
    import sys as _records_sys
    from pathlib import Path as _RecordsPath
    _records_sys.path.insert(0, str(_RecordsPath(__file__).resolve().parent))
    _records_sys.path.insert(1, str(_RecordsPath(__file__).resolve().parents[1]))
    from common import start as _start_records
    _test_record = _start_records(142, __file__, globals())
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
import time
from collections import Counter
from datetime import datetime
from io import BytesIO
from pathlib import Path
import torch
import torch.nn.functional as F
from datasets import Dataset as HFDataset
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from transformers import AutoImageProcessor, AutoModelForImageClassification
MODEL_ID = 'RickyIG/emotion_face_image_classification_v3'
ROOT = __import__('model_paths').resolve_paths()['project']
MODEL_CACHE_DIR = __import__('model_paths').resolve_paths()['weights']
FER2013_DIR = __import__('model_paths').resolve_paths()['dataset'] / 'fer2013'
PRIVATE_TEST_PATH = FER2013_DIR / 'data/privateTest-00000-of-00001-4b8a0715cf1b7560.parquet'
FERPLUS_DIR = __import__('model_paths').resolve_paths()['dataset'] / 'FERPlus'
FERPLUS_LABELS_PATH = FERPLUS_DIR / 'fer2013new.csv'
RESULTS_PATH = _test_record.output / 'metrics.json'
if __name__ == '__main__':
    RESULTS_PATH = _test_record.output / 'metrics.json'
BATCH_SIZE = int(os.environ.get('MODEL_BATCH_SIZE', '64'))
NUM_WORKERS = int(os.environ.get('MODEL_WORKERS', '0'))
WARMUP_STEPS = int(os.environ.get('MODEL_WARMUP', '3'))
ECE_BINS = 15
MODEL_LABELS = ['anger', 'contempt', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
FERPLUS_VOTE_COLUMNS = ['neutral', 'happiness', 'surprise', 'sadness', 'anger', 'disgust', 'fear', 'contempt', 'unknown', 'NF']
FERPLUS_EMOTIONS = FERPLUS_VOTE_COLUMNS[:8]
FERPLUS_TO_MODEL_LABEL = {'neutral': 5, 'happiness': 4, 'surprise': 7, 'sadness': 6, 'anger': 0, 'disgust': 2, 'fear': 3, 'contempt': 1}

class FERPlusTestDataset(Dataset):

    def __init__(self, dataset, labels, processor):
        self.dataset = dataset
        self.labels = labels
        self.processor = processor

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        image_data = self.dataset[index]['image']
        image = Image.open(BytesIO(image_data['bytes'])).convert('RGB')
        pixel_values = self.processor(images=image, return_tensors='pt')['pixel_values'].squeeze(0)
        return (pixel_values, self.labels[index])

def safe_divide(numerator, denominator):
    return float(numerator) / float(denominator) if denominator else 0.0

def majority_label(votes):
    """Apply Microsoft FER+ majority-mode processing to one vote vector."""
    cleaned_votes = [0 if vote <= 1 else vote for vote in votes]
    vote_total = sum(cleaned_votes)
    if vote_total == 0:
        return (None, 'no_valid_votes')
    label_index = max(range(len(cleaned_votes)), key=cleaned_votes.__getitem__)
    if cleaned_votes[label_index] <= 0.5 * vote_total:
        return (None, 'no_strict_majority')
    if label_index >= len(FERPLUS_EMOTIONS):
        return (None, 'unknown_or_non_face')
    return (FERPLUS_TO_MODEL_LABEL[FERPLUS_EMOTIONS[label_index]], None)

def load_ferplus_private_test_labels():
    if not FERPLUS_LABELS_PATH.is_file():
        raise FileNotFoundError(f'未找到 FER+ 官方标注：{FERPLUS_LABELS_PATH}')
    targets = []
    accepted_counts = Counter()
    excluded_counts = Counter()
    with FERPLUS_LABELS_PATH.open(encoding='utf-8', newline='') as handle:
        for row in csv.DictReader(handle):
            if row['Usage'] != 'PrivateTest':
                continue
            votes = [int(row[column]) for column in FERPLUS_VOTE_COLUMNS]
            (target, exclusion_reason) = majority_label(votes)
            targets.append(target)
            if target is None:
                excluded_counts[exclusion_reason] += 1
            else:
                accepted_counts[MODEL_LABELS[target]] += 1
    if not targets:
        raise ValueError('FER+ 标注中没有 PrivateTest 行')
    return (targets, accepted_counts, excluded_counts)

def expected_calibration_error(labels, predictions, confidences):
    ece = 0.0
    for bin_index in range(ECE_BINS):
        lower = bin_index / ECE_BINS
        upper = (bin_index + 1) / ECE_BINS
        indices = [index for (index, confidence) in enumerate(confidences) if lower <= confidence < upper or (bin_index == ECE_BINS - 1 and confidence == upper)]
        if indices:
            accuracy = sum((labels[index] == predictions[index] for index in indices)) / len(indices)
            mean_confidence = sum((confidences[index] for index in indices)) / len(indices)
            ece += len(indices) / len(labels) * abs(accuracy - mean_confidence)
    return ece

def classification_metrics(labels, predictions, top3_predictions, losses, confidences, ranks):
    class_count = len(MODEL_LABELS)
    confusion = [[0 for _ in range(class_count)] for _ in range(class_count)]
    for (label, prediction) in zip(labels, predictions):
        confusion[label][prediction] += 1
    per_class = {}
    precisions = []
    recalls = []
    f1_scores = []
    for (class_index, class_name) in enumerate(MODEL_LABELS):
        true_positive = confusion[class_index][class_index]
        support = sum(confusion[class_index])
        predicted = sum((row[class_index] for row in confusion))
        precision = safe_divide(true_positive, predicted)
        recall = safe_divide(true_positive, support)
        f1 = safe_divide(2.0 * precision * recall, precision + recall)
        precisions.append(precision)
        recalls.append(recall)
        f1_scores.append(f1)
        per_class[class_name] = {'support': support, 'precision_pct': round(100.0 * precision, 4), 'recall_pct': round(100.0 * recall, 4), 'f1_pct': round(100.0 * f1, 4)}
    image_count = len(labels)
    top1_correct = sum((label == prediction for (label, prediction) in zip(labels, predictions)))
    top3_correct = sum((label in candidates for (label, candidates) in zip(labels, top3_predictions)))
    return {'image_count': image_count, 'top1_accuracy_pct': round(100.0 * safe_divide(top1_correct, image_count), 4), 'top3_accuracy_pct': round(100.0 * safe_divide(top3_correct, image_count), 4), 'mean_reciprocal_rank': round(safe_divide(sum((1.0 / rank for rank in ranks)), image_count), 6), 'macro_precision_pct': round(100.0 * sum(precisions) / class_count, 4), 'macro_recall_pct': round(100.0 * sum(recalls) / class_count, 4), 'balanced_accuracy_pct': round(100.0 * sum(recalls) / class_count, 4), 'macro_f1_pct': round(100.0 * sum(f1_scores) / class_count, 4), 'cross_entropy_loss': round(safe_divide(sum(losses), image_count), 6), 'expected_calibration_error': round(expected_calibration_error(labels, predictions, confidences), 6), 'mean_prediction_confidence_pct': round(100.0 * safe_divide(sum(confidences), image_count), 4), 'per_class': per_class, 'confusion_matrix_rows_true_cols_pred': confusion, 'confusion_matrix_labels': MODEL_LABELS}

def save_result(result):
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = RESULTS_PATH.with_suffix('.json.tmp')
    temporary_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    os.replace(temporary_path, RESULTS_PATH)

def main():
    if not MODEL_CACHE_DIR.is_dir():
        raise FileNotFoundError(f'未找到模型缓存：{MODEL_CACHE_DIR}')
    if not PRIVATE_TEST_PATH.is_file():
        raise FileNotFoundError(f'未找到 FER2013 PrivateTest 图像：{PRIVATE_TEST_PATH}')
    with _test_record.timer.loading():
        processor = AutoImageProcessor.from_pretrained(str(MODEL_CACHE_DIR), local_files_only=True)
    with _test_record.timer.loading():
        model = AutoModelForImageClassification.from_pretrained(str(MODEL_CACHE_DIR), local_files_only=True)
    id2label = {int(index): label for (index, label) in model.config.id2label.items()}
    model_labels = [id2label[index] for index in range(model.config.num_labels)]
    if model_labels != MODEL_LABELS:
        raise ValueError(f'模型标签顺序应为 {MODEL_LABELS}，实际为 {model_labels}')
    full_test = HFDataset.from_parquet(str(PRIVATE_TEST_PATH), cache_dir=str(FER2013_DIR / '.cache'))
    if set(full_test.column_names) != {'label', 'image'}:
        raise ValueError(f'FER2013 PrivateTest 列应为 label、image，实际为 {full_test.column_names}')
    (all_targets, accepted_counts, excluded_counts) = load_ferplus_private_test_labels()
    if len(full_test) != len(all_targets):
        raise ValueError(f'FER2013 PrivateTest 图像数与 FER+ PrivateTest 标注数不一致：{len(full_test)} != {len(all_targets)}')
    selected_indices = [index for (index, target) in enumerate(all_targets) if target is not None]
    selected_labels = [all_targets[index] for index in selected_indices]
    test_subset = full_test.select(selected_indices)
    wrapped_dataset = FERPlusTestDataset(test_subset, selected_labels, processor)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device_name = torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'
    with _test_record.timer.loading():
        model = model.to(device).eval()
    dataloader = _test_record.loader(DataLoader, wrapped_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=device.type == 'cuda')
    print(f'使用设备: {device_name}')
    print(f'模型: {MODEL_ID}')
    print('数据集: FER+ PrivateTest（微软 majority-vote hard labels）')
    print(f'PrivateTest 原始图片数量: {len(full_test)}')
    print(f'有有效单标签的测试图片数量: {len(wrapped_dataset)}')
    print(f'类别数量: {len(MODEL_LABELS)}')
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
    metrics = classification_metrics(labels, predictions, top3_predictions, losses, confidences, ranks)
    result = {'recorded_at': datetime.now().astimezone().isoformat(timespec='seconds'), 'model_id': MODEL_ID, 'model_path': str(MODEL_CACHE_DIR), 'script_path': str(Path(__file__).resolve()), 'model': {'id': MODEL_ID, 'architecture': model.__class__.__name__, 'parameter_count_millions': round(sum((parameter.numel() for parameter in model.parameters())) / 1000000.0, 4), 'num_labels': model.config.num_labels, 'classifier_labels': model_labels, 'scope_note': '模型仓库无 model card；八类标签与 FER+ 兼容，但无法从本地缓存确认其训练数据。'}, 'dataset': {'name': 'FER+', 'images_source': 'FER2013 PrivateTest images from Aaryan333/fer2013_train_publicTest_privateTest', 'labels_source': 'microsoft/FERPlus fer2013new.csv', 'images_path': str(PRIVATE_TEST_PATH), 'labels_path': str(FERPLUS_LABELS_PATH), 'source_split': 'PrivateTest', 'source_image_count': len(full_test), 'evaluated_image_count': len(wrapped_dataset), 'evaluated_coverage_pct': round(100.0 * len(wrapped_dataset) / len(full_test), 4), 'accepted_class_counts': dict(accepted_counts), 'excluded_image_counts': dict(excluded_counts), 'label_protocol': 'Microsoft FER+ majority mode: remove one-vote outliers, require a strict majority (>50% of valid votes), then discard unknown/non-face or ambiguous samples.', 'alignment_note': 'FER+ annotation rows and FER2013 PrivateTest images are paired by their shared PrivateTest row order; both contain 3589 rows.'}, 'environment': {'device': device_name, 'precision': 'FP32', 'batch_size': BATCH_SIZE, 'warmup_steps': WARMUP_STEPS, 'num_workers': NUM_WORKERS}, 'preprocessing': {'input': 'FER2013 48×48 image bytes decoded to RGB', 'processor': processor.__class__.__name__, 'resize': '224×224, bilinear', 'rescale': '1/255', 'normalize': '(x - 0.5) / 0.5 per RGB channel', 'augmentations': 'none'}, 'official_reference': {'model_card_metric': None, 'comparability': '模型仓库没有 model card 或作者公布的 FER+ 指标；本结果可与相同 FER+ majority-label PrivateTest 协议的结果比较。'}, 'accuracy': metrics, 'timing': {'timed_split': 'FER+ PrivateTest valid majority-label subset only', 'model_forward_total_seconds': round(forward_seconds, 6), 'model_latency_ms_per_image': round(forward_seconds / len(wrapped_dataset) * 1000.0, 6), 'model_throughput_images_per_second': round(len(wrapped_dataset) / forward_seconds, 6), 'end_to_end_total_seconds': round(wall_seconds, 6), 'end_to_end_latency_ms_per_image': round(wall_seconds / len(wrapped_dataset) * 1000.0, 6), 'end_to_end_throughput_images_per_second': round(len(wrapped_dataset) / wall_seconds, 6)}}
    if device.type == 'cuda':
        result['environment']['peak_gpu_memory_mib'] = round(torch.cuda.max_memory_allocated(device) / 1024 ** 2, 4)
    save_result(result)
    print('=' * 60)
    print(f"Top-1 accuracy: {metrics['top1_accuracy_pct']:.4f}%")
    print(f"Top-3 accuracy: {metrics['top3_accuracy_pct']:.4f}%")
    print(f"Macro-F1: {metrics['macro_f1_pct']:.4f}%")
    print(f"Balanced accuracy: {metrics['balanced_accuracy_pct']:.4f}%")
    print(f"Cross-entropy loss: {metrics['cross_entropy_loss']:.6f}")
    print(f"模型平均推理时间: {result['timing']['model_latency_ms_per_image']:.4f} ms/image")
    print(f"模型吞吐量: {result['timing']['model_throughput_images_per_second']:.2f} images/s")
    print(f'评测结果已更新: {RESULTS_PATH}')
    _test_record.finish(locals())
if __name__ == '__main__':
    main()
