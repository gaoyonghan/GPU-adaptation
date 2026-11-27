"""在 CelebA 官方 test split 上评测 dima806/attractive_faces_celebs_detection。"""
if __name__ == '__main__':
    import sys as _records_sys
    from pathlib import Path as _RecordsPath
    _records_sys.path.insert(0, str(_RecordsPath(__file__).resolve().parent))
    _records_sys.path.insert(1, str(_RecordsPath(__file__).resolve().parents[1]))
    from common import start as _start_records
    _test_record = _start_records(115, __file__, globals())
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
from collections import defaultdict
from datetime import datetime
from pathlib import Path
import torch
import torch.nn.functional as F
from datasets import Dataset as HFDataset
from torch.utils.data import DataLoader, Dataset
from transformers import AutoImageProcessor, AutoModelForImageClassification
MODEL_ID = 'dima806/attractive_faces_celebs_detection'
ROOT = __import__('model_paths').resolve_paths()['project']
MODEL_CACHE_DIR = __import__('model_paths').resolve_paths()['weights']
DATASET_DIR = __import__('model_paths').resolve_paths()['dataset'] / 'CelebA'
RESULTS_PATH = _test_record.output / 'metrics.json'
if __name__ == '__main__':
    RESULTS_PATH = _test_record.output / 'metrics.json'
BATCH_SIZE = int(os.environ.get('MODEL_BATCH_SIZE', '64'))
NUM_WORKERS = int(os.environ.get('MODEL_WORKERS', '0'))
WARMUP_STEPS = int(os.environ.get('MODEL_WARMUP', '3'))
ECE_BINS = 15
CLASS_NAMES = ['attractive', 'not attractive']

class CelebATestDataset(Dataset):

    def __init__(self, dataset, processor):
        self.dataset = dataset
        self.processor = processor

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        example = self.dataset[index]
        label = 0 if bool(example['Attractive']) else 1
        pixel_values = self.processor(images=example['image'].convert('RGB'), return_tensors='pt')['pixel_values'].squeeze(0)
        return (pixel_values, label, int(bool(example['Male'])), int(bool(example['Young'])))

def safe_divide(numerator, denominator):
    return float(numerator) / float(denominator) if denominator else 0.0

def roc_auc(labels, positive_scores):
    """ROC-AUC with average ranks for tied scores; class 0 is positive."""
    positive_indices = [index for (index, label) in enumerate(labels) if label == 0]
    negative_indices = [index for (index, label) in enumerate(labels) if label == 1]
    if not positive_indices or not negative_indices:
        return None
    ranked = sorted(enumerate(positive_scores), key=lambda item: item[1])
    rank_sum = 0.0
    index = 0
    while index < len(ranked):
        end = index + 1
        while end < len(ranked) and ranked[end][1] == ranked[index][1]:
            end += 1
        average_rank = (index + 1 + end) / 2.0
        rank_sum += sum((average_rank for (item_index, _) in ranked[index:end] if labels[item_index] == 0))
        index = end
    positive_count = len(positive_indices)
    negative_count = len(negative_indices)
    return safe_divide(rank_sum - positive_count * (positive_count + 1) / 2.0, positive_count * negative_count)

def binary_metrics(labels, predictions, positive_scores, losses, confidences):
    confusion = [[0, 0], [0, 0]]
    for (label, prediction) in zip(labels, predictions):
        confusion[label][prediction] += 1
    per_class = {}
    recalls = []
    f1_scores = []
    for (class_index, class_name) in enumerate(CLASS_NAMES):
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
    correct = sum((label == prediction for (label, prediction) in zip(labels, predictions)))
    return {'image_count': len(labels), 'accuracy_pct': round(100 * safe_divide(correct, len(labels)), 4), 'balanced_accuracy_pct': round(50 * sum(recalls), 4), 'macro_f1_pct': round(50 * sum(f1_scores), 4), 'roc_auc': None if roc_auc(labels, positive_scores) is None else round(roc_auc(labels, positive_scores), 6), 'cross_entropy_loss': round(safe_divide(sum(losses), len(labels)), 6), 'expected_calibration_error': round(ece, 6), 'per_class': per_class, 'confusion_matrix_rows_true_cols_pred': confusion}

def subgroup_accuracy(group_values, labels, predictions, group_names):
    grouped = defaultdict(lambda : [0, 0])
    for (group, label, prediction) in zip(group_values, labels, predictions):
        grouped[int(group)][0] += 1
        grouped[int(group)][1] += int(label == prediction)
    return {group_names[index]: {'image_count': values[0], 'accuracy_pct': round(100 * safe_divide(values[1], values[0]), 4)} for (index, values) in sorted(grouped.items())}

def find_test_shards():
    shards = sorted(DATASET_DIR.glob('img_align+identity+attr/test-*.parquet'))
    if not shards:
        raise FileNotFoundError(f'未找到 CelebA test parquet：{DATASET_DIR}')
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
    id2label = {int(key): value for (key, value) in model.config.id2label.items()}
    if model.config.num_labels != 2 or [id2label[index] for index in range(2)] != CLASS_NAMES:
        raise ValueError(f'模型标签应为 {CLASS_NAMES}，实际为 {id2label}')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device_name = torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'
    with _test_record.timer.loading():
        model = model.to(device)
    with _test_record.timer.loading():
        model.eval()
    print(f'使用设备: {device_name}')
    print(f'模型: {MODEL_ID}')
    print('数据集: CelebA 官方 test split，Attractive 属性')
    dataset = HFDataset.from_parquet([str(path) for path in shards], cache_dir=str(_test_record.output / '.dataset_cache'))
    expected_columns = {'image', 'Attractive', 'Male', 'Young'}
    if not expected_columns.issubset(set(dataset.column_names)):
        raise ValueError(f'CelebA 列不完整，实际列：{dataset.column_names}')
    wrapped_dataset = CelebATestDataset(dataset, processor)
    dataloader = _test_record.loader(DataLoader, wrapped_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=device.type == 'cuda')
    print(f'测试图片数量: {len(wrapped_dataset)}')
    print(f'模型参数量: {sum((parameter.numel() for parameter in model.parameters())) / 1000000.0:.2f} M')
    print(f'batch size: {BATCH_SIZE}')
    (warmup_pixel_values, *_) = next(iter(dataloader))
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
    (labels, predictions, positive_scores) = ([], [], [])
    (losses, confidences, male_values, young_values) = ([], [], [], [])
    forward_seconds = 0.0
    wall_start = time.perf_counter()
    with torch.inference_mode():
        for (batch_index, batch) in enumerate(dataloader, start=1):
            _test_record.before_batch(locals())
            (pixel_values, batch_labels, batch_male, batch_young) = batch
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
            _test_record.prediction_ready(locals())
            batch_loss = F.cross_entropy(logits, batch_labels, reduction='none')
            labels.extend(batch_labels.cpu().tolist())
            predictions.extend(batch_predictions.cpu().tolist())
            positive_scores.extend(probabilities[:, 0].cpu().tolist())
            losses.extend(batch_loss.cpu().tolist())
            confidences.extend(confidence.cpu().tolist())
            male_values.extend(batch_male.tolist())
            young_values.extend(batch_young.tolist())
            if batch_index % 50 == 0 or batch_index == len(dataloader):
                print(f'已处理: {min(batch_index * BATCH_SIZE, len(wrapped_dataset))}/{len(wrapped_dataset)}')
            _test_record.capture(locals())
    wall_seconds = time.perf_counter() - wall_start
    metrics = binary_metrics(labels, predictions, positive_scores, losses, confidences)
    metrics['subgroup_accuracy'] = {'gender': subgroup_accuracy(male_values, labels, predictions, ['Female', 'Male']), 'young_attribute': subgroup_accuracy(young_values, labels, predictions, ['not Young', 'Young'])}
    result = {'recorded_at': datetime.now().astimezone().isoformat(timespec='seconds'), 'model_id': MODEL_ID, 'model_path': str(MODEL_CACHE_DIR), 'script_path': str(Path(__file__).resolve()), 'model': {'id': MODEL_ID, 'architecture': model.__class__.__name__, 'parameter_count_millions': round(sum((parameter.numel() for parameter in model.parameters())) / 1000000.0, 4), 'num_labels': model.config.num_labels, 'id2label': {str(key): value for (key, value) in id2label.items()}}, 'dataset': {'name': 'CelebA', 'source': 'flwrlabs/celeba', 'path': str(DATASET_DIR), 'split': 'test', 'parquet_shards': [str(path) for path in shards], 'image_count': len(wrapped_dataset), 'target_attribute': 'Attractive', 'target_mapping': {'True': 'attractive (class 0)', 'False': 'not attractive (class 1)'}, 'split_note': 'Official CelebA test split with the original binary Attractive attribute.'}, 'preprocessing': {'processor': 'AutoImageProcessor loaded from local model cache', 'image_size': 224, 'batch_size': BATCH_SIZE}, 'environment': {'device': str(device), 'device_name': device_name, 'torch_version': torch.__version__, 'cuda_available': torch.cuda.is_available()}, 'official_model_card_reference': {'accuracy_pct': 83.78, 'macro_f1_pct': 83.78, 'image_count': 10384, 'note': "Model-card values are reported on the author's 10,384-image attractiveness evaluation set, not this CelebA test split."}, 'metrics': metrics, 'timing': {'warmup_steps': WARMUP_STEPS, 'forward_seconds': round(forward_seconds, 6), 'forward_ms_per_image': round(1000 * forward_seconds / len(wrapped_dataset), 6), 'forward_images_per_second': round(safe_divide(len(wrapped_dataset), forward_seconds), 4), 'wall_seconds_including_dataloader': round(wall_seconds, 6), 'wall_ms_per_image': round(1000 * wall_seconds / len(wrapped_dataset), 6), 'peak_cuda_memory_mib': round(torch.cuda.max_memory_allocated(device) / 1024 ** 2, 2) if device.type == 'cuda' else None}}
    save_result(result)
    print('\n评测完成')
    print(f"Accuracy: {metrics['accuracy_pct']:.4f}%")
    print(f"ROC-AUC: {metrics['roc_auc']:.6f}")
    print(f"Macro-F1: {metrics['macro_f1_pct']:.4f}%")
    print(f"Cross-entropy loss: {metrics['cross_entropy_loss']:.6f}")
    print(f'Forward time: {forward_seconds:.4f}s ({1000 * forward_seconds / len(wrapped_dataset):.4f} ms/image)')
    print(f'结果已保存: {RESULTS_PATH}')
    _test_record.finish(locals())
if __name__ == '__main__':
    main()
