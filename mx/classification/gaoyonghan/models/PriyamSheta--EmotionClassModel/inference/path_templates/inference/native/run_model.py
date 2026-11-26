"""Evaluate PriyamSheta/EmotionClassModel on FER2013 privateTest.

The model has only Happy and Sad outputs, while FER2013 has seven emotions.
Metrics are therefore calculated only on the Happy/Sad subset of privateTest;
the result explicitly records that this is not a seven-class FER2013 score.
"""
if __name__ == '__main__':
    import sys as _records_sys
    from pathlib import Path as _RecordsPath
    _records_sys.path.insert(0, str(_RecordsPath(__file__).resolve().parent))
    _records_sys.path.insert(1, str(_RecordsPath(__file__).resolve().parents[1]))
    from common import start as _start_records
    _test_record = _start_records(139, __file__, globals())
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
from io import BytesIO
from datetime import datetime
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from datasets import Dataset as HFDataset
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForImageClassification
try:
    from transformers.models.vit.image_processing_pil_vit import ViTImageProcessorPil as LegacyPilProcessor
except ImportError:
    from transformers import ViTImageProcessor as LegacyPilProcessor
MODEL_ID = 'PriyamSheta/EmotionClassModel'
ROOT = __import__('model_paths').resolve_paths()['project']
MODEL_CACHE_DIR = __import__('model_paths').resolve_paths()['weights']
DATASET_DIR = __import__('model_paths').resolve_paths()['dataset'] / 'fer2013'
TEST_PATH = DATASET_DIR / 'data/privateTest-00000-of-00001-4b8a0715cf1b7560.parquet'
RESULTS_PATH = _test_record.output / 'metrics.json'
if __name__ == '__main__':
    RESULTS_PATH = _test_record.output / 'metrics.json'
BATCH_SIZE = int(os.environ.get('MODEL_BATCH_SIZE', '64'))
NUM_WORKERS = int(os.environ.get('MODEL_WORKERS', '0'))
WARMUP_STEPS = int(os.environ.get('MODEL_WARMUP', '3'))
ECE_BINS = 15
FER2013_LABELS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
FER_TO_MODEL_LABEL = {3: 0, 4: 1}
MODEL_LABELS = ['Happy', 'Sad']

class FER2013HappySadDataset(Dataset):

    def __init__(self, dataset, processor):
        self.dataset = dataset
        self.processor = processor

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        example = self.dataset[index]
        image_data = example['image']
        image = Image.open(BytesIO(image_data['bytes'])).convert('RGB')
        pixel_values = self.processor(images=image, return_tensors='pt')['pixel_values'].squeeze(0)
        return (pixel_values, FER_TO_MODEL_LABEL[int(example['label'])])

def safe_divide(numerator, denominator):
    return float(numerator) / float(denominator) if denominator else 0.0

def binary_roc_auc(labels, sad_probabilities):
    labels = np.asarray(labels, dtype=bool)
    scores = np.asarray(sad_probabilities, dtype=np.float64)
    positive_count = int(labels.sum())
    negative_count = int((~labels).sum())
    if positive_count == 0 or negative_count == 0:
        return 0.0
    order = np.argsort(-scores, kind='mergesort')
    sorted_scores = scores[order]
    sorted_labels = labels[order]
    true_positive = np.cumsum(sorted_labels)
    false_positive = np.cumsum(~sorted_labels)
    last_indices = np.r_[np.where(np.diff(sorted_scores))[0], len(scores) - 1]
    tpr = np.r_[0.0, true_positive[last_indices] / positive_count]
    fpr = np.r_[0.0, false_positive[last_indices] / negative_count]
    integrate = np.trapezoid if hasattr(np, 'trapezoid') else np.trapz
    return float(integrate(tpr, fpr))

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

def classification_metrics(labels, predictions, losses, confidences, sad_probabilities):
    confusion = [[0, 0], [0, 0]]
    for (label, prediction) in zip(labels, predictions):
        confusion[label][prediction] += 1
    per_class = {}
    recalls = []
    f1_scores = []
    for (class_index, class_name) in enumerate(MODEL_LABELS):
        true_positive = confusion[class_index][class_index]
        false_positive = sum((row[class_index] for row in confusion)) - true_positive
        false_negative = sum(confusion[class_index]) - true_positive
        precision = safe_divide(true_positive, true_positive + false_positive)
        recall = safe_divide(true_positive, true_positive + false_negative)
        f1 = safe_divide(2.0 * precision * recall, precision + recall)
        recalls.append(recall)
        f1_scores.append(f1)
        per_class[class_name] = {'support': sum(confusion[class_index]), 'precision_pct': round(100.0 * precision, 4), 'recall_pct': round(100.0 * recall, 4), 'f1_pct': round(100.0 * f1, 4)}
    correct = sum((label == prediction for (label, prediction) in zip(labels, predictions)))
    sad_targets = np.asarray(labels, dtype=np.int64)
    sad_probabilities = np.asarray(sad_probabilities, dtype=np.float64)
    return {'image_count': len(labels), 'accuracy_pct': round(100.0 * safe_divide(correct, len(labels)), 4), 'balanced_accuracy_pct': round(100.0 * sum(recalls) / len(recalls), 4), 'macro_f1_pct': round(100.0 * sum(f1_scores) / len(f1_scores), 4), 'cross_entropy_loss': round(safe_divide(sum(losses), len(labels)), 6), 'expected_calibration_error': round(expected_calibration_error(labels, predictions, confidences), 6), 'mean_prediction_confidence_pct': round(100.0 * safe_divide(sum(confidences), len(confidences)), 4), 'sad_vs_happy_roc_auc': round(binary_roc_auc(sad_targets, sad_probabilities), 6), 'sad_vs_happy_brier_score': round(float(np.mean((sad_probabilities - sad_targets) ** 2)), 6), 'per_class': per_class, 'confusion_matrix_rows_true_cols_pred': confusion, 'confusion_matrix_labels': MODEL_LABELS}

def save_result(result):
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = RESULTS_PATH.with_suffix('.json.tmp')
    temporary_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    os.replace(temporary_path, RESULTS_PATH)

def main():
    if not MODEL_CACHE_DIR.is_dir():
        raise FileNotFoundError(f'未找到模型缓存：{MODEL_CACHE_DIR}')
    if not TEST_PATH.is_file():
        raise FileNotFoundError(f'未找到 FER2013 privateTest 文件：{TEST_PATH}')
    with _test_record.timer.loading():
        processor = LegacyPilProcessor.from_pretrained(str(MODEL_CACHE_DIR), local_files_only=True)
    with _test_record.timer.loading():
        model = AutoModelForImageClassification.from_pretrained(str(MODEL_CACHE_DIR), local_files_only=True)
    model_labels = [model.config.id2label[str(index)] if str(index) in model.config.id2label else model.config.id2label[index] for index in range(model.config.num_labels)]
    if model_labels != MODEL_LABELS:
        raise ValueError(f'模型标签应为 {MODEL_LABELS}，实际为 {model_labels}')
    full_test = HFDataset.from_parquet(str(TEST_PATH), cache_dir=str(_test_record.output / '.dataset_cache'))
    if set(full_test.column_names) != {'label', 'image'}:
        raise ValueError(f'FER2013 列应为 label、image，实际为 {full_test.column_names}')
    selected_indices = [index for (index, label) in enumerate(full_test['label']) if int(label) in FER_TO_MODEL_LABEL]
    binary_test = full_test.select(selected_indices)
    source_counts = {FER2013_LABELS[index]: sum((int(label) == index for label in full_test['label'])) for index in range(len(FER2013_LABELS))}
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device_name = torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'
    with _test_record.timer.loading():
        model = model.to(device).eval()
    wrapped_dataset = FER2013HappySadDataset(binary_test, processor)
    dataloader = _test_record.loader(DataLoader, wrapped_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=device.type == 'cuda')
    print(f'使用设备: {device_name}')
    print(f'模型: {MODEL_ID}')
    print('数据集: FER2013 privateTest 的 Happy/Sad 二分类子集')
    print(f'FER2013 privateTest 原始图片数量: {len(full_test)}')
    print(f'实际评测图片数量: {len(wrapped_dataset)}')
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
    (labels, predictions, losses, confidences, sad_probabilities) = ([], [], [], [], [])
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
            _test_record.prediction_ready(locals())
            batch_loss = F.cross_entropy(logits, batch_labels, reduction='none')
            labels.extend(batch_labels.cpu().tolist())
            predictions.extend(batch_predictions.cpu().tolist())
            losses.extend(batch_loss.cpu().tolist())
            confidences.extend(confidence.cpu().tolist())
            sad_probabilities.extend(probabilities[:, 1].cpu().tolist())
            if batch_index % 10 == 0 or batch_index == len(dataloader):
                print(f'已处理: {min(batch_index * BATCH_SIZE, len(wrapped_dataset))}/{len(wrapped_dataset)}')
            _test_record.capture(locals())
    if device.type == 'cuda':
        torch.cuda.synchronize()
    wall_seconds = time.perf_counter() - wall_start
    metrics = classification_metrics(labels, predictions, losses, confidences, sad_probabilities)
    result = {'recorded_at': datetime.now().astimezone().isoformat(timespec='seconds'), 'model_id': MODEL_ID, 'model_path': str(MODEL_CACHE_DIR), 'script_path': str(Path(__file__).resolve()), 'model': {'id': MODEL_ID, 'architecture': model.__class__.__name__, 'parameter_count_millions': round(sum((parameter.numel() for parameter in model.parameters())) / 1000000.0, 4), 'num_labels': model.config.num_labels, 'classifier_labels': model_labels}, 'dataset': {'name': 'FER2013', 'source': 'Aaryan333/fer2013_train_publicTest_privateTest（本地 parquet）', 'path': str(DATASET_DIR), 'source_split': 'privateTest', 'source_image_count': len(full_test), 'evaluated_subset': 'Happy (FER2013 label 3) and Sad (FER2013 label 4)', 'evaluated_image_count': len(wrapped_dataset), 'evaluated_coverage_pct': round(100.0 * len(wrapped_dataset) / len(full_test), 4), 'source_class_counts': source_counts, 'label_mapping': {'FER2013 Happy (3)': 'model Happy (0)', 'FER2013 Sad (4)': 'model Sad (1)'}, 'scope_note': '该模型仅有 Happy/Sad 两类输出；其余 Angry、Disgust、Fear、Surprise、Neutral 五类未映射，故不参与指标计算。'}, 'environment': {'device': device_name, 'precision': 'FP32', 'batch_size': BATCH_SIZE, 'warmup_steps': WARMUP_STEPS, 'num_workers': NUM_WORKERS}, 'preprocessing': {'input': 'FER2013 48×48 image bytes decoded to RGB', 'processor': processor.__class__.__name__, 'resize': '224×224, bilinear', 'rescale': '1/255', 'normalize': '(x - 0.5) / 0.5 per RGB channel', 'augmentations': 'none'}, 'official_reference': {'model_card_metric': None, 'comparability': '模型仓库没有 model card，无法确定其训练/验证数据；因此不提供可直接比较的作者官方指标。'}, 'accuracy': metrics, 'timing': {'timed_split': 'FER2013 privateTest Happy/Sad subset only', 'model_forward_total_seconds': round(forward_seconds, 6), 'model_latency_ms_per_image': round(forward_seconds / len(wrapped_dataset) * 1000.0, 6), 'model_throughput_images_per_second': round(len(wrapped_dataset) / forward_seconds, 6), 'end_to_end_total_seconds': round(wall_seconds, 6), 'end_to_end_latency_ms_per_image': round(wall_seconds / len(wrapped_dataset) * 1000.0, 6), 'end_to_end_throughput_images_per_second': round(len(wrapped_dataset) / wall_seconds, 6)}}
    if device.type == 'cuda':
        result['environment']['peak_gpu_memory_mib'] = round(torch.cuda.max_memory_allocated(device) / 1024 ** 2, 4)
    save_result(result)
    print('=' * 60)
    print(f"Accuracy: {metrics['accuracy_pct']:.4f}%")
    print(f"Balanced accuracy: {metrics['balanced_accuracy_pct']:.4f}%")
    print(f"Macro-F1: {metrics['macro_f1_pct']:.4f}%")
    print(f"Sad-vs-Happy ROC-AUC: {metrics['sad_vs_happy_roc_auc']:.6f}")
    print(f"Cross-entropy loss: {metrics['cross_entropy_loss']:.6f}")
    print(f"模型平均推理时间: {result['timing']['model_latency_ms_per_image']:.4f} ms/image")
    print(f"模型吞吐量: {result['timing']['model_throughput_images_per_second']:.2f} images/s")
    print(f'评测结果已更新: {RESULTS_PATH}')
    _test_record.finish(locals())
if __name__ == '__main__':
    main()
