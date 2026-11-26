"""在 FairFace validation split 上评测 syntheticbot/clip-face-attribute-classifier。"""
if __name__ == '__main__':
    import sys as _records_sys
    from pathlib import Path as _RecordsPath
    _records_sys.path.insert(0, str(_RecordsPath(__file__).resolve().parent))
    _records_sys.path.insert(1, str(_RecordsPath(__file__).resolve().parents[1]))
    from common import start as _start_records
    _test_record = _start_records(149, __file__, globals())
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
import torch.nn as nn
import torch.nn.functional as F
from datasets import Dataset as HFDataset
from safetensors.torch import load_file
from torch.utils.data import DataLoader, Dataset
from transformers import CLIPVisionConfig, CLIPVisionModel
try:
    from transformers.models.clip.image_processing_pil_clip import CLIPImageProcessorPil as CLIPImageProcessor
except ImportError:
    from transformers import CLIPImageProcessor
MODEL_ID = 'syntheticbot/clip-face-attribute-classifier'
ROOT = __import__('model_paths').resolve_paths()['project']
MODEL_CACHE_DIR = __import__('model_paths').resolve_paths()['weights']
DATASET_DIR = __import__('model_paths').resolve_paths()['dataset'] / 'FairFace'
RESULTS_PATH = _test_record.output / 'metrics.json'
if __name__ == '__main__':
    RESULTS_PATH = _test_record.output / 'metrics.json'
BATCH_SIZE = int(os.environ.get('MODEL_BATCH_SIZE', '64'))
NUM_WORKERS = int(os.environ.get('MODEL_WORKERS', '0'))
WARMUP_STEPS = int(os.environ.get('MODEL_WARMUP', '3'))
ECE_BINS = 15
DATASET_LABELS = {'age': ['0-2', '3-9', '10-19', '20-29', '30-39', '40-49', '50-59', '60-69', 'more than 70'], 'gender': ['Male', 'Female'], 'race': ['East Asian', 'Indian', 'Black', 'White', 'Middle Eastern', 'Latino_Hispanic', 'Southeast Asian']}
MODEL_LABELS = {task: sorted(names) for (task, names) in DATASET_LABELS.items()}
DATASET_TO_MODEL = {task: [MODEL_LABELS[task].index(name) for name in names] for (task, names) in DATASET_LABELS.items()}
TASK_ORDER = ['age', 'gender', 'race']

class MultiTaskClipVisionModel(nn.Module):
    """与模型仓库 train_clip.py 相同的视觉编码器和三个分类头。"""

    def __init__(self, vision_config):
        super().__init__()
        self.vision_model = CLIPVisionModel(vision_config)
        hidden_size = vision_config.hidden_size
        self.age_head = nn.Linear(hidden_size, len(MODEL_LABELS['age']))
        self.gender_head = nn.Linear(hidden_size, len(MODEL_LABELS['gender']))
        self.race_head = nn.Linear(hidden_size, len(MODEL_LABELS['race']))

    def forward(self, pixel_values):
        pooled_output = self.vision_model(pixel_values=pixel_values).pooler_output
        return {'age': self.age_head(pooled_output), 'gender': self.gender_head(pooled_output), 'race': self.race_head(pooled_output)}

class FairFaceDataset(Dataset):

    def __init__(self, dataset, processor):
        self.dataset = dataset
        self.processor = processor

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        example = self.dataset[index]
        raw_labels = torch.tensor([int(example[task]) for task in TASK_ORDER], dtype=torch.long)
        model_labels = torch.tensor([DATASET_TO_MODEL[task][int(example[task])] for task in TASK_ORDER], dtype=torch.long)
        pixel_values = self.processor(images=example['image'].convert('RGB'), return_tensors='pt')['pixel_values'].squeeze(0)
        return (pixel_values, model_labels, raw_labels)

def safe_divide(numerator, denominator):
    return float(numerator) / float(denominator) if denominator else 0.0

def classification_metrics(labels, predictions, losses, confidences, class_names):
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
    correct = sum((label == prediction for (label, prediction) in zip(labels, predictions)))
    ece = 0.0
    for bin_index in range(ECE_BINS):
        lower = bin_index / ECE_BINS
        upper = (bin_index + 1) / ECE_BINS
        indices = [index for (index, confidence) in enumerate(confidences) if lower <= confidence < upper or (bin_index == ECE_BINS - 1 and confidence == upper)]
        if indices:
            accuracy = sum((labels[index] == predictions[index] for index in indices)) / len(indices)
            average_confidence = sum((confidences[index] for index in indices)) / len(indices)
            ece += len(indices) / len(labels) * abs(accuracy - average_confidence)
    return {'image_count': len(labels), 'accuracy_pct': round(100 * safe_divide(correct, len(labels)), 4), 'macro_f1_pct': round(100 * sum(f1_scores) / class_count, 4), 'balanced_accuracy_pct': round(100 * sum(recalls) / class_count, 4), 'cross_entropy_loss': round(safe_divide(sum(losses), len(labels)), 6), 'expected_calibration_error': round(ece, 6), 'per_class': per_class, 'confusion_matrix_rows_true_cols_pred': confusion}

def subgroup_accuracy(group_values, labels, predictions, group_names):
    grouped = defaultdict(lambda : [0, 0])
    for (group, label, prediction) in zip(group_values, labels, predictions):
        grouped[int(group)][0] += 1
        grouped[int(group)][1] += int(label == prediction)
    return {group_names[index]: {'image_count': values[0], 'accuracy_pct': round(100 * safe_divide(values[1], values[0]), 4)} for (index, values) in sorted(grouped.items())}

def find_validation_shards():
    shards = sorted(DATASET_DIR.glob('0.25/validation-*.parquet'))
    if not shards:
        raise FileNotFoundError(f"未找到 FairFace validation parquet：{DATASET_DIR / '0.25'}")
    return shards

def save_result(result):
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = RESULTS_PATH.with_suffix('.json.tmp')
    temporary_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    os.replace(temporary_path, RESULTS_PATH)

def main():
    if not MODEL_CACHE_DIR.is_dir():
        raise FileNotFoundError(f'未找到模型缓存：{MODEL_CACHE_DIR}')
    shards = find_validation_shards()
    with _test_record.timer.loading():
        processor = CLIPImageProcessor.from_pretrained(str(MODEL_CACHE_DIR), local_files_only=True)
    config_path = MODEL_CACHE_DIR / 'config.json'
    with _test_record.timer.loading():
        config = json.loads(config_path.read_text(encoding='utf-8'))
    vision_config = CLIPVisionConfig(**config['vision_config'])
    with _test_record.timer.loading():
        model = MultiTaskClipVisionModel(vision_config)
    with _test_record.timer.loading():
        state_dict = load_file(str(MODEL_CACHE_DIR / 'model.safetensors'))
    expected_keys = set(model.state_dict())
    compatible_state = {}
    for (key, value) in state_dict.items():
        candidate = key.replace('vision_model.vision_model.', 'vision_model.', 1)
        if key not in expected_keys and candidate in expected_keys:
            key = candidate
        if key in compatible_state:
            raise ValueError(f'CLIP 权重键名映射冲突：{key}')
        compatible_state[key] = value
    with _test_record.timer.loading():
        model.load_state_dict(compatible_state, strict=True)
    del state_dict, compatible_state
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device_name = torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'
    with _test_record.timer.loading():
        model = model.to(device)
    with _test_record.timer.loading():
        model.eval()
    print(f'使用设备: {device_name}')
    print(f'模型: {MODEL_ID}')
    print('数据集: FairFace validation（官方公开 validation split）')
    dataset = HFDataset.from_parquet([str(path) for path in shards], cache_dir=str(_test_record.output / '.dataset_cache'))
    expected_columns = {'image', 'age', 'gender', 'race'}
    if not expected_columns.issubset(set(dataset.column_names)):
        raise ValueError(f'FairFace 列不完整，实际列：{dataset.column_names}')
    for task in TASK_ORDER:
        actual_names = list(dataset.features[task].names)
        if actual_names != DATASET_LABELS[task]:
            raise ValueError(f'FairFace {task} 标签顺序变化：{actual_names}')
    wrapped_dataset = FairFaceDataset(dataset, processor)
    dataloader = _test_record.loader(DataLoader, wrapped_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=device.type == 'cuda')
    print(f'验证图片数量: {len(wrapped_dataset)}')
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
    task_labels = {task: [] for task in TASK_ORDER}
    task_predictions = {task: [] for task in TASK_ORDER}
    task_losses = {task: [] for task in TASK_ORDER}
    task_confidences = {task: [] for task in TASK_ORDER}
    raw_values = {task: [] for task in TASK_ORDER}
    forward_seconds = 0.0
    wall_start = time.perf_counter()
    with torch.inference_mode():
        for (batch_index, batch) in enumerate(dataloader, start=1):
            _test_record.before_batch(locals())
            (pixel_values, batch_model_labels, batch_raw_labels) = batch
            pixel_values = pixel_values.to(device, non_blocking=True)
            batch_model_labels = batch_model_labels.to(device, non_blocking=True)
            if device.type == 'cuda':
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)
                start_event.record()
                logits = _test_record.timer.invoke(model, pixel_values=pixel_values)
                end_event.record()
                torch.cuda.synchronize()
                forward_seconds += start_event.elapsed_time(end_event) / 1000.0
            else:
                forward_start = time.perf_counter()
                logits = _test_record.timer.invoke(model, pixel_values=pixel_values)
                forward_seconds += time.perf_counter() - forward_start
            for (task_index, task) in enumerate(TASK_ORDER):
                task_logits = logits[task]
                labels = batch_model_labels[:, task_index]
                probabilities = F.softmax(task_logits, dim=1)
                (confidence, predictions) = probabilities.max(dim=1)
                losses = F.cross_entropy(task_logits, labels, reduction='none')
                task_labels[task].extend(labels.cpu().tolist())
                task_predictions[task].extend(predictions.cpu().tolist())
                task_losses[task].extend(losses.cpu().tolist())
                task_confidences[task].extend(confidence.cpu().tolist())
                raw_values[task].extend(batch_raw_labels[:, task_index].tolist())
            _test_record.prediction_ready(locals())
            if batch_index % 25 == 0 or batch_index == len(dataloader):
                print(f'已处理: {min(batch_index * BATCH_SIZE, len(wrapped_dataset))}/{len(wrapped_dataset)}')
            _test_record.capture(locals())
    wall_seconds = time.perf_counter() - wall_start
    task_metrics = {}
    for task in TASK_ORDER:
        metrics = classification_metrics(task_labels[task], task_predictions[task], task_losses[task], task_confidences[task], MODEL_LABELS[task])
        metrics['subgroup_accuracy'] = {'age': subgroup_accuracy(raw_values['age'], task_labels[task], task_predictions[task], DATASET_LABELS['age']), 'gender': subgroup_accuracy(raw_values['gender'], task_labels[task], task_predictions[task], DATASET_LABELS['gender']), 'race': subgroup_accuracy(raw_values['race'], task_labels[task], task_predictions[task], DATASET_LABELS['race'])}
        task_metrics[task] = metrics
    result = {'recorded_at': datetime.now().astimezone().isoformat(timespec='seconds'), 'model_id': MODEL_ID, 'model_path': str(MODEL_CACHE_DIR), 'script_path': str(Path(__file__).resolve()), 'model': {'id': MODEL_ID, 'architecture': 'CLIPVisionModel + age/gender/race linear heads', 'parameter_count_millions': round(sum((parameter.numel() for parameter in model.parameters())) / 1000000.0, 4), 'task_num_labels': {task: len(MODEL_LABELS[task]) for task in TASK_ORDER}, 'model_head_label_order': MODEL_LABELS}, 'dataset': {'name': 'FairFace', 'source': 'HuggingFaceM4/FairFace', 'path': str(DATASET_DIR), 'split': 'validation', 'parquet_shards': [str(path) for path in shards], 'image_count': len(wrapped_dataset), 'dataset_label_order': DATASET_LABELS, 'dataset_to_model_label_index': DATASET_TO_MODEL, 'split_note': "This is FairFace's publicly available validation split (10,954 images); the downloaded public version does not provide a separate hidden test split."}, 'preprocessing': {'processor': 'CLIPImageProcessor loaded from local model cache', 'image_size': 224, 'batch_size': BATCH_SIZE}, 'environment': {'device': str(device), 'device_name': device_name, 'torch_version': torch.__version__, 'cuda_available': torch.cuda.is_available()}, 'official_model_card_reference': {'dataset': 'FairFace validation', 'gender_accuracy_pct': 96.38, 'race_accuracy_pct': 73.22, 'age_accuracy_pct': 59.17, 'note': 'Metrics reported by syntheticbot in the model card; evaluated here with the local validation parquet.'}, 'metrics': task_metrics, 'timing': {'warmup_steps': WARMUP_STEPS, 'forward_seconds': round(forward_seconds, 6), 'forward_ms_per_image': round(1000 * forward_seconds / len(wrapped_dataset), 6), 'forward_images_per_second': round(safe_divide(len(wrapped_dataset), forward_seconds), 4), 'wall_seconds_including_dataloader': round(wall_seconds, 6), 'wall_ms_per_image': round(1000 * wall_seconds / len(wrapped_dataset), 6), 'peak_cuda_memory_mib': round(torch.cuda.max_memory_allocated(device) / 1024 ** 2, 2) if device.type == 'cuda' else None}}
    save_result(result)
    print('\n评测完成')
    for task in TASK_ORDER:
        print(f"{task} accuracy: {task_metrics[task]['accuracy_pct']:.4f}% | macro-F1: {task_metrics[task]['macro_f1_pct']:.4f}% | loss: {task_metrics[task]['cross_entropy_loss']:.6f}")
    print(f'Forward time: {forward_seconds:.4f}s ({1000 * forward_seconds / len(wrapped_dataset):.4f} ms/image)')
    print(f'结果已保存: {RESULTS_PATH}')
    _test_record.finish(locals())
if __name__ == '__main__':
    main()
