"""在 HVPD 的 Chart Identification Subset 上评测 chart-recognizer。"""
if __name__ == '__main__':
    import sys as _records_sys
    from pathlib import Path as _RecordsPath
    _records_sys.path.insert(0, str(_RecordsPath(__file__).resolve().parent))
    _records_sys.path.insert(1, str(_RecordsPath(__file__).resolve().parents[1]))
    from common import start as _start_records
    _test_record = _start_records(120, __file__, globals())
import os
import sys
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['PYTHONUNBUFFERED'] = '1'
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(line_buffering=True, write_through=True)
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(line_buffering=True, write_through=True)
import io
import json
import time
import zipfile
from datetime import datetime
from pathlib import Path
import numpy as np
import timm
import torch
from PIL import Image
from safetensors.torch import load_file
from timm.data import create_transform, resolve_data_config
from torch.utils.data import DataLoader, Dataset
MODEL_ID = 'StephanAkkerman/chart-recognizer'
ROOT = __import__('model_paths').resolve_paths()['project']
MODEL_DIR = __import__('model_paths').resolve_paths()['weights']
DATASET_ID = 'WieszczUrsynowa/HVPD-HarmfulVisualPlotsDataset'
DATASET_ARCHIVE = __import__('model_paths').resolve_paths()['dataset'] / 'HVPD/CIS.zip'
RESULTS_PATH = _test_record.output / 'metrics.json'
if __name__ == '__main__':
    RESULTS_PATH = _test_record.output / 'metrics.json'
EXPECTED_IMAGES = 58501
CLASS_NAMES = ['chart', 'non-chart']
BATCH_SIZE = int(os.environ.get('MODEL_BATCH_SIZE', '256'))
NUM_WORKERS = int(os.environ.get('MODEL_WORKERS', '4'))
WARMUP_STEPS = int(os.environ.get('MODEL_WARMUP', '5'))

class HVPDChartIdentificationDataset(Dataset):

    def __init__(self, archive_path, transform):
        self.archive_path = archive_path
        self.transform = transform
        self.archive = None
        with zipfile.ZipFile(archive_path) as archive:
            self.samples = []
            for name in archive.namelist():
                normalized = name.replace('/', '\\')
                basename = normalized.rsplit('\\', 1)[-1].lower()
                if basename.startswith('chart_'):
                    label = 0
                elif basename.startswith('just_image_'):
                    label = 1
                else:
                    continue
                self.samples.append((name, label))
        chart_count = sum((label == 0 for (_, label) in self.samples))
        non_chart_count = sum((label == 1 for (_, label) in self.samples))
        if len(self.samples) != EXPECTED_IMAGES:
            raise ValueError(f'HVPD CIS 应有 {EXPECTED_IMAGES} 张图，实际识别到 {len(self.samples)} 张')
        if (chart_count, non_chart_count) != (20738, 37763):
            raise ValueError(f'HVPD CIS 类别数量异常：chart={chart_count}, non-chart={non_chart_count}')

    def __len__(self):
        return len(self.samples)

    def __getstate__(self):
        with _test_record.timer.loading():
            state = self.__dict__.copy()
        state['archive'] = None
        return state

    def __getitem__(self, index):
        if self.archive is None:
            self.archive = zipfile.ZipFile(self.archive_path)
        (name, label) = self.samples[index]
        try:
            with self.archive.open(name) as file:
                image = Image.open(io.BytesIO(file.read())).convert('RGB')
            return (self.transform(image), label, name, None)
        except Exception as error:
            return (None, label, name, f'{type(error).__name__}: {error}')

def collate_batch(batch):
    valid = [sample for sample in batch if sample[0] is not None]
    invalid = [{'path': name, 'error': error} for (image, _, name, error) in batch if image is None]
    if not valid:
        return (None, None, invalid)
    images = torch.stack([sample[0] for sample in valid])
    labels = torch.tensor([sample[1] for sample in valid], dtype=torch.long)
    return (images, labels, invalid)

def binary_roc_auc(labels, scores):
    labels = np.asarray(labels, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    order = np.argsort(scores, kind='mergesort')
    sorted_scores = scores[order]
    ranks = np.empty(len(scores), dtype=np.float64)
    start = 0
    while start < len(scores):
        end = start + 1
        while end < len(scores) and sorted_scores[end] == sorted_scores[start]:
            end += 1
        ranks[order[start:end]] = (start + 1 + end) / 2.0
        start = end
    positive_count = int((labels == 1).sum())
    negative_count = int((labels == 0).sum())
    rank_sum = ranks[labels == 1].sum()
    return (rank_sum - positive_count * (positive_count + 1) / 2) / (positive_count * negative_count)

def calculate_metrics(confusion, labels, chart_scores):
    true_positive = confusion.diag().to(torch.float64)
    support = confusion.sum(dim=1).to(torch.float64)
    predicted = confusion.sum(dim=0).to(torch.float64)
    precision = torch.where(predicted > 0, true_positive / predicted, torch.zeros_like(predicted))
    recall = torch.where(support > 0, true_positive / support, torch.zeros_like(support))
    f1 = torch.where(precision + recall > 0, 2 * precision * recall / (precision + recall), torch.zeros_like(precision))
    image_count = int(support.sum().item())
    return {'image_count': image_count, 'accuracy_pct': round(100 * true_positive.sum().item() / image_count, 4), 'balanced_accuracy_pct': round(100 * recall.mean().item(), 4), 'macro_precision_pct': round(100 * precision.mean().item(), 4), 'macro_recall_pct': round(100 * recall.mean().item(), 4), 'macro_f1_pct': round(100 * f1.mean().item(), 4), 'chart_roc_auc': round(binary_roc_auc(np.asarray(labels) == 0, chart_scores), 6), 'per_class': {name: {'support': int(support[index].item()), 'precision_pct': round(100 * precision[index].item(), 4), 'recall_pct': round(100 * recall[index].item(), 4), 'f1_pct': round(100 * f1[index].item(), 4)} for (index, name) in enumerate(CLASS_NAMES)}, 'confusion_matrix_rows_true_cols_pred': confusion.tolist()}

def save_result(result):
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = RESULTS_PATH.with_suffix('.json.tmp')
    temporary_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    os.replace(temporary_path, RESULTS_PATH)

def main():
    if not MODEL_DIR.is_dir():
        raise FileNotFoundError(f'未找到模型：{MODEL_DIR}')
    if not DATASET_ARCHIVE.is_file():
        raise FileNotFoundError(f'未找到 HVPD CIS：{DATASET_ARCHIVE}')
    with _test_record.timer.loading():
        config = json.loads((MODEL_DIR / 'config.json').read_text(encoding='utf-8'))
    if config.get('architecture') != 'efficientnet_b0':
        raise ValueError(f"不支持的模型结构：{config.get('architecture')}")
    if config.get('label_names') != CLASS_NAMES:
        raise ValueError(f"模型标签应为 {CLASS_NAMES}，实际为 {config.get('label_names')}")
    with _test_record.timer.loading():
        model = timm.create_model('efficientnet_b0', pretrained=False, num_classes=2)
    with _test_record.timer.loading():
        model.load_state_dict(load_file(MODEL_DIR / 'model.safetensors'), strict=True)
    data_config = resolve_data_config({}, model=model)
    transform = create_transform(**data_config, is_training=False)
    dataset = HVPDChartIdentificationDataset(DATASET_ARCHIVE, transform)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device_name = torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'
    with _test_record.timer.loading():
        model = model.to(device).eval()
    dataloader = _test_record.loader(DataLoader, dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=device.type == 'cuda', persistent_workers=NUM_WORKERS > 0, collate_fn=collate_batch)
    print(f'使用设备: {device_name}')
    print(f'模型: {MODEL_ID}')
    print(f'数据集: HVPD Chart Identification Subset（外部评测）')
    print(f'测试图片数量: {len(dataset)}')
    print(f'batch size: {BATCH_SIZE}')
    (warmup_images, _, invalid) = next(iter(dataloader))
    if warmup_images is None:
        raise RuntimeError(f'首个 batch 全部无法解码：{invalid[:3]}')
    warmup_images = warmup_images.to(device, non_blocking=True)
    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = True
        torch.cuda.reset_peak_memory_stats(device)
    with torch.inference_mode():
        with _test_record.timer.warmup():
            for _ in range(WARMUP_STEPS):
                _test_record.timer.invoke(model, warmup_images)
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
    del warmup_images
    confusion = torch.zeros((2, 2), dtype=torch.long)
    all_labels = []
    chart_scores = []
    invalid_images = []
    forward_seconds = 0.0
    wall_start = time.perf_counter()
    with torch.inference_mode():
        for (batch_index, (images, labels, invalid)) in enumerate(dataloader, 1):
            _test_record.before_batch(locals())
            invalid_images.extend(invalid)
            if images is None:
                continue
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            if device.type == 'cuda':
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)
                start_event.record()
                logits = _test_record.timer.invoke(model, images)
                end_event.record()
                torch.cuda.synchronize(device)
                forward_seconds += start_event.elapsed_time(end_event) / 1000.0
            else:
                forward_start = time.perf_counter()
                logits = _test_record.timer.invoke(model, images)
                forward_seconds += time.perf_counter() - forward_start
            predictions = logits.argmax(dim=1)
            _test_record.prediction_ready(locals())
            confusion += torch.bincount((labels * 2 + predictions).cpu(), minlength=4).reshape(2, 2)
            all_labels.extend(labels.cpu().tolist())
            chart_scores.extend(torch.softmax(logits, dim=1)[:, 0].cpu().tolist())
            if batch_index % 25 == 0 or batch_index == len(dataloader):
                processed = min(batch_index * BATCH_SIZE, len(dataset))
                print(f'已处理: {processed}/{len(dataset)}')
            _test_record.capture(locals())
    wall_seconds = time.perf_counter() - wall_start
    metrics = calculate_metrics(confusion, all_labels, chart_scores)
    valid_count = metrics['image_count']
    result = {'recorded_at': datetime.now().astimezone().isoformat(timespec='seconds'), 'model_id': MODEL_ID, 'model_path': str(MODEL_DIR), 'script_path': str(Path(__file__).resolve()), 'model': {'architecture': 'EfficientNet-B0', 'parameter_count_millions': round(sum((parameter.numel() for parameter in model.parameters())) / 1000000.0, 4), 'class_names': CLASS_NAMES}, 'dataset': {'name': 'HVPD Chart Identification Subset (CIS)', 'source': DATASET_ID, 'archive': str(DATASET_ARCHIVE), 'evaluation_scope': 'entire CIS archive; the dataset does not publish a separate test split', 'expected_images': EXPECTED_IMAGES, 'valid_images': valid_count, 'invalid_images': len(invalid_images), 'class_counts': {'chart': 20738, 'non-chart': 37763}, 'dvcs_excluded': "DVCS labels deceptive chart properties, which do not match the model's chart/non-chart output."}, 'protocol': {'type': 'external cross-dataset evaluation', 'official_comparability': False, 'notes': ['HVPD CIS and the model share the same binary chart/non-chart task.', 'All ZIP entries share a misleading root folder; labels are read from the chart_ and just_image_ filename prefixes.', "No overlap audit between HVPD web-scraped sources and the model's training datasets is available.", "Results must not be compared directly with the model card's own test-set accuracy/F1."]}, 'preprocessing': {'input_size': list(data_config['input_size']), 'interpolation': data_config['interpolation'], 'crop_pct': data_config['crop_pct'], 'mean': list(data_config['mean']), 'std': list(data_config['std']), 'batch_size': BATCH_SIZE}, 'metrics': metrics, 'timing': {'warmup_steps': WARMUP_STEPS, 'forward_seconds': round(forward_seconds, 6), 'forward_ms_per_image': round(1000 * forward_seconds / valid_count, 6), 'forward_images_per_second': round(valid_count / forward_seconds, 4), 'wall_seconds_including_zip_decode': round(wall_seconds, 6), 'wall_ms_per_image': round(1000 * wall_seconds / valid_count, 6), 'peak_cuda_memory_mib': round(torch.cuda.max_memory_allocated(device) / 1024 ** 2, 2) if device.type == 'cuda' else None}, 'environment': {'device': str(device), 'device_name': device_name, 'torch_version': torch.__version__, 'timm_version': timm.__version__}, 'official_model_card_reference': {'dataset': "the author's own Test Set, not HVPD", 'accuracy': 0.9782, 'f1': 0.9685}, 'invalid_image_examples': invalid_images[:20]}
    save_result(result)
    print('\n评测完成')
    print(f"Accuracy: {metrics['accuracy_pct']:.4f}%")
    print(f"Balanced accuracy: {metrics['balanced_accuracy_pct']:.4f}%")
    print(f"Macro-F1: {metrics['macro_f1_pct']:.4f}%")
    print(f"Chart ROC-AUC: {metrics['chart_roc_auc']:.6f}")
    print(f"纯前向推理: {result['timing']['forward_ms_per_image']:.6f} ms/图")
    print(f'损坏/无法解码图片: {len(invalid_images)}')
    print(f'结果已保存: {RESULTS_PATH}')
    _test_record.finish(locals())
if __name__ == '__main__':
    main()
