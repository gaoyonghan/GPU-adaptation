"""在 Xenova/quickdraw-small 官方 test split 上评测 Xenova/quickdraw-mobilevit-small。"""
if __name__ == '__main__':
    import sys as _records_sys
    from pathlib import Path as _RecordsPath
    _records_sys.path.insert(0, str(_RecordsPath(__file__).resolve().parent))
    _records_sys.path.insert(1, str(_RecordsPath(__file__).resolve().parents[1]))
    from common import start as _start_records
    _test_record = _start_records(113, __file__, globals())
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
import numpy as np
import torch
import torch.nn.functional as F
from datasets import Dataset as HFDataset
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForImageClassification
try:
    from transformers.models.mobilevit.image_processing_pil_mobilevit import MobileViTImageProcessorPil as LegacyPilProcessor
except ImportError:
    from transformers import MobileViTImageProcessor as LegacyPilProcessor
MODEL_ID = 'Xenova/quickdraw-mobilevit-small'
ROOT = __import__('model_paths').resolve_paths()['project']
MODEL_CACHE_DIR = __import__('model_paths').resolve_paths()['weights']
DATASET_ID = 'Xenova/quickdraw-small'
DATASET_DIR = __import__('model_paths').resolve_paths()['dataset'] / 'quickdraw-small'
RESULTS_PATH = _test_record.output / 'metrics.json'
if __name__ == '__main__':
    RESULTS_PATH = _test_record.output / 'metrics.json'
NUM_CLASSES = 345
EXPECTED_TEST_IMAGES = 250000
BATCH_SIZE = int(os.environ.get('MODEL_BATCH_SIZE', '512'))
NUM_WORKERS = int(os.environ.get('MODEL_WORKERS', '4'))
WARMUP_STEPS = int(os.environ.get('MODEL_WARMUP', '3'))

def json_metadata_default(value):
    """Serialize library metadata without changing inference or preprocessing."""
    if isinstance(value, (set, frozenset)):
        return sorted(value)
    if is_dataclass(value) and (not isinstance(value, type)):
        return {key: item for (key, item) in asdict(value).items() if item is not None}
    raise TypeError(f'Object of type {type(value).__name__} is not JSON serializable')

class QuickDrawTestDataset(Dataset):

    def __init__(self, dataset, processor):
        self.dataset = dataset
        self.processor = processor

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        example = self.dataset[index]
        image = np.asarray(example['image'].convert('L'), dtype=np.uint8)[..., None]
        pixel_values = self.processor(images=image, input_data_format='channels_last', return_tensors='pt')['pixel_values'].squeeze(0)
        return (pixel_values, int(example['label']))

def load_quickdraw_test():
    shards = sorted(DATASET_DIR.glob('data/test-*.parquet'))
    if not shards:
        raise FileNotFoundError(f'未找到 QuickDraw test parquet：{DATASET_DIR}')
    dataset = HFDataset.from_parquet([str(path) for path in shards], cache_dir=str(_test_record.output / '.dataset_cache'))
    if len(dataset) != EXPECTED_TEST_IMAGES:
        raise ValueError(f'QuickDraw-small test 应有 {EXPECTED_TEST_IMAGES} 张图，实际为 {len(dataset)}')
    if set(dataset.column_names) != {'image', 'label'}:
        raise ValueError(f'QuickDraw test 列应为 image/label，实际为 {dataset.column_names}')
    label_feature = dataset.features['label']
    if not hasattr(label_feature, 'names') or len(label_feature.names) != NUM_CLASSES:
        raise ValueError(f'QuickDraw test 应有 {NUM_CLASSES} 类标签')
    return (dataset, list(label_feature.names), shards)

def calculate_metrics(confusion, top1_correct, top5_correct, loss_sum, class_names):
    true_positive = confusion.diag().to(torch.float64)
    support = confusion.sum(dim=1).to(torch.float64)
    predicted = confusion.sum(dim=0).to(torch.float64)
    precision = torch.where(predicted > 0, true_positive / predicted, torch.zeros_like(predicted))
    recall = torch.where(support > 0, true_positive / support, torch.zeros_like(support))
    f1 = torch.where(precision + recall > 0, 2 * precision * recall / (precision + recall), torch.zeros_like(precision))
    image_count = int(support.sum().item())
    per_class = {class_name: {'support': int(support[index].item()), 'precision_pct': round(100 * precision[index].item(), 4), 'recall_pct': round(100 * recall[index].item(), 4), 'f1_pct': round(100 * f1[index].item(), 4)} for (index, class_name) in enumerate(class_names)}
    off_diagonal = confusion.clone()
    off_diagonal.fill_diagonal_(0)
    confusion_pairs = []
    for flat_index in torch.argsort(off_diagonal.flatten(), descending=True).tolist():
        count = int(off_diagonal.flatten()[flat_index].item())
        if count == 0 or len(confusion_pairs) == 20:
            break
        true_index = flat_index // NUM_CLASSES
        predicted_index = flat_index % NUM_CLASSES
        confusion_pairs.append({'true': class_names[true_index], 'predicted': class_names[predicted_index], 'count': count})
    return {'image_count': image_count, 'top1_accuracy_pct': round(100 * top1_correct / image_count, 4), 'top5_accuracy_pct': round(100 * top5_correct / image_count, 4), 'macro_precision_pct': round(100 * precision.mean().item(), 4), 'macro_recall_pct': round(100 * recall.mean().item(), 4), 'balanced_accuracy_pct': round(100 * recall.mean().item(), 4), 'macro_f1_pct': round(100 * f1.mean().item(), 4), 'cross_entropy_loss': round(loss_sum / image_count, 6), 'per_class': per_class, 'top_confusion_pairs': confusion_pairs, 'confusion_matrix_rows_true_cols_pred': confusion.tolist()}

def save_result(result):
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = RESULTS_PATH.with_suffix('.json.tmp')
    temporary_path.write_text(json.dumps(result, default=json_metadata_default, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    os.replace(temporary_path, RESULTS_PATH)

def main():
    if not MODEL_CACHE_DIR.is_dir():
        raise FileNotFoundError(f'未找到模型缓存：{MODEL_CACHE_DIR}')
    (dataset, dataset_class_names, shards) = load_quickdraw_test()
    with _test_record.timer.loading():
        processor = LegacyPilProcessor.from_pretrained(str(MODEL_CACHE_DIR), local_files_only=True)
    with _test_record.timer.loading():
        model = AutoModelForImageClassification.from_pretrained(str(MODEL_CACHE_DIR), local_files_only=True)
    id2label = {int(index): name for (index, name) in model.config.id2label.items()}
    model_class_names = [id2label[index] for index in range(NUM_CLASSES)]
    if model.config.num_labels != NUM_CLASSES or dataset_class_names != model_class_names:
        raise ValueError('QuickDraw 数据集标签顺序与模型输出标签不一致，拒绝评测。')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device_name = torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'
    with _test_record.timer.loading():
        model = model.to(device)
    with _test_record.timer.loading():
        model.eval()
    wrapped_dataset = QuickDrawTestDataset(dataset, processor)
    dataloader = _test_record.loader(DataLoader, wrapped_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=device.type == 'cuda', persistent_workers=NUM_WORKERS > 0)
    print(f'使用设备: {device_name}')
    print(f'模型: {MODEL_ID}')
    print(f'数据集: {DATASET_ID} 官方 test split')
    print(f'测试图片数量: {len(wrapped_dataset)}')
    print(f'类别数量: {NUM_CLASSES}')
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
        torch.cuda.synchronize(device)
    del warmup_pixel_values
    confusion = torch.zeros((NUM_CLASSES, NUM_CLASSES), dtype=torch.long)
    top1_correct = 0
    top5_correct = 0
    loss_sum = 0.0
    forward_seconds = 0.0
    wall_start = time.perf_counter()
    with torch.inference_mode():
        for (batch_index, (pixel_values, labels)) in enumerate(dataloader, 1):
            _test_record.before_batch(locals())
            pixel_values = pixel_values.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            if device.type == 'cuda':
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)
                start_event.record()
                logits = _test_record.timer.invoke(model, pixel_values=pixel_values).logits
                end_event.record()
                torch.cuda.synchronize(device)
                forward_seconds += start_event.elapsed_time(end_event) / 1000.0
            else:
                forward_start = time.perf_counter()
                logits = _test_record.timer.invoke(model, pixel_values=pixel_values).logits
                forward_seconds += time.perf_counter() - forward_start
            top5 = logits.topk(k=5, dim=1).indices
            predictions = top5[:, 0]
            _test_record.prediction_ready(locals())
            top1_correct += predictions.eq(labels).sum().item()
            top5_correct += top5.eq(labels.unsqueeze(1)).any(dim=1).sum().item()
            loss_sum += F.cross_entropy(logits, labels, reduction='sum').item()
            confusion += torch.bincount((labels * NUM_CLASSES + predictions).cpu(), minlength=NUM_CLASSES ** 2).reshape(NUM_CLASSES, NUM_CLASSES)
            if batch_index % 50 == 0 or batch_index == len(dataloader):
                print(f'已处理: {min(batch_index * BATCH_SIZE, len(wrapped_dataset))}/{len(wrapped_dataset)}')
            _test_record.capture(locals())
    wall_seconds = time.perf_counter() - wall_start
    metrics = calculate_metrics(confusion, top1_correct, top5_correct, loss_sum, model_class_names)
    result = {'recorded_at': datetime.now().astimezone().isoformat(timespec='seconds'), 'model_id': MODEL_ID, 'model_path': str(MODEL_CACHE_DIR), 'script_path': str(Path(__file__).resolve()), 'model': {'id': MODEL_ID, 'architecture': model.__class__.__name__, 'parameter_count_millions': round(sum((parameter.numel() for parameter in model.parameters())) / 1000000.0, 4), 'num_labels': NUM_CLASSES, 'id2label': {str(index): id2label[index] for index in range(NUM_CLASSES)}, 'execution_backend': "PyTorch CUDA (the repository's pytorch_model.bin)"}, 'dataset': {'name': 'quickdraw-small', 'source': DATASET_ID, 'path': str(DATASET_DIR), 'split': 'test', 'parquet_shards': [str(path) for path in shards], 'image_count': len(wrapped_dataset), 'num_classes': NUM_CLASSES, 'image_mode': 'grayscale (L)', 'split_note': 'Uses the published 250,000-image test parquet; no local random split is created.'}, 'preprocessing': {'processor': processor.__class__.__name__, 'size': processor.size, 'crop_size': processor.crop_size, 'do_convert_rgb': processor.do_convert_rgb, 'rescale_factor': processor.rescale_factor, 'batch_size': BATCH_SIZE}, 'environment': {'device': str(device), 'device_name': device_name, 'torch_version': torch.__version__, 'cuda_available': torch.cuda.is_available()}, 'official_model_card_reference': {'metrics': None, 'note': 'The Xenova ONNX-conversion model card does not publish a QuickDraw-small test metric.'}, 'metrics': metrics, 'timing': {'warmup_steps': WARMUP_STEPS, 'forward_seconds': round(forward_seconds, 6), 'forward_ms_per_image': round(1000 * forward_seconds / len(wrapped_dataset), 6), 'forward_images_per_second': round(len(wrapped_dataset) / forward_seconds, 4), 'wall_seconds_including_dataloader': round(wall_seconds, 6), 'wall_ms_per_image': round(1000 * wall_seconds / len(wrapped_dataset), 6), 'peak_cuda_memory_mib': round(torch.cuda.max_memory_allocated(device) / 1024 ** 2, 2) if device.type == 'cuda' else None}}
    save_result(result)
    print('\n评测完成')
    print(f"Top-1: {metrics['top1_accuracy_pct']:.4f}%")
    print(f"Top-5: {metrics['top5_accuracy_pct']:.4f}%")
    print(f"Macro-F1: {metrics['macro_f1_pct']:.4f}%")
    print(f"Cross-entropy loss: {metrics['cross_entropy_loss']:.6f}")
    print(f"纯前向推理: {result['timing']['forward_ms_per_image']:.6f} ms/图")
    print(f'结果已保存: {RESULTS_PATH}')
    _test_record.finish(locals())
if __name__ == '__main__':
    main()
