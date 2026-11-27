"""在 TrashNet 官方 431 张 test 清单上评测 yangy50/garbage-classification。"""
if __name__ == '__main__':
    import sys as _records_sys
    from pathlib import Path as _RecordsPath
    _records_sys.path.insert(0, str(_RecordsPath(__file__).resolve().parent))
    _records_sys.path.insert(1, str(_RecordsPath(__file__).resolve().parents[1]))
    from common import start as _start_records
    _test_record = _start_records(105, __file__, globals())
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
import zipfile
from datetime import datetime
from pathlib import Path
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from transformers import AutoImageProcessor, AutoModelForImageClassification
MODEL_ID = 'yangy50/garbage-classification'
ROOT = __import__('model_paths').resolve_paths()['project']
MODEL_CACHE_DIR = __import__('model_paths').resolve_paths()['weights']
DATASET_DIR = __import__('model_paths').resolve_paths()['dataset'] / 'trashnet'
IMAGE_DIR = DATASET_DIR / 'dataset-resized'
IMAGE_ARCHIVE = DATASET_DIR / 'dataset-resized.zip'
TEST_MANIFEST = DATASET_DIR / 'one-indexed-files-notrash_test.txt'
RESULTS_PATH = _test_record.output / 'metrics.json'
if __name__ == '__main__':
    RESULTS_PATH = _test_record.output / 'metrics.json'
BATCH_SIZE = int(os.environ.get('MODEL_BATCH_SIZE', '64'))
NUM_WORKERS = int(os.environ.get('MODEL_WORKERS', '0'))
WARMUP_STEPS = int(os.environ.get('MODEL_WARMUP', '3'))
MANIFEST_LABEL_TO_NAME = {'1': 'glass', '2': 'paper', '3': 'cardboard', '4': 'plastic', '5': 'metal', '6': 'trash'}
CLASS_NAMES = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']
EXPECTED_TEST_COUNTS = {'cardboard': 70, 'glass': 82, 'metal': 68, 'paper': 108, 'plastic': 74, 'trash': 29}

def json_metadata_default(value):
    """Serialize library metadata without changing inference or preprocessing."""
    if isinstance(value, (set, frozenset)):
        return sorted(value)
    if is_dataclass(value) and (not isinstance(value, type)):
        return {key: item for (key, item) in asdict(value).items() if item is not None}
    raise TypeError(f'Object of type {type(value).__name__} is not JSON serializable')

class TrashNetTestDataset(Dataset):

    def __init__(self, samples, processor):
        self.samples = samples
        self.processor = processor

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        (image_path, label) = self.samples[index]
        with Image.open(image_path) as image:
            pixel_values = self.processor(images=image.convert('RGB'), return_tensors='pt')['pixel_values'].squeeze(0)
        return (pixel_values, label)

def ensure_images_extracted():
    if IMAGE_DIR.is_dir():
        return
    if not IMAGE_ARCHIVE.is_file():
        raise FileNotFoundError(f'未找到 TrashNet 图像目录或压缩包：{IMAGE_DIR} / {IMAGE_ARCHIVE}')
    print(f'解压本地图像包：{IMAGE_ARCHIVE}')
    with zipfile.ZipFile(IMAGE_ARCHIVE) as archive:
        archive.extractall(DATASET_DIR)
    if not IMAGE_DIR.is_dir():
        raise FileNotFoundError(f'解压后仍未找到图像目录：{IMAGE_DIR}')

def load_official_test_samples(model_label_to_id):
    if not TEST_MANIFEST.is_file():
        raise FileNotFoundError(f'未找到官方测试清单：{TEST_MANIFEST}。请从 TrashNet GitHub 的 data/one-indexed-files-notrash_test.txt 下载。')
    samples = []
    counts = {name: 0 for name in CLASS_NAMES}
    for (line_number, line) in enumerate(TEST_MANIFEST.read_text(encoding='utf-8').splitlines(), 1):
        if not line.strip():
            continue
        parts = line.split()
        if len(parts) != 2:
            raise ValueError(f"第 {line_number} 行不是 'filename label' 格式：{line!r}")
        (filename, manifest_label) = parts
        class_name = MANIFEST_LABEL_TO_NAME.get(manifest_label)
        if class_name is None:
            raise ValueError(f'第 {line_number} 行有未知 TrashNet 标签：{manifest_label}')
        filename_class = ''.join((character for character in filename if not character.isdigit())).removesuffix('.jpg')
        if filename_class != class_name:
            raise ValueError(f'第 {line_number} 行文件名类别与清单标签不一致：{filename} / {manifest_label}')
        image_path = IMAGE_DIR / class_name / filename
        if not image_path.is_file():
            raise FileNotFoundError(f'官方 test 图像缺失：{image_path}')
        samples.append((image_path, model_label_to_id[class_name]))
        counts[class_name] += 1
    if len(samples) != 431 or counts != EXPECTED_TEST_COUNTS:
        raise ValueError(f'TrashNet 官方 test 应为 431 张且类别数为 {EXPECTED_TEST_COUNTS}，实际为 {len(samples)} 张 / {counts}')
    return (samples, counts)

def safe_divide(numerator, denominator):
    return float(numerator) / float(denominator) if denominator else 0.0

def calculate_metrics(confusion, loss_sum, top1_correct, top5_correct, class_names):
    true_positive = confusion.diag().to(torch.float64)
    support = confusion.sum(dim=1).to(torch.float64)
    predicted = confusion.sum(dim=0).to(torch.float64)
    precision = torch.where(predicted > 0, true_positive / predicted, torch.zeros_like(predicted))
    recall = torch.where(support > 0, true_positive / support, torch.zeros_like(support))
    f1 = torch.where(precision + recall > 0, 2 * precision * recall / (precision + recall), torch.zeros_like(precision))
    image_count = int(support.sum().item())
    per_class = {class_name: {'support': int(support[index].item()), 'precision_pct': round(100 * precision[index].item(), 4), 'recall_pct': round(100 * recall[index].item(), 4), 'f1_pct': round(100 * f1[index].item(), 4)} for (index, class_name) in enumerate(class_names)}
    return {'image_count': image_count, 'top1_accuracy_pct': round(100 * safe_divide(top1_correct, image_count), 4), 'top5_accuracy_pct': round(100 * safe_divide(top5_correct, image_count), 4), 'macro_precision_pct': round(100 * precision.mean().item(), 4), 'macro_recall_pct': round(100 * recall.mean().item(), 4), 'balanced_accuracy_pct': round(100 * recall.mean().item(), 4), 'macro_f1_pct': round(100 * f1.mean().item(), 4), 'cross_entropy_loss': round(safe_divide(loss_sum, image_count), 6), 'per_class': per_class, 'confusion_matrix_rows_true_cols_pred': confusion.tolist()}

def save_result(result):
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = RESULTS_PATH.with_suffix('.json.tmp')
    temporary_path.write_text(json.dumps(result, default=json_metadata_default, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    os.replace(temporary_path, RESULTS_PATH)

def main():
    if not MODEL_CACHE_DIR.is_dir():
        raise FileNotFoundError(f'未找到模型缓存：{MODEL_CACHE_DIR}')
    ensure_images_extracted()
    with _test_record.timer.loading():
        processor = AutoImageProcessor.from_pretrained(str(MODEL_CACHE_DIR), local_files_only=True)
    with _test_record.timer.loading():
        model = AutoModelForImageClassification.from_pretrained(str(MODEL_CACHE_DIR), local_files_only=True)
    id2label = {int(index): name for (index, name) in model.config.id2label.items()}
    if model.config.num_labels != len(CLASS_NAMES) or set(id2label.values()) != set(CLASS_NAMES):
        raise ValueError(f'模型标签应为 {CLASS_NAMES}，实际为 {id2label}')
    model_label_to_id = {name: index for (index, name) in id2label.items()}
    ordered_class_names = [id2label[index] for index in range(len(CLASS_NAMES))]
    (samples, class_counts) = load_official_test_samples(model_label_to_id)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device_name = torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'
    with _test_record.timer.loading():
        model = model.to(device)
    with _test_record.timer.loading():
        model.eval()
    dataset = TrashNetTestDataset(samples, processor)
    dataloader = _test_record.loader(DataLoader, dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=device.type == 'cuda')
    print(f'使用设备: {device_name}')
    print(f'模型: {MODEL_ID}')
    print('数据集: TrashNet 官方 431 张 test 清单')
    print(f'测试图片数量: {len(dataset)}')
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
    class_count = len(CLASS_NAMES)
    confusion = torch.zeros((class_count, class_count), dtype=torch.long)
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
                torch.cuda.synchronize()
                forward_seconds += start_event.elapsed_time(end_event) / 1000.0
            else:
                forward_start = time.perf_counter()
                logits = _test_record.timer.invoke(model, pixel_values=pixel_values).logits
                forward_seconds += time.perf_counter() - forward_start
            topk = logits.topk(k=min(5, class_count), dim=1).indices
            predictions = topk[:, 0]
            _test_record.prediction_ready(locals())
            top1_correct += predictions.eq(labels).sum().item()
            top5_correct += topk.eq(labels.unsqueeze(1)).any(dim=1).sum().item()
            loss_sum += F.cross_entropy(logits, labels, reduction='sum').item()
            batch_confusion = torch.bincount((labels * class_count + predictions).cpu(), minlength=class_count ** 2).reshape(class_count, class_count)
            confusion += batch_confusion
            if batch_index % 10 == 0 or batch_index == len(dataloader):
                print(f'已处理: {min(batch_index * BATCH_SIZE, len(dataset))}/{len(dataset)}')
            _test_record.capture(locals())
    wall_seconds = time.perf_counter() - wall_start
    metrics = calculate_metrics(confusion, loss_sum, top1_correct, top5_correct, ordered_class_names)
    result = {'recorded_at': datetime.now().astimezone().isoformat(timespec='seconds'), 'model_id': MODEL_ID, 'model_path': str(MODEL_CACHE_DIR), 'script_path': str(Path(__file__).resolve()), 'model': {'id': MODEL_ID, 'architecture': model.__class__.__name__, 'parameter_count_millions': round(sum((parameter.numel() for parameter in model.parameters())) / 1000000.0, 4), 'num_labels': model.config.num_labels, 'id2label': {str(index): id2label[index] for index in range(class_count)}}, 'dataset': {'name': 'TrashNet', 'source': 'garythung/trashnet', 'path': str(DATASET_DIR), 'split': 'official test manifest', 'image_count': len(dataset), 'class_counts': class_counts, 'manifest': str(TEST_MANIFEST), 'manifest_url': 'https://github.com/garythung/trashnet/blob/master/data/one-indexed-files-notrash_test.txt', 'split_note': "Uses the repository's published 431-image test manifest; no random split is created."}, 'preprocessing': {'processor': 'AutoImageProcessor loaded from local model cache', 'resize': processor.size, 'image_mean': processor.image_mean, 'image_std': processor.image_std, 'batch_size': BATCH_SIZE}, 'environment': {'device': str(device), 'device_name': device_name, 'torch_version': torch.__version__, 'cuda_available': torch.cuda.is_available()}, 'official_model_card_reference': {'reported_test_accuracy_pct': 95.0, 'note': 'The model card reports 95% test accuracy but does not publish its exact split; it is not strictly comparable with this official TrashNet manifest.'}, 'metrics': metrics, 'timing': {'warmup_steps': WARMUP_STEPS, 'forward_seconds': round(forward_seconds, 6), 'forward_ms_per_image': round(1000 * forward_seconds / len(dataset), 6), 'forward_images_per_second': round(len(dataset) / forward_seconds, 4), 'wall_seconds_including_dataloader': round(wall_seconds, 6), 'wall_ms_per_image': round(1000 * wall_seconds / len(dataset), 6), 'peak_cuda_memory_mib': round(torch.cuda.max_memory_allocated(device) / 1024 ** 2, 2) if device.type == 'cuda' else None}}
    save_result(result)
    print('\n评测完成')
    print(f"Top-1: {metrics['top1_accuracy_pct']:.4f}%")
    print(f"Macro-F1: {metrics['macro_f1_pct']:.4f}%")
    print(f"Balanced Accuracy: {metrics['balanced_accuracy_pct']:.4f}%")
    print(f"Cross-entropy loss: {metrics['cross_entropy_loss']:.6f}")
    print(f"纯前向推理: {result['timing']['forward_ms_per_image']:.6f} ms/图")
    print(f'结果已保存: {RESULTS_PATH}')
    _test_record.finish(locals())
if __name__ == '__main__':
    main()
