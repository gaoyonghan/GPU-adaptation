"""评测 mrm8488/vit-base-patch16-224_finetuned-kvasirv2-colonoscopy。

本地 San-D/Kvasir_V2 仓库只提供 800 张 train 标注图像（每类 100 张），
因此这里明确将它记录为 local_800_image_test_candidate，而不是官方 test split。
"""
if __name__ == '__main__':
    import sys as _records_sys
    from pathlib import Path as _RecordsPath
    _records_sys.path.insert(0, str(_RecordsPath(__file__).resolve().parent))
    _records_sys.path.insert(1, str(_RecordsPath(__file__).resolve().parents[1]))
    from common import start as _start_records
    _test_record = _start_records(157, __file__, globals())
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
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from transformers import AutoImageProcessor, AutoModelForImageClassification
MODEL_ID = 'mrm8488/vit-base-patch16-224_finetuned-kvasirv2-colonoscopy'
ROOT = __import__('model_paths').resolve_paths()['project']
MODEL_CACHE_DIR = __import__('model_paths').resolve_paths()['weights']
DATASET_DIR = __import__('model_paths').resolve_paths()['dataset'] / 'Kvasir_V2/kvasir-dataset-v2'
RESULTS_PATH = _test_record.output / 'metrics.json'
if __name__ == '__main__':
    RESULTS_PATH = _test_record.output / 'metrics.json'
BATCH_SIZE = int(os.environ.get('MODEL_BATCH_SIZE', '64'))
NUM_WORKERS = int(os.environ.get('MODEL_WORKERS', '0'))
WARMUP_STEPS = int(os.environ.get('MODEL_WARMUP', '3'))
ECE_BINS = 15
CLASS_NAMES = ['dyed-lifted-polyps', 'dyed-resection-margins', 'esophagitis', 'normal-cecum', 'normal-pylorus', 'normal-z-line', 'polyps', 'ulcerative-colitis']
IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}

class KvasirDataset(Dataset):

    def __init__(self, root: Path, processor):
        self.processor = processor
        self.samples = []
        for (label, class_name) in enumerate(CLASS_NAMES):
            class_dir = root / class_name
            if not class_dir.is_dir():
                raise FileNotFoundError(f'未找到类别目录：{class_dir}')
            paths = sorted((path for path in class_dir.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS))
            if not paths:
                raise FileNotFoundError(f'类别目录没有图像：{class_dir}')
            self.samples.extend(((path, label) for path in paths))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        (path, label) = self.samples[index]
        with Image.open(path) as image:
            pixel_values = self.processor(images=image.convert('RGB'), return_tensors='pt')['pixel_values'].squeeze(0)
        return (pixel_values, label)

def safe_divide(numerator, denominator):
    return float(numerator) / float(denominator) if denominator else 0.0

def compute_metrics(labels, predictions, top5_predictions, losses, confidences, class_count):
    image_count = len(labels)
    confusion = [[0 for _ in range(class_count)] for _ in range(class_count)]
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
    top1_correct = sum((label == prediction for (label, prediction) in zip(labels, predictions)))
    top5_correct = sum((label in candidates for (label, candidates) in zip(labels, top5_predictions)))
    ece = 0.0
    for bin_index in range(ECE_BINS):
        lower = bin_index / ECE_BINS
        upper = (bin_index + 1) / ECE_BINS
        if bin_index == ECE_BINS - 1:
            selected = [i for (i, confidence) in enumerate(confidences) if lower <= confidence <= upper]
        else:
            selected = [i for (i, confidence) in enumerate(confidences) if lower <= confidence < upper]
        if selected:
            accuracy = sum((labels[i] == predictions[i] for i in selected)) / len(selected)
            average_confidence = sum((confidences[i] for i in selected)) / len(selected)
            ece += len(selected) / image_count * abs(accuracy - average_confidence)
    return {'image_count': image_count, 'top1_accuracy_pct': round(100 * safe_divide(top1_correct, image_count), 4), 'top5_accuracy_pct': round(100 * safe_divide(top5_correct, image_count), 4), 'macro_f1_pct': round(100 * sum(f1_scores) / class_count, 4), 'balanced_accuracy_pct': round(100 * sum(recalls) / class_count, 4), 'cross_entropy_loss': round(safe_divide(sum(losses), image_count), 6), 'expected_calibration_error': round(ece, 6), 'per_class': per_class, 'confusion_matrix_rows_true_cols_pred': confusion}

def save_result(result):
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = RESULTS_PATH.with_suffix('.json.tmp')
    temporary_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    os.replace(temporary_path, RESULTS_PATH)

def main():
    if not MODEL_CACHE_DIR.is_dir():
        raise FileNotFoundError(f'未找到模型缓存：{MODEL_CACHE_DIR}')
    if not DATASET_DIR.is_dir():
        raise FileNotFoundError(f'未找到 Kvasir-v2 数据：{DATASET_DIR}')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device_name = torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'
    print(f'使用设备: {device_name}')
    print(f'模型: {MODEL_ID}')
    print('数据集: Kvasir-v2 本地 800 张分层测试候选集')
    with _test_record.timer.loading():
        processor = AutoImageProcessor.from_pretrained(str(MODEL_CACHE_DIR), local_files_only=True)
    with _test_record.timer.loading():
        model = AutoModelForImageClassification.from_pretrained(str(MODEL_CACHE_DIR), local_files_only=True).to(device)
    with _test_record.timer.loading():
        model.eval()
    class_count = int(model.config.num_labels)
    if class_count != len(CLASS_NAMES):
        raise ValueError(f'模型输出类别数为 {class_count}，预期为 {len(CLASS_NAMES)}')
    id2label = {int(key): value for (key, value) in model.config.id2label.items()}
    if [id2label[index] for index in range(class_count)] != CLASS_NAMES:
        raise ValueError(f'模型标签顺序与数据目录不一致：{id2label}')
    dataset = KvasirDataset(DATASET_DIR, processor)
    dataloader = _test_record.loader(DataLoader, dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=device.type == 'cuda')
    class_support = [0] * class_count
    for (_, label) in dataset.samples:
        class_support[label] += 1
    if len(set(class_support)) != 1 or class_support[0] != 100:
        raise ValueError(f'预期每类 100 张，实际类别数量：{class_support}')
    print(f'验证图片数量: {len(dataset)}')
    print(f'每类图片数量: {class_support}')
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
            batch_top5 = logits.topk(min(5, class_count), dim=1).indices
            _test_record.prediction_ready(locals())
            batch_loss = F.cross_entropy(logits, batch_labels, reduction='none')
            labels.extend(batch_labels.cpu().tolist())
            predictions.extend(batch_predictions.cpu().tolist())
            top5_predictions.extend(batch_top5.cpu().tolist())
            losses.extend(batch_loss.cpu().tolist())
            confidences.extend(confidence.cpu().tolist())
            if batch_index % 5 == 0 or batch_index == len(dataloader):
                print(f'已处理: {min(batch_index * BATCH_SIZE, len(dataset))}/{len(dataset)}')
            _test_record.capture(locals())
    wall_seconds = time.perf_counter() - wall_start
    metrics = compute_metrics(labels, predictions, top5_predictions, losses, confidences, class_count)
    result = {'recorded_at': datetime.now().astimezone().isoformat(timespec='seconds'), 'model_id': MODEL_ID, 'model_path': str(MODEL_CACHE_DIR), 'script_path': str(Path(__file__).resolve()), 'model': {'id': MODEL_ID, 'parameter_count_millions': round(sum((parameter.numel() for parameter in model.parameters())) / 1000000.0, 4), 'num_labels': class_count, 'id2label': {str(key): value for (key, value) in id2label.items()}}, 'dataset': {'name': 'Kvasir-v2', 'source': 'San-D/Kvasir_V2', 'path': str(DATASET_DIR), 'split': 'local_800_image_test_candidate', 'image_count': len(dataset), 'class_names': CLASS_NAMES, 'class_support': dict(zip(CLASS_NAMES, class_support)), 'split_note': "The local San-D/Kvasir_V2 repository exposes one train split of 800 images (100 per class); it is not the model card's unreleased 480-image evaluation split.", 'original_dataset_note': 'Kvasir-v2 is an 8-class endoscopy image dataset from Simula Research Laboratory.'}, 'preprocessing': {'processor': 'AutoImageProcessor loaded from local model cache', 'image_size': 224, 'batch_size': BATCH_SIZE}, 'environment': {'device': str(device), 'device_name': device_name, 'torch_version': torch.__version__, 'cuda_available': torch.cuda.is_available()}, 'metrics': metrics, 'timing': {'warmup_steps': WARMUP_STEPS, 'forward_seconds': round(forward_seconds, 6), 'forward_ms_per_image': round(1000 * forward_seconds / len(dataset), 6), 'forward_images_per_second': round(safe_divide(len(dataset), forward_seconds), 4), 'wall_seconds_including_dataloader': round(wall_seconds, 6), 'wall_ms_per_image': round(1000 * wall_seconds / len(dataset), 6), 'peak_cuda_memory_mib': round(torch.cuda.max_memory_allocated(device) / 1024 ** 2, 2) if device.type == 'cuda' else None}}
    save_result(result)
    print('\n评测完成')
    print(f"Top-1 accuracy: {metrics['top1_accuracy_pct']:.4f}%")
    print(f"Top-5 accuracy: {metrics['top5_accuracy_pct']:.4f}%")
    print(f"Macro-F1: {metrics['macro_f1_pct']:.4f}%")
    print(f"Cross-entropy loss: {metrics['cross_entropy_loss']:.6f}")
    print(f'Forward time: {forward_seconds:.4f}s ({1000 * forward_seconds / len(dataset):.4f} ms/image)')
    print(f'结果已保存: {RESULTS_PATH}')
    _test_record.finish(locals())
if __name__ == '__main__':
    main()
