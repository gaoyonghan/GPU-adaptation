"""Evaluate Bombek1/ai-image-detector-siglip-dinov2 on AIGCDetectionBenchmark."""
if __name__ == '__main__':
    import sys as _records_sys
    from pathlib import Path as _RecordsPath
    _records_sys.path.insert(0, str(_RecordsPath(__file__).resolve().parent))
    _records_sys.path.insert(1, str(_RecordsPath(__file__).resolve().parents[1]))
    from common import start as _start_records
    _test_record = _start_records(151, __file__, globals())
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
import math
import time
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
import timm
import torch
import torch.nn as nn
from peft import LoraConfig, get_peft_model
from PIL import Image, ImageFile
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from transformers import SiglipVisionConfig, SiglipVisionModel
try:
    from transformers.models.siglip.image_processing_pil_siglip import SiglipImageProcessorPil as SiglipImageProcessor
except ImportError:
    from transformers import SiglipImageProcessor
MODEL_ID = 'Bombek1/ai-image-detector-siglip-dinov2'
ROOT = __import__('model_paths').resolve_paths()['project']
MODEL_DIR = __import__('model_paths').resolve_paths()['weights']
CHECKPOINT_PATH = MODEL_DIR / 'pytorch_model.pt'
DATASET_DIR = __import__('model_paths').resolve_paths()['dataset'] / 'AIGCDetectionBenchmark/test'
RESULTS_PATH = _test_record.output / 'metrics.json'
if __name__ == '__main__':
    RESULTS_PATH = _test_record.output / 'metrics.json'
SIGLIP_MODEL_ID = 'google/siglip2-so400m-patch14-384'
DINOV2_MODEL_ID = 'vit_large_patch14_dinov2.lvd142m'
BATCH_SIZE = int(os.environ.get('MODEL_BATCH_SIZE', '256'))
NUM_WORKERS = int(os.environ.get('MODEL_WORKERS', '4'))
WARMUP_STEPS = int(os.environ.get('MODEL_WARMUP', '3'))
IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.webp', '.bmp', '.tif', '.tiff'}
REAL_LABEL = 0
FAKE_LABEL = 1
IMAGE_SIZE = 392
ImageFile.LOAD_TRUNCATED_IMAGES = True

class LoRALinear(nn.Module):

    def __init__(self, original, rank, alpha, dropout=0.1):
        super().__init__()
        self.original = original
        self.scaling = alpha / rank
        for parameter in self.original.parameters():
            parameter.requires_grad = False
        self.lora_A = nn.Linear(original.in_features, rank, bias=False)
        self.lora_B = nn.Linear(rank, original.out_features, bias=False)
        self.dropout = nn.Dropout(dropout)
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)

    def forward(self, values):
        return self.original(values) + self.lora_B(self.lora_A(self.dropout(values))) * self.scaling

class ClassificationHead(nn.Module):

    def __init__(self, input_dim, hidden_dim=512, dropout=0.3):
        super().__init__()
        self.head = nn.Sequential(nn.LayerNorm(input_dim), nn.Linear(input_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden_dim, hidden_dim // 2), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden_dim // 2, 1))

    def forward(self, values):
        return self.head(values).squeeze(-1)

class EnsembleAIDetector(nn.Module):

    def __init__(self, siglip, dinov2, classifier):
        super().__init__()
        self.siglip = siglip
        self.dinov2 = dinov2
        self.classifier = classifier

    def forward(self, siglip_pixels, dinov2_pixels):
        siglip_features = self.siglip(pixel_values=siglip_pixels).pooler_output
        dinov2_features = self.dinov2(dinov2_pixels)
        combined = torch.cat([siglip_features.float(), dinov2_features], dim=-1)
        return self.classifier(combined)

class BenchmarkDataset(Dataset):

    def __init__(self, records, siglip_processor, dinov2_transform):
        self.records = records
        self.siglip_processor = siglip_processor
        self.dinov2_transform = dinov2_transform

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        (path, _source, _subcategory, label) = self.records[index]
        with Image.open(path) as image:
            image = image.convert('RGB')
            siglip_pixels = self.siglip_processor(images=image, return_tensors='pt')['pixel_values'].squeeze(0)
            dinov2_pixels = self.dinov2_transform(image)
        return (siglip_pixels, dinov2_pixels, label)

def safe_divide(numerator, denominator):
    return float(numerator) / float(denominator) if denominator else 0.0

def collect_records():
    if not DATASET_DIR.is_dir():
        raise FileNotFoundError(f'未找到 AIGCDetectionBenchmark 测试目录：{DATASET_DIR}')
    records = []
    for path in sorted(DATASET_DIR.rglob('*')):
        if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        parts = path.relative_to(DATASET_DIR).parts
        label_index = next((index for (index, part) in enumerate(parts) if part in {'0_real', '1_fake'}), None)
        if label_index is None or label_index == 0:
            raise ValueError(f'图片路径缺少有效的生成器/标签目录：{path}')
        source = parts[0]
        subcategory = '/'.join(parts[1:label_index]) or None
        label = REAL_LABEL if parts[label_index] == '0_real' else FAKE_LABEL
        records.append((path, source, subcategory, label))
    if not records:
        raise ValueError(f'测试目录中没有可识别的图片：{DATASET_DIR}')
    return records

def build_model():
    if not MODEL_DIR.is_dir() or not CHECKPOINT_PATH.is_file():
        raise FileNotFoundError(f'未找到本地模型 checkpoint：{CHECKPOINT_PATH}')
    checkpoint = torch.load(CHECKPOINT_PATH, map_location='cpu', weights_only=False)
    model_config = checkpoint.get('config', {})
    backbone_config = json.loads((MODEL_DIR / 'siglip_config.json').read_text(encoding='utf-8'))
    siglip_config = SiglipVisionConfig.from_dict(backbone_config['vision_config'])
    siglip = SiglipVisionModel(siglip_config).to(dtype=torch.bfloat16)
    siglip = get_peft_model(siglip, LoraConfig(r=model_config.get('lora_rank', 32), lora_alpha=model_config.get('lora_alpha', 64), target_modules=['q_proj', 'v_proj'], lora_dropout=model_config.get('lora_dropout', 0.1), bias='none'))
    dinov2 = timm.create_model(model_config.get('dinov2_model', DINOV2_MODEL_ID), pretrained=False, num_classes=0, img_size=model_config.get('image_size', IMAGE_SIZE))
    for (_name, module) in dinov2.named_modules():
        if hasattr(module, 'qkv') and isinstance(module.qkv, nn.Linear):
            module.qkv = LoRALinear(module.qkv, model_config.get('lora_rank', 32), model_config.get('lora_alpha', 64), model_config.get('lora_dropout', 0.1))
    with _test_record.timer.loading():
        model = EnsembleAIDetector(siglip, dinov2, ClassificationHead(2176))
    expected_keys = set(model.state_dict())
    state_dict = {}
    renamed = 0
    for (key, value) in checkpoint['model_state_dict'].items():
        target = key
        old_prefix = 'siglip.base_model.model.vision_model.'
        if key not in expected_keys and key.startswith(old_prefix):
            candidate = 'siglip.base_model.model.' + key[len(old_prefix):]
            if candidate in expected_keys:
                target = candidate
                renamed += 1
        if target in state_dict:
            raise ValueError(f'checkpoint 参数名称映射冲突：{target}')
        state_dict[target] = value
    (missing, unexpected) = model.load_state_dict(state_dict, strict=True)
    print(f'checkpoint 参数严格匹配成功，兼容映射 {renamed} 个 SigLIP 参数名')
    del state_dict
    del checkpoint
    if missing or unexpected:
        raise ValueError(f'checkpoint 参数未完全匹配，missing={missing[:5]} unexpected={unexpected[:5]}')
    return model

def binary_metrics(labels, predictions, fake_scores):
    true_negative = sum((label == REAL_LABEL and prediction == REAL_LABEL for (label, prediction) in zip(labels, predictions)))
    false_positive = sum((label == REAL_LABEL and prediction == FAKE_LABEL for (label, prediction) in zip(labels, predictions)))
    false_negative = sum((label == FAKE_LABEL and prediction == REAL_LABEL for (label, prediction) in zip(labels, predictions)))
    true_positive = sum((label == FAKE_LABEL and prediction == FAKE_LABEL for (label, prediction) in zip(labels, predictions)))
    image_count = len(labels)
    real_count = sum((label == REAL_LABEL for label in labels))
    fake_count = sum((label == FAKE_LABEL for label in labels))
    real_recall = safe_divide(true_negative, real_count)
    fake_recall = safe_divide(true_positive, fake_count)
    fake_precision = safe_divide(true_positive, true_positive + false_positive)
    fake_f1 = safe_divide(2.0 * fake_precision * fake_recall, fake_precision + fake_recall)
    binary_log_loss = safe_divide(sum((-math.log(max(min(score, 1.0 - 1e-15), 1e-15)) if label == FAKE_LABEL else -math.log(max(min(1.0 - score, 1.0 - 1e-15), 1e-15)) for (label, score) in zip(labels, fake_scores))), image_count)
    return {'image_count': image_count, 'real_count': real_count, 'fake_count': fake_count, 'accuracy_pct': round(100.0 * safe_divide(true_negative + true_positive, image_count), 4), 'balanced_accuracy_pct': round(50.0 * (real_recall + fake_recall), 4), 'fake_precision_pct': round(100.0 * fake_precision, 4), 'fake_recall_pct': round(100.0 * fake_recall, 4), 'fake_f1_pct': round(100.0 * fake_f1, 4), 'real_specificity_pct': round(100.0 * real_recall, 4), 'binary_log_loss': round(binary_log_loss, 6), 'confusion_matrix_rows_true_cols_pred': [[true_negative, false_positive], [false_negative, true_positive]], 'confusion_matrix_labels': ['Real', 'Fake'], 'mean_fake_probability_pct': round(100.0 * safe_divide(sum(fake_scores), image_count), 4)}

def binary_auroc(labels, scores):
    positive_count = sum((label == FAKE_LABEL for label in labels))
    negative_count = len(labels) - positive_count
    if not positive_count or not negative_count:
        return None
    order = sorted(range(len(scores)), key=scores.__getitem__)
    rank_sum = 0.0
    index = 0
    while index < len(order):
        end = index + 1
        score = scores[order[index]]
        while end < len(order) and scores[order[end]] == score:
            end += 1
        average_rank = (index + 1 + end) / 2.0
        rank_sum += sum((average_rank for position in order[index:end] if labels[position] == FAKE_LABEL))
        index = end
    return round((rank_sum - positive_count * (positive_count + 1) / 2.0) / (positive_count * negative_count), 6)

def expected_calibration_error(labels, predictions, confidences, bins=15):
    ece = 0.0
    for bin_index in range(bins):
        lower = bin_index / bins
        upper = (bin_index + 1) / bins
        indices = [index for (index, confidence) in enumerate(confidences) if lower <= confidence < upper or (bin_index == bins - 1 and confidence == upper)]
        if indices:
            accuracy = sum((labels[index] == predictions[index] for index in indices)) / len(indices)
            mean_confidence = sum((confidences[index] for index in indices)) / len(indices)
            ece += len(indices) / len(labels) * abs(accuracy - mean_confidence)
    return round(ece, 6)

def build_source_metrics(source_states):
    result = {}
    for source in sorted(source_states):
        state = source_states[source]
        metrics = binary_metrics(state['labels'], state['predictions'], state['scores'])
        metrics['auroc'] = binary_auroc(state['labels'], state['scores'])
        metrics['subcategories'] = {}
        for (name, value) in sorted(state['subcategories'].items()):
            sub_metrics = binary_metrics(value['labels'], value['predictions'], value['scores'])
            sub_metrics['auroc'] = binary_auroc(value['labels'], value['scores'])
            metrics['subcategories'][name] = sub_metrics
        result[source] = metrics
    return result

def save_result(result):
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = RESULTS_PATH.with_suffix('.json.tmp')
    temporary_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    os.replace(temporary_path, RESULTS_PATH)

def main():
    records = collect_records()
    with _test_record.timer.loading():
        model = build_model()
    with _test_record.timer.loading():
        processor = SiglipImageProcessor.from_dict(json.loads((MODEL_DIR / 'siglip_preprocessor_config.json').read_text(encoding='utf-8')))
    dinov2_transform = transforms.Compose([transforms.Resize((IMAGE_SIZE, IMAGE_SIZE), interpolation=transforms.InterpolationMode.BICUBIC), transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    dataset = BenchmarkDataset(records, processor, dinov2_transform)
    dataloader = _test_record.loader(DataLoader, dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=torch.cuda.is_available())
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device_name = torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'
    with _test_record.timer.loading():
        model = model.to(device).eval()
    source_counts = Counter((source for (_path, source, _subcategory, _label) in records))
    label_counts = Counter((label for (_path, _source, _subcategory, label) in records))
    parameter_count = sum((parameter.numel() for parameter in model.parameters()))
    print(f'使用设备: {device_name}')
    print(f'模型: {MODEL_ID}')
    print('数据集: AIGCDetectionBenchmark/test')
    print(f'测试图片数量: {len(dataset)}')
    print(f'真实/伪造数量: {label_counts[REAL_LABEL]}/{label_counts[FAKE_LABEL]}')
    print(f'生成器数量: {len(source_counts)}')
    print(f'模型参数量: {parameter_count / 1000000.0:.4f} M')
    print(f'batch size: {BATCH_SIZE}')
    (warmup_siglip, warmup_dino, _) = next(iter(dataloader))
    warmup_siglip = warmup_siglip.to(device, non_blocking=True)
    warmup_dino = warmup_dino.to(device, non_blocking=True)
    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = True
        torch.cuda.reset_peak_memory_stats(device)
    with torch.inference_mode():
        with _test_record.timer.warmup():
            for _ in range(WARMUP_STEPS):
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    _test_record.timer.invoke(model, warmup_siglip, warmup_dino)
    if device.type == 'cuda':
        torch.cuda.synchronize()
    del warmup_siglip, warmup_dino
    (labels, predictions, fake_scores, confidences) = ([], [], [], [])
    source_states = defaultdict(lambda : {'labels': [], 'predictions': [], 'scores': [], 'subcategories': defaultdict(lambda : {'labels': [], 'predictions': [], 'scores': []})})
    forward_seconds = 0.0
    wall_start = time.perf_counter()
    with torch.inference_mode():
        for (batch_index, (siglip_pixels, dinov2_pixels, batch_labels)) in enumerate(dataloader, start=1):
            _test_record.before_batch(locals())
            siglip_pixels = siglip_pixels.to(device, non_blocking=True)
            dinov2_pixels = dinov2_pixels.to(device, non_blocking=True)
            if device.type == 'cuda':
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)
                start_event.record()
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    logits = _test_record.timer.invoke(model, siglip_pixels, dinov2_pixels)
                end_event.record()
                torch.cuda.synchronize()
                forward_seconds += start_event.elapsed_time(end_event) / 1000.0
            else:
                forward_start = time.perf_counter()
                logits = _test_record.timer.invoke(model, siglip_pixels, dinov2_pixels)
                forward_seconds += time.perf_counter() - forward_start
            batch_fake_scores = torch.sigmoid(logits).float()
            batch_predictions = batch_fake_scores.ge(0.5).long()
            _test_record.prediction_ready(locals())
            batch_confidences = torch.where(batch_predictions.eq(FAKE_LABEL), batch_fake_scores, 1.0 - batch_fake_scores)
            batch_labels_list = batch_labels.tolist()
            batch_predictions_list = batch_predictions.cpu().tolist()
            batch_scores_list = batch_fake_scores.cpu().tolist()
            labels.extend(batch_labels_list)
            predictions.extend(batch_predictions_list)
            fake_scores.extend(batch_scores_list)
            confidences.extend(batch_confidences.cpu().tolist())
            start_index = (batch_index - 1) * BATCH_SIZE
            for (record_index, (label, prediction, score)) in enumerate(zip(batch_labels_list, batch_predictions_list, batch_scores_list), start=start_index):
                (_path, source, subcategory, _record_label) = records[record_index]
                state = source_states[source]
                state['labels'].append(label)
                state['predictions'].append(prediction)
                state['scores'].append(score)
                subcategory_state = state['subcategories'][subcategory or '(root)']
                subcategory_state['labels'].append(label)
                subcategory_state['predictions'].append(prediction)
                subcategory_state['scores'].append(score)
            if batch_index % 25 == 0 or batch_index == len(dataloader):
                print(f'已处理: {min(batch_index * BATCH_SIZE, len(dataset))}/{len(dataset)}')
            _test_record.capture(locals())
    if device.type == 'cuda':
        torch.cuda.synchronize()
    wall_seconds = time.perf_counter() - wall_start
    metrics = binary_metrics(labels, predictions, fake_scores)
    metrics['auroc'] = binary_auroc(labels, fake_scores)
    metrics['expected_calibration_error'] = expected_calibration_error(labels, predictions, confidences)
    result = {'recorded_at': datetime.now().astimezone().isoformat(timespec='seconds'), 'model_id': MODEL_ID, 'model_path': str(MODEL_DIR), 'checkpoint_path': str(CHECKPOINT_PATH), 'script_path': str(Path(__file__).resolve()), 'model': {'id': MODEL_ID, 'architecture': 'SigLIP2 + DINOv2 Large ensemble with LoRA', 'parameter_count_millions': round(parameter_count / 1000000.0, 4), 'num_labels': 2, 'id2label': {'0': 'real', '1': 'ai-generated'}, 'checkpoint_step': 2000, 'checkpoint_best_auc': 0.99975}, 'dataset': {'name': 'AIGCDetectionBenchmark', 'path': str(DATASET_DIR), 'split': 'test', 'image_count': len(dataset), 'source_count': len(source_counts), 'source_image_counts': dict(sorted(source_counts.items())), 'binary_label_counts': {'Real': label_counts[REAL_LABEL], 'Fake': label_counts[FAKE_LABEL]}, 'label_protocol': '0_real and 1_fake directory labels are mapped to Real and Fake.'}, 'evaluation': {'task': 'binary real-vs-AI-generated classification', 'fake_probability': "sigmoid of the checkpoint's scalar classifier logit", 'three_class_accuracy_note': 'The model is binary; no separate deepfake subclass is reported.'}, 'environment': {'device': device_name, 'precision': 'bfloat16 autocast on CUDA', 'batch_size': BATCH_SIZE, 'warmup_steps': WARMUP_STEPS, 'num_workers': NUM_WORKERS}, 'preprocessing': {'siglip': 'SiglipImageProcessor from google/siglip2-so400m-patch14-384 (384x384, model processor normalization)', 'dinov2': 'Resize 392x392 bicubic, ToTensor, ImageNet mean/std normalization', 'truncated_image_policy': 'PIL LOAD_TRUNCATED_IMAGES=True; truncated files are evaluated using decodable content'}, 'metrics': metrics, 'raw_predictions': {'top1_prediction_counts': {'Real': sum((prediction == REAL_LABEL for prediction in predictions)), 'Fake': sum((prediction == FAKE_LABEL for prediction in predictions))}, 'top1_prediction_pct': {'Real': round(100.0 * safe_divide(sum((prediction == REAL_LABEL for prediction in predictions)), len(dataset)), 4), 'Fake': round(100.0 * safe_divide(sum((prediction == FAKE_LABEL for prediction in predictions)), len(dataset)), 4)}}, 'per_source': build_source_metrics(source_states), 'timing': {'model_latency_ms_per_image': round(forward_seconds / len(dataset) * 1000.0, 4), 'model_throughput_images_per_second': round(len(dataset) / forward_seconds, 4), 'end_to_end_latency_ms_per_image': round(wall_seconds / len(dataset) * 1000.0, 4), 'end_to_end_throughput_images_per_second': round(len(dataset) / wall_seconds, 4)}}
    if device.type == 'cuda':
        result['memory'] = {'peak_allocated_gib': round(torch.cuda.max_memory_allocated(device) / 2 ** 30, 4), 'peak_reserved_gib': round(torch.cuda.max_memory_reserved(device) / 2 ** 30, 4)}
    save_result(result)
    print('=' * 64)
    print(f"Binary accuracy: {metrics['accuracy_pct']:.2f}%")
    print(f"Balanced accuracy: {metrics['balanced_accuracy_pct']:.2f}%")
    print(f"Fake F1: {metrics['fake_f1_pct']:.2f}%")
    print(f"AUROC: {metrics['auroc']:.6f}")
    print(f"模型平均推理时间: {result['timing']['model_latency_ms_per_image']:.4f} ms/image")
    print(f"模型吞吐量: {result['timing']['model_throughput_images_per_second']:.2f} images/s")
    print(f'评测结果已保存: {RESULTS_PATH}')
    _test_record.finish(locals())
if __name__ == '__main__':
    main()
