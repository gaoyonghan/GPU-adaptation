"""在 DeepFashion 官方 test split 的 14 个兼容类别上评测服装分类 ViT。"""
if __name__ == '__main__':
    import sys as _records_sys
    from pathlib import Path as _RecordsPath
    _records_sys.path.insert(0, str(_RecordsPath(__file__).resolve().parent))
    _records_sys.path.insert(1, str(_RecordsPath(__file__).resolve().parents[1]))
    from common import start as _start_records
    _test_record = _start_records(112, __file__, globals())
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
from collections import Counter
from datetime import datetime
from pathlib import Path
import h5py
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from transformers import AutoConfig, AutoModelForImageClassification
try:
    from transformers.models.vit.image_processing_pil_vit import ViTImageProcessorPil as ViTPilProcessor
except ImportError:
    from transformers import ViTImageProcessor as ViTPilProcessor
MODEL_ID = 'aalonso-developer/vit-base-patch16-224-in21k-clothing-classifier'
ROOT = __import__('model_paths').resolve_paths()['project']
MODEL_DIR = __import__('model_paths').resolve_paths()['weights']
DATASET_DIR = __import__('model_paths').resolve_paths()['dataset'] / 'DeepFashion-Category-and-Attribute-Prediction-Benchmark'
RESULTS_PATH = _test_record.output / 'metrics.json'
if __name__ == '__main__':
    RESULTS_PATH = _test_record.output / 'metrics.json'
NUM_CLASSES = 14
BATCH_SIZE = int(os.environ.get('MODEL_BATCH_SIZE', '64'))
NUM_WORKERS = int(os.environ.get('MODEL_WORKERS', '4'))
MAX_SAMPLES = int(os.environ.get('MODEL_MAX_IMAGES', '0'))
WARMUP_STEPS = int(os.environ.get('MODEL_WARMUP', '3'))

def _tensor(dataset, transpose=False, convolution=False):
    array = np.asarray(dataset)
    if convolution:
        array = array.transpose(3, 2, 0, 1)
    elif transpose:
        array = array.T
    return torch.from_numpy(array.copy())

def load_tensorflow_h5_as_pytorch(config):
    """把仓库中的 TensorFlow ViT H5 权重严格映射到 PyTorch 模型。"""
    with _test_record.timer.loading():
        model = AutoModelForImageClassification.from_config(config)
    h5_path = MODEL_DIR / 'tf_model.h5'
    if not h5_path.is_file():
        raise FileNotFoundError(f'未找到 TensorFlow 权重：{h5_path}')
    state = {}
    with h5py.File(h5_path, 'r') as handle:
        root = 'vit/tf_vi_t_for_image_classification_3/vit'
        classifier = 'classifier/tf_vi_t_for_image_classification_3/classifier'
        with _test_record.timer.loading():
            state['vit.embeddings.cls_token'] = _tensor(handle[f'{root}/embeddings/cls_token:0'])
        with _test_record.timer.loading():
            state['vit.embeddings.position_embeddings'] = _tensor(handle[f'{root}/embeddings/position_embeddings:0'])
        with _test_record.timer.loading():
            state['vit.embeddings.patch_embeddings.projection.weight'] = _tensor(handle[f'{root}/embeddings/patch_embeddings/projection/kernel:0'], convolution=True)
        with _test_record.timer.loading():
            state['vit.embeddings.patch_embeddings.projection.bias'] = _tensor(handle[f'{root}/embeddings/patch_embeddings/projection/bias:0'])
        for layer_index in range(config.num_hidden_layers):
            tf_layer = f'{root}/encoder/layer_._{layer_index}'
            pt_layer = f'vit.encoder.layer.{layer_index}'
            dense_modules = ('attention/attention/query', 'attention/attention/key', 'attention/attention/value', 'attention/output/dense', 'intermediate/dense', 'output/dense')
            for module in dense_modules:
                pt_module = module.replace('/', '.')
                with _test_record.timer.loading():
                    state[f'{pt_layer}.{pt_module}.weight'] = _tensor(handle[f'{tf_layer}/{module}/kernel:0'], transpose=True)
                with _test_record.timer.loading():
                    state[f'{pt_layer}.{pt_module}.bias'] = _tensor(handle[f'{tf_layer}/{module}/bias:0'])
            for layernorm in ('layernorm_before', 'layernorm_after'):
                with _test_record.timer.loading():
                    state[f'{pt_layer}.{layernorm}.weight'] = _tensor(handle[f'{tf_layer}/{layernorm}/gamma:0'])
                with _test_record.timer.loading():
                    state[f'{pt_layer}.{layernorm}.bias'] = _tensor(handle[f'{tf_layer}/{layernorm}/beta:0'])
        with _test_record.timer.loading():
            state['vit.layernorm.weight'] = _tensor(handle[f'{root}/layernorm/gamma:0'])
        with _test_record.timer.loading():
            state['vit.layernorm.bias'] = _tensor(handle[f'{root}/layernorm/beta:0'])
        with _test_record.timer.loading():
            state['classifier.weight'] = _tensor(handle[f'{classifier}/kernel:0'], transpose=True)
        with _test_record.timer.loading():
            state['classifier.bias'] = _tensor(handle[f'{classifier}/bias:0'])
    if 'vit.layers.0.attention.q_proj.weight' in model.state_dict():
        replacements = (('vit.encoder.layer.', 'vit.layers.'), ('.attention.attention.query.', '.attention.q_proj.'), ('.attention.attention.key.', '.attention.k_proj.'), ('.attention.attention.value.', '.attention.v_proj.'), ('.attention.output.dense.', '.attention.o_proj.'), ('.intermediate.dense.', '.mlp.fc1.'), ('.output.dense.', '.mlp.fc2.'))
        renamed = {}
        for (key, value) in state.items():
            for (old, new) in replacements:
                key = key.replace(old, new)
            if key in renamed:
                raise ValueError(f'ViT 权重键名映射冲突：{key}')
            renamed[key] = value
        state = renamed
    with _test_record.timer.loading():
        model.load_state_dict(state, strict=True)
    return model

def read_rows(path):
    lines = path.read_text(encoding='utf-8').splitlines()
    return [line.rsplit(maxsplit=1) for line in lines[2:] if line.strip()]

def load_deepfashion_test(config):
    category_rows = read_rows(DATASET_DIR / 'Anno/list_category_cloth.txt')
    categories = {index: row[0] for (index, row) in enumerate(category_rows, start=1)}
    image_categories = {image_path: int(category_id) for (image_path, category_id) in read_rows(DATASET_DIR / 'Anno/list_category_img.txt')}
    partitions = read_rows(DATASET_DIR / 'Eval/list_eval_partition.txt')
    split_counts = Counter((split for (_, split) in partitions))
    if split_counts != Counter({'train': 209222, 'val': 40000, 'test': 40000}):
        raise ValueError(f'DeepFashion 官方划分数量异常：{dict(split_counts)}')
    label_to_model_id = {str(label): int(model_id) for (label, model_id) in config.label2id.items()}
    samples = []
    excluded = Counter()
    for (relative_path, split) in partitions:
        if split != 'test':
            continue
        category_name = categories[image_categories[relative_path]]
        if category_name not in label_to_model_id:
            excluded[category_name] += 1
            continue
        image_path = DATASET_DIR / relative_path
        if not image_path.is_file():
            raise FileNotFoundError(f'标注对应的图片不存在：{image_path}')
        samples.append((image_path, label_to_model_id[category_name]))
    if len(samples) != 32692:
        raise ValueError(f'14 类 test 子集应有 32692 张图片，实际为 {len(samples)}')
    if MAX_SAMPLES > 0:
        samples = samples[:MAX_SAMPLES]
    return (samples, split_counts, excluded)

class DeepFashionDataset(Dataset):

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

def calculate_metrics(confusion, top1_correct, top3_correct, loss_sum, labels):
    true_positive = confusion.diag().to(torch.float64)
    support = confusion.sum(dim=1).to(torch.float64)
    predicted = confusion.sum(dim=0).to(torch.float64)
    precision = torch.where(predicted > 0, true_positive / predicted, torch.zeros_like(predicted))
    recall = torch.where(support > 0, true_positive / support, torch.zeros_like(support))
    f1 = torch.where(precision + recall > 0, 2 * precision * recall / (precision + recall), torch.zeros_like(precision))
    image_count = int(support.sum().item())
    per_class = {labels[index]: {'support': int(support[index].item()), 'precision_pct': round(precision[index].item() * 100.0, 4), 'recall_pct': round(recall[index].item() * 100.0, 4), 'f1_pct': round(f1[index].item() * 100.0, 4)} for index in range(NUM_CLASSES)}
    return {'image_count': image_count, 'top1_accuracy_pct': round(100.0 * top1_correct / image_count, 4), 'top3_accuracy_pct': round(100.0 * top3_correct / image_count, 4), 'cross_entropy_loss': round(loss_sum / image_count, 6), 'macro_f1_pct': round(100.0 * f1.mean().item(), 4), 'balanced_accuracy_pct': round(100.0 * recall.mean().item(), 4), 'per_class': per_class, 'confusion_matrix_rows_true_cols_pred': confusion.tolist()}

def save_result(result):
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = RESULTS_PATH.with_suffix('.json.tmp')
    temporary_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    os.replace(temporary_path, RESULTS_PATH)

def main():
    if not torch.cuda.is_available():
        raise RuntimeError('当前进程无法访问 CUDA；为保证推理时间可比，本脚本不会回退到 CPU。')
    device = torch.device('cuda')
    print(f"使用设备: {(torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU')}")
    print(f'模型: {MODEL_ID}')
    print('数据集: DeepFashion Category and Attribute Prediction Benchmark')
    print('划分: 官方 test，仅模型支持的 14 类，不使用 bbox 裁剪')
    with _test_record.timer.loading():
        config = AutoConfig.from_pretrained(str(MODEL_DIR), local_files_only=True)
    with _test_record.timer.loading():
        processor = ViTPilProcessor.from_pretrained(str(MODEL_DIR), local_files_only=True, use_fast=False)
    with _test_record.timer.loading():
        model = load_tensorflow_h5_as_pytorch(config).to(device).eval()
    if config.num_labels != NUM_CLASSES:
        raise ValueError(f'模型应输出 {NUM_CLASSES} 类，实际为 {config.num_labels}')
    labels = [config.id2label[index] for index in range(NUM_CLASSES)]
    (samples, split_counts, excluded) = load_deepfashion_test(config)
    dataset = DeepFashionDataset(samples, processor)
    dataloader = _test_record.loader(DataLoader, dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=device.type == 'cuda', persistent_workers=NUM_WORKERS > 0)
    print(f'测试图片数量: {len(dataset)}')
    print(f'类别数量: {NUM_CLASSES}')
    print(f'batch size: {BATCH_SIZE}')
    (warmup_images, _) = next(iter(dataloader))
    warmup_images = warmup_images.to(device, non_blocking=True)
    with torch.inference_mode():
        with _test_record.timer.warmup():
            for _ in range(WARMUP_STEPS):
                _test_record.timer.invoke(model, pixel_values=warmup_images)
    if device.type == 'cuda':
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats(device)
    del warmup_images
    confusion = torch.zeros((NUM_CLASSES, NUM_CLASSES), dtype=torch.long)
    top1_correct = 0
    top3_correct = 0
    loss_sum = 0.0
    forward_seconds = 0.0
    wall_start = time.perf_counter()
    with torch.inference_mode():
        for (batch_index, (pixel_values, targets)) in enumerate(dataloader, start=1):
            _test_record.before_batch(locals())
            pixel_values = pixel_values.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
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
            loss_sum += F.cross_entropy(logits, targets, reduction='sum').item()
            top3 = logits.topk(k=3, dim=1).indices
            predictions = top3[:, 0]
            _test_record.prediction_ready(locals())
            top1_correct += predictions.eq(targets).sum().item()
            top3_correct += top3.eq(targets.unsqueeze(1)).any(dim=1).sum().item()
            bins = torch.bincount((targets * NUM_CLASSES + predictions).cpu(), minlength=NUM_CLASSES * NUM_CLASSES).reshape(NUM_CLASSES, NUM_CLASSES)
            confusion += bins
            if batch_index % 25 == 0:
                completed = min(batch_index * BATCH_SIZE, len(dataset))
                print(f'已处理: {completed}/{len(dataset)}')
            _test_record.capture(locals())
    wall_seconds = time.perf_counter() - wall_start
    metrics = calculate_metrics(confusion, top1_correct, top3_correct, loss_sum, labels)
    image_count = len(dataset)
    result = {'recorded_at': datetime.now().astimezone().isoformat(timespec='seconds'), 'model_id': MODEL_ID, 'model_path': str(MODEL_DIR), 'script_path': str(Path(__file__).resolve()), 'model': {'architecture': model.__class__.__name__, 'source_weight_format': 'TensorFlow H5', 'runtime_backend': 'PyTorch', 'parameter_count_millions': round(sum((parameter.numel() for parameter in model.parameters())) / 1000000.0, 4), 'num_labels': NUM_CLASSES, 'labels_by_output_id': labels}, 'dataset': {'name': 'DeepFashion Category and Attribute Prediction Benchmark', 'path': str(DATASET_DIR), 'split': 'official test', 'official_split_counts': dict(split_counts), 'selection': '14 categories whose names exactly match model labels', 'selected_image_count': image_count, 'full_selected_image_count': 32692, 'excluded_test_image_count': sum(excluded.values()), 'excluded_categories': dict(sorted(excluded.items()))}, 'preprocessing': {'processor': processor.__class__.__name__, 'resize': '224x224 according to preprocessor_config.json', 'normalization_mean': processor.image_mean, 'normalization_std': processor.image_std, 'bounding_box_crop': False, 'note': 'Uses ordinary DeepFashion img files without bbox cropping.'}, 'environment': {'device': torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU', 'precision': 'FP32', 'batch_size': BATCH_SIZE, 'num_workers': NUM_WORKERS, 'warmup_steps': WARMUP_STEPS}, 'metrics': metrics, 'timing': {'model_latency_ms_per_image': round(1000.0 * forward_seconds / image_count, 4), 'model_throughput_images_per_second': round(image_count / forward_seconds, 4), 'end_to_end_latency_ms_per_image': round(1000.0 * wall_seconds / image_count, 4), 'end_to_end_throughput_images_per_second': round(image_count / wall_seconds, 4)}, 'official_model_card_metrics': {'validation_accuracy_pct': 79.06, 'validation_top3_accuracy_pct': 94.37, 'validation_loss': 0.7294, 'comparability_note': 'Model card says the original training/evaluation dataset is unknown; these values are not directly comparable with this DeepFashion test.'}}
    if device.type == 'cuda':
        result['memory'] = {'peak_allocated_gib': round(torch.cuda.max_memory_allocated(device) / 2 ** 30, 4), 'peak_reserved_gib': round(torch.cuda.max_memory_reserved(device) / 2 ** 30, 4)}
    save_result(result)
    print('=' * 64)
    print(f"Top-1 accuracy: {metrics['top1_accuracy_pct']:.2f}%")
    print(f"Top-3 accuracy: {metrics['top3_accuracy_pct']:.2f}%")
    print(f"Cross-entropy loss: {metrics['cross_entropy_loss']:.6f}")
    print(f"Macro-F1: {metrics['macro_f1_pct']:.2f}%")
    print(f"Balanced accuracy: {metrics['balanced_accuracy_pct']:.2f}%")
    print(f"模型平均推理时间: {result['timing']['model_latency_ms_per_image']:.4f} ms/image")
    print(f'结果已保存: {RESULTS_PATH}')
    _test_record.finish(locals())
if __name__ == '__main__':
    main()
