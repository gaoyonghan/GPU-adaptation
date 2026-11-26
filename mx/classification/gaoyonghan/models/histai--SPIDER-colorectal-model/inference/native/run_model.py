"""在 SPIDER-colorectal 官方 test split 上评测同源基线模型。"""
if __name__ == '__main__':
    import sys as _records_sys
    from pathlib import Path as _RecordsPath
    _records_sys.path.insert(0, str(_RecordsPath(__file__).resolve().parent))
    _records_sys.path.insert(1, str(_RecordsPath(__file__).resolve().parents[1]))
    from common import start as _start_records
    _test_record = _start_records(146, __file__, globals())
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
from dataclasses import asdict, is_dataclass
from functools import lru_cache
from itertools import chain
from pathlib import Path
import torch
import torch.nn.functional as F
from PIL import Image
from safetensors.torch import load_file
from transformers import AutoConfig, AutoImageProcessor, AutoModel
MODEL_ID = 'histai/SPIDER-colorectal-model'
DATASET_ID = 'histai/SPIDER-colorectal'
ROOT = __import__('model_paths').resolve_paths()['project']
MODEL_DIR = __import__('model_paths').resolve_paths()['weights']
DATASET_ROOT = __import__('model_paths').resolve_paths()['dataset'] / 'SPIDER-colorectal/SPIDER-colorectal'
METADATA_PATH = DATASET_ROOT / 'metadata.json'
TEST_ZIP_PATH = DATASET_ROOT / 'test-images.zip'
RESULTS_PATH = _test_record.output / 'metrics.json'
if __name__ == '__main__':
    RESULTS_PATH = _test_record.output / 'metrics.json'
NUM_CLASSES = 13
EXPECTED_TEST_IMAGES = 13193
BATCH_SIZE = int(os.environ.get('MODEL_BATCH_SIZE', '8'))
MAX_SAMPLES = int(os.environ.get('MODEL_MAX_IMAGES', '0'))
WARMUP_STEPS = int(os.environ.get('MODEL_WARMUP', '3'))
PATCH_SIZE = 224
GRID_SIZE = 5

def load_test_records(class_names):
    metadata = json.loads(METADATA_PATH.read_text(encoding='utf-8'))
    records = [record for record in metadata if record.get('split') == 'test']
    if len(records) != EXPECTED_TEST_IMAGES:
        raise ValueError(f'官方 test split 应有 {EXPECTED_TEST_IMAGES} 条，实际为 {len(records)} 条')
    class_to_id = {name: index for (index, name) in enumerate(class_names)}
    unknown = sorted({record['class'] for record in records} - set(class_to_id))
    if unknown:
        raise ValueError(f'数据集包含模型未知类别：{unknown}')
    if any((len(record.get('context_info', {})) != 24 for record in records)):
        raise ValueError('test split 中存在上下文patch不完整的记录')
    if MAX_SAMPLES > 0:
        records = records[:MAX_SAMPLES]
    return (records, class_to_id)

class TestImageBuilder:

    def __init__(self, zip_path):
        self.archive = zipfile.ZipFile(zip_path, 'r')
        self.available = set(self.archive.namelist())

    @lru_cache(maxsize=4096)
    def load_patch(self, filename):
        if filename not in self.available:
            raise FileNotFoundError(f'test ZIP 中缺少patch：{filename}')
        with Image.open(io.BytesIO(self.archive.read(filename))) as image:
            return image.convert('RGB')

    def build(self, record):
        stitched = Image.new('RGB', (GRID_SIZE * PATCH_SIZE, GRID_SIZE * PATCH_SIZE))
        for row in range(GRID_SIZE):
            for column in range(GRID_SIZE):
                if row == GRID_SIZE // 2 and column == GRID_SIZE // 2:
                    filename = record['image_name']
                else:
                    filename = record['context_info'][f'{row}_{column}']
                stitched.paste(self.load_patch(filename), (column * PATCH_SIZE, row * PATCH_SIZE))
        return stitched

    def close(self):
        self.load_patch.cache_clear()
        self.archive.close()

def iter_batches(records, class_to_id, builder):
    for start in range(0, len(records), BATCH_SIZE):
        batch_records = records[start:start + BATCH_SIZE]
        images = [builder.build(record) for record in batch_records]
        labels = torch.tensor([class_to_id[record['class']] for record in batch_records], dtype=torch.long)
        yield (images, labels)

def calculate_metrics(confusion, top1_correct, top3_correct, loss_sum, class_names):
    true_positive = confusion.diag().to(torch.float64)
    support = confusion.sum(dim=1).to(torch.float64)
    predicted = confusion.sum(dim=0).to(torch.float64)
    precision = torch.where(predicted > 0, true_positive / predicted, torch.zeros_like(predicted))
    recall = torch.where(support > 0, true_positive / support, torch.zeros_like(support))
    f1 = torch.where(precision + recall > 0, 2.0 * precision * recall / (precision + recall), torch.zeros_like(precision))
    image_count = int(support.sum().item())
    weights = support / image_count
    per_class = {class_names[index]: {'support': int(support[index].item()), 'precision_pct': round(100.0 * precision[index].item(), 4), 'recall_pct': round(100.0 * recall[index].item(), 4), 'f1_pct': round(100.0 * f1[index].item(), 4)} for index in range(NUM_CLASSES)}
    return {'image_count': image_count, 'top1_accuracy_pct': round(100.0 * top1_correct / image_count, 4), 'top3_accuracy_pct': round(100.0 * top3_correct / image_count, 4), 'cross_entropy_loss': round(loss_sum / image_count, 6), 'macro_precision_pct': round(100.0 * precision.mean().item(), 4), 'macro_recall_pct': round(100.0 * recall.mean().item(), 4), 'macro_f1_pct': round(100.0 * f1.mean().item(), 4), 'weighted_precision_pct': round(100.0 * (precision * weights).sum().item(), 4), 'weighted_recall_pct': round(100.0 * (recall * weights).sum().item(), 4), 'weighted_f1_pct': round(100.0 * (f1 * weights).sum().item(), 4), 'per_class': per_class, 'confusion_matrix_rows_true_cols_pred': confusion.tolist()}

def json_metadata_default(value):
    """只转换库元数据，不改变预处理或推理。"""
    if is_dataclass(value) and (not isinstance(value, type)):
        return {key: item for (key, item) in asdict(value).items() if item is not None}
    raise TypeError(f'Object of type {type(value).__name__} is not JSON serializable')

def save_result(result):
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = RESULTS_PATH.with_suffix('.json.tmp')
    temporary_path.write_text(json.dumps(result, default=json_metadata_default, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    os.replace(temporary_path, RESULTS_PATH)

def main():
    if not torch.cuda.is_available():
        raise RuntimeError('当前进程无法访问 CUDA；为保证推理时间可比，本脚本不会回退到 CPU。')
    if not TEST_ZIP_PATH.is_file():
        raise FileNotFoundError(f'未找到test缓存：{TEST_ZIP_PATH}，请先运行prepare脚本')
    device = torch.device('cuda')
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    print(f'使用设备: {torch.cuda.get_device_name(0)}')
    print(f'模型: {MODEL_ID}')
    print(f'数据集: {DATASET_ID}, 官方 slide-level test split')
    print('输入: 5x5个224x224 patch拼接为1120x1120')
    with _test_record.timer.loading():
        processor = AutoImageProcessor.from_pretrained(str(MODEL_DIR), local_files_only=True)
    with _test_record.timer.loading():
        config = AutoConfig.from_pretrained(str(MODEL_DIR), trust_remote_code=True, local_files_only=True)
    with _test_record.timer.loading():
        model = AutoModel.from_config(config, trust_remote_code=True)
    with _test_record.timer.loading():
        state = load_file(str(MODEL_DIR / 'model.safetensors'))
    with _test_record.timer.loading():
        model.load_state_dict(state, strict=True)
    del state
    with _test_record.timer.loading():
        model = model.to(device).eval()
    class_names = list(model.config.class_names)
    if len(class_names) != NUM_CLASSES:
        raise ValueError(f'模型应有{NUM_CLASSES}类，实际为{len(class_names)}类')
    (records, class_to_id) = load_test_records(class_names)
    builder = TestImageBuilder(TEST_ZIP_PATH)
    print(f'测试图片数量: {len(records)}')
    print(f'类别数量: {NUM_CLASSES}')
    print(f'batch size: {BATCH_SIZE}')
    batch_iterator = iter_batches(records, class_to_id, builder)
    first_batch = next(batch_iterator)
    warmup_values = processor(images=first_batch[0], return_tensors='pt')['pixel_values'].to(device)
    with torch.inference_mode():
        with _test_record.timer.warmup():
            for _ in range(WARMUP_STEPS):
                _test_record.timer.invoke(model, pixel_values=warmup_values)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats(device)
    del warmup_values
    confusion = torch.zeros((NUM_CLASSES, NUM_CLASSES), dtype=torch.long)
    top1_correct = 0
    top3_correct = 0
    loss_sum = 0.0
    forward_seconds = 0.0
    processed = 0
    wall_start = time.perf_counter()
    with torch.inference_mode():
        for (batch_index, (images, labels)) in enumerate(chain([first_batch], batch_iterator), start=1):
            _test_record.before_batch(locals())
            pixel_values = processor(images=images, return_tensors='pt')['pixel_values'].to(device, non_blocking=True)
            targets = labels.to(device, non_blocking=True)
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
            logits = _test_record.timer.invoke(model, pixel_values=pixel_values).logits
            end_event.record()
            torch.cuda.synchronize()
            forward_seconds += start_event.elapsed_time(end_event) / 1000.0
            loss_sum += F.cross_entropy(logits, targets, reduction='sum').item()
            top3 = logits.topk(k=3, dim=1).indices
            predictions = top3[:, 0]
            _test_record.prediction_ready(locals())
            top1_correct += predictions.eq(targets).sum().item()
            top3_correct += top3.eq(targets.unsqueeze(1)).any(dim=1).sum().item()
            confusion += torch.bincount((targets * NUM_CLASSES + predictions).cpu(), minlength=NUM_CLASSES * NUM_CLASSES).reshape(NUM_CLASSES, NUM_CLASSES)
            processed += len(labels)
            if batch_index % 25 == 0:
                print(f'已处理: {processed}/{len(records)}')
            _test_record.capture(locals())
    wall_seconds = time.perf_counter() - wall_start
    builder.close()
    if processed != len(records):
        raise ValueError(f'应处理{len(records)}张，实际处理{processed}张')
    metrics = calculate_metrics(confusion, top1_correct, top3_correct, loss_sum, class_names)
    image_count = len(records)
    official = {'accuracy_pct': 91.4, 'precision_pct': 91.7, 'f1_pct': 91.5, 'source': 'histai/SPIDER-colorectal-model model card'}
    result = {'recorded_at': datetime.now().astimezone().isoformat(timespec='seconds'), 'model_id': MODEL_ID, 'model_path': str(MODEL_DIR), 'script_path': str(Path(__file__).resolve()), 'model': {'architecture': model.__class__.__name__, 'feature_extractor': 'Hibou-L / DINOv2-L/14 with 4 register tokens', 'parameter_count_millions': round(sum((parameter.numel() for parameter in model.parameters())) / 1000000.0, 4), 'num_labels': NUM_CLASSES, 'class_names': class_names, 'weight_loading': {'missing_keys': [], 'unexpected_keys': [], 'mismatched_keys': []}}, 'dataset': {'id': DATASET_ID, 'path': str(DATASET_ROOT), 'test_cache': str(TEST_ZIP_PATH), 'split': 'official slide-level test', 'image_count': image_count, 'class_count': NUM_CLASSES, 'input_patch_count_per_sample': 25, 'central_patch_size': '224x224', 'composite_size': '1120x1120'}, 'preprocessing': {'processor': processor.__class__.__name__, 'do_resize': processor.do_resize, 'do_center_crop': processor.do_center_crop, 'crop_size': processor.crop_size, 'image_mean': processor.image_mean, 'image_std': processor.image_std, 'context_order': 'official 5x5 row-major grid, central patch at 2_2'}, 'environment': {'device': torch.cuda.get_device_name(0), 'precision': 'FP32 with TF32 matrix multiplication enabled', 'batch_size': BATCH_SIZE, 'warmup_steps': WARMUP_STEPS}, 'metrics': metrics, 'official_metrics': official, 'comparison_to_official_percentage_points': {'accuracy': round(metrics['top1_accuracy_pct'] - official['accuracy_pct'], 4), 'macro_precision': round(metrics['macro_precision_pct'] - official['precision_pct'], 4), 'macro_f1': round(metrics['macro_f1_pct'] - official['f1_pct'], 4)}, 'timing': {'model_latency_ms_per_image': round(1000.0 * forward_seconds / image_count, 4), 'model_throughput_images_per_second': round(image_count / forward_seconds, 4), 'end_to_end_latency_ms_per_image': round(1000.0 * wall_seconds / image_count, 4), 'end_to_end_throughput_images_per_second': round(image_count / wall_seconds, 4)}, 'memory': {'peak_allocated_gib': round(torch.cuda.max_memory_allocated(device) / 2 ** 30, 4), 'peak_reserved_gib': round(torch.cuda.max_memory_reserved(device) / 2 ** 30, 4)}}
    save_result(result)
    print('=' * 68)
    print(f"Top-1 accuracy: {metrics['top1_accuracy_pct']:.2f}%")
    print(f"Macro precision: {metrics['macro_precision_pct']:.2f}%")
    print(f"Macro-F1: {metrics['macro_f1_pct']:.2f}%")
    print(f"Cross-entropy loss: {metrics['cross_entropy_loss']:.6f}")
    print(f"模型平均推理时间: {result['timing']['model_latency_ms_per_image']:.4f} ms/image")
    print(f'结果已保存: {RESULTS_PATH}')
    _test_record.finish(locals())
if __name__ == '__main__':
    main()
