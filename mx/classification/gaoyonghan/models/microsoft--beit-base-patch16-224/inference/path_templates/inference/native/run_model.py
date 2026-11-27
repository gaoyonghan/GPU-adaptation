if __name__ == '__main__':
    import sys as _records_sys
    from pathlib import Path as _RecordsPath
    _records_sys.path.insert(0, str(_RecordsPath(__file__).resolve().parent))
    _records_sys.path.insert(1, str(_RecordsPath(__file__).resolve().parents[1]))
    from common import start as _start_records
    _test_record = _start_records(103, __file__, globals())
import os
import sys
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['PYTHONUNBUFFERED'] = '1'
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(line_buffering=True, write_through=True)
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(line_buffering=True, write_through=True)
import csv
import json
from dataclasses import asdict, is_dataclass
import time
from datetime import datetime
from pathlib import Path
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from transformers import BeitForImageClassification, BeitImageProcessor
ROOT = __import__('model_paths').resolve_paths()['project']
MODEL_CACHE_DIR = str(__import__('model_paths').resolve_paths()['weights'])
DATASET_DIR = __import__('model_paths').resolve_paths()['dataset'] / 'ImageNet-1K'
VAL_DIR = DATASET_DIR / 'val'
VAL_LABELS_PATH = DATASET_DIR / 'LOC_val_solution.csv'
SYNSET_MAPPING_PATH = DATASET_DIR / 'LOC_synset_mapping.txt'
RESULTS_PATH = _test_record.output / 'metrics.json'
if __name__ == '__main__':
    RESULTS_PATH = _test_record.output / 'metrics.json'
MODEL_ID = 'microsoft/beit-base-patch16-224'
BATCH_SIZE = int(os.environ.get('MODEL_BATCH_SIZE', '64'))
NUM_WORKERS = int(os.environ.get('MODEL_WORKERS', '8'))
WARMUP_STEPS = int(os.environ.get('MODEL_WARMUP', '10'))
os.makedirs(MODEL_CACHE_DIR, exist_ok=True)
os.environ['HF_HOME'] = MODEL_CACHE_DIR
os.environ['HF_HUB_CACHE'] = MODEL_CACHE_DIR
os.environ['TRANSFORMERS_CACHE'] = MODEL_CACHE_DIR

def json_metadata_default(value):
    """Serialize library metadata without changing inference or preprocessing."""
    if isinstance(value, (set, frozenset)):
        return sorted(value)
    if is_dataclass(value) and (not isinstance(value, type)):
        return {key: item for (key, item) in asdict(value).items() if item is not None}
    raise TypeError(f'Object of type {type(value).__name__} is not JSON serializable')

class ImageNetValidationDataset(Dataset):

    def __init__(self, processor):
        with SYNSET_MAPPING_PATH.open(encoding='utf-8') as f:
            synsets = [line.split(maxsplit=1)[0] for line in f if line.strip()]
        synset_to_index = {synset: index for (index, synset) in enumerate(synsets)}
        with VAL_LABELS_PATH.open(newline='', encoding='utf-8') as f:
            rows = csv.DictReader(f)
            labels_by_image_id = {row['ImageId']: synset_to_index[row['PredictionString'].split()[0]] for row in rows}
        self.image_paths = sorted(VAL_DIR.glob('*.JPEG'))
        self.labels = [labels_by_image_id[path.stem] for path in self.image_paths]
        self.processor = processor
        if len(synsets) != 1000:
            raise ValueError(f'类别映射应包含 1000 类，实际为 {len(synsets)} 类')
        if len(self.image_paths) != 50000:
            raise ValueError(f'ImageNet-1K val 应包含 50000 张图片，实际为 {len(self.image_paths)} 张')

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        with Image.open(self.image_paths[index]) as image:
            pixel_values = self.processor(images=image.convert('RGB'), return_tensors='pt')['pixel_values'][0]
        return (pixel_values, self.labels[index])
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'使用设备: {device}')
print(f'模型: {MODEL_ID}')
with _test_record.timer.loading():
    processor = BeitImageProcessor.from_pretrained(MODEL_ID, cache_dir=MODEL_CACHE_DIR)
with _test_record.timer.loading():
    model = BeitForImageClassification.from_pretrained(MODEL_ID, cache_dir=MODEL_CACHE_DIR, use_safetensors=True).to(device)
with _test_record.timer.loading():
    model.eval()
dataset = ImageNetValidationDataset(processor)
if model.config.num_labels != 1000:
    raise ValueError(f'模型分类头应输出 1000 类，实际为 {model.config.num_labels} 类')
dataloader = _test_record.loader(DataLoader, dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=device == 'cuda', persistent_workers=NUM_WORKERS > 0)
print(f'验证图片数量: {len(dataset)}')
print(f'batch size: {BATCH_SIZE}')
print(f'图像处理器: {type(processor).__name__}')
with Image.open(dataset.image_paths[0]) as image:
    warmup_inputs = processor(images=[image.convert('RGB')] * BATCH_SIZE, return_tensors='pt').to(device)
if device == 'cuda':
    torch.backends.cudnn.benchmark = True
with torch.inference_mode():
    with _test_record.timer.warmup():
        for _ in range(WARMUP_STEPS):
            _test_record.timer.invoke(model, **warmup_inputs)
if device == 'cuda':
    torch.cuda.synchronize()
del warmup_inputs
correct_top1 = torch.zeros((), dtype=torch.long, device=device)
correct_top5 = torch.zeros((), dtype=torch.long, device=device)
gpu_events = []
cpu_forward_seconds = 0.0
wall_start = time.perf_counter()
with torch.inference_mode():
    for (batch_index, (pixel_values, labels)) in enumerate(dataloader, start=1):
        _test_record.before_batch(locals())
        pixel_values = pixel_values.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        if device == 'cuda':
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
            logits = _test_record.timer.invoke(model, pixel_values=pixel_values).logits
            end_event.record()
            gpu_events.append((start_event, end_event))
        else:
            forward_start = time.perf_counter()
            logits = _test_record.timer.invoke(model, pixel_values=pixel_values).logits
            cpu_forward_seconds += time.perf_counter() - forward_start
        top5_indices = logits.topk(k=5, dim=1).indices
        _test_record.prediction_ready(locals())
        matches = top5_indices.eq(labels.unsqueeze(1))
        correct_top1 += matches[:, :1].sum()
        correct_top5 += matches.any(dim=1).sum()
        if batch_index % 100 == 0:
            processed = min(batch_index * BATCH_SIZE, len(dataset))
            print(f'已处理: {processed}/{len(dataset)}')
        _test_record.capture(locals())
if device == 'cuda':
    torch.cuda.synchronize()
wall_seconds = time.perf_counter() - wall_start
if device == 'cuda':
    forward_seconds = sum((start.elapsed_time(end) for (start, end) in gpu_events)) / 1000
else:
    forward_seconds = cpu_forward_seconds
image_count = len(dataset)
top1_accuracy = correct_top1.item() / image_count * 100
top5_accuracy = correct_top5.item() / image_count * 100
average_forward_ms = forward_seconds / image_count * 1000
model_fps = image_count / forward_seconds
end_to_end_ms = wall_seconds / image_count * 1000
end_to_end_fps = image_count / wall_seconds
print('\n' + '=' * 60)
print('ImageNet-1K val 上的 BEiT-Base 评测结果')
print('=' * 60)
print(f'Top-1 accuracy: {top1_accuracy:.2f}%')
print(f'Top-5 accuracy: {top5_accuracy:.2f}%')
print(f'纯模型平均推理时间: {average_forward_ms:.3f} ms/image')
print(f'纯模型吞吐量: {model_fps:.2f} images/s')
print(f'端到端平均时间: {end_to_end_ms:.3f} ms/image')
print(f'端到端吞吐量: {end_to_end_fps:.2f} images/s')
result = {'recorded_at': datetime.now().astimezone().isoformat(timespec='seconds'), 'model_id': MODEL_ID, 'model_path': MODEL_CACHE_DIR, 'script_path': str(Path(__file__).resolve()), 'dataset': {'name': 'ImageNet-1K val', 'path': str(DATASET_DIR), 'image_count': image_count}, 'environment': {'device': torch.cuda.get_device_name(torch.cuda.current_device()) if device == 'cuda' else 'CPU', 'precision': 'FP32', 'batch_size': BATCH_SIZE, 'warmup_steps': WARMUP_STEPS, 'num_workers': NUM_WORKERS}, 'preprocessing': processor.to_dict(), 'accuracy': {'image_count': image_count, 'top1_pct': round(top1_accuracy, 4), 'top5_pct': round(top5_accuracy, 4)}, 'timing': {'model_latency_ms_per_image': round(average_forward_ms, 4), 'model_throughput_images_per_second': round(model_fps, 4), 'end_to_end_latency_ms_per_image': round(end_to_end_ms, 4), 'end_to_end_throughput_images_per_second': round(end_to_end_fps, 4)}}
RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
temporary_results_path = RESULTS_PATH.with_suffix('.json.tmp')
with temporary_results_path.open('w', encoding='utf-8') as f:
    json.dump(result, f, default=json_metadata_default, ensure_ascii=False, indent=2)
    f.write('\n')
os.replace(temporary_results_path, RESULTS_PATH)
print(f'评测结果已更新: {RESULTS_PATH}')
_test_record.finish(globals())
