"""在本地 MNIST test split 上评测 fxmarty/resnet-tiny-mnist。"""
if __name__ == '__main__':
    import sys as _records_sys
    from pathlib import Path as _RecordsPath
    _records_sys.path.insert(0, str(_RecordsPath(__file__).resolve().parent))
    _records_sys.path.insert(1, str(_RecordsPath(__file__).resolve().parents[1]))
    from common import start as _start_records
    _test_record = _start_records(159, __file__, globals())
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
from datasets import Dataset as HFDataset, DatasetDict, load_from_disk
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import Normalize, ToTensor
from transformers import AutoModelForImageClassification
MODEL_ID = 'fxmarty/resnet-tiny-mnist'
ROOT = __import__('model_paths').resolve_paths()['project']
MODEL_CACHE_DIR = __import__('model_paths').resolve_paths()['weights']
DATASET_DIR = __import__('model_paths').resolve_paths()['dataset'] / 'mnist'
RESULTS_PATH = _test_record.output / 'metrics.json'
if __name__ == '__main__':
    RESULTS_PATH = _test_record.output / 'metrics.json'
BATCH_SIZE = int(os.environ.get('MODEL_BATCH_SIZE', '256'))
NUM_WORKERS = int(os.environ.get('MODEL_WORKERS', '4'))
WARMUP_STEPS = int(os.environ.get('MODEL_WARMUP', '10'))

def load_test_split():
    if not DATASET_DIR.is_dir():
        raise FileNotFoundError(f'未找到本地 MNIST 数据：{DATASET_DIR}。请先运行 download_test_datasets.py。')
    dataset_info_path = DATASET_DIR / 'dataset_info.json'
    parquet_path = DATASET_DIR / 'test-00000-of-00001.parquet'
    if dataset_info_path.is_file():
        loaded = load_from_disk(str(DATASET_DIR))
    elif parquet_path.is_file():
        loaded = HFDataset.from_parquet(str(parquet_path), cache_dir=str(_test_record.output / '.dataset_cache'))
    else:
        raise FileNotFoundError(f'未找到 MNIST test 数据文件：{parquet_path}')
    if isinstance(loaded, DatasetDict):
        if 'test' not in loaded:
            raise ValueError(f'MNIST 数据集没有 test split，可用 split：{list(loaded)}')
        loaded = loaded['test']
    if 'image' not in loaded.column_names or 'label' not in loaded.column_names:
        raise ValueError(f'MNIST 数据列应包含 image 和 label，实际为：{loaded.column_names}')
    return loaded

class MnistResNetDataset(Dataset):

    def __init__(self, dataset):
        self.dataset = dataset
        self.transform = Normalize(mean=[0.45], std=[0.22])

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        example = self.dataset[index]
        image = example['image'].convert('L')
        pixel_values = self.transform(ToTensor()(image))
        return (pixel_values, int(example['label']))

def save_result(result):
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = RESULTS_PATH.with_suffix('.json.tmp')
    temporary_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    os.replace(temporary_path, RESULTS_PATH)

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {(torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU')}")
    print(f'模型: {MODEL_ID}')
    print('数据集: 本地 MNIST test split')
    with _test_record.timer.loading():
        model = AutoModelForImageClassification.from_pretrained(str(MODEL_CACHE_DIR), local_files_only=True).to(device)
    with _test_record.timer.loading():
        model.eval()
    if model.config.num_labels != 10:
        raise ValueError(f'模型分类头应输出 10 类，实际为 {model.config.num_labels} 类')
    parameter_count_millions = sum((parameter.numel() for parameter in model.parameters())) / 1000000.0
    dataset = load_test_split()
    wrapped_dataset = MnistResNetDataset(dataset)
    dataloader = _test_record.loader(DataLoader, wrapped_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=device.type == 'cuda', persistent_workers=NUM_WORKERS > 0)
    print(f'测试图片数量: {len(wrapped_dataset)}')
    print(f'batch size: {BATCH_SIZE}')
    print('预处理: 灰度单通道、28x28、Normalize(mean=0.45, std=0.22)')
    (warmup_pixel_values, _) = next(iter(dataloader))
    warmup_pixel_values = warmup_pixel_values[:BATCH_SIZE].to(device, non_blocking=True)
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
    correct_top1 = 0
    correct_top5 = 0
    forward_seconds = 0.0
    wall_start = time.perf_counter()
    with torch.inference_mode():
        for (batch_index, (pixel_values, labels)) in enumerate(dataloader, start=1):
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
            top5 = logits.topk(k=5, dim=1).indices
            _test_record.prediction_ready(locals())
            correct_top1 += top5[:, 0].eq(labels).sum().item()
            correct_top5 += top5.eq(labels.unsqueeze(1)).any(dim=1).sum().item()
            if batch_index % 25 == 0:
                print(f'已处理: {min(batch_index * BATCH_SIZE, len(wrapped_dataset))}/{len(wrapped_dataset)}')
            _test_record.capture(locals())
    wall_seconds = time.perf_counter() - wall_start
    image_count = len(wrapped_dataset)
    top1_pct = correct_top1 / image_count * 100.0
    top5_pct = correct_top5 / image_count * 100.0
    result = {'recorded_at': datetime.now().astimezone().isoformat(timespec='seconds'), 'model_id': MODEL_ID, 'model_path': str(MODEL_CACHE_DIR), 'script_path': str(Path(__file__).resolve()), 'model': {'id': MODEL_ID, 'parameter_count_millions': round(parameter_count_millions, 4), 'num_labels': 10}, 'dataset': {'name': 'MNIST', 'split': 'test', 'path': str(DATASET_DIR), 'image_count': image_count, 'class_count': 10}, 'environment': {'device': torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU', 'precision': 'FP32', 'batch_size': BATCH_SIZE, 'warmup_steps': WARMUP_STEPS, 'num_workers': NUM_WORKERS}, 'preprocessing': {'mode': 'L', 'resize': False, 'normalize_mean': 0.45, 'normalize_std': 0.22}, 'accuracy': {'image_count': image_count, 'top1_pct': round(top1_pct, 4), 'top5_pct': round(top5_pct, 4)}, 'timing': {'model_latency_ms_per_image': round(forward_seconds / image_count * 1000.0, 4), 'model_throughput_images_per_second': round(image_count / forward_seconds, 4), 'end_to_end_latency_ms_per_image': round(wall_seconds / image_count * 1000.0, 4), 'end_to_end_throughput_images_per_second': round(image_count / wall_seconds, 4)}}
    if device.type == 'cuda':
        result['memory'] = {'peak_allocated_gib': round(torch.cuda.max_memory_allocated(device) / 2 ** 30, 4), 'peak_reserved_gib': round(torch.cuda.max_memory_reserved(device) / 2 ** 30, 4)}
    save_result(result)
    print('=' * 60)
    print(f'Top-1 accuracy: {top1_pct:.2f}%')
    print(f'Top-5 accuracy: {top5_pct:.2f}%')
    print(f"模型平均推理时间: {result['timing']['model_latency_ms_per_image']:.4f} ms/image")
    print(f"模型吞吐量: {result['timing']['model_throughput_images_per_second']:.2f} images/s")
    print(f'评测结果已更新: {RESULTS_PATH}')
    _test_record.finish(locals())
if __name__ == '__main__':
    main()
