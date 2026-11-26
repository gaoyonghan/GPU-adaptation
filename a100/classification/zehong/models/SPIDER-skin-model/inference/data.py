"""
utils/evaluate_model.py —— SPIDER-breast 昇腾 NPU 适配版 (对齐 MambaVision 评测口径)
"""
import os
os.environ['HF_HUB_OFFLINE'] = '1'

import csv
import json
import time
from datetime import datetime
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
PATCH_SIZE = 224
FULL_GRID_SIZE = 5
CENTER_INDEX = FULL_GRID_SIZE // 2
_THIS_DIR = Path(__file__).resolve().parent
MODEL_PROJECT_DIR = _THIS_DIR.parent
DATASETS = {'spider-breast': {'root': (Path(__file__).resolve().parents[5] / 'dataset' / 'SPIDER-breast/SPIDER-breast'), 'model_id': 'histai/SPIDER-breast-model', 'splits_dir': MODEL_PROJECT_DIR / 'data_splits' / 'spider-breast', 'default_split': 'test'}, 'spider-skin': {'root': (Path(__file__).resolve().parents[5] / 'dataset' / 'SPIDER-skin/SPIDER-skin'), 'model_id': 'histai/SPIDER-skin-model', 'splits_dir': MODEL_PROJECT_DIR / 'data_splits' / 'spider-skin', 'default_split': 'test'}, 'spider-thorax': {'root': (Path(__file__).resolve().parents[5] / 'dataset' / 'SPIDER-thorax/SPIDER-thorax'), 'model_id': 'histai/SPIDER-thorax-model', 'splits_dir': MODEL_PROJECT_DIR / 'data_splits' / 'spider-thorax', 'default_split': 'test'}}
OFFICIAL_METRICS = {'spider-breast': {'accuracy': 0.902, 'precision': 0.896, 'f1': 0.897, 'note': '来自 histai/SPIDER-breast-model 模型卡 Results / 论文 Table 2 (test set)。'}, 'spider-skin': {'accuracy': 0.94, 'precision': 0.936, 'f1': 0.938, 'note': '来自 histai/SPIDER-skin-model 模型卡 Results (test set, context=1120x1120)。'}, 'spider-thorax': {'accuracy': 0.962, 'precision': 0.958, 'f1': 0.96, 'note': '来自 histai/SPIDER-thorax-model 模型卡 / 论文 Table 2 (test set)。'}}

class SPIDERStitchedDataset(Dataset):

    def __init__(self, records, images_dir, context_size=5):
        if context_size not in (1, 3, 5):
            raise ValueError('context_size 只能是 1/3/5')
        self.records = records
        self.images_dir = Path(images_dir)
        self.context_size = context_size
        self.patch_size = PATCH_SIZE
        self.full_grid_size = FULL_GRID_SIZE
        self.center_index = CENTER_INDEX
        lower_bound = (self.full_grid_size - self.context_size) // 2
        self.lower_bound = lower_bound
        self.upper_bound = lower_bound + self.context_size - 1

    def __len__(self):
        return len(self.records)

    @staticmethod
    def _stitch_record(record, images_dir, context_size, patch_size, full_grid_size, center_index, lower_bound, upper_bound):
        stitched_size = context_size * patch_size
        stitched = Image.new('RGB', (stitched_size, stitched_size))
        for i in range(lower_bound, upper_bound + 1):
            for j in range(lower_bound, upper_bound + 1):
                if i == center_index and j == center_index:
                    patch_filename = record['image_name']
                else:
                    key = f'{i}_{j}'
                    patch_filename = record['context_info'].get(key)
                    if patch_filename is None:
                        continue
                patch_path = images_dir / patch_filename
                try:
                    patch = Image.open(patch_path).convert('RGB')
                except Exception:
                    raise
                paste_x = (j - lower_bound) * patch_size
                paste_y = (i - lower_bound) * patch_size
                stitched.paste(patch, (paste_x, paste_y))
        if context_size == 1:
            return Image.open(images_dir / record['image_name']).convert('RGB')
        return stitched

    def __getitem__(self, index):
        record = self.records[index]
        image = self._stitch_record(record, self.images_dir, self.context_size, self.patch_size, self.full_grid_size, self.center_index, self.lower_bound, self.upper_bound)
        return (image, index)

def load_records_for_split(dataset_name, split):
    if dataset_name not in DATASETS:
        choices = ', '.join(sorted(DATASETS))
        raise ValueError(f'未知数据集 {dataset_name!r}，可选: {choices}')
    cfg = DATASETS[dataset_name]
    root = Path(cfg['root'])
    manifest = Path(cfg['splits_dir']) / f'{split}.json'
    source = None
    if manifest.with_suffix('.json.gz').exists():
        import gzip
        with gzip.open(manifest.with_suffix('.json.gz'),'rt',encoding='utf-8') as f: records=json.load(f)
        return records, 'manifest:'+str(manifest.with_suffix('.json.gz'))
    if manifest.exists():
        with open(manifest, 'r', encoding='utf-8') as f:
            records = json.load(f)
        source = f'manifest:{manifest}'
    else:
        metadata_path = root / 'metadata.json'
        if not metadata_path.exists():
            raise FileNotFoundError(f'metadata.json 不存在: {metadata_path}')
        with open(metadata_path, 'r', encoding='utf-8') as f:
            all_records = json.load(f)
        records = [r for r in all_records if r.get('split', 'train') == split]
        source = f'metadata:{metadata_path} (按 split 字段={split})'
    if not records:
        raise ValueError(f'数据集 {dataset_name} 在 split={split} 下没有记录（{source}）')
    return (records, source)

def get_class_names(model):
    class_names = getattr(getattr(model, 'config', None), 'class_names', None)
    if class_names:
        return list(class_names)
    raise ValueError('模型 config 中没有 class_names，无法确定类别顺序。')

def _label_name(class_names, index):
    return class_names[index] if 0 <= index < len(class_names) else str(index)

def build_processor(model_id):
    from transformers import ViTImageProcessor
    return ViTImageProcessor.from_pretrained(model_id, trust_remote_code=True)

def make_collate_fn(processor):

    def collate_fn(batch):
        images = [item[0] for item in batch]
        indices = torch.tensor([item[1] for item in batch], dtype=torch.long)
        encoded = processor(images=images, return_tensors='pt')
        return (encoded['pixel_values'], indices)
    return collate_fn

def _model_device(model):
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device('cpu')

def _sync_device(device):
    if device.type in {'npu', 'cuda'}:
        getattr(torch, device.type).synchronize(device)

def _reset_peak_memory(device):
    if device.type in {'npu', 'cuda'}:
        getattr(torch, device.type).reset_peak_memory_stats(device)

def _time_forward(model, pixel_values, device):
    """同时记录 NPU 事件纯耗时与同步墙钟时间 (对齐 MambaVision 风格)"""
    if device.type in {'npu', 'cuda'}:
        backend = getattr(torch, device.type)
        backend.synchronize(device)
        start = backend.Event(enable_timing=True)
        end = backend.Event(enable_timing=True)
        wall_start = time.perf_counter()
        start.record()
        outputs = model(pixel_values)
        end.record()
        end.synchronize()
        wall_seconds = time.perf_counter() - wall_start
        elapsed_seconds = start.elapsed_time(end) / 1000.0
    else:
        start_time = time.perf_counter()
        outputs = model(pixel_values)
        elapsed_seconds = time.perf_counter() - start_time
        wall_seconds = elapsed_seconds
    return (outputs, elapsed_seconds, wall_seconds)

def prepare_inputs(model, dataset_name='spider-breast', split=None, context_size=5, output_dir=None, timing_mode='batched', batch_size=8, num_workers=4, max_samples=None, warmup_steps=3, input_resolution=(3, 1120, 1120), processor=None):
    if dataset_name not in DATASETS:
        raise ValueError(f'未知数据集 {dataset_name!r}，可选: {sorted(DATASETS)}')
    if timing_mode not in {'batched', 'single'}:
        raise ValueError("timing_mode 必须是 'batched' 或 'single'")
    cfg = DATASETS[dataset_name]
    split = split or cfg.get('default_split', 'test')
    effective_batch_size = 1 if timing_mode == 'single' else batch_size
    class_names = get_class_names(model)
    num_classes = len(class_names)
    print(f'[evaluate_model] 数据集={dataset_name} split={split} context_size={context_size} 类别数={num_classes}')
    if processor is None:
        processor = build_processor(cfg['model_id'])
    records, source = load_records_for_split(dataset_name, split)
    if max_samples is not None and max_samples > 0:
        records = records[:min(max_samples, len(records))]
    num_samples = len(records)
    if num_samples == 0:
        raise ValueError('待评估记录数为 0')
    labels = []
    for rec in records:
        try:
            labels.append(class_names.index(rec['class']))
        except ValueError as e:
            raise ValueError(f"记录类别 {rec['class']!r} 不在模型类别列表中: {e}") from e
    labels = np.asarray(labels, dtype=np.int64)
    images_dir = Path(cfg['root']) / 'images'
    dataset = SPIDERStitchedDataset(records, images_dir, context_size=context_size)
    collate_fn = make_collate_fn(processor)
    device = _model_device(model)
    return locals()
