import csv
import json
import time
from datetime import datetime
from pathlib import Path
import numpy as np
import torch
from datasets import Dataset, concatenate_datasets
from timm.data.transforms_factory import create_transform
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
DATASETS = {'rsscn7': {'root': (Path(__file__).resolve().parents[5] / 'dataset' / 'jonathan-roberts1___rsscn7/split_data'), 'split': 'test'}}

class _TransformedDataset(torch.utils.data.Dataset):

    def __init__(self, dataset, transform):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        sample = self.dataset[index]
        image = sample['image']
        if image.mode != 'RGB':
            image = image.convert('RGB')
        return (self.transform(image), int(sample['label']), index)

def _load_arrow_split(dataset_name, max_samples=None):
    if dataset_name not in DATASETS:
        choices = ', '.join(sorted(DATASETS))
        raise ValueError(f'Unknown dataset {dataset_name!r}. Available datasets: {choices}')
    dataset_config = DATASETS[dataset_name]
    root = dataset_config['root']
    split = dataset_config['split']
    shard_pattern = f'{split}/*.arrow'
    shard_paths = sorted(root.glob(shard_pattern))
    if not shard_paths:
        shard_paths = sorted(root.glob(f'**/*{split}*.arrow'))
    if not shard_paths:
        raise FileNotFoundError(f'No Arrow shards found for {dataset_name}/{split} under {root}')
    shards = [Dataset.from_file(str(path)) for path in shard_paths]
    dataset = concatenate_datasets(shards)
    if max_samples is not None:
        if max_samples <= 0:
            raise ValueError('max_samples must be greater than zero')
        dataset = dataset.select(range(min(max_samples, len(dataset))))
    return (dataset, dataset_config, shard_paths)

def _model_device(model):
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device('cpu')

def _extract_logits(outputs):
    if isinstance(outputs, dict):
        return outputs['logits']
    if hasattr(outputs, 'logits'):
        return outputs.logits
    raise TypeError('Model output does not contain logits')

def _label_name(model, index):
    id2label = getattr(model.config, 'id2label', {})
    return id2label.get(index, id2label.get(str(index), str(index)))

def _check_class_mapping(model, dataset):
    label_feature = dataset.features.get('label')
    dataset_names = getattr(label_feature, 'names', None)
    if not dataset_names:
        return {'checked': False, 'mismatch_count': None}
    mismatches = []
    for index, dataset_name in enumerate(dataset_names):
        model_name = _label_name(model, index)
        if model_name != dataset_name:
            mismatches.append({'index': index, 'dataset_label': dataset_name, 'model_label': model_name})
    return {'checked': True, 'mismatch_count': len(mismatches), 'first_mismatches': mismatches[:10]}

def _time_forward(model, images, device):
    if device.type == 'cuda':
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        outputs = model(images)
        end.record()
        end.synchronize()
        elapsed_seconds = start.elapsed_time(end) / 1000.0
    else:
        start_time = time.perf_counter()
        outputs = model(images)
        elapsed_seconds = time.perf_counter() - start_time
    return (outputs, elapsed_seconds)

def prepare_inputs(model, dataset_name='rsscn7', output_dir=None, timing_mode='batched', batch_size=64, num_workers=8, max_samples=None, warmup_steps=3, input_resolution=(3, 224, 224)):
    """Evaluate a classification model and save per-image logits and timings.

    timing_mode="batched" records batch latency divided by its sample count.
    timing_mode="single" forces batch size 1 and records true per-image latency.
    """
    if timing_mode not in {'batched', 'single'}:
        raise ValueError("timing_mode must be either 'batched' or 'single'")
    if batch_size <= 0:
        raise ValueError('batch_size must be greater than zero')
    if num_workers < 0:
        raise ValueError('num_workers cannot be negative')
    if warmup_steps < 0:
        raise ValueError('warmup_steps cannot be negative')
    effective_batch_size = 1 if timing_mode == 'single' else batch_size
    dataset, dataset_config, shard_paths = _load_arrow_split(dataset_name, max_samples=max_samples)
    class_mapping = _check_class_mapping(model, dataset)
    if class_mapping.get('mismatch_count'):
        print(f"Warning: model and dataset class names differ at {class_mapping['mismatch_count']} indices.")
    mean = getattr(model.config, 'image_mean', getattr(model.config, 'mean', (0.5, 0.5, 0.5)))
    std = getattr(model.config, 'image_std', getattr(model.config, 'std', (0.5, 0.5, 0.5)))
    crop_mode = getattr(model.config, 'crop_mode', None)
    crop_pct = getattr(model.config, 'crop_pct', None)
    transform = create_transform(input_size=input_resolution, is_training=False, mean=mean, std=std, crop_mode=crop_mode, crop_pct=crop_pct)
    transformed_dataset = _TransformedDataset(dataset, transform)
    device = _model_device(model)
    return locals()
