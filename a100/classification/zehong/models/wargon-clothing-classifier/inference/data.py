import csv
import json
import time
from datetime import datetime
from pathlib import Path
import numpy as np
import torch
from datasets import load_from_disk
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import AutoImageProcessor
DATASETS = {'clothingdatasetsecondhand': {'root': (Path(__file__).resolve().parents[5] / 'dataset' / 'clothingdataset_clean_test'), 'split': 'test'}}

class _TransformedDataset(torch.utils.data.Dataset):

    def __init__(self, dataset, processor):
        self.dataset = dataset
        self.processor = processor

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        sample = self.dataset[index]
        image = sample['image']
        if image.mode != 'RGB':
            image = image.convert('RGB')
        pixel_values = self.processor(image, return_tensors='pt')['pixel_values'].squeeze(0)
        target_idx = sample.get('target', -1)
        return (pixel_values, int(target_idx), index)

def _load_clean_dataset(dataset_name, max_samples=None):
    if dataset_name not in DATASETS:
        choices = ', '.join(sorted(DATASETS))
        raise ValueError(f'Unknown dataset {dataset_name!r}. Available datasets: {choices}')
    dataset_config = DATASETS[dataset_name]
    root = dataset_config['root']
    if not root.exists():
        raise FileNotFoundError(f'Cleaned dataset path does not exist: {root}')
    dataset = load_from_disk(str(root))
    if max_samples is not None:
        if max_samples <= 0:
            raise ValueError('max_samples must be greater than zero')
        dataset = dataset.select(range(min(max_samples, len(dataset))))
    shard_paths = list(root.glob('*.arrow'))
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

def _label_name(model, index, id2label_mapping=None):
    if id2label_mapping and index in id2label_mapping:
        return id2label_mapping[index]
    id2label = getattr(model.config, 'id2label', {})
    return id2label.get(index, id2label.get(str(index), f'LABEL_{index}'))

def _check_class_mapping(model, dataset):
    num_classes = int(getattr(model.config, 'num_labels', 27))
    unique_targets = set()
    if 'target' in dataset.column_names:
        unique_targets = {t for t in dataset.unique('target') if t is not None}
    mismatches = [t for t in unique_targets if t >= num_classes or t < 0]
    return {'checked': True, 'model_classes_count': num_classes, 'dataset_classes_count': len(unique_targets), 'out_of_range_targets': mismatches}

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

def prepare_inputs(model, dataset_name='clothingdatasetsecondhand', output_dir=None, timing_mode='batched', batch_size=64, num_workers=8, max_samples=None, warmup_steps=3, input_resolution=(3, 224, 224), processor=None):
    """Evaluate a classification model and save per-image logits and timings."""
    if timing_mode not in {'batched', 'single'}:
        raise ValueError("timing_mode must be either 'batched' or 'single'")
    if batch_size <= 0:
        raise ValueError('batch_size must be greater than zero')
    if num_workers < 0:
        raise ValueError('num_workers cannot be negative')
    if warmup_steps < 0:
        raise ValueError('warmup_steps cannot be negative')
    effective_batch_size = 1 if timing_mode == 'single' else batch_size
    dataset, dataset_config, shard_paths = _load_clean_dataset(dataset_name, max_samples=max_samples)
    class_mapping = _check_class_mapping(model, dataset)
    mapping_path = dataset_config['root'] / 'clean_label_mapping.json'
    id2label_mapping = {}
    if mapping_path.exists():
        with open(mapping_path, 'r', encoding='utf-8') as f:
            mapping_data = json.load(f)
            id2label_mapping = {int(k): v for k, v in mapping_data.get('id2label', {}).items()}
    if processor is None:
        model_name_or_path = getattr(model.config, '_name_or_path', 'wargoninnovation/wargon-clothing-classifier')
        processor = AutoImageProcessor.from_pretrained(model_name_or_path)
    transformed_dataset = _TransformedDataset(dataset, processor)
    device = _model_device(model)
    return locals()
