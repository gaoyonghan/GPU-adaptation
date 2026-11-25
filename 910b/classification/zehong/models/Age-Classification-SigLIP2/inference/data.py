import csv
import json
import re
import time
from datetime import datetime
from pathlib import Path
import numpy as np
import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
DATASETS = {'Age-Classification-Set': {'root': (Path(__file__).resolve().parents[5] / 'dataset' / 'Age-Classification-Set'), 'split': 'train'}}

class _TransformedDataset(torch.utils.data.Dataset):

    def __init__(self, dataset, transform):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        sample = self.dataset[index]
        image = sample['image'].convert('RGB')
        return (self.transform(image), int(sample['label']), index)

class _ImageProcessorTransform:

    def __init__(self, image_processor):
        self.image_processor = image_processor

    def __call__(self, image):
        processed = self.image_processor(images=image, return_tensors='pt')
        return processed['pixel_values'][0]

def _load_parquet_split(dataset_name, max_samples=None):
    dataset_config = DATASETS[dataset_name]
    root = dataset_config['root']
    split = dataset_config['split']
    shard_paths = sorted(root.glob('*.parquet'))
    if not shard_paths:
        raise FileNotFoundError(f'No Parquet files found under {root}')
    dataset = load_dataset('parquet', data_files={split: [str(path) for path in shard_paths]}, split=split)
    if max_samples is not None:
        if max_samples <= 0:
            raise ValueError('max_samples must be greater than zero')
        dataset = dataset.select(range(min(max_samples, len(dataset))))
    return (dataset, dataset_config, shard_paths)

def _model_device(model):
    return next(model.parameters()).device

def _label_name(model, index):
    id2label = getattr(getattr(model, 'config', None), 'id2label', {})
    return id2label.get(index, id2label.get(str(index), str(index)))

def _age_range(name):
    match = re.search('\\d+\\s*-\\s*\\d+|\\d+\\s*\\+', name)
    return re.sub('\\s+', '', match.group()) if match else name

def _check_class_mapping(model, dataset):
    dataset_names = getattr(dataset.features.get('label'), 'names', None)
    if not dataset_names or not getattr(getattr(model, 'config', None), 'id2label', None):
        return {'checked': False, 'mismatch_count': None}
    mismatches = [{'index': i, 'dataset_label': name, 'model_label': _label_name(model, i)} for i, name in enumerate(dataset_names) if _age_range(name) != _age_range(_label_name(model, i))]
    return {'checked': True, 'mismatch_count': len(mismatches), 'first_mismatches': mismatches[:10]}

def prepare_inputs(model, dataset_name='Age-Classification-Set', output_dir=None, timing_mode='batched', batch_size=64, num_workers=8, max_samples=None, warmup_steps=3, input_resolution=None, image_processor=None, crop_pct=None, interpolation=None):
    """Prepare native dataset and transform only; inference and offline scoring are separate."""
    if timing_mode not in {'batched', 'single'}:
        raise ValueError("timing_mode must be either 'batched' or 'single'")
    if batch_size <= 0 or num_workers < 0 or warmup_steps < 0:
        raise ValueError('Invalid batch_size, num_workers or warmup_steps')
    effective_batch_size = 1 if timing_mode == 'single' else batch_size
    dataset, dataset_config, shard_paths = _load_parquet_split(dataset_name, max_samples)
    class_mapping = _check_class_mapping(model, dataset)
    if class_mapping.get('mismatch_count'):
        raise ValueError(f'Class indices do not match: {class_mapping}')
    model_config = getattr(model, 'config', None)
    effective_input_resolution = input_resolution or (3, 224, 224)
    preprocessing = {'type': 'huggingface' if image_processor is not None else 'timm'}
    if image_processor is not None:
        transform = _ImageProcessorTransform(image_processor)
        processor_size = getattr(image_processor, 'size', {})
        if 'height' in processor_size and 'width' in processor_size:
            effective_input_resolution = (3, processor_size['height'], processor_size['width'])
        preprocessing['processor_class'] = type(image_processor).__name__
        preprocessing['processor_config'] = image_processor.to_dict()
    else:
        from timm.data import resolve_model_data_config
        from timm.data.transforms_factory import create_transform
        data_config = resolve_model_data_config(model) if hasattr(model, 'pretrained_cfg') else {}
        effective_input_resolution = input_resolution or data_config.get('input_size', (3, 224, 224))
        resolved_crop_pct = crop_pct if crop_pct is not None else data_config.get('crop_pct', getattr(model_config, 'crop_pct', 0.875))
        resolved_interpolation = interpolation or data_config.get('interpolation', getattr(model_config, 'interpolation', 'bilinear'))
        resolved_crop_mode = data_config.get('crop_mode', getattr(model_config, 'crop_mode', 'center'))
        transform = create_transform(input_size=effective_input_resolution, is_training=False, mean=data_config.get('mean', getattr(model_config, 'mean', (0.485, 0.456, 0.406))), std=data_config.get('std', getattr(model_config, 'std', (0.229, 0.224, 0.225))), crop_mode=resolved_crop_mode, crop_pct=resolved_crop_pct, interpolation=resolved_interpolation)
        preprocessing.update(crop_mode=resolved_crop_mode, crop_pct=resolved_crop_pct, interpolation=resolved_interpolation)
    device = _model_device(model)
    transformed_dataset = _TransformedDataset(dataset, transform)
    return locals()
