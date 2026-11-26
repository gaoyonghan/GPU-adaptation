import csv
import json
import time
from datetime import datetime
from pathlib import Path
import numpy as np
import torch
from datasets import Dataset, concatenate_datasets
from timm.data import resolve_model_data_config
from timm.data.transforms_factory import create_transform
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
DATASETS = {'imagenet-1k': {'root': (Path(__file__).resolve().parents[5] / 'dataset' / 'ILSVRC___imagenet-1k'), 'split': 'validation'}}

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

class _ImageProcessorTransform:

    def __init__(self, image_processor):
        self.image_processor = image_processor

    def __call__(self, image):
        processed = self.image_processor(images=image, return_tensors='pt')
        pixel_values = processed['pixel_values']
        if pixel_values.ndim != 4 or pixel_values.shape[0] != 1:
            raise ValueError(f'Image processor must return pixel_values with shape (1, C, H, W), got {tuple(pixel_values.shape)}')
        return pixel_values[0]

def _load_arrow_split(dataset_name, max_samples=None):
    if dataset_name not in DATASETS:
        choices = ', '.join(sorted(DATASETS))
        raise ValueError(f'Unknown dataset {dataset_name!r}. Available datasets: {choices}')
    dataset_config = DATASETS[dataset_name]
    root = dataset_config['root']
    split = dataset_config['split']
    shard_pattern = f'**/imagenet-1k-{split}-*.arrow'
    shard_paths = sorted(root.glob(shard_pattern))
    if not shard_paths:
        raise FileNotFoundError(f'No Arrow shards found for {dataset_name}/{split} under {root}')
    shards = [Dataset.from_file(str(path)) for path in shard_paths]
    dataset = concatenate_datasets(shards)
    if max_samples is not None:
        if max_samples <= 0:
            raise ValueError('max_samples must be greater than zero')
        dataset = dataset.select(range(min(max_samples, len(dataset))))
    # Keep all 21843 classifier outputs; map independent 1K GT by official timm WordNet synsets.
    from timm.data import ImageNetInfo
    from datasets import ClassLabel
    source_info=ImageNetInfo('imagenet-1k');target_info=ImageNetInfo('imagenet-21k-goog')
    source_names=source_info.label_names();target_names=target_info.label_names()
    assert len(source_names)==1000 and len(target_names)==21843
    target_index={name:i for i,name in enumerate(target_names)}
    unsupported=[name for name in source_names if name not in target_index]
    # GT-only categories are never model outputs; keep these images and count them as errors.
    evaluation_names=target_names+unsupported
    mapping=[evaluation_names.index(name) for name in source_names]
    assert len(set(mapping))==1000
    if not isinstance(dataset.features['label'],ClassLabel) or dataset.features['label'].num_classes!=1000:
        raise ValueError('Expected standard ImageNet-1K ClassLabel order')
    labels=[mapping[int(x)] for x in dataset['label']]
    dataset=dataset.remove_columns('label').add_column('label',labels).cast_column('label',ClassLabel(names=evaluation_names))
    return (dataset, dataset_config, shard_paths)

def _model_device(model):
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device('cpu')

def _label_name(model, index):
    if hasattr(model, 'config'):
        id2label = getattr(model.config, 'id2label', {})
        return id2label.get(index, id2label.get(str(index), str(index)))
    return str(index)

def _check_class_mapping(model, dataset):
    if not getattr(getattr(model, 'config', None), 'id2label', None):
        return {'checked': False, 'mismatch_count': None}
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

def prepare_inputs(model, dataset_name='imagenet-1k', output_dir=None, timing_mode='batched', batch_size=64, num_workers=8, max_samples=None, warmup_steps=3, input_resolution=None, image_processor=None, crop_pct=None, interpolation=None):
    """Prepare native dataset and transform only; inference and offline scoring are separate."""
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
    model_config = getattr(model, 'config', None)
    timm_data_config = resolve_model_data_config(model) if hasattr(model, 'pretrained_cfg') else {}
    effective_input_resolution = input_resolution or timm_data_config.get('input_size', (3, 224, 224))
    preprocessing = {'type': 'huggingface' if image_processor is not None else 'timm'}
    if image_processor is not None:
        transform = _ImageProcessorTransform(image_processor)
        processor_size = getattr(image_processor, 'size', None)
        if getattr(image_processor, 'do_resize', False) and isinstance(processor_size, dict) and ('height' in processor_size) and ('width' in processor_size):
            effective_input_resolution = (effective_input_resolution[0], processor_size['height'], processor_size['width'])
        preprocessing['processor_class'] = type(image_processor).__name__
    else:
        resolved_crop_pct = crop_pct if crop_pct is not None else timm_data_config.get('crop_pct', getattr(model_config, 'crop_pct', 0.875))
        resolved_interpolation = interpolation if interpolation is not None else timm_data_config.get('interpolation', getattr(model_config, 'interpolation', 'bilinear'))
        resolved_crop_mode = timm_data_config.get('crop_mode', getattr(model_config, 'crop_mode', 'center'))
        transform = create_transform(input_size=effective_input_resolution, is_training=False, mean=timm_data_config.get('mean', getattr(model_config, 'mean', (0.485, 0.456, 0.406))), std=timm_data_config.get('std', getattr(model_config, 'std', (0.229, 0.224, 0.225))), crop_mode=resolved_crop_mode, crop_pct=resolved_crop_pct, interpolation=resolved_interpolation)
        preprocessing.update({'crop_mode': resolved_crop_mode, 'crop_pct': resolved_crop_pct, 'interpolation': resolved_interpolation})
    transformed_dataset = _TransformedDataset(dataset, transform)
    device = _model_device(model)
    return locals()
