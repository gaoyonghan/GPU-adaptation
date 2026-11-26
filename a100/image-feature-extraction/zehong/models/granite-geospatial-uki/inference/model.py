"""NVIDIA A100入口：官方模型加载 → 结构/环境记录 → 全部图片提取。"""
import os
from datetime import datetime
from pathlib import Path
import tifffile
import numpy as np
import yaml
ROOT = Path(__file__).resolve().parent
GPU_ID = '0'
MODEL_ID = 'ibm-granite/granite-geospatial-uki'
BUNDLE_DIR = (Path(__file__).resolve().parents[5] / 'dataset' / 'granite_biomass_bundle')
BATCH_SIZE = 64
NUM_WORKERS = 8
WARMUP_STEPS = 3
os.environ.setdefault('CUDA_VISIBLE_DEVICES', GPU_ID)
os.environ.setdefault('HF_ENDPOINT', 'https://hf-mirror.com')
os.environ.setdefault('HF_HUB_CACHE', str(Path(__file__).parent.parent/'weights/.runtime_cache'))
import torch
_orig_torch_load = torch.load

def _patched_torch_load(*args, **kwargs):
    kwargs['weights_only'] = False
    return _orig_torch_load(*args, **kwargs)
torch.load = _patched_torch_load
import terratorch.models.backbones.prithvi_vit
import inference.native.granite_geospatial_uki
from huggingface_hub import hf_hub_download, snapshot_download
from terratorch.tasks import SemanticSegmentationTask
from inference.utils.evaluate_model import evaluate_model
from inference.utils.print_model_info import print_model_and_environment
from inference.utils.protocol import checkpoint_fingerprints

class PrepareBatch:
    """多波段遥感切片逐图读取、通道精确对齐与拼批。"""

    def __init__(self):
        self.band_indices = [1, 2, 3, 4, 5, 6, 7, 8, 9]

    def __call__(self, image_inputs):
        tensors = []
        for item in image_inputs:
            if isinstance(item, (str, Path)):
                arr = tifffile.imread(str(item))
            elif hasattr(item, 'filename'):
                arr = tifffile.imread(item.filename)
            elif isinstance(item, np.ndarray):
                arr = item
            else:
                arr = tifffile.imread(item)
            if arr.ndim == 3 and arr.shape[-1] <= 16:
                arr = np.transpose(arr, (2, 0, 1))
            elif arr.ndim == 2:
                arr = arr[np.newaxis, :, :]
            if arr.shape[0] >= 10:
                arr = arr[self.band_indices, :, :]
            elif arr.shape[0] == 9:
                arr = arr[:9, :, :]
            else:
                pad = np.zeros((9 - arr.shape[0], arr.shape[1], arr.shape[2]), dtype=arr.dtype)
                arr = np.concatenate([arr, pad], axis=0)
            tensor = torch.from_numpy(arr.copy()).float()
            tensors.append(tensor)
        batch_tensor = torch.stack(tensors, dim=0)
        return {'x': batch_tensor}

def forward_fn(model, inputs):
    if isinstance(inputs, dict):
        return model(**inputs)
    return model(inputs)

def select_output(outputs):
    if isinstance(outputs, torch.Tensor):
        return outputs
    if hasattr(outputs, 'output'):
        return outputs.output
    if hasattr(outputs, 'last_hidden_state'):
        return outputs.last_hidden_state
    if isinstance(outputs, dict):
        if 'output' in outputs:
            return outputs['output']
        for v in outputs.values():
            if isinstance(v, torch.Tensor):
                return v
    raise ValueError(f'无法识别的输出结构: {type(outputs)}')

def load_model_from_checkpoint(config_path, ckpt_path):
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)
    model_init_args = cfg.get('model', {}).get('init_args', {})
    model_args = model_init_args.get('model_args', {})
    if model_args.get('backbone')!='granite_geospatial_uki':raise ValueError('Unexpected official UKI backbone config')
    model_args['backbone_pretrained'] = False
    model_args['num_classes'] = model_args.get('num_classes', 2)
    if 'backbone_ckpt_path' in model_args:
        del model_args['backbone_ckpt_path']
    if 'aux_heads' in model_init_args:
        del model_init_args['aux_heads']
    if 'aux_loss' in model_init_args:
        del model_init_args['aux_loss']
    task = SemanticSegmentationTask(model_args=model_args, model_factory='EncoderDecoderFactory', loss='ce')
    ckpt = torch.load(ckpt_path, map_location='cpu')
    raw_dict = ckpt['state_dict'] if 'state_dict' in ckpt else ckpt
    clean_state_dict = {}
    for k, v in raw_dict.items():
        if k.startswith('model.model.'):
            clean_state_dict[k.replace('model.model.', 'model.')] = v
        elif k.startswith('model.'):
            clean_state_dict[k] = v
        else:
            clean_state_dict[f'model.{k}'] = v
    incompatible = task.load_state_dict(clean_state_dict, strict=False)
    print(f'权重加载情况: missing={len(incompatible.missing_keys)}, unexpected={len(incompatible.unexpected_keys)}')
    print('LOAD_MISSING_KEYS',incompatible.missing_keys,flush=True)
    print('LOAD_UNEXPECTED_KEYS',incompatible.unexpected_keys,flush=True)
    allowed=lambda key: key=='model.criterion.weight' or key.startswith('model.aux_heads.aux_head.')
    if incompatible.missing_keys or any(not allowed(k) for k in incompatible.unexpected_keys):
        raise RuntimeError('Unsupported checkpoint mismatch: '+str(incompatible))
    # Only training loss / auxiliary-head keys may be excluded; strictly reload every active state key.
    active=task.state_dict()
    task.load_state_dict({k:v for k,v in clean_state_dict.items() if k in active},strict=True)

    return task.model

def main():
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA不可用，请检查环境；不自动改用CPU')
    snapshot = snapshot_download(MODEL_ID, allow_patterns=['config.yaml', '*.ckpt', '*.pt', '*.json', '*.py'])
    ckpt_path = str(hf_hub_download(repo_id=MODEL_ID, filename='granite_geospatial_uki_flood_detection_v2.ckpt'))
    config_path = str(hf_hub_download(repo_id=MODEL_ID, filename='config.yaml'))
    model = load_model_from_checkpoint(config_path, ckpt_path)
    model.cuda().eval()
    output_dir = RUN_ARGS.output_dir.absolute()
    output_dir.mkdir(parents=True, exist_ok=False)
    print_model_and_environment(model, output_dir)
    preprocessing_info = {'framework': 'terratorch_multiband_tiff', 'channels': '9_channels_optical_sar_cloud', 'transpose': '[C, H, W]'}
    summary = evaluate_model(model, PrepareBatch(), forward_fn, select_output, bundle_dir=BUNDLE_DIR, output_dir=output_dir, model_info={'source': 'huggingface', 'id': MODEL_ID, 'revision': checkpoint_revision(snapshot), 'checkpoint_files': checkpoint_fingerprints(snapshot)}, preprocessing_info=preprocessing_info, output_name='uki_segmentation_output', batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, warmup_steps=WARMUP_STEPS)
    print('Results:', output_dir)
    print('Status:', summary['status'], 'images:', summary['num_images'])
    print('Performance:', summary['performance'])
    if summary['status'] != 'complete':
        raise RuntimeError('存在无效输出，请查看summary和feature_index')

RUN_ARGS=None

def local_timm_create(snapshot, **overrides):
    import json, timm
    config=json.loads((Path(snapshot)/'config.json').read_text())
    cfg=dict(config.get('pretrained_cfg',{}))
    for key in ['num_classes','label_names','label_descriptions']:
        if key in config:cfg[key]=config[key]
    model_args=dict(config.get('model_args',{}));model_args.update(overrides)
    weight=next((Path(snapshot)/f for f in ['model.safetensors','pytorch_model.bin'] if (Path(snapshot)/f).is_file()),None)
    if weight is None:raise FileNotFoundError('Local timm checkpoint missing')
    return timm.create_model(config['architecture'],pretrained=True,pretrained_cfg=cfg,
        pretrained_cfg_overlay={'source':'','file':str(weight.absolute()),'custom_load':False},**model_args)

def snapshot_download(*args,**kwargs):
    if not RUN_ARGS.weights.is_dir():raise FileNotFoundError(RUN_ARGS.weights)
    return str(RUN_ARGS.weights.absolute())

def hf_hub_download(repo_id=None,filename=None,**kwargs):
    path=RUN_ARGS.weights/filename
    if not path.is_file():raise FileNotFoundError(path)
    return str(path.absolute())



def checkpoint_revision(path):
    import re,json
    path=Path(path)
    if re.fullmatch(r'[0-9a-f]{40}',path.name):return path.name
    config=path/'config.json'
    if config.is_file():
        value=json.loads(config.read_text()).get('_commit_hash')
        if isinstance(value,str) and re.fullmatch(r'[0-9a-f]{40}',value):return value
    return None
