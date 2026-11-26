"""NVIDIA / 跨设备入口：TTPlanet SDXL ControlNet Tile v2 特征提取与跨设备比对。"""
import os
from datetime import datetime
from pathlib import Path
from PIL import Image
import numpy as np
ROOT = Path(__file__).resolve().parent
GPU_ID = '0'
MODEL_ID = 'TTPlanet/TTPLanet_SDXL_Controlnet_Tile_Realistic'
WEIGHT_NAME = 'TTPLANET_Controlnet_Tile_realistic_v2_fp16.safetensors'
BUNDLE_DIR = (Path(__file__).resolve().parents[5] / 'dataset' / 'aimv2_features/imagenet_validation_1000')
BATCH_SIZE = 64
NUM_WORKERS = 8
WARMUP_STEPS = 3
os.environ.setdefault('CUDA_VISIBLE_DEVICES', GPU_ID)
os.environ.setdefault('HF_ENDPOINT', 'https://hf-mirror.com')
os.environ.setdefault('HF_HUB_CACHE', str(Path(__file__).parent.parent / 'weights/.runtime_cache'))
import torch
from huggingface_hub import hf_hub_download
from diffusers import ControlNetModel
from inference.utils.evaluate_model import evaluate_model
from inference.utils.print_model_info import print_model_and_environment
from inference.utils.protocol import checkpoint_fingerprints

class PrepareBatch:
    """ControlNet Tile 输入处理：将原始图像调整至 1024x1024 并转为 [0, 1] 浮点 Tensor。
    类置于模块顶层以支持多进程 spawn 模式序列化。
    """

    def __init__(self, target_size=(1024, 1024)):
        self.target_size = target_size

    def __call__(self, images):
        tensors = []
        for img in images:
            img = img.convert('RGB').resize(self.target_size, resample=Image.BICUBIC)
            arr = np.array(img).astype(np.float32) / 255.0
            tensor = torch.from_numpy(arr).permute(2, 0, 1)
            tensors.append(tensor)
        batch_cond = torch.stack(tensors, dim=0)
        return {'controlnet_cond': batch_cond}

def prepare_device_inputs(model, inputs):
    """仅包围官方前向计算，适配 SDXL ControlNet 所需的完整条件张量。"""
    cond = inputs['controlnet_cond']
    batch_size = cond.shape[0]
    device = cond.device
    dtype = next(model.parameters()).dtype
    dummy_latents = torch.zeros((batch_size, 4, cond.shape[2] // 8, cond.shape[3] // 8), device=device, dtype=dtype)
    timestep = torch.tensor([0] * batch_size, device=device, dtype=torch.long)
    dummy_text_embeds = torch.zeros((batch_size, 77, 2048), device=device, dtype=dtype)
    dummy_pooled_text_embeds = torch.zeros((batch_size, 1280), device=device, dtype=dtype)
    dummy_time_ids = torch.tensor([[1024, 1024, 0, 0, 1024, 1024]] * batch_size, device=device, dtype=dtype)
    added_cond_kwargs = {'text_embeds': dummy_pooled_text_embeds, 'time_ids': dummy_time_ids}
    return {'sample': dummy_latents, 'timestep': timestep, 'encoder_hidden_states': dummy_text_embeds, 'controlnet_cond': cond.to(dtype=dtype), 'added_cond_kwargs': added_cond_kwargs, 'return_dict': True}

def select_output(outputs):
    """提取 ControlNet 的中心瓶颈层输出 (mid_block_res_sample)。
    该张量形状固定为 [B, 1280, 16, 16]，完整表征了空间与条件融合后的深层特征。
    """
    return outputs.mid_block_res_sample

def main():
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA不可用，请检查环境；不自动改用CPU')
    weight_path = hf_hub_download(repo_id=MODEL_ID, filename=WEIGHT_NAME)
    config_path = hf_hub_download(repo_id=MODEL_ID, filename='config.json')
    model_snapshot_dir = Path(weight_path).parent
    model = ControlNetModel.from_single_file(weight_path, config=config_path, torch_dtype=torch.float16)
    model.cuda().eval()
    output_dir = RUN_ARGS.output_dir.absolute()
    output_dir.mkdir(parents=True, exist_ok=False)
    print_model_and_environment(model, output_dir)
    preprocessing_info = {'processor_class': 'ResizeAndNormalize', 'target_size': [1024, 1024], 'resample': 'BICUBIC', 'normalization': 'range_[0,1]'}
    summary = evaluate_model(model, PrepareBatch(target_size=(1024, 1024)), forward_fn, select_output, bundle_dir=BUNDLE_DIR, output_dir=output_dir, model_info={'source': 'huggingface', 'id': MODEL_ID, 'weight_file': WEIGHT_NAME, 'revision': checkpoint_revision(model_snapshot_dir), 'checkpoint_files': checkpoint_fingerprints(str(model_snapshot_dir))}, preprocessing_info=preprocessing_info, output_name='mid_block_res_sample', batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, warmup_steps=WARMUP_STEPS, prepare_device_inputs=prepare_device_inputs)
    print('Results:', output_dir)
    print('Status:', summary['status'], 'images:', summary['num_images'])
    print('Performance:', summary['performance'])
    if summary['status'] != 'complete':
        raise RuntimeError('存在无效输出，请查看summary和feature_index')
RUN_ARGS = None

def local_timm_create(snapshot, **overrides):
    import json, timm
    config = json.loads((Path(snapshot) / 'config.json').read_text())
    cfg = dict(config.get('pretrained_cfg', {}))
    for key in ['num_classes', 'label_names', 'label_descriptions']:
        if key in config:
            cfg[key] = config[key]
    model_args = dict(config.get('model_args', {}))
    model_args.update(overrides)
    weight = next((Path(snapshot) / f for f in ['model.safetensors', 'pytorch_model.bin'] if (Path(snapshot) / f).is_file()), None)
    if weight is None:
        raise FileNotFoundError('Local timm checkpoint missing')
    return timm.create_model(config['architecture'], pretrained=True, pretrained_cfg=cfg, pretrained_cfg_overlay={'source': '', 'file': str(weight.absolute()), 'custom_load': False}, **model_args)

def snapshot_download(*args, **kwargs):
    if not RUN_ARGS.weights.is_dir():
        raise FileNotFoundError(RUN_ARGS.weights)
    return str(RUN_ARGS.weights.absolute())

def hf_hub_download(repo_id=None, filename=None, **kwargs):
    path = RUN_ARGS.weights / filename
    if not path.is_file():
        raise FileNotFoundError(path)
    return str(path.absolute())

def forward_fn(model, inputs):
    return model(**inputs)


def checkpoint_revision(path):
    import re,json
    path=Path(path)
    if re.fullmatch(r'[0-9a-f]{40}',path.name):return path.name
    config=path/'config.json'
    if config.is_file():
        value=json.loads(config.read_text()).get('_commit_hash')
        if isinstance(value,str) and re.fullmatch(r'[0-9a-f]{40}',value):return value
    return None
