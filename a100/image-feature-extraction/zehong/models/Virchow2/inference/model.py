"""NVIDIA 入口：Virchow2 官方模型加载 -> 结构/环境记录 -> 全部图片提取。"""
import os
from datetime import datetime
from pathlib import Path
ROOT = Path(__file__).resolve().parent
GPU_ID = '0'
MODEL_ID = 'paige-ai/Virchow2'
BUNDLE_DIR = (Path(__file__).resolve().parents[5] / 'dataset' / 'nct_crc_nonorm_bundle_1000')
BATCH_SIZE = 64
NUM_WORKERS = 8
WARMUP_STEPS = 3
os.environ.setdefault('CUDA_VISIBLE_DEVICES', GPU_ID)
os.environ.setdefault('HF_ENDPOINT', 'https://hf-mirror.com')
os.environ.setdefault('HF_HUB_CACHE', str(Path(__file__).parent.parent/'weights/.runtime_cache'))
import timm
import torch
from huggingface_hub import snapshot_download
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from timm.layers import SwiGLUPacked
from inference.utils.evaluate_model import evaluate_model
from inference.utils.print_model_info import print_model_and_environment
from inference.utils.protocol import checkpoint_fingerprints, processor_protocol

class PrepareBatch:
    """官方 transform 逐图处理并拼批；类在顶层以支持 spawn worker。"""

    def __init__(self, transform):
        self.transform = transform

    def __call__(self, images):
        tensors = [self.transform(image.convert('RGB')) for image in images]
        return torch.stack(tensors)

def forward_fn(model, inputs):
    return model(inputs)

def select_output(outputs):
    """
    Virchow2 官方推荐推理特征提取：
    outputs 形状为 [B, 261, 1280]
    outputs[:, 0] 是 class_token
    outputs[:, 1:5] 是 4 个 register tokens（官方指明忽略）
    outputs[:, 5:] 是 256 个 patch tokens
    拼接 class token 与 patch tokens 的 mean，生成 [B, 2560] 特征向量
    """
    class_token = outputs[:, 0]
    patch_tokens = outputs[:, 5:]
    embedding = torch.cat([class_token, patch_tokens.mean(dim=1)], dim=-1)
    return embedding

def main():
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA 不可用，请检查环境；不自动改用 CPU')
    snapshot = snapshot_download(MODEL_ID, allow_patterns=['*.safetensors', '*.bin', '*.json', '*.py'])
    model = local_timm_create(snapshot, mlp_layer=SwiGLUPacked, act_layer=torch.nn.SiLU)
    model.cuda().eval()
    transform_config = resolve_data_config(model.pretrained_cfg, model=model)
    transform = create_transform(**transform_config, is_training=False)
    output_dir = RUN_ARGS.output_dir.absolute()
    output_dir.mkdir(parents=True, exist_ok=False)
    print_model_and_environment(model, output_dir)
    summary = evaluate_model(model, PrepareBatch(transform), forward_fn, select_output, bundle_dir=BUNDLE_DIR, output_dir=output_dir, model_info={'source': 'huggingface', 'id': MODEL_ID, 'revision': checkpoint_revision(snapshot), 'checkpoint_files': checkpoint_fingerprints(snapshot)}, preprocessing_info=processor_protocol(transform), output_name='cls_mean_patch_concat', batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, warmup_steps=WARMUP_STEPS)
    print('Results:', output_dir)
    print('Status:', summary['status'], 'images:', summary['num_images'])
    print('Performance:', summary['performance'])
    if summary['status'] != 'complete':
        raise RuntimeError('存在无效输出，请查看 summary 和 feature_index')

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
