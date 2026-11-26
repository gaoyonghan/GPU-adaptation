"""NVIDIA入口：官方模型加载 → 结构/环境记录 → 全部图片提取。"""
import os
from datetime import datetime
from pathlib import Path
ROOT = Path(__file__).resolve().parent
GPU_ID = '0'
MODEL_ID = 'OpenGVLab/InternViT-6B-448px-V1-2'
BUNDLE_DIR = (Path(__file__).resolve().parents[5] / 'dataset' / 'coco_val_bundle_1000')
BATCH_SIZE = 32
NUM_WORKERS = 8
WARMUP_STEPS = 3
os.environ.setdefault('CUDA_VISIBLE_DEVICES', GPU_ID)
os.environ.setdefault('HF_ENDPOINT', 'https://hf-mirror.com')
os.environ.setdefault('HF_HUB_CACHE', str(Path(__file__).parent.parent/'weights/.runtime_cache'))
import torch
from huggingface_hub import snapshot_download
from transformers import AutoImageProcessor, AutoModel
from inference.utils.evaluate_model import evaluate_model
from inference.utils.print_model_info import print_model_and_environment
from inference.utils.protocol import checkpoint_fingerprints, processor_protocol

class PrepareBatch:
    """官方processor逐图处理并拼批；类在顶层以支持spawn worker。"""

    def __init__(self, processor):
        self.processor = processor

    def __call__(self, images):
        samples = [self.processor(images=image, return_tensors='pt') for image in images]
        return {key: torch.cat([sample[key] for sample in samples]) for key in samples[0]}

def forward_fn(model, inputs):
    return model(**inputs)

def select_output(outputs):
    return outputs.last_hidden_state

def main():
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA不可用，请检查环境；不自动改用CPU')
    snapshot = snapshot_download(MODEL_ID, allow_patterns=['*.safetensors', '*.bin', '*.json', '*.py'])
    processor = AutoImageProcessor.from_pretrained(snapshot, trust_remote_code=True)
    model = AutoModel.from_pretrained(snapshot, trust_remote_code=True)
    model.cuda().eval()
    output_dir = RUN_ARGS.output_dir.absolute()
    output_dir.mkdir(parents=True, exist_ok=False)
    print_model_and_environment(model, output_dir)
    summary = evaluate_model(model, PrepareBatch(processor), forward_fn, select_output, bundle_dir=BUNDLE_DIR, output_dir=output_dir, model_info={'source': 'huggingface', 'id': MODEL_ID, 'revision': checkpoint_revision(snapshot), 'checkpoint_files': checkpoint_fingerprints(snapshot)}, preprocessing_info=processor_protocol(processor), output_name='last_hidden_state', batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, warmup_steps=WARMUP_STEPS)
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
