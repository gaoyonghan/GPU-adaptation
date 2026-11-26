"""NVIDIA入口：官方模型加载 → 结构/环境记录 → 全部图片提取。"""
import os
from datetime import datetime
from pathlib import Path
ROOT = Path(__file__).resolve().parent
GPU_ID = '0'
MODEL_ID = 'facebook/PE-Spatial-G14-448'
MODEL_CONFIG = 'PE-Spatial-G14-448'
BUNDLE_DIR = (Path(__file__).resolve().parents[5] / 'dataset' / 'NYUv2_bundle_1000')
BATCH_SIZE = 64
NUM_WORKERS = 8
WARMUP_STEPS = 3
os.environ.setdefault('CUDA_VISIBLE_DEVICES', GPU_ID)
os.environ.setdefault('HF_ENDPOINT', 'https://hf-mirror.com')
os.environ.setdefault('HF_HUB_CACHE', str(Path(__file__).resolve().parents[1] / 'weights/.runtime_cache'))
import torch
from huggingface_hub import snapshot_download
import sys
_PERCEPTION_DIR = str(Path(__file__).resolve().parent/'perception_models')
if _PERCEPTION_DIR not in sys.path: sys.path.insert(0,_PERCEPTION_DIR)
import core.vision_encoder.pe as pe
import core.vision_encoder.transforms as transforms
from inference.utils.evaluate_model import evaluate_model
from inference.utils.print_model_info import print_model_and_environment
from inference.utils.protocol import checkpoint_fingerprints

class PrepareBatch:
    """只保存尺寸，官方transform在调用时创建，兼容spawn worker。"""

    def __init__(self, image_size):
        self.image_size = image_size

    def __call__(self, images):
        preprocess = transforms.get_image_transform(self.image_size)
        return torch.stack([preprocess(image) for image in images], dim=0)

def forward_fn(model, inputs):
    return model.forward_features(inputs, strip_cls_token=True)

def select_output(outputs):
    return outputs

def main():
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA不可用，请检查环境；不自动改用CPU')
    snapshot = snapshot_download(MODEL_ID, allow_patterns=[f'{MODEL_CONFIG}.pt'])
    checkpoint_path = Path(snapshot) / f'{MODEL_CONFIG}.pt'
    model = pe.VisionTransformer.from_config(MODEL_CONFIG, pretrained=True, checkpoint_path=str(checkpoint_path))
    model.cuda().eval()
    preprocess = transforms.get_image_transform(model.image_size)
    output_dir = RUN_ARGS.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    print_model_and_environment(model, output_dir)
    summary = evaluate_model(model, PrepareBatch(model.image_size), forward_fn, select_output, bundle_dir=BUNDLE_DIR, output_dir=output_dir, model_info={'source': 'huggingface', 'id': MODEL_ID, 'revision': Path(snapshot).name, 'checkpoint_files': checkpoint_fingerprints(snapshot)}, preprocessing_info={'class': 'core.vision_encoder.transforms.get_image_transform', 'config': {'image_size': model.image_size, 'center_crop': False, 'interpolation': 'bilinear', 'mean': [0.5, 0.5, 0.5], 'std': [0.5, 0.5, 0.5], 'transform': repr(preprocess)}, 'decode': 'Pillow open, first frame, convert RGB; no EXIF transpose', 'input_hash_protocol': 'named-tensors-original-dtype-le-v2'}, output_name='forward_features(strip_cls_token=True,norm=False,layer_idx=-1)', batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, warmup_steps=WARMUP_STEPS, precision='none')
    print('Results:', output_dir)
    print('Status:', summary['status'], 'images:', summary['num_images'])
    print('Performance:', summary['performance'])
    if summary['status'] != 'complete':
        raise RuntimeError('存在无效输出，请查看summary和feature_index')
RUN_ARGS = None

def snapshot_download(*args, **kwargs):
    return str(RUN_ARGS.weights.resolve())
if __name__ == '__main__':
    main()
