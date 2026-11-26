"""NVIDIA入口：官方模型加载 → 结构/环境记录 → 全部图片提取。"""
import os
from datetime import datetime
from pathlib import Path
ROOT = Path(__file__).resolve().parent
GPU_ID = '0'
MODEL_ID = 'GoodBaiBai88/M3D-CLIP'
BUNDLE_DIR = (Path(__file__).resolve().parents[5] / 'dataset' / 'aimv2_features/imagenet_validation_1000')
BATCH_SIZE = 1
NUM_WORKERS = 8
WARMUP_STEPS = 3
os.environ.setdefault('CUDA_VISIBLE_DEVICES', GPU_ID)
os.environ.setdefault('HF_ENDPOINT', 'https://hf-mirror.com')
os.environ.setdefault('HF_HUB_CACHE', str(Path(__file__).resolve().parents[1] / 'weights/.runtime_cache'))
import numpy as np
import torch
import torch.nn.functional as F
from huggingface_hub import snapshot_download
from transformers import AutoModel
from inference.utils.evaluate_model import evaluate_model
from inference.utils.print_model_info import print_model_and_environment
from inference.utils.protocol import checkpoint_fingerprints

class PrepareBatch:
    """自定义ImageNet二维→三维适配；不是官方ImageNet评测流程。"""

    def __call__(self, images):
        samples = []
        for image in images:
            array = np.array(image.convert('L'), dtype=np.float32)
            image_tensor = torch.from_numpy(array)
            image_tensor = image_tensor - image_tensor.min()
            image_tensor = image_tensor / image_tensor.max().clamp_min(1e-08)
            image_tensor = image_tensor[None, None, None, :, :]
            image_tensor = F.interpolate(image_tensor, size=(32, 256, 256), mode='trilinear', align_corners=False)
            samples.append(image_tensor[0])
        return torch.stack(samples, dim=0)

def forward_fn(model, inputs):
    return model.encode_image(inputs)

def select_output(outputs):
    return outputs[:, 0]

def main():
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA不可用，请检查环境；不自动改用CPU')
    snapshot = snapshot_download(MODEL_ID, allow_patterns=['*.safetensors', '*.bin', '*.json', '*.py'])
    model = AutoModel.from_pretrained(snapshot, trust_remote_code=True)
    model.cuda().eval()
    output_dir = RUN_ARGS.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    print_model_and_environment(model, output_dir)
    summary = evaluate_model(model, PrepareBatch(), forward_fn, select_output, bundle_dir=BUNDLE_DIR, output_dir=output_dir, model_info={'source': 'huggingface', 'id': MODEL_ID, 'revision': Path(snapshot).name, 'checkpoint_files': checkpoint_fingerprints(snapshot)}, preprocessing_info={'class': 'ImageNet2DToM3DVolume', 'config': {'official_imagenet_protocol': False, 'grayscale': 'Pillow convert L', 'normalization': 'per-image min-max; denominator clamp_min=1e-8', 'crop': 'none', 'resize': [32, 256, 256], 'interpolation': 'torch interpolate trilinear', 'align_corners': False, 'input_axis_order': 'B,C,D,H,W', 'input_dtype': 'float32', 'source_depth': 1, 'augmentation': 'none', 'numpy_version': np.__version__}, 'decode': 'Pillow open, first frame, convert RGB; no EXIF transpose', 'input_hash_protocol': 'named-tensors-original-dtype-le-v2'}, output_name='encode_image.cls_projected_l2_normalized', batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, warmup_steps=WARMUP_STEPS)
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
