"""NVIDIA入口：官方模型加载 → 结构/环境记录 → 全部图片提取。"""
import os
from datetime import datetime
from pathlib import Path
ROOT = Path(__file__).resolve().parent
GPU_ID = '0'
MODEL_ID = 'bytedance-research/LVFace'
NETWORK = 'vit_b_dp005_mask_005'
WEIGHT_FILE = 'LVFace-B_Glint360K/LVFace-B_Glint360K.pt'
LVFACE_REPO = Path(__file__).resolve().parent / 'upstream/LVFace'
BUNDLE_DIR = (Path(__file__).resolve().parents[5] / 'dataset' / 'LVdataset/ijbb_bundle_1000')
BATCH_SIZE = 64
NUM_WORKERS = 8
WARMUP_STEPS = 3
os.environ.setdefault('CUDA_VISIBLE_DEVICES', GPU_ID)
os.environ.setdefault('HF_ENDPOINT', 'https://hf-mirror.com')
os.environ.setdefault('HF_HUB_CACHE', str(Path(__file__).resolve().parents[1] / 'weights/.runtime_cache'))
import sys
import subprocess
import cv2
import numpy as np
import torch
from huggingface_hub import snapshot_download
# Use the vendored package directly; do not shadow inference/ with vendor/LVFace/inference.py.
from .upstream.LVFace.backbones import get_model
from inference.utils.evaluate_model import evaluate_model
from inference.utils.print_model_info import print_model_and_environment
from inference.utils.protocol import checkpoint_fingerprints, sha256_file

class PrepareBatch:
    """保留样例Pillow RGB读取；采用官方推理的resize和归一化。"""

    def __call__(self, images):
        samples = []
        for image in images:
            rgb = np.asarray(image, dtype=np.uint8)
            rgb = cv2.resize(rgb, (112, 112), interpolation=cv2.INTER_LINEAR)
            chw = np.ascontiguousarray(np.transpose(rgb, (2, 0, 1)))
            tensor = torch.from_numpy(chw).float()
            tensor.div_(255).sub_(0.5).div_(0.5)
            samples.append(tensor)
        return torch.stack(samples, dim=0)

def forward_fn(model, inputs):
    return model(inputs)

def select_output(outputs):
    return outputs

def main():
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA不可用，请检查环境；不自动改用CPU')
    snapshot = snapshot_download(MODEL_ID, allow_patterns=[WEIGHT_FILE])
    checkpoint = Path(snapshot) / WEIGHT_FILE
    github_revision = 'vendored-source'.strip()
    model = get_model(NETWORK, fp16=False)
    state_dict = torch.load(checkpoint, map_location='cpu', weights_only=True)
    model.load_state_dict(state_dict, strict=True)
    model.cuda().eval()
    output_dir = RUN_ARGS.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    print_model_and_environment(model, output_dir)
    summary = evaluate_model(model, PrepareBatch(), forward_fn, select_output, bundle_dir=BUNDLE_DIR, output_dir=output_dir, model_info={'source': 'huggingface_weights+github_code', 'id': MODEL_ID, 'revision': Path(snapshot).name, 'variant': 'LVFace-B_Glint360K', 'network': NETWORK, 'checkpoint_files': checkpoint_fingerprints(checkpoint), 'github_repository': 'https://github.com/bytedance/LVFace', 'github_revision': github_revision, 'code_fingerprints': {'backbones/__init__.py': sha256_file(LVFACE_REPO / 'backbones' / '__init__.py'), 'backbones/vit.py': sha256_file(LVFACE_REPO / 'backbones' / 'vit.py')}, 'numerical_policy': {'outer_autocast': False, 'model_internal_autocast': 'official local CUDA AMP', 'parameters': 'default FP32', 'output_postprocessing': 'none'}}, preprocessing_info={'class': 'LVFaceOfficialInferenceAdaptedToSharedReader', 'decode': 'Pillow open, first frame, convert RGB; no EXIF transpose', 'resize': {'library': 'OpenCV', 'size': [112, 112], 'interpolation': 'INTER_LINEAR'}, 'layout': 'CHW', 'dtype': 'torch.float32', 'normalization': 'div_(255).sub_(0.5).div_(0.5)', 'face_alignment': False, 'flip_test': False, 'input_hash_protocol': 'named-tensors-original-dtype-le-v2'}, output_name='face_embedding', batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, warmup_steps=WARMUP_STEPS, precision='none')
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
