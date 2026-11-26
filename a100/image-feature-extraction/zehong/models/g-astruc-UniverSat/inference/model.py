"""NVIDIA入口：官方模型加载 → 结构/环境记录 → 全部图片提取。"""
import os
import sys
from datetime import datetime
from pathlib import Path
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
GPU_ID = '0'
MODEL_ID = 'g-astruc/UniverSat'
BUNDLE_DIR = (Path(__file__).resolve().parents[5] / 'dataset' / 'pastis_spot_bundle_1000')
BATCH_SIZE = 8
NUM_WORKERS = 8
WARMUP_STEPS = 3
UNIVERSAT_REPO = Path(__file__).resolve().parent / 'upstream/UniverSat'
IMAGE_SIZE = 320
PATCH_SIZE_M = 40
OUTPUT_GRID = 32
os.environ.setdefault('CUDA_VISIBLE_DEVICES', GPU_ID)
os.environ.setdefault('HF_ENDPOINT', 'https://hf-mirror.com')
os.environ.setdefault('HF_HUB_CACHE', str(Path(__file__).resolve().parents[1] / 'weights/.runtime_cache'))
import torch
from huggingface_hub import snapshot_download
from inference.utils.evaluate_model import evaluate_model
from inference.utils.print_model_info import print_model_and_environment
from inference.utils.protocol import checkpoint_fingerprints, processor_protocol, sha256_file
import json
import sys
import subprocess
import numpy as np
from PIL import Image

class PrepareBatch:

    def __init__(self, normalization_file):
        path = Path(normalization_file)
        self.stats = json.loads(path.read_text(encoding='utf-8'))
        self.stats_sha256 = sha256_file(path)
        self.mean = torch.tensor(self.stats['mean'], dtype=torch.float32).view(3, 1, 1)
        self.std = torch.tensor(self.stats['std'], dtype=torch.float32).view(3, 1, 1).clamp_min(1e-06)
        if not torch.isfinite(self.mean).all():
            raise ValueError('Invalid mean')
        if not torch.isfinite(self.std).all():
            raise ValueError('Invalid std')

    def __call__(self, images):
        samples = []
        for image in images:
            array = np.asarray(image)
            if array.shape != (3, IMAGE_SIZE, IMAGE_SIZE):
                raise ValueError(f'Unexpected input shape: {array.shape}')
            if array.dtype != np.uint8:
                raise ValueError(f'Unexpected input dtype: {array.dtype}')
            x = torch.from_numpy(np.array(array, dtype=np.float32, copy=True))
            samples.append((x - self.mean) / self.std)
        return {'spot': torch.stack(samples)}

    def to_dict(self):
        return {'recipe': 'pastis_spot_native_320_dataset_zscore_v1', 'input_shape': [3, IMAGE_SIZE, IMAGE_SIZE], 'channel_order': 'source TIFF band order, RGB', 'resize': None, 'rescale': None, 'normalization': self.stats, 'normalization_sha256': self.stats_sha256, 'modality': 'spot', 'wavelengths_um': [0.665, 0.56, 0.49], 'input_res_m': 1.0, 'subpatches': 10, 'patch_size_m': PATCH_SIZE_M, 'output_grid_side': OUTPUT_GRID, 'compile': False}

def forward_fn(model, inputs):
    return model.encode(inputs, patch_size=PATCH_SIZE_M, output_grid=OUTPUT_GRID)

def select_output(outputs):
    return outputs[0]

def main():
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA不可用，请检查环境；不自动改用CPU')
    sys.path.insert(0, str(UNIVERSAT_REPO))
    from hubconf import UniverSat
    snapshot = snapshot_download(MODEL_ID, allow_patterns=['*.safetensors', '*.json'])
    processor = PrepareBatch(BUNDLE_DIR / 'normalization.json')
    model = UniverSat.from_pretrained(str(snapshot), compile=False)
    model.cuda().eval()
    code_revision = 'vendored-source'.strip()
    output_dir = RUN_ARGS.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    print_model_and_environment(model, output_dir)
    preprocessing_info = processor_protocol(processor)
    preprocessing_info['decode'] = 'rasterio MemoryFile; source TIFF band order CHW; preserve raw uint8 values; no Pillow RGB conversion'
    summary = evaluate_model(model, processor, forward_fn, select_output, bundle_dir=BUNDLE_DIR, output_dir=output_dir, model_info={'source': 'huggingface', 'id': MODEL_ID, 'revision': Path(snapshot).name, 'checkpoint_files': checkpoint_fingerprints(snapshot), 'code_repository': 'https://github.com/gastruc/UniverSat', 'code_revision': code_revision}, preprocessing_info=preprocessing_info, output_name='encode_spatial_tokens', batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, warmup_steps=WARMUP_STEPS, precision='none')
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
