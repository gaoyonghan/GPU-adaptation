"""NVIDIA入口：官方模型加载 → 结构/环境记录 → 全部图片提取。"""
import os
from datetime import datetime
from pathlib import Path
ROOT = Path(__file__).resolve().parent
GPU_ID = '0'
MODEL_ID = 'kaiko-ai/midnight'
BUNDLE_DIR = (Path(__file__).resolve().parents[5] / 'dataset' / 'pathology/crc_val_he_7k_bundle')
BATCH_SIZE = 64
NUM_WORKERS = 8
WARMUP_STEPS = 3
os.environ.setdefault('CUDA_VISIBLE_DEVICES', GPU_ID)
os.environ.setdefault('HF_ENDPOINT', 'https://hf-mirror.com')
os.environ.setdefault('HF_HUB_CACHE', str(Path(__file__).resolve().parents[1] / 'weights/.runtime_cache'))
import torch
from huggingface_hub import snapshot_download
from transformers import AutoModel
from torchvision.transforms import v2, InterpolationMode
from inference.utils.evaluate_model import evaluate_model
from inference.utils.print_model_info import print_model_and_environment
from inference.utils.protocol import checkpoint_fingerprints

class PrepareBatch:

    def __init__(self, transform):
        self.transform = transform

    def __call__(self, images):
        return {'pixel_values': torch.stack([self.transform(image) for image in images])}

def forward_fn(model, inputs):
    return model(**inputs)

def select_output(outputs):
    tokens = outputs.last_hidden_state
    return torch.cat([tokens[:, 0, :], tokens[:, 1:, :].mean(dim=1)], dim=-1)

def main():
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA不可用，请检查环境；不自动改用CPU')
    snapshot = snapshot_download(MODEL_ID, allow_patterns=['*.safetensors', '*.bin', '*.json', '*.py'])
    processor = v2.Compose([v2.Resize(224, interpolation=InterpolationMode.BILINEAR, antialias=True), v2.CenterCrop(224), v2.ToTensor(), v2.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))])
    model = AutoModel.from_pretrained(snapshot)
    model.cuda().eval()
    output_dir = RUN_ARGS.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    print_model_and_environment(model, output_dir)
    summary = evaluate_model(model, PrepareBatch(processor), forward_fn, select_output, bundle_dir=BUNDLE_DIR, output_dir=output_dir, model_info={'source': 'huggingface', 'id': MODEL_ID, 'revision': Path(snapshot).name, 'checkpoint_files': checkpoint_fingerprints(snapshot)}, preprocessing_info={'class': 'torchvision.transforms.v2.Compose', 'config': {'resize_shorter_edge': 224, 'interpolation': 'bilinear', 'antialias': True, 'center_crop': [224, 224], 'to_tensor': 'float32 CHW; uint8 divided by 255', 'mean': [0.5, 0.5, 0.5], 'std': [0.5, 0.5, 0.5]}, 'decode': 'Pillow open, first frame, convert RGB; no EXIF transpose', 'input_hash_protocol': 'named-tensors-original-dtype-le-v2'}, output_name='cls_concat_mean_patch', batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, warmup_steps=WARMUP_STEPS)
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
