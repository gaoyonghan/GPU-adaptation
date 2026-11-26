"""NVIDIA入口：官方模型加载 → 结构/环境记录 → 全部图片提取。"""
import os
from datetime import datetime
from pathlib import Path
ROOT = Path(__file__).resolve().parent
GPU_ID = '0'
MODEL_ID = 'Lin-Chen/ShareGPT4V-7B_Pretrained_vit-large336-l12'
BUNDLE_DIR = (Path(__file__).resolve().parents[5] / 'dataset' / 'mm-vet/mmvet_bundle')
BATCH_SIZE = 64
NUM_WORKERS = 8
WARMUP_STEPS = 3
os.environ.setdefault('CUDA_VISIBLE_DEVICES', GPU_ID)
os.environ.setdefault('HF_ENDPOINT', 'https://hf-mirror.com')
os.environ.setdefault('HF_HUB_CACHE', str(Path(__file__).resolve().parents[1] / 'weights/.runtime_cache'))
import torch
from huggingface_hub import snapshot_download
from PIL import Image
from transformers import CLIPImageProcessor, CLIPVisionModel
from inference.utils.evaluate_model import evaluate_model
from inference.utils.print_model_info import print_model_and_environment
from inference.utils.protocol import checkpoint_fingerprints, processor_protocol

class PrepareBatch:
    """沿用官方 pad 预处理；保持顶层类以支持 spawn。"""

    def __init__(self, processor):
        self.processor = processor

    def __call__(self, images):
        samples = []
        background = tuple((int(x * 255) for x in self.processor.image_mean))
        for image in images:
            width, height = image.size
            if width != height:
                side = max(width, height)
                square = Image.new(image.mode, (side, side), background)
                square.paste(image, ((side - width) // 2, (side - height) // 2))
                image = square
            sample = self.processor.preprocess(image, return_tensors='pt')
            samples.append(sample['pixel_values'])
        return {'pixel_values': torch.cat(samples, dim=0).to(torch.float16)}

def forward_fn(model, inputs):
    return model(**inputs, output_hidden_states=True, return_dict=True)

def select_output(outputs):
    return outputs.hidden_states[-2][:, 1:, :]

def main():
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA不可用，请检查环境；不自动改用CPU')
    snapshot = snapshot_download(MODEL_ID, allow_patterns=['*.safetensors', '*.bin', '*.json', '*.py'])
    processor = CLIPImageProcessor.from_pretrained(snapshot)
    model = CLIPVisionModel.from_pretrained(snapshot, torch_dtype=torch.float16, attn_implementation='eager')
    model.cuda().eval()
    output_dir = RUN_ARGS.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    print_model_and_environment(model, output_dir)
    preprocessing_info = processor_protocol(processor)
    preprocessing_info.update({'image_aspect_ratio': 'pad', 'pad_method': 'center square, background=int(image_mean*255)', 'pixel_values_cast': 'torch.float16 after processor'})
    summary = evaluate_model(model, PrepareBatch(processor), forward_fn, select_output, bundle_dir=BUNDLE_DIR, output_dir=output_dir, model_info={'source': 'huggingface', 'id': MODEL_ID, 'revision': Path(snapshot).name, 'checkpoint_files': checkpoint_fingerprints(snapshot)}, preprocessing_info=preprocessing_info, output_name='hidden_states[-2][:,1:,:]', batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, warmup_steps=WARMUP_STEPS)
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
