from inference.options import ARGS
from inference.runtime import local_weight
import os
import sys
from datetime import datetime
from pathlib import Path
import torch
import torch.nn as nn
current_dir = os.path.dirname(os.path.abspath(__file__))
model_dir = str(ARGS.weights)
if model_dir not in sys.path:
    sys.path.append(model_dir)
sys.path.insert(0, str(Path(__file__).parent / 'sumsub_synthetic_mj_5'))
from pipeline import PreTrainedPipeline
from inference.runner import evaluate_model
from inference.runtime import print_model_and_environment

class ModelConfig:

    def __init__(self):
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]
        self.crop_mode = 'center'
        self.crop_pct = 0.875
        self.num_labels = 2
        self.id2label = {0: 'by human', 1: 'by AI'}

class WrappedModel(nn.Module):

    def __init__(self, raw_model, config):
        super().__init__()
        self.model = raw_model
        self.config = config

    def forward(self, x):
        out = self.model(x)
        if isinstance(out, torch.Tensor):
            return {'logits': out}
        return out

def main():
    device = torch.device(ARGS.device)
    print(f'正在加载模型: {model_dir} ...')
    pipe = PreTrainedPipeline(model_dir)
    config = ModelConfig()
    wrapped_model = WrappedModel(pipe.model, config).to(device)
    output_dir = ARGS.output_dir
    print_model_and_environment(wrapped_model, output_dir)
    print('开始评估...')
    evaluate_model(model=wrapped_model, dataset_name='coco-val2017', output_dir=output_dir, batch_size=64, num_workers=8)

