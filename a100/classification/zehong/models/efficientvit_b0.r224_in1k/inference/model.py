from inference.options import ARGS
from inference.runtime import local_weight
import os
from datetime import datetime
from pathlib import Path
import torch
import timm
device = torch.device(ARGS.device)
model = timm.create_model('efficientvit_b0.r224_in1k', pretrained=True, pretrained_cfg_overlay={'file': str(local_weight().absolute()), 'custom_load': False})
model = model.to(device)
model = model.eval()
data_config = timm.data.resolve_model_data_config(model)

class TimmToHFWrapper(torch.nn.Module):

    def __init__(self, timm_model, data_cfg):
        super().__init__()
        self.model = timm_model

        class Config:
            pass
        self.config = Config()
        self.config.mean = data_cfg['mean']
        self.config.std = data_cfg['std']
        self.config.crop_pct = data_cfg['crop_pct']
        self.config.crop_mode = 'center'
        self.config.num_labels = timm_model.num_classes
        self.config.id2label = {i: str(i) for i in range(timm_model.num_classes)}

    def forward(self, x):
        logits = self.model(x)

        class Output:
            pass
        res = Output()
        res.logits = logits
        return res
wrapped_model = TimmToHFWrapper(model, data_config)
from inference.runner import evaluate_model
from inference.runtime import print_model_and_environment
output_dir = ARGS.output_dir
print_model_and_environment(model, output_dir)
evaluate_model(wrapped_model, dataset_name='imagenet-1k', output_dir=output_dir, batch_size=64, num_workers=8, input_resolution=data_config['input_size'])
