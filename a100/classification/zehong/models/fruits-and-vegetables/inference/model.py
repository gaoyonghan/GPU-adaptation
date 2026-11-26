from inference.options import ARGS
from inference.runtime import local_weight
import os
import torch
from transformers import AutoModelForImageClassification
from datetime import datetime
from pathlib import Path
from inference.runner import evaluate_model
from inference.runtime import print_model_and_environment
device = torch.device(ARGS.device)
model = AutoModelForImageClassification.from_pretrained(str(ARGS.weights), local_files_only=True).to(device)
model.config.mean = [0.485, 0.456, 0.406]
model.config.std = [0.229, 0.224, 0.225]
model.config.crop_mode = 'squash'
model.config.crop_pct = 1.0
output_dir = ARGS.output_dir
print_model_and_environment(model, output_dir)
evaluate_model(model, dataset_name='fruits-and-vegetables', output_dir=output_dir, batch_size=64, num_workers=8)
