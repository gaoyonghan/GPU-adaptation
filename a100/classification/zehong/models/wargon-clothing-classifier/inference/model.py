from inference.options import ARGS
from inference.runtime import local_weight
import os
from datetime import datetime
from pathlib import Path
import torch
from transformers import AutoImageProcessor, AutoModelForImageClassification
device = torch.device(ARGS.device)
print(f'Using device: {device}')
model_id = 'wargoninnovation/wargon-clothing-classifier'
dataset_name = 'clothingdatasetsecondhand'
input_resolution = (3, 224, 224)
processor = AutoImageProcessor.from_pretrained(str(ARGS.weights), local_files_only=True)
model = AutoModelForImageClassification.from_pretrained(str(ARGS.weights), local_files_only=True)
model.to(device)
model.eval()
from inference.runner import evaluate_model
from inference.runtime import print_model_and_environment
output_dir = ARGS.output_dir
None
print_model_and_environment(model, output_dir)
evaluate_model(model=model, dataset_name=dataset_name, output_dir=output_dir, batch_size=64, num_workers=8, input_resolution=input_resolution, processor=processor)
