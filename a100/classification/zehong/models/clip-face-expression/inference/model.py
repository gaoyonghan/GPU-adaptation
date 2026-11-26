from inference.options import ARGS
from inference.runtime import local_weight
import os
from datetime import datetime
from pathlib import Path
import torch
import torch.nn as nn
from PIL import Image
from transformers import AutoProcessor, CLIPVisionModel
from safetensors.torch import load_file
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
import requests
from inference.runner import evaluate_model
from inference.runtime import print_model_and_environment
REPO_ID = 'syntheticbot/clip-face-expression'
FILENAME = 'model.safetensors'
DEVICE = torch.device(ARGS.device)
CLASS_NAMES = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
NUM_CLASSES = len(CLASS_NAMES)

class ClipClassifier(nn.Module):

    def __init__(self, vision_model, num_classes):
        super(ClipClassifier, self).__init__()
        self.vision_model = vision_model
        self.classifier = nn.Linear(vision_model.config.hidden_size, num_classes)
        self.config = vision_model.config
        self.config.num_labels = num_classes
        self.config.id2label = {i: name for i, name in enumerate(CLASS_NAMES)}
        self.config.label2id = {name: i for i, name in enumerate(CLASS_NAMES)}
        self.config.mean = (0.48145466, 0.4578275, 0.40821073)
        self.config.std = (0.26862954, 0.26130258, 0.27577711)
        self.config.crop_mode = 'center'
        self.config.crop_pct = 1.0

    def forward(self, pixel_values):
        outputs = self.vision_model(pixel_values=pixel_values)
        image_features = outputs.pooler_output
        logits = self.classifier(image_features)
        return logits
print('Loading CLIP Vision Model...')
vision_model = CLIPVisionModel.from_pretrained(str(ARGS.vision_weights), local_files_only=True).to(DEVICE)
model = ClipClassifier(vision_model, NUM_CLASSES).to(DEVICE)
print(f'Downloading model weights from {REPO_ID}...')
model_path = str(local_weight())
state_dict = load_file(model_path, device='cpu')
# All 393 checkpoint keys and shapes match the complete classifier exactly.
model.load_state_dict(state_dict, strict=True)
model.eval()
output_dir = ARGS.output_dir
print_model_and_environment(model, output_dir)
evaluate_model(model=model, dataset_name='fer-2013', output_dir=output_dir, batch_size=64, num_workers=8)
