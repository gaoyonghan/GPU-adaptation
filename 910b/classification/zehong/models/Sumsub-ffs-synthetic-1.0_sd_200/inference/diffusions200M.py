import timm
import torch
from torch import nn

class Model200M(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = timm.create_model('convnext_large_mlp.clip_laion2b_soup_ft_in12k_in1k_384',
                                       pretrained=False,
                                       num_classes=0)

        self.clf = nn.Sequential(
            nn.Linear(1536, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 2))

    def forward(self, image):
        image_features = self.model(image)
        return self.clf(image_features)