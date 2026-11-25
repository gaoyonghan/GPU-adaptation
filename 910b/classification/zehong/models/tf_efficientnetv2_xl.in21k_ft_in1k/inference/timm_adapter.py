"""Official timm eval transform and forward, without added AMP or dtype conversion."""
from timm.data import resolve_model_data_config,create_transform
class Processor:
    def __init__(self,model):
        self.config=resolve_model_data_config(model)
        self.transform=create_transform(**self.config,is_training=False)
    def __call__(self,images,return_tensors='pt'):
        return {'pixel_values':self.transform(images).unsqueeze(0)}
    def __str__(self):return str(self.transform)
def forward(model,inputs):return model(inputs['pixel_values'])
