import timm,json
def load_model(weights):
 weights=weights if weights.is_file() else next((weights/n for n in ['model.safetensors','pytorch_model.bin'] if (weights/n).is_file()),None)
 if weights is None:raise FileNotFoundError('Local timm checkpoint missing')
 model=timm.create_model('vit_tiny_patch16_224.augreg_in21k_ft_in1k',pretrained=True,pretrained_cfg_overlay={'file':str(weights.absolute()),'custom_load':False})
 if model.num_classes!=1000:raise ValueError('Expected 1000-class ImageNet head')
 return model,None
