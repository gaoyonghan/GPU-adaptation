from types import SimpleNamespace
class TimmProcessor:
    def __init__(self,model):
        from timm.data import resolve_model_data_config,create_transform
        self.transform=create_transform(**resolve_model_data_config(model),is_training=False)
    def __call__(self,images,return_tensors='pt'):return {'pixel_values':self.transform(images).unsqueeze(0)}
class CifarProcessor:
    def __init__(self):
        from torchvision import transforms as T
        self.transform=T.Compose([T.Resize((32,32)),T.ToTensor(),T.Normalize((.5071,.4867,.4408),(.2675,.2565,.2761))])
    def __call__(self,images,return_tensors='pt'):return {'pixel_values':self.transform(images).unsqueeze(0)}
class FastaiProcessor:
    def __init__(self,dls):self.dls=dls.cpu()
    def __call__(self,images,return_tensors='pt'):
        import numpy as np
        dl=self.dls.test_dl([np.asarray(images)],bs=1,num_workers=0,shuffle=False)
        batch=dl.one_batch()
        return {'pixel_values':batch[0].as_subclass(__import__('torch').Tensor).cpu()}
def tensor_forward(model,inputs):return model(inputs['pixel_values'])
