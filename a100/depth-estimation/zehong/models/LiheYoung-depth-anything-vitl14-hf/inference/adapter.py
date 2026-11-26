"""模型官方加载和原项目预/后处理；不计算GT指标。"""
from pathlib import Path
import cv2,numpy as np,torch
import torch.nn.functional as F
class Adapter:
 def __init__(self,kind,weights,device,lotus_dtype='float16',attention=None):
  self.kind,self.device,self.processor=kind,device,None
  if kind.startswith('v2_'):
   from depth_anything_v2.dpt import DepthAnythingV2
   enc=kind[3:];cfg={'vitb':dict(features=128,out_channels=[96,192,384,768]),'vitl':dict(features=256,out_channels=[256,512,1024,1024])}
   self.model=DepthAnythingV2(encoder=enc,**cfg[enc]);self.model.load_state_dict(torch.load(weights/f'depth_anything_v2_{enc}.pth',map_location='cpu',weights_only=True))
  elif kind=='v1_native':
   from depth_anything.dpt import DepthAnything
   # The published V1 snapshot has pytorch_model.bin; newer HubMixin local loading expects safetensors.
   import json
   self.model=DepthAnything(json.loads((weights/'config.json').read_text()))
   if (weights/'model.safetensors').exists():
    from safetensors.torch import load_file
    state=load_file(str(weights/'model.safetensors'))
   else:state=torch.load(weights/'pytorch_model.bin',map_location='cpu',weights_only=True)
   self.model.load_state_dict(state,strict=True)
  elif kind=='lotus':
   import sys
   sys.path.insert(0,str(Path(__file__).parent/'Lotus'))
   from Lotus.pipeline import LotusGPipeline
   self.model=LotusGPipeline.from_pretrained(weights,torch_dtype=getattr(torch,lotus_dtype),local_files_only=True)
  else:
   from transformers import AutoImageProcessor,AutoModelForDepthEstimation
   self.processor=AutoImageProcessor.from_pretrained(weights,local_files_only=True)
   self.model=AutoModelForDepthEstimation.from_pretrained(weights,local_files_only=True)
   if attention:
    backbone=getattr(self.model,'backbone',None)
    if backbone is None or not hasattr(backbone,'set_attn_implementation'):raise RuntimeError('This ZoeDepth environment needs the validated set_attn_implementation API; use recorded Transformers 5.17 environment')
    backbone.set_attn_implementation(attention)
  self.model=self.model.to(device)
  if hasattr(self.model,'eval'):self.model.eval().requires_grad_(False)
 def prepare(self,rgb,name):
  kind=self.kind
  if kind.startswith('v2_') or kind=='v1_native':
   if kind=='v1_native':from depth_anything.util.transform import Resize,NormalizeImage,PrepareForNet
   else:from depth_anything_v2.util.transform import Resize,NormalizeImage,PrepareForNet
   d={'image':rgb/255.0}
   for fn in [Resize(width=518,height=518,resize_target=False,keep_aspect_ratio=True,ensure_multiple_of=14,resize_method='lower_bound',image_interpolation_method=cv2.INTER_CUBIC),NormalizeImage(mean=[.485,.456,.406],std=[.229,.224,.225]),PrepareForNet()]:d=fn(d)
   return torch.from_numpy(np.ascontiguousarray(d['image'])).float()
  if kind=='lotus':
   if rgb.shape[0]%8 or rgb.shape[1]%8:raise ValueError('Lotus native input_size=None requires dimensions divisible by 8')
   return torch.from_numpy(np.ascontiguousarray(rgb.transpose(2,0,1)))
  if kind=='zoe' and name in ['KITTI','NYUv2']:
   return torch.from_numpy(rgb.astype(np.float32)).permute(2,0,1)
  return self.processor(images=rgb,return_tensors='pt')['pixel_values'][0]
 def inputs(self,cpu,name):
  x=cpu.unsqueeze(0).to(self.device)
  if self.kind=='zoe' and name in ['KITTI','NYUv2']:x=(F.interpolate(x/255.,size=(384,512),mode='bilinear',align_corners=True)-.5)/.5
  if self.kind=='lotus':x=(x.float()/127.5-1).to(self.model.unet.dtype)
  # Cast floating inputs only to the model's actual parameter dtype, not a fixed FP32 policy.
  if self.kind!='lotus' and x.is_floating_point():x=x.to(next(self.model.parameters()).dtype)
  return x
 def predict(self,x,timer):
  if self.kind=='lotus':
   emb=torch.tensor([[1.,0.]],device=self.device);emb=torch.cat([emb.sin(),emb.cos()],-1).to(x.dtype)
   out=self.model(rgb_in=x,prompt='',num_inference_steps=1,timesteps=[999],task_emb=emb,processing_res=0,output_type='np')
   return out
  if self.kind.startswith('v2_') or self.kind=='v1_native':return timer.call(self.model,x)
  y=timer.call(self.model,pixel_values=x)
  if self.kind=='zoe':
   flipped=torch.flip(x,[-1])  # Outside device-event range.
   return y,timer.call(self.model,pixel_values=flipped)
  return y
 def postprocess(self,out,shape,name):
  if self.kind=='lotus':
   a=np.asarray(out.images)
   if a.ndim!=4 or a.shape[0]!=1:raise ValueError(f'Unexpected Lotus output {a.shape}')
   return cv2.resize(a[0].mean(-1),(shape[1],shape[0]),interpolation=cv2.INTER_LINEAR)
  if self.kind=='zoe':
   y,f=out
   if name in ['KITTI','NYUv2']:
    d=(y.predicted_depth+torch.flip(f.predicted_depth,[-1]))/2
    d=F.interpolate(d[:,None],size=shape,mode='bilinear',align_corners=True)[0,0]
   else:d=self.processor.post_process_depth_estimation(y,source_sizes=[shape],outputs_flipped=f)[0]['predicted_depth']
  else:
   native=self.kind.startswith('v2_') or self.kind=='v1_native'
   d=out if native else out.predicted_depth
   d=F.interpolate(d[:,None],size=shape,mode='bilinear' if native else 'bicubic',align_corners=native)[0,0]
  return d.float().cpu().numpy()
