"""Model-owned native loading, preprocessing, forward and v4 serialization."""
import json
from pathlib import Path
import torch
from .utils.evaluate_model import evaluate_model
from .utils.protocol import checkpoint_fingerprints,processor_protocol
from .utils.print_model_info import print_model_and_environment
MODEL_ID='facebook/dino-vitb16'
FAMILY='hf'

class PrepareBatch:
 def __init__(self,processor,hf=False):self.processor,self.hf=processor,hf
 def __call__(self,images):
  if self.hf:
   samples=[self.processor(images=image.convert('RGB'),return_tensors='pt') for image in images]
   return {key:torch.cat([s[key] for s in samples]) for key in samples[0]}
  return {'pixel_values':torch.stack([self.processor(image.convert('RGB')) for image in images])}

def forward_fn(model,inputs):
 if FAMILY=='conch':return model.encode_image(inputs['pixel_values'],proj_contrast=False,normalize=False)
 if FAMILY=='vtp':return model.get_last_layer_feature(inputs['pixel_values'])
 if FAMILY=='uni':return model(inputs['pixel_values'])
 if FAMILY=='titan':return model(inputs['images'],inputs['coords'])
 return model(**inputs)

def select_output(outputs):
 if FAMILY=='hf':return outputs.last_hidden_state
 if FAMILY=='vtp':return torch.cat([outputs['cls_token'].unsqueeze(1),outputs['patch_tokens']],dim=1)
 return outputs

def extract(a):
 if a.device.startswith('npu'):
  import torch_npu
  if not torch.npu.is_available():raise RuntimeError('NPU unavailable; source CANN set_env.sh first')
  torch.npu.set_device(a.device)
 elif not torch.cuda.is_available():raise RuntimeError('CUDA unavailable')
 snapshot=a.weights
 if FAMILY=='hf':
  from transformers import AutoImageProcessor,AutoModel
  processor=AutoImageProcessor.from_pretrained(str(snapshot),local_files_only=True,use_fast=False)
  model=AutoModel.from_pretrained(str(snapshot),local_files_only=True,trust_remote_code=True)
  prepare=PrepareBatch(processor,True);preprocessing=processor_protocol(processor);output='last_hidden_state'
 elif FAMILY=='uni':
  import timm
  from timm.data import resolve_data_config,create_transform
  cfg=json.loads((snapshot/'config.json').read_text())
  kwargs=dict(cfg.get('model_args',{}))
  kwargs.update(init_values=1e-5,num_classes=0,dynamic_img_size=True)
  if MODEL_ID.endswith('UNI2-h'):
   kwargs.update(img_size=224,patch_size=14,depth=24,num_heads=24,embed_dim=1536,mlp_ratio=2.66667*2,no_embed_class=True,mlp_layer=timm.layers.SwiGLUPacked,act_layer=torch.nn.SiLU,reg_tokens=8)
  model=timm.create_model(cfg['architecture'],pretrained=False,pretrained_cfg=cfg.get('pretrained_cfg',{}),**kwargs)
  model.load_state_dict(torch.load(snapshot/'pytorch_model.bin',map_location='cpu',weights_only=True),strict=True)
  data=resolve_data_config(model.pretrained_cfg,model=model)
  processor=create_transform(**data,is_training=False)
  prepare=PrepareBatch(processor);preprocessing={'source':'official timm config','data_config':data,'transform':repr(processor)};output='image_embedding'
 elif FAMILY=='conch':
  from .conch.open_clip_custom import create_model_from_pretrained
  model,processor=create_model_from_pretrained('conch_ViT-B-16',checkpoint_path=str(snapshot/'pytorch_model.bin'))
  prepare=PrepareBatch(processor);preprocessing={'source':'official CONCH transform','transform':repr(processor),'proj_contrast':False,'normalize':False};output='image_embedding'
 elif FAMILY=='vtp':
  from .vtp.models.vtp_hf import VTPModel
  from torchvision import transforms as T
  model=VTPModel.from_pretrained(str(snapshot),local_files_only=True)
  processor=T.Compose([T.Resize((256,256)),T.ToTensor(),T.Normalize([.485,.456,.406],[.229,.224,.225])])
  prepare=PrepareBatch(processor);preprocessing={'source':'VTP official Quick Start','transform':repr(processor)};output='cls_and_patch_tokens'
 else:
  from .titan_native.configuration_titan import TitanConfig
  from .titan_native.modeling_titan import Titan
  from .titan_native.conch_v1_5 import build_conch
  from .titan_region import TitanImagePipeline,PrepareBatch as PrepareRegion
  titan=Titan.from_pretrained(str(snapshot),config=TitanConfig.from_pretrained(str(snapshot),local_files_only=True),local_files_only=True)
  conch,processor=build_conch(titan.conch_config,str(snapshot/'conch_v1_5_pytorch_model.bin'))
  model=TitanImagePipeline(titan,conch);prepare=PrepareRegion(processor)
  preprocessing={'source':'custom RGB 2x2 simulated region + official CONCH v1.5','grid':[2,2],'patch_size_lv0':512,'real_histology_magnification':False,'transform':repr(processor)};output='titan_simulated_region_embedding'
 model.to(a.device).eval()
 print_model_and_environment(model,a.output_dir)
 from .utils import evaluate_model as runner
 runner.MAX_SAMPLES=a.max_samples
 # Local location names differ across hosts; content fingerprints are authoritative.
 identity={'source':'huggingface','id':MODEL_ID,'revision':'local-content-v1','checkpoint_files':checkpoint_fingerprints(snapshot)}
 summary=evaluate_model(model,prepare,forward_fn,select_output,bundle_dir=a.bundle,output_dir=a.output_dir,model_info=identity,preprocessing_info=preprocessing,output_name=output,batch_size=a.batch_size,num_workers=a.num_workers,warmup_steps=a.warmup_steps)
 if summary['status']!='complete':raise RuntimeError('Invalid features: '+str(summary))
 print(json.dumps(summary,ensure_ascii=False,indent=2))
