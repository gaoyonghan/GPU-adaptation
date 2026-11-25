import argparse, hashlib, json, re, os
from pathlib import Path
MODEL_DIR = Path(__file__).resolve().parent
def main():
 p=argparse.ArgumentParser()
 p.add_argument('--bundle',type=Path)
 p.add_argument('--weights',type=Path,default=Path('weights'))
 p.add_argument('--output-root','--output-dir',type=Path,default=Path('result/run_001'))
 p.add_argument('--max-samples',type=int)
 a=p.parse_args()
 # Asset paths are model-relative; absolute paths remain supported.
 for _name in ('weights', 'dataset_root', 'dataset', 'bundle', 'vision_weights', 'hf_cache', 'image'):
     _path = getattr(a, _name, None)
     if isinstance(_path, Path) and not _path.is_absolute():
         setattr(a, _name, (MODEL_DIR / _path).resolve())
 root=Path(__file__).resolve().parent
 cfg=json.loads((root/'model.json').read_text())
 import torch,timm
 from torchvision import transforms
 if cfg['device'].startswith('npu'):
  import torch_npu
  torch.npu.matmul.allow_hf32=False
  torch.npu.conv.allow_hf32=False
 else:
  torch.backends.cuda.matmul.allow_tf32=False
  torch.backends.cudnn.allow_tf32=False
 from inference.utils import evaluate_model as runner
 checkpoint=a.weights if a.weights.is_file() else a.weights/'backbone.pt'
 model=timm.create_model(cfg['architecture'],pretrained=False,num_classes=0)
 for block in getattr(model,'blocks',[]):
  block.attn.fused_attn=False
 model.load_state_dict(torch.load(checkpoint,map_location='cpu',weights_only=True),strict=True)
 model.to(cfg['device']).eval()
 out=root/a.output_root
 out.mkdir(parents=True,exist_ok=False)
 runner.MAX_SAMPLES=a.max_samples
 transform=transforms.Compose([transforms.Resize(256),transforms.CenterCrop(224),transforms.ToTensor(),transforms.Normalize([.485,.456,.406],[.229,.224,.225])])
 with checkpoint.open('rb') as f:
  digest=hashlib.file_digest(f,'sha256').hexdigest() if hasattr(hashlib,'file_digest') else hashlib.sha256(f.read()).hexdigest()
 result=runner.evaluate_model(model,lambda images:torch.stack([transform(im.convert('RGB')) for im in images]),lambda m,x:m(x),lambda x:x,bundle_dir=a.bundle or (root / cfg['bundle']).resolve(),output_dir=out,model_info={'source':'Xiaojie0903/genview_pretrained_models','id':root.name,'revision':cfg['source'],'checkpoint_files':{'backbone.pt':digest}},preprocessing_info={'resize':256,'crop':224,'interpolation':'bilinear','mean':[.485,.456,.406],'std':[.229,.224,.225]},output_name='backbone_embedding',batch_size=8,num_workers=0,warmup_steps=1)
 if result['status']!='complete':raise RuntimeError(result['status'])
 (out/'inference_output.json').write_text(json.dumps({'state':'completed','task':'image-feature-extraction'}))
if __name__=='__main__':main()

