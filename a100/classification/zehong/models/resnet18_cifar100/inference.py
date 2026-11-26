MODEL_ID='edadaltocg/resnet18_cifar100'
KIND='cifar'
DATA_NAME='cifar100_set'
CIFAR_NAMES=['apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle', 'bicycle', 'bottle', 'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel', 'can', 'castle', 'caterpillar', 'cattle', 'chair', 'chimpanzee', 'clock', 'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup', 'dinosaur', 'dolphin', 'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster', 'house', 'kangaroo', 'keyboard', 'lamp', 'lawn_mower', 'leopard', 'lion', 'lizard', 'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain', 'mouse', 'mushroom', 'oak_tree', 'orange', 'orchid', 'otter', 'palm_tree', 'pear', 'pickup_truck', 'pine_tree', 'plain', 'plate', 'poppy', 'porcupine', 'possum', 'rabbit', 'raccoon', 'ray', 'road', 'rocket', 'rose', 'sea', 'seal', 'shark', 'shrew', 'skunk', 'skyscraper', 'snail', 'snake', 'spider', 'squirrel', 'streetcar', 'sunflower', 'sweet_pepper', 'table', 'tank', 'telephone', 'television', 'tiger', 'tractor', 'train', 'trout', 'tulip', 'turtle', 'wardrobe', 'whale', 'willow_tree', 'wolf', 'woman', 'worm']
"""仅推理和标准评分包准备；准确率由任务evaluate.py独立计算。"""
import argparse,json,sys,os
from pathlib import Path
from types import SimpleNamespace
from datetime import datetime
HERE=Path(__file__).resolve().parent
PROJECT=HERE.parents[3]
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--weights',type=Path,default=HERE/'weights')
    p.add_argument('--dataset-root',type=Path,default=PROJECT/'dataset'/DATA_NAME)
    p.add_argument('--output-root', '--output-dir', dest='output_dir', type=Path, default=HERE/'result'/'run_001', help='Run directory; relative paths resolve against this model folder (--output-dir is a compatibility alias)')
    p.add_argument('--batch-size',type=int,default=64)
    p.add_argument('--num-workers', '--workers',type=int,default=0 if KIND=='pet' else 8)
    p.add_argument('--warmup-steps', '--warmup',type=int,default=3)
    p.add_argument('--max-samples',type=int)
    p.add_argument('--label-mapping',type=Path,default=HERE/'inference/map22kto1k.txt')
    a = p.parse_args()
    # Asset paths are model-relative; absolute paths remain supported.
    for _name in ('weights', 'dataset_root', 'dataset', 'bundle', 'vision_weights', 'hf_cache', 'image'):
        _path = getattr(a, _name, None)
        if isinstance(_path, Path) and not _path.is_absolute():
            setattr(a, _name, (HERE / _path).resolve())
    if not a.output_dir.is_absolute():
        a.output_dir = HERE / a.output_dir
    a.output_dir = a.output_dir.resolve()
    if a.output_dir.exists():
        p.error('Output directory already exists; use a new --output-root to preserve previous results')
    if a.max_samples is not None and a.max_samples<1:p.error('max-samples must be positive')
    import torch
    if not torch.cuda.is_available():raise RuntimeError('CUDA unavailable')
    device=torch.device('cuda:0');torch.cuda.set_device(device)
    from inference.adapter import TimmProcessor,CifarProcessor,FastaiProcessor,tensor_forward
    forward=None
    if KIND=='pet':
        from fastai.learner import load_learner
        learner=load_learner(a.weights/'model.pkl',cpu=True)
        vocab=list(learner.dls.vocab)
        processor=FastaiProcessor(learner.dls)
        model=learner.model.to(device).eval()
        model.config=SimpleNamespace(id2label=dict(enumerate(vocab)))
        forward=tensor_forward
    elif KIND in ['timm','cifar']:
        import timm
        if KIND=='cifar':
            # Official detectors _create_resnet_small construction (same as validated 910B).
            model=timm.create_model('resnet18',pretrained=False,num_classes=100)
            model.conv1=torch.nn.Conv2d(3,64,kernel_size=3,stride=1,padding=1,bias=False)
            model.maxpool=torch.nn.Identity()
            checkpoint=a.weights/'pytorch_model.bin' if a.weights.is_dir() else a.weights
            model.load_state_dict(torch.load(checkpoint,map_location='cpu',weights_only=True))
            processor=CifarProcessor()
            names=CIFAR_NAMES
        else:
            checkpoint=next((a.weights/name for name in ['model.safetensors','pytorch_model.bin'] if (a.weights/name).is_file()),a.weights/'model.safetensors') if a.weights.is_dir() else a.weights
            model=timm.create_model(MODEL_ID,pretrained=True,pretrained_cfg_overlay={'source':'','file':str(checkpoint.absolute()),'custom_load':False})
            processor=TimmProcessor(model)
            names=[f'ImageNet index {i}' for i in range(model.num_classes)]
        model=model.to(device).eval()
        model.config=SimpleNamespace(id2label=dict(enumerate(names)))
        forward=tensor_forward
    else:
        from transformers import AutoImageProcessor,AutoModelForImageClassification
        processor=AutoImageProcessor.from_pretrained(a.weights,local_files_only=True)
        model=AutoModelForImageClassification.from_pretrained(a.weights,local_files_only=True).to(device).eval()
    mapping=None
    if KIND=='22k':
        from inference.label_mapping import _build_imagenet1k_to_22k_mapping
        mapping,_=_build_imagenet1k_to_22k_mapping(model,str(a.label_mapping))
    from inference.data import load
    dataset,categories,dataset_info=load(a.dataset_root,model,KIND,a.max_samples,mapping)
    model_info={'id':MODEL_ID,'weights':str(a.weights.resolve()),'parameter_dtypes':sorted({str(p.dtype) for p in model.parameters()}),'device':str(device)}
    a.output_dir.mkdir(parents=True,exist_ok=False)
    from inference.run_inference import run_inference
    from inference.prepare import publish
    kwargs={} if forward is None else {'forward_fn':forward}
    (a.output_dir/'model_structure.txt').write_text(str(model))
    import importlib.metadata as metadata
    versions={}
    for package in ['torch','torch_npu','torchvision','transformers','timm','fastai','detectors','numpy','datasets','Pillow']:
        try:versions[package]=metadata.version(package)
        except metadata.PackageNotFoundError:pass
    (a.output_dir/'model_environment.json').write_text(json.dumps({'python':sys.executable,'packages':versions,'device':str(device)},indent=2))
    try:
        run_inference(model,processor,dataset,categories,a.output_dir,batch_size=a.batch_size,num_workers=a.num_workers,warmup_steps=a.warmup_steps,model_info=model_info,dataset_info=dataset_info,**kwargs)
        publish(a.output_dir,model_info,dataset_info)
    except Exception:
        (a.output_dir/'inference_output.json').write_text(json.dumps({'state':'failed','task':'classification'}));raise
    print(a.output_dir)
if __name__=='__main__':main()
