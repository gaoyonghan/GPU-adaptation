"""Translate deprecated training-only Rotate option names; original checkpoint config stays unchanged."""
from contextlib import contextmanager
from pathlib import Path
import tempfile,inspect,yaml
@contextmanager
def compatible_config(path):
    import albumentations as A
    cfg=yaml.safe_load(Path(path).read_text())
    transforms=cfg.get('data',{}).get('init_args',{}).get('train_transform',[])
    changed=False
    for transform in transforms if isinstance(transforms,list) else []:
        if isinstance(transform,dict) and transform.get('class_path')=='ToTensorV2':
            transform['class_path']='albumentations.pytorch.ToTensorV2';changed=True
    params=inspect.signature(A.Rotate.__init__).parameters
    for transform in transforms if isinstance(transforms,list) else []:
        if not isinstance(transform,dict) or str(transform.get('class_path','')).split('.')[-1]!='Rotate':continue
        args=transform.get('init_args',{})
        for old,new in [('value','fill'),('mask_value','fill_mask')]:
            if old in args and old not in params and new in params:
                if new in args:raise ValueError('Ambiguous old/new Rotate option')
                args[new]=args.pop(old);changed=True
    if not changed:
        yield str(path);return
    with tempfile.TemporaryDirectory(prefix='granite_inference_config_') as folder:
        target=Path(folder)/'config.yaml';target.write_text(yaml.safe_dump(cfg,sort_keys=False))
        print('Config compatibility: training-only Rotate value/mask_value renamed to fill/fill_mask; prediction preprocessing unchanged',flush=True)
        yield str(target)
