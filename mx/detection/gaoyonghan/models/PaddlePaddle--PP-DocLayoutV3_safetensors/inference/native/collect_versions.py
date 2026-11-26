import importlib.metadata as metadata
import json
import platform
import sys
versions={}
for name in ['torch','torchvision','transformers','huggingface-hub','ultralytics','paddleocr','paddlex','paddlepaddle-gpu','onnxruntime','pycocotools','numpy','pandas','scipy','Pillow','opencv-python','opencv-python-headless','pyarrow','dill','omegaconf','Cython','tokenizers','safetensors']:
    try:versions[name]=metadata.version(name)
    except metadata.PackageNotFoundError:pass
try:
    import torch
    versions['torch_cuda_wheel']=torch.version.cuda
except ImportError:pass
print(json.dumps({'python':platform.python_version(),'executable':sys.executable,'versions':versions}))
