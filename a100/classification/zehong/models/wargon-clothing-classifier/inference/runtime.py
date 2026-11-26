from pathlib import Path
from inference.options import ARGS
def local_weight():
 p=ARGS.weights
 if p.is_file():return p
 names=['model.safetensors','pytorch_model.safetensors','pytorch_model.bin','midjourney5M.pt']
 found=next((p/n for n in names if (p/n).is_file()),None)
 if found is None:raise FileNotFoundError('No local checkpoint in '+str(p))
 return found
def print_model_and_environment(*args,**kwargs):pass
