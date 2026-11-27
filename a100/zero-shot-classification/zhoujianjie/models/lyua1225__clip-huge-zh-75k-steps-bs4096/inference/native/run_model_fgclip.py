import os,sys
from pathlib import Path
from model_paths import weight,NATIVE
sys.dont_write_bytecode=True
os.environ.update(USE_TF='0',USE_FLAX='0',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
sys.path.insert(0,str(NATIVE))
import importlib.util
_find=importlib.util.find_spec
def _safe_find(name,package=None):
    return None if name.split('.')[0] in {'flash_attn','deepspeed','transformer_engine'} else _find(name,package)
importlib.util.find_spec=_safe_find
import transformers.utils.import_utils as iu
import transformers.utils as tu
iu.is_flash_attn_2_available=lambda:False
tu.is_flash_attn_2_available=lambda:False
try:
    import evaluate_b01 as b,validate_stage2_models as v
    b.model_dir=v.local_dir=weight
except ModuleNotFoundError as error:
    if error.name not in {'evaluate_b01','validate_stage2_models'}:raise
if '--check-imports' in sys.argv:raise SystemExit(0)
raise RuntimeError('This model-local bootstrap only configures native imports; it is not a model router')
