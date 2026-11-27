import capture_environment
import os
os.environ.update(USE_TF="0",USE_FLAX="0",HF_HUB_OFFLINE="1",TRANSFORMERS_OFFLINE="1")
import importlib.util
_original_find_spec = importlib.util.find_spec
_disabled_optional = {"flash_attn", "deepspeed", "transformer_engine"}
def _find_spec(name, package=None):
    if name.split(".")[0] in _disabled_optional:
        return None
    return _original_find_spec(name, package)
importlib.util.find_spec = _find_spec
import transformers.utils.import_utils as iu
import transformers.utils as tu
# Broken optional system CUDA extension; this run does not request flash_attention_2.
iu.is_flash_attn_2_available=lambda:False
tu.is_flash_attn_2_available=lambda:False

from pathlib import Path
from shared.handoff_runtime import inference_main
raise SystemExit(inference_main(Path(__file__).resolve().parent.parent))
