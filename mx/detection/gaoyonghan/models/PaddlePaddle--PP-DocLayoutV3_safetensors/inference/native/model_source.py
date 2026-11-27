"""Use the PP framework installed in the activated environment."""
from pathlib import Path
import sys
import transformers
SOURCE_ROOT = Path(transformers.__file__).resolve().parent.parent
assert transformers.__version__ == '5.3.0', transformers.__version__
assert str(SOURCE_ROOT).startswith(str(Path(sys.prefix).resolve()) + '/')
