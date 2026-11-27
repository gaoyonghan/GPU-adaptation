from pathlib import Path
import sys
import ultralytics
SOURCE_ROOT=Path(ultralytics.__file__).resolve().parent.parent
assert SOURCE_ROOT.is_relative_to(Path(sys.prefix).resolve())
assert ultralytics.__version__=='8.1.34'
