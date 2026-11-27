import capture_environment
from pathlib import Path
import sys,os,runpy
from model_paths import prepare_metadata,NATIVE
prepare_metadata()
sys.path.insert(0,str(NATIVE))
sys.argv=[str(NATIVE/'run_openclip_native.py')]+['openai/clip-vit-large-patch14']+(['--small'] if int(os.environ.get('ZERO_SHOT_SMOKE_LIMIT','0')) else [])
try:
 runpy.run_path(sys.argv[0],run_name='__main__')
except SystemExit as error:
 if error.code:raise
from convert_native import main
main()
