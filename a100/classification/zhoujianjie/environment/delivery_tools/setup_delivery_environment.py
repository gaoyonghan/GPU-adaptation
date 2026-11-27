"""Copy observed dependency overlays into a NEW isolated validation environment."""
from pathlib import Path
import json,sys,subprocess,shutil,argparse
root=Path('/data/zhoujianjie/delivery_validation_20261126')
p=argparse.ArgumentParser();p.add_argument('--profile',default='zeroshot-main');a=p.parse_args()
allowed={'zeroshot-main','zeroshot-metaclip2','zeroshot-onealign','zeroshot-mme5','zeroshot-fgclip','zeroshot-llm2clip'}
assert a.profile in allowed
venv=root/'environments'/a.profile
if not (venv/'bin/python').exists():subprocess.run([sys.executable,'-m','venv','--copies','--without-pip','--system-site-packages',str(venv)],check=True)
paths=[]
for name in ([a.profile] if a.profile=='zeroshot-main' else [a.profile,'zeroshot-main']):
    source=Path('/data/zhoujianjie/.venvs')/name/'lib/python3.10/site-packages'
    target=root/'dependency_overlays'/name
    if not (target/'.copy_complete').exists():
        shutil.copytree(source,target,dirs_exist_ok=True,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
        (target/'.copy_complete').write_text(str(source)+'\n')
    paths.append(str(target))
source=Path('/data/zhoujianjie/environment_setup/runtime_overlays/openclip')
if source.is_dir():
    target=root/'dependency_overlays/openclip'
    if not (target/'.copy_complete').exists():
        shutil.copytree(source,target,dirs_exist_ok=True,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
        (target/'.copy_complete').write_text(str(source)+'\n')
    paths.append(str(target))
site=venv/'lib/python3.10/site-packages'
(site/'00-delivery-overlays.pth').write_text('\n'.join(paths)+'\n')
env=__import__('os').environ.copy();env.update(PYTHONPATH=__import__('os').pathsep.join(paths),USE_TF='0',USE_FLAX='0',PYTHONDONTWRITEBYTECODE='1')
# The shared storage disallows execve of a copied interpreter. Keep dependencies
# independent, and invoke the existing system interpreter without modifying it.
profile={'profile':a.profile,'python':sys.executable,'PYTHONPATH':env['PYTHONPATH'],'copied_interpreter_executable':__import__('os').access(venv/'bin/python',__import__('os').X_OK)}
(root/'profiles').mkdir(exist_ok=True);(root/'profiles'/(a.profile+'.json')).write_text(json.dumps(profile,indent=2)+'\n')
probe=subprocess.run([sys.executable,'-c','import sys,torch,transformers;print(sys.version);print(torch.__version__,transformers.__version__);print(transformers.__file__)'],env=env,capture_output=True,text=True)
print(json.dumps({**profile,'probe_exit':probe.returncode,'stdout':probe.stdout,'stderr':probe.stderr},ensure_ascii=False))
raise SystemExit(probe.returncode)
