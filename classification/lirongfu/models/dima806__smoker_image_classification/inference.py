#!/usr/bin/env python3
"""Standalone inference with model-owned logs, state and configurable paths."""
from pathlib import Path
import argparse,datetime,importlib.util,json,os,platform,subprocess,sys,uuid
HERE=Path(__file__).resolve().parent
s=importlib.util.spec_from_file_location('_model_paths',HERE/'inference/paths.py')
paths=importlib.util.module_from_spec(s);s.loader.exec_module(paths)

def save(path,value):
    tmp=path.with_name(path.name+'.tmp')
    tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')
    tmp.replace(path)

def main():
    command=[sys.executable,'-u','-B',str(HERE/'inference/native.py'),*sys.argv[1:]]
    if any(x in sys.argv[1:] for x in ('--check','--help','-h')):
        return subprocess.call(command)
    run=datetime.datetime.now().strftime('%Y%m%d_%H%M%S')+'_'+uuid.uuid4().hex[:8]
    directory=paths.RESULT_DIR/'runs'/run;directory.mkdir(parents=True,exist_ok=False)
    status=directory/'status.json';log=directory/'stdout.log'
    env=dict(os.environ,CLASSIFICATION_RUN_ID=run,PYTHONNOUSERSITE='1',PYTHONDONTWRITEBYTECODE='1')
    options=argparse.ArgumentParser(add_help=False)
    options.add_argument('--image')
    options.add_argument('--max-images','--max-samples',dest='limit',type=int,default=0)
    options.add_argument('--checkpoint')
    selected,_=options.parse_known_args()
    scope={'selection':'single image' if selected.image else 'configured dataset',
           'max_images':selected.limit,'image':selected.image}
    weight=paths.project_path(selected.checkpoint) if selected.checkpoint else paths.WEIGHTS_PATH
    record={'state':'running','model':HERE.name,
            'command':['python',str((HERE/'inference.py').relative_to(paths.PROJECT_ROOT)),*sys.argv[1:]],
            'environment':{'name':Path(os.environ.get('VIRTUAL_ENV') or os.environ.get('CONDA_PREFIX') or sys.prefix).name,
                           'python_version':platform.python_version()},
            'input_scope':scope,'output_directory':os.path.relpath(directory,paths.PROJECT_ROOT),
            'dataset_root':os.path.relpath(paths.DATASET_ROOT,paths.PROJECT_ROOT),
            'weights':os.path.relpath(weight,paths.PROJECT_ROOT),
            'log':'stdout.log'}
    save(status,record)
    code=None
    try:
        with log.open('xb') as stream:
            child=subprocess.Popen(command,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,env=env)
            try:
                while True:
                    block=os.read(child.stdout.fileno(),65536)
                    if not block:break
                    stream.write(block);stream.flush()
                    sys.stdout.buffer.write(block);sys.stdout.buffer.flush()
                code=child.wait()
            except BaseException:
                child.terminate()
                try:child.wait(timeout=15)
                except subprocess.TimeoutExpired:child.kill();child.wait()
                raise
        record.update(state='passed' if code==0 else 'failed',exit_code=code)
        reference=directory/'evaluation_bundle.json'
        if reference.is_file():
            marker=json.loads(reference.read_text())
            bundle=(directory/marker['bundle']).resolve()
            if not bundle.is_relative_to(paths.DATASET_ROOT):
                raise ValueError('Evaluation package must belong to configured shared dataset root')
            record['evaluation_bundle']=os.path.relpath(bundle,paths.PROJECT_ROOT)
            p=json.loads((bundle/'protocol.json').read_text())
            record['input_scope'].update(inference_images=p.get('inference_sample_count'),
                                        evaluated_images=p.get('prediction_count'),max_images=p.get('sample_limit'))
    except BaseException as error:
        record.update(state='interrupted' if isinstance(error,KeyboardInterrupt) else 'failed',error=str(error))
        save(status,record);raise
    save(status,record)
    return code

if __name__=='__main__':raise SystemExit(main())
