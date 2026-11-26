#!/usr/bin/env python3
"""Model-owned inference, logging, forward timing, COCO export and evaluation."""
from pathlib import Path
import argparse,datetime,importlib.util,json,os,platform,subprocess,sys,uuid
sys.dont_write_bytecode=True

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('_mask_entry_paths',HERE/'inference/paths.py')
PATHS=importlib.util.module_from_spec(spec);spec.loader.exec_module(PATHS)

def save(path,value):
    temporary=path.with_name(path.name+'.tmp')
    temporary.write_text(json.dumps(value,ensure_ascii=False,allow_nan=False,indent=2)+'\n')
    temporary.replace(path)

def main():
    args=sys.argv[1:]
    command=[sys.executable,'-u','-B',str(HERE/'inference/native.py'),*args]
    if '--no-env-switch' not in args:command.append('--no-env-switch')
    if any(x in args for x in ('--help','-h')):return subprocess.call(command)
    ready=HERE.parents[1]/'environment/deployment_ready.json'
    if not ready.is_file() or json.loads(ready.read_text()).get('complete') is not True:
        raise RuntimeError('Model/data migration is not verified yet; inference was not started.')
    if '--check' in args:return subprocess.call(command)
    config=json.loads((HERE/'inference/config.json').read_text())
    options=argparse.ArgumentParser(add_help=False)
    options.add_argument('--max-images','--max-samples',dest='limit',type=int,default=0)
    options.add_argument('--batch-size',type=int)
    selected,_=options.parse_known_args(args)
    run=datetime.datetime.now().strftime('%Y%m%d_%H%M%S')+'_'+uuid.uuid4().hex[:8]
    directory=PATHS.RESULT_DIR/'runs'/run;directory.mkdir(parents=True,exist_ok=False)
    status=directory/'status.json';log=directory/'stdout.log'
    env=dict(os.environ,MASK_RUN_ID=run,PYTHONNOUSERSITE='1',PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1')
    record={'state':'running','model':HERE.name,'started_at':datetime.datetime.now().astimezone().isoformat(),
            'command':['python',str((HERE/'inference.py').relative_to(PATHS.PROJECT_ROOT)),*args],
            'environment':{'name':Path(os.environ.get('VIRTUAL_ENV') or os.environ.get('CONDA_PREFIX') or sys.prefix).name,
                           'python_version':platform.python_version()},
            'input_scope':{'dataset':config['dataset'],'sample_limit':selected.limit},
            'batch_policy':config['batch_policy'],'requested_batch_override':selected.batch_size,
            'configured_prompt_batch_size':(selected.batch_size or config['prompt_batch_size']) if config['batch_policy']=='project_prompt_batch' else None,
            'dataset_root':os.path.relpath(PATHS.DATASET_ROOT,PATHS.PROJECT_ROOT),
            'weights':os.path.relpath(PATHS.WEIGHTS_PATH,PATHS.PROJECT_ROOT),
            'output_directory':os.path.relpath(directory,PATHS.PROJECT_ROOT),'log':'stdout.log'}
    save(status,record)
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
        if code==0:
            reference=json.loads((directory/'evaluation_bundle.json').read_text())
            bundle=(directory/reference['bundle']).resolve()
            if not bundle.is_relative_to(PATHS.DATASET_ROOT):raise ValueError('Evaluation package is outside the configured datasets')
            protocol=json.loads((bundle/'protocol.json').read_text())
            if protocol['model']!=HERE.name or protocol['run_id']!=run or protocol['complete'] is not True:
                raise ValueError('Wrong or incomplete evaluation package')
            timing=json.loads((directory/'inference_time.json').read_text())[HERE.name]
            result=json.loads((directory/'results.json').read_text())
            record['input_scope'].update(evaluation_units=protocol['sample_count'],timing_samples=timing['sample_count'])
            record.update(evaluation_bundle=os.path.relpath(bundle,PATHS.PROJECT_ROOT),metrics=result)
        record.update(state='passed' if code==0 else 'failed',exit_code=code,
                      finished_at=datetime.datetime.now().astimezone().isoformat())
        save(status,record)
        return code
    except BaseException as error:
        record.update(state='interrupted' if isinstance(error,KeyboardInterrupt) else 'failed',error=str(error))
        save(status,record);raise

if __name__=='__main__':raise SystemExit(main())
