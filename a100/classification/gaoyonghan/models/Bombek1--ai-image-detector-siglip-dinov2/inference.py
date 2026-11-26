#!/usr/bin/env python3
"""Public inference entry: run the native workflow, then prepare standard output."""
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import subprocess
import sys

MODEL=Path(__file__).resolve().parent
sys.path.insert(0,str(MODEL/'inference'))
from common import parser,runtime_options,model_spec,native_command,command_details,execution_environment,save,preflight
CONFIG=json.loads((MODEL/'inference/entrypoint.json').read_text())

def main():
    p=parser();args=p.parse_args();options=runtime_options(args,p,model_spec());root=Path(options['output_root'])
    if options['phase']=='preflight':
        print(json.dumps(preflight(options),ensure_ascii=False,indent=2));return
    if args.show_command:
        print(json.dumps(command_details(options),ensure_ascii=False,indent=2));return
    env=execution_environment(options);manifest=root/'inference_output.json'
    if not args.prepare_only and manifest.exists():
        p.error('This run directory already has a manifest; select a new --output-root')
    try:
        if not args.prepare_only:
            save(manifest,dict(state='running',task='classification',sequence=CONFIG['sequence'],phase=options['phase'],bundles=[]))
            save(root/'run_command.json',dict(command_details(options),started_at=datetime.now(timezone.utc).isoformat(),
              python=sys.executable,python_version=sys.version,conda_environment=os.environ.get('CONDA_DEFAULT_ENV')))
            with (root/'inference.log').open('w') as log:
                subprocess.run(native_command(options),cwd=MODEL,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
        subprocess.run([sys.executable,'-B',str(MODEL/'inference/prepare.py'),'--run-dir',str(root),'--phase',options['phase']],cwd=MODEL,env=env,check=True)
        result=json.loads(manifest.read_text())
        if result.get('state')!='completed':raise RuntimeError('Standardized output preparation did not complete')
        print('Standardized inference output: '+str(manifest))
    except BaseException as exc:
        if not manifest.exists() or json.loads(manifest.read_text()).get('state')!='completed':
            save(manifest,dict(state='failed',task='classification',sequence=CONFIG['sequence'],phase=options['phase'],bundles=[],reason=str(exc)))
        raise

if __name__=='__main__':main()

