#!/usr/bin/env python3
"""Self-contained model entry; --smoke never authorizes a full evaluation."""
from pathlib import Path
import argparse,json,os,subprocess,sys
MODEL_DIR=Path(__file__).resolve().parent
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--weights',type=Path,default=Path(os.environ.get('ZERO_SHOT_WEIGHTS',MODEL_DIR/'weights')))
    p.add_argument('--weights-root',type=Path,default=Path(os.environ.get('ZERO_SHOT_WEIGHTS_ROOT',MODEL_DIR.parent)))
    p.add_argument('--dataset-root',type=Path,default=Path(os.environ.get('ZERO_SHOT_DATASET_ROOT',MODEL_DIR.parents[3]/'dataset')))
    p.add_argument('--output-root',type=Path,default=Path('result/run_001'))
    p.add_argument('--smoke',action='store_true')
    p.add_argument('--max-images',type=int,default=8)
    p.add_argument('--check-inputs',action='store_true')
    p.add_argument('--show-command',action='store_true')
    args=p.parse_args()
    cfg=json.loads((MODEL_DIR/'inference/delivery_config.json').read_text())
    out=(args.output_root if args.output_root.is_absolute() else MODEL_DIR/args.output_root).resolve()
    if not out.is_relative_to((MODEL_DIR/'result').resolve()):p.error('output-root must belong to this model result/')
    if args.max_images<1:p.error('max-images must be positive')
    command=[sys.executable,'-u',str(MODEL_DIR/'inference/native_entry.py')]
    resources={'weights':str(args.weights.resolve()),'weights_root':str(args.weights_root.resolve()),'dataset_root':str(args.dataset_root.resolve())}
    missing=[]
    for root,names in [('weights',cfg['required_weights']),('dataset_root',cfg['required_data'])]:
        for name in names:
            if not (Path(resources[root])/name).exists():missing.append(str(Path(resources[root])/name))
    report={'model':cfg['model_id'],'resources':resources,'missing':missing,'command':command,'task':cfg['native_task'],'public_evaluator':cfg['public_evaluator'],'smoke_supported':cfg['bounded_smoke_supported']}
    print(json.dumps(report,ensure_ascii=False),flush=True)
    if args.show_command:return 0
    if args.check_inputs:return 2 if missing else 0
    if missing:return 2
    if args.smoke and not cfg['bounded_smoke_supported']:
        print('Bounded smoke is not yet implemented for this native protocol; full run was NOT started.',file=sys.stderr);return 3
    out.mkdir(parents=True,exist_ok=False)
    env={**os.environ,'ZERO_SHOT_WEIGHTS':resources['weights'],'ZERO_SHOT_WEIGHTS_ROOT':resources['weights_root'],'ZERO_SHOT_DATASET_ROOT':resources['dataset_root'],'ZERO_SHOT_RUN_DIR':str(out),'ZERO_SHOT_SMOKE_LIMIT':str(args.max_images if args.smoke else 0),'PYTHONDONTWRITEBYTECODE':'1','HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1'}
    env['PYTHONPATH']=str(MODEL_DIR/'inference/native')+os.pathsep+env.get('PYTHONPATH','')
    env.update(HF_HOME=str(out/'cache'),TORCH_HOME=str(out/'cache'),HF_MODULES_CACHE=str(out/'cache/modules'),TMPDIR=str(out/'tmp'))
    cache_vars={'HF_HOME':str(out/'cache/huggingface'),'HF_HUB_CACHE':str(out/'cache/huggingface/hub'),'HF_MODULES_CACHE':str(out/'cache/huggingface/modules'),'TORCH_HOME':str(out/'cache/torch'),'XDG_CACHE_HOME':str(out/'cache'),'CUDA_CACHE_PATH':str(out/'cache/cuda'),'TRITON_CACHE_DIR':str(out/'cache/triton'),'TORCH_EXTENSIONS_DIR':str(out/'cache/torch_extensions'),'TMPDIR':str(out/'tmp')}
    (out/'tmp').mkdir(exist_ok=True)
    env.update(cache_vars)
    def save(name,obj):(out/name).write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    save('config.json',{**report,'smoke_only':args.smoke,'max_images':args.max_images if args.smoke else None,'python':sys.version})
    save('status.json',{'state':'running'})
    with (out/'execution.log').open('w',encoding='utf-8') as log:
        proc=subprocess.Popen(command,cwd=MODEL_DIR/'inference/native',env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,encoding='utf-8',errors='replace')
        for line in proc.stdout:print(line,end='',flush=True);log.write(line);log.flush()
        code=proc.wait()
    save('status.json',{'state':('smoke_completed' if args.smoke else 'native_completed') if code==0 else 'failed','exit_code':code,'public_evaluator':cfg['public_evaluator']})
    return code
if __name__=='__main__':raise SystemExit(main())
