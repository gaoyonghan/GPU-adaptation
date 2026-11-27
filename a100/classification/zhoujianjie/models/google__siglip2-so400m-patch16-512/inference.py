#!/usr/bin/env python3
"""Model-local entry. --check-inputs never imports torch or starts GPU work."""
from pathlib import Path
import argparse,json,os,subprocess,sys
from datetime import datetime
MODEL_DIR=Path(__file__).resolve().parent
def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check-inputs',action='store_true')
    parser.add_argument('--weights',type=Path,default=Path(os.environ.get('ZERO_SHOT_WEIGHTS',MODEL_DIR/'weights')))
    parser.add_argument('--dataset-root',type=Path,default=Path(os.environ.get('ZERO_SHOT_DATASET_ROOT',MODEL_DIR.parents[3]/'dataset')))
    parser.add_argument('--smoke',action='store_true')
    parser.add_argument('--max-images',type=int,default=8)
    parser.add_argument('--output-root',type=Path)
    args,rest=parser.parse_known_args()
    if args.max_images<1:parser.error('max-images must be positive')
    cfg=json.loads((MODEL_DIR/'model_config.json').read_text())
    status=json.loads((MODEL_DIR/'inference/submission_status.json').read_text())
    weights=args.weights.resolve(); datasets=args.dataset_root.resolve()
    missing=[]
    if not (weights/'download_receipt.json').is_file():missing.append(str(weights/'download_receipt.json'))
    required=cfg.get('submission_required_dataset_directories',[])
    for relative in required:
        if not (datasets/relative).is_dir():missing.append(str(datasets/relative))
    print(json.dumps({'model':cfg['model_id'],'revision':cfg.get('revision'),
        'dataset':cfg.get('official_dataset',cfg.get('official_datasets')),
        'reference':cfg.get('official_claim'),'weights':str(weights),'dataset_root':str(datasets),
        'historical_a100_status':status['historical_a100_status'],
        'submission_entry_status':status['submission_entry_status'],'missing':missing},ensure_ascii=False),flush=True)
    if args.check_inputs:
        return 2 if missing or status['submission_entry_status']!='independent_entry_prepared' else 0
    if status['submission_entry_status']!='independent_entry_prepared':
        print(status['reason'],file=sys.stderr)
        print('原始脚本、最新A100适配代码和启动参数保存在 inference/original_sources/ 与 native_launch.json；本副本尚未宣称独立可运行。',file=sys.stderr)
        return 2
    if missing:return 2
    os.environ['ZERO_SHOT_WEIGHTS']=str(weights)
    os.environ['ZERO_SHOT_DATASET_ROOT']=str(datasets)
    out=MODEL_DIR/'result/run_001'
    if args.output_root is not None:
        out=(args.output_root if args.output_root.is_absolute() else MODEL_DIR/args.output_root).resolve()
        if not out.is_relative_to((MODEL_DIR/'result').resolve()):parser.error('output-root must be in this model result/')
    out.mkdir(parents=True,exist_ok=False)
    cache_vars={'HF_HOME':str(out/'cache/huggingface'),'HF_HUB_CACHE':str(out/'cache/huggingface/hub'),'HF_MODULES_CACHE':str(out/'cache/huggingface/modules'),'TORCH_HOME':str(out/'cache/torch'),'XDG_CACHE_HOME':str(out/'cache'),'CUDA_CACHE_PATH':str(out/'cache/cuda'),'TRITON_CACHE_DIR':str(out/'cache/triton'),'TORCH_EXTENSIONS_DIR':str(out/'cache/torch_extensions'),'TMPDIR':str(out/'tmp')}
    (out/'tmp').mkdir(exist_ok=True)
    os.environ.update(cache_vars)
    os.environ['ZERO_SHOT_SMOKE_LIMIT']=str(args.max_images if args.smoke else 0)
    os.environ['ZERO_SHOT_RUN_DIR']=str(out)
    os.environ['PYTHONPATH']=str(MODEL_DIR/'inference')+os.pathsep+os.environ.get('PYTHONPATH','')
    command=[sys.executable,'-u',str(MODEL_DIR/'inference/native_entry.py')]+rest
    def save(name,value):(out/name).write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    save('config.json',{'model':cfg['model_id'],'command':command,'weights':str(weights),'dataset_root':str(datasets),'python':sys.version,'gpu_retested_in_submission':False})
    save('status.json',{'state':'running'})
    with (out/'execution.log').open('w',encoding='utf-8') as log:
        try:
            process=subprocess.Popen(command,cwd=MODEL_DIR,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,encoding='utf-8',errors='replace')
            for line in process.stdout:
                print(line,end='',flush=True);log.write(line);log.flush()
            code=process.wait()
        except BaseException as error:
            save('status.json',{'state':'failed','error':repr(error)})
            raise
    save('status.json',{'state':('smoke_completed' if args.smoke else 'completed') if code==0 else 'failed','exit_code':code})
    return code
if __name__=='__main__':
    raise SystemExit(main())
