"""Serial per-GPU dispatch: isolate library stacks, preserve every model and variant."""
import argparse
from collections import Counter
import csv
import json
import os
from pathlib import Path
import subprocess

from common import HERE, locked, save


def summarize(args, models, failures, blocked):
    with locked(args.output_root/'.summary.lock'):
        rows=[json.loads(path.read_text()) for path in sorted((args.output_root/args.phase).glob('*/status.json'))]
        groups={}
        for row in rows: groups.setdefault(row['sequence'],[]).append(row)
        by_model={}
        for n,results in groups.items():
            expected=len(models[n].get('variants',[None]))
            statuses=[r['status'] for r in results]
            by_model[n]=statuses[0] if len(results)==expected and len(set(statuses))==1 else 'mixed_or_incomplete'
        save(args.output_root/f'audit_summary_{args.phase}.json',{'phase':args.phase,'models':by_model,
             'counts_by_unique_model':dict(Counter(by_model.values())),'variant_results':rows,
             'this_invocation':{'requested_models':args.models,'failed':failures,'blocked':blocked}})
        fields=['sequence','model_id','variant','status','dataset_relation','fidelity','images','device','batch_size',
                'accuracy','top5_accuracy','macro_f1','weighted_f1','forward_seconds','forward_ms_per_image',
                'forward_images_per_second','inference_e2e_seconds','e2e_images_per_second','result_directory']
        with (args.output_root/f'results_{args.phase}.csv').open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
            for row in rows:
                n=row['sequence'];spec=models[n]
                key=f'm{n}'+('-'+Path(row['variant']).stem if row.get('variant') else '')
                folder=args.output_root/args.phase/key
                timing=json.loads((folder/'timing.json').read_text()) if (folder/'timing.json').exists() else {}
                metrics=json.loads((folder/'metrics.json').read_text()) if (folder/'metrics.json').exists() else {}
                accuracy=metrics.get('accuracy',metrics.get('metrics/accuracy_top1'))
                top5=metrics.get('top5_accuracy',metrics.get('metrics/accuracy_top5'))
                if 'top1' in metrics: accuracy=metrics['top1']/100
                if 'top5' in metrics: top5=metrics['top5']/100
                writer.writerow({'sequence':n,'model_id':spec['model_id'],'variant':row.get('variant'),'status':row['status'],
                    'dataset_relation':spec.get('dataset_relation'),'fidelity':spec.get('fidelity'),'images':row.get('images'),
                    'device':row.get('protocol',{}).get('device'),'batch_size':row.get('protocol',{}).get('batch_size'),
                    'accuracy':accuracy,'top5_accuracy':top5,'macro_f1':metrics.get('macro_f1_over_evaluated_target_classes'),
                    'weighted_f1':metrics.get('weighted_f1'),'forward_seconds':timing.get('forward_seconds'),
                    'forward_ms_per_image':timing.get('forward_amortized_ms_per_image'),
                    'forward_images_per_second':timing.get('forward_images_per_second'),
                    'inference_e2e_seconds':timing.get('inference_e2e_seconds'),'e2e_images_per_second':timing.get('e2e_images_per_second'),
                    'result_directory':str(folder)})


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--phase',choices=['preflight','smoke','full'],default='preflight')
    p.add_argument('--models',type=int,nargs='+',default=list(range(171,201)))
    p.add_argument('--output-root',type=Path,default=Path('__MODEL_ROOT__/resources/runtime/assets/workspace/results/classification171_200_v1'))
    p.add_argument('--device',default=os.environ.get('DEVICE','cuda:0'))
    p.add_argument('--batch-size',type=int,default=1)
    p.add_argument('--warmup',type=int,default=10)
    p.add_argument('--workers',type=int,default=2)
    args=p.parse_args()
    # Coordinate this package's GPU runs even when output roots differ.
    gpu_lock=locked(HERE/'.runtime'/('dispatch-'+args.device.replace(':','_')+'.lock'))
    gpu_lock.__enter__()
    manifest=json.loads((HERE/'models.json').read_text())
    models={s['sequence']:s for s in manifest['models']}
    if not set(args.models)<=set(models): p.error('Unknown model number')
    base='__MODEL_ROOT__/resources/runtime/assets/workspace/classification8_eval/.venv/bin/python'
    failures=[];blocked=[]
    for number in args.models:
        spec=models[number]
        family=spec.get('environment','shared_hf')
        python=base if family in ['shared_hf','js'] or args.phase=='preflight' else str(HERE/f'.venv-{family}'/'bin/python')
        for variant in spec.get('variants',[None]):
            key=f'm{number}'+('-'+Path(variant).stem if variant else '')
            out=args.output_root/args.phase/key;out.mkdir(parents=True,exist_ok=True)
            if not Path(python).exists():
                save(out/'status.json',{'sequence':number,'variant':variant,'status':'blocked','reason':f'Environment missing; run bash setup_env.sh {family}'})
                blocked.append(key);continue
            phases=['smoke','full'] if args.phase=='full' else [args.phase]
            for phase in phases:
                command=[python,str(HERE/'run_model.py'),'--model',str(number),'--phase',phase,
                         '--output-root',str(args.output_root),'--device','cpu' if family=='js' else args.device,'--batch-size',str(args.batch_size),
                         '--warmup',str(args.warmup),'--workers',str(args.workers)]
                if variant:command+=['--variant',variant]
                log_dir=args.output_root/phase/key;log_dir.mkdir(parents=True,exist_ok=True)
                print(f'{key}: {phase} → {log_dir}',flush=True)
                with (log_dir/'console.log').open('a') as log:
                    result=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT)
                if result.returncode:
                    (blocked if result.returncode==2 else failures).append(key)
                    if phase!=args.phase and not (out/'status.json').exists():
                        save(out/'status.json',{'sequence':number,'variant':variant,
                             'status':'blocked' if result.returncode==2 else 'failed',
                             'reason':f'Required {phase} did not pass; see {log_dir}'})
                    print(f'{key}: exit {result.returncode}; see console.log',flush=True)
                    break
            summarize(args,models,failures,blocked)
    summarize(args,models,failures,blocked)
    print(json.dumps({'failed':failures,'blocked':blocked,'phase':args.phase},ensure_ascii=False),flush=True)
    return 1 if failures else 2 if blocked else 0


if __name__=='__main__':
    raise SystemExit(main())
