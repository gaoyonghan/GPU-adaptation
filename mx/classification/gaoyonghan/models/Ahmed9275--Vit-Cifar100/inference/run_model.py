#!/usr/bin/env python3
"""Outer loops, timing and archiving around the publisher's inference entry points."""
import argparse
from collections import Counter
import importlib.metadata
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

from common import HERE, Images, digest, fingerprint, locked, read_index, save, smoke_rows
from timing import ForwardTimer, synchronize

for key in ['HF_HUB_OFFLINE', 'TRANSFORMERS_OFFLINE', 'HF_DATASETS_OFFLINE']:
    os.environ[key] = '1'
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
os.environ.setdefault('HF_MODULES_CACHE', str(HERE/'.runtime/hf_modules'))
os.environ.setdefault('MPLCONFIGDIR', str(HERE/'.runtime/matplotlib'))


def environment():
    import torch
    versions = {}
    for package in ['torch','torchvision','transformers','timm','Pillow','numpy','scikit-learn','pyarrow','ultralytics','mamba-ssm']:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            pass
    return {'python':sys.version,'versions':versions,'cuda_runtime':torch.version.cuda,
            'cuda_available':torch.cuda.is_available()}


def gpu_snapshot():
    try:
        return subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,name,driver_version,memory.used,utilization.gpu','--format=csv,noheader'],text=True).strip()
    except (OSError,subprocess.CalledProcessError):
        return None


def check(spec, manifest, smoke=False):
    root=Path(spec['model_path'])
    for name,item in spec['files'].items():
        p=root/name
        if not p.is_file() or p.stat().st_size!=item['size'] or digest(p)!=item['sha256']:
            raise ValueError(f'Model file missing or changed: {p}')
    for p,item in spec['data_containers'].items():
        stat=Path(p).stat()
        if stat.st_size!=item['size'] or stat.st_mtime_ns!=item['mtime_ns']:
            raise ValueError(f'Data container changed: {p}')
    rows=read_index(spec)
    # Validate selected image bytes only in smoke; retain full index validation.
    selected_rows = smoke_rows(rows) if smoke else rows
    for row in selected_rows:
        if 'path' in row:
            path=Path(row['path'])
            if not path.is_file(): raise ValueError(f'Missing selected image: {path}')
            if 'sha256' in row and digest(path)!=row['sha256']:
                raise ValueError(f'Image hash changed: {path}')
            if 'crc32' in row and zlib.crc32(path.read_bytes())!=row['crc32']:
                raise ValueError(f'Image CRC changed: {path}')
        if row['target'] is not None and not 0<=row['target']<spec['num_output_classes']:
            raise ValueError('Target is outside the full native head')
    return rows


def score(records, spec):
    import numpy as np
    from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, precision_recall_fscore_support, roc_auc_score
    truth=np.array([r['target'] for r in records]);pred=np.array([r['prediction'] for r in records])
    labels=sorted(set(truth.tolist()))
    precision,recall,f1,support=precision_recall_fscore_support(truth,pred,labels=labels,zero_division=0)
    result={'accuracy':float(accuracy_score(truth,pred)),
            'balanced_accuracy':float(balanced_accuracy_score(truth,pred)),
            'macro_f1_over_evaluated_target_classes':float(f1.mean()),
            'macro_precision_over_evaluated_target_classes':float(precision.mean()),
            'macro_recall_over_evaluated_target_classes':float(recall.mean()),
            'weighted_f1':float(np.average(f1,weights=support)),
            'weighted_precision':float(np.average(precision,weights=support)),
            'weighted_recall':float(np.average(recall,weights=support)),
            'micro_f1':float(f1_score(truth,pred,average='micro',zero_division=0)),
            'cross_entropy_reference':float(np.mean([-math.log(max(r['true_probability'],1e-300)) for r in records])),
            'units':'fractions, not percent',
            'per_class':{str(label):{'precision':float(p),'recall':float(r),'f1':float(f),'support':int(s)} for label,p,r,f,s in zip(labels,precision,recall,f1,support)},
            'scoring_source':'scikit-learn; publisher model inference retained; not a publisher end-to-end evaluator',
            'reference_comparison':'not_officially_comparable' if spec['dataset_relation']!='original_exact' else 'manual_reference_required'}
    if spec['num_output_classes']>=5:
        result['top5_accuracy']=float(np.mean([r['target'] in r['top5'] for r in records]))
    if spec['num_output_classes']==2 and len(labels)==2:
        result['auroc_class_1']=float(roc_auc_score(truth,[r['positive_probability'] for r in records]))
        result['binary_log_loss']=result['cross_entropy_reference']
    return result


def journal_records(path, rows):
    if not path.exists():
        return [],[]
    data=path.read_bytes()
    # Preserve an interrupted final write for inspection; only a non-newline tail is removed.
    if data and not data.endswith(b'\n'):
        boundary=data.rfind(b'\n')+1
        path.with_name(f'interrupted-tail-{time.time_ns()}.bin').write_bytes(data[boundary:])
        with path.open('r+b') as f:
            f.truncate(boundary)
        data=data[:boundary]
    batches=[json.loads(line) for line in data.splitlines()]
    records=[r for b in batches for r in b['records']]
    if [r['id'] for r in records]!=[r['id'] for r in rows[:len(records)]]:
        raise ValueError('Resume journal is not an exact prefix of the frozen selection')
    for r,source in zip(records,rows):
        if r['target']!=source['target']:
            raise ValueError('Resume label mismatch')
    return records,batches


def hf_run(spec, rows, args, out):
    import numpy as np
    import torch
    import transformers
    start=time.perf_counter()
    kwargs={'local_files_only':True,'output_loading_info':True}
    model_class=transformers.AutoModelForImageClassification
    model,loading=model_class.from_pretrained(spec['model_path'],**kwargs)
    if loading.get('missing_keys') or loading.get('unexpected_keys') or loading.get('mismatched_keys'):
        raise ValueError(f'Checkpoint does not load exactly into publisher model: {loading}')
    model=model.to(args.device).eval()
    processor=transformers.AutoImageProcessor.from_pretrained(spec['model_path'],local_files_only=True,use_fast=False)
    def infer(images):
        inputs=processor(images=images,return_tensors='pt').to(args.device)
        return model(**inputs).logits.softmax(dim=-1).float().cpu().numpy()
    synchronize(args.device)
    load_seconds=time.perf_counter()-start
    image_reader=Images()
    journal=out/'batches.jsonl'
    records,batches=journal_records(journal,rows)
    pending=rows[len(records):]
    timer=ForwardTimer(model,args.device)
    warmup_seconds=0.0
    instrumentation_verified=None
    try:
        with torch.inference_mode():
            baseline=None
            if pending:
                images=[image_reader.image(r) for r in pending[:args.batch_size]]
                synchronize(args.device);warm_start=time.perf_counter()
                for _ in range(args.warmup):
                    baseline=infer(images)
                synchronize(args.device);warmup_seconds=time.perf_counter()-warm_start
            if str(args.device).startswith('cuda'):
                torch.cuda.reset_peak_memory_stats(args.device)
            timer.active=True
            with journal.open('a') as f:
                for offset in range(0,len(pending),args.batch_size):
                    chunk=pending[offset:offset+args.batch_size]
                    synchronize(args.device);batch_start=time.perf_counter()
                    images=[image_reader.image(row) for row in chunk]
                    probs=infer(images)
                    synchronize(args.device);elapsed=time.perf_counter()-batch_start
                    if probs.shape!=(len(chunk),spec['num_output_classes']) or not np.isfinite(probs).all():
                        raise ValueError('Native output shape or values invalid')
                    if offset==0 and baseline is not None:
                        instrumentation_verified=bool(np.allclose(baseline,probs,rtol=1e-5,atol=1e-7) and np.array_equal(baseline.argmax(-1),probs.argmax(-1)))
                        if not instrumentation_verified:
                            raise ValueError('Enabling timing changed predictions beyond the frozen tolerance')
                    forward=timer.summary()
                    if forward['timed_images']!=len(chunk):
                        raise ValueError('Timed forwards do not cover exactly this batch')
                    last=forward['batch_forward_seconds'][-1]
                    timer.records.clear()
                    batch_records=[]
                    for row,prob in zip(chunk,probs):
                        top=np.argsort(-prob,kind='stable')[:min(5,len(prob))].tolist()
                        batch_records.append({'id':row['id'],'target':row['target'],'prediction':int(top[0]),'top5':top,
                                              'confidence':float(prob[top[0]]),'true_probability':float(prob[row['target']]) if row['target'] is not None else None,
                                              'positive_probability':float(prob[1]) if len(prob)==2 else None})
                    batch={'records':batch_records,'forward_seconds':last,'inference_e2e_seconds':elapsed,
                           'images':len(chunk),'segment':args.segment}
                    f.write(json.dumps(batch,ensure_ascii=False,allow_nan=False)+'\n');f.flush()
                    records.extend(batch_records);batches.append(batch)
                    if len(records)%100==0 or len(records)==len(rows):
                        print(f'{spec["sequence"]}: {len(records)}/{len(rows)} predictions saved',flush=True)
    finally:
        timer.close();image_reader.close()
    assert [r['id'] for r in records]==[r['id'] for r in rows]
    forward=sum(b['forward_seconds'] for b in batches)
    e2e=sum(b['inference_e2e_seconds'] for b in batches)
    timing={'model_load_seconds_this_segment':load_seconds,'warmup_seconds_this_segment':warmup_seconds,
            'warmup_batches':args.warmup,'timed_images':len(records),'forward_seconds':forward,'inference_e2e_seconds':e2e,
            'forward_amortized_ms_per_image':1000*forward/len(records),'forward_images_per_second':len(records)/forward,
            'e2e_images_per_second':len(records)/e2e,'actual_batch_sizes':dict(Counter(b['images'] for b in batches)),
            'forward_method':'CUDA events' if str(args.device).startswith('cuda') else 'perf_counter',
            'e2e_scope':'Per batch image read/decode, native processor, transfer, forward, publisher postprocess, CPU prediction availability; excludes journal writes and metrics',
            'resume_segments':len({b['segment'] for b in batches}),'instrumentation_predictions_unchanged_this_segment':instrumentation_verified,
            'peak_allocated_bytes_this_segment':torch.cuda.max_memory_allocated(args.device) if str(args.device).startswith('cuda') else None}
    with (out/'predictions.jsonl').open('w') as f:
        for record in records:
            f.write(json.dumps(record,ensure_ascii=False)+'\n')
    metrics=None
    if args.phase=='full' and spec['scoring']!='prediction_only':
        start=time.perf_counter();metrics=score(records,spec);timing['evaluation_seconds']=time.perf_counter()-start
    return timing,metrics


def run_one(args, spec, manifest):
    key=f'm{spec["sequence"]}'+('-'+Path(args.variant).stem if args.variant else '')
    out=args.output_root/args.phase/key
    out.mkdir(parents=True,exist_ok=True)
    with locked(out/'.lock'):
        status_path=out/'status.json'
        if spec['status'] in ['blocked','skipped']:
            save(status_path,{'sequence':spec['sequence'],'status':spec['status'],'reason':spec['reason']})
            return 0 if spec['status']=='skipped' else 2
        env=environment()
        code={p.name:digest(p) for p in HERE.glob('*.py')}
        protocol={'model':spec['protocol_fingerprint'],'code':code,'environment':env,'device':args.device,
                  'batch_size':args.batch_size,'warmup':args.warmup,'workers':args.workers,'variant':args.variant}
        fp=fingerprint(protocol)
        previous=json.loads(status_path.read_text()) if status_path.exists() else None
        if previous and previous.get('protocol_fingerprint') not in [None,fp]:
            raise ValueError('Protocol/code/environment changed: use a new --output-root; old results preserved')
        if previous and previous.get('status') in ['complete','partial_complete','prediction_only_complete']:
            if not (out/'timing.json').exists():
                raise ValueError('Completed result is missing timing.json')
            if previous['status']=='complete' and not (out/'metrics.json').exists():
                raise ValueError('Completed evaluation is missing metrics.json')
            print(f'{key}: already finished this phase',flush=True);return 0
        state={'sequence':spec['sequence'],'model_id':spec['model_id'],'variant':args.variant,'phase':args.phase,
               'protocol_fingerprint':fp,'protocol':protocol,'model_manifest':spec,'status':'running',
               'gpu_at_start':gpu_snapshot(),'full_evaluation':False}
        save(status_path,state)
        started=time.perf_counter()
        try:
            rows=check(spec,manifest,smoke=args.phase=='smoke')
            if args.phase=='preflight':
                state.update(status='preflight_ready',images=len(rows));save(status_path,state);return 0
            if str(args.device).startswith('cuda') and not env['cuda_available']:
                raise RuntimeError('CUDA unavailable; no silent CPU fallback')
            if str(args.device).startswith('cuda'):
                import torch
                torch.cuda.set_device(args.device)
            if args.phase=='smoke': rows=smoke_rows(rows)
            elif args.phase=='full':
                print("Direct full evaluation: no prior smoke prerequisite; smoke equivalence not established.", flush=True)
            args.segment=str(time.time_ns())
            timing,metrics=hf_run(spec,rows,args,out)
            timing.update(run_wall_seconds_this_segment=time.perf_counter()-started,
                          gpu_at_end=gpu_snapshot(),phase=args.phase,
                          performance_comparison_valid=False if args.phase=='smoke' else 'requires review of recorded GPU co-tenancy and matching batch/dtype/resolution')
            import torch
            timing['backend_flags']={'cuda_matmul_allow_tf32':torch.backends.cuda.matmul.allow_tf32,
                                     'cudnn_allow_tf32':torch.backends.cudnn.allow_tf32,
                                     'cudnn_benchmark':torch.backends.cudnn.benchmark}
            save(out/'timing.json',timing)
            if metrics is not None: save(out/'metrics.json',metrics)
            state.update(status='partial_complete' if args.phase=='smoke' else 'prediction_only_complete' if spec['scoring']=='prediction_only' else 'complete',
                         images=len(rows),timing_file=str(out/'timing.json'),full_evaluation=args.phase=='full' and spec['scoring']!='prediction_only',
                         run_wall_seconds_this_segment=time.perf_counter()-started)
            save(status_path,state);return 0
        except Exception as error:
            state.update(status='failed',error_type=type(error).__name__,error=str(error),traceback=traceback.format_exc())
            save(out/f'failure-{time.time_ns()}.json',state);save(status_path,state)
            traceback.print_exc();return 1


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--phase',choices=['preflight','smoke','full'],default='preflight')
    parser.add_argument('--describe',action='store_true')
    parser.add_argument('--output-root',type=Path,default=HERE.parent/'results/run_001')
    parser.add_argument('--device',default='cuda:0')
    parser.add_argument('--batch-size',type=int,default=1)
    parser.add_argument('--warmup',type=int,default=10)
    parser.add_argument('--workers',type=int,default=2)
    parser.add_argument('--variant')
    args=parser.parse_args()
    if not args.output_root.is_absolute():
        args.output_root=HERE.parent/args.output_root
    if args.batch_size<1 or args.warmup<1 or args.workers<1: parser.error('batch-size, warmup and workers must be positive')
    manifest=json.loads((HERE/'models.json').read_text())
    if len(manifest['models']) != 1:
        raise ValueError('Expected one model in this folder')
    spec=manifest['models'][0]
    # Resolve the moved files against this model directory.
    spec['model_path']=str((HERE.parent/spec['model_path']).resolve())
    spec['index']=str((HERE.parent/spec['index']).resolve())
    spec['data_containers']={str((HERE.parent/p).resolve()):v for p,v in spec['data_containers'].items()}
    if args.describe: print(json.dumps(spec,ensure_ascii=False,indent=2));return 0
    if spec.get('variants') and args.variant not in spec['variants']: parser.error('Choose one of the frozen --variant weight filenames, or use run_all.sh')
    return run_one(args,spec,manifest)


if __name__=='__main__':
    raise SystemExit(main())
