#!/usr/bin/env python3
"""Optional delivery tests only; individual model entries never depend on this file."""
from pathlib import Path
import argparse,collections,csv,fcntl,json,os,signal,subprocess,sys,time,traceback
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--only',action='append');a=p.parse_args()
r=a.root.resolve();code=r/'GPU-adaptation-submit/a100/zero-shot-classification/zhoujianjie'
original=Path('/data/zhoujianjie/model_inference_for_GPU_adaptation')
lock=(r/'validation.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
rows=list(csv.DictReader((code/'environments/model_environment.csv').open()))
models=[]
for row in rows:
 if row['historical_a100_status']!='completed':continue
 d=code/'models'/row['model_directory'];s=json.loads((d/'inference/submission_status.json').read_text())
 if s['historical_a100_status']=='completed':models.append((d,row))
models.sort(key=lambda item:(0 if item[0].name.startswith('flax-community__') else 1 if (item[0]/'inference/delivery_config.json').exists() else 2,item[0].name))
statefile=r/'delivery_validation_state.json';state=json.loads(statefile.read_text()) if statefile.exists() else {}
# If a prior manager exited while its detached child survived, never duplicate it.
for _mid,_record in state.items():
 if _record.get('gpu_smoke')=='running' and _record.get('pid'):
  try:os.kill(_record['pid'],0)
  except ProcessLookupError:pass
  else:raise SystemExit('Existing validation child may still be running: '+_mid+'; preserve it and inspect its status before resuming')
def save():
 tmp=statefile.with_suffix('.tmp');tmp.write_text(json.dumps(state,ensure_ascii=False,indent=2)+'\n');tmp.replace(statefile)
def normalized(bundle):
 pred=json.loads((bundle/'predictions.json').read_text());truth={x['sample_id']:x['label'] for x in json.loads((bundle/'annotations.json').read_text())['samples']}
 return [{**x,'target':truth[x['sample_id']]} for x in pred]
def compare(d,out,profile,env):
 manifest=json.loads((out/'inference_output.json').read_text());new=[]
 for bundle in manifest['bundles']:new.extend(normalized(out/bundle))
 if not new:return {'state':'not_compared','reason':'No standard prediction bundle'}
 base=original/'classification/zhoujianjie/models'/d.name/'result'
 hits=[]
 for f in sorted(base.rglob('predictions.jsonl'),reverse=True):
  if any(x in str(f.relative_to(base)).lower() for x in ['smoke','small_','preflight','history','4090']):continue
  old={}
  with f.open() as stream:
   for ordinal,line in enumerate(stream):
    x=json.loads(line);sid=x.get('sample_id')
    if sid is None and 'shard' in x:sid=x['shard']+':'+str(x['batch']*16+x['index'])
    if sid is None and d.name in {'Xenova__clip-vit-base-patch32','philschmid__clip-zero-shot-image-classification'}:sid=str(ordinal)
    if sid is not None:old[str(sid)]=x
  ground_file=f.with_name('ground_truth.jsonl')
  if ground_file.exists():
   for line in ground_file.read_text().splitlines():
    ground=json.loads(line);sid=str(ground.get('sample_id'))
    if sid in old:old[sid]['target']=ground['target']
  if len(old)<=len(new) or not all(x['sample_id'] in old for x in new):continue
  same=0;rank_same=0;targets_equal=0
  for x in new:
   y=old[x['sample_id']];rank=y.get('ranked_predictions',y.get('ranking',y.get('top5_predictions')));pred=y.get('prediction',rank[0] if rank else None)
   same+=pred==x['prediction'];rank_same+=rank==x['topk'] if rank is not None else False
   targets_equal+=y.get('target',y.get('label'))==x['target']
  return {'state':'equal' if same==len(new) and targets_equal==len(new) else 'mismatch','samples':len(new),'top1_equal':same,'ground_truth_equal':targets_equal,'ordered_ranking_equal':rank_same,'original_prediction_path':str(f),'scope':'same fixed sample identity/order with ground-truth verification; no model rerun'}
 # These native protocols use a frozen test-row order and save full arrays.
 patterns={'geolocal__StreetCLIP':'country_predictions.json','google__medsiglip-448':'predictions.npz','qihoo360__fg-clip2-base':'rank0_predictions.jsonl'}
 pattern=patterns.get(d.name)
 if pattern:
  for f in sorted(base.rglob(pattern),reverse=True):
   if any(x in str(f.relative_to(base)).lower() for x in ['smoke','small_','preflight','history','4090']):continue
   if pattern.endswith('.npz'):
    import numpy as np
    with np.load(f) as payload:
     if not {'targets','predictions'}.issubset(payload.files):continue
     old=[{'target':int(t),'prediction':int(p)} for t,p in zip(payload['targets'],payload['predictions'])]
   elif pattern.endswith('.json'):
    payload=json.loads(f.read_text());old=[{'target':t,'prediction':p,'topk':k} for t,p,k in zip(payload['targets'],payload['predictions'],payload['top5_predictions'])]
   else:
    old=[]
    for line in f.read_text().splitlines():
     item=json.loads(line);ranking=item.get('ranking',item.get('ranked_predictions'))
     if not ranking:break
     old.append({'target':item['target'],'prediction':ranking[0],'topk':ranking})
   if len(old)<=len(new) or not all(x['sample_id'].isdigit() and int(x['sample_id'])<len(old) for x in new):continue
   targets_equal=sum(old[int(x['sample_id'])]['target']==x['target'] for x in new)
   same=sum(old[int(x['sample_id'])]['prediction']==x['prediction'] for x in new)
   return {'state':'equal' if same==len(new) and targets_equal==len(new) else 'mismatch','samples':len(new),'top1_equal':same,'ground_truth_equal':targets_equal,'original_prediction_path':str(f),'scope':'frozen native test-row order; ground truth checked; no original model rerun'}
 return {'state':'not_compared','reason':'No matching original full-A100 per-sample predictions found; no consistency pass claimed','samples':len(new)}

for d,row in models:
 state.setdefault(row['model_id'],{'original_a100_full':'completed','independent_code':'prepared_pending_runtime_validation','cpu_static':'passed','gpu_smoke':'pending','public_evaluator':'not_run','prediction_consistency':'not_compared','final_delivery':'pending_validation'})
save()

for d,row in models:
 mid=row['model_id']
 if a.only and mid not in a.only:continue
 previous=state.get(mid,{})
 if previous.get('gpu_smoke')=='passed':continue
 cfgfile=d/'inference/delivery_config.json';cfg=json.loads(cfgfile.read_text()) if cfgfile.exists() else {}
 if cfg and not cfg['bounded_smoke_supported']:
  state[mid]={'original_a100_full':'completed','gpu_smoke':'not_run','final_delivery':'blocked_bounded_diagnostic','reason':'No bounded diagnostic adapter; full evaluation was NOT started'};save();continue
 profile_name=row['environment_file'].removesuffix('.txt').removesuffix('-overlay')
 profilefile=r/'profiles'/(profile_name+'.json')
 if not profilefile.exists():
  setup=subprocess.run([sys.executable,str(r/'setup_delivery_environment.py'),'--profile',profile_name],capture_output=True,text=True)
  if setup.returncode:
   state[mid]={'original_a100_full':'completed','gpu_smoke':'not_run','final_delivery':'blocked_isolated_environment','error':setup.stderr[-8000:],'environment_setup_stdout':setup.stdout[-8000:]};save();continue
 profile=json.loads(profilefile.read_text());stamp=time.strftime('%Y%m%dT%H%M%S');out=d/'result'/('delivery_smoke_'+stamp)
 count=max(8,cfg.get('minimum_smoke_images',0));count=16 if count>8 else count
 weights=original/'classification/zhoujianjie/models'/d.name/'weights';data=original/'dataset'
 env={**os.environ,'PYTHONPATH':profile['PYTHONPATH'],'ZERO_SHOT_FORBIDDEN_SOURCE_ROOTS':str(original)+':/data/zhoujianjie/environment_setup','ZERO_SHOT_WEIGHTS_ROOT':str(weights.parent.parent),'PYTHONDONTWRITEBYTECODE':'1','USE_TF':'0','USE_FLAX':'0'}
 cmd=[profile['python'],str(d/'inference.py'),'--smoke','--max-images',str(count),'--weights',str(weights),'--dataset-root',str(data),'--output-root',str(out)]
 if cfg:cmd+=['--weights-root',str(weights.parent.parent)]
 record={'original_a100_full':'completed','independent_code':'prepared','cpu_static':'passed','gpu_smoke':'running','public_evaluator':'not_run','prediction_consistency':'not_compared','final_delivery':'validating','command':cmd,'environment':profile_name,'run_dir':str(out),'started_at':time.strftime('%Y-%m-%dT%H:%M:%S%z'),'full_evaluation_executed':False,'smoke_scope':cfg.get('smoke_scope','fixed prefix classification samples')}
 record['previous_attempts']=previous.get('previous_attempts',[])+([previous] if previous.get('log_path') else [])
 state[mid]=record;save();print('START',mid,flush=True)
 log=r/'logs';log.mkdir(exist_ok=True);logfile=log/(d.name+'-'+stamp+'.log')
 try:
  with logfile.open('w') as stream:
   process=subprocess.Popen(cmd,env=env,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
   record['pid']=process.pid;record['log_path']=str(logfile);save()
   try:exitcode=process.wait(timeout=1200)
   except subprocess.TimeoutExpired:
    os.killpg(process.pid,signal.SIGTERM)
    try:process.wait(timeout=30)
    except subprocess.TimeoutExpired:os.killpg(process.pid,signal.SIGKILL);process.wait(timeout=10)
    exitcode=124
  record.update(exit_code=exitcode,finished_at=time.strftime('%Y-%m-%dT%H:%M:%S%z'))
  if exitcode:
   record.update(gpu_smoke='failed',final_delivery='blocked_smoke_failure',error=logfile.read_text(errors='replace')[-12000:]);save();print('FAILED',mid,exitcode,flush=True);continue
  manifest=json.loads((out/'inference_output.json').read_text())
  artifacts=list(out.rglob('source_isolation_*.json'))
  if not artifacts:raise RuntimeError('No active source-isolation guard evidence')
  record.update(gpu_smoke='passed',source_isolation_evidence=[str(f) for f in artifacts])
  if manifest.get('task')!='classification':
   record.update(public_evaluator='unsupported_original_protocol',final_delivery='blocked_public_protocol',reason='Original '+manifest.get('task','unknown')+' protocol cannot be represented by the unchanged single-label evaluator');save();print('NATIVE_PASS_PUBLIC_BLOCKED',mid,flush=True);continue
  evaluation=subprocess.run([profile['python'],str(r/'GPU-adaptation-submit/a100/zero-shot-classification/zhoujianjie/evaluate.py'),'--run-dir',str(out)],env={**os.environ,'PYTHONPATH':profile['PYTHONPATH'],'PYTHONDONTWRITEBYTECODE':'1'},capture_output=True,text=True)
  record['public_evaluator']='passed' if evaluation.returncode==0 else 'failed'
  (out/'public_evaluator.log').write_text(evaluation.stdout+evaluation.stderr)
  comparison=compare(d,out,profile,env);record['prediction_consistency']=comparison['state'];record['comparison']=comparison
  (out/'comparison.json').write_text(json.dumps(comparison,ensure_ascii=False,indent=2)+'\n')
  record['final_delivery']='validated_smoke_and_same_sample_predictions' if evaluation.returncode==0 and comparison['state']=='equal' else 'blocked_prediction_mismatch' if evaluation.returncode==0 and comparison['state']=='mismatch' else 'pending_prediction_comparison' if evaluation.returncode==0 else 'blocked_public_evaluator'
  print('DONE',mid,record['public_evaluator'],record['prediction_consistency'],flush=True)
 except Exception:
  record.update(final_delivery='blocked_validation_error',error=traceback.format_exc());print('ERROR',mid,record['error'],flush=True)
 save()
print('QUEUE_FINISHED',dict(collections.Counter(x.get('final_delivery') for x in state.values())),flush=True)
