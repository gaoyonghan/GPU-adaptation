from pathlib import Path
import json,os
r=Path('/data/zhoujianjie/delivery_validation_20261126')
original=Path('/data/zhoujianjie/model_inference_for_GPU_adaptation')
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

state=json.loads((r/'delivery_validation_state.json').read_text());supplement={}
for mid,record in state.items():
 if record.get('gpu_smoke')!='passed' or record.get('public_evaluator')!='passed':continue
 d=r/'GPU-adaptation-submit/a100/classification/zhoujianjie/models'/mid.replace('/','__');out=Path(record['run_dir'])
 result=compare(d,out,None,{})
 (out/'comparison_with_ground_truth.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
 supplement[mid]=result
 print(mid,result['state'],result.get('samples'),result.get('top1_equal'),flush=True)
(r/'delivery_prediction_comparison_supplement.json').write_text(json.dumps(supplement,ensure_ascii=False,indent=2)+'\n')
