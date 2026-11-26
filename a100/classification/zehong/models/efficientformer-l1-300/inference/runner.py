"""Model-local inference and scoring-bundle conversion; no accuracy computation."""
import csv,json,time,hashlib,sys
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from inference.options import ARGS
from inference import data

POLICY={'task':'single_label_classification','top_k':[1,5],'top5_when_classes_below_five':'not_applicable','top5_when_ranking_missing':'unavailable','ranking':'preserve_native_order_no_new_tie_break','macro_labels':'observed_ground_truth_classes','zero_division':0,'unit':'fraction','sample_weight':'uniform'}
def dump(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
def build_processor(*args,**kwargs):
 from transformers import ViTImageProcessor
 return ViTImageProcessor.from_pretrained(ARGS.weights,trust_remote_code=True,local_files_only=True)
def evaluate_model(model,*pos,**kwargs):
 import inspect,importlib.metadata
 name=kwargs.get('dataset_name',pos[0] if pos else next(iter(data.DATASETS)))
 kwargs['dataset_name']=name
 kwargs.update(output_dir=ARGS.output_dir,batch_size=ARGS.batch_size,num_workers=ARGS.num_workers,max_samples=ARGS.max_samples,warmup_steps=ARGS.warmup_steps)
 if ARGS.dataset_root is not None:data.DATASETS[name]['root']=ARGS.dataset_root.absolute()
 prepared=data.prepare_inputs(model,**{k:v for k,v in kwargs.items() if k in inspect.signature(data.prepare_inputs).parameters})
 ds=prepared.get('transformed_dataset',prepared['dataset']);raw=prepared['dataset']
 spider='records' in prepared and hasattr(data,'SPIDERStitchedDataset')
 multi=hasattr(data,'CHEXPERT_LABELS')
 if spider:
  labels=prepared['labels'].tolist();names=prepared['class_names'];ids=[str(r.get('image_name',i)) for i,r in enumerate(prepared['records'])]
 else:
  labels=list(raw['labels' if multi else 'target' if 'target' in raw.column_names else 'label'])
  names=list(data.CHEXPERT_LABELS) if multi else [str(getattr(model.config,'id2label',{}).get(i,getattr(model.config,'id2label',{}).get(str(i),i))) for i in range(int(model.config.num_labels))]
  ids=[str(i) for i in range(len(ds))]
 if len(set(ids))!=len(ids):raise ValueError('Duplicate sample IDs')
 targets=np.asarray(labels)
 if multi:
  if targets.shape!=(len(ds),len(names)) or not np.isin(targets,[0,1]).all():raise ValueError('CheXpert GT must be explicit binary [N,14]; uncertain/missing labels require declared protocol')
 elif targets.shape!=(len(ds),) or not np.isin(targets,range(len(names))).all():raise ValueError('GT outside declared categories; no sample silently dropped')
 out=ARGS.output_dir;out.mkdir(parents=True,exist_ok=False)
 bundle=out/'scoring_bundle';bundle.mkdir()
 ann={'labels':[{'id':i,'name':n} for i,n in enumerate(names)],'samples':[{'sample_id':sid,'labels':list(map(int,label))} for sid,label in zip(ids,labels)]} if multi else {'categories':[{'id':i,'name':n} for i,n in enumerate(names)],'samples':[{'sample_id':sid,'label':int(label)} for sid,label in zip(ids,labels)]}
 dump(bundle/'annotations.json',ann)
 device=next(model.parameters()).device
 if device.type not in {'cuda','npu'}:raise RuntimeError('No CPU fallback')
 api=getattr(torch,device.type);model.eval()
 loader=DataLoader(ds,batch_size=ARGS.batch_size,shuffle=False,num_workers=ARGS.num_workers,pin_memory=device.type=='cuda',persistent_workers=ARGS.num_workers>0,collate_fn=prepared.get('collate_fn'))
 def unpack(batch):
  return (batch[0],batch[-1])
 warm,idx=unpack(next(iter(DataLoader(ds,batch_size=ARGS.batch_size,num_workers=0,collate_fn=prepared.get('collate_fn')))))
 warm=warm.to(device)
 with torch.inference_mode():
  for _ in range(ARGS.warmup_steps):model(warm)
 api.synchronize(device);del warm
 api.reset_peak_memory_stats(device)
 logits_file=None;predictions=[];timings=[];done=0
 begin,end=api.Event(enable_timing=True),api.Event(enable_timing=True)
 start=time.perf_counter()
 try:
  with torch.inference_mode(),(out/'predictions.csv').open('w',newline='',encoding='utf-8') as f,(out/'batch_timings.csv').open('w',newline='',encoding='utf-8') as t:
   writer=csv.DictWriter(f,fieldnames=['sample_id','prediction','topk','batch_index','forward_ms_per_image']);writer.writeheader()
   tw=csv.DictWriter(t,fieldnames=['batch_index','sample_count','forward_ms']);tw.writeheader()
   for bi,batch in enumerate(loader):
    images,indices=unpack(batch);images=images.to(device,non_blocking=device.type=='cuda')
    begin.record();output=model(images);end.record();end.synchronize();ms=begin.elapsed_time(end)
    logits=output['logits'] if isinstance(output,dict) else output.logits if hasattr(output,'logits') else output
    if logits.ndim!=2 or logits.shape[1]!=len(names) or not torch.isfinite(logits).all():raise ValueError('Invalid logits shape/values')
    arr=(logits.float() if logits.dtype==torch.bfloat16 else logits).cpu().numpy()
    if logits_file is None:logits_file=np.lib.format.open_memmap(out/'logits.npy',mode='w+',dtype=arr.dtype,shape=(len(ds),len(names)))
    logits_file[indices.tolist()]=arr
    if multi:
     probs=logits.sigmoid().float().cpu().tolist();preds=(logits.sigmoid()>0.5).int().cpu().tolist();ranks=[[]]*len(preds)
    else:
     ranks=logits.topk(min(5,len(names)),1).indices.cpu().tolist();preds=[r[0] for r in ranks]
    for j,(ix,pred,rank) in enumerate(zip(indices.tolist(),preds,ranks)):
     row={'sample_id':ids[ix],'prediction':pred}
     if multi:row['probabilities']=probs[j]
     else:row['topk']=rank
     predictions.append(row);writer.writerow(dict(sample_id=ids[ix],prediction=json.dumps(pred),topk=json.dumps(rank),batch_index=bi,forward_ms_per_image=ms/len(indices)))
    tw.writerow(dict(batch_index=bi,sample_count=len(indices),forward_ms=ms));timings.append(ms);done+=len(indices)
    print(f'Inference {done}/{len(ds)}',flush=True)
 finally:
  if logits_file is not None:logits_file.flush();del logits_file
 api.synchronize(device);wall=time.perf_counter()-start
 if done!=len(ds):raise RuntimeError('Incomplete inference')
 dump(bundle/'predictions.json',predictions)
 protocol={'schema':'chexpert-auroc-v1' if multi else 'model-eval/score-classification/v1','task':'multi_label_classification' if multi else 'single_label_classification','scoring_ready':True,'evaluation':{'threshold':0.5,'comparison':'strict_greater','uncertain_gt':'reject','auc_single_class':'unavailable'} if multi else POLICY,'model':{'id':ARGS.model_name,'source':str(ARGS.weights.absolute())},'dataset':{'name':name,'root':str(data.DATASETS[name]['root']),'scope':'debug_subset' if ARGS.max_samples else 'full_native_split'},'completed_sample_ids':sorted(ids),'prediction_count':len(predictions),'artifact_sha256':{x:hashlib.sha256((bundle/x).read_bytes()).hexdigest() for x in ['annotations.json','predictions.json']}}
 dump(bundle/'protocol.json',protocol)
 dump(out/'inference_summary.json',dict(sample_count=done,device=str(device),batch_size=ARGS.batch_size,num_workers=ARGS.num_workers,warmup_steps=ARGS.warmup_steps,parameter_dtype=str(next(model.parameters()).dtype),forward_seconds=sum(timings)/1000,forward_ms_per_image=sum(timings)/done,end_to_end_seconds=wall,timing={'forward':'device Events around model(images) only','end_to_end':'formal data iteration through output saves; excludes loading,warmup,offline score'},preprocessing=str(prepared.get('transform',prepared.get('processor')))))
 dump(out/'model_environment.json',{'python':sys.version,'packages':{x:importlib.metadata.version(x) for x in ['torch','torchvision','transformers','timm','numpy'] if list(importlib.metadata.distributions(name=x))}})
 (out/'model_structure.txt').write_text(str(model),encoding='utf-8')
 dump(out/'inference_output.json',{'state':'completed','task':'classification','bundles':['scoring_bundle']})
 return {'state':'completed'}
