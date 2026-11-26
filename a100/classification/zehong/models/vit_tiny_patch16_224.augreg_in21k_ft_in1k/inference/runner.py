"""Native single-label inference and standard artifacts; scoring is offline."""
import csv
import hashlib
import importlib.metadata as metadata
import inspect
import json
import sys
import time
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from inference.options import ARGS
from inference import data

POLICY = {'task':'single_label_classification','top_k':[1,5],
          'top5_when_classes_below_five':'not_applicable','top5_when_ranking_missing':'unavailable',
          'ranking':'preserve_native_order_no_new_tie_break','macro_labels':'observed_ground_truth_classes',
          'zero_division':0,'unit':'fraction','sample_weight':'uniform'}

def dump(path, value):
    Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')

def evaluate_model(model, *, dataset_name, image_processor=None, timing_mode='batched'):
    a = ARGS
    data.DATASETS[dataset_name]['root'] = a.dataset_root.absolute()
    kwargs = dict(dataset_name=dataset_name, image_processor=image_processor, timing_mode=timing_mode,
                  output_dir=a.output_dir,batch_size=a.batch_size,num_workers=a.num_workers,
                  max_samples=a.max_samples,warmup_steps=a.warmup_steps)
    prepared = data.prepare_inputs(model,**{k:v for k,v in kwargs.items() if k in inspect.signature(data.prepare_inputs).parameters})
    raw, dataset = prepared['dataset'], prepared['transformed_dataset']
    labels = [int(x) for x in raw['label']]
    num_classes = int(getattr(getattr(model,'config',None),'num_labels',getattr(model,'num_classes',0)))
    if num_classes < 1: raise ValueError('Missing classifier category count')
    id2label = getattr(getattr(model,'config',None),'id2label',{})
    native_names = getattr(raw.features.get('label'),'names',None)
    names = [str(id2label.get(i,id2label.get(str(i),native_names[i] if native_names and len(native_names)==num_classes else i))) for i in range(num_classes)]
    ids = [str(i) for i in range(len(dataset))]
    if not ids or len(labels)!=len(ids) or not np.isin(labels,range(num_classes)).all():
        raise ValueError('Empty, mismatched or out-of-range GT; no samples may be silently discarded')
    out=a.output_dir
    out.mkdir(parents=True,exist_ok=False)
    bundle=out/'scoring_bundle'
    bundle.mkdir()
    annotation={'categories':[{'id':i,'name':n} for i,n in enumerate(names)],
                'samples':[{'sample_id':sid,'label':y} for sid,y in zip(ids,labels)]}
    # Independent ground truth exists before the first model invocation.
    dump(bundle/'annotations.json',annotation)
    device=next(model.parameters()).device
    if device.type not in {'cuda','npu'}: raise RuntimeError('No CPU fallback')
    api=getattr(torch,device.type)
    model.eval()
    loader=DataLoader(dataset,batch_size=a.batch_size,shuffle=False,num_workers=a.num_workers,
                      pin_memory=device.type=='cuda',persistent_workers=a.num_workers>0,
                      multiprocessing_context='spawn' if a.num_workers else None)
    warm=next(iter(DataLoader(dataset,batch_size=a.batch_size,num_workers=0)))[0].to(device)
    with torch.inference_mode():
        for _ in range(a.warmup_steps): model(warm)
    api.synchronize(device)
    del warm
    api.reset_peak_memory_stats(device)
    begin,end=api.Event(enable_timing=True),api.Event(enable_timing=True)
    timings,predictions,seen=[],[],set()
    logits_file=None
    start=time.perf_counter()
    try:
        with torch.inference_mode(),(out/'predictions.csv').open('w',newline='',encoding='utf-8') as f,(out/'batch_timings.csv').open('w',newline='',encoding='utf-8') as t:
            writer=csv.DictWriter(f,fieldnames=['sample_id','prediction','topk','batch_index','forward_ms_per_image'])
            batches=csv.DictWriter(t,fieldnames=['batch_index','sample_count','forward_ms'])
            writer.writeheader();batches.writeheader()
            for bi,(images,_,indices) in enumerate(loader):
                indices=indices.tolist()
                if len(set(indices))!=len(indices) or seen.intersection(indices) or any(i<0 or i>=len(ids) for i in indices):
                    raise ValueError('Invalid/duplicate sample indices')
                images=images.to(device,non_blocking=device.type=='cuda')
                begin.record()
                output=model(images)
                end.record()
                end.synchronize()
                ms=begin.elapsed_time(end)
                logits=output['logits'] if isinstance(output,dict) else output.logits if hasattr(output,'logits') else output
                if logits.ndim!=2 or tuple(logits.shape)!=(len(indices),num_classes) or not torch.isfinite(logits).all():
                    raise ValueError('Invalid logits shape or nonfinite values')
                arr=(logits.float() if logits.dtype==torch.bfloat16 else logits).cpu().numpy()
                if logits_file is None:
                    logits_file=np.lib.format.open_memmap(out/'logits.npy',mode='w+',dtype=arr.dtype,shape=(len(ids),num_classes))
                logits_file[indices]=arr
                ranks=logits.topk(min(5,num_classes),dim=1).indices.cpu().tolist()
                for ix,rank in zip(indices,ranks):
                    row={'sample_id':ids[ix],'prediction':rank[0],'topk':rank}
                    predictions.append(row)
                    writer.writerow(dict(row,topk=json.dumps(rank),batch_index=bi,forward_ms_per_image=ms/len(indices)))
                batches.writerow(dict(batch_index=bi,sample_count=len(indices),forward_ms=ms))
                timings.append(ms);seen.update(indices)
                print(f'Inference {len(seen)}/{len(ids)}',flush=True)
    finally:
        if logits_file is not None: logits_file.flush();del logits_file
    api.synchronize(device)
    wall=time.perf_counter()-start
    if seen!=set(range(len(ids))): raise RuntimeError('Incomplete sample coverage')
    dump(bundle/'predictions.json',predictions)
    dump(bundle/'protocol.json',dict(schema='model-eval/score-classification/v1',task='single_label_classification',
         scoring_ready=True,evaluation=POLICY,model={'id':a.model_name,'source':str(a.weights.absolute())},
         dataset={'name':dataset_name,'root':str(a.dataset_root.absolute()),'split':data.DATASETS[dataset_name]['split'],
                  'scope':'debug_subset' if a.max_samples else 'full_native_split','label_order':'native integer labels; no reordering'},
         completed_sample_ids=sorted(ids),prediction_count=len(predictions),
         artifact_sha256={n:hashlib.sha256((bundle/n).read_bytes()).hexdigest() for n in ['annotations.json','predictions.json']}))
    dump(out/'inference_summary.json',dict(sample_count=len(seen),device=str(device),batch_size=a.batch_size,
         num_workers=a.num_workers,warmup_steps=a.warmup_steps,timing_mode=timing_mode,parameter_dtype=str(next(model.parameters()).dtype),
         forward_seconds=sum(timings)/1000,forward_ms_per_image=sum(timings)/len(seen),end_to_end_seconds=wall,
         peak_allocated_bytes=api.max_memory_allocated(device),peak_reserved_bytes=api.max_memory_reserved(device),
         preprocessing=str(prepared['transform']),class_mapping=prepared.get('class_mapping'),
         timing={'forward':'device Events around model(images) only','end_to_end':'formal iteration through CSV/logits saves; excludes load,warmup,bundle publication,offline score'}))
    dump(out/'model_environment.json',{'python':sys.version,'executable':sys.executable,'packages':{
         n:metadata.version(n) for n in ['torch','torch_npu','torchvision','transformers','timm','datasets','numpy','einops'] if list(metadata.distributions(name=n))}})
    (out/'model_structure.txt').write_text(str(model),encoding='utf-8')
    # Publish the completion manifest last; failed runs are never score-ready.
    dump(out/'inference_output.json',{'state':'completed','task':'classification','bundles':['scoring_bundle']})
