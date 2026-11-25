"""Marigold原pipeline不变；网络组件计时与正式读图到保存计时分开。"""
import csv,json,time,functools
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from .data import _DepthEvaluationDataset,_collate_as_list


def _readable_timing_fields(value):
    """Add readable aliases; preserve measured timing boundaries and legacy keys."""
    if not isinstance(value, dict):
        return value
    for child in value.values():
        if isinstance(child, dict):
            _readable_timing_fields(child)
    if 'mean_forward_ms' in value:
        value['forward_ms_per_image'] = value['mean_forward_ms']
    if 'forward_fps' in value:
        value['forward_images_per_second'] = value['forward_fps']
    if 'total_forward_seconds' in value:
        value['forward_seconds'] = value['total_forward_seconds']
    datasets = value.get('datasets')
    if isinstance(datasets, dict) and datasets:
        rows = list(datasets.values())
        rows = [r.get('evaluation', r) for r in rows]
        counts = [r.get('forward_recorded_images', r.get('stage1_completed_images', r.get('num_samples', r.get('sample_count')))) for r in rows]
        seconds = [r.get('total_forward_seconds', r.get('forward_seconds')) for r in rows]
        if all(n is not None for n in counts) and all(s is not None for s in seconds):
            n, total = sum(counts), sum(seconds)
            overall = value.setdefault('overall', {})
            info = overall.get('evaluation', overall)
            info['forward_seconds'] = total
            info['forward_ms_per_image'] = total * 1000 / n if n else None
            info['forward_images_per_second'] = n / total if total > 0 else None
    return value

class NetworkTimer:
    """每次网络调用记录设备事件；读取事件在pipeline返回后，不改变随机数或网络输出。"""
    def __init__(self,pipe,device):
        self.api=torch.cuda if device.type=='cuda' else torch.npu
        self.events=[]
        self.original=[]
        targets=[(pipe.vae,'encode','vae_encode'),(pipe.vae,'decode','vae_decode'),(pipe.unet,'forward','unet')]
        if getattr(pipe,'text_encoder',None) is not None:targets.append((pipe.text_encoder,'forward','text_encoder'))
        for module,name,stage in targets:
            fn=getattr(module,name);self.original.append((module,name,fn))
            def wrap(fn,stage):
                @functools.wraps(fn)
                def measured(*args,**kwargs):
                    begin,end=self.api.Event(enable_timing=True),self.api.Event(enable_timing=True)
                    begin.record();result=fn(*args,**kwargs);end.record()
                    self.events.append((stage,begin,end))
                    return result
                return measured
            setattr(module,name,wrap(fn,stage))
    def reset(self):self.events.clear()
    def read(self):
        self.api.synchronize()
        stages={}
        for name,start,end in self.events:stages[name]=stages.get(name,0.)+start.elapsed_time(end)
        if 'unet' not in stages or 'vae_decode' not in stages:raise RuntimeError('Network timing incomplete; do not report pipeline wall time as forward')
        return sum(stages.values()),stages
    def close(self):
        for module,name,fn in self.original:setattr(module,name,fn)

def run(pipe,a,device):
    out=a.output_dir;out.mkdir(parents=True,exist_ok=False)
    cfg={'datasets':[a.dataset],'dataset_root':str(a.dataset_root.resolve()),'prediction_type':'scale-and-shift-invariant depth','max_samples':a.max_samples,'evaluation_scope':'debug_subset' if a.max_samples else 'full_validation','num_inference_steps':a.steps,'ensemble_size':a.ensemble,'processing_resolution':a.processing_resolution,'seed':a.seed,'dtype':str(pipe.unet.dtype),'forward_scope':'sum of VAE encode, all UNet calls, VAE decode and actual text encoder calls; excludes scheduler/ensemble and external CPU processing'}
    (out/'evaluation_config.json').write_text(json.dumps(_readable_timing_fields(cfg),indent=2))
    ds=_DepthEvaluationDataset(a.dataset,a.dataset_root/a.dataset,a.max_samples)
    timer=NetworkTimer(pipe,device)
    def predict(sample):
        generator=torch.Generator(device=device).manual_seed(a.seed+int(sample['index']))
        return pipe(sample['image'],num_inference_steps=a.steps,ensemble_size=a.ensemble,processing_resolution=a.processing_resolution,match_input_resolution=True,resample_method_input='bilinear',resample_method_output='bilinear',batch_size=1,generator=generator,output_type='np',output_uncertainty=False)
    with torch.inference_mode():
        warm=ds[0]
        for _ in range(a.warmup_runs):predict(warm)
        timer.api.synchronize();timer.reset()
        d=out/a.dataset;(d/'raw_predictions').mkdir(parents=True)
        loader=DataLoader(ds,batch_size=1,shuffle=False,num_workers=a.num_workers,collate_fn=_collate_as_list,multiprocessing_context='spawn' if a.num_workers else None)
        fields=['sample_id','image_path','depth_path','mask_path','raw_prediction_path','height','width','forward_ms','forward_timing_kind']
        times=[];stages_all={};start=time.perf_counter()
        try:
            with (d/'inference_records.csv').open('w',newline='') as f:
                writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
                for batch in loader:
                    s=batch[0];timer.reset();result=predict(s);ms,stages=timer.read()
                    pred=np.asarray(result.prediction if hasattr(result,'prediction') else result[0]).squeeze().astype(np.float32)
                    if pred.shape!=tuple(s['output_shape']) or not np.isfinite(pred).all():raise ValueError('Invalid prediction shape or values')
                    rel=Path(a.dataset)/'raw_predictions'/f"{s['index']:08d}.npy";np.save(out/rel,pred)
                    writer.writerow(dict(sample_id=s['id'],image_path=s['image_path'],depth_path=s['depth_path'],mask_path=ds.samples[s['index']].get('mask',''),raw_prediction_path=str(rel),height=pred.shape[0],width=pred.shape[1],forward_ms=ms,forward_timing_kind='network_components_sum'))
                    times.append(ms)
                    for key,value in stages.items():stages_all[key]=stages_all.get(key,0.)+value
                f.flush();timer.api.synchronize();wall=time.perf_counter()-start
        finally:timer.close()
    seconds=sum(times)/1000
    summary={'datasets':{a.dataset:{'sample_count':len(times),'mean_forward_ms':sum(times)/len(times),'median_forward_ms':float(np.median(times)),'p95_forward_ms':float(np.percentile(times,95)),'forward_seconds':seconds,'forward_fps':len(times)/seconds,'stage1_dataset_seconds':wall,'stage1_dataset_fps':len(times)/wall,'component_forward_ms':stages_all}},'overall':{'sample_count':len(times),'stage1_global_seconds':wall,'mean_forward_ms':sum(times)/len(times),'forward_fps':len(times)/seconds}}
    (out/'inference_summary.json').write_text(json.dumps(_readable_timing_fields(summary),indent=2))
    (out/'inference_output.json').write_text(json.dumps(_readable_timing_fields({'state':'completed','task':'depth-estimation','datasets':[a.dataset]})))
    print(json.dumps(_readable_timing_fields(summary)))
