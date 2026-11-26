"""完整预测与纯网络/提取端到端计时；任务evaluate.py独立读取GT评分。"""
import csv,json,os,time,sys,importlib.metadata as metadata
from pathlib import Path
import cv2,numpy as np,torch
from torch.utils.data import Dataset,DataLoader
from .data import _discover_samples
from .timer import Timer

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

class Images(Dataset):
 def __init__(self,root,name,adapter,limit):
  self.root,self.name,self.kind,self.processor=Path(root),name,adapter.kind,adapter.processor
  self.samples=_discover_samples(name,self.root/name)
  if limit is not None:self.samples=self.samples[:limit]
  if not self.samples:raise ValueError('No RGB/GT pairs found')
 def __len__(self):return len(self.samples)
 def __getitem__(self,i):
  from .adapter import Adapter
  s=self.samples[i];bgr=cv2.imread(str(s['image']))
  if bgr is None:raise OSError(s['image'])
  if self.name=='KITTI':
   h,w=bgr.shape[:2]
   if h<352 or w<1216:raise ValueError('KITTI image too small for benchmark crop')
   bgr=bgr[h-352:,(w-1216)//2:(w-1216)//2+1216]
  rgb=cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB)
  prep=object.__new__(Adapter);prep.kind,prep.processor=self.kind,self.processor
  inputs=prep.prepare(rgb,self.name)
  row=dict(sample_id=s['id'],image_path=str(s['image'].relative_to(self.root)),depth_path=str(s['depth'].relative_to(self.root)),mask_path=str(s['mask'].relative_to(self.root)) if 'mask' in s else '',height=rgb.shape[0],width=rgb.shape[1])
  return inputs,row

def collate(xs):return xs

def run(adapter,a,api):
 output=a.output_dir;output.mkdir(parents=True,exist_ok=False)
 timer=Timer(api);name=a.dataset;dataset=Images(a.dataset_root,name,adapter,a.max_samples)
 if adapter.kind=='lotus':
  for obj,method in [(adapter.model.vae,'encode'),(adapter.model.vae,'decode'),(adapter.model.unet,'forward'),(adapter.model.text_encoder,'forward')]:timer.wrap(obj,method)
 first=adapter.inputs(dataset[0][0],name)
 cpu_state=torch.random.get_rng_state();device_state=api.get_rng_state()
 try:
  with torch.inference_mode():
   for _ in range(a.warmup_steps):adapter.predict(first,timer)
  api.synchronize();torch.random.set_rng_state(cpu_state);api.set_rng_state(device_state);del first
  api.reset_peak_memory_stats()
  loader=DataLoader(dataset,batch_size=1,shuffle=False,num_workers=a.num_workers,collate_fn=collate,multiprocessing_context='spawn' if a.num_workers else None)
  rows=[];start=time.perf_counter();timer.enabled=True
  with torch.inference_mode():
   for batch in loader:
    cpu,row=batch[0];x=adapter.inputs(cpu,name);timer.events=[]
    out=adapter.predict(x,timer);ms=timer.milliseconds()
    array=adapter.postprocess(out,(row['height'],row['width']),name)
    if not np.isfinite(array).all():raise ValueError('Prediction contains NaN/Inf')
    relative=Path(name)/'raw_predictions'/(row['sample_id']+'.npy');(output/relative).parent.mkdir(parents=True,exist_ok=True);np.save(output/relative,np.asarray(array,dtype=np.float32))
    row.update(dataset=name,raw_prediction_path=str(relative),input_height=x.shape[-2],input_width=x.shape[-1],forward_ms=ms,forward_timing_kind='two_network_calls_with_flip' if adapter.kind=='zoe' else 'network_components_sum' if adapter.kind=='lotus' else 'single_image')
    rows.append(row);print(f'Inference {len(rows)}/{len(dataset)}',flush=True)
  with (output/name/'inference_records.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
  api.synchronize();wall=time.perf_counter()-start
  times=[r['forward_ms'] for r in rows];seconds=sum(times)/1000
  info=dict(num_samples=len(rows),stage1_completed_images=len(rows),forward_recorded_images=len(rows),stage1_status='completed',accuracy_status='not_run',total_forward_seconds=seconds,mean_forward_ms=float(np.mean(times)),median_forward_ms=float(np.median(times)),p95_forward_ms=float(np.percentile(times,95)),forward_fps=len(rows)/seconds,stage1_dataset_seconds=wall,stage1_dataset_fps=len(rows)/wall,peak_allocated_bytes=api.max_memory_allocated(),peak_reserved_bytes=api.max_memory_reserved())
  config=dict(datasets=[name],prediction_type='metric depth' if adapter.kind=='zoe' else 'scale-and-shift-invariant depth' if adapter.kind=='lotus' else 'affine-invariant inverse depth',dataset_root=os.path.relpath(a.dataset_root.resolve(),output.resolve()),max_samples=a.max_samples,evaluation_scope='functional_subset' if a.max_samples is not None else 'full_dataset',batch_size=1,num_workers=a.num_workers,warmup_runs=a.warmup_steps,seed=a.seed,kind=adapter.kind,attention=a.attention,parameter_dtypes=sorted({str(p.dtype) for c in adapter.model.components.values() if hasattr(c,'parameters') for p in c.parameters()}) if adapter.kind=='lotus' else sorted({str(p.dtype) for p in adapter.model.parameters()}),timing={'forward':'device events around actual network calls only; ZoeDepth sums original+flipped forwards; Lotus sums VAE encode/decode, UNet, text encoder','end_to_end':'formal data iteration through prediction and CSV save; excludes loading,warmup,offline scoring'})
  versions={}
  for pkg in ['torch','torch_npu','torchvision','transformers','diffusers','numpy','opencv-python','huggingface-hub']:
   try:versions[pkg]=metadata.version(pkg)
   except metadata.PackageNotFoundError:pass
  for file,data in [('evaluation_config.json',config),('inference_summary.json',{'datasets':{name:info},'overall':{'stage1_completed_images':len(rows)}}),('model_environment.json',{'python':sys.executable,'versions':versions,'device':str(adapter.device),'torch_num_threads':torch.get_num_threads()})]:(output/file).write_text(json.dumps(_readable_timing_fields(data),indent=2))
  (output/'model_structure.txt').write_text(str(adapter.model))
  (output/'inference_output.json').write_text(json.dumps(_readable_timing_fields({'task':'depth-estimation','state':'completed','datasets':[name]}),indent=2))
 finally:timer.close()
