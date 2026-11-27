from model_paths import legacy as _legacy, legacy_str as _legacy_str, relative as _relative, revision as _revision, weight as _weight, result as _result
import sys,runpy,json,argparse,time
from pathlib import Path
from datetime import datetime
r=_legacy('/data/zhoujianjie');mid='BAAI/AltCLIP'
sys.argv=['bootstrap','--check-imports']
try:runpy.run_path(str(_legacy('/data/zhoujianjie/environment_setup/run_model.py')),run_name='__main__')
except SystemExit as e:
 if e.code:raise
sys.path.insert(0,str(_legacy('/data/zhoujianjie/environment_setup/native_sources/evaluation_audit')))
import evaluate_downloaded_retrieval as n
import torch
n.b01.model_dir=lambda m:_weight(m)
n.DATASETS['flickr30k']['directory']=_legacy('/data/zhoujianjie/model_inference_for_GPU_adaptation/dataset/raw/flickr30k_1')
out=_result(mid)/('native_'+datetime.now().strftime('%Y%m%dT%H%M%S'));out.mkdir(parents=True)
(out/'config.json').write_text(json.dumps({'model':mid,'dataset':'flickr30k','original_script':'evaluate_downloaded_retrieval.py','image_batch_size':16,'text_batch_size':256},indent=2))
old=n.recall_metrics
def capture(images,texts,captions,targets):
 torch.save({'images':images,'texts':texts,'targets':targets},out/'predictions_embeddings.pt')
 with (out/'predictions.jsonl').open('w') as stream:
  for offset in range(0,len(texts),256):
   ranks=(texts[offset:offset+256]@images.T).topk(10,dim=1).indices.tolist()
   for i,rank in enumerate(ranks):stream.write(json.dumps({'query_index':offset+i,'target':targets[offset+i],'ranking':rank})+'\n')
 return old(images,texts,captions,targets)
n.recall_metrics=capture
loader=n.b01.load_adapter;timing={}
def load(*args,**kw):
 a=loader(*args,**kw)
 for name in ['image_features','text_features']:
  old_method=getattr(a,name)
  def timed(*aa,_old=old_method,_name=name,**kk):
   torch.cuda.synchronize();tick=time.perf_counter();v=_old(*aa,**kk);torch.cuda.synchronize();timing[_name]=timing.get(_name,0.0)+time.perf_counter()-tick;return v
  setattr(a,name,timed)
 return a
n.b01.load_adapter=load
print('MAPPING',mid,'original Flickr30k 1000 images /5000 captions, original FP16 Transformers adapter',flush=True)
args=argparse.Namespace(worker_model_id=mid,worker_backend='Transformers AutoModel/AutoProcessor',worker_output=str(out/'metrics.json'),dataset='flickr30k',image_batch_size=16,text_batch_size=256)
code=n.worker(args)
d=json.loads((out/'metrics.json').read_text());d['synchronized_feature_pipeline_seconds']=timing;(out/'metrics.json').write_text(json.dumps(d,indent=2));raise SystemExit(code)
