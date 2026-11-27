from model_paths import legacy as _legacy, legacy_str as _legacy_str, relative as _relative, revision as _revision, weight as _weight, result as _result
import sys,runpy,json,time
from pathlib import Path
from datetime import datetime
r=_legacy('/data/zhoujianjie');mid='HuggingFaceM4/siglip-so400m-14-980-flash-attn2-navit'
sys.argv=['bootstrap','--check-imports']
try:runpy.run_path(str(_legacy('/data/zhoujianjie/environment_setup/run_model.py')),run_name='__main__')
except SystemExit as e:
 if e.code:raise
sys.path.insert(0,str(_legacy('/data/zhoujianjie/environment_setup/native_sources/evaluation_audit')));sys.path.insert(0,str(_legacy('/data/zhoujianjie/environment_setup')))
import navit_native_a100 as n
import torch
out=_result(mid)/('native_'+datetime.now().strftime('%Y%m%dT%H%M%S'));out.mkdir(parents=True)
n.route.DATASET_ROOT=_legacy('/data/zhoujianjie/model_inference_for_GPU_adaptation/dataset/raw/mteb_flickr30k')
n.route.verify_data();assert n.receipt_revision(mid)==n.MODEL_REVISION
corpus,queries,relevant=n.flickr.load_dataset()
_limit=int(__import__('os').environ.get('ZERO_SHOT_SMOKE_LIMIT','0'))
if _limit:
 corpus=corpus[:_limit];_ids={x['id'] for x in corpus};queries=[q for q in queries if relevant[q['id']] in _ids][:_limit];relevant={q['id']:relevant[q['id']] for q in queries}
original=n.flickr.calculate_metrics
def metrics(texts,images,qs,cs,qrels):
 torch.save({'texts':texts,'images':images,'queries':qs,'corpus':cs,'qrels':qrels},out/'predictions_embeddings.pt')
 return original(texts,images,qs,cs,qrels)
n.flickr.calculate_metrics=metrics
(out/'config.json').write_text(json.dumps({'model':mid,'processor':n.PARENT_ID,'protocol':'original NaViT Flickr30k diagnostic; parent not rerun','images':len(corpus),'queries':len(queries)},indent=2))
print('MAPPING',mid,n.MODEL_REVISION,'MTEB Flickr30k original NaViT 384px parent processor; no completed parent evaluation',flush=True)
torch.cuda.synchronize();tick=time.perf_counter();d=n.evaluate(mid,True,corpus,queries,relevant);torch.cuda.synchronize()
d.update(status='completed',samples={'corpus':len(corpus),'queries':len(queries)},elapsed_seconds_synchronized=time.perf_counter()-tick)
(out/'metrics.json').write_text(json.dumps(d,indent=2))
