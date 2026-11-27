from model_paths import legacy as _legacy, legacy_str as _legacy_str, relative as _relative, revision as _revision, weight as _weight, result as _result
from model_paths import DATA
import sys,runpy,json,os
from pathlib import Path
from datetime import datetime
root=_legacy('/data/zhoujianjie');args=sys.argv[1:];mid=args[0];small='--small' in args
sys.argv=['bootstrap','--check-imports']
try:runpy.run_path(str(_legacy('/data/zhoujianjie/environment_setup/run_model.py')),run_name='__main__')
except SystemExit as e:
 if e.code:raise
sys.path.insert(0,str(_legacy('/data/zhoujianjie/environment_setup/native_sources/evaluation_audit')));sys.path.insert(0,str(_legacy('/data/zhoujianjie/environment_setup')))
import siglip_native_a100 as native
native.ROOT=_legacy('/data/zhoujianjie/model_inference_for_GPU_adaptation');native.route.ROOT=native.ROOT;native.route.DATASET_ROOT=DATA/'raw/mteb_flickr30k'
assert mid in native.route.MODELS
weights=_weight(mid)
receipt=json.loads((weights/'download_receipt.json').read_text());assert (receipt.get('revision') or receipt.get('resolved_revision'))==native.route.MODELS[mid]['revision']
verified=native.route.verify_data()
print('MAPPING',mid,native.route.MODELS[mid]['revision'],native.route.DATASET_ID,native.route.DATASET_REVISION,'test; FP32; exact original processor and metric functions; paired baseline',flush=True)
corpus,queries,relevant=native.load_dataset()
if small:
 corpus=corpus[:8];ids={x['id'] for x in corpus};queries=[q for q in queries if relevant[q['id']] in ids][:16];relevant={q['id']:relevant[q['id']] for q in queries}
out=_result(mid)/(("small_validation_" if small else 'native_')+datetime.now().strftime('%Y%m%dT%H%M%S'))
native.RESULT_ROOT=out;out.mkdir(parents=True,exist_ok=False)
(out/'config.json').write_text(json.dumps({'model':mid,'model_revision':receipt.get('revision') or receipt.get('resolved_revision'),'dataset_verified':verified,'small_validation':small,'precision':'FP32','image_batch_size':16,'text_batch_size':256,'source':'evaluation_audit/evaluate_siglip_mteb_flickr30k_official.py'},indent=2))
result=native.evaluate_one(mid,corpus,queries,relevant,16,256)
result['small_validation']=small;result['strict_official_acceptance']=False;result['purpose']='A100_P800_same_4090_protocol'
result.update(model_name=mid,dataset_name=native.route.DATASET_ID,num_samples=len(corpus),prediction_path=str(out/'predictions.jsonl'),config_path=str(out/'config.json'))
(out/'metrics.json').write_text(json.dumps(result,indent=2));print('COMPLETED',out,json.dumps(result['measured']),flush=True)
