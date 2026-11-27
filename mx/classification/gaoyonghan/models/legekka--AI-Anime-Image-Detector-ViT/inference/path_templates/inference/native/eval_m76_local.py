import argparse,hashlib,json,time
from pathlib import Path
import torch
from PIL import Image
from transformers import AutoImageProcessor,AutoModelForImageClassification
ROOT=Path('__DATASET_ROOT__/m76_animedl_test_v1/prepared')
MODEL=Path('__MODEL_ROOT__/weights/legekka/AI-Anime-Image-Detector-ViT')
SHARDS=['0050','0110','0240']
GENS=['FLUX1','SDXL','SD']
def key(path):return hashlib.sha256(('m76-proxy-v1:'+str(path)).encode()).hexdigest()
def manifest(per_shard):
 rows=[]
 for sh in SHARDS:
  real=sorted((ROOT/'real'/sh).iterdir(),key=key)[:per_shard]
  for p in real:rows.append({'path':str(p),'label':0,'shard':sh,'generator':'human'})
  base=sorted((ROOT/'ai'/sh/'text2image').iterdir(),key=key)[:per_shard]
  for p in base:
   gi=int(key(p)[:8],16)%3;gen=GENS[gi];f=p/(gen+'_text2image_'+p.name+'.png')
   if not f.exists():raise FileNotFoundError(f)
   rows.append({'path':str(f),'label':1,'shard':sh,'generator':gen})
 return rows

def main():
 p=argparse.ArgumentParser();p.add_argument('--per-shard',type=int,default=500);p.add_argument('--out',required=True);p.add_argument('--batch-size',type=int,default=8);p.add_argument('--threads',type=int,default=8);p.add_argument('--max-images',type=int,default=0);a=p.parse_args()
 out=Path(a.out);out.mkdir(parents=True,exist_ok=True);torch.set_num_threads(a.threads)
 rows=manifest(a.per_shard)
 if a.max_images:rows=rows[:a.max_images]
 (out/'manifest.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in rows))
 t0=time.perf_counter();proc=AutoImageProcessor.from_pretrained(str(MODEL),local_files_only=True);mod=AutoModelForImageClassification.from_pretrained(str(MODEL),local_files_only=True);device='cuda' if torch.cuda.is_available() else 'cpu';mod.to(device).eval();load_s=time.perf_counter()-t0
 conf=[[0,0],[0,0]];bygen={};fwd=prep=0.;n=0
 with (out/'predictions.jsonl').open('w') as pred,torch.inference_mode():
  for i in range(0,len(rows),a.batch_size):
   batch=rows[i:i+a.batch_size];ts=time.perf_counter();imgs=[]
   for x in batch:
    with Image.open(x['path']) as im:imgs.append(im.convert('RGB'))
   inputs=proc(images=imgs,return_tensors='pt').to(device);prep+=time.perf_counter()-ts
   ts=time.perf_counter();logits=mod(**inputs).logits
   if device=='cuda':torch.cuda.synchronize()
   fwd+=time.perf_counter()-ts
   probs=logits.softmax(dim=1).cpu().tolist()
   for x,pr in zip(batch,probs):
    y=x['label'];yh=max(range(len(pr)),key=lambda j:pr[j]);conf[y][yh]+=1;bygen.setdefault(x['generator'],[0,0]);bygen[x['generator']][0]+=int(yh==y);bygen[x['generator']][1]+=1;n+=1
    pred.write(json.dumps({**x,'prediction':yh,'confidence_ai':pr[1]})+'\n')
   if n%100==0 or n==len(rows):print(f'm76 {n}/{len(rows)}',flush=True)
 tp=conf[1][1];fp=conf[0][1];fn=conf[1][0];tn=conf[0][0];prec=tp/(tp+fp) if tp+fp else None;rec=tp/(tp+fn) if tp+fn else None
 result={'model_number':76,'model_id':'legekka/AI-Anime-Image-Detector-ViT','dataset':'AnimeDL-2M local prepared test subset','shards':SHARDS,'per_shard_per_label':a.per_shard,'images':n,'confusion_matrix_real_ai':conf,'accuracy':(tp+tn)/n,'precision_ai':prec,'recall_ai':rec,'f1_ai':2*prec*rec/(prec+rec) if prec and rec else None,'by_generator':{k:{'correct':v[0],'images':v[1],'accuracy':v[0]/v[1]} for k,v in bygen.items()},'device':device,'load_seconds':load_s,'forward_seconds':fwd,'preprocess_seconds':prep,'total_seconds':time.perf_counter()-t0,'scope':'balanced local test-shard subset; excludes inpainting','officially_comparable':False,'manifest_sha256':hashlib.sha256((out/'manifest.jsonl').read_bytes()).hexdigest(),'predictions_sha256':hashlib.sha256((out/'predictions.jsonl').read_bytes()).hexdigest()}
 (out/'metrics.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n');print(json.dumps(result,ensure_ascii=False),flush=True)
if __name__=='__main__':main()
