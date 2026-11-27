import argparse,hashlib,json,time
from pathlib import Path
import torch
from PIL import Image
from transformers import AutoImageProcessor,AutoModelForImageClassification
MODEL='/data2/gaoyonghan/gpu/model/touchtech/fashion-images-gender-age-vit-large-patch16-224-in21k-v2'
LABELS=['model-female-adult','model-female-child','model-male-adult','model-male-child','pack']
def main():
 p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--threads',type=int,default=8);p.add_argument('--batch-size',type=int,default=8);args=p.parse_args()
 root=Path(args.root);rows=[json.loads(s) for s in (root/'manifest.jsonl').read_text().splitlines() if s.strip()];assert len(rows)==50
 assert {i:sum(r['label']==i for r in rows) for i in range(5)}=={i:10 for i in range(5)}
 torch.set_num_threads(args.threads);start=time.perf_counter();t=time.perf_counter()
 processor=AutoImageProcessor.from_pretrained(MODEL,local_files_only=True);model=AutoModelForImageClassification.from_pretrained(MODEL,local_files_only=True).eval();device='cuda' if torch.cuda.is_available() else 'cpu';model.to(device);load_s=time.perf_counter()-t
 assert [model.config.id2label[i] for i in range(5)]==LABELS
 preds=[];pre_s=0.;forward_s=0.;conf=[[0]*5 for _ in range(5)]
 with torch.inference_mode():
  for k in range(0,len(rows),args.batch_size):
   batch=rows[k:k+args.batch_size];t=time.perf_counter();ims=[Image.open(r['path']).convert('RGB') for r in batch];inputs=processor(images=ims,return_tensors='pt').to(device);pre_s+=time.perf_counter()-t
   t=time.perf_counter();logits=model(**inputs).logits
   if device=='cuda':torch.cuda.synchronize()
   forward_s+=time.perf_counter()-t
   for r,v in zip(batch,logits.argmax(-1).cpu().tolist()):
    conf[r['label']][v]+=1;preds.append({'id':r['id'],'gold':r['label'],'pred':v,'correct':r['label']==v,'source':r['source']})
   print(f'90 {min(k+args.batch_size,len(rows))}/{len(rows)}',flush=True)
 predpath=root/'predictions.jsonl';predpath.write_text(''.join(json.dumps(r)+'\n' for r in preds))
 metrics={'model_number':90,'model_id':'touchtech/fashion-images-gender-age-vit-large-patch16-224-in21k-v2','dataset':'PA-100K test + manually selected ImageNet-1k val garment-only images','images':len(rows),'per_class':{LABELS[i]:{'n':sum(r['label']==i for r in rows),'correct':conf[i][i],'accuracy':conf[i][i]/10} for i in range(5)},'accuracy':sum(conf[i][i] for i in range(5))/len(rows),'confusion_matrix':conf,'device':device,'load_seconds':load_s,'forward_seconds':forward_s,'data_preprocess_seconds':pre_s,'total_seconds':time.perf_counter()-start,'scope':'50-image balanced diagnostic proxy; surveillance person images vs studio-fashion model; pack uses selected isolated garment images','officially_comparable':False,'manifest_sha256':hashlib.sha256((root/'manifest.jsonl').read_bytes()).hexdigest(),'predictions_sha256':hashlib.sha256(predpath.read_bytes()).hexdigest()}
 (root/'metrics.json').write_text(json.dumps(metrics,ensure_ascii=False,indent=2)+'\n');print(json.dumps(metrics,ensure_ascii=False),flush=True)
if __name__=='__main__':main()
