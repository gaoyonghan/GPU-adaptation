import argparse,csv,hashlib,io,json,os,sys,time,zipfile
from pathlib import Path
import torch
from PIL import Image
from transformers import AutoImageProcessor,AutoModelForImageClassification

MODELS={85:'microsoft/swin-large-patch4-window12-384-in22k',86:'microsoft/swinv2-large-patch4-window12-192-22k',100:'facebook/convnext-base-224-22k'}
ZIP='__DATASET_ROOT__/ImageNet/imagenet-object-localization-challenge.zip'
MODEL_ROOT='__MODEL_ROOT__/weights'
SYNSETS='__MODEL_ROOT__/resources/runtime/assets/labels/imagenet22k_synsets.txt'

def main():
 p=argparse.ArgumentParser();p.add_argument('--model',type=int,choices=MODELS,required=True);p.add_argument('--out',required=True);p.add_argument('--per-class',type=int,default=0);p.add_argument('--batch-size',type=int,default=8);p.add_argument('--threads',type=int,default=8);p.add_argument('--max-images',type=int,default=0);args=p.parse_args()
 out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
 torch.set_num_threads(args.threads)
 model_dir=Path(MODEL_ROOT)/MODELS[args.model]
 t0=time.perf_counter(); processor=AutoImageProcessor.from_pretrained(str(model_dir),local_files_only=True);model=AutoModelForImageClassification.from_pretrained(str(model_dir),local_files_only=True);device='cuda' if torch.cuda.is_available() else 'cpu';model.to(device).eval();load_s=time.perf_counter()-t0
 synsets=Path(SYNSETS).read_text().splitlines();to22={w:i for i,w in enumerate(synsets)}
 with zipfile.ZipFile(ZIP) as z:
  raw=z.read('LOC_val_solution.csv').decode('utf8').splitlines();rows=list(csv.DictReader(raw));groups={}
  for row in rows:
   wnid=row['PredictionString'].split()[0]
   if wnid in to22: groups.setdefault(wnid,[]).append((row['ImageId'],to22[wnid]))
  selected=[]
  for wnid in sorted(groups):
   grp=sorted(groups[wnid]);selected.extend(grp[:args.per_class] if args.per_class else grp)
  if args.max_images: selected=selected[:args.max_images]
  valid_indices=torch.tensor(sorted(set(to22[w] for w in groups)),dtype=torch.long,device=device)
  global_correct=restricted_correct=global_top5=restricted_top5=0;forward_s=0.;read_s=0.;n=0
  pred_path=out/'predictions.jsonl'
  with pred_path.open('w') as pred,torch.inference_mode():
   for bi in range(0,len(selected),args.batch_size):
    batch=selected[bi:bi+args.batch_size];ts=time.perf_counter();imgs=[]
    for image_id,_ in batch:
     data=z.read('ILSVRC/Data/CLS-LOC/val/'+image_id+'.JPEG');imgs.append(Image.open(io.BytesIO(data)).convert('RGB'))
    inputs=processor(images=imgs,return_tensors='pt').to(device);read_s+=time.perf_counter()-ts
    ts=time.perf_counter();logits=model(**inputs).logits
    if device=='cuda':torch.cuda.synchronize()
    forward_s+=time.perf_counter()-ts
    global_preds=logits.topk(5,dim=1).indices.cpu().tolist();res_logits=logits.index_select(1,valid_indices);res_preds=valid_indices[res_logits.topk(5,dim=1).indices].cpu().tolist()
    for (image_id,gold),gp,rp in zip(batch,global_preds,res_preds):
     global_correct+=int(gp[0]==gold);restricted_correct+=int(rp[0]==gold);global_top5+=int(gold in gp);restricted_top5+=int(gold in rp);n+=1
     pred.write(json.dumps({'image_id':image_id,'gold_index':gold,'global_top5':gp,'restricted_top5':rp})+'\n')
    if n%100==0 or n==len(selected):print(f'{args.model} {n}/{len(selected)}',flush=True)
 total_s=time.perf_counter()-t0
 result={'model_number':args.model,'model_id':MODELS[args.model],'dataset':'ILSVRC2012 val','source_zip':ZIP,'scope':'ImageNet-1k mapped to ImageNet-22k head','total_1k_classes':1000,'mapped_classes':len(groups),'unmapped_synsets':sorted(set(row['PredictionString'].split()[0] for row in rows)-set(groups)),'images':n,'per_class_limit':args.per_class or None,'global_top1':global_correct/n,'restricted_top1':restricted_correct/n,'global_top5':global_top5/n,'restricted_top5':restricted_top5/n,'device':device,'load_seconds':load_s,'forward_seconds':forward_s,'data_preprocess_seconds':read_s,'total_seconds':total_s,'model_file_size':(model_dir/'model.safetensors').stat().st_size if (model_dir/'model.safetensors').exists() else None,'predictions_sha256':hashlib.sha256(pred_path.read_bytes()).hexdigest(),'officially_comparable':False}
 (out/'metrics.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n');print(json.dumps(result,ensure_ascii=False),flush=True)
if __name__=='__main__':main()
