"""OmniDocBench replacement evaluation: detection, polygons and reading order."""
import argparse
import fcntl
import importlib.metadata
import json
import time
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.optimize import linear_sum_assignment

from run_model import core, digest, file_hash, metrics

DATA = Path('__DATASET_ROOT__/omnidocbench_eval')
GT_MAP = {'text_block':'text','title':'title','figure':'image','figure_caption':'caption',
          'table':'table','table_caption':'caption','equation_isolated':'formula',
          'header':'header','footer':'footer','page_number':'number',
          'page_footnote':'footnote','reference':'reference','figure_footnote':'footnote'}
PRED_MAP = {'text':'text','abstract':'text','aside_text':'text','content':'text','doc_title':'title',
            'paragraph_title':'title','image':'image','figure_title':'caption','table':'table',
            'formula':'formula','header':'header','footer':'footer','number':'number',
            'footnote':'footnote','vision_footnote':'footnote','reference':'reference','reference_content':'reference'}
NAMES = list(dict.fromkeys(GT_MAP.values()))
CAT = {n:i+1 for i,n in enumerate(NAMES)}


def order_counts(gt, predictions):
    gt = [g for g in gt if g.get('order') is not None and not g.get('iscrowd')]
    predictions = [p for p in predictions if p.get('order') is not None]
    total_pairs = sum(a['order'] != b['order'] for i,a in enumerate(gt) for b in gt[i+1:])
    if not gt or not predictions:
        return [0,0,total_pairs,0,len(gt)]
    ious = np.zeros((len(gt),len(predictions)))
    for i,g in enumerate(gt):
        gx,gy,gw,gh=g['bbox']
        for j,p in enumerate(predictions):
            if g['category_id'] != p['category_id']:
                continue
            px,py,pw,ph=p['bbox']
            inter=max(0,min(gx+gw,px+pw)-max(gx,px))*max(0,min(gy+gh,py+ph)-max(gy,py))
            ious[i,j]=inter/(gw*gh+pw*ph-inter) if inter else 0
    rows,cols=linear_sum_assignment(-ious)
    pairs=[(gt[i]['order'],predictions[j]['order']) for i,j in zip(rows,cols) if ious[i,j]>=0.5]
    correct=count=0
    for i,(ga,pa) in enumerate(pairs):
        for gb,pb in pairs[i+1:]:
            if ga==gb:
                continue
            count+=1
            correct+=int((ga-gb)*(pa-pb)>0)
    return [correct,count,total_pairs,len(pairs),len(gt)]


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--model',type=int,choices=[38,39],required=True)
    parser.add_argument('--max-images',type=int,default=10)
    parser.add_argument('--gpu',type=int,default=0)
    parser.add_argument('--output-root',type=Path,default=Path('__MODEL_ROOT__/resources/runtime/assets/workspace/results/remaining25_pp_v1'))
    args=parser.parse_args()
    name='PP-DocLayoutV2' if args.model==38 else 'PP-DocLayoutV3_safetensors'
    model_dir=Path('__MODEL_ROOT__/weights/PaddlePaddle')/name
    out=args.output_root/name
    out.mkdir(parents=True,exist_ok=True)
    with (out/'.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        pages=json.loads((DATA/'OmniDocBench.json').read_text())
        assert len(pages)==1651
        versions={}
        for package in ['transformers','paddlepaddle-gpu','paddlex','paddleocr','torch','numpy','pycocotools','scipy']:
            try: versions[package]=importlib.metadata.version(package)
            except importlib.metadata.PackageNotFoundError: pass
        protocol={'sequence':args.model,'model_id':'PaddlePaddle/'+name,'dataset':'OmniDocBench local 1651-page snapshot',
                  'dataset_sha256':file_hash(DATA/'OmniDocBench.json'),'script_sha256':file_hash(Path(__file__)),
                  'weights':{p.name:file_hash(p) for p in model_dir.iterdir() if p.suffix in {'.safetensors','.pdiparams','.json','.yml'}},
                  'gt_mapping':GT_MAP,'prediction_mapping':PRED_MAP,'score_threshold':0.3,
                  'polygon_protocol':'Original annotated polygons; no prediction-derived ground truth.',
                  'order_protocol':'Hungarian class-aware bbox IoU>=0.5; pair accuracy plus matched-pair/GT coverage.',
                  'runtime_versions':versions,'gpu':args.gpu,'fidelity':'reference_baseline','dataset_relation':'public_replacement'}
        if args.model==39:
            from transformers.models.pp_doclayout_v3 import image_processing_pp_doclayout_v3_fast
            protocol['postprocessor_sha256']=file_hash(Path(image_processing_pp_doclayout_v3_fast.__file__))
            protocol['empty_mask_fallback']='Predicted rectangle for empty crops/contours, consistent with native degenerate-polygon fallback.'
        fingerprint=digest(protocol)
        status_path=out/'status.json'
        old=json.loads(status_path.read_text()) if status_path.exists() else None
        if old and old['protocol_fingerprint']!=fingerprint: raise RuntimeError('Protocol changed; preserve old output')
        if old and old['state']=='complete':return
        if args.max_images==0 and not(old and old.get('smoke_passed')): raise RuntimeError('First run 10-image smoke')
        status={'sequence':args.model,'model_id':protocol['model_id'],'state':'running','protocol':protocol,
                'protocol_fingerprint':fingerprint,'started_at':time.time()}
        core.atomic_json(status_path,status)
        try:
            if args.model==38:
                from paddleocr import LayoutDetection
                model=LayoutDetection(model_name=name,model_dir=str(model_dir),device=f'gpu:{args.gpu}')
                def predict(image_path):
                    result=model.predict(str(image_path),batch_size=1,threshold=0.3)[0].json['res']['boxes']
                    return [{'label':b['label'],'score':b['score'],'box':b['coordinate'],'order':b.get('order')} for b in result]
            else:
                import torch
                from transformers import AutoImageProcessor,AutoModelForObjectDetection
                processor=AutoImageProcessor.from_pretrained(model_dir,local_files_only=True,use_fast=True)
                model,info=AutoModelForObjectDetection.from_pretrained(model_dir,local_files_only=True,output_loading_info=True)
                assert not info['missing_keys'] and not info['unexpected_keys'] and not info['mismatched_keys'],info
                device=f'cuda:{args.gpu}'
                model.eval().to(device)
                def predict(image_path):
                    image=Image.open(image_path).convert('RGB')
                    with torch.inference_mode():
                        outputs=model(**processor(images=image,return_tensors='pt').to(device))
                    result=processor.post_process_object_detection(outputs,threshold=0.3,target_sizes=[(image.height,image.width)])[0]
                    return [{'label':model.config.id2label[int(label)],'score':float(score),'box':box.cpu().tolist(),
                             'polygon':np.asarray(poly).reshape(-1).tolist(),'order':i} for i,(label,score,box,poly) in
                            enumerate(zip(result['labels'],result['scores'],result['boxes'],result['polygon_points']))]
            annotation={'images':[],'annotations':[],'categories':[{'id':CAT[n],'name':n} for n in NAMES],'info':{},'licenses':[]}
            records=core.read_jsonl(out/'predictions.jsonl')
            existing={r['image_id'] for r in records}
            assert len(existing)==len(records)
            for index,page in enumerate(pages,1):
                info=page['page_info']
                annotation['images'].append({'id':index,'file_name':info['image_path'],'width':info['width'],'height':info['height']})
                gold=[]
                for region in page['layout_dets']:
                    category=region['category_type']
                    if category not in GT_MAP:
                        if not (category.endswith('_mask') or category=='abandon'):continue
                        poly=region['poly'];x1,y1,x2,y2=min(poly[::2]),min(poly[1::2]),max(poly[::2]),max(poly[1::2])
                        for cat in CAT.values():
                            annotation['annotations'].append({'id':len(annotation['annotations'])+1,'image_id':index,
                                'category_id':cat,'bbox':[x1,y1,x2-x1,y2-y1],'segmentation':[poly],
                                'area':(x2-x1)*(y2-y1),'iscrowd':1,'order':None})
                        continue
                    poly=region['poly'];x1,y1,x2,y2=min(poly[::2]),min(poly[1::2]),max(poly[::2]),max(poly[1::2])
                    item={'id':len(annotation['annotations'])+1,'image_id':index,'category_id':CAT[GT_MAP[region['category_type']]],
                          'bbox':[x1,y1,x2-x1,y2-y1],'segmentation':[poly],'area':(x2-x1)*(y2-y1),
                          'iscrowd':int(region.get('ignore',False)),'order':region.get('order')}
                    annotation['annotations'].append(item);gold.append(item)
                if index in existing or (args.max_images and index>args.max_images):continue
                predictions=[]
                for r in predict(DATA/'images'/Path(info['image_path']).name):
                    if r['label'] not in PRED_MAP:continue
                    x1,y1,x2,y2=r['box']
                    pred={'image_id':index,'category_id':CAT[PRED_MAP[r['label']]],'score':r['score'],
                          'bbox':[x1,y1,x2-x1,y2-y1],'order':r['order']}
                    if args.model==39:
                        assert len(r['polygon'])>=6
                        pred['segmentation']=[r['polygon']]
                    predictions.append(pred)
                core.append_jsonl(out/'predictions.jsonl',{'image_id':index,'detections':predictions,'order_counts':order_counts(gold,predictions)})
                if index==1 or index%100==0 or index==args.max_images or index==len(pages): print(name,index,'/',args.max_images or len(pages),flush=True)
            records=core.read_jsonl(out/'predictions.jsonl')
            completed={r['image_id'] for r in records}
            assert len(completed)==len(records)
            assert set(range(1,(args.max_images or len(pages))+1))<=completed
            status.update(state='partial_complete',smoke_passed=True,completed_images=len(records),total_images=1651)
            if args.max_images==0:
                from pycocotools.coco import COCO
                from pycocotools.cocoeval import COCOeval
                predictions=[p for r in records for p in r['detections']]
                result={'bbox':metrics(annotation,predictions),'unit':'fraction_0_to_1','reference_check':'not_officially_comparable'}
                counts=np.array([r['order_counts'] for r in records],dtype=np.int64).sum(axis=0).tolist()
                result['reading_order']={'correct_pairs':counts[0],'matched_pairs':counts[1],'all_gt_pairs':counts[2],
                    'matched_elements':counts[3],'gt_ordered_elements':counts[4],
                    'pair_accuracy':counts[0]/counts[1] if counts[1] else None,
                    'matched_element_recall':counts[3]/counts[4] if counts[4] else None,
                    'end_to_end_correct_pair_fraction':counts[0]/counts[2] if counts[2] else None}
                if args.model==39:
                    gt=COCO();gt.dataset=annotation;gt.createIndex()
                    masks=[{k:v for k,v in p.items() if k not in {'bbox','order'}} for p in predictions]
                    dt=gt.loadRes(masks)
                    evaluator=COCOeval(gt,dt,'segm');evaluator.evaluate();evaluator.accumulate();evaluator.summarize()
                    result['polygon_segm_AP']=float(evaluator.stats[0]);result['polygon_segm_AP50']=float(evaluator.stats[1])
                core.atomic_json(out/'metrics.json',result)
                status.update(state='complete',metrics=result)
        except BaseException as exc:
            status.update(state='failed',error=repr(exc));raise
        finally:
            status['elapsed_seconds']=time.time()-status['started_at'];core.atomic_json(status_path,status)


if __name__=='__main__':
    main()
