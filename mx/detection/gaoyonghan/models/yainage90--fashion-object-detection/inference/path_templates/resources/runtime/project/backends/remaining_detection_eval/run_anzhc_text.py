"""Text-segmentation subtask of the multi-task Anzhc repository on OmniDocBench."""
import argparse
import fcntl
import json
from pathlib import Path
import time
from PIL import Image
from pycocotools import mask as mask_utils
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from run_model import core,file_hash,digest,metrics

DATA=Path('__DATASET_ROOT__/omnidocbench_eval')
TEXT={'text_block','title','figure_caption','table_caption','header','footer','page_number','page_footnote','reference','figure_footnote'}

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--max-images',type=int,default=10);args=parser.parse_args()
    out=Path('__MODEL_ROOT__/resources/runtime/assets/workspace/results/remaining25_anzhc_v3/text_seg');out.mkdir(parents=True,exist_ok=True)
    with (out/'.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        weight=Path('__MODEL_ROOT__/weights/Anzhc/Anzhcs_YOLOs/Anzhcs-text-seg-v9-y11m.pt')
        protocol={'sequence':8,'weight_sha256':file_hash(weight),'script_sha256':file_hash(Path(__file__)),'gt_sha256':file_hash(DATA/'OmniDocBench.json'),
          'dataset_relation':'public_replacement','fidelity':'reference_baseline','input_size':640,'conf':.001,'iou':.7,'max_det':300,'half':False,'device':'cuda:0',
          'gt_text_classes':sorted(TEXT),'prediction_class':'Inline text (1 of 9 native classes)','known_differences':['Document paragraph/line region masks, not private illustration text mask protocol.','Text masks are genuine annotated polygons; no bounding-box-as-GT-mask or pseudo labels.'],
          'mask_protocol':'Native retina_masks=False; result.masks.xy contours in original image coordinates, converted to COCO RLE one at a time.','runtime_versions':core.runtime_versions()}
        fp=digest(protocol);path=out/'status.json';old=json.loads(path.read_text()) if path.exists() else None
        if old and old['protocol_fingerprint']!=fp:raise RuntimeError('Changed protocol')
        if old and old['state']=='complete':return
        if args.max_images==0 and not(old and old.get('smoke_passed')):raise RuntimeError('Run smoke first')
        status={'sequence':8,'model_id':'Anzhc/Anzhcs_YOLOs','task':'text_seg','state':'running','protocol':protocol,'protocol_fingerprint':fp,'started_at':time.time()};core.atomic_json(path,status)
        try:
            from ultralytics import YOLO
            model=YOLO(str(weight));assert model.task=='segment' and model.names[1]=='Inline text',model.names
            pages=json.loads((DATA/'OmniDocBench.json').read_text());ann={'images':[],'annotations':[],'categories':[{'id':1,'name':'text'}],'info':{},'licenses':[]}
            records=core.read_jsonl(out/'predictions.jsonl');done={r['image_id'] for r in records}
            for index,page in enumerate(pages,1):
                info=page['page_info'];ann['images'].append({'id':index,'width':info['width'],'height':info['height'],'file_name':info['image_path']})
                for r in page['layout_dets']:
                    if r['category_type'] not in TEXT and not(r['category_type'].endswith('_mask') or r['category_type']=='abandon'):continue
                    poly=r['poly'];x1,y1,x2,y2=min(poly[::2]),min(poly[1::2]),max(poly[::2]),max(poly[1::2]);crowd=r['category_type'] not in TEXT or r.get('ignore',False)
                    ann['annotations'].append({'id':len(ann['annotations'])+1,'image_id':index,'category_id':1,'bbox':[x1,y1,x2-x1,y2-y1],'segmentation':[poly],'area':(x2-x1)*(y2-y1),'iscrowd':int(crowd)})
                if index in done or (args.max_images and index>args.max_images):continue
                result=model.predict(Image.open(DATA/'images'/Path(info['image_path']).name).convert('RGB'),imgsz=640,device='cuda:0',conf=.001,iou=.7,max_det=300,half=False,retina_masks=False,verbose=False)[0]
                predictions=[]
                if result.masks is not None:
                    assert len(result.masks)==len(result.boxes)
                    for box,score,label,mask in zip(result.boxes.xyxy.cpu().tolist(),result.boxes.conf.cpu().tolist(),result.boxes.cls.cpu().tolist(),result.masks.xy):
                        if int(label)!=1:continue
                        import numpy as np
                        if len(mask)>=3:
                            rle=mask_utils.merge(mask_utils.frPyObjects([mask.reshape(-1).tolist()],info['height'],info['width']))
                        else:
                            rle=mask_utils.encode(np.zeros((info['height'],info['width']),dtype=np.uint8,order='F'))
                        rle['counts']=rle['counts'].decode()
                        x1,y1,x2,y2=box;predictions.append({'image_id':index,'category_id':1,'score':score,'bbox':[x1,y1,x2-x1,y2-y1],'segmentation':rle})
                core.append_jsonl(out/'predictions.jsonl',{'image_id':index,'detections':predictions})
                if index==1 or index%100==0 or index==args.max_images or index==len(pages):print('Anzhc text',index,'/',args.max_images or len(pages),flush=True)
            records=core.read_jsonl(out/'predictions.jsonl');status.update(state='partial_complete',smoke_passed=True,completed_images=len(records),total_images=1651)
            if args.max_images==0:
                assert len(records)==1651 and {r['image_id'] for r in records}==set(range(1,1652))
                pred=[p for r in records for p in r['detections']];score={'bbox':metrics(ann,pred),'all_model_tasks_complete':False,'unit':'fraction_0_to_1'}
                sizes={i['id']:(i['height'],i['width']) for i in ann['images']}
                for a in ann['annotations']:a['area']=float(mask_utils.area(mask_utils.merge(mask_utils.frPyObjects(a['segmentation'],*sizes[a['image_id']]))))
                gt=COCO();gt.dataset=ann;gt.createIndex();dt=gt.loadRes([{k:v for k,v in p.items() if k!='bbox'} for p in pred]);e=COCOeval(gt,dt,'segm');e.evaluate();e.accumulate();e.summarize()
                score.update(segm_AP=float(e.stats[0]),segm_AP50=float(e.stats[1]));core.atomic_json(out/'metrics.json',score);status.update(state='complete',metrics=score,all_model_tasks_complete=False)
        except BaseException as exc:status.update(state='failed',error=repr(exc));raise
        finally:status['elapsed_seconds']=time.time()-status['started_at'];core.atomic_json(path,status)

if __name__=='__main__':main()
