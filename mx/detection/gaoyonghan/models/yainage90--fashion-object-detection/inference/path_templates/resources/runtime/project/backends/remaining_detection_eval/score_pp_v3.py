"""Score saved PP V3 predictions after converting polygon detections to COCO RLE."""
import json
from pathlib import Path
import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from pycocotools import mask as mask_utils
from run_model import core,metrics,file_hash,digest
from run_pp_layout import DATA,GT_MAP,NAMES,CAT

SOURCE=Path('__MODEL_ROOT__/resources/runtime/assets/workspace/results/remaining25_pp_v2/PP-DocLayoutV3_safetensors')
OUT=Path('__MODEL_ROOT__/resources/runtime/assets/workspace/results/remaining25_pp_v3/PP-DocLayoutV3_safetensors')

def main():
    prior=json.loads((SOURCE/'status.json').read_text());records=core.read_jsonl(SOURCE/'predictions.jsonl')
    assert len(records)==1651 and {r['image_id'] for r in records}==set(range(1,1652))
    assert prior['completed_images']==1651
    pages=json.loads((DATA/'OmniDocBench.json').read_text())
    assert file_hash(DATA/'OmniDocBench.json')==prior['protocol']['dataset_sha256']
    ann={'images':[],'annotations':[],'categories':[{'id':CAT[n],'name':n} for n in NAMES],'info':{},'licenses':[]}
    for index,page in enumerate(pages,1):
        info=page['page_info'];ann['images'].append({'id':index,'file_name':info['image_path'],'width':info['width'],'height':info['height']})
        for r in page['layout_dets']:
            category=r['category_type']
            if category in GT_MAP: cats=[CAT[GT_MAP[category]]];crowd=int(r.get('ignore',False))
            elif category.endswith('_mask') or category=='abandon':cats=list(CAT.values());crowd=1
            else:continue
            poly=r['poly'];x1,y1,x2,y2=min(poly[::2]),min(poly[1::2]),max(poly[::2]),max(poly[1::2])
            for cat in cats:
                ann['annotations'].append({'id':len(ann['annotations'])+1,'image_id':index,'category_id':cat,
                  'bbox':[x1,y1,x2-x1,y2-y1],'segmentation':[poly],'area':(x2-x1)*(y2-y1),'iscrowd':crowd})
    predictions=[p for r in records for p in r['detections']]
    result={'bbox':metrics(ann,predictions),'unit':'fraction_0_to_1','reference_check':'not_officially_comparable'}
    counts=np.array([r['order_counts'] for r in records],dtype=np.int64).sum(axis=0).tolist()
    result['reading_order']={'correct_pairs':counts[0],'matched_pairs':counts[1],'all_gt_pairs':counts[2],
      'matched_elements':counts[3],'gt_ordered_elements':counts[4],'pair_accuracy':counts[0]/counts[1],
      'matched_element_recall':counts[3]/counts[4],'end_to_end_correct_pair_fraction':counts[0]/counts[2]}
    sizes={i['id']:(i['height'],i['width']) for i in ann['images']}
    for a in ann['annotations']:
        a['area']=float(mask_utils.area(mask_utils.merge(mask_utils.frPyObjects(a['segmentation'],*sizes[a['image_id']]))))
    gt=COCO();gt.dataset=ann;gt.createIndex()
    masks=[]
    for p in predictions:
        rle=mask_utils.merge(mask_utils.frPyObjects(p['segmentation'],*sizes[p['image_id']]))
        masks.append({'image_id':p['image_id'],'category_id':p['category_id'],'score':p['score'],'segmentation':rle})
    dt=gt.loadRes(masks);evaluator=COCOeval(gt,dt,'segm');evaluator.evaluate();evaluator.accumulate();evaluator.summarize()
    result.update(polygon_segm_AP=float(evaluator.stats[0]),polygon_segm_AP50=float(evaluator.stats[1]))
    protocol={'sequence':39,'source_protocol':prior['protocol'],'source_protocol_fingerprint':prior['protocol_fingerprint'],
      'predictions':str(SOURCE/'predictions.jsonl'),'predictions_sha256':file_hash(SOURCE/'predictions.jsonl'),
      'scorer_sha256':file_hash(Path(__file__)),'scoring_fix':'Convert predicted polygons to RLE for loadRes; true polygon area for segmentation size bins; inference and ground-truth coordinates unchanged.'}
    core.atomic_json(OUT/'metrics.json',result)
    core.atomic_json(OUT/'status.json',{'sequence':39,'model_id':'PaddlePaddle/PP-DocLayoutV3_safetensors','state':'complete',
      'completed_images':1651,'total_images':1651,'protocol':protocol,'protocol_fingerprint':digest(protocol),'metrics':result})
    print(result,flush=True)

if __name__=='__main__':main()
