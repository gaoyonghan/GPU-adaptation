"""COCO WholeBody replacement: bbox and visible 2D landmark diagnostics."""
import argparse
from collections import defaultdict
import fcntl
import io
import json
from pathlib import Path
import time
import zipfile
import numpy as np
from PIL import Image
from scipy.optimize import linear_sum_assignment
from run_model import core,file_hash,digest,metrics
from qualcomm_inference import Pipeline,ROOT

DATA=Path('__DATASET_ROOT__')
GT_PATH=DATA/'coco_wholebody_val/coco_wholebody_val_v1.0.json'
# Corresponding visible nose tip and lip/eye corners; no inferred ear/tragion labels.
FACE_DENSE_MAP={1:30,61:48,291:54,33:36,133:39,362:42,263:45}
PALM_IDS=[0,1,2,5,9,13,17]

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--model',type=int,choices=[32,35],required=True);parser.add_argument('--max-images',type=int,default=10)
    parser.add_argument('--output-root',type=Path,default=Path('__MODEL_ROOT__/resources/runtime/assets/workspace/results/remaining25_qualcomm_v3'));args=parser.parse_args()
    kind='hand' if args.model==32 else 'face'
    out=args.output_root/kind;out.mkdir(parents=True,exist_ok=True)
    with (out/'.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        annotation=json.loads(GT_PATH.read_text());by_image=defaultdict(list)
        for a in annotation['annotations']:by_image[a['image_id']].append(a)
        protocol={'sequence':args.model,'model_id':f'qualcomm/MediaPipe-{kind.title()}-Detection','fidelity':'reference_baseline','dataset_relation':'public_replacement',
          'gt_sha256':file_hash(GT_PATH),'weights':{str(p.relative_to(ROOT)):file_hash(p) for p in (ROOT/f'mediapipe_{kind}-onnx-float').rglob('*') if p.is_file()},
          'source':{str(p.relative_to(ROOT)):file_hash(p) for p in (ROOT/'source').glob('*') if p.suffix in {'.py','.npy'}},
          'runner_sha256':file_hash(Path(__file__)),'inference_sha256':file_hash(Path(__file__).with_name('qualcomm_inference.py')),'geometry_sha256':file_hash(Path(__file__).with_name('qualcomm_geometry.py')),
          'runtime':{'onnxruntime':__import__('onnxruntime').__version__,'provider':'CPUExecutionProvider','torch':__import__('torch').__version__},
          'parameters':{'detector_size':256,'landmark_size':256 if kind=='hand' else 192,'conf':.001,'nms':.3,'max_det':300,'landmark_detection_threshold':.5,'landmark_score_threshold':.5},
          'mapping':PALM_IDS if kind=='hand' else FACE_DENSE_MAP,
          'known_differences':['Keep all NMS detections rather than native demo first-object-only loop; retain out-of-frame boxes.','PCK uses GT bbox diagonal normalization and class-free Hungarian bbox IoU >= .5 matching.','Coincident quantized rotation keypoints retain the detection and record landmark_failure; no NaN pose or invented landmarks.'],
          'coverage_limit':'All 21 hand x/y and anatomical handedness; no 3D gold in COCO. Palm boxes geometrically derived from seven annotated wrist/MCP/thumb-base points.' if kind=='hand' else 'Face bbox plus seven corresponding vertices of 468-point mesh. Detector six-point ears/eye centers lack exactly matching gold; not a full six/468-point benchmark.'}
        fp=digest(protocol);path=out/'status.json';old=json.loads(path.read_text()) if path.exists() else None
        if old and old['protocol_fingerprint']!=fp:raise RuntimeError('Changed protocol; preserve results')
        if old and old['state']=='complete':return
        if args.max_images==0 and not(old and old.get('smoke_passed')):raise RuntimeError('Run smoke first')
        status={'sequence':args.model,'model_id':protocol['model_id'],'task':'coco_2d','state':'running','protocol':protocol,'protocol_fingerprint':fp,'started_at':time.time()}
        core.atomic_json(path,status)
        try:
            pipeline=Pipeline(kind)
            ann={'images':annotation['images'],'categories':[{'id':1,'name':'palm_proxy' if kind=='hand' else 'face'}],'annotations':[]}
            golds={}
            for image in annotation['images']:
                gold=[]
                for a in by_image[image['id']]:
                    for side in (['left','right'] if kind=='hand' else ['face']):
                        prefix=side+'hand' if kind=='hand' else 'face'
                        if not a[prefix+'_valid']:
                            box=a['bbox'];crowd=1;kp=None
                        else:
                            kp=np.array(a[prefix+'_kpts']).reshape(-1,3);box=a[prefix+'_box'];crowd=0
                            if kind=='hand':
                                points=kp[PALM_IDS];points=points[points[:,2]>0]
                                if len(points)<3:box=a['bbox'];crowd=1
                                else:
                                    x1,y1=points[:,:2].min(axis=0);x2,y2=points[:,:2].max(axis=0);box=[x1,y1,x2-x1,y2-y1]
                        x,y,w,h=box
                        if w<=0 or h<=0:continue
                        ann['annotations'].append({'id':len(ann['annotations'])+1,'image_id':image['id'],'category_id':1,'bbox':list(map(float,box)),'area':float(w*h),'iscrowd':crowd})
                        if not crowd:gold.append({'bbox':box,'keypoints':kp,'right':side=='right'})
                golds[image['id']]=gold
            records=core.read_jsonl(out/'predictions.jsonl');done={r['image_id'] for r in records}
            with zipfile.ZipFile(DATA/'COCO2017/val2017.zip') as z:
                samples=annotation['images'][:args.max_images] if args.max_images else annotation['images']
                for index,info in enumerate(samples,1):
                    if info['id'] in done:continue
                    image=Image.open(io.BytesIO(z.read('val2017/'+info['file_name']))).convert('RGB')
                    preds=pipeline.predict(image);core.append_jsonl(out/'predictions.jsonl',{'image_id':info['id'],'predictions':preds})
                    if index==1 or index%100==0 or index==len(samples):print(kind,index,'/',len(samples),flush=True)
            records=core.read_jsonl(out/'predictions.jsonl');status.update(state='partial_complete',smoke_passed=True,completed_images=len(records),total_images=5000)
            if args.max_images==0:
                assert len(records)==5000 and {r['image_id'] for r in records}=={i['id'] for i in ann['images']}
                detections=[];all_points=matched_points=good_points=matched_hands=correct_lr=0;errors=[];counts_per_point=np.zeros(21 if kind=='hand' else 7,dtype=int)
                for record in records:
                    gold=golds[record['image_id']];pred=record['predictions'];indices=list(range(21)) if kind=='hand' else list(FACE_DENSE_MAP.values())
                    all_points+=sum(int((g['keypoints'][indices,2]>0).sum()) for g in gold)
                    for p in pred:
                        x1,y1,x2,y2=p['bbox_xyxy'];detections.append({'image_id':record['image_id'],'category_id':1,'bbox':[x1,y1,x2-x1,y2-y1],'score':p['score']})
                    pose=[p for p in pred if p.get('landmark_score',0)>=.5]
                    if not gold or not pose:continue
                    ious=np.zeros((len(gold),len(pose)))
                    for i,g in enumerate(gold):
                        x,y,w,h=g['bbox']
                        for j,p in enumerate(pose):
                            a,b,c,d=p['bbox_xyxy'];inter=max(0,min(x+w,c)-max(x,a))*max(0,min(y+h,d)-max(y,b));union=w*h+(c-a)*(d-b)-inter
                            ious[i,j]=inter/union if union>0 else 0
                    rows,cols=linear_sum_assignment(-ious)
                    for i,j in zip(rows,cols):
                        if ious[i,j]<.5:continue
                        g,p=gold[i],pose[j];truth=g['keypoints'][indices];prediction=np.array(p['landmarks'])[list(range(21)) if kind=='hand' else list(FACE_DENSE_MAP)]
                        visible=truth[:,2]>0;counts_per_point+=visible.astype(int)
                        norm=np.hypot(g['bbox'][2],g['bbox'][3]);distance=np.linalg.norm(prediction[visible,:2]-truth[visible,:2],axis=1)/norm
                        matched_points+=len(distance);good_points+=int((distance<=.1).sum());errors.extend(distance.tolist())
                        if kind=='hand':matched_hands+=1;correct_lr+=int((p['right_hand_probability']>=.5)==g['right'])
                result={'bbox':metrics(ann,detections),'landmarks_2d':{'visible_gt_points':all_points,'matched_visible_points':matched_points,'coverage':matched_points/all_points if all_points else None,'NME_bbox_diagonal':float(np.mean(errors)) if errors else None,'PCK_0_1_conditional':good_points/matched_points if matched_points else None,'PCK_0_1_end_to_end':good_points/all_points if all_points else None,'matched_counts_per_point':counts_per_point.tolist()},'all_model_tasks_complete':False,'coverage_limit':protocol['coverage_limit'],'unit':'fraction_0_to_1'}
                if kind=='hand':result['handedness']={'correct':correct_lr,'matched_hands':matched_hands,'accuracy':correct_lr/matched_hands if matched_hands else None}
                core.atomic_json(out/'metrics.json',result);status.update(state='complete',metrics=result,all_model_tasks_complete=False)
        except BaseException as exc:status.update(state='failed',error=repr(exc));raise
        finally:status['elapsed_seconds']=time.time()-status['started_at'];core.atomic_json(path,status)

if __name__=='__main__':main()
