"""Shared bundle integrity checks; no model routing or metric computation."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import platform
import tempfile
import shutil
from detection import read, write, require, integer, number, box, digest, file_hash, validate_annotation

COCO_SCHEMA='model-eval/score-coco-bbox/v1'
LVIS_SCHEMA='model-eval/score-lvis-bbox/v1'
GRITS_SCHEMA='model-eval/score-grits-cells/v1'
LVIS_POLICY=dict(iou_type='bbox',max_detections=300,iou_thresholds=[(50+5*i)/100 for i in range(10)],
                 use_negative_and_non_exhaustive_categories=True,frequency_groups=['r','c','f'],undefined_metric=None)
GRITS_POLICY=dict(metrics=['GriTS_Top','GriTS_Con','GriTS_Loc'],aggregation='mean_over_all_tables',
                  empty_predictions='score_with_empty_grid',cell_bbox_format='xyxy',text='provided_cell_text_unchanged')


def validate_lvis(annotation):
    # Reuse geometry checks without inventing persistent COCO crowd semantics.
    validation=dict(annotation,images=[dict(im,file_name=im.get('file_name',im.get('coco_url',''))) for im in annotation['images']],
                    annotations=[dict(a,iscrowd=a.get('iscrowd',0)) for a in annotation['annotations']])
    ids=validate_annotation(validation)
    for im in annotation['images']:
        for field in ('neg_category_ids','not_exhaustive_category_ids'):
            values=im.get(field);require(isinstance(values,list) and all(integer(v) and v in ids['categories'] for v in values),
                                        'LVIS '+field+' must be explicit valid category IDs')
            require(len(values)==len(set(values)), 'duplicate LVIS category metadata')
    for cat in annotation['categories']:
        require(cat.get('frequency') in ('r','c','f'),'LVIS frequency required')
    negatives={im['id']:set(im['neg_category_ids']) for im in annotation['images']}
    for ann in annotation['annotations']:
        require(not ann.get('iscrowd',0),'LVIS crowd annotations are unsupported')
        require(ann['category_id'] not in negatives[ann['image_id']], 'positive GT contradicts LVIS negative category')
    return ids


def validate_bbox_predictions(predictions, ids):
    for row in predictions:
        require(set(row)=={'image_id','category_id','bbox','score'},'four prediction fields required')
        require(integer(row['image_id']) and row['image_id'] in ids['images'],'unknown prediction image')
        require(integer(row['category_id']) and row['category_id'] in ids['categories'],'unknown prediction category')
        box(row['bbox']);require(number(row['score']),'finite ranking score required')


def load_inputs(bundle,schema,policy):
    bundle=Path(bundle);p=read(bundle/'protocol.json')
    require(p.get('schema')==schema,'wrong scoring protocol; use the matching fixed scorer')
    require(p.get('evaluation')==policy,'scoring policy differs from this fixed scorer')
    require(p.get('scoring_ready') is True,'bundle not ready for scoring')
    require(set(p.get('artifact_sha256',{}))=={'annotations.json','predictions.json'},'input hash manifest incomplete')
    for name,value in p['artifact_sha256'].items():require(file_hash(bundle/name)==value,'input hash mismatch: '+name)
    return read(bundle/'annotations.json'),read(bundle/'predictions.json'),p


def check_bbox_scope(annotation,predictions,p,validator):
    ids=validator(annotation);validate_bbox_predictions(predictions,ids)
    completed=p.get('completed_image_ids')
    require(isinstance(completed,list) and all(integer(i) for i in completed) and len(completed)==len(set(completed))
            and set(completed)==ids['images'],'full selected image coverage mismatch')
    require(p.get('prediction_count')==len(predictions),'prediction count mismatch')
    return ids


def result_header(p,metrics,evaluator,scope):
    return dict(schema='model-eval/metrics/v1',protocol=p['schema'],metric_unit='fraction',metrics=metrics,
                model=p.get('model'),dataset=p.get('dataset'),scope=scope,evaluation=p['evaluation'],
                evaluator=evaluator,input_sha256=p['artifact_sha256'],protocol_sha256=digest(p),
                inference_executed=False,python=platform.python_version())


def publish_bbox(output,annotation,predictions,protocol):
    output=Path(output);require(not output.exists(),'choose a new output directory; no overwrite')
    output.parent.mkdir(parents=True,exist_ok=True)
    tmp=Path(tempfile.mkdtemp(prefix='.'+output.name+'-',dir=output.parent))
    try:
        write(tmp/'annotations.json',annotation);write(tmp/'predictions.json',predictions)
        protocol['artifact_sha256']={name:file_hash(tmp/name) for name in ['annotations.json','predictions.json']}
        write(tmp/'protocol.json',protocol);tmp.rename(output)
    finally:
        if tmp.exists():shutil.rmtree(tmp)
    return protocol


def cli(evaluate,description):
    p=argparse.ArgumentParser(description=description);p.add_argument('--bundle',type=Path,required=True)
    args=p.parse_args();result=evaluate(args.bundle);print(json.dumps(result['metrics'],ensure_ascii=False))
