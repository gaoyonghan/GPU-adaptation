"""Serialize existing predictions for the unmodified classification/evaluate.py."""
from pathlib import Path
import hashlib,json,os
POLICY = {'task': 'single_label_classification', 'top_k': [1, 5], 'top5_when_classes_below_five': 'not_applicable', 'top5_when_ranking_missing': 'unavailable', 'ranking': 'preserve_native_order_no_new_tie_break', 'macro_labels': 'observed_ground_truth_classes', 'zero_division': 0, 'unit': 'fraction', 'sample_weight': 'uniform'}
def write(p, value):
    p.write_text(json.dumps(value, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
def export_bundle(run, categories, rows, model, dataset):
    run=Path(run); bundle=run/'scoring'; bundle.mkdir(exist_ok=False)
    samples=[]; predictions=[]
    for row in rows:
        sid=row['sample_id']; target=row.get('target',row.get('label'))
        ranking=row.get('top5_predictions',row.get('topk',[row['prediction']]))
        samples.append({'sample_id':sid,'label':target})
        predictions.append({'sample_id':sid,'prediction':row['prediction'],'topk':ranking})
    write(bundle/'annotations.json',{'categories':categories,'samples':samples})
    write(bundle/'predictions.json',predictions)
    write(bundle/'protocol.json',{'schema':'model-eval/score-classification/v1',
        'evaluation':POLICY,'scoring_ready':True,'model':model,'dataset':dataset,
        'completed_sample_ids':[row['sample_id'] for row in predictions],
        'prediction_count':len(predictions),
        'qualification':'delivery smoke only; not official metric validation' if int(os.environ.get('ZERO_SHOT_SMOKE_LIMIT','0')) else 'same-protocol baseline; not automatic official metric acceptance',
        'artifact_sha256':{name:hashlib.sha256((bundle/name).read_bytes()).hexdigest()
                           for name in ['annotations.json','predictions.json']}})
    write(run/'inference_output.json',{'state':'completed','task':'classification','bundles':['scoring'],'smoke_only':bool(int(os.environ.get('ZERO_SHOT_SMOKE_LIMIT','0')))})
    return bundle
