"""Generate a scoring package from independent GT and recorded predictions; no accuracy calculation."""
import csv,json,hashlib
from pathlib import Path
POLICY = {'task': 'single_label_classification', 'top_k': [1, 5], 'top5_when_classes_below_five': 'not_applicable', 'top5_when_ranking_missing': 'unavailable', 'ranking': 'preserve_native_order_no_new_tie_break', 'macro_labels': 'observed_ground_truth_classes', 'zero_division': 0, 'unit': 'fraction', 'sample_weight': 'uniform'}

def publish(run,model_info,dataset_info):
    run=Path(run); bundle=run/'scoring_bundle'; bundle.mkdir(exist_ok=False)
    annotation=json.loads((run/'annotations.json').read_text())
    rows=list(csv.DictReader((run/'predictions.csv').open()))
    predictions=[{'sample_id':r['sample_id'],'prediction':int(r['prediction']),'topk':json.loads(r['topk'])} for r in rows]
    for name,value in [('annotations.json',annotation),('predictions.json',predictions)]:
        (bundle/name).write_text(json.dumps(value,ensure_ascii=False,allow_nan=False))
    protocol=dict(schema='model-eval/score-classification/v1',scoring_ready=True,evaluation=POLICY,
        model=model_info,dataset=dataset_info,completed_sample_ids=sorted(r['sample_id'] for r in predictions),prediction_count=len(predictions),
        artifact_sha256={n:hashlib.sha256((bundle/n).read_bytes()).hexdigest() for n in ['annotations.json','predictions.json']})
    (bundle/'protocol.json').write_text(json.dumps(protocol,ensure_ascii=False,indent=2,allow_nan=False))
    (run/'inference_output.json').write_text(json.dumps({'state':'completed','task':'classification','bundles':['scoring_bundle']},indent=2))
