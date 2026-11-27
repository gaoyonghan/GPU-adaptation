"""One preregistered diagnostic: model-card usage-example threshold 0.7."""
import json
from pathlib import Path
from run import OUTPUT, evaluate, save, sha

status = json.loads((OUTPUT / 'status.json').read_text())
assert status['state'] == 'complete' and status['completed_images'] == 498
annotation = json.loads((OUTPUT / 'annotation.json').read_text())
predictions = json.loads((OUTPUT / 'predictions.json').read_text())
metrics = evaluate(annotation, [p for p in predictions if p['score'] > .7], {i['id'] for i in annotation['images']})
save(OUTPUT / 'threshold_0p7_diagnostic.json', {
    'purpose': 'Single diagnostic at model-card usage-example threshold .7; not a claim that author evaluates at this threshold.',
    'threshold': .7, 'inference_repeated': False, 'script_sha256': sha(Path(__file__)),
    'source_predictions_sha256': sha(OUTPUT / 'predictions.json'), 'metrics': metrics,
    'unit': 'fraction_0_to_1', 'model_card_reference': .571,
    'limitation': 'Dataset is original498 and scores are pooled. No confirmed domain mapping; author 57.1 dataset/protocol unknown.',
})
print('Diagnostic AP:', metrics['AP'], flush=True)
