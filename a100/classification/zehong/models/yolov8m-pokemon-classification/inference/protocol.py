"""评分包写出：model-eval/score-classification/v1。"""
import hashlib
import json
from pathlib import Path

SCHEMA = 'model-eval/score-classification/v1'
POLICY = {
    'task': 'single_label_classification', 'top_k': [1, 5],
    'top5_when_classes_below_five': 'not_applicable',
    'top5_when_ranking_missing': 'unavailable',
    'ranking': 'preserve_native_order_no_new_tie_break',
    'macro_labels': 'observed_ground_truth_classes', 'zero_division': 0,
    'unit': 'fraction', 'sample_weight': 'uniform',
}


def write_json(path, value):
    path = Path(path)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def write_json_compact(path, value):
    path = Path(path)
    path.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False) + '\n', encoding='utf-8')


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def write_bundle(output_dir, annotation, predictions, *, model_info, dataset_info,
                 scope, native_execution, limitations):
    bundle = Path(output_dir) / 'scoring_bundle'
    bundle.mkdir(parents=True, exist_ok=False)
    write_json_compact(bundle / 'annotations.json', annotation)
    write_json_compact(bundle / 'predictions.json', predictions)
    protocol = {
        'schema': SCHEMA,
        'task': 'single_label_classification',
        'scoring_ready': True,
        'evaluation': POLICY,
        'model': model_info,
        'dataset': dict(dataset_info, scope=scope),
        'completed_sample_ids': [row['sample_id'] for row in annotation['samples']],
        'prediction_count': len(predictions),
        'artifact_sha256': {
            'annotations.json': sha256_file(bundle / 'annotations.json'),
            'predictions.json': sha256_file(bundle / 'predictions.json'),
        },
        'native_execution': native_execution,
        'limitations': limitations,
    }
    write_json(bundle / 'protocol.json', protocol)
    return bundle
