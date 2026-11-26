"""Convert completed local native records without inference or scoring."""
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from model_paths import configuration
from classification import file_hash, load_bundle, publish, require


def finalize_run(root, sequence, phase, successful=True):
    root = Path(root).resolve()
    spec = configuration()
    require(sequence == spec['sequence'] and successful, 'Wrong model or failed native execution')
    native = root / phase / f'm{sequence:03d}'
    status = json.loads((native / 'status.json').read_text())
    require(status['state'] == 'complete' and status['phase'] == phase, 'Native execution is not complete')
    require(status['sequence'] == sequence and status['model_id'] == spec['model_id'], 'Native status belongs to another model')
    branches = spec['native_script'].get('branches', ['default'])
    result = []
    for branch in branches:
        folder = native if branch == 'default' else native / branch
        capture = json.loads((folder / 'classification_capture.json').read_text())
        require(capture['state'] == 'complete', 'Native capture is incomplete')
        raw_gt = json.loads((folder / 'classification_annotations.json').read_text())
        annotation = dict(schema='model-eval/classification-annotations/v1',
                          categories=raw_gt['categories'],
                          samples=[dict(sample_id=r['sample_id'], label=r['label']) for r in raw_gt['samples']])
        declared = spec['label_mapping'] if branch == 'default' else spec['label_mapping'][branch]
        require(annotation['categories'] == declared, 'Native category mapping differs from frozen e07 mapping')
        truth = {r['sample_id']: r['label'] for r in annotation['samples']}
        predictions = []
        for line in (folder / 'predictions.jsonl').read_text().splitlines():
            r = json.loads(line)
            sid = r.get('sample_id', r.get('id'))
            require(not Path(sid).is_absolute(), 'Absolute native sample ID is forbidden')
            require(sid in truth and r['target'] == truth[sid], 'Native prediction target differs from loader GT')
            predictions.append(dict(sample_id=sid, prediction=r['prediction'],
                                    topk=r.get('topk', r.get('top5')), scores=r.get('scores')))
        require(len(predictions) == len(annotation['samples']) == capture['completed_images'], 'Capture count mismatch')
        require(status['completed_images'] == len(predictions), 'Native status sample count differs from branch coverage')
        if phase == 'full':
            require(len(predictions) == spec['expected_images'], 'Full output differs from registered e07 valid sample count')
        selected = status['total_images']
        model = dict(sequence=sequence, model_id=spec['model_id'])
        if branch != 'default':
            model['branch'] = branch
        metadata = dict(
            model=model,
            dataset=spec['dataset'], qualification=spec['qualification'],
            limitations=spec.get('limitations', []) + ['MX native backend and precision retained. New-machine environment rebuild has not been verified.'],
            source=dict(path_base='bundle_parent',
                        predictions=dict(path='predictions.jsonl', sha256=file_hash(folder / 'predictions.jsonl')),
                        ground_truth=dict(kind='native_dataset_loader', path='classification_annotations.json',
                                          sha256=file_hash(folder / 'classification_annotations.json'))),
            conversion=dict(adapter='native_labels', label_space='native_recorded_class_ids', dropped_samples=0,
                            excluded_samples=[], class_mapping=None, prediction_mapping=None,
                            merged_classes=False, restricted_candidates=False, ranking='native_saved_order',
                            adapter_sha256=file_hash(__file__), inventory_sha256=file_hash(Path(__file__).with_name('classification_inventory.json'))),
            selection=dict(selected_images=selected, expected_images=spec['expected_images'] if phase == 'full' else selected,
                           phase=phase, completion_evidence=[dict(path=str((native / name).relative_to(root)), sha256=file_hash(native / name)) for name in ('status.json','metrics.json')],
                           launcher_completed=True, scored_images=len(predictions),
                           excluded_decode_samples=status.get('excluded_sample_ids', []),
                           scope_note='Native full registered e07 scope' if phase == 'full' else 'Native MAX_SAMPLES selection after complete native scope validation'),
            native_execution=[dict(path=str((native / 'status.json').relative_to(root)), variant=None, **status['native_execution'])])
        output = folder / 'evaluation'
        if output.exists():
            old_gt, old_pred, old_protocol = load_bundle(output)
            require(old_gt == annotation and old_pred == predictions, 'Existing converted bundle differs; use a new output-root')
            require(old_protocol['selection']['phase'] == phase, 'Existing phase differs')
        else:
            publish(output, annotation, predictions, metadata)
        result.append(dict(output=str(output)))
    return dict(state='exported', items=result)
