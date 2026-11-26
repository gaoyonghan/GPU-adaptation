"""Convert existing classification predictions/GT; preserve native inference semantics."""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
import subprocess
import sys
from classification import file_hash, integer, publish, read, require, write
ROOT = Path(__file__).resolve().parents[2] / 'resources/runtime/project'

def registry():
    return {r['sequence']: r for r in read(Path(__file__).with_name('classification_inventory.json'))['models']}

def categories_for(spec):
    if spec.get('categories'):
        return spec['categories']
    names = {}
    ref = spec.get('category_reference')
    if ref:
        model = next((r for r in read(ROOT / ref['file'])['models'] if r['sequence'] == spec['sequence']))
        names = model.get(ref['field'], {}) or model.get('config', {}).get('id2label', {})
    return [dict(id=i, name=str(names.get(str(i), f'output_class_{i}'))) for i in range(spec['num_classes'])]

def records(path):
    path = Path(path)
    if path.suffix == '.csv':
        with path.open(encoding='utf-8-sig', newline='') as stream:
            return list(csv.DictReader(stream))
    if path.suffix == '.jsonl':
        with path.open(encoding='utf-8') as stream:
            return [json.loads(line) for line in stream if line.strip()]
    data = read(path)
    if isinstance(data, dict):
        for key in ('records', 'rows', 'samples'):
            if isinstance(data.get(key), list):
                return data[key]
    require(isinstance(data, list), 'unsupported native record container')
    return data

def source_id(row):
    for key in ('sample_id', 'id', 'image_id'):
        if key in row:
            v = row[key]
            require(isinstance(v, (str, int)) and (not isinstance(v, bool)), 'invalid native sample ID')
            return str(v)
    if 'member' in row:
        return (str(row['source']) + ':' if 'source' in row else '') + str(row['member'])
    for key in ('path', 'source_path', 'row', 'index'):
        if key in row:
            return str(row[key])
    raise ValueError('native sample has no explicit identity')

def as_label(value):
    if integer(value):
        return value
    if isinstance(value, str) and value.lstrip('-').isdigit():
        return int(value)
    raise ValueError('missing/unresolved native GT or prediction label')

def normalize(row, spec):
    adapter = spec['adapter']
    sid = source_id(row)
    if adapter == 'fer_csv':
        target, pred = (row['target_fer_id'], row['prediction_fer_id'])
    elif adapter == 'fer_provisional':
        target, pred = (row['target_fer_id'], row['provisional_prediction_fer_id'])
    elif adapter == 'imagein':
        target, pred = (row['target_id'], row['prediction_id'])
    elif adapter == 'imagenet22k':
        target, pred = (row['gold_index'], row['global_top5'][0])
    elif adapter == 'fashion_proxy':
        target, pred = (row['gold'], row['pred'])
    elif adapter == 'body_proxy':
        target, pred = (row['target_id'], row['prediction'])
    else:
        target = row.get('target', row.get('label'))
        pred = row.get('prediction')
    target, pred = (as_label(target), as_label(pred))
    ranked = row.get('topk', row.get('top5'))
    if adapter == 'imagenet22k':
        ranked = row['global_top5']
    ranked = [as_label(v) for v in ranked] if ranked is not None else [pred]
    scores = row.get('topk_scores')
    normalized = dict(sample_id=sid, prediction=pred, topk=ranked, scores=scores)
    return (normalized, dict(sample_id=sid, label=target))

def nearby_documents(source, root):
    result = []
    parent = source.parent
    names = ('classification_capture.json', 'status.json', 'metrics.json', 'provisional_metrics.json', 'protocol.json', 'evaluation_protocol.json')
    while True:
        for name in names:
            p = parent / name
            if p.is_file():
                try:
                    data = read(p)
                    if isinstance(data, dict):
                        result.append((p, data))
                except (ValueError, OSError):
                    pass
        if parent == root or root not in parent.parents:
            break
        parent = parent.parent
    return result

def declared_scope(source, root, count, phase, spec, successful):
    docs = nearby_documents(source, root)
    expected = None
    evidence = []
    for path, doc in docs:
        state = doc.get('state', doc.get('status'))
        if state in ('failed', 'running', 'error') and (path.parent == source.parent or source.parent.name == 'evaluation'):
            require(False, 'native output is incomplete or failed: ' + str(path))
        for key in ('expected_images', 'selected_images', 'images', 'evaluation_images', 'evaluation_samples', 'n_images', 'total_images', 'n_eval'):
            v = doc.get(key)
            if integer(v) and v > 0:
                if expected is None:
                    expected = v
                break
        evidence.append(dict(path=str(path), sha256=file_hash(path)))
    if phase == 'full' and spec.get('expected_images'):
        require(count == spec['expected_images'], f"full scope differs from registered protocol: {count}/{spec['expected_images']}")
        if expected is None:
            expected = spec['expected_images']
    if phase == 'smoke':
        for _, doc in docs:
            if (doc.get('requested_images') or doc.get('status', doc.get('state')) == 'partial_complete') and integer(doc.get('completed_images')) and doc['completed_images'] > 0:
                expected = doc['completed_images']
                break
    if expected is not None:
        require(count == expected, f'native selected scope incomplete: {count}/{expected}')
    require(expected is not None or successful is True, 'no independent completion evidence; do not infer full coverage from predictions')
    return dict(selected_images=count, expected_images=expected, phase=phase, completion_evidence=evidence, launcher_completed=successful, scope_note='Complete declared selection only; not proof of a complete official dataset.')

def convert(source, output, sequence, *, root=None, phase='full', successful=None, gt_path=None, binding=None):
    source, output = (Path(source), Path(output))
    root = Path(root) if root else source.parent
    spec = registry()[sequence]
    require(spec['state'] != 'historically_skipped', spec.get('reason', 'historically skipped'))
    raw = records(source)
    require(raw, 'no native predictions')
    before = file_hash(source)
    scope = declared_scope(source, root, len(raw), phase, spec, successful)
    pred, embedded_gt = ([], [])
    excluded = []
    index_path = ROOT / spec['gt_index'] if spec.get('gt_index') else None
    if index_path is not None and (not index_path.exists()):
        for _, doc in nearby_documents(source, root):
            recorded = doc.get('model_manifest', {}).get('index')
            if recorded and Path(recorded).is_file():
                index_path = Path(recorded)
                break
        else:
            index_path = None
    index_rows = records(index_path) if index_path else None
    indexed = {source_id(r): r for r in index_rows} if index_rows is not None else None
    if indexed is not None:
        require(len(indexed) == len(index_rows), 'Duplicate IDs in native GT index')
        native_ids = [source_id(r) for r in raw]
        require(len(set(native_ids)) == len(native_ids), 'Duplicate native predictions')
        require(set(native_ids) <= set(indexed), 'Prediction IDs not in native dataset index')
        if phase == 'full':
            require(set(native_ids) == set(indexed), 'Full prediction/GT index coverage differs')
    qualified = spec['adapter'] in ('qualified_architecture', 'qualified_age')
    for row in raw:
        if qualified:
            require(indexed is not None, 'Qualified mapping requires independent source-label index')
            sid = source_id(row)
            label = indexed[sid]['source_label']
            if label in spec['excluded_labels']:
                excluded.append(dict(sample_id=sid, source_label=label))
                continue
            require(label in spec['gt_mapping'], 'Unmapped source category: ' + label)
            prediction = as_label(row['prediction'])
            if spec.get('prediction_mapping'):
                prediction = spec['prediction_mapping'][str(prediction)]
            ranking = [prediction] if spec.get('prediction_mapping') else [as_label(v) for v in row.get('top5', [prediction])]
            pr = dict(sample_id=sid, prediction=prediction, topk=ranking, scores=None)
            gt = dict(sample_id=sid, label=spec['gt_mapping'][label])
        else:
            pr, gt = normalize(row, spec)
            if indexed is not None:
                target = as_label(indexed[pr['sample_id']]['target'])
                require(target == gt['label'], 'Saved target differs from independent native dataset index')
        pred.append(pr)
        embedded_gt.append(gt)
    if qualified and phase == 'full':
        require(len(pred) == spec['expected_scored_images'], 'Qualified scoring selection differs from original protocol')
    categories = categories_for(spec)
    gt_source = dict(kind='native_recorded_dataset_targets', path=str(source), sha256=before, note='Targets carried unchanged from the existing dataset loader, never derived from predicted labels.')
    gt = embedded_gt
    if indexed is not None:
        gt_source = dict(kind='native_dataset_index', path=str(index_path), sha256=file_hash(index_path))
    captured_gt = source.parent / 'classification_annotations.json'
    if captured_gt.exists() and (not qualified):
        gt = read(captured_gt)['samples']
        require({r['sample_id']: r['label'] for r in gt} == {r['sample_id']: r['label'] for r in embedded_gt}, 'Native metric targets differ from captured dataset manifest')
        gt_source = dict(kind='native_dataset_manifest', path=str(captured_gt), sha256=file_hash(captured_gt))
    elif not qualified and indexed is None:
        manifest_names = {72: 'dataset_index.json', 76: 'manifest.jsonl', 90: 'manifest.jsonl', 92: 'manifest.json', 94: 'test_manifest.json', 99: 'data_manifest.json'}
        name = manifest_names.get(sequence)
        for parent in (source.parent, source.parent.parent):
            manifest = parent / name if name else None
            if manifest is None or not manifest.is_file():
                continue
            dataset_rows = records(manifest)
            converted_gt = [dict(sample_id=source_id(r), label=as_label(r.get('target', r.get('label')))) for r in dataset_rows]
            actual = {r['sample_id']: r['label'] for r in embedded_gt}
            selected = {r['sample_id']: r['label'] for r in converted_gt}
            require(len(selected) == len(converted_gt), 'Duplicate native manifest sample IDs')
            if phase == 'full':
                require(actual == selected, 'Native GT manifest and predictions differ')
            else:
                require(all((selected.get(k) == v for k, v in actual.items())), 'Smoke GT labels differ from dataset manifest')
            gt = [r for r in converted_gt if r['sample_id'] in actual]
            gt_source = dict(kind='native_dataset_manifest', path=str(manifest), sha256=file_hash(manifest))
            break
    if gt_path is not None:
        data = read(gt_path)
        require(isinstance(data, dict) and 'samples' in data and ('categories' in data), 'explicit GT must use samples/categories')
        gt, categories = (data['samples'], data['categories'])
        gt_source = dict(kind='independent_gt', path=str(gt_path), sha256=file_hash(gt_path))
    if binding is not None:
        require(gt_path is not None, 'A class binding requires explicit GT in the destination class space')
        require(isinstance(binding, dict), 'binding must be an object')
        cmap = binding.get('category_ids', {})
        require(len(set(cmap.values())) == len(cmap), 'class merging is a protocol change, not a format conversion')
        for row in pred:
            require(all((str(c) in cmap for c in row['topk'])), 'missing explicit class binding')
            row['topk'] = [cmap[str(c)] for c in row['topk']]
            row['prediction'] = row['topk'][0]
    annotation = dict(schema='model-eval/classification-annotations/v1', categories=categories, samples=gt)
    metadata = dict(model=dict(sequence=sequence, model_id=spec['model_id']), dataset=spec['dataset'], qualification=spec['qualification'], limitations=spec.get('limitations', []), source=dict(predictions=dict(path=str(source), sha256=before), ground_truth=gt_source), conversion=dict(adapter=spec['adapter'], label_space='existing_qualified_protocol' if qualified else 'native_recorded_class_ids', dropped_samples=len(excluded), excluded_samples=excluded, class_mapping=spec.get('gt_mapping'), prediction_mapping=spec.get('prediction_mapping'), merged_classes=bool(spec.get('prediction_mapping')), restricted_candidates=False, ranking='native_saved_order', adapter_sha256=file_hash(Path(__file__)), inventory_sha256=file_hash(Path(__file__).with_name('classification_inventory.json'))), selection=scope)
    native_execution = []
    for path, doc in nearby_documents(source, root):
        fields = {k: doc[k] for k in ('device', 'variant', 'nv_rerun_required', 'original_onnx_hardware_comparison_valid', 'native_baseline') if k in doc}
        if fields:
            native_execution.append(dict(path=str(path), **fields))
    metadata['native_execution'] = native_execution
    require(file_hash(source) == before, 'native predictions changed during conversion')
    if index_path is not None:
        metadata['source']['ground_truth']['index_sha256'] = file_hash(index_path)
    metadata['selection']['scored_images'] = len(pred)
    if gt_source['kind'] == 'native_recorded_dataset_targets':
        metadata['limitations'] = metadata['limitations'] + ['GT comes from saved dataset targets; independent annotation-file identity is not revalidated by this conversion.']
    return publish(output, annotation, pred, metadata)

def prediction_sources(root, phase='full'):
    root = Path(root)
    found = []
    for name in ('predictions.jsonl', 'predictions.csv', 'predictions.json', 'classification_predictions.jsonl'):
        for path in root.rglob(name):
            parts = path.relative_to(root).parts
            if any((p.startswith(('standardized', 'score_', '.', '_prepared')) or p in ('baseline', 'scoring_history') or (p == 'smoke_check' and phase != 'smoke') for p in parts)):
                continue
            if phase == 'full' and 'smoke' in parts:
                continue
            if phase == 'smoke' and 'full' in parts:
                continue
            found.append(path)
    parents = {p.parent for p in found if p.name == 'classification_predictions.jsonl'}
    return sorted((p for p in found if p.parent not in parents or p.name == 'classification_predictions.jsonl'))

def finalize_run(root, sequence, phase, successful=None):
    root = Path(root)
    spec = registry()[sequence]
    report = dict(sequence=sequence, phase=phase, state='preparing', items=[], inference_executed=False, source_run_success=successful)
    if spec['state'] == 'historically_skipped':
        report.update(state='not_applicable', reason=spec['reason'])
    else:
        for source in prediction_sources(root, phase):
            entry = dict(source=str(source.relative_to(root)))
            try:
                output = source.parent / 'standardized_classification'
                if output.exists():
                    p = read(output / 'protocol.json')
                    gt_source = p.get('source', {}).get('ground_truth', {})
                    gt_unchanged = gt_source.get('path') and Path(gt_source['path']).is_file() and (file_hash(gt_source['path']) == gt_source.get('sha256'))
                    old_evidence = p.get('selection', {}).get('completion_evidence', [])
                    evidence_unchanged = all((Path(e['path']).is_file() and file_hash(e['path']) == e['sha256'] for e in old_evidence))
                    if p.get('source', {}).get('predictions', {}).get('sha256') == file_hash(source) and p.get('conversion', {}).get('adapter_sha256') == file_hash(Path(__file__)) and (p.get('conversion', {}).get('inventory_sha256') == file_hash(Path(__file__).with_name('classification_inventory.json'))) and gt_unchanged and evidence_unchanged and all((file_hash(output / n) == h for n, h in p.get('artifact_sha256', {}).items())):
                        entry.update(state='exported', output=str(output))
                        report['items'].append(entry)
                        continue
                    from datetime import datetime
                    output = output.with_name(output.name + '-' + datetime.now().strftime('%Y%m%dT%H%M%S%f'))
                convert(source, output, sequence, root=root, phase=phase, successful=successful)
                entry.update(state='exported', output=str(output))
            except Exception as exc:
                entry.update(state='failed', reason=f'{type(exc).__name__}: {exc}')
            report['items'].append(entry)
        report['state'] = 'exported' if report['items'] and all((r['state'] == 'exported' for r in report['items'])) else 'failed'
        if not report['items']:
            report['reason'] = 'No per-sample predictions; rerun with classification capture enabled. Aggregate metrics are not predictions.'
        if successful is False:
            report.update(state='failed', reason='Native launch failed; exported older results do not prove this launch succeeded.')
    write(root / ('classification_export_' + phase + '.json'), report)
    return report
if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', type=int, required=True)
    p.add_argument('--source', type=Path)
    p.add_argument('--output', type=Path)
    p.add_argument('--root', type=Path)
    p.add_argument('--annotations', type=Path)
    p.add_argument('--binding', type=Path)
    p.add_argument('--phase', choices=['smoke', 'full'], default='full')
    a = p.parse_args()
    if a.source:
        require(a.output is not None, '--output required with --source')
        convert(a.source, a.output, a.model, root=a.root, phase=a.phase, gt_path=a.annotations, binding=read(a.binding) if a.binding else None)
    else:
        require(a.root is not None, '--root required')
        result = finalize_run(a.root, a.model, a.phase)
        print(json.dumps(result, ensure_ascii=False))
        raise SystemExit(0 if result['state'] in ('exported', 'not_applicable') else 1)
