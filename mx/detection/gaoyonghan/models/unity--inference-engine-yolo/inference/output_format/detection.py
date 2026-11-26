"""Versioned COCO bbox exchange and scoring; no inference or GPU dependency."""
from __future__ import annotations
import argparse
import copy
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import platform
import zipfile
SCHEMA = 'model-eval/detection-coco-bbox/v1'
METRICS = ['AP', 'AP50', 'AP75', 'AP_small', 'AP_medium', 'AP_large', 'AR1', 'AR10', 'AR100', 'AR_small', 'AR_medium', 'AR_large']
POLICY = dict(iou_type='bbox', iou_thresholds=[(50 + 5 * i) / 100 for i in range(10)], recall_thresholds=[i / 100 for i in range(101)], max_detections=[1, 10, 100], area_ranges=[[0, 10000000000.0], [0, 1024], [1024, 9216], [9216, 10000000000.0]], area_labels=['all', 'small', 'medium', 'large'], use_categories=True, score_unit='fraction', undefined_metric=None)

def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))

def write(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    tmp.replace(path)

def digest(data):
    return hashlib.sha256(json.dumps(data, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()

def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def require(ok, message):
    if not ok:
        raise ValueError(message)

def integer(value):
    return isinstance(value, int) and (not isinstance(value, bool))

def number(value):
    return isinstance(value, (int, float)) and (not isinstance(value, bool)) and math.isfinite(value)

def box(value):
    require(isinstance(value, list) and len(value) == 4 and all((number(x) for x in value)), 'bbox must be four finite numbers: original-pixel xywh')
    require(value[2] >= 0 and value[3] >= 0, 'negative bbox width/height')

def validate_annotation(data):
    for key in ['images', 'categories', 'annotations']:
        require(isinstance(data.get(key), list), f'missing annotation list: {key}')
    ids = {}
    for key in ['images', 'categories', 'annotations']:
        values = [v.get('id') for v in data[key]]
        require(all((integer(x) for x in values)), f'{key} IDs must be integers')
        require(len(values) == len(set(values)), f'duplicate {key} IDs')
        ids[key] = set(values)
    require(ids['images'] and ids['categories'], 'empty image/category scope')
    for im in data['images']:
        require(all((integer(im.get(k)) and im[k] > 0 for k in ['width', 'height'])), 'image width/height must be positive integers')
        require(isinstance(im.get('file_name'), str), 'image file_name is required')
    for cat in data['categories']:
        require(isinstance(cat.get('name'), str), 'category name is required')
    for ann in data['annotations']:
        require(integer(ann.get('image_id')) and ann['image_id'] in ids['images'], 'unknown GT image ID')
        require(integer(ann.get('category_id')) and ann['category_id'] in ids['categories'], 'unknown GT category ID')
        box(ann.get('bbox'))
        require(number(ann.get('area')) and ann['area'] >= 0, 'GT area must be preserved and nonnegative')
        require(integer(ann.get('iscrowd')) and ann['iscrowd'] in [0, 1], 'explicit GT iscrowd is required')
    return ids

def validate_predictions(rows, ids):
    require(isinstance(rows, list), 'predictions must be a COCO detection list')
    for row in rows:
        require(set(row) == {'image_id', 'category_id', 'bbox', 'score'}, 'prediction fields must exactly match v1')
        require(integer(row['image_id']) and row['image_id'] in ids['images'], 'unknown prediction image ID')
        require(integer(row['category_id']) and row['category_id'] in ids['categories'], 'unknown prediction category ID')
        box(row['bbox'])
        require(number(row['score']) and 0 <= row['score'] <= 1, 'score must be finite in [0,1]')

def flatten_records(records, annotation):
    """Adapter for existing official_object_detection_eval per-image JSONL ONLY.

    Inputs already contain postprocessed COCO xywh and dataset category IDs.
    No second model-label mapping, NMS, thresholding, clipping, sorting or rounding.
    """
    ids = validate_annotation(annotation)
    seen = set()
    predictions = []
    empty = []
    for record in records:
        im = record.get('image_id')
        require(integer(im) and im in ids['images'], 'unknown completed image ID')
        require(im not in seen, 'duplicate image record')
        seen.add(im)
        det = record.get('detections')
        require(isinstance(det, list), 'detections list is required even if empty')
        if not det:
            empty.append(im)
        for row in det:
            require(row.get('image_id') == im, 'nested prediction image ID mismatch')
            predictions.append(copy.deepcopy(row))
    require(seen == ids['images'], f"incomplete image coverage: {len(seen)}/{len(ids['images'])}")
    validate_predictions(predictions, ids)
    return (predictions, sorted(seen), sorted(empty))

def export_bundle(annotation_path, records_path, status_path, output, annotation_member=None):
    output = Path(output)
    require(not output.exists(), 'choose a new output directory; no overwrite')
    if annotation_member:
        with zipfile.ZipFile(annotation_path) as z:
            data = z.read(annotation_member)
    else:
        data = Path(annotation_path).read_bytes()
    annotation = json.loads(data)
    status = read(status_path)
    require(status.get('state') in ['complete', 'verified_existing'], 'source run is not complete')
    snapshot = status.get('dataset_snapshot', {})
    require(snapshot.get('annotation_sha256') == digest(annotation), 'annotation differs from the original scored snapshot')
    with Path(records_path).open(encoding='utf-8') as f:
        predictions, completed, empty = flatten_records((json.loads(line) for line in f if line.strip()), annotation)
    require(status.get('completed_images') == len(completed) == status.get('total_images'), 'source completion counts disagree with the full annotation scope')
    mapping = status.get('class_mapping')
    require(isinstance(mapping, dict) and mapping, 'source category mapping is required')
    category_ids = {c['id'] for c in annotation['categories']}
    require(all((integer(v) and v in category_ids for v in mapping.values())), 'source mapping contains unknown category IDs')
    spec = status.get('manifest_snapshot', {})
    protocol = dict(schema=SCHEMA, task='object_detection', adapter='official_object_detection_eval.jsonl/v1', model=dict(sequence=status.get('sequence'), model_id=status.get('model_id')), dataset=spec.get('dataset', {}), image_ids=[i['id'] for i in annotation['images']], category_ids=[c['id'] for c in annotation['categories']], completed_image_ids=completed, empty_prediction_image_ids=empty, source_label_to_category_id=mapping, mapping_applied_before_export=True, bbox_format='xywh', coordinate_space='original_image_pixels', prediction_stage='after_original_postprocessing', prediction_count=len(predictions), conversion=dict(additional_filter=False, additional_nms=False, clipping=False, rounding=False), evaluation=copy.deepcopy(POLICY), source_runtime=status.get('runtime_versions', {}), source_device=status.get('device'), source_protocol=spec.get('protocol', {}), artifacts=dict(annotations='annotations.json', predictions='predictions.json'), source=dict(predictions_file_sha256=file_hash(records_path), status_file_sha256=file_hash(status_path), annotation_file_sha256=file_hash(annotation_path), annotation_member=annotation_member), content_sha256=dict(annotations=digest(annotation), predictions=digest(predictions)))
    output.mkdir(parents=True)
    write(output / 'annotations.json', annotation)
    write(output / 'predictions.json', predictions)
    write(output / 'protocol.json', protocol)
    return protocol

def load_bundle(bundle):
    bundle = Path(bundle)
    protocol = read(bundle / 'protocol.json')
    require(protocol.get('schema') == SCHEMA, 'unsupported schema')
    require(protocol.get('task') == 'object_detection' and protocol.get('bbox_format') == 'xywh' and (protocol.get('coordinate_space') == 'original_image_pixels'), 'unsupported task or coordinate semantics')
    require(protocol.get('evaluation') == POLICY, 'unsupported evaluation policy; create a new named protocol')
    annotation = read(bundle / 'annotations.json')
    predictions = read(bundle / 'predictions.json')
    ids = validate_annotation(annotation)
    validate_predictions(predictions, ids)
    require(protocol['content_sha256'] == dict(annotations=digest(annotation), predictions=digest(predictions)), 'bundle content hash mismatch')
    for key, expected in [('image_ids', ids['images']), ('completed_image_ids', ids['images']), ('category_ids', ids['categories'])]:
        values = protocol[key]
        require(all((integer(x) for x in values)) and len(values) == len(set(values)) and (set(values) == expected), f'{key} disagrees with annotation scope')
    require(protocol['prediction_count'] == len(predictions), 'prediction count mismatch')
    with_predictions = {p['image_id'] for p in predictions}
    require(set(protocol['empty_prediction_image_ids']) == ids['images'] - with_predictions, 'empty-image scope mismatch')
    return (annotation, predictions, protocol)
PREDICTION_SCHEMA = 'model-eval/detection-predictions/v2'
CORE_FIELDS = ('image_id', 'category_id', 'bbox', 'score')

def iter_json_array(path):
    """Stream large native JSON arrays without loading millions of boxes."""
    decoder = json.JSONDecoder()
    with Path(path).open(encoding='utf-8') as stream:
        buffer = ''
        position = 0
        eof = False

        def fill():
            nonlocal buffer, position, eof
            chunk = stream.read(1024 * 1024)
            buffer = buffer[position:] + chunk
            position = 0
            eof = not chunk

        def token():
            nonlocal position
            while True:
                while position < len(buffer) and buffer[position].isspace():
                    position += 1
                if position < len(buffer):
                    return buffer[position]
                if eof:
                    return None
                fill()
        require(token() == '[', 'native predictions must be a JSON array')
        position += 1
        if token() == ']':
            position += 1
        else:
            while True:
                token()
                while True:
                    try:
                        value, end = decoder.raw_decode(buffer, position)
                        position = end
                        break
                    except json.JSONDecodeError:
                        if eof:
                            raise
                        fill()
                yield value
                separator = token()
                position += 1
                if separator == ']':
                    break
                require(separator == ',', 'invalid prediction array separator')
        require(token() is None, 'trailing data after prediction array')

def iter_source(path, adapter):
    if adapter in ('B_coco_flat', 'G_wilddet3d') or (adapter == 'F_omnidoc' and Path(path).suffix == '.json'):
        yield from iter_json_array(path)
    else:
        with Path(path).open(encoding='utf-8') as stream:
            for line in stream:
                if line.strip():
                    yield json.loads(line)
SPECIAL_ADAPTERS = ('D_xyxy_records', 'E_widerface', 'F_omnidoc', 'G_wilddet3d', 'H_metrics_only')
FLAT_ADAPTERS = ('B_coco_flat', 'G_wilddet3d')

def xyxy_box(value):
    require(isinstance(value, (list, tuple)) and len(value) == 4 and all((number(x) for x in value)), 'four finite xyxy coordinates required')
    x1, y1, x2, y2 = value
    result = [x1, y1, x2 - x1, y2 - y1]
    box(result)
    return result

def native_record(record, adapter, sequence, category_names=None):
    """Represent native geometry without altering inference, ranking or class selection.

    Source fields remain in extras. Encoded category names are local to this
    export and are explicitly NOT annotation IDs. UI referring expressions have
    no category, represented by null rather than a fabricated semantic class.
    """
    if adapter in ('A_coco_records', 'B_coco_flat', 'C_zeroshot', 'G_wilddet3d', 'H_metrics_only'):
        return record
    result = dict(record)
    converted = []
    if adapter == 'D_xyxy_records':
        image_id = record['file_name'] if sequence == 54 else record['image_id']
        if sequence == 54:
            for row in record['native_xyxy_conf_class']:
                require(isinstance(row, list) and len(row) == 6, 'SKU row needs six fields')
                require(number(row[5]) and int(row[5]) == row[5], 'SKU class must be integral')
                converted.append(dict(image_id=image_id, category_id=int(row[5]), bbox=xyxy_box(row[:4]), score=row[4], native_xyxy_conf_class=row))
            result.pop('native_xyxy_conf_class')
        else:
            require(sequence in (25, 32, 35), 'unknown xyxy model semantics')
            field = 'detections' if sequence == 25 else 'predictions'
            for row in record[field]:
                converted.append(dict(row, image_id=image_id, category_id=None if sequence == 25 else 1, bbox=xyxy_box(row['bbox_xyxy'])))
            result.pop(field)
    elif adapter == 'E_widerface':
        require(all((isinstance(record[k], str) for k in ('event', 'filename'))), 'WIDER event/filename required')
        image_id = json.dumps([record['event'], record['filename']], ensure_ascii=False, separators=(',', ':'))
        for row in record['boxes']:
            require(isinstance(row, list) and len(row) == 5, 'WIDER row needs xywh and score')
            converted.append(dict(image_id=image_id, category_id=1, bbox=row[:4], score=row[4], native_box=row))
        result.pop('boxes')
    elif adapter == 'F_omnidoc':
        if 'page_info' in record:
            image_id = record['page_info']['image_path']
            for row in record['layout_dets']:
                poly = row['poly']
                require(isinstance(poly, list) and len(poly) >= 6 and (len(poly) % 2 == 0) and all((number(x) for x in poly)), 'finite polygon coordinates required')
                bounds = [min(poly[::2]), min(poly[1::2]), max(poly[::2]), max(poly[1::2])]
                converted.append(dict(row, image_id=image_id, category_id=category_names[row['category_type']], bbox=xyxy_box(bounds)))
            result.pop('layout_dets')
        else:
            image_id = record['image_id']
            for row in record['boxes']:
                cid = row['cls_id'] if sequence == 38 else row['native_class_id']
                converted.append(dict(row, image_id=image_id, category_id=cid, bbox=xyxy_box(row['coordinate'])))
            result.pop('boxes')
    else:
        raise ValueError('unsupported native record adapter: ' + adapter)
    result.update(image_id=image_id, detections=converted)
    return result

def prepared_rows(source, adapter, sequence, category_names=None):
    for record in iter_source(source, adapter):
        require(isinstance(record, dict), 'prediction record must be an object')
        yield native_record(record, adapter, sequence, category_names)

def install_table_capture(official, run):
    """Patch only the per-run copied evaluator, before importing it (model 15).

    Observe the existing postprocessor result and existing raw GriTS inputs.
    No additional forward call, filtering, softmax or coordinate transform.
    """
    path = Path(official) / 'src/eval.py'
    content = path.read_text()
    marker = '        # Unified output capture: existing native results only.'
    require(marker not in content, 'table capture already installed in this evaluation copy')
    anchor = "        results = postprocessors['bbox'](outputs, orig_target_sizes)"
    require(content.count(anchor) == 1, 'native table postprocessing anchor changed')
    insertion = anchor + '\n' + marker + '\n        from detection_exchange_capture import capture_table_batch\n'
    insertion += '        capture_table_batch(' + repr(str(Path(run) / 'predictions.jsonl')) + ', targets, results, outputs)'
    updated = content.replace(anchor, insertion)
    cell_anchor = '    metrics[\'id\'] = target["img_path"].split(\'/\')[-1].split(\'.\')[0]'
    require(updated.count(cell_anchor) == 1, 'native table cell capture anchor changed')
    cell_capture = cell_anchor + '\n    from detection_exchange_capture import capture_table_cells\n' + '    capture_table_cells(' + repr(str(Path(run) / 'grits_cells')) + ', metrics["id"], "ground_truth_table_crop_padding2_pixels", true_cells, pred_cells)'
    updated = updated.replace(cell_anchor, cell_capture)
    path.write_text(updated)
    write(Path(run) / 'detection_capture.json', dict(schema='model-eval/table-prediction-capture/v1', original_evaluator_sha256=hashlib.sha256(content.encode()).hexdigest(), captured_evaluator_sha256=file_hash(path), capture_helper_sha256=file_hash(Path(__file__)), evaluator_path='official_source/src/eval.py', captures='existing_bbox_result_raw_grits_inputs_and_native_table_cells', additional_forward_calls=0, coordinate_space='ground_truth_table_crop_padding2_pixels', timing_note='Adds serialization after postprocessing; not part of model forward time.'))

def capture_table_cells(directory, table_id, coordinate_space, ground_truth_cells, prediction_cells):
    """Observe already-computed native cells; unique per-table files permit CPU workers."""
    import sys
    root = Path(directory)
    try:

        def native(value):
            if hasattr(value, 'tolist'):
                return value.tolist()
            if hasattr(value, 'item'):
                return value.item()
            raise TypeError('unsupported native cell value ' + type(value).__name__)
        payload = json.loads(json.dumps(dict(table_id=str(table_id), coordinate_space=coordinate_space, ground_truth_cells=ground_truth_cells, prediction_cells=prediction_cells), default=native, allow_nan=False))
        target = root / (hashlib.sha256(str(table_id).encode()).hexdigest() + '.json')
        require(not target.exists(), 'duplicate table cell capture')
        write(target, payload)
    except Exception as exc:
        print('Table cell capture failed: ' + str(exc), file=sys.stderr, flush=True)
        try:
            write(root.parent / 'detection_capture_error.json', dict(state='capture_failed', error=str(exc), table_id=str(table_id)))
        except Exception:
            pass

def capture_table_batch(path, targets, results, outputs):
    """Do not let optional serialization replace an original evaluation result."""
    import sys
    error_path = Path(path).parent / 'detection_capture_error.json'
    if error_path.exists():
        return
    try:
        _capture_table_batch(path, targets, results, outputs)
    except Exception as exc:
        error = dict(state='capture_failed', error=type(exc).__name__ + ': ' + str(exc), native_evaluation_continues=True, partial_predictions_must_not_be_treated_as_complete=True)
        print('Table prediction capture failed: ' + error['error'], file=sys.stderr, flush=True)
        try:
            write(error_path, error)
        except Exception as write_error:
            print('Cannot save capture failure: ' + str(write_error), file=sys.stderr, flush=True)

def _capture_table_batch(path, targets, results, outputs):
    """Save completed table images, including GriTS raw tensors and crop identity."""

    def values(value):
        return value.detach().cpu().tolist() if hasattr(value, 'detach') else value
    require(len(targets) == len(results), 'table target/result batch differs')
    with Path(path).open('a', encoding='utf-8') as stream:
        for index, (target, result) in enumerate(zip(targets, results)):
            image_id = values(target['image_id'])
            if isinstance(image_id, list):
                require(len(image_id) == 1, 'single table image ID required')
                image_id = image_id[0]
            scores = values(result['scores'])
            labels = values(result['labels'])
            boxes = values(result['boxes'])
            require(len(scores) == len(labels) == len(boxes), 'table result fields differ in length')
            rows = [dict(image_id=image_id, category_id=label, bbox=xyxy_box(bounds), score=score, bbox_xyxy=bounds) for label, bounds, score in zip(labels, boxes, scores)]
            record = dict(image_id=image_id, image_path=target['img_path'], orig_size=values(target['orig_size']), detections=rows, grits_raw_pred_logits=values(outputs['pred_logits'][index]), grits_raw_pred_boxes_cxcywh_normalized=values(outputs['pred_boxes'][index]))
            stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + '\n')

def typed_id(value):
    require(integer(value) or isinstance(value, str), 'image ID must be an integer or string')
    return ('integer' if integer(value) else 'string', value)

def normalize_rows(rows, adapter, image_mapping, *, allow_classless=False):
    """Only express saved, postprocessed boxes; never filter/reorder predictions."""
    seen = set()
    for record_index, record in enumerate(rows):
        require(isinstance(record, dict), 'prediction record must be an object')
        original_id = record.get('image_id')
        key = typed_id(original_id)
        require(key in image_mapping, 'prediction image absent from explicit image mapping')
        image_id = image_mapping[key]
        if adapter == 'B_coco_flat':
            detections = [record]
            record_extra = {}
        else:
            require(key not in seen, 'duplicate per-image prediction record')
            seen.add(key)
            field = 'detections' if adapter == 'A_coco_records' else 'predictions'
            detections = record.get(field)
            require(isinstance(detections, list), field + ' list required, including empty images')
            record_extra = {k: v for k, v in record.items() if k not in ('image_id', field)}
        normalized = []
        extras = []
        for index, row in enumerate(detections):
            require(isinstance(row, dict), 'detection must be an object')
            require(typed_id(row.get('image_id')) == key, 'nested prediction image ID mismatch')
            require(integer(row.get('category_id')) or (allow_classless and row.get('category_id') is None), 'explicit integer category ID required')
            box(row.get('bbox'))
            require(number(row.get('score')), 'finite ranking score required')
            normalized.append(dict(image_id=image_id, category_id=row['category_id'], bbox=row['bbox'], score=row['score']))
            extra = {k: v for k, v in row.items() if k not in CORE_FIELDS}
            if extra:
                extras.append(dict(detection_index=index, fields=extra))
        yield dict(image_id=image_id, source_image_id=original_id, detections=normalized, record_index=record_index, record_fields=record_extra, detection_fields=extras)

def export_predictions(source, output, adapter, *, sequence, model_id, completed_image_ids=None, coverage_evidence=None, source_metadata=None, coordinate_space='original_image_pixels', native_evaluator=None, source_run_success=None):
    """Atomic, streaming native prediction output. Does not assert GT alignment or score equivalence."""
    import os
    import shutil
    import tempfile
    source = Path(source)
    output = Path(output)
    require(adapter in ('A_coco_records', 'B_coco_flat', 'C_zeroshot') + SPECIAL_ADAPTERS, 'unsupported prediction adapter')
    canonical_adapter = 'B_coco_flat' if adapter in FLAT_ADAPTERS else 'C_zeroshot' if adapter == 'C_zeroshot' else 'A_coco_records'
    require(not output.exists(), 'choose a new output directory; no overwrite')
    before = source.stat()
    category_names = {}
    if adapter == 'F_omnidoc' and source.suffix == '.json':
        names = {row['category_type'] for record in iter_source(source, adapter) for row in record['layout_dets']}
        require(all((isinstance(name, str) for name in names)), 'Omni category names must be strings')
        category_names = {name: index + 1 for index, name in enumerate(sorted(names))}
    source_ids = set()
    record_ids = []
    for row in prepared_rows(source, adapter, sequence, category_names):
        key = typed_id(row.get('image_id'))
        source_ids.add(key)
        if adapter not in FLAT_ADAPTERS:
            record_ids.append(key)
    if adapter not in FLAT_ADAPTERS:
        require(len(record_ids) == len(set(record_ids)), 'duplicate per-image prediction record')
        if completed_image_ids is None:
            completed_image_ids = [key[1] for key in record_ids]
    scope = None if completed_image_ids is None else [typed_id(x) for x in completed_image_ids]
    if scope is not None:
        require(len(scope) == len(set(scope)), 'duplicate coverage image ID')
        require(source_ids <= set(scope), 'predictions contain images outside coverage')
        if adapter not in FLAT_ADAPTERS:
            require(source_ids == set(scope), 'record and coverage image IDs differ')
    all_ids = set(scope) if scope is not None else source_ids
    integer_ids = {v for kind, v in all_ids if kind == 'integer'}
    mapping = {key: key[1] for key in all_ids if key[0] == 'integer'}
    next_id = max(integer_ids | {0}) + 1
    for key in sorted(all_ids):
        if key[0] == 'string':
            mapping[key] = next_id
            next_id += 1
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix='.' + output.name + '-', dir=output.parent))
    count = 0
    categories = set()
    with_predictions = set()
    extras_count = 0
    core_hash = hashlib.sha256()
    try:
        with (temporary / 'predictions.json').open('w', encoding='utf-8') as flat, (temporary / 'extras.jsonl').open('w', encoding='utf-8') as extras:
            flat.write('[')
            for item in normalize_rows(prepared_rows(source, adapter, sequence, category_names), canonical_adapter, mapping, allow_classless=sequence == 25 and adapter == 'D_xyxy_records'):
                if item['record_fields'] or item['detection_fields']:
                    payload = {k: v for k, v in item.items() if k != 'detections'}
                    extras.write(json.dumps(payload, ensure_ascii=False, allow_nan=False) + '\n')
                    extras_count += 1
                for row in item['detections']:
                    text = json.dumps(row, ensure_ascii=False, separators=(',', ':'), allow_nan=False)
                    if count:
                        flat.write(',')
                    flat.write(text)
                    core_hash.update((text + '\n').encode())
                    count += 1
                    categories.add(row['category_id'])
                    with_predictions.add(row['image_id'])
            flat.write(']\n')
        after = source.stat()
        require((before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns), 'source predictions changed during export')
        image_map = [dict(source_type=k[0], source_id=k[1], image_id=mapping[k]) for k in sorted(mapping)]
        write(temporary / 'image_mapping.json', image_map)
        completed = None if scope is None else [mapping[key] for key in scope]
        protocol = dict(schema=PREDICTION_SCHEMA, model=dict(sequence=sequence, model_id=model_id), adapter=adapter, bbox_format='xywh', coordinate_space=coordinate_space, prediction_stage='after_original_postprocessing', prediction_count=count, score_semantics='finite_original_ranking_score', category_ids_observed=sorted((c for c in categories if c is not None)), category_mapping='preserved_saved_category_ids; GT alignment requires explicit annotation binding', completed_image_ids=completed, empty_prediction_image_ids=None if completed is None else sorted(set(completed) - with_predictions), coverage=dict(state='recorded_image_ids' if completed is not None else 'unknown_no_image_manifest', evidence=coverage_evidence, full_dataset_verified=False), source_run_success=source_run_success, native_evaluator=native_evaluator, scoring_ready=False, scoring_note='Prediction exchange only; bind original GT and evaluation protocol before scoring.', conversion=dict(additional_filter=False, additional_nms=False, clipping=False, rounding=False, reordered=False, image_ids_encoded=any((k[0] == 'string' for k in mapping))), source=dict(path=os.path.relpath(source, output), sha256=file_hash(source), adapter_sha256=file_hash(Path(__file__)), metadata=source_metadata or []), artifacts=dict(predictions='predictions.json', image_mapping='image_mapping.json', extras='extras.jsonl'), ordered_prediction_sha256=core_hash.hexdigest(), extra_records=extras_count)
        protocol['category_semantics'] = dict(namespace='saved_native_ids', mapping=None)
        if category_names:
            protocol['category_semantics'] = dict(namespace='export_local_name_ids', mapping=category_names, note='Reversible name encoding only. Bind by original category name; IDs may differ across exports.')
        if sequence in (26, 32, 35):
            protocol['category_semantics'] = dict(namespace='native_single_class_protocol', mapping={'face' if sequence in (26, 35) else 'hand': 1})
        if sequence == 54:
            protocol['category_semantics'] = dict(namespace='native_model_ids', mapping={'empty': 0, 'product': 1}, note='Both retained; native scorer selects product and uses inclusive +1 geometry.')
        if sequence == 25:
            protocol['category_semantics'] = dict(namespace='classless_referring_expression', mapping=None, note='category_id=null; instruction and target remain in extras; ordinary category AP is not defined.')
        protocol['category_mapping'] = protocol['category_semantics']['namespace'] + '; GT alignment requires explicit annotation binding'
        protocol['conversion']['geometry'] = 'polygon_envelope_xywh' if category_names else 'xyxy_to_xywh' if adapter in ('D_xyxy_records', 'F_omnidoc', 'H_metrics_only') else 'preserved_xywh'
        protocol['artifact_sha256'] = {name: file_hash(temporary / name) for name in ['predictions.json', 'image_mapping.json', 'extras.jsonl']}
        write(temporary / 'protocol.json', protocol)
        require(not output.exists(), 'output appeared during export')
        temporary.rename(output)
        return protocol
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)

def model_adapter(sequence):
    inventory = read(Path(__file__).with_name('detection_inventory.json'))
    item = next((m for m in inventory['models'] if m['sequence'] == sequence), None)
    if item is None or item['adapter_group'] == 'I_skipped':
        return None
    return item

def prediction_sources(root, sequence):
    """Search only the selected model's run outputs, never prepared data or weights."""
    import os
    item = model_adapter(sequence)
    if item is None or not Path(root).exists():
        return []
    names = {'detect_2D_results.json'} if sequence == 27 else {'native_predictions.jsonl', 'predictions_omni.json'} if sequence in (38, 39) else {'predictions.mt.json'} if sequence == 23 else {'predictions.json', 'predictions_validation.json', 'predictions.jsonl'} if sequence == 70 else {'predictions.json'} if item['adapter_group'] == 'B_coco_flat' else {'predictions.jsonl'}
    sources = []
    for folder, dirs, files in os.walk(root, followlinks=False):
        dirs[:] = [n for n in dirs if not n.startswith(('.', '_', 'standardized', 'score_')) and n not in ('standardized', 'timing', 'images', 'labels', 'assets', 'node_modules', 'site-packages')]
        for name in sorted(names.intersection(files)):
            path = Path(folder) / name
            if sequence in (38, 39) and name == 'predictions_omni.json' and ('native_predictions.jsonl' in files):
                continue
            if sequence == 70 and name != 'predictions.jsonl' and ('predictions.jsonl' in files):
                continue
            sources.append(path)
    return sorted(sources)

def prediction_snapshot(root, sequence):
    files = prediction_sources(root, sequence)
    if sequence == 15 and Path(root).exists():
        files += list(Path(root).rglob('detection_capture_error.json'))
    return {str(p): (p.stat().st_size, p.stat().st_mtime_ns) for p in files}

def source_context(source, root, sequence, phase):
    """Keep explicit evidence; absence of a manifest must never imply zero detections."""
    import os
    source = Path(source)
    root = Path(root)
    metadata = []
    documents = []
    current = source.parent
    while True:
        for name in ('coverage.json', 'detection_scope.json', 'detection_capture.json', 'detection_capture_error.json', 'status.json', 'protocol.json', 'expected_ids.json'):
            path = current / name
            if path.is_file():
                value = read(path)
                documents.append((path, value))
                metadata.append(dict(path=os.path.relpath(path, source.parent / 'standardized'), sha256=file_hash(path)))
        if current == root or root not in current.parents:
            break
        current = current.parent

    def nested(obj, path):
        for key in path:
            if not isinstance(obj, dict):
                return None
            obj = obj.get(key)
        return obj
    paths = [('image_ids',), ('expected_ids',), ('processed_image_ids',), ('protocol', 'dataset', 'image_ids'), ('protocol', 'dataset_snapshot', 'image_ids'), ('dataset', 'image_ids')]
    for path, value in documents:
        candidates = [value] if path.name == 'expected_ids.json' and isinstance(value, list) else []
        candidates += [nested(value, key) for key in paths]
        for candidate in candidates:
            if isinstance(candidate, list) and all((integer(x) or isinstance(x, str) for x in candidate)):
                return (candidate, dict(kind='explicit_image_list', path=os.path.relpath(path, source.parent / 'standardized'), sha256=file_hash(path), phase=phase), metadata)
        if sequence == 12:
            files = nested(value, ('protocol', 'source_files'))
            if isinstance(files, list) and all((isinstance(x, dict) and isinstance(x.get('image'), str) for x in files)):
                return ([Path(x['image']).stem for x in files], dict(kind='native_m12_image_stems', path=os.path.relpath(path, source.parent / 'standardized'), sha256=file_hash(path), phase=phase), metadata)
    if sequence == 24 and (source.parent / 'val.txt').is_file():
        path = source.parent / 'val.txt'
        ids = [int(Path(x).stem) for x in path.read_text().splitlines() if x.strip()]
        return (ids, dict(kind='native_m24_numeric_filenames', path=os.path.relpath(path, source.parent / 'standardized'), sha256=file_hash(path), phase=phase), metadata)
    if sequence == 41 and (source.parent / 'ground_truth.json').is_file():
        path = source.parent / 'ground_truth.json'
        ids = [x['id'] for x in read(path)['images']]
        return (ids, dict(kind='original_selected_ground_truth', path=os.path.relpath(path, source.parent / 'standardized'), sha256=file_hash(path), phase=phase), metadata)
    if sequence == 46:
        for path, value in documents:
            if path.name == 'coverage.json' and isinstance(value.get('images'), list):
                ids = [int(Path(x).stem) if Path(x).stem.isnumeric() else Path(x).stem for x in value['images']]
                return (ids, dict(kind='native_m46_validator_filenames', path=os.path.relpath(path, source.parent / 'standardized'), sha256=file_hash(path), phase=phase), metadata)
    return (None, dict(kind='per_image_records' if source.suffix == '.jsonl' else 'not_recorded', phase=phase), metadata)

def _finalize_run(root, sequence, phase, before=None, run_success=True):
    """Called by modelctl for supported detection models on every device; independent of GPU libraries.

    A failed scorer may leave valid inference outputs. Export them as such and
    preserve the failed run state, never converting a scoring failure to success.
    """
    import sys
    root = Path(root)
    item = model_adapter(sequence)
    if item is None:
        return None
    report = dict(schema='model-eval/detection-export-report/v1', sequence=sequence, phase=phase, source_run_success=run_success, items=[], state='no_new_predictions')
    for source in prediction_sources(root, sequence):
        parts = source.relative_to(root).parts
        if phase == 'full' and 'smoke' in parts or (phase == 'smoke' and 'full' in parts):
            continue
        stamp = (source.stat().st_size, source.stat().st_mtime_ns)
        if before is not None and before.get(str(source)) == stamp:
            continue
        relative = str(source.relative_to(root))
        output = source.parent / 'standardized'
        entry = dict(source=relative, output=str(output.relative_to(root)))
        try:
            ids, evidence, metadata = source_context(source, root, sequence, phase)
            adapter = item['adapter_group']
            if sequence == 70 and source.suffix == '.json':
                adapter = 'B_coco_flat'
            if output.exists():
                previous = read(output / 'protocol.json')
                if previous['source']['sha256'] == file_hash(source) and previous['source'].get('adapter_sha256') == file_hash(Path(__file__)) and (previous['source'].get('metadata') == metadata) and (previous.get('source_run_success') == run_success) and all(((output / name).is_file() and file_hash(output / name) == value for name, value in previous.get('artifact_sha256', {}).items())) and (len(previous.get('artifact_sha256', {})) == 3):
                    entry.update(state='existing', prediction_count=previous['prediction_count'])
                    report['items'].append(entry)
                    continue
                import time
                output = output.with_name('standardized-' + str(time.time_ns()))
                entry['output'] = str(output.relative_to(root))
            protocol = export_predictions(source, output, adapter, sequence=sequence, model_id=item['model_id'], completed_image_ids=ids, coverage_evidence=evidence, source_metadata=metadata, coordinate_space='ground_truth_table_crop_padding2_pixels' if sequence == 15 else 'author_padded_canvas_evaluation_pixels' if sequence == 30 else 'original_image_pixels', native_evaluator=dict(preserved=True, task_notes=item['special_notes']), source_run_success=run_success)
            entry.update(state='exported', prediction_count=protocol['prediction_count'], coverage=protocol['coverage']['state'], scoring_ready=protocol['scoring_ready'])
        except Exception as exc:
            entry.update(state='export_failed', error=type(exc).__name__ + ': ' + str(exc))
            print('统一检测输出失败（原运行结果保留）: ' + relative + ' — ' + entry['error'], file=sys.stderr, flush=True)
        report['items'].append(entry)
    if report['items']:
        report['state'] = 'export_failed' if any((x['state'] == 'export_failed' for x in report['items'])) else 'exported'
    if sequence == 15 and root.exists():
        errors = [dict(path=str(path.relative_to(root)), detail=read(path)) for path in sorted(root.rglob('detection_capture_error.json')) if not any((part.startswith(('standardized', '_', '.')) for part in path.relative_to(root).parts)) and (before is None or before.get(str(path)) != (path.stat().st_size, path.stat().st_mtime_ns))]
        if errors:
            report.update(state='export_failed', capture_errors=errors)
    root.mkdir(parents=True, exist_ok=True)
    write(root / ('detection_export_' + phase + '.json'), report)
    return report

def finalize_run(root, sequence, phase, before=None, run_success=True):
    import sys
    try:
        return _finalize_run(root, sequence, phase, before, run_success)
    except Exception as exc:
        report = dict(sequence=sequence, phase=phase, state='export_failed', source_run_success=run_success, error=type(exc).__name__ + ': ' + str(exc))
        print('统一检测输出未完成: ' + report['error'], file=sys.stderr, flush=True)
        try:
            write(Path(root) / ('detection_export_' + phase + '.json'), report)
        except Exception as write_error:
            print('无法写导出状态: ' + str(write_error), file=sys.stderr, flush=True)
        return report

def capture_scoring_inputs(annotation, image_ids, category_ids, source):
    """Observe the evaluator's actual GT and selected scope, without changing it."""
    import os
    root = os.environ.get('BENCH_SCORING_CAPTURE_ROOT')
    if not root:
        return
    try:
        images = set(image_ids)
        categories = set(category_ids)
        snapshot = dict(annotation, images=[r for r in annotation['images'] if r['id'] in images], categories=[r for r in annotation['categories'] if r['id'] in categories], annotations=[r for r in annotation['annotations'] if r['image_id'] in images and r['category_id'] in categories])
        fingerprint = digest(snapshot)
        folder = Path(root) / fingerprint
        folder.mkdir(parents=True, exist_ok=True)
        if not (folder / 'annotations.json').exists():
            write(folder / 'annotations.json', snapshot)
            write(folder / 'source.json', dict(source=source, content_sha256=fingerprint, note='GT actually passed to the native evaluator, including its selected scope.'))
    except Exception as exc:
        import sys
        print('Unified GT capture failed: ' + repr(exc), file=sys.stderr, flush=True)
        write(Path(root) / ('capture-error-' + str(os.getpid()) + '.json'), dict(error=repr(exc), source=source))

def capture_yolo_ground_truth(validator):
    """Read the validator's original normalized labels and actual JSON class map."""
    import os
    root = os.environ.get('BENCH_SCORING_CAPTURE_ROOT')
    if not root:
        return
    try:
        labels = validator.dataloader.dataset.labels
        class_map = validator.class_map
        names = validator.names
        items = names.items() if isinstance(names, dict) else enumerate(names)
        categories = [dict(id=int(class_map[int(index)]), name=str(name)) for index, name in items]
        annotation = dict(images=[], annotations=[], categories=categories)
        source_ids = []
        for index, label in enumerate(labels, 1):
            require(label.get('normalized') is True and label.get('bbox_format') == 'xywh', 'unsupported native YOLO label geometry')
            shape = label.get('shape')
            if shape is None:
                from PIL import Image, ImageOps
                with Image.open(label['im_file']) as image:
                    width, height = ImageOps.exif_transpose(image).size
            else:
                height, width = (int(v) for v in shape)
            stem = Path(label['im_file']).stem
            source_id = int(stem) if stem.isnumeric() else stem
            source_ids.append(source_id)
            annotation['images'].append(dict(id=index, file_name=Path(label['im_file']).name, width=width, height=height))
            for raw_box, raw_class in zip(label['bboxes'], label['cls']):
                cx, cy, w, h = (float(x) for x in raw_box)
                cid = int(raw_class.item() if hasattr(raw_class, 'item') else raw_class[0])
                bbox = [(cx - w / 2) * width, (cy - h / 2) * height, w * width, h * height]
                annotation['annotations'].append(dict(id=len(annotation['annotations']) + 1, image_id=index, category_id=int(class_map[cid]), bbox=bbox, area=bbox[2] * bbox[3], iscrowd=0))
        fingerprint = digest(annotation)
        capture_scoring_inputs(annotation, [r['id'] for r in annotation['images']], [r['id'] for r in categories], 'ultralytics_native_dataset_labels')
        write(Path(root) / fingerprint / 'image_binding.json', [dict(annotation_id=i, prediction_source_id=value) for i, value in enumerate(source_ids, 1)])
    except Exception as exc:
        import sys
        print('YOLO GT observation unavailable: ' + repr(exc), file=sys.stderr, flush=True)
        write(Path(root) / ('yolo-error-' + str(os.getpid()) + '.json'), dict(error=repr(exc)))

def install_yolo_capture_when_imported():
    """Do not import a competing Ultralytics source before the selected runner."""
    import builtins
    import sys
    original_import = builtins.__import__

    def observe():
        for name in ('ultralytics.engine.validator', 'ultralytics.yolo.engine.validator'):
            module = sys.modules.get(name)
            cls = getattr(module, 'BaseValidator', None)
            if cls is None or getattr(cls.__call__, '_unified_gt_capture', False):
                continue
            original_call = cls.__call__

            def call(self, *args, _original=original_call, **kwargs):
                try:
                    return _original(self, *args, **kwargs)
                finally:
                    capture_yolo_ground_truth(self)
            call._unified_gt_capture = True
            cls.__call__ = call

    def imported(name, globals=None, locals=None, fromlist=(), level=0):
        result = original_import(name, globals, locals, fromlist, level)
        if name.startswith('ultralytics'):
            observe()
        return result
    builtins.__import__ = imported
    observe()

def install_scoring_capture():
    """Optional observation at evaluator boundaries; no model or GPU patches."""
    import importlib
    import os
    if not os.environ.get('BENCH_SCORING_CAPTURE_ROOT'):
        return
    install_yolo_capture_when_imported()
    for module_name, class_name, method_name, gt_attr, image_attr, category_attr in [('pycocotools.cocoeval', 'COCOeval', 'evaluate', 'cocoGt', 'imgIds', 'catIds'), ('lvis.eval', 'LVISEval', 'evaluate', 'lvis_gt', 'img_ids', 'cat_ids')]:
        try:
            cls = getattr(importlib.import_module(module_name), class_name)
        except ImportError:
            continue
        original = getattr(cls, method_name)
        if getattr(original, '_unified_gt_capture', False):
            continue

        def observed(self, *args, _original=original, _gt=gt_attr, _images=image_attr, _cats=category_attr, _source=module_name, **kwargs):
            ground_truth = getattr(self, _gt)
            capture_scoring_inputs(ground_truth.dataset, getattr(self.params, _images), getattr(self.params, _cats), _source)
            return _original(self, *args, **kwargs)
        observed._unified_gt_capture = True
        setattr(cls, method_name, observed)
