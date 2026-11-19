"""Prepare explicit scoring inputs. Dataset conversion belongs here, never in scorers."""
from __future__ import annotations
import argparse
import copy
import json
from pathlib import Path
import zipfile
from detection import read, write, require, typed_id, integer, number, file_hash, digest, POLICY, validate_annotation, PREDICTION_SCHEMA
from scoring_common import COCO_SCHEMA, LVIS_SCHEMA, GRITS_SCHEMA, LVIS_POLICY, GRITS_POLICY, validate_lvis, validate_bbox_predictions, publish_bbox

def fresh_output(path):
    """Preserve prior scoring bundles, including incomplete previous attempts."""
    from datetime import datetime, timezone
    path = Path(path)
    if not path.exists():
        return path
    return path.with_name(path.name + '-' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%f'))

def read_annotation(path, member=None):
    if member:
        with zipfile.ZipFile(path) as z:
            return json.loads(z.read(member))
    return read(path)

def exchange_inputs(exchange):
    exchange = Path(exchange)
    p = read(exchange / 'protocol.json')
    require(p.get('schema') == PREDICTION_SCHEMA, 'prediction exchange v2 required')
    require(set(p.get('artifact_sha256', {})) == {'predictions.json', 'image_mapping.json', 'extras.jsonl'}, 'exchange hashes incomplete')
    for name, h in p['artifact_sha256'].items():
        require(file_hash(exchange / name) == h, 'exchange hash mismatch: ' + name)
    pred = read(exchange / 'predictions.json')
    mapping = read(exchange / 'image_mapping.json')
    require(len(pred) == p['prediction_count'], 'prediction count differs')
    ids = [r['image_id'] for r in mapping]
    require(all((integer(i) for i in ids)) and len(ids) == len(set(ids)), 'duplicate exchange image IDs')
    keys = [typed_id(r['source_id']) for r in mapping]
    require(len(keys) == len(set(keys)), 'duplicate exchange source IDs')
    require(all((r['source_type'] == typed_id(r['source_id'])[0] for r in mapping)), 'source ID type mismatch')
    return (p, pred, mapping)

def bind_bbox(exchange, annotation, binding, output, protocol_name, source):
    """Use exact explicit ID maps; never infer classes or match file basenames."""
    p, pred, image_mapping = exchange_inputs(exchange)
    require(protocol_name in ('coco', 'lvis'), 'unknown preparation protocol')
    schema, policy, validator = (COCO_SCHEMA, POLICY, validate_annotation) if protocol_name == 'coco' else (LVIS_SCHEMA, LVIS_POLICY, validate_lvis)
    ids = validator(annotation)
    require(binding.get('annotation_coordinate_space') == p.get('coordinate_space'), 'annotation/prediction coordinate space not explicitly matched')
    require(isinstance(binding.get('dataset'), dict) and binding['dataset'], 'dataset identity required')
    by_source = {typed_id(r['source_id']): r['image_id'] for r in image_mapping}
    mode = binding.get('image_ids')
    if mode == 'source_identity':
        imap = {gt: by_source[typed_id(gt)] for gt in ids['images'] if typed_id(gt) in by_source}
    elif isinstance(mode, list):
        imap = {}
        used = set()
        for row in mode:
            gt = row['annotation_id']
            require(integer(gt) and gt in ids['images'] and (gt not in imap), 'invalid or duplicate GT image map')
            key = typed_id(row['prediction_source_id'])
            require(key in by_source and key not in used, 'prediction image mapping missing or duplicated')
            imap[gt] = by_source[key]
            used.add(key)
    else:
        raise ValueError('explicit image_ids mapping or source_identity required')
    require(set(imap) == ids['images'], 'not every selected GT image has a prediction source ID')
    require(set(imap.values()) == {r['image_id'] for r in image_mapping}, 'prediction and selected GT scope differ; subset GT explicitly first')
    completed = p.get('completed_image_ids')
    if completed is None:
        evidence = binding.get('coverage_evidence')
        require(isinstance(evidence, dict) and evidence.get('file') and evidence.get('sha256'), 'flat predictions require a verified completed-image manifest')
        require(file_hash(evidence['file']) == evidence['sha256'], 'coverage evidence file differs')
        done = read(evidence['file'])
        require(isinstance(done, list), 'coverage evidence must be a source image ID list')
        require(len(done) == len(set((typed_id(i) for i in done))) and set((typed_id(i) for i in done)) == set(by_source), 'coverage evidence differs from selected GT')
        completed = list(imap.values())
    require(len(completed) == len(set(completed)) and set(completed) == set(imap.values()), 'incomplete selected prediction coverage')
    cmap = binding.get('category_ids')
    require(cmap == 'dataset_identity' or isinstance(cmap, dict), 'explicit category_ids mapping or dataset_identity required')
    namespace = p.get('category_semantics', {}).get('namespace', 'saved_native_ids')
    require(namespace != 'classless_referring_expression', 'classless instruction localization is outside the three scoring protocols')
    require(cmap != 'dataset_identity' or namespace != 'export_local_name_ids', 'export-local names require explicit GT class mapping')
    if isinstance(cmap, dict):
        require(all((isinstance(k, str) and integer(v) and (v in ids['categories']) for k, v in cmap.items())), 'invalid category map')
        require(len(set(cmap.values())) == len(cmap), 'many-to-one class merging is not a pure format conversion')
    converted = []
    for row in pred:
        cid = row['category_id']
        require(integer(cid), 'numeric prediction class required')
        if cmap != 'dataset_identity':
            require(str(cid) in cmap, 'prediction class has no declared GT mapping')
        converted.append(dict(row, category_id=cid if cmap == 'dataset_identity' else cmap[str(cid)]))
    gt = copy.deepcopy(annotation)
    for im in gt['images']:
        im['id'] = imap[im['id']]
    for ann in gt['annotations']:
        ann['image_id'] = imap[ann['image_id']]
    validated = validator(gt)
    validate_bbox_predictions(converted, validated)
    protocol = dict(schema=schema, scoring_ready=True, evaluation=copy.deepcopy(policy), model=p['model'], dataset=binding['dataset'], completed_image_ids=sorted(completed), prediction_count=len(converted), bbox_format='xywh', coordinate_space=binding['annotation_coordinate_space'], binding=copy.deepcopy(binding), source=dict(**source, prediction_protocol_sha256=file_hash(Path(exchange) / 'protocol.json'), prediction_file_sha256=file_hash(Path(exchange) / 'predictions.json')), source_run_success=p.get('source_run_success'), scope_note='All explicitly selected GT images, including empty predictions; not a claim about a larger dataset.', new_metric_policy=bool(binding.get('new_metric_policy', False)))
    return publish_bbox(output, gt, converted, protocol)

def prepare_bbox(exchange, annotations, binding_path, output, protocol, member=None):
    return bind_bbox(exchange, read_annotation(annotations, member), read(binding_path), output, protocol, dict(annotation_sha256=file_hash(annotations), annotation_member=member, binding_sha256=file_hash(binding_path)))

def prepare_wider(exchange, dataset, output):
    """Project COCO policy: full val, valid=positive, invalid-positive boxes=COCO crowd.

    Not official WIDER Easy/Medium/Hard. No +1 coordinate change or size clipping.
    Nonpositive-area placeholders have no measurable region and are recorded as omitted.
    """
    from PIL import Image
    root = Path(dataset)
    split = root / 'data/wider_face_split.zip'
    images = root / 'data/WIDER_val.zip'
    with zipfile.ZipFile(split) as z:
        member = next((n for n in z.namelist() if n.endswith('/wider_face_val_bbx_gt.txt')))
        lines = iter(z.read(member).decode().splitlines())
    gt = dict(images=[], categories=[dict(id=1, name='face')], annotations=[])
    mapping = []
    omitted = 0
    invalid = 0
    with zipfile.ZipFile(images) as archive:
        names = set(archive.namelist())
        for line in lines:
            name = line.strip()
            if not name:
                continue
            count = int(next(lines))
            require(count >= 0, 'negative WIDER annotation count')
            native = []
            for _ in range(max(1, count)):
                native.append([int(v) for v in next(lines).split()])
            key = 'WIDER_val/images/' + name
            require(key in names, 'WIDER image missing: ' + key)
            with archive.open(key) as f:
                with Image.open(f) as im:
                    width, height = im.size
            iid = len(gt['images']) + 1
            gt['images'].append(dict(id=iid, file_name=name, width=width, height=height))
            event, filename = name.split('/', 1)
            source_id = json.dumps([event, Path(filename).stem], ensure_ascii=False, separators=(',', ':'))
            mapping.append(dict(annotation_id=iid, prediction_source_id=source_id))
            for fields in native[:count]:
                require(len(fields) == 10, 'WIDER annotation needs ten fields')
                x, y, w, h = fields[:4]
                flag = fields[7]
                require(flag in (0, 1), 'invalid WIDER invalid flag')
                if w <= 0 or h <= 0:
                    omitted += 1
                    continue
                invalid += flag
                gt['annotations'].append(dict(id=len(gt['annotations']) + 1, image_id=iid, category_id=1, bbox=[x, y, w, h], area=w * h, iscrowd=flag, wider_attributes=fields[4:]))
    binding = dict(dataset=dict(id='widerface_val_project_coco', split='validation', images=len(gt['images']), policy='All positive-area valid faces scored; invalid positive-area boxes represented as COCO crowd.', invalid_as_crowd=invalid, nonpositive_boxes_omitted=omitted, official_wider_comparable=False), annotation_coordinate_space='original_image_pixels', image_ids=mapping, category_ids={'1': 1}, new_metric_policy=True)
    return bind_bbox(exchange, gt, binding, output, 'coco', dict(annotation_sha256=file_hash(split), annotation_member=member, image_archive=str(images), image_archive_bytes=images.stat().st_size, conversion='wider_val_to_project_coco_v1'))

def prepare_grits(pairs, output, dataset, table_ids):
    from table_schema import validate_tables
    gt = []
    pred = []
    path = Path(pairs)
    if path.is_dir():
        sources = sorted(path.glob('*.json'))
        require(sources, 'no captured table cells')
        rows = (read(p) for p in sources)
        hashes = {p.name: file_hash(p) for p in sources}
    else:
        rows = (json.loads(line) for line in path.read_text().splitlines() if line.strip())
        hashes = {path.name: file_hash(path)}
    for row in rows:
        key = row['table_id']
        space = row['coordinate_space']
        gt.append(dict(table_id=key, coordinate_space=space, cells=row['ground_truth_cells']))
        pred.append(dict(table_id=key, coordinate_space=space, cells=row['prediction_cells']))
    annotation = {'tables': gt}
    predictions = {'tables': pred}
    ids = validate_tables(annotation)
    require(ids and ids == validate_tables(predictions), 'paired table scope mismatch or empty')
    expected = read(table_ids)
    require(isinstance(expected, list) and all((isinstance(i, str) for i in expected)) and (len(expected) == len(set(expected))) and (set(expected) == ids), 'missing or extra captured tables relative to selected table manifest')
    require(not (path.parent / 'detection_capture_error.json').exists(), 'table capture reported failure; do not score partial cells')
    p = dict(schema=GRITS_SCHEMA, scoring_ready=True, evaluation=GRITS_POLICY, model=None, dataset=dict(id=dataset), completed_table_ids=sorted(ids), source=dict(paired_cells_sha256=hashes, table_manifest_sha256=file_hash(table_ids)), scope_note='All paired table records in supplied file. No inferred missing/empty tables.')
    return publish_bbox(output, annotation, predictions, p)

def canonicalize_reversed_rectangles(annotation):
    """Repair reversed corners only when an explicit rectangular polygon proves them."""
    gt = copy.deepcopy(annotation)
    corrections = []
    for ann in gt['annotations']:
        x, y, w, h = ann['bbox']
        if w >= 0 and h >= 0:
            continue
        polys = ann.get('segmentation')
        require(isinstance(polys, list) and len(polys) == 1 and (len(polys[0]) == 8), 'Reversed GT bbox lacks an unambiguous rectangle')
        points = list(zip(polys[0][::2], polys[0][1::2]))
        expected = {(x, y), (x + w, y), (x + w, y + h), (x, y + h)}
        require(len(expected) == 4 and set(points) == expected, 'Reversed GT bbox and polygon corners disagree')
        require(ann['area'] == abs(w * h), 'Reversed GT bbox area disagrees with polygon')
        corrected = [min(x, x + w), min(y, y + h), abs(w), abs(h)]
        corrections.append(dict(annotation_id=ann['id'], image_id=ann['image_id'], before=ann['bbox'], after=corrected, evidence='All four rectangular polygon corners and positive recorded area agree.'))
        ann['bbox'] = corrected
    return (gt, corrections)

def declared_annotation_candidates(item, source, config):
    """Bind registered GT explicitly; never choose by matching integer IDs alone."""
    policy = item.get('fixed_scoring', {}).get('annotation_binding')
    if not policy:
        return None
    source = Path(source)
    if policy['kind'] == 'sibling':
        path = source.parent / policy['filename']
        require(path.is_file(), 'Declared sibling annotation missing: ' + str(path))
        return [(read(path), 'source_identity', dict(annotation_file=str(path), annotation_sha256=file_hash(path), source='declared_prediction_sibling'))]
    require(policy['kind'] == 'registered_archive', 'Unknown annotation binding policy')
    candidates = {}
    for spec in item.get('annotation_source_candidates', []):
        if spec.get('dataset_id') != policy['dataset_id']:
            continue
        definition = spec.get('definition', {})
        archive = definition.get('annotation_archive') or definition.get('archive')
        member = definition.get('annotation_member')
        if not archive or not member:
            continue
        path = Path(config['dataset_root']) / archive
        require(path.is_file(), 'Declared annotation archive missing: ' + str(path))
        gt = read_annotation(path, member)
        candidates[digest(gt)] = (gt, 'source_identity', dict(annotation_file=str(path), annotation_member=member, annotation_sha256=file_hash(path), source='declared_dataset_archive', dataset_id=policy['dataset_id']))
    require(len(candidates) == 1, 'Declared dataset must resolve to one authoritative annotation')
    return list(candidates.values())

def selected_dataset(item, source):
    """Keep branch/split identity on each result, independent of hardware."""
    dataset = dict(item['dataset'], scope='full_selected_native_evaluation')
    variants = item.get('fixed_scoring', {}).get('dataset_by_output_directory', {})
    if variants:
        branch = Path(source).parent.name
        require(branch in variants, 'Unregistered output dataset branch: ' + branch)
        dataset.update(variants[branch])
    return dataset

def native_annotation_candidates(sequence, source, config=None):
    """Read saved native GT and its identity; do not guess labels or geometry."""
    source = Path(source)
    if sequence == 30:
        path = source.parent / 'annotation.json'
        audit_path = source.parent / 'data_audit.json'
        if not path.is_file():
            return []
        gt = read(path)
        audit = read(audit_path)
        images = {r['image_id']: r for r in audit['images_metadata']}
        require(len(images) == len(audit['images_metadata']), 'Duplicate native image audit IDs')
        if (config or {}).get('phase') == 'smoke':
            require(gt['images'] and {im['id'] for im in gt['images']} <= set(images), 'Smoke GT outside native image audit')
        else:
            require(len(images) == len(gt['images']), 'Native image audit scope differs')
        for row in gt['images']:
            meta = images[row['id']]
            require((row['width'], row['height']) == (meta['width'], meta['height']), 'Native GT image dimensions differ')
            row.setdefault('file_name', 'embedded-sha256/' + meta['image_sha256'])
        return [(gt, 'source_identity', dict(annotation_file=str(path), annotation_sha256=file_hash(path), audit_sha256=file_hash(audit_path), source='verified_native_dataset', conversion='Add embedded-image identity only; retain native padded-canvas GT boxes.'))]
    if sequence == 54:
        import csv
        protocol_path = source.parent / 'protocol.json'
        protocol = read(protocol_path)
        path = source.parent / 'annotations_test.csv'
        original = path
        if file_hash(path) != protocol['original_csv_sha256']:
            prefix = '/project/detection/gaoyonghan/models/foduucom--table-detection-and-extraction/resources/runtime/assets/workspace/results/remaining25_prepared/sku110k_test/annotations_test.csv'
            for old, new in (config or {}).get('path_rewrites', {}).items():
                prefix = prefix.replace(old, new)
            original = Path(prefix)
            require(original.is_file() and file_hash(original) == protocol['original_csv_sha256'], 'SKU annotation CSV differs from native evaluation')
            selected_names = {r['name'] for r in protocol['source_images']}
            from collections import Counter
            with path.open(newline='') as f:
                actual = Counter((tuple(r) for r in csv.reader(f)))
            with original.open(newline='') as f:
                expected = Counter((tuple(r) for r in csv.reader(f) if r[0] in selected_names))
            require(actual == expected, 'Saved SKU annotation rows differ from original')
        images = protocol['source_images']
        lookup = {r['name']: dict(id=i + 1, file_name=r['name'], width=r['width'], height=r['height']) for i, r in enumerate(images)}
        require(len(lookup) == len(images), 'duplicate SKU image name')
        gt = dict(images=list(lookup.values()), categories=[dict(id=0, name='empty'), dict(id=1, name='product')], annotations=[])
        with path.open(newline='') as stream:
            for row in csv.reader(stream):
                name, x1, y1, x2, y2, label, width, height = row
                if name not in lookup:
                    continue
                im = lookup[name]
                require(label == 'object' and (int(width), int(height)) == (im['width'], im['height']), 'SKU GT label/dimensions differ')
                x1, y1, x2, y2 = map(float, (x1, y1, x2, y2))
                w = x2 - x1
                h = y2 - y1
                require(all((number(v) for v in (x1, y1, w, h))) and w > 0 and (h > 0), 'invalid SKU GT rectangle')
                gt['annotations'].append(dict(id=len(gt['annotations']) + 1, image_id=im['id'], category_id=1, bbox=[x1, y1, w, h], area=w * h, iscrowd=0))
        mapping = [dict(annotation_id=im['id'], prediction_source_id=im['file_name']) for im in gt['images']]
        return [(gt, mapping, dict(annotation_file=str(path), annotation_sha256=file_hash(path), protocol_sha256=file_hash(protocol_path), original_annotation_sha256=file_hash(original), source='verified_native_dataset', conversion='SKU CSV xyxy to COCO xywh; retain empty/product predictions. Empty has no GT and is not averaged by COCO. No inclusive +1 adjustment.'))]
    return []

def native_definition(path, name, namespace):
    """Load a frozen pure preparation definition without importing a model stack."""
    import ast
    tree = ast.parse(Path(path).read_text())
    nodes = [node for node in tree.body if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name == name]
    require(len(nodes) == 1, 'Native preparation definition missing or ambiguous: ' + name)
    future = ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0)
    module = ast.fix_missing_locations(ast.Module(body=[future, nodes[0]], type_ignores=[]))
    exec(compile(module, str(path), 'exec'), namespace)
    return namespace[name]

def prepare_wilddet(exchange, source, config, output):
    """Use native 2D boxes; declare project COCO crowd policy for ignored GT."""
    import contextlib, io, os, time
    from collections import defaultdict
    import numpy as np
    from pycocotools.coco import COCO
    source = Path(source)
    run = next((p for p in source.parents if (p / 'expected_ids.json').is_file()), None)
    require(run is not None, 'Missing WildDet native coverage manifest')
    protocol = read(run / 'protocol.json')
    path = Path(config['dataset_root']) / 'wilddet3d_val/annotations/InTheWild_v3_val.json'
    require(file_hash(path) == protocol['annotation_sha256'], 'WildDet annotation differs from inference')
    raw = read(path)
    require(set(read(run / 'expected_ids.json')) == {im['id'] for im in raw['images']}, 'WildDet full image scope differs')
    definition = Path(config['official_source_root']) / 'WildDet3D/wilddet3d/data/datasets/coco3d.py'
    namespace = dict(np=np, COCO=COCO, defaultdict=defaultdict, contextlib=contextlib, io=io, json=json, os=os, time=time)
    loader = native_definition(definition, 'COCO3D', namespace)
    gt = loader(str(path), [c['name'] for c in raw['categories']]).dataset
    gt = copy.deepcopy(gt)
    omitted = 0
    ignored = 0
    annotations = []
    for im in gt['images']:
        im['file_name'] = im['file_path']
    for ann in gt['annotations']:
        w, h = ann['bbox'][2:]
        if w <= 0 or h <= 0:
            omitted += 1
            continue
        flag = int(bool(ann['ignore2D']))
        ignored += flag
        annotations.append(dict(id=ann['id'], image_id=ann['image_id'], category_id=ann['category_id'], bbox=ann['bbox'], area=ann['area'], iscrowd=flag))
    gt['annotations'] = annotations
    p, pred, mapping = exchange_inputs(exchange)
    require(p.get('source_run_success') is True, 'WildDet native run did not finish')
    from detection import export_predictions
    done = read(run / 'expected_ids.json')
    new_exchange = fresh_output(source.parent / 'standardized_scoring')
    export_predictions(source, new_exchange, p['adapter'], sequence=27, model_id=p['model']['model_id'], completed_image_ids=done, coverage_evidence=dict(file=str(run / 'expected_ids.json'), sha256=file_hash(run / 'expected_ids.json')), coordinate_space=p['coordinate_space'], source_run_success=True)
    binding = dict(dataset=dict(id='wilddet3d_val_project_coco', images=len(gt['images']), native_ignored_as_crowd=ignored, nonpositive_boxes_omitted=omitted, official_wilddet_comparable=False, policy='Native truncated/projection 2D boxes and original class IDs; native ignore2D mapped to COCO crowd. COCO bbox AP, not native 3D/proximity AP.'), image_ids='source_identity', category_ids='dataset_identity', annotation_coordinate_space=p['coordinate_space'], new_metric_policy=True)
    return bind_bbox(new_exchange, gt, binding, output, 'coco', dict(annotation_sha256=file_hash(path), native_loader_sha256=file_hash(definition), native_coverage_sha256=file_hash(run / 'expected_ids.json')))

def prepare_omni(exchange, source, config, output):
    """Preserve the previously accepted seven-class Omni GT/prediction mapping."""
    import os
    from collections import Counter
    import numpy as np
    import yaml
    project = Path(__file__).resolve().parents[2] / 'resources/runtime/project'
    baseline = project / 'protocol_alignment/m038_baseline'
    reference = read(baseline / 'protocol.json')
    task = yaml.safe_load((baseline / 'layout_detection.yaml').read_text())['detection_eval']
    source = Path(source)
    inference = read(source.parent / 'protocol.json')
    status = read(source.parent / 'status.json')
    import hashlib
    require(hashlib.sha256(json.dumps(inference, sort_keys=True, ensure_ascii=False).encode()).hexdigest() == status['protocol_fingerprint'], 'Omni native identity differs')
    path = Path(config['dataset_root']) / 'omnidocbench_eval/OmniDocBench.json'
    require(file_hash(path) == inference['dataset_sha256'] == reference['dataset_sha256'], 'Omni GT differs from accepted scope')
    pages = read(path)
    smoke = config.get('phase') == 'smoke'
    selected_count = status['completed_images'] if smoke else len(pages)
    require(isinstance(selected_count, int) and 0 < selected_count <= len(pages), 'Invalid Omni selected image count')
    require(status['state'] in (('smoke_complete', 'complete') if smoke else ('complete',)) and status['completed_images'] == selected_count, 'Omni native coverage incomplete')
    asset = project.parent / 'assets/OmniDocBench'
    for name in ['dataset/detection_dataset.py', 'utils/ocr_utils.py']:
        require(file_hash(asset / name) == reference['source_files_sha256'][name], 'Omni native preprocessing source changed')
    namespace = dict(json=json, os=os, np=np)
    poly = native_definition(asset / 'utils/ocr_utils.py', 'poly2bbox', namespace)
    import ast
    tree = ast.parse((asset / 'dataset/detection_dataset.py').read_text())
    cls = next((n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'DetectionDataset'))
    cls.decorator_list = []
    module = ast.fix_missing_locations(ast.Module(body=[cls], type_ignores=[]))
    exec(compile(module, str(asset / 'dataset/detection_dataset.py'), 'exec'), namespace)
    loader = namespace['DetectionDataset'].__new__(namespace['DetectionDataset'])
    classes = reference['classes']
    categories = task['categories']
    require(categories['eval_cat']['block_level'] == classes and categories['pred_cat_mapping'] == reference['native_prediction_mapping'], 'Omni category policy differs')
    gts, names = loader.get_gts_and_img_list(None, str(path), classes, categories['eval_cat'], categories['gt_cat_mapping'])
    records = (json.loads(line) for line in source.open() if line.strip())
    expected = 0
    for expected, record in enumerate(records, 1):
        require(record['image_id'] == expected and expected <= len(pages), 'Omni prediction order/coverage differs')
        require(record['image_sha256'] == inference['images'][expected - 1]['sha256'], 'Omni image identity differs')
        require(Path(record['image_path']).name == Path(names[expected - 1]).name, 'Omni image name differs')
    require(expected == selected_count, 'Omni prediction count differs')
    p, pred, mapping = exchange_inputs(exchange)
    by_source = {typed_id(r['source_id']): r['image_id'] for r in mapping}
    require(set(by_source) == {typed_id(i) for i in range(1, selected_count + 1)}, 'Omni exchange image scope differs')
    gt = dict(images=[], categories=[dict(id=i, name=name) for i, name in enumerate(classes)], annotations=[])
    full_counts = Counter(classes[int(cid)] for row in gts for cid in row['labels'])
    require(dict(full_counts) == read(baseline / 'coverage.json')['gt_instances'], 'Omni native GT counts differ from accepted baseline')
    for idx, row in enumerate(gts[:selected_count], 1):
        iid = by_source[typed_id(idx)]
        gt['images'].append(dict(id=iid, file_name=names[idx - 1], width=row['width'], height=row['height']))
        for box, cid, ignore in zip(row['bboxes'], row['labels'], row['ignore_flags']):
            x1, y1, x2, y2 = map(float, box)
            w = x2 - x1
            h = y2 - y1
            gt['annotations'].append(dict(id=len(gt['annotations']) + 1, image_id=iid, category_id=int(cid), bbox=[x1, y1, w, h], area=w * h, iscrowd=int(ignore)))
    labels = inference['canonical_labels_by_id']
    cmap = reference['native_prediction_mapping']
    converted = []
    excluded = Counter()
    for row in pred:
        label = labels[row['category_id']]
        target = cmap.get(label, label)
        if target not in classes:
            excluded[label] += 1
            continue
        converted.append(dict(row, category_id=classes.index(target)))
    counts = Counter((classes[a['category_id']] for a in gt['annotations']))
    require(sum(counts.values()) == len(gt['annotations']), 'Omni selected GT count differs')
    validate_bbox_predictions(converted, validate_annotation(gt))
    protocol = dict(schema=COCO_SCHEMA, scoring_ready=True, evaluation=copy.deepcopy(POLICY), model=p['model'], dataset=dict(id='omnidocbench_eval', images=selected_count, classes=classes, scope='completed_smoke_images' if smoke else 'accepted_seven_class_layout_detection'), completed_image_ids=sorted(by_source.values()), prediction_count=len(converted), bbox_format='xywh', coordinate_space=p['coordinate_space'], source=dict(annotation_sha256=file_hash(path), prediction_sha256=file_hash(source), baseline_sha256=file_hash(baseline / 'protocol.json')), class_policy=dict(native_id_to_name=labels, prediction_mapping=cmap, excluded_predictions=dict(excluded), rationale='Reuse accepted baseline mapping including many-to-one semantic classes; this is an explicit evaluation policy, not lossless prediction exchange.'), source_run_success=True, new_metric_policy=True, scope_note='Same seven-class selection and native GT preprocessing; fixed common COCO scoring policy.')
    return publish_bbox(output, gt, converted, protocol)

def select_smoke_annotation(annotation, image_ids, completed, source_keys):
    """Bind smoke GT to the completed image manifest, including empty predictions."""
    require(completed is not None and len(completed) > 0, 'Smoke scoring requires a nonempty completion manifest')
    keys = {typed_id(x) for x in completed}
    require(len(keys) == len(completed) and keys == source_keys, 'Smoke completion manifest differs from prediction image mapping')
    if image_ids == 'source_identity':
        lookup = {typed_id(im['id']): im['id'] for im in annotation['images']}
        require(len(lookup) == len(annotation['images']), 'Duplicate GT image identity')
        require(keys <= set(lookup), 'Completed smoke images outside GT scope')
        selected = {lookup[k] for k in keys}
    else:
        lookup = {typed_id(im['prediction_source_id']): im['annotation_id'] for im in image_ids}
        require(len(lookup) == len(image_ids), 'Duplicate prediction source binding')
        require(keys <= set(lookup), 'Completed smoke images outside GT binding')
        selected = {lookup[k] for k in keys}
        require(len(selected) == len(keys), 'Smoke image binding is not one-to-one')
        image_ids = [im for im in image_ids if typed_id(im['prediction_source_id']) in keys]
    scoped = copy.deepcopy(annotation)
    scoped['images'] = [im for im in scoped['images'] if im['id'] in selected]
    require(len(scoped['images']) == len(keys), 'Smoke binding references missing GT images')
    scoped['annotations'] = [a for a in scoped['annotations'] if a['image_id'] in selected]
    return scoped, image_ids

def _score_model_run(root, sequence, config, python, source_directories=None, score=True):
    """Prepare per-dataset bundles then invoke one fixed scorer per bundle."""
    import subprocess
    import tempfile
    from detection import model_adapter, export_predictions, source_context
    root = Path(root)
    item = model_adapter(sequence)
    kind = item.get('fixed_scoring', {}).get('protocol') if item else None
    report = dict(sequence=sequence, state='preparing', protocol=kind, items=[])
    destination = root / 'unified_scoring.json'
    previous = read(destination) if source_directories and destination.exists() else {}
    retained = {r['source']: r for r in previous.get('items', [])}
    write(destination, report)
    if not kind:
        report.update(state='not_applicable', reason='No approved common protocol for this task')
        write(destination, report)
        return report

    def run_score(bundle, entry):
        entry.update(state='prepared', bundle=str(bundle))
    if kind == 'grits':
        paths = sorted(root.rglob('grits_cells'))
        for cells in paths:
            entry = dict(source=str(cells))
            try:
                bundle = fresh_output(cells.parent / 'score_grits')
                prepare_grits(cells, bundle, item['dataset']['id'], cells.parent / 'grits_scope.json')
                run_score(bundle, entry)
            except Exception as exc:
                entry.update(state='failed', reason=repr(exc))
            report['items'].append(entry)
        if not paths:
            report['items'].append(dict(state='failed', reason='No native cell capture found'))
    else:
        index = root / ('detection_export_' + config.get('phase', 'full') + '.json')
        if not index.is_file():
            nested = sorted(root.glob('pytorch_fp32/full/detection_export_full.json'))
            require(len(nested) == 1, 'Missing or ambiguous full prediction export index')
            index = nested[0]
        export_root = index.parent
        exported = read(index)
        captures = sorted(root.rglob('scoring_inputs/*/annotations.json'))
        if source_directories:
            require(set(source_directories) <= {Path(r['source']).parent.name for r in exported['items']}, 'Requested scoring directory not found')
        for row in exported['items']:
            if source_directories and Path(row['source']).parent.name not in source_directories:
                require(row['source'] in retained, 'No previous result for untouched source')
                report['items'].append(retained[row['source']])
                continue
            entry = dict(source=row['source'])
            report['items'].append(entry)
            try:
                require(row['state'] in ('exported', 'existing'), 'Prediction export failed')
                exchange = export_root / row['output']
                source = export_root / row['source']
                bundle = fresh_output(source.parent / ('score_' + kind))
                if sequence == 27:
                    prepare_wilddet(exchange, source, config, bundle)
                    run_score(bundle, entry)
                    continue
                if sequence == 39:
                    prepare_omni(exchange, source, config, bundle)
                    run_score(bundle, entry)
                    continue
                if sequence == 26:
                    prepare_wider(exchange, Path(config['dataset_root']) / 'widerface_val', bundle)
                    run_score(bundle, entry)
                    continue
                original = read(exchange / 'protocol.json')
                mapping = read(exchange / 'image_mapping.json')
                source_keys = {typed_id(r['source_id']) for r in mapping}
                candidates = []
                for path in captures:
                    custom = path.with_name('image_binding.json')
                    candidates.append((read(path), read(custom) if custom.exists() else 'source_identity', dict(annotation_file=str(path), annotation_sha256=file_hash(path), source='native_evaluator_observation')))
                for name in ('annotations.coco.json', 'annotation.json', 'ground_truth.json'):
                    path = source.parent / name
                    if path.is_file():
                        candidates.append((read(path), 'source_identity', dict(annotation_file=str(path), annotation_sha256=file_hash(path))))
                for spec in item.get('annotation_source_candidates', []):
                    definition = spec.get('definition', {})
                    archive, member = (definition.get('annotation_archive') or definition.get('archive'), definition.get('annotation_member'))
                    if archive and member:
                        path = Path(config['dataset_root']) / archive
                        if path.is_file():
                            candidates.append((read_annotation(path, member), 'source_identity', dict(annotation_file=str(path), annotation_member=member, annotation_sha256=file_hash(path))))
                declared = declared_annotation_candidates(item, source, config)
                native = native_annotation_candidates(sequence, source, config) if declared is None else None
                if declared is not None:
                    candidates = declared
                elif native:
                    candidates = native
                errors = []
                viable = {}
                for annotation, image_ids, evidence in candidates:
                    try:
                        if sequence == 19:
                            annotation, corrections = canonicalize_reversed_rectangles(annotation)
                            evidence = dict(evidence, annotation_corrections=corrections, correction_policy='Canonicalize explicitly reversed rectangle corners using saved segmentation; retain all images and GT objects.')
                        validator = validate_lvis if kind == 'lvis' else validate_annotation
                        validator(annotation)
                        candidate_keys = {typed_id(im['id']) for im in annotation['images']} if image_ids == 'source_identity' else {typed_id(im['prediction_source_id']) for im in image_ids}
                        completed = original.get('completed_image_ids')
                        if config.get('phase') == 'smoke':
                            original_image_count = len(annotation['images'])
                            # Completion IDs in the exchange are encoded; GT binding uses native source IDs.
                            reverse_ids = {r['image_id']: r['source_id'] for r in mapping}
                            require(len(reverse_ids) == len(mapping), 'Duplicate encoded image identity')
                            require(completed is not None and set(completed) <= set(reverse_ids), 'Completed IDs outside image mapping')
                            source_completed = [reverse_ids[i] for i in completed]
                            annotation, image_ids = select_smoke_annotation(annotation, image_ids, source_completed, source_keys)
                            evidence = dict(evidence, scope='completed_smoke_images', original_image_count=original_image_count, selected_image_count=len(annotation['images']), completed_source_ids=completed)
                        elif completed is not None:
                            require(candidate_keys == source_keys, 'GT and completed prediction scope differ')
                        else:
                            require(source_keys <= candidate_keys, 'Predictions outside GT scope')
                        classes = {c['id'] for c in annotation['categories']}
                        predictions = read(exchange / 'predictions.json')
                        require(all((r['category_id'] in classes for r in predictions)), 'GT and prediction category IDs differ')
                        key = digest(dict(annotation=annotation, image_ids=image_ids))
                        viable[key] = (annotation, image_ids, evidence)
                    except Exception as exc:
                        errors.append(str(exc))
                require(viable, 'No verified GT matches this output: ' + '; '.join(errors[:5]))
                observed = {k: v for k, v in viable.items() if v[2].get('source') == 'native_evaluator_observation'}
                if observed:
                    viable = observed
                require(len(viable) == 1, 'More than one distinct GT matches; explicit dataset binding required')
                annotation, image_ids, evidence = next(iter(viable.values()))
                if original.get('completed_image_ids') is None:
                    require(original.get('source_run_success') is True, 'Missing completion manifest for an unsuccessful native run')
                    done = [im['id'] for im in annotation['images']] if image_ids == 'source_identity' else [im['prediction_source_id'] for im in image_ids]
                    new_exchange = fresh_output(source.parent / 'standardized_scoring')
                    export_predictions(source, new_exchange, original['adapter'], sequence=sequence, model_id=item['model_id'], completed_image_ids=done, coverage_evidence=dict(kind='completed_native_evaluator_scope', **evidence), coordinate_space=original['coordinate_space'], source_run_success=True)
                    exchange = new_exchange
                binding = dict(dataset=selected_dataset(item, source), annotation_coordinate_space=original['coordinate_space'], image_ids=image_ids, category_ids='dataset_identity', new_metric_policy=True)
                if config.get('phase') == 'smoke':
                    binding['dataset']['scope'] = 'completed_smoke_images'
                bind_bbox(exchange, annotation, binding, bundle, kind, evidence)
                run_score(bundle, entry)
            except Exception as exc:
                entry.update(state='failed', reason=repr(exc))
            write(destination, report)
        if not report['items']:
            report['items'].append(dict(state='failed', reason='No exported prediction datasets found'))
    expected_state = 'completed' if score else 'prepared'
    report['state'] = expected_state if report['items'] and all((r['state'] == expected_state for r in report['items'])) else 'failed'
    write(destination, report)
    return report

def score_model_run(root, sequence, config, python, source_directories=None, score=True):
    """Publish a terminal failure even when preparation fails before item iteration."""
    try:
        return _score_model_run(root, sequence, config, python, source_directories, score=score)
    except Exception as exc:
        import traceback
        report = dict(sequence=sequence, state='failed', items=[], reason=repr(exc), traceback=traceback.format_exc())
        write(Path(root) / 'unified_scoring.json', report)
        return report

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='kind', required=True)
    batch = commands.add_parser('run')
    batch.add_argument('--root', type=Path, required=True)
    batch.add_argument('--model', type=int, required=True)
    batch.add_argument('--config', type=Path, required=True)
    batch.add_argument('--source-directory', action='append')
    for name in ['coco', 'lvis']:
        cmd = commands.add_parser(name)
        cmd.add_argument('--exchange', type=Path, required=True)
        cmd.add_argument('--annotations', type=Path, required=True)
        cmd.add_argument('--annotation-member')
        cmd.add_argument('--binding', type=Path, required=True)
        cmd.add_argument('--output', type=Path, required=True)
    cmd = commands.add_parser('wider')
    cmd.add_argument('--exchange', type=Path, required=True)
    cmd.add_argument('--dataset', type=Path, required=True)
    cmd.add_argument('--output', type=Path, required=True)
    cmd = commands.add_parser('grits')
    cmd.add_argument('--pairs', type=Path, required=True)
    cmd.add_argument('--dataset', required=True)
    cmd.add_argument('--table-ids', type=Path, required=True)
    cmd.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.kind == 'run':
        import sys
        result = score_model_run(args.root, args.model, read(args.config), sys.executable, args.source_directory)
        print(json.dumps(result, ensure_ascii=False))
        raise SystemExit(0 if result['state'] in ('completed', 'not_applicable') else 1)
    if args.kind in ('coco', 'lvis'):
        result = prepare_bbox(args.exchange, args.annotations, args.binding, args.output, args.kind, args.annotation_member)
    elif args.kind == 'wider':
        result = prepare_wider(args.exchange, args.dataset, args.output)
    else:
        result = prepare_grits(args.pairs, args.output, args.dataset, args.table_ids)
    print(json.dumps(dict(schema=result['schema'], scoring_ready=result['scoring_ready']), ensure_ascii=False))
if __name__ == '__main__':
    main()
