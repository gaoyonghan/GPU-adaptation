#!/usr/bin/env python3
"""Evaluate completed detection outputs with their validated COCO or LVIS bbox rules.

Model inference.py performs output conversion. Each bundle selects its fixed scorer by its validated protocol schema.
"""
from __future__ import annotations
import argparse
import copy
import hashlib
import importlib
import importlib.metadata
import importlib.util
import json
import math
from pathlib import Path
import platform
import sys
import types

HERE = Path(__file__).resolve().parent.parent  # Relative inputs stay task-relative.

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

def evaluate_coco_data(bundle, annotation, predictions, protocol):
    import numpy as np
    import pycocotools.coco as coco_module
    import pycocotools.cocoeval as evaluator_module
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    gt = COCO()
    gt.dataset = copy.deepcopy(annotation)
    gt.dataset.setdefault('info', {})
    gt.createIndex()
    if predictions:
        dt = gt.loadRes(copy.deepcopy(predictions))
    else:
        dt = COCO()
        dt.dataset = dict(images=copy.deepcopy(annotation['images']), categories=copy.deepcopy(annotation['categories']), annotations=[])
        dt.createIndex()
    ev = COCOeval(gt, dt, 'bbox')
    params = ev.params
    params.imgIds = protocol['image_ids']
    params.catIds = protocol['category_ids']
    params.useCats = 1
    params.iouThrs = np.linspace(0.5, 0.95, 10, endpoint=True)
    params.recThrs = np.linspace(0, 1, 101, endpoint=True)
    params.maxDets = list(POLICY['max_detections'])
    params.areaRng = copy.deepcopy(POLICY['area_ranges'])
    params.areaRngLbl = list(POLICY['area_labels'])
    ev.evaluate()
    ev.accumulate()
    ev.summarize()
    metrics = {k: None if float(v) < 0 else float(v) for k, v in zip(METRICS, ev.stats)}
    per_category = []
    for i, cat_id in enumerate(ev.params.catIds):
        precision = ev.eval['precision'][:, :, i, 0, -1]
        valid = precision[precision > -1]
        at50 = precision[0]
        at50 = at50[at50 > -1]
        per_category.append(dict(category_id=int(cat_id), name=gt.cats[cat_id]['name'], AP=float(valid.mean()) if valid.size else None, AP50=float(at50.mean()) if at50.size else None))
    result = dict(schema='model-eval/metrics/v1', task='object_detection', protocol=protocol['schema'], metric_unit='fraction', metrics=metrics, metric_aliases={'mAP_COCO': 'AP'}, per_category=per_category, scope=dict(images=len(protocol['image_ids']), completed_images=len(protocol['completed_image_ids']), categories=len(protocol['category_ids']), predictions=len(predictions), empty_prediction_images=len(protocol['empty_prediction_image_ids'])), model=protocol['model'], dataset=protocol['dataset'], input_content_sha256=protocol['content_sha256'], protocol_sha256=digest(protocol), evaluation=copy.deepcopy(POLICY), evaluator=dict(name='pycocotools.COCOeval', version=importlib.metadata.version('pycocotools'), numpy=importlib.metadata.version('numpy'), python=platform.python_version(), coco_py_sha256=file_hash(coco_module.__file__), cocoeval_py_sha256=file_hash(evaluator_module.__file__)), inference_executed=False)
    write(Path(bundle) / 'metrics.json', result)
    return result

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

def coco_evaluate(bundle):
    annotation, predictions, protocol = load_inputs(bundle, COCO_SCHEMA, POLICY)
    ids = check_bbox_scope(annotation, predictions, protocol, validate_annotation)
    evaluator_protocol = dict(
        protocol,
        image_ids=sorted(ids["images"]),
        category_ids=sorted(ids["categories"]),
        empty_prediction_image_ids=sorted(
            ids["images"] - {row["image_id"] for row in predictions}
        ),
        content_sha256=dict(
            annotations=digest(annotation), predictions=digest(predictions)
        ),
    )
    # Reuse the same COCO computation as the verified historical entry point.
    result = evaluate_coco_data(bundle, annotation, predictions, evaluator_protocol)
    result.update(
        input_sha256=protocol["artifact_sha256"],
        protocol_sha256=digest(protocol),
        new_metric_policy=protocol.get("new_metric_policy", False),
    )
    write(Path(bundle) / "metrics.json", result)
    return result

def load_lvis(source):
    # Import unchanged upstream computation without optional LVISVis (OpenCV).
    saved = {
        key: value for key, value in sys.modules.items()
        if key == "lvis" or key.startswith("lvis.")
    }
    for name in saved:
        del sys.modules[name]
    package = types.ModuleType("lvis")
    package.__path__ = [str(source / "lvis")]
    sys.modules["lvis"] = package
    try:
        return (
            importlib.import_module("lvis.lvis").LVIS,
            importlib.import_module("lvis.results").LVISResults,
            importlib.import_module("lvis.eval").LVISEval,
        )
    finally:
        for name in list(sys.modules):
            if name == "lvis" or name.startswith("lvis."):
                del sys.modules[name]
        sys.modules.update(saved)

def lvis_evaluate(bundle):
    import numpy as np

    annotation, predictions, protocol = load_inputs(bundle, LVIS_SCHEMA, LVIS_POLICY)
    ids = check_bbox_scope(annotation, predictions, protocol, validate_lvis)
    source = Path(__file__).resolve().parent / "vendor/lvis-api"
    LVIS, LVISResults, LVISEval = load_lvis(source)
    if "float" not in np.__dict__:
        np.float = float  # upstream NumPy compatibility; no metric change
    ground_truth = LVIS(str(Path(bundle) / "annotations.json"))
    if predictions:
        detections = LVISResults(ground_truth, copy.deepcopy(predictions), max_dets=300)
    else:
        # Upstream LVISResults indexes results[0]. Preserve an empty detection set.
        detections = LVISResults.__new__(LVISResults)
        detections.__dict__ = copy.deepcopy(ground_truth.__dict__)
        detections.dataset["annotations"] = []
        detections._create_index()
    evaluator = LVISEval(ground_truth, detections, "bbox")
    evaluator.params.img_ids = sorted(ids["images"])
    evaluator.params.cat_ids = sorted(ids["categories"])
    evaluator.run()
    metrics = {
        key: None if float(value) < 0 else float(value)
        for key, value in evaluator.get_results().items()
    }
    result = result_header(
        protocol, metrics,
        dict(
            name="lvis.LVISEval", numpy=np.__version__,
            source_sha256={
                name: file_hash(source / "lvis" / name)
                for name in ["eval.py", "results.py", "lvis.py"]
            },
        ),
        dict(
            images=len(ids["images"]), categories=len(ids["categories"]),
            predictions=len(predictions),
            empty_prediction_images=len(
                ids["images"] - {row["image_id"] for row in predictions}
            ),
        ),
    )
    write(Path(bundle) / "metrics.json", result)
    return result

def validate_tables(data):
    require(isinstance(data, dict) and isinstance(data.get("tables"), list),
            "tables list required")
    ids = set()
    for table in data["tables"]:
        key = table.get("table_id")
        require(isinstance(key, str) and key and key not in ids,
                "unique string table_id required")
        ids.add(key)
        require(isinstance(table.get("coordinate_space"), str) and table["coordinate_space"],
                "coordinate space required")
        cells = table.get("cells")
        require(isinstance(cells, list), "cells must be a list, including empty predictions")
        for cell in cells:
            require(isinstance(cell.get("cell_text"), str),
                    "cell_text must be explicit, including empty text")
            bbox = cell.get("bbox")
            require(
                isinstance(bbox, list) and len(bbox) == 4
                and all(number(x) for x in bbox)
                and bbox[2] >= bbox[0] and bbox[3] >= bbox[1],
                "cell bbox must be finite xyxy",
            )
            for field in ("row_nums", "column_nums"):
                values = cell.get(field)
                require(isinstance(values, list) and values
                        and all(integer(x) and x >= 0 for x in values),
                        "nonnegative span indices required")
                # Native spans may be set-derived. Keep their original order.
                require(sorted(values) == list(range(min(values), max(values) + 1)),
                        "cell spans must contain unique contiguous indices")
        # Native GriTS assigns cells to a grid in source order. Overlapping
        # predictions are retained and resolved by that same upstream operation;
        # rejecting or dropping them would change the accepted evaluation scope.
    return ids

def load_grits():
    source = Path(__file__).resolve().parent / "vendor/grits.py"
    spec = importlib.util.spec_from_file_location("official_grits_fixed", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    original = module.iou
    # Existing PyMuPDF input compatibility; retain the official scoring formula.
    module.iou = lambda a, b: original(tuple(a), tuple(b))
    return module, source

def grits_evaluate(bundle):
    import numpy as np

    annotation, predictions, protocol = load_inputs(bundle, GRITS_SCHEMA, GRITS_POLICY)
    ids = validate_tables(annotation)
    require(ids, "empty GT scope")
    require(ids == validate_tables(predictions), "table coverage differs")
    completed = protocol.get("completed_table_ids", [])
    require(len(completed) == len(ids) and set(completed) == ids,
            "completed table scope differs")
    grits, source = load_grits()
    predicted = {row["table_id"]: row for row in predictions["tables"]}
    rows = []
    for index, ground_truth in enumerate(annotation["tables"], 1):
        actual = predicted[ground_truth["table_id"]]
        require(actual["coordinate_space"] == ground_truth["coordinate_space"],
                "GT/prediction coordinate spaces differ")
        true_cells, pred_cells = ground_truth["cells"], actual["cells"]
        top = grits.grits_top(
            np.array(grits.cells_to_relspan_grid(true_cells)),
            np.array(grits.cells_to_relspan_grid(pred_cells)),
        )
        con = grits.grits_con(
            np.array(grits.cells_to_grid(true_cells, "cell_text"), dtype=object),
            np.array(grits.cells_to_grid(pred_cells, "cell_text"), dtype=object),
        )
        loc = grits.grits_loc(
            np.array(grits.cells_to_grid(true_cells, "bbox")),
            np.array(grits.cells_to_grid(pred_cells, "bbox")),
        )
        rows.append(dict(
            table_id=ground_truth["table_id"],
            GriTS_Top=float(top[0]), GriTS_Con=float(con[0]), GriTS_Loc=float(loc[0]),
        ))
        if index == 1 or index % 100 == 0 or index == len(ids):
            write(Path(bundle) / "progress.json", dict(state="running",completed_tables=index,total_tables=len(ids)))
            print(f"GriTS {index}/{len(ids)}", flush=True)
    metrics = {
        key: float(np.mean([row[key] for row in rows]))
        for key in GRITS_POLICY["metrics"]
    }
    result = result_header(
        protocol, metrics,
        dict(name="microsoft/table-transformer grits", source_sha256=file_hash(source),
             numpy=np.__version__, compatibility="Rect inputs converted to tuples"),
        dict(tables=len(ids)),
    )
    write(Path(bundle) / "per_table_metrics.json", rows)
    write(Path(bundle) / "metrics.json", result)
    write(Path(bundle) / "progress.json", dict(state="completed",completed_tables=len(ids),total_tables=len(ids)))
    return result

SCORERS = {COCO_SCHEMA: coco_evaluate, LVIS_SCHEMA: lvis_evaluate}

def resolve_input(path):
    return (path if path.is_absolute() else HERE / path).resolve()

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--bundle', type=Path, help='One standardized scoring bundle')
    source.add_argument('--run-dir', type=Path, help='One completed run containing inference_output.json')
    args = parser.parse_args()
    if args.bundle:
        bundles = [resolve_input(args.bundle)]
        output = None
    else:
        root = resolve_input(args.run_dir)
        manifest = read(root / 'inference_output.json')
        if manifest.get('state') != 'completed' or manifest.get('task') != 'detection':
            raise SystemExit('Inference output is incomplete or belongs to another task')
        if manifest.get('scoring_ready') is False:
            raise SystemExit(manifest.get('reason', 'No validated scoring protocol for this output'))
        bundles = [(root / p).resolve() for p in manifest['bundles']]
        if any(not p.is_relative_to(root) for p in bundles):
            raise SystemExit('Bundle path escapes its run directory')
        output = root / 'evaluation_summary.json'
    if not bundles or len(bundles) != len(set(bundles)):
        raise SystemExit('No unique scoring bundles provided')
    results = []
    for bundle in bundles:
        protocol = read(bundle / 'protocol.json')
        scorer = SCORERS.get(protocol.get('schema'))
        if not scorer:
            raise SystemExit('Unsupported scoring protocol: ' + str(protocol.get('schema')))
        metrics = scorer(bundle)
        results.append({'bundle': str(bundle), 'metrics': metrics})
    if output:
        write(output, {'state': 'completed', 'task': 'detection', 'results': results})
    print(json.dumps({'state': 'completed', 'task': 'detection', 'scored_bundles': len(results)}))

if __name__ == '__main__':
    main()
