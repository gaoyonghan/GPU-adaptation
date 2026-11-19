"""Internal helpers used by this model's inference.py entrypoint."""
import hashlib
import json
import detection_core as core


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def metrics(annotation, predictions):
    import numpy as np
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    gt = COCO()
    gt.dataset = core.ensure_coco_fields(annotation)
    gt.createIndex()
    if predictions:
        dt = gt.loadRes(predictions)
    else:
        dt = COCO()
        dt.dataset = {**gt.dataset, 'annotations': []}
        dt.createIndex()
    evaluator = COCOeval(gt, dt, 'bbox')
    evaluator.evaluate()
    evaluator.accumulate()
    evaluator.summarize()
    names = ['AP', 'AP50', 'AP75', 'AP_small', 'AP_medium', 'AP_large',
             'AR1', 'AR10', 'AR100', 'AR_small', 'AR_medium', 'AR_large']
    result = {k: float(v) if v >= 0 else None for k, v in zip(names, evaluator.stats)}
    precision = evaluator.eval['precision']
    category_names = {c['id']: c['name'] for c in annotation['categories']}
    result['per_class_AP'] = {}
    for index, cat_id in enumerate(evaluator.params.catIds):
        values = precision[:, :, index, 0, -1]
        values = values[values > -1]
        result['per_class_AP'][category_names[cat_id]] = float(np.mean(values)) if values.size else None
    return result
