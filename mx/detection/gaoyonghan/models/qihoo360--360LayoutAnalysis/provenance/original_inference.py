"""Four public-replacement layout evaluations, using OpenDataLab's label merges."""
import argparse
from collections import Counter
from datetime import datetime
import hashlib
import json
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parent / '_deps'))

import numpy as np
import torch
import ultralytics
from ultralytics import YOLO
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


DATA = Path('/data2/gaoyonghan/gpu/dataset/omnidocbench_eval')
WEIGHTS = Path('/data2/gaoyonghan/gpu/model/qihoo360/360LayoutAnalysis')
SOURCE = 'https://github.com/opendatalab/PDF-Extract-Kit/blob/release/0.1.1/assets/validation/README-zh_CN.md'
LABELS = ['title', 'plain text', 'abandon', 'figure', 'figure_caption',
          'table', 'table_caption', 'isolate_formula']
MODEL_MAP = {
    'Text': 'plain text', 'Title': 'title', 'Figure': 'figure',
    'Figure caption': 'figure_caption', 'Table': 'table',
    'Table caption': 'table_caption', 'Header': 'abandon',
    'Footer': 'abandon', 'Reference': 'plain text',
    'Equation': 'isolate_formula', 'Toc': 'plain text',
}
# Omni-specific adaptation. Unmapped regions are explicitly ignored, not scored.
OMNI_MAP = {
    'title': 'title', 'text_block': 'plain text', 'figure': 'figure',
    'figure_caption': 'figure_caption', 'table': 'table',
    'table_caption': 'table_caption', 'table_footnote': 'plain text',
    'equation_caption': 'plain text', 'reference': 'plain text',
    'equation_isolated': 'isolate_formula', 'header': 'abandon',
    'footer': 'abandon', 'page_number': 'abandon',
    'page_footnote': 'abandon', 'abandon': 'abandon',
}
SUBSETS = {'academic': ('academic_literature', 215),
           'textbook': ('colorful_textbook', 159)}


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def ground_truth(pages, task):
    names = LABELS if task == 'paper' else LABELS[:-1]
    ids = {name: i + 1 for i, name in enumerate(names)}
    mapping = dict(OMNI_MAP)
    if task == 'report':
        mapping['equation_isolated'] = 'plain text'
    images, annotations = [], []
    positive_counts, ignored_counts = Counter(), Counter()
    for image_id, page in enumerate(pages, 1):
        info = page['page_info']
        images.append({'id': image_id, 'file_name': Path(info['image_path']).name,
                       'width': info['width'], 'height': info['height']})
        for element in page['layout_dets']:
            poly = element['poly']
            x1, y1 = max(0, min(poly[::2])), max(0, min(poly[1::2]))
            x2, y2 = min(info['width'], max(poly[::2])), min(info['height'], max(poly[1::2]))
            if x2 <= x1 or y2 <= y1:
                raise ValueError(f'Invalid annotation: {info["image_path"]}')
            original = element['category_type']
            name = mapping.get(original)
            ignored = bool(element.get('ignore', False)) or name is None
            if ignored:
                ignored_counts[original] += 1
            else:
                positive_counts[name] += 1
            # A prediction in an unscored/masked region must not become an FP.
            target_ids = list(ids.values()) if ignored else [ids[name]]
            for category_id in target_ids:
                annotations.append({'id': len(annotations) + 1, 'image_id': image_id,
                                    'category_id': category_id, 'bbox': [x1, y1, x2-x1, y2-y1],
                                    'area': (x2-x1)*(y2-y1), 'iscrowd': int(ignored)})
    assert len({im['file_name'] for im in images}) == len(images)
    annotation = {'info': {}, 'images': images, 'annotations': annotations,
                  'categories': [{'id': cid, 'name': name} for name, cid in ids.items()]}
    coverage = {'positive_instances': dict(positive_counts),
                'ignored_regions_by_source_class': dict(ignored_counts),
                'scored_classes': names, 'omni_mapping': mapping}
    return annotation, ids, coverage


def evaluate(annotation, predictions):
    gt = COCO()
    gt.dataset = annotation
    gt.createIndex()
    if predictions:
        dt = gt.loadRes(predictions)
    else:
        dt = COCO()
        dt.dataset = {**annotation, 'annotations': []}
        dt.createIndex()
    evaluator = COCOeval(gt, dt, 'bbox')
    evaluator.params.maxDets = [1, 10, 100]
    evaluator.evaluate()
    evaluator.accumulate()
    evaluator.summarize()
    result = {'mAP_percent': float(evaluator.stats[0] * 100), 'per_class_AP_percent': {}}
    for i, category in enumerate(annotation['categories']):
        values = evaluator.eval['precision'][:, :, i, 0, -1]
        valid = values[values >= 0]
        result['per_class_AP_percent'][category['name']] = float(np.mean(valid)*100) if valid.size else None
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='0')
    parser.add_argument('--smoke', action='store_true', help='Only two pages per domain; not final metrics')
    args = parser.parse_args()
    pages = json.loads((DATA / 'OmniDocBench.json').read_text(encoding='utf-8'))
    subsets = {}
    for key, (source, expected) in SUBSETS.items():
        selected = [p for p in pages if p['page_info']['page_attribute']['data_source'] == source]
        assert len(selected) == expected, (source, len(selected), expected)
        for page in selected:
            assert (DATA / 'images' / Path(page['page_info']['image_path']).name).is_file()
        subsets[key] = selected[:2] if args.smoke else selected
    mode = 'smoke' if args.smoke else 'full'
    output = Path(__file__).resolve().parent.parent / 'results' / 'm41_omni_opendatalab_four'
    output = output / f'{mode}_{datetime.now():%Y%m%d_%H%M%S_%f}'
    output.mkdir(parents=True)
    print(f'Output: {output}', flush=True)
    protocol = {
        'dataset': 'OmniDocBench local 1651-page snapshot; public replacement',
        'original_opendatalab_data': '402 academic + 587 textbook pages; not used',
        'label_mapping_source': SOURCE, 'model_mapping': MODEL_MAP,
        'subsets': {key: {'data_source': SUBSETS[key][0], 'pages': len(value)} for key, value in subsets.items()},
        'book_pages_included': False, 'language_filter': None,
        'metric': 'COCO bbox AP@[0.50:0.95], maxDets=[1,10,100]',
        'evaluator': 'pycocotools.COCOeval (OpenDataLab reports mmeval.COCODetection)',
        'inference': {'imgsz': 640, 'conf': 0.001, 'iou': 0.7, 'max_det': 300, 'half': False, 'device': args.device},
        'limitation': 'Public replacement, not reproduction of OpenDataLab reported numbers. '
                      'Omni-specific mapping and all-class crowd ignores for unscored/ignored regions. '
                      'Low confidence threshold retained for AP; original complete inference settings unconfirmed.',
        'versions': {'torch': torch.__version__, 'ultralytics': ultralytics.__version__},
        'annotation_sha256': sha256(DATA / 'OmniDocBench.json'),
        'script_sha256': sha256(Path(__file__)), 'smoke_only': args.smoke,
        'weights_sha256': {task: sha256(WEIGHTS / f'{task}-8n.pt') for task in ['paper', 'report']},
    }
    save(output / 'protocol.json', protocol)
    summary = {'state': 'running', 'smoke_only': args.smoke, 'results': {}}
    save(output / 'summary.json', summary)
    for task in ['paper', 'report']:
        model = YOLO(str(WEIGHTS / f'{task}-8n.pt'))
        assert set(model.names.values()) <= set(MODEL_MAP), model.names
        for domain, selected in subsets.items():
            key = f'{task}_on_{domain}'
            dest = output / key
            dest.mkdir()
            annotation, ids, coverage = ground_truth(selected, task)
            if not args.smoke:
                assert all(coverage['positive_instances'].get(name, 0) > 0 for name in ids), coverage
            save(dest / 'ground_truth.json', annotation)
            save(dest / 'coverage.json', coverage)
            predictions = []
            for index, info in enumerate(annotation['images'], 1):
                result = model.predict(source=str(DATA / 'images' / info['file_name']),
                                       device=args.device, imgsz=640, conf=0.001, iou=0.7,
                                       max_det=300, half=False, verbose=False, save=False,
                                       project=str(dest), name='inference', exist_ok=True)[0]
                for box in result.boxes.data.cpu().tolist():
                    x1, y1, x2, y2, score, class_id = box
                    name = MODEL_MAP[model.names[int(class_id)]]
                    predictions.append({'image_id': info['id'], 'category_id': ids[name],
                                        'bbox': [x1, y1, x2-x1, y2-y1], 'score': score})
                if index % 25 == 0 or index == len(selected):
                    print(f'{key}: {index}/{len(selected)}', flush=True)
            save(dest / 'predictions.json', predictions)
            metrics = evaluate(annotation, predictions)
            metrics.update({'images': len(selected), 'scored_classes': len(ids), 'smoke_only': args.smoke})
            save(dest / 'metrics.json', metrics)
            summary['results'][key] = metrics
            save(output / 'summary.json', summary)
        del model
    summary['state'] = 'complete'
    save(output / 'summary.json', summary)
    print('\n' + ('SMOKE ONLY — not final scores' if args.smoke else 'Four public-replacement mAP scores'))
    for key, value in summary['results'].items():
        print(f'{key}: mAP={value["mAP_percent"]:.4f}% ({value["images"]} pages)')
    print(f'M41 completed: {output}', flush=True)


if __name__ == '__main__':
    main()
