"""Readers for existing annotated replacement datasets and native NVIDIA weights."""
import importlib
import json
import sys
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

from PIL import Image


def prepare(dataset, root, cache, core):
    kind = dataset['kind']
    if kind == 'xml_detection':
        return xml_detection(dataset, root, core)
    if kind == 'wholebody_face_person':
        source = json.loads((root / dataset['annotation']).read_text())
        archive = zipfile.ZipFile(root / dataset['image_archive'])
        _, basenames = core.zip_member_index(archive)
        annotations = []
        for original in source['annotations']:
            for name, cat, box, crowd in [
                ('person', 1, original['bbox'], original.get('iscrowd', 0)),
                ('face', 2, original['face_box'] if original['face_valid'] else original['bbox'],
                 0 if original['face_valid'] else 1),
            ]:
                if box[2] <= 0 or box[3] <= 0:
                    continue
                annotations.append({'id': len(annotations)+1, 'image_id': original['image_id'],
                                    'category_id': cat, 'bbox': box, 'area': box[2]*box[3], 'iscrowd': crowd})
        annotation = {'images': source['images'], 'annotations': annotations,
                      'categories': [{'id': 1, 'name': 'person'}, {'id': 2, 'name': 'face'}]}
        def read_image(info):
            return core.image_from_zip(archive, basenames[Path(info['file_name']).name])
        return core.PreparedDataset(core.ensure_coco_fields(annotation), read_image, archive.close)
    if kind == 'omni_layout':
        pages = json.loads((root / dataset['annotation']).read_text())
        mapping = dataset['category_mapping']
        names = list(dict.fromkeys(mapping.values()))
        ids = {n: i+1 for i, n in enumerate(names)}
        images, annotations = [], []
        for page in pages:
            info = page['page_info']
            image_id = len(images)+1
            images.append({'id': image_id, 'file_name': info['image_path'],
                           'width': info['width'], 'height': info['height']})
            for element in page['layout_dets']:
                if element['category_type'] not in mapping:
                    if element['category_type'].endswith('_mask') or element['category_type'] == 'abandon':
                        poly = element['poly']
                        x1, y1, x2, y2 = min(poly[::2]), min(poly[1::2]), max(poly[::2]), max(poly[1::2])
                        for cat_id in ids.values():
                            annotations.append({'id': len(annotations)+1, 'image_id': image_id,
                                                'category_id': cat_id, 'bbox': [x1,y1,x2-x1,y2-y1],
                                                'area': (x2-x1)*(y2-y1), 'iscrowd': 1})
                    continue
                poly = element['poly']
                x1, y1, x2, y2 = min(poly[::2]), min(poly[1::2]), max(poly[::2]), max(poly[1::2])
                annotations.append({'id': len(annotations)+1, 'image_id': image_id,
                                    'category_id': ids[mapping[element['category_type']]],
                                    'bbox': [x1, y1, x2-x1, y2-y1], 'area': (x2-x1)*(y2-y1),
                                    'iscrowd': int(element.get('ignore', False)),
                                    'segmentation': [poly], 'reading_order': element.get('order')})
        def read_image(info):
            with Image.open(root / dataset['image_path'] / Path(info['file_name']).name) as opened:
                return opened.convert('RGB')
        return core.PreparedDataset(core.ensure_coco_fields({'images': images, 'annotations': annotations,
                                    'categories': [{'id': ids[n], 'name': n} for n in names]}), read_image, lambda: None)
    return core.prepare_dataset(dataset, root, cache)


def xml_detection(dataset, root, core):
    archive = zipfile.ZipFile(root / dataset['archive']) if 'archive' in dataset else None
    if archive:
        members = sorted(n for n in archive.namelist() if n.startswith(dataset['prefix']) and n.endswith('.xml'))
        trees = [(n, ET.fromstring(archive.read(n))) for n in members]
        image_members = [n for n in archive.namelist() if n.startswith(dataset['prefix'])
                         and Path(n).suffix.lower() in core.IMAGE_SUFFIXES]
        image_index = {Path(n).name: n for n in image_members}
        assert len(image_index) == len(image_members), 'Duplicate images within selected split'
    else:
        base = root / dataset['path']
        trees = [(str(p), ET.parse(p).getroot()) for p in sorted(base.rglob('*.xml'))]
        image_index = {p.name: p for p in base.rglob('*') if p.suffix.lower() in core.IMAGE_SUFFIXES}
    names = list(dataset['class_names'])
    cat_ids = {n: i+1 for i, n in enumerate(names)}
    images, annotations, payloads = [], [], {}
    for xml_path, tree in trees:
        filename = tree.get('filename') or tree.findtext('filename')
        if filename is None:
            raise ValueError(f'XML has no filename: {xml_path}')
        filename = Path(filename).name
        image_path = image_index[filename]
        if archive:
            opened = core.image_from_zip(archive, image_path)
        else:
            opened = Image.open(image_path)
        width, height = opened.size
        opened.close()
        image_id = len(images)+1
        images.append({'id': image_id, 'file_name': filename, 'width': width, 'height': height})
        payloads[image_id] = image_path
        if dataset.get('format') == 'ctdar':
            for table in tree.findall('table'):
                points = [list(map(float, x.split(','))) for x in table.find('Coords').get('points').split()]
                xs, ys = zip(*points)
                box = [min(xs), min(ys), max(xs)-min(xs), max(ys)-min(ys)]
                annotations.append({'id': len(annotations)+1, 'image_id': image_id, 'category_id': 1,
                                    'bbox': box, 'area': box[2]*box[3], 'iscrowd': 0})
        else:
            for obj in tree.findall('object'):
                name = obj.findtext('name').strip()
                if name not in cat_ids:
                    continue
                b = obj.find('bndbox')
                x1, y1, x2, y2 = [float(b.findtext(k)) for k in ('xmin', 'ymin', 'xmax', 'ymax')]
                x1, y1 = max(0, x1-1), max(0, y1-1)
                x2, y2 = min(width, x2), min(height, y2)
                if x2 <= x1 or y2 <= y1:
                    raise ValueError(f'Invalid annotated box: {xml_path}')
                annotations.append({'id': len(annotations)+1, 'image_id': image_id, 'category_id': cat_ids[name],
                                    'bbox': [x1, y1, x2-x1, y2-y1], 'area': (x2-x1)*(y2-y1),
                                    'iscrowd': int(obj.findtext('difficult', '0'))})
    def read_image(info):
        if archive:
            return core.image_from_zip(archive, payloads[info['id']])
        with Image.open(payloads[info['id']]) as opened:
            return opened.convert('RGB')
    return core.PreparedDataset(core.ensure_coco_fields({'images': images, 'annotations': annotations,
                                'categories': [{'id': cat_ids[n], 'name': n} for n in names]}),
                                read_image, archive.close if archive else lambda: None)


def load_nemotron(weight, task, device):
    import numpy as np
    import torch
    root = weight.parent.parent
    package = task['package']
    sys.path.insert(0, str(root))
    config_module = importlib.import_module(package + '.' + task['config_module'])
    model_module = importlib.import_module(package + '.model')
    config = config_module.Exp()
    config.device = device
    model = config.get_model()
    state = torch.load(weight, map_location='cpu', weights_only=False)
    model.load_state_dict(state['model'], strict=True)
    model = model_module.YoloXWrapper(model, config).eval().to(device)
    utils = importlib.import_module(package + '.utils')
    names = dict(enumerate(config.labels))
    def predict(image):
        array = np.asarray(image).copy()
        with torch.inference_mode():
            result = model(model.preprocess(array), array.shape)[0]
        if task.get('element_type') == 'table':
            boxes, labels, scores = utils.postprocess_preds_table_structure(result, model.threshold, model.labels)
        elif task.get('element_type') == 'graphic':
            boxes, labels, scores = utils.postprocess_preds_graphic_element(result, model.threshold, model.labels)
        else:
            boxes, labels, scores = utils.postprocess_preds_page_element(result, model.thresholds_per_class, model.labels)
        if config.normalize_boxes:
            boxes *= np.array([image.width, image.height, image.width, image.height])
        return [(box.tolist(), float(score), int(label)) for box, score, label in
                zip(boxes, scores, labels)]
    return names, predict
