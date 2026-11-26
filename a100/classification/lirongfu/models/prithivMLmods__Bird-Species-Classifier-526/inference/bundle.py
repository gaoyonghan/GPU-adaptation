"""Per-model output conversion; all metrics are computed by the shared scorer.

Original shared images stay in project/dataset. Embedded images are decoded
once to dataset/_decoded_images. Prediction/label JSON is saved beside the original dataset in
<dataset>_<model>_imagenet/runs/<run-id>/. Final model metrics stay in result/. All stored image paths are relative; no soft links.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import io
import json
import math
import os
from pathlib import Path
import shutil
import tempfile
import uuid

MODEL_DIR = Path(__file__).resolve().parent.parent
PROJECT = MODEL_DIR.parents[3]
_spec = importlib.util.spec_from_file_location('_bundle_paths', Path(__file__).with_name('paths.py'))
paths = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(paths)
ROOT = paths.DATASET_ROOT


def load_evaluator(project=PROJECT):
    path = Path(project)/'classification'/'evaluate.py'
    spec = importlib.util.spec_from_file_location('_shared_classification_evaluator', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix='.' + path.name, dir=path.parent)
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as stream:
            json.dump(value, stream, ensure_ascii=False, allow_nan=False, separators=(',', ':'))
            stream.write('\n'); stream.flush(); os.fsync(stream.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp): os.unlink(tmp)


def name(value):
    if not isinstance(value, str) or not value or Path(value).name != value or value in ('.', '..'):
        raise ValueError('Invalid model name')
    return value


def typed_label(value):
    if type(value) not in (int, str):
        raise ValueError('Class labels must be integers or strings')
    return type(value).__name__, value


def relative(path, root):
    return Path(os.path.relpath(path, root)).as_posix()



def dataset_name(specification):
    names = set()
    for value in specification.get('dataset_links', {}).values():
        if not isinstance(value, str) or not value.startswith('../../Dataset/'):
            raise ValueError('Unexpected original dataset reference')
        names.add(name(value[len('../../Dataset/'):].split('/')[0]))
    if len(names) != 1:
        raise ValueError('Exactly one original dataset directory is required per model')
    return next(iter(names))


def package_root(model, dataset):
    return paths.DATASET_ROOT / (name(dataset) + '_' + name(model) + '_imagenet')


def validate_bundle_location(bundle, model, dataset):
    expected = package_root(model, dataset).resolve()
    if bundle.parent.name != 'runs' or bundle.parent.parent != expected:
        raise ValueError('Bundle is outside the dataset/model-specific evaluation directory')


class BundleWriter:
    def __init__(self, model_dir, specification, labels, *, sample_limit=0):
        self.model_dir = Path(model_dir).resolve()
        self.project = self.model_dir.parents[3]
        self.root = paths.DATASET_ROOT
        self.model = name(self.model_dir.name)
        run_id = name(os.environ.get('CLASSIFICATION_RUN_ID') or uuid.uuid4().hex)
        self.dataset = dataset_name(specification)
        self.package_root = package_root(self.model, self.dataset)
        self.result_run = paths.RESULT_DIR/'runs'/run_id
        self.directory = self.package_root/'runs'/run_id
        self.directory.mkdir(parents=True, exist_ok=True)
        if any((self.directory/p).exists() for p in ('protocol.json','annotations.json','predictions.json')):
            raise FileExistsError('This run already contains predictions; refusing to reuse it')
        self.evaluator = load_evaluator(self.project)
        self.spec = specification
        self.limit = sample_limit
        self.options = specification.get('options', {})
        self.mapping = self.options.get('pred_label_map') or {}
        self.ignored = self.options.get('ignore_pred_label')
        self.categories, self.category_ids = [], {}
        self.samples, self.predictions, self.excluded = [], [], []
        self.seen = set()
        self.input_count = 0
        self.labels = labels
        for label, title in labels.items():
            mapped = self.map_label(label)
            if not self.is_ignored(mapped): self.category(mapped, str(title))
        self.declared_label_count = len(labels)
        for row in specification.get('data_spec', {}).get('rows', []):
            if 'true_id' in row: self.category(row['true_id'])

    def map_label(self, value):
        typed_label(value)
        return self.mapping.get(value, self.mapping.get(str(value), value))

    def is_ignored(self, value):
        return self.ignored is not None and typed_label(value) == typed_label(self.ignored)

    def category(self, value, title=None):
        key = typed_label(value)
        if key not in self.category_ids:
            cid = len(self.categories)
            self.category_ids[key] = cid
            self.categories.append({'id': cid, 'name': title or str(value), 'original_label': value})
        return self.category_ids[key]

    def media(self, image, sid, category):
        from PIL import Image
        source = image.info.get('_coco_source_path') if isinstance(image, Image.Image) else None
        if isinstance(image, (str, os.PathLike)): source = image
        if source is not None:
            source = Path(source).resolve(strict=True)
            if not source.is_relative_to(paths.DATASET_ROOT):
                raise ValueError('Native image must belong to the original Dataset')
            suffix = source.suffix.lower()
        else:
            if not isinstance(image, Image.Image):
                raise TypeError('Expected a PIL image or native image path')
            if image.mode not in ('RGB','RGBA','L','LA','I','I;16'): image = image.convert('RGB')
            buffer = io.BytesIO(); image.save(buffer, format='PNG')
            content = buffer.getvalue()
            digest = hashlib.sha256(content).hexdigest()
            source = self.root/'_decoded_images'/digest[:2]/(digest+'.png')
            source.parent.mkdir(parents=True, exist_ok=True)
            if not source.exists():
                fd, temporary = tempfile.mkstemp(prefix='.image-', dir=source.parent)
                try:
                    with os.fdopen(fd, 'wb') as stream: stream.write(content)
                    os.replace(temporary, source)
                finally:
                    if os.path.exists(temporary): os.unlink(temporary)
            elif file_hash(source) != digest:
                raise ValueError('Shared image checksum mismatch')
            suffix = '.png'
        # ImageNet-style image/label manifests reference shared images directly.
        # No symlinks or per-model dataset copies are needed.
        ref = relative(source, self.directory)
        return ref, ref

    def add(self, sample_id, prediction, truth, image):
        key = typed_label(sample_id)
        sid = json.dumps(key, ensure_ascii=False, separators=(',', ':'))
        if sid in self.seen: raise ValueError('Duplicate sample ID')
        self.seen.add(sid); self.input_count += 1
        # Declare the full model output space, not just observed winners.
        if len(self.labels) != self.declared_label_count:
            for label, title in self.labels.items():
                mapped = self.map_label(label)
                if not self.is_ignored(mapped): self.category(mapped, str(title))
            self.declared_label_count = len(self.labels)
        mapped = self.map_label(prediction['pred_id'])
        if self.is_ignored(mapped):
            self.excluded.append({'sample_id': sid, 'prediction': mapped, 'reason': 'original_ignore_pred_label'})
            return
        gt = self.category(truth)
        predicted = self.category(mapped)
        ranking = prediction.get('topk_ids', [prediction['pred_id']])
        if not ranking or ranking[0] != prediction['pred_id']:
            raise ValueError('Native ranking must start with the existing prediction')
        ranked, used = [], set()
        for label in ranking:
            mapped_label = self.map_label(label)
            if self.is_ignored(mapped_label): continue
            cid = self.category(mapped_label)
            if cid not in used: ranked.append(cid); used.add(cid)
        if not ranked or ranked[0] != predicted:
            raise ValueError('Label mapping changed the top-1 decision')
        file, source = self.media(image, sid, gt)
        self.samples.append({'sample_id': sid, 'label': gt, 'image': file, 'source': source})
        row = {'sample_id': sid, 'prediction': predicted, 'topk': ranked}
        scores = prediction.get('topk_scores')
        if scores is not None and len(scores) == len(ranked): row['scores'] = [float(v) for v in scores]
        self.predictions.append(row)

    def finish(self, expected):
        if self.input_count != expected or not self.samples:
            raise ValueError(f'Incomplete or empty evaluation scope: {self.input_count}/{expected}')
        annotation = {'categories': self.categories, 'samples': self.samples}
        self.evaluator.validate_annotation(annotation)
        self.evaluator.validate_predictions(self.predictions, {c['id'] for c in self.categories})
        write_json(self.directory/'annotations.json', annotation)
        write_json(self.directory/'predictions.json', self.predictions)
        if self.excluded: write_json(self.directory/'excluded.json', self.excluded)
        datasets = [self.dataset]
        protocol = {'schema': self.evaluator.SCHEMA, 'evaluation': self.evaluator.POLICY,
            'scoring_ready': True, 'model': self.model, 'dataset': datasets,
            'completed_sample_ids': [r['sample_id'] for r in self.samples],
            'prediction_count': len(self.predictions), 'inference_sample_count': self.input_count,
            'sample_limit': self.limit,
            'artifact_sha256': {p: file_hash(self.directory/p) for p in ['annotations.json','predictions.json']},
            'qualification': {'kind': 'original_configured_dataset', 'data_kind': self.spec.get('data_spec', {}).get('kind'),
                              'pred_label_map': self.mapping, 'ignore_pred_label': self.ignored,
                              'excluded_samples': len(self.excluded), 'image_layout': 'shared images with ImageNet-style image/label manifests'},
            'native_execution': ['existing native top1 retained; available native class rankings exported'],
            'limitations': ['Top-5 is unavailable when native output provides only a final label.']}
        write_json(self.directory/'protocol.json', protocol)
        write_json(self.directory/'inference_output.json', {
            'state':'completed', 'task':'classification', 'model':self.model,
            'bundles':['.'], 'input_scope':{'inference_images':self.input_count,
                'evaluated_images':len(self.predictions),'max_images':self.limit}})
        write_json(self.result_run/'evaluation_bundle.json', {
            'model':self.model, 'dataset':self.dataset, 'complete':True,
            'bundle':relative(self.directory, self.result_run), 'schema':self.evaluator.SCHEMA})
        return self.directory


def latest_bundle(model_dir):
    model_dir = Path(model_dir).resolve()
    parent = paths.RESULT_DIR
    marker = json.loads((parent/'latest.json').read_text())
    value = marker['bundle']
    if not isinstance(value, str) or Path(value).is_absolute(): raise ValueError('Bundle must be relative')
    bundle = (parent/value).resolve(strict=True)
    validate_bundle_location(bundle, model_dir.name, marker['dataset'])
    return bundle


def validate_media(bundle, annotation):
    bundle = Path(bundle).resolve()
    root = paths.DATASET_ROOT
    for row in annotation['samples']:
        value = row['image']
        if Path(value).is_absolute(): raise ValueError('Expected relative image path')
        file = (bundle/value).resolve(strict=True)
        if not file.is_file() or not file.is_relative_to(root.resolve()):
            raise ValueError('Missing or external exported image')


def score_bundle(model_dir, bundle=None, output=None, *, publish=False):
    model_dir = Path(model_dir).resolve()
    bundle = latest_bundle(model_dir) if bundle is None else Path(bundle).resolve(strict=True)
    root = paths.RESULT_DIR
    if not bundle.is_relative_to(paths.DATASET_ROOT):
        raise ValueError('Bundle must be in the shared dataset directory')
    evaluator = load_evaluator(model_dir.parents[3])
    annotation, _, protocol = evaluator.load_bundle(bundle)
    if protocol.get('model') != model_dir.name: raise ValueError('Wrong model in protocol')
    datasets = protocol.get('dataset')
    if not isinstance(datasets, list) or len(datasets) != 1: raise ValueError('Invalid dataset identity')
    validate_bundle_location(bundle, model_dir.name, datasets[0])
    target = (Path(output) if output is not None else paths.RESULT_DIR/'results.json').resolve()
    if protocol.get('sample_limit') and target == paths.RESULT_DIR/'results.json':
        raise ValueError('Limited-sample evaluation requires a separate result file')
    if not target.is_relative_to(paths.RESULT_DIR) or target.is_relative_to(paths.RESULT_DIR/'runs'):
        raise ValueError('Results must remain in this model result directory, outside input bundles')
    validate_media(bundle, annotation)
    result = evaluator.evaluate(bundle)
    write_json(target, result)
    if publish and not protocol.get('sample_limit') and target == paths.RESULT_DIR/'results.json':
        parent = root
        base = package_root(model_dir.name, datasets[0])
        write_json(base/'inference_output.json', {
            'state':'completed', 'task':'classification', 'model':model_dir.name,
            'bundles':[relative(bundle,base)], 'input_scope':{
                'inference_images':protocol['inference_sample_count'],
                'evaluated_images':protocol['prediction_count'], 'max_images':0}})
        write_json(parent/'latest.json', {'model':model_dir.name, 'dataset':datasets[0],
                                        'bundle':relative(bundle,parent),
                                        'schema':evaluator.SCHEMA, 'complete':True})
    return result

