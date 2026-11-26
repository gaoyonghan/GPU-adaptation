"""Portable COCO-style evaluation bundles. No model loading or inference.

Standard images/annotations/categories and RLE masks are supplemented with
evaluation_units for prompts, video objects, volumes and pairwise preferences.
Media references are relative to the project root; shard paths are relative to their run directory.
"""
from __future__ import annotations
import hashlib
import io
import json
import math
import os
from pathlib import Path
import tempfile
import uuid

import importlib.util
import time
_paths_spec = importlib.util.spec_from_file_location('_mask_bundle_paths', Path(__file__).with_name('paths.py'))
PATHS = importlib.util.module_from_spec(_paths_spec)
_paths_spec.loader.exec_module(PATHS)
SCHEMA = 'model-eval/mask-coco/v1' 
SHARD_SIZE = 256


def _json(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False, separators=(',', ':')).encode('utf-8')


def _sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def _write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = _json(value)
    fd, temporary = tempfile.mkstemp(prefix='.' + path.name, dir=path.parent)
    try:
        with os.fdopen(fd, 'wb') as stream:
            stream.write(data); stream.flush(); os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return hashlib.sha256(data).hexdigest()


def _relative(root, path):
    value = os.path.relpath(path, root)
    if Path(value).is_absolute():
        raise ValueError('A persisted path must be relative')
    return Path(value).as_posix()




def _name(value):
    if not value or Path(value).name != value or value in ('.', '..'):
        raise ValueError('Invalid model/dataset directory name')
    return value


class MediaStore:
    def __init__(self, root):
        self.root = Path(root)
        self.project = PATHS.PROJECT_ROOT
        self.native = PATHS.DATASET_ROOT.resolve()
        self.images = {}
        self.references = {}

    def source(self, path):
        key = os.fspath(path)
        if key not in self.references:
            resolved = Path(path).resolve(strict=True)
            if not resolved.is_relative_to(self.native): raise ValueError('Original media is outside the configured dataset root')
            self.references[key] = _relative(self.project, resolved)
        return self.references[key]

    def image(self, value):
        from PIL import Image
        if isinstance(value, (str, os.PathLike)):
            key = os.fspath(value)
            if key not in self.images:
                ref = self.source(value)
                with Image.open(value) as im:
                    self.images[key] = {'file_name': ref, 'width': im.width, 'height': im.height}
            return dict(self.images[key])
        if isinstance(value, dict):
            return self.image(value.get('bytes') if value.get('bytes') is not None else value.get('path'))
        if isinstance(value, (bytes, bytearray, memoryview)):
            with Image.open(io.BytesIO(bytes(value))) as im: image = im.copy()
        elif isinstance(value, Image.Image): image = value
        else:
            import numpy as np
            image = Image.fromarray(np.asarray(value))
        if image.mode not in ('L', 'LA', 'RGB', 'RGBA', 'I', 'I;16'): image = image.convert('RGB')
        stream = io.BytesIO(); image.save(stream, format='PNG'); content = stream.getvalue()
        digest = hashlib.sha256(content).hexdigest()
        file = self.native / '_decoded_images' / (digest + '.png')
        file.parent.mkdir(parents=True, exist_ok=True)
        if not file.exists():
            fd, tmp = tempfile.mkstemp(prefix='.image-', dir=file.parent)
            try:
                with os.fdopen(fd, 'wb') as handle: handle.write(content)
                os.replace(tmp, file)
            finally:
                if os.path.exists(tmp): os.unlink(tmp)
        elif _sha(file) != digest: raise ValueError('Stored image checksum mismatch')
        return {'file_name': _relative(self.project, file), 'width': image.width, 'height': image.height}


def _rle(mask):
    import numpy as np
    from pycocotools import mask as api
    if isinstance(mask, dict):
        mask = dict(mask)
        if isinstance(mask['counts'], list):
            mask = api.frPyObjects(mask, *mask['size'])
        if isinstance(mask['counts'], bytes): mask['counts'] = mask['counts'].decode('ascii')
        mask['size'] = [int(v) for v in mask['size']]
        return mask
    array = np.asarray(mask, dtype=bool)
    if array.ndim != 2: raise ValueError('COCO RLE requires a 2D slice')
    value = api.encode(np.asfortranarray(array, dtype=np.uint8))
    value['counts'] = value['counts'].decode('ascii')
    return value


def _annotation(mask, annotation_id, image_id, unit_id):
    from pycocotools import mask as api
    rle = _rle(mask)
    encoded = dict(rle, counts=rle['counts'].encode('ascii'))
    return {'id': annotation_id, 'image_id': image_id, 'category_id': 1,
            'segmentation': rle, 'area': int(api.area(encoded)),
            'bbox': [float(x) for x in api.toBbox(encoded)], 'iscrowd': 0, 'eval_id': unit_id}


class BundleWriter:
    def __init__(self, model_dir, task, protocol=None):
        self.model_dir = Path(model_dir).resolve()
        if self.model_dir != PATHS.MODEL_DIR: raise ValueError('This exporter belongs to another model')
        if task != 'mask_generation': raise ValueError('Only mask evaluation is supported')
        self.model = _name(self.model_dir.name)
        self.task = task
        self.protocol = dict(protocol or {})
        dataset = _name(self.protocol['dataset'])
        self.package = PATHS.DATASET_ROOT / (dataset + '_' + self.model + '_coco')
        self.run = _name(os.environ.get('MASK_RUN_ID') or uuid.uuid4().hex)
        self.root = self.package / 'runs' / self.run
        self.directory = self.root / 'predictions'
        self.directory.mkdir(parents=True, exist_ok=False)
        self.media = MediaStore(self.root)
        self.categories = []
        self.images, self.annotations, self.units, self.predictions = [], [], [], []
        self.shards, self.count, self.image_id, self.annotation_id = [], 0, 0, 0
        self.seen = set()
        self.last_progress = time.monotonic()
        print(f'COCO export: {_relative(PATHS.PROJECT_ROOT, self.root)}', flush=True)

    def _unit(self, record, kind):
        sid = record['sample_id']
        if isinstance(sid, bool) or not isinstance(sid, (str, int)):
            raise ValueError('Invalid sample_id')
        identity = (type(sid).__name__, sid)
        if identity in self.seen: raise ValueError(f'Duplicate sample_id: {sid}')
        self.seen.add(identity); self.count += 1
        return {'id': self.count, 'sample_id': sid, 'kind': kind,
                'group': str(record.get('group', sid)), 'metadata': record.get('metadata', {}), 'image_ids': []}


    def mask(self, record):
        import numpy as np
        p, g = record['prediction'], record['truth']
        is_volume = not isinstance(g, (dict, list)) and np.asarray(g).ndim == 3
        kind = 'volume' if is_volume else 'instance_set' if isinstance(g, list) else 'mask_pair'
        unit = self._unit(record, kind)
        self.categories = [{'id': 1, 'name': 'foreground'}]
        if is_volume:
            if np.asarray(p).shape != np.asarray(g).shape: raise ValueError('Volume shape mismatch')
            unit['volume_shape'] = list(g.shape)
            frames = [(z, [p[z]], [g[z]]) for z in range(len(g))]
            source = record['volume_source']
            unit['metadata'] = {**unit['metadata'], 'volume_file': self.media.source(source)}
            import SimpleITK as sitk
            original = sitk.GetArrayFromImage(sitk.ReadImage(str(source))).astype(np.float32)
        else:
            frames = [(None, p if isinstance(p, list) else [p], g if isinstance(g, list) else [g])]
        for z, preds, truths in frames:
            self.image_id += 1
            if is_volume:
                # Lossless native CT remains referenced; PNG is only the exact
                # windowed RGB-equivalent image used by this model.
                pixels = ((np.clip(original[z], -160, 240) + 160) / 400 * 255).astype(np.uint8)
                media = self.media.image(pixels)
            else:
                media = self.media.image(record['image'])
            image = {'id': self.image_id, **media}
            if z is not None: image['slice_index'] = z
            self.images.append(image); unit['image_ids'].append(self.image_id)
            for mask in truths:
                self.annotation_id += 1
                ann = _annotation(mask, self.annotation_id, self.image_id, unit['id'])
                if ann['segmentation']['size'] != [media['height'], media['width']]: raise ValueError('GT image grid mismatch')
                self.annotations.append(ann)
            for mask in preds:
                ann = _annotation(mask, len(self.predictions) + 1, self.image_id, unit['id'])
                if ann['segmentation']['size'] != [media['height'], media['width']]: raise ValueError('Prediction image grid mismatch')
                ann.pop('id'); self.predictions.append(ann)
        self.units.append(unit)
        if len(self.units) >= SHARD_SIZE or is_volume: self.flush()

    def flush(self):
        if not self.units: return
        gt = {'images': self.images, 'annotations': self.annotations,
              'categories': self.categories, 'evaluation_units': self.units}
        index = len(self.shards)
        gtfile = self.root / 'annotations' / f'annotations-{index:06d}.json'
        predfile = self.directory / f'predictions-{index:06d}.json'
        gh, ph = _write(gtfile, gt), _write(predfile, self.predictions)
        self.shards.append({'annotations': _relative(self.root, gtfile), 'annotations_sha256': gh,
                            'predictions': _relative(self.root, predfile), 'predictions_sha256': ph,
                            'sample_count': len(self.units)})
        self.images, self.annotations, self.units, self.predictions = [], [], [], []
        if time.monotonic() - self.last_progress >= 5:
            print(f'COCO export: {self.count} units, {len(self.shards)} shards', flush=True)
            self.last_progress = time.monotonic()

    def finish(self):
        if not self.count: raise ValueError('Cannot publish an empty evaluation package')
        self.flush()
        protocol = {'schema': SCHEMA, 'model': self.model, 'task': self.task, 'complete': True,
                    'sample_count': self.count, 'shards': self.shards, 'evaluation': self.protocol,
                    'path_base': 'project_root', 'media_policy': 'shared-relative-native-or-lossless-png',
                    'model_directory': _relative(PATHS.PROJECT_ROOT, self.model_dir),
                    'dataset_root': _relative(PATHS.PROJECT_ROOT, PATHS.DATASET_ROOT), 'run_id': self.run}
        _write(self.root / 'protocol.json', protocol)
        _write(self.root / 'inference_output.json', {'state': 'completed', 'task': self.task, 'model': self.model,
                                                   'run_directory': '.'})
        model_run = PATHS.RESULT_DIR / 'runs' / self.run
        _write(model_run / 'evaluation_bundle.json', {'bundle': _relative(model_run, self.root),
                                                    'schema': SCHEMA, 'model': self.model})
        print(f'COCO export complete: {self.count} units', flush=True)
        return self.root / 'protocol.json'

