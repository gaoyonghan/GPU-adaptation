import contextlib
import fcntl
import hashlib
import io
import json
import os
from pathlib import Path
import pickle
import tempfile
import zipfile

HERE = Path(__file__).resolve().parent
WORKSPACE = Path('__MODEL_ROOT__/resources/runtime/assets/workspace')
SOURCE = Path('__MODEL_ROOT__/resources/runtime/assets/workspace/171-200/inference_code')


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent)
    try:
        with os.fdopen(fd, 'w') as f:
            json.dump(value, f, indent=2, ensure_ascii=False, allow_nan=False)
            f.write('\n')
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


@contextlib.contextmanager
def locked(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a') as f:
        fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield


class DataOnlyUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        raise ValueError(f'Dataset pickle contains executable object: {module}.{name}')


class Images:
    """Read source image bytes in their original containers, without re-encoding."""
    def __init__(self):
        self.archives = {}
        self.parquet_key = None
        self.parquet_rows = None
        self.fer = None
        self.idx = None

    def raw(self, row):
        if 'path' in row:
            return Path(row['path']).read_bytes()
        if 'archive' in row:
            key = (row['archive'], row.get('inner_archive'))
            if key not in self.archives:
                outer = zipfile.ZipFile(row['archive'])
                if row.get('inner_archive'):
                    data = outer.read(row['inner_archive'])
                    outer.close()
                    outer = zipfile.ZipFile(io.BytesIO(data))
                self.archives[key] = outer
            data = self.archives[key].read(row['member'])
            return data
        if 'parquet' in row:
            import pyarrow.parquet as pq
            key = (row['parquet'], row['group'], row['image_column'])
            if key != self.parquet_key:
                self.parquet_rows = pq.ParquetFile(key[0]).read_row_group(key[1], columns=[key[2]]).column(0).to_pylist()
                self.parquet_key = key
            image = self.parquet_rows[row['offset']]
            if not image.get('bytes'):
                raise ValueError('Parquet image bytes missing; no implicit path or network fallback')
            return image['bytes']
        if 'fer_pickle' in row:
            if self.fer is None:
                with open(row['fer_pickle'], 'rb') as f:
                    self.fer = DataOnlyUnpickler(f).load()
            return self.fer[row['offset']]['img_bytes']
        raise ValueError(f'No encoded image for row {row["id"]}')

    def image(self, row):
        from PIL import Image
        if 'fashion_idx' in row:
            import gzip
            import struct
            if self.idx is None:
                self.idx = gzip.open(row['fashion_idx'], 'rb').read()
                assert struct.unpack('>IIII', self.idx[:16]) == (2051, 10000, 28, 28)
            start = 16 + row['offset'] * 784
            return Image.frombytes('L', (28, 28), self.idx[start:start + 784]).convert('RGB')
        with Image.open(io.BytesIO(self.raw(row))) as im:
            return im.convert('RGB')

    def close(self):
        for archive in self.archives.values():
            archive.close()


def read_index(spec):
    path = Path(spec['index'])
    if digest(path) != spec['index_sha256']:
        raise ValueError('Frozen dataset index changed')
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    if len(rows) != spec['images'] or len({r['id'] for r in rows}) != len(rows):
        raise ValueError('Dataset coverage or sample IDs invalid')
    return rows


def smoke_rows(rows, count=10):
    # Prefer different target classes, then fill deterministically; preserve source order.
    chosen, seen = [], set()
    for i, row in enumerate(rows):
        if row.get('target') not in seen:
            seen.add(row.get('target'))
            chosen.append(i)
        if len(chosen) == count:
            break
    chosen += [i for i in range(len(rows)) if i not in chosen][:max(0, count-len(chosen))]
    return [rows[i] for i in sorted(chosen)]
