"""Read the same ImageNet validation bytes from a ZIP or its extracted directory."""
import csv
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import zipfile
import zlib

VAL = 'ILSVRC/Data/CLS-LOC/val'


class ImageNetInput:
    def __init__(self, archive_path):
        path = Path(archive_path)
        if path.is_file():
            self.kind, self.source = 'zip', path
            self.archive = zipfile.ZipFile(path)
        else:
            self.kind, self.source = 'directory', path if path.is_dir() else path.parent
            self.archive = None
            if not (self.source / 'LOC_val_solution.csv').is_file() or not (self.source / VAL).is_dir():
                raise FileNotFoundError(f'ImageNet ZIP or extracted validation directory missing: {path}')
        self._infos = {}

    def _path(self, member):
        member = str(member)
        parts = PurePosixPath(member).parts
        if member.startswith('/') or '..' in parts:
            raise ValueError(f'Invalid ImageNet member: {member}')
        return self.source.joinpath(*parts)

    def read(self, member):
        return self.archive.read(member) if self.archive else self._path(member).read_bytes()

    def open(self, member):
        return self.archive.open(member) if self.archive else self._path(member).open('rb')

    def namelist(self):
        if self.archive:
            return self.archive.namelist()
        labels = [p.name for p in self.source.glob('LOC_*.csv')] + [p.name for p in self.source.glob('LOC_*.txt')]
        return sorted(labels + [f'{VAL}/{p.name}' for p in (self.source / VAL).glob('*.JPEG')])

    def getinfo(self, member):
        if self.archive:
            return self.archive.getinfo(member)
        if member not in self._infos:
            data = self.read(member)
            info = zipfile.ZipInfo(member)
            info.file_size, info.CRC = len(data), zlib.crc32(data)
            self._infos[member] = info
        return self._infos[member]

    def infolist(self):
        return self.archive.infolist() if self.archive else [self.getinfo(n) for n in self.namelist()]

    def close(self):
        if self.archive:
            self.archive.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def frozen_directory_index(archive_path, cache, expected, canonical, save, use_prepared=False):
    """Preserve the frozen ZIP index; verify its labels/bytes against the shared directory."""
    marker = cache / 'index.json'
    if not marker.is_file():
        raise FileNotFoundError(f'The frozen ImageNet index is required for directory input: {marker}')
    original = json.loads(marker.read_text())
    classes, rows = original['classes'], original['rows']
    if len(rows) != expected['images'] or len(classes) != expected['classes']:
        raise ValueError('Frozen ImageNet sample/class count changed')
    with ImageNetInput(archive_path) as source:
        raw = source.read('LOC_val_solution.csv')
        annotation_sha = hashlib.sha256(raw).hexdigest()
        if annotation_sha != original['annotation_sha256']:
            raise ValueError('Shared ImageNet annotation differs from the frozen index')
        annotations = {}
        for entry in csv.DictReader(io.StringIO(raw.decode())):
            synsets = set(entry['PredictionString'].split()[::5])
            if len(synsets) != 1 or entry['ImageId'] in annotations:
                raise ValueError('Ambiguous or duplicate ImageNet annotation')
            annotations[entry['ImageId']] = classes.index(synsets.pop())
        if len({r['id'] for r in rows}) != len(rows) or set(annotations) != {r['id'] for r in rows}:
            raise ValueError('Shared ImageNet sample IDs differ from the frozen index')
        relocated = []
        for row in rows:
            if annotations[row['id']] != row['label']:
                raise ValueError(f'ImageNet label changed: {row["id"]}')
            name = f'{VAL}/{row["id"]}.JPEG'
            data = source.read(name)
            if zlib.crc32(data) != row['crc32']:
                raise ValueError(f'Shared ImageNet image differs from the frozen index: {name}')
            path = source.source / name
            if use_prepared:
                path = cache / 'val' / classes[row['label']] / f'{row["id"]}.JPEG'
                if not path.is_file() or zlib.crc32(path.read_bytes()) != row['crc32']:
                    raise ValueError(f'Prepared ImageNet image differs from the frozen index: {path}')
            relocated.append({**row, 'path': str(path)})
        result = {**original, 'rows': relocated, 'folder': str(cache if use_prepared else source.source),
                  'input_format': source.kind, 'source_archive_snapshot': original['sources'],
                  'sources': [{'path': str(source.source), 'annotation_sha256': annotation_sha}]}
    result.pop('fingerprint', None)
    result['fingerprint'] = canonical(result)
    save(cache / 'index.directory.json', result)
    return result
