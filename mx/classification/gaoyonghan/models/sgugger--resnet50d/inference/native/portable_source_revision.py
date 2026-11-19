"""Verify a packaged source snapshot without executing Git on the target host."""
import hashlib
import json
import os
from pathlib import Path

IGNORED_DIRS = {'.git', '__pycache__', '.pytest_cache', '.mypy_cache', '.ruff_cache'}
IGNORED_SUFFIXES = {'.pyc', '.pyo'}


def source_files(root):
    root = Path(root)
    files = {}
    for directory, dirs, names in os.walk(root, followlinks=False):
        dirs[:] = sorted(d for d in dirs if d not in IGNORED_DIRS)
        for d in dirs:
            if (Path(directory) / d).is_symlink():
                raise ValueError(f'Source directory symlink is unsupported: {Path(directory) / d}')
        for name in sorted(names):
            p = Path(directory) / name
            if p.suffix in IGNORED_SUFFIXES:
                continue
            relative = p.relative_to(root).as_posix()
            if p.is_symlink() and not p.resolve().is_relative_to(root.resolve()):
                raise ValueError(f'Source file symlink escapes snapshot: {p}')
            files[relative] = p
    return files


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def verified_source_revision(directory, expected=None, manifest_path=None):
    """Return the recorded revision only after verifying all packaged source files."""
    directory = Path(directory)
    manifest_path = Path(manifest_path) if manifest_path else Path(__file__).with_name('source_revisions.json')
    manifest = json.loads(manifest_path.read_text())
    if manifest.get('schema') != 'portable-source-revisions-v1':
        raise ValueError('Unsupported source revision manifest')
    source = manifest['sources'].get(directory.name)
    if source is None or not directory.is_dir():
        raise ValueError(f'Unrecorded or missing source directory: {directory}')
    if expected is not None and source['commit'] != expected:
        raise ValueError(f'Official source revision changed: {directory.name}')
    actual = source_files(directory)
    if set(actual) != set(source['files']):
        missing = sorted(set(source['files']) - set(actual))[:5]
        extra = sorted(set(actual) - set(source['files']))[:5]
        raise ValueError(f'Source inventory changed: {directory.name}; missing={missing}; extra={extra}')
    for name, item in source['files'].items():
        link = os.readlink(actual[name]) if actual[name].is_symlink() else None
        if link != item.get('symlink'):
            raise ValueError(f'Source link changed: {directory.name}/{name}')
        if actual[name].stat().st_size != item['size'] or sha256(actual[name]) != item['sha256']:
            raise ValueError(f'Source file changed: {directory.name}/{name}')
    print(f'[source-verification] {directory.name}: commit={source["commit"]}; '
          f'files={len(actual)}; SHA256 verified; Git not required', flush=True)
    return source['commit']
