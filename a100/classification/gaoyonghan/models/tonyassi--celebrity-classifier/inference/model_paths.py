"""Resolve the model's relative paths and its packaged runtime resources."""
import json
import os
from pathlib import Path

MODEL = Path(__file__).resolve().parents[1]


def resolve(name, key):
    config_path = MODEL / 'paths.json'
    if not config_path.is_file():
        config_path = MODEL / 'inference/path_templates/paths.json'
    config = json.loads(config_path.read_text(encoding='utf-8'))
    path = Path(os.environ.get(name, config[key])).expanduser()
    return (path if path.is_absolute() else MODEL / path).resolve()


def runtime_path(key):
    config = json.loads((MODEL / 'inference/local_paths.json').read_text(encoding='utf-8'))
    return MODEL / config['runtime_resources'][key]
