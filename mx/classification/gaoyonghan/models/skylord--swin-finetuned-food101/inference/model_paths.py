"""Model-local paths and relocation templates for the MX delivery layout."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import sys

MODEL_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = MODEL_ROOT.parents[3]


def configuration():
    manifest = json.loads((MODEL_ROOT / 'inference/native/models.json').read_text())
    if len(manifest['models']) != 1:
        raise ValueError('Expected exactly one model-local configuration')
    return manifest['models'][0]


def _relative(value, base=MODEL_ROOT):
    path = Path(value).expanduser()
    return (path if path.is_absolute() else base / path).resolve()


def resolve_paths():
    settings = json.loads((MODEL_ROOT / 'paths.json').read_text())
    data = _relative(os.environ.get('MODEL_DATASET_ROOT', settings['dataset_root']))
    weights = _relative(os.environ.get('MODEL_WEIGHTS_ROOT', settings.get('weights_root', 'weights')))
    if not data.is_relative_to(PROJECT_ROOT / 'dataset'):
        raise ValueError('Dataset must remain inside the MX platform dataset/')
    if not weights.is_relative_to(MODEL_ROOT / 'weights'):
        raise ValueError('Weights must remain inside this model weights/')
    return dict(project=PROJECT_ROOT, model=MODEL_ROOT, dataset=data, weights=weights)


def values():
    paths = resolve_paths()
    output = _relative(os.environ.get('MODEL_OUTPUT_ROOT', 'result/run_001'))
    if not output.is_relative_to(MODEL_ROOT / 'result') or output == MODEL_ROOT / 'result':
        raise ValueError('Output must be a run directory inside this model result/')
    return {'__MODEL_ROOT__': str(MODEL_ROOT), '__PROJECT_ROOT__': str(PROJECT_ROOT),
            '__DATASET_ROOT__': str(paths['dataset']), '__WEIGHTS_ROOT__': str(paths['weights']),
            '__OUTPUT_ROOT__': str(output), '__PYTHON__': sys.executable,
            '__SCORE_PYTHON__': os.environ.get('MODEL_SCORE_PYTHON', sys.executable)}


def expand(value, mapping):
    for old, new in mapping.items():
        value = value.replace(old, new)
    return value


def prepare():
    mapping = values()
    plan = json.loads((MODEL_ROOT / 'inference/local_paths.json').read_text())
    if plan.get('links'):
        raise ValueError('Symbolic links are not part of this delivery')
    with (MODEL_ROOT / 'inference/.local_paths.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        for item in plan['text_templates']:
            source = (MODEL_ROOT / item['source']).resolve()
            target = (MODEL_ROOT / item['destination']).resolve()
            if not source.is_relative_to(MODEL_ROOT / 'inference/path_templates'):
                raise ValueError('Template source escapes path_templates/')
            if not target.is_relative_to(MODEL_ROOT / 'inference'):
                raise ValueError('Template destination escapes inference/')
            content = expand(source.read_text(), mapping)
            if target.is_file() and target.read_text() == content:
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = target.with_name(target.name + '.relocating')
            temporary.write_text(content)
            if target.exists():
                temporary.chmod(target.stat().st_mode)
            temporary.replace(target)
    return mapping


def preflight():
    paths = resolve_paths()
    spec = configuration()
    for row in spec['weights_files']:
        path = paths['weights'] / row['path']
        if not path.is_file() or path.is_symlink() or path.stat().st_size != row['size']:
            raise ValueError('Missing, linked or wrong-size weight/config: ' + str(path))
    for relative in spec['dataset_files']:
        path = paths['dataset'] / relative
        if not path.is_file() or path.is_symlink():
            raise ValueError('Required dataset file is missing or linked: ' + str(path))
    return dict(state='static_preflight_passed', sequence=spec['sequence'],
                weight_files=len(spec['weights_files']), dataset_files=len(spec['dataset_files']),
                inference_executed=False, gpu_checked=False, environment_rebuilt=False)


def native_arguments():
    """Apply the public command arguments before the preserved native code starts."""
    spec = configuration()
    parser = argparse.ArgumentParser(description='Preserved native MX inference')
    parser.add_argument('--phase', choices=['full', 'smoke'], default=os.environ.get('MODEL_PHASE', 'full'))
    parser.add_argument('--output-root', default=os.environ.get('MODEL_OUTPUT_ROOT', 'result/run_001'))
    parser.add_argument('--device', default=os.environ.get('MX_DEVICE', 'cuda:0'))
    for flag, variable, key in [('batch-size', 'MODEL_BATCH_SIZE', 'batch_size'),
                                ('workers', 'MODEL_WORKERS', 'workers'),
                                ('warmup', 'MODEL_WARMUP', 'warmup')]:
        parser.add_argument('--' + flag, type=int, default=int(os.environ.get(variable, spec['runtime'][key])))
    parser.add_argument('--max-images', type=int)
    args = parser.parse_args()
    if not args.device.startswith('cuda:') or not args.device[5:].isdigit():
        parser.error('--device must be cuda:N on MX')
    if args.batch_size < 1 or args.workers < 0 or args.warmup < 1:
        parser.error('batch-size/warmup must be positive and workers nonnegative')
    if spec['sequence'] == 146 and args.workers != 0:
        parser.error('SPIDER requires --workers 0')
    if args.phase == 'smoke' and not spec.get('supports_max_samples'):
        parser.error('This native workflow does not support limited smoke')
    if args.max_images is not None and (args.phase != 'smoke' or args.max_images < 1):
        parser.error('--max-images must be positive and requires --phase smoke')
    output = _relative(args.output_root)
    if not output.is_relative_to(MODEL_ROOT / 'result') or output == MODEL_ROOT / 'result':
        parser.error('output-root must be a run directory inside this model result/')
    os.environ.update(MODEL_PHASE=args.phase, MODEL_OUTPUT_ROOT=str(output), MX_DEVICE=args.device,
                      MODEL_BATCH_SIZE=str(args.batch_size), MODEL_WORKERS=str(args.workers),
                      MODEL_WARMUP=str(args.warmup),
                      MODEL_MAX_IMAGES=str(args.max_images or 10) if args.phase == 'smoke' else '0')
    os.environ.pop('MAX_SAMPLES', None)

