#!/usr/bin/env python3
"""Run this model's recorded inference and publish standardized outputs."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

MODEL = Path(__file__).resolve().parent
sys.path.insert(0, str(MODEL / "inference"))
CONFIG = json.loads((MODEL / 'inference/entrypoint.json').read_text())

def argument_value(args, name, default=None):
    for i in range(len(args) - 1, -1, -1):
        if args[i].startswith(name + '='):
            return args[i].split('=', 1)[1]
        if args[i] == name and i + 1 < len(args):
            return args[i + 1]
    return default

def replace_option(args, name, value):
    result = []
    i = 0
    while i < len(args):
        if args[i] == name:
            i += 2
        elif args[i].startswith(name + '='):
            i += 1
        else:
            result.append(args[i]); i += 1
    return result + [name, str(value)]

def publish(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2))
    temporary.replace(path)

def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument('--output-root', type=Path, help='Run directory; relative paths start at this model folder')
    parser.add_argument('--prepare-only', action='store_true', help='Convert a completed existing native run without inference')
    parser.add_argument('--phase', choices=['smoke', 'full', 'preflight'], default=None)
    parser.add_argument('--show-command', action='store_true', help='Print preserved commands without running them')
    args, native_args = parser.parse_known_args()
    if CONFIG.get('state') == 'historically_skipped':
        raise SystemExit('This model has no verified runnable workflow: ' + CONFIG['reason'])
    root = args.output_root or Path(os.environ.get('MODEL_OUTPUT_ROOT', 'result/run_001'))
    if not root.is_absolute():
        root = MODEL / root
    root = root.resolve()
    os.environ['MODEL_OUTPUT_ROOT'] = str(root)
    phase = args.phase or ('smoke' if '--smoke' in native_args or int(argument_value(native_args, '--max-images', '0')) > 0 else 'full')
    if args.prepare_only and phase == 'preflight':
        parser.error('--prepare-only requires --phase smoke or full')
    if CONFIG.get('single_model_sample'):
        mapping = {'__MODEL_ROOT__': str(MODEL), '__DATASET_ROOT__': str((MODEL / '../../../../dataset').resolve()),
                   '__OUTPUT_ROOT__': str(root), '__PYTHON__': sys.executable}
        def expand(value, values):
            for old, new in values.items(): value = value.replace(old, new)
            return value
    else:
        from model_paths import prepare, expand, values
        mapping = values() if args.show_command else prepare()
    commands = []
    for index, record in enumerate(CONFIG['commands']):
        command = [expand(v, mapping) for v in record['arguments']]
        if index == 0:
            if '--phase' in command:
                command = replace_option(command, '--phase', phase)
            elif args.phase == 'smoke':
                if CONFIG.get('accepts_smoke_flag'):
                    command.append('--smoke')
                elif '--max-images' not in command and not CONFIG.get('accepts_max_images'):
                    raise SystemExit('This native workflow has no verified smoke option; see provenance/original_launch.json')
                else:
                    command = replace_option(command, '--max-images', '10')
            elif args.phase == 'preflight':
                raise SystemExit('This native workflow does not expose --phase preflight')
            command.extend(native_args)
        elif '--phase' in command and phase != 'preflight':
            command = replace_option(command, '--phase', phase)
        elif phase == 'smoke' and '--max-images' in command:
            command = replace_option(command, '--max-images', argument_value(native_args, '--max-images', '10'))
        environment = dict(os.environ)
        environment.update({key: expand(value, mapping) for key, value in record['environment'].items()})
        if CONFIG['task'] == 'detection':
            environment['BENCH_UNIFIED_SCORING'] = '1'
            environment['BENCH_SCORING_CAPTURE_ROOT'] = str(root / 'scoring_inputs')
        commands.append((command, environment))
    if args.show_command:
        print(json.dumps({'commands': [c for c, e in commands], 'output_root': str(root), 'phase': phase}, ensure_ascii=False, indent=2))
        return
    introspection = any(flag in native_args for flag in ('--help', '--describe', '--preflight', '--paths-only', '--list-models', '--summarize', '--check-environment', '--setup-sources')) or phase == 'preflight'
    status = root / 'inference_output.json'
    if not introspection:
        publish(status, {'state': 'running', 'task': CONFIG['task'], 'sequence': CONFIG['sequence'], 'phase': phase, 'bundles': []})
    try:
        if not args.prepare_only:
            for index, (command, environment) in enumerate(commands):
                cwd = MODEL if CONFIG.get('single_model_sample') else MODEL / 'inference/native'
                needs_smoke = index in CONFIG.get('smoke_commands', [])
                if not introspection and phase == 'full' and needs_smoke:
                    smoke = replace_option(command, '--phase', 'smoke') if '--phase' in command else replace_option(command, '--max-images', '10')
                    subprocess.run(smoke, cwd=cwd, env=environment, check=True)
                subprocess.run(command, cwd=cwd, env=environment, check=True)
                if introspection:
                    return
        subprocess.run([sys.executable, str(MODEL / 'inference/prepare.py'), '--run-dir', str(root), '--phase', phase], cwd=MODEL, check=True)
        result = json.loads(status.read_text())
        if result.get('state') != 'completed':
            raise RuntimeError('Standardized output preparation did not complete')
        print('Standardized inference output: ' + str(status))
    except BaseException as exc:
        if not introspection:
            publish(status, {'state': 'failed', 'task': CONFIG['task'], 'sequence': CONFIG['sequence'], 'phase': phase, 'reason': str(exc), 'bundles': []})
        raise

if __name__ == '__main__':
    main()
