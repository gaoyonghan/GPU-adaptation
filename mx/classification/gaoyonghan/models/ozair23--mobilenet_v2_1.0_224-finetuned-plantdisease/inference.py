#!/usr/bin/env python3
"""Independent native MX inference and standard classification conversion."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import time

MODEL = Path(__file__).resolve().parent
sys.path.insert(0, str(MODEL / 'inference'))
from model_paths import configuration, expand, preflight, prepare, resolve_paths, values


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(path)


def main():
    config = configuration()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-root', default=os.environ.get('MODEL_OUTPUT_ROOT', 'result/run_001'))
    parser.add_argument('--phase', choices=['full', 'smoke', 'preflight'], default='full')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--batch-size', type=int, default=config['runtime']['batch_size'])
    parser.add_argument('--workers', type=int, default=config['runtime']['workers'])
    parser.add_argument('--warmup', type=int, default=config['runtime']['warmup'])
    parser.add_argument('--max-images', type=int)
    parser.add_argument('--show-command', action='store_true')
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args()
    if not args.device.startswith('cuda:') or not args.device[5:].isdigit():
        parser.error('--device must be cuda:N on MX; CPU fallback is forbidden')
    if args.batch_size < 1 or args.workers < 0 or args.warmup < 1:
        parser.error('batch-size/warmup must be positive and workers nonnegative')
    if config['sequence'] == 146 and args.workers != 0:
        parser.error('SPIDER preserves its native synchronous generator; --workers must be 0')
    if args.max_images is not None and (args.phase != 'smoke' or args.max_images < 1):
        parser.error('--max-images is positive and supported only with --phase smoke')
    if args.phase == 'smoke' and not config.get('supports_max_samples'):
        parser.error('Limited native smoke is supported only for m112/m146; use full')
    output = Path(args.output_root).expanduser()
    output = (output if output.is_absolute() else MODEL / output).resolve()
    if not output.is_relative_to(MODEL / 'result') or output == MODEL / 'result':
        parser.error('output-root must be a run directory inside this model result/')
    os.environ['MODEL_OUTPUT_ROOT'] = str(output)
    mapping = values() if args.show_command else prepare()
    if args.phase == 'preflight':
        print(json.dumps(preflight(), ensure_ascii=False, indent=2))
        return
    entry = json.loads((MODEL / 'inference/entrypoint.json').read_text())
    if entry['sequence'] != config['sequence'] or len(entry['commands']) != 1:
        raise ValueError('Invalid model-local entrypoint configuration')
    command = [expand(value, mapping) for value in entry['commands'][0]['arguments']]
    for flag, value in [('--phase', args.phase), ('--output-root', str(output)),
                         ('--device', args.device), ('--batch-size', str(args.batch_size)),
                         ('--workers', str(args.workers)), ('--warmup', str(args.warmup))]:
        command[command.index(flag) + 1] = value
    if args.phase == 'smoke':
        command.extend(['--max-images', str(args.max_images or 10)])
    env = os.environ.copy()
    env.update({key: expand(value, mapping) for key, value in entry['commands'][0].get('environment', {}).items()})
    env.update(MODEL_PHASE=args.phase, MX_DEVICE=args.device,
               MODEL_BATCH_SIZE=str(args.batch_size), MODEL_WORKERS=str(args.workers),
               MODEL_WARMUP=str(args.warmup),
               MODEL_MAX_IMAGES=str(args.max_images or 10) if args.phase == 'smoke' else '0')
    env.pop('MAX_SAMPLES', None)
    preparation = [sys.executable, str(MODEL / 'inference/prepare.py'), '--run-dir', str(output), '--phase', args.phase]
    if args.show_command:
        print(json.dumps(dict(native=shlex.join(command), conversion=shlex.join(preparation),
                              cwd=str(MODEL / 'inference/native'),
                              parameters={key: env[key] for key in ('MODEL_PHASE', 'MX_DEVICE', 'MODEL_BATCH_SIZE', 'MODEL_WORKERS', 'MODEL_WARMUP', 'MODEL_MAX_IMAGES')},
                              paths={key: str(value) for key, value in resolve_paths().items()}), ensure_ascii=False, indent=2))
        return
    preflight()
    output.mkdir(parents=True, exist_ok=True)
    with (output / '.entrypoint.lock').open('a') as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        manifest = dict(task='classification', sequence=config['sequence'], phase=args.phase,
                        state='running', scoring_ready=False, bundles=[], command=sys.argv,
                        python_executable=sys.executable, native_command=command,
                        conversion_command=preparation,
                        runtime=dict(device=args.device, batch_size=args.batch_size,
                                     workers=args.workers, warmup=args.warmup),
                        environment='e07', started_at=time.time())
        if args.prepare_only:
            with (output / (args.phase + '.log')).open('a') as log:
                log.write('\nPrepare-only command: ' + shlex.join(preparation) + '\n')
                log.flush()
                subprocess.run(preparation, check=True, cwd=MODEL, env=env, stdout=log, stderr=subprocess.STDOUT)
            return
        if (output / 'inference_output.json').exists():
            archive = output / '_history' / ('entrypoint-' + str(time.time_ns()))
            archive.mkdir(parents=True)
            for name in ('inference_output.json', 'evaluation_summary.json', args.phase + '.log'):
                if (output / name).is_file():
                    shutil.copy2(output / name, archive / name)
        save(output / 'inference_output.json', manifest)
        try:
            with (output / (args.phase + '.log')).open('w') as log:
                log.write('Native command: ' + shlex.join(command) + '\n')
                log.flush()
                process = subprocess.Popen(command, cwd=MODEL / 'inference/native', env=env,
                                           stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
                try:
                    for line in process.stdout:
                        sys.stdout.write(line)
                        log.write(line)
                        log.flush()
                    result = process.wait()
                except BaseException:
                    process.terminate()
                    process.wait(timeout=30)
                    raise
                if result:
                    raise subprocess.CalledProcessError(result, command)
                log.write('\nConversion command: ' + shlex.join(preparation) + '\n')
                log.flush()
                subprocess.run(preparation, check=True, cwd=MODEL, env=env, stdout=log, stderr=subprocess.STDOUT)
                log.write('Conversion completed.\n')
        except BaseException as exc:
            manifest.update(state='failed', reason=str(exc), bundles=[], scoring_ready=False)
            save(output / 'inference_output.json', manifest)
            raise


if __name__ == '__main__':
    main()
