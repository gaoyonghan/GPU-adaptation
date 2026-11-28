#!/usr/bin/env python3
"""Run model stages sequentially, with validated model-level resume.

Examples (run from the project directory):
  /usr/bin/python3 -I -S -B run_all_inference.py --venv auto --dry-run
  /usr/bin/python3 -I -S -B run_all_inference.py --mode inference --start-from PramaLLC__BEN2
  /usr/bin/python3 -I -S -B run_all_inference.py --mode export --model Xenova__modnet
  /usr/bin/python3 -I -S -B run_all_inference.py --mode evaluation

A completed bundle is reused only after code/configuration, model/run, source
manifest and four JSON payload hashes agree. Mismatched existing outputs fail
closed; --force explicitly requests replacement. Prediction-cache/shard resume
is delegated to the selected model's standalone inference.py. No model package
is imported here, and no root log/result file is created. --dry-run never writes.
All relative command-line paths are relative to this project directory.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys

sys.dont_write_bytecode = True
PAYLOADS = ('ground_truth.json', 'predictions.json', 'semantic_ground_truth.json', 'semantic_predictions.json')
METRICS = ('mIoU', 'IoU', 'Mask AP@[.50:.95]', 'AP50', 'AP75')
TF_MODELS = {'Narsil__pet-segmentation', 'keras-io__semantic-segmentation',
             'SerdarHelli__Segmentation-of-Teeth-in-Panoramic-X-ray-Image-Using-U-Net'}


def positive(value):
    n = int(value)
    if n < 1:
        raise argparse.ArgumentTypeError('must be positive')
    return n


def nonnegative(value):
    n = int(value)
    if n < 0:
        raise argparse.ArgumentTypeError('must be nonnegative')
    return n


def parser_for():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--mode', choices=('inference', 'export', 'evaluation', 'full'), default='full')
    selection = p.add_mutually_exclusive_group()
    selection.add_argument('--model', action='append', help='Exact model directory name; repeat for several models')
    selection.add_argument('--start-from', help='Inclusive start model in directory-name sort order')
    p.add_argument('--force', action='store_true', help='Explicitly replace selected stage outputs')
    p.add_argument('--resume', action='store_true', default=True, help='Default: reuse validated completed work')
    p.add_argument('--venv', help='auto, current, or project environment path; defaults to auto with -S, otherwise current')
    p.add_argument('--environment', choices=('auto', 'current'), default='auto', help='current uses the caller interpreter/packages')
    p.add_argument('--python', help='Child Python interpreter; default is the current executable')
    p.add_argument('--dataset')
    p.add_argument('--dataset-root')
    p.add_argument('--split', action='append')
    p.add_argument('--seed', type=int)
    p.add_argument('--device', choices=('auto', 'cpu', 'cuda'))
    p.add_argument('--precision', choices=('fp32', 'fp16', 'bf16'))
    p.add_argument('--limit', type=positive)
    p.add_argument('--warmup', type=nonnegative)
    p.add_argument('--threads', type=positive)
    p.add_argument('--result-dir', help='Single model only; path within that model/result for isolated outputs')
    p.add_argument('--export-mode', choices=('json', 'full'), help='Explicitly require this export format; omitted preserves a validated existing bundle')
    p.add_argument('--prediction-images', choices=('none', 'labels', 'preview', 'both'), help='Explicit prediction image selection for JSON export')
    p.add_argument('--skip-evaluation', action='store_true', help='In full mode, stop after exporting the bundle')
    p.add_argument('--check-level', choices=('metadata', 'payload', 'full'), default='payload', help='Evaluation check level; completed inference reuse always checks payload hashes')
    p.add_argument('--dry-run', action='store_true', help='Inspect and print commands without starting a child or writing')
    p.add_argument('--stop-on-error', action='store_true')
    return p


def read_object(path):
    def reject(value):
        raise ValueError('Non-finite JSON value: ' + value)
    value = json.loads(Path(path).read_text(encoding='utf-8'), parse_constant=reject)
    if not isinstance(value, dict):
        raise ValueError('Expected JSON object: ' + str(path))
    return value


def no_symlinks(path):
    path = Path(path).absolute()
    if any(p.is_symlink() for p in (path, *path.parents)):
        raise ValueError('Path must not traverse symlinks: ' + str(path))
    return path


def digest(path):
    path = no_symlinks(path)
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def project_path(value, root):
    p = Path(value)
    return (p if p.is_absolute() else root / p).absolute()


def selected_models(root, args):
    directory = no_symlinks(root / 'Model')
    available = {p.name: p for p in sorted(directory.iterdir()) if p.is_dir()}
    wanted = set(args.model or available)
    missing = wanted - set(available)
    if missing:
        raise ValueError('Unknown model(s): ' + ', '.join(sorted(missing)))
    if args.start_from is not None:
        if args.start_from not in available:
            raise ValueError('Unknown start model: ' + args.start_from)
        wanted = {name for name in available if name >= args.start_from}
    models = []
    for name in sorted(wanted):
        model = no_symlinks(available[name])
        no_symlinks(model / 'inference.py')
        no_symlinks(model / 'weights/model.json')
        config = read_object(model / 'weights/model.json')
        if config.get('model_id') != name:
            raise ValueError('Configuration model_id differs from directory: ' + name)
        if not (model / 'inference.py').is_file():
            raise ValueError('Missing inference.py: ' + name)
        if args.dataset and args.dataset not in {x.get('name') for x in config.get('datasets', [])}:
            continue
        models.append((model, config))
    if not models:
        raise ValueError('No models match the selection')
    if args.result_dir and len(models) != 1:
        raise ValueError('--result-dir requires exactly one model')
    return models


def result_path(model, args, root):
    path = project_path(args.result_dir, root) if args.result_dir else model / 'result'
    path = no_symlinks(path)
    if not path.resolve().is_relative_to((model / 'result').resolve()):
        raise ValueError('--result-dir must remain within this model/result')
    return path


def evaluation_complete(result, manifest_digest):
    metrics = result.get('metrics')
    if (result.get('status') != 'complete' or result.get('schema_version') != 'segmentation-evaluation/v2'
            or result.get('bundle_manifest_sha256') != manifest_digest
            or result.get('protocol', {}).get('ap', {}).get('definition') != 'class_region_ap_v1'
            or not isinstance(metrics, dict) or set(metrics) != set(METRICS)):
        return False
    return all(value is None or type(value) in (float, int) and math.isfinite(value) and 0 <= value <= 1 for value in metrics.values())


def check_requested(args, request, metadata, data_root):
    """Only explicitly supplied options constrain historical execution settings."""
    for name in ('device', 'precision', 'warmup', 'threads', 'seed'):
        wanted = getattr(args, name)
        previous = request.get(name)
        if previous is None and name == 'seed':
            previous = metadata.get('seed')
        if wanted is not None and (type(previous) is not type(wanted) or previous != wanted):
            raise ValueError('Requested ' + name + ' differs from completed inference; use --force to rerun')
    if args.dataset is not None and metadata.get('dataset') != args.dataset:
        raise ValueError('Requested dataset differs from completed inference')
    if args.split is not None:
        previous = metadata.get('splits', request.get('split'))
        previous = [previous] if isinstance(previous, str) else previous
        if not isinstance(previous, list) or sorted(previous) != sorted(args.split):
            raise ValueError('Requested split differs from completed inference')
    if args.dataset_root is not None and Path(args.dataset_root).resolve() != data_root:
        raise ValueError('Requested dataset-root differs from completed inference')
    full, selected = metadata.get('full_source_samples'), metadata.get('selected_source_samples')
    if type(full) is not int or full < 1 or type(selected) is not int or not 0 < selected <= full:
        raise ValueError('Missing or invalid source sample counts')
    expected = min(args.limit, full) if args.limit is not None else full
    if selected != expected or metadata.get('scope') != ('full' if selected == full else 'subset'):
        raise ValueError('Requested limit/full scope differs from completed inference')


def validate_pending_cache(model, config, args, result_directory, current):
    """Validate new inference before delegating, ignoring the retained old bundle.

    Shard sizes/coverage are checked here. ZIP index/record validation remains
    mandatory in the standalone worker, before any cache reuse or publication.
    """
    cache_dir = no_symlinks(result_directory / 'prediction_cache')
    cache = read_object(no_symlinks(cache_dir / 'manifest.json'))
    run_id = cache.get('run_id')
    request = cache.get('request')
    if (cache.get('schema_version') != 'deferred-predictions/v1' or cache.get('status') != 'complete'
            or cache.get('model_id') != model.name or not isinstance(run_id, str) or not run_id
            or current.get('model_id') != model.name or current.get('run_id') != run_id
            or not isinstance(request, dict)):
        raise ValueError('Cached result and completed prediction cache model/run do not match')
    if json.dumps(current.get('request'), sort_keys=True, allow_nan=False) != json.dumps(request, sort_keys=True, allow_nan=False):
        raise ValueError('Cached result request differs from prediction cache')
    if request.get('environment') != 'current':
        raise ValueError('Cached request does not match the standalone current runtime protocol')
    for field, relative in [('inference_sha256', 'inference.py'), ('config_sha256', 'weights/model.json')]:
        if request.get(field) != digest(model / relative):
            raise ValueError('Pending cache code/configuration hash mismatch: ' + field)
    dataset = cache.get('dataset')
    if not isinstance(dataset, dict):
        raise ValueError('Pending cache has no dataset description')
    name = dataset.get('name')
    if (not isinstance(name, str) or Path(name).name != name or request.get('dataset') != name
            or name not in {item.get('name') for item in config.get('datasets', [])}):
        raise ValueError('Pending cache dataset identity differs from current configuration')
    data_root = (model / config['dataset_root']).resolve()
    if not isinstance(request.get('dataset_root'), str) or Path(request['dataset_root']).resolve() != data_root:
        raise ValueError('Pending cache dataset root differs from current configuration')
    data = data_root if (data_root / 'evaluation_samples.jsonl').is_file() else data_root / name
    if request.get('source_manifest_sha256') != digest(data / 'evaluation_samples.jsonl'):
        raise ValueError('Pending cache source manifest hash mismatch')
    dataset_manifest = no_symlinks(data / 'dataset_manifest.json')
    if request.get('dataset_manifest_sha256') != (digest(dataset_manifest) if dataset_manifest.is_file() else None):
        raise ValueError('Pending cache dataset manifest hash mismatch')
    state = read_object(no_symlinks(result_directory / 'prediction_cache_state.json'))
    if (state.get('status') != 'complete' or state.get('run_id') != run_id
            or state.get('model_id') not in (None, model.name)):
        raise ValueError('Pending cache completion state does not match its run')
    metadata = {'dataset': name, 'splits': dataset.get('splits'), 'seed': request.get('seed'),
                'scope': dataset.get('scope'), 'selected_source_samples': dataset.get('count'),
                'full_source_samples': dataset.get('full_count')}
    check_requested(args, request, metadata, data_root)
    if current.get('scope') != dataset.get('scope'):
        raise ValueError('Cached result scope differs from prediction cache')
    # Worker make_request uses exact arguments, including a literal limit. Do
    # not silently rerun inference when two limits happen to cover equal counts.
    for field in ('limit', 'split'):
        supplied = getattr(args, field)
        if supplied is not None and supplied != request.get(field):
            raise ValueError('Requested ' + field + ' differs from cached request; use --force to rerun')
    shards = cache.get('shards')
    if not isinstance(shards, list) or not shards:
        raise ValueError('Pending cache has no prediction shards')
    total = 0
    for index, shard in enumerate(shards):
        name = f'predictions_{index:05d}.zip'
        if not isinstance(shard, dict) or shard.get('name') != name:
            raise ValueError('Pending cache shard order/name mismatch')
        path = no_symlinks(cache_dir / name)
        count = shard.get('count')
        size = shard.get('bytes')
        fingerprint = shard.get('index_sha256')
        if (type(count) is not int or count < 1 or type(size) is not int or size < 1
                or not path.is_file() or path.stat().st_size != size
                or not isinstance(fingerprint, str) or len(fingerprint) != 64
                or any(c not in '0123456789abcdef' for c in fingerprint)):
            raise ValueError('Pending cache shard metadata/size mismatch: ' + name)
        total += count
    if total != dataset.get('count'):
        raise ValueError('Pending cache shard coverage is incomplete')
    # Preserve unspecified verified settings so the worker sees the exact same
    # request. This Namespace belongs only to the current model in main().
    for field in ('dataset', 'dataset_root', 'split', 'seed', 'device', 'precision', 'limit', 'warmup', 'threads'):
        if getattr(args, field) is None:
            setattr(args, field, request.get(field))


def reusable_bundle(model, config, args, result_directory):
    """Return validated completion or None; malformed existing evidence never reruns silently."""
    bundle = result_directory / 'coco_bundle'
    result_file = result_directory / 'results.json'
    if result_file.exists() or result_file.is_symlink():
        pending = read_object(no_symlinks(result_file))
        if pending.get('status') == 'cached':
            validate_pending_cache(model, config, args, result_directory, pending)
            return None
    if not bundle.exists() and not bundle.is_symlink():
        if result_file.exists() and read_object(result_file).get('status') in ('complete', 'exported'):
            raise ValueError('Completed result is missing its coco_bundle')
        return None
    no_symlinks(bundle)
    manifest = read_object(bundle / 'manifest.json')
    metadata = manifest.get('metadata')
    if (manifest.get('schema_version') != 'segmentation-coco-bundle/v1' or manifest.get('complete') is not True
            or not isinstance(metadata, dict) or metadata.get('model_id') != model.name):
        raise ValueError('Bundle is incomplete or belongs to another model')
    run_id = metadata.get('run_id')
    count = manifest.get('expected_samples')
    ids = manifest.get('coverage_sample_ids')
    if (not isinstance(run_id, str) or not run_id or type(count) is not int or count < 1
            or manifest.get('evaluated_samples') != count or not isinstance(ids, list)
            or len(ids) != count or len(set(ids)) != count):
        raise ValueError('Bundle run identity or coverage is invalid')
    hashes = manifest.get('payload_sha256')
    if not isinstance(hashes, dict) or set(hashes) != set(PAYLOADS):
        raise ValueError('Invalid payload hash manifest')
    for name in PAYLOADS:
        if digest(bundle / name) != hashes[name]:
            raise ValueError('Bundle payload hash mismatch: ' + name)
    manifest_sha = digest(bundle / 'manifest.json')
    current = read_object(no_symlinks(result_file))
    current_meta = current.get('metadata') or {}
    if current.get('status') not in ('complete', 'exported'):
        raise ValueError('Current result is not exported/complete')
    for item in (current, current_meta):
        if item.get('model_id') not in (None, model.name) or item.get('run_id') not in (None, run_id):
            raise ValueError('Current result model/run differs from bundle')
    if not any(item.get('model_id') == model.name for item in (current, current_meta)):
        raise ValueError('Current result lacks model identity')
    if not any(item.get('run_id') == run_id for item in (current, current_meta)):
        raise ValueError('Current result lacks run identity')
    if current.get('bundle_manifest_sha256') not in (None, manifest_sha):
        raise ValueError('Current result manifest hash mismatch')

    proof_path = result_directory / 'layout_migration.json'
    request = None
    cache_path = result_directory / 'prediction_cache/manifest.json'
    cache = read_object(no_symlinks(cache_path)) if cache_path.exists() or cache_path.is_symlink() else None
    fresh_cache = False
    if cache is not None:
        candidate = cache.get('request')
        fresh_cache = (cache.get('schema_version') == 'deferred-predictions/v1'
                       and cache.get('status') == 'complete' and cache.get('model_id') == model.name
                       and cache.get('run_id') == run_id and isinstance(candidate, dict)
                       and candidate.get('inference_sha256') == digest(model / 'inference.py')
                       and candidate.get('config_sha256') == digest(model / 'weights/model.json'))
    proof_exists = proof_path.exists() or proof_path.is_symlink()
    if proof_exists and not fresh_cache:
        proof = read_object(no_symlinks(proof_path))
        if (proof.get('schema_version') != 'segmentation-layout-migration/v1'
                or proof.get('model_id') != model.name or proof.get('run_id') != run_id
                or proof.get('bundle_manifest_sha256') != manifest_sha):
            raise ValueError('Unrecognized or mismatched layout migration evidence')
        files = proof.get('current_files')
        if not isinstance(files, dict) or set(files) != {'inference.py', 'weights/model.json'}:
            raise ValueError('Migration evidence must bind inference and configuration hashes')
        for relative, expected in files.items():
            if digest(model / relative) != expected:
                raise ValueError('Code/configuration hash changed: ' + relative)
        if not isinstance(proof.get('dataset_root'), str) or Path(proof['dataset_root']).is_absolute():
            raise ValueError('Migration dataset_root must be relative to model')
        data_root = (model / proof['dataset_root']).resolve()
        dataset_name = proof.get('dataset_name')
        source_sha = proof.get('source_manifest_sha256')
        dataset_sha = proof.get('dataset_manifest_sha256')
    elif cache is not None:
        request = cache.get('request')
        if (cache.get('schema_version') != 'deferred-predictions/v1' or cache.get('status') != 'complete'
                or cache.get('model_id') != model.name or cache.get('run_id') != run_id or not isinstance(request, dict)):
            raise ValueError('Current bundle lacks a matching completed inference cache')
        for field, relative in [('inference_sha256', 'inference.py'), ('config_sha256', 'weights/model.json')]:
            if request.get(field) != digest(model / relative):
                raise ValueError('Current cache code/configuration hash mismatch: ' + field)
        data_root = (model / config['dataset_root']).resolve()
        if not isinstance(request.get('dataset_root'), str) or Path(request['dataset_root']).resolve() != data_root:
            raise ValueError('Current cache dataset root mismatch')
        dataset_name = request.get('dataset')
        source_sha = request.get('source_manifest_sha256')
        dataset_sha = request.get('dataset_manifest_sha256')
    else:
        raise ValueError('Existing bundle has no recognized migration/current-cache evidence; use --force for a new run')
    if (not isinstance(dataset_name, str) or Path(dataset_name).name != dataset_name
            or metadata.get('dataset') != dataset_name or (model / config['dataset_root']).resolve() != data_root):
        raise ValueError('Bundle dataset identity/configuration differs from evidence')
    data = data_root if (data_root / 'evaluation_samples.jsonl').is_file() else data_root / dataset_name
    if not isinstance(source_sha, str) or digest(data / 'evaluation_samples.jsonl') != source_sha:
        raise ValueError('Source manifest hash mismatch')
    if metadata.get('source_manifest_sha256') != source_sha:
        raise ValueError('Bundle source manifest differs from evidence')
    data_manifest = data / 'dataset_manifest.json'
    if (digest(data_manifest) if data_manifest.is_file() else None) != dataset_sha:
        raise ValueError('Dataset manifest hash mismatch')
    if request is None:
        # Historical bundle-only models have no prediction cache. The original
        # run configuration remains a provenance record, not a runtime path.
        saved = read_object(no_symlinks(result_directory / 'run_config.json'))
        request = saved.get('arguments')
        if not isinstance(request, dict):
            raise ValueError('Missing original inference request')
    check_requested(args, request, metadata, data_root)
    if args.export_mode is not None or args.prediction_images is not None:
        actual_mode = metadata.get('export_mode')
        if actual_mode is None and isinstance(manifest.get('image_artifacts_sha256'), dict):
            actual_mode = 'full'
        if args.export_mode is not None and actual_mode != args.export_mode:
            raise ValueError('Requested export-mode differs from existing bundle; use --force to export again')
        actual_pictures = metadata.get('prediction_images', 'none')
        if args.prediction_images is not None and actual_pictures != args.prediction_images:
            raise ValueError('Requested prediction-images differs from existing bundle; use --force to export again')
        if actual_mode == 'full' and args.prediction_images not in (None, 'none'):
            raise ValueError('Full bundles already include pictures; prediction-images requires JSON export')
    if digest(bundle / 'manifest.json') != manifest_sha:
        raise ValueError('Bundle changed during validation')
    return {'bundle': bundle, 'evaluated': evaluation_complete(current, manifest_sha),
            'reuse_basis': 'fresh_run_supersedes_migration' if fresh_cache and proof_exists else 'validated_completion'}


def child_prefix(root, model, mode, args):
    interpreter = args.python or sys.executable
    if args.python:
        interpreter = shutil.which(interpreter) if '/' not in interpreter else str(project_path(interpreter, root))
        if interpreter is None:
            raise ValueError('Requested Python interpreter was not found')
    chosen = args.venv
    if args.environment == 'current':
        if chosen not in (None, 'current'):
            raise ValueError('--environment current cannot be combined with --venv')
        chosen = 'current'
    if chosen is None:
        chosen = 'auto' if sys.flags.no_site else 'current'
    if chosen == 'auto':
        chosen = 'venv_tf' if mode in ('inference', 'full') and model.name in TF_MODELS else 'venv'
    if chosen == 'current':
        if sys.flags.no_site and not args.python:
            raise ValueError('Current Python was started with -S; select --venv auto or a usable --python')
        return [str(interpreter), '-B'], ['--environment', 'current']
    environment = project_path(chosen, root)
    if not environment.resolve().is_relative_to(root):
        raise ValueError('--venv must refer to an environment inside this project')
    no_symlinks(environment)
    return [str(interpreter), '-I', '-S', '-B'], ['--venv', str(environment)]


def child_command(root, model, mode, args, result_directory):
    prefix, runtime = child_prefix(root, model, mode, args)
    environment = os.environ.copy()
    if mode in ('export', 'evaluation'):
        environment['CUDA_VISIBLE_DEVICES'] = ''
    environment['PYTHONDONTWRITEBYTECODE'] = '1'
    if mode == 'evaluation':
        command = prefix + [str(no_symlinks(root / 'evaluate.py')), '--model', model.name] + runtime
        command += ['--check-level', args.check_level]
        if args.result_dir:
            command += ['--bundle', str(result_directory / 'coco_bundle'), '--output', str(result_directory / 'results.json')]
        if args.force:
            command.append('--force')
        return command, environment
    command = prefix + [str(model / 'inference.py'), '--stage', mode] + runtime
    for option in ('dataset', 'dataset_root', 'seed', 'device', 'precision', 'limit', 'warmup', 'threads', 'result_dir', 'export_mode', 'prediction_images'):
        value = getattr(args, option)
        if value is not None:
            command += ['--' + option.replace('_', '-'), str(value)]
    for split in args.split or []:
        command += ['--split', split]
    if args.force:
        command.append('--force')
    if mode == 'full' and args.skip_evaluation:
        command.append('--skip-evaluation')
    return command, environment


def main(argv=None):
    parser = parser_for()
    args = parser.parse_args(argv)
    root = Path(__file__).resolve().parent
    try:
        if args.export_mode == 'full' and args.prediction_images not in (None, 'none'):
            raise ValueError('--export-mode full requires --prediction-images none')
        if args.mode in ('inference', 'evaluation') and (args.export_mode is not None or args.prediction_images is not None):
            raise ValueError('Export options apply only to --mode export or full')
        if args.skip_evaluation and args.mode != 'full':
            raise ValueError('--skip-evaluation applies only to --mode full')
        if args.mode == 'evaluation' and any(getattr(args, key) is not None for key in ('device', 'precision', 'limit', 'warmup', 'threads', 'seed', 'split', 'dataset_root')):
            raise ValueError('Inference options cannot be applied to evaluation of an existing bundle')
        for key in ('dataset_root', 'result_dir'):
            value = getattr(args, key)
            if value is not None:
                setattr(args, key, str(project_path(value, root)))
        models = selected_models(root, args)
    except (OSError, ValueError, TypeError, KeyError) as exc:
        parser.error(str(exc))
    completed = skipped = failed = 0
    for model, config in models:
        try:
            model_args = argparse.Namespace(**vars(args))
            output = result_path(model, model_args, root)
            mode = args.mode
            if mode == 'evaluation':
                current_path = output / 'results.json'
                if current_path.exists() or current_path.is_symlink():
                    current = read_object(no_symlinks(current_path))
                    if current.get('status') == 'cached':
                        raise ValueError('Current status is cached: export this run before evaluation; the retained old bundle is not current')
            if mode != 'evaluation' and not args.force:
                reusable = reusable_bundle(model, config, model_args, output)
                if reusable:
                    if mode != 'full' or args.skip_evaluation or reusable['evaluated']:
                        print('SKIP ' + model.name + ': validated completed ' + mode + ' (' + reusable['reuse_basis'] + ')', flush=True)
                        skipped += 1
                        continue
                    mode = 'evaluation'
            command, environment = child_command(root, model, mode, model_args, output)
            print(('DRY-RUN ' if args.dry_run else 'RUN ') + model.name + ': ' + shlex.join(command), flush=True)
            if args.dry_run:
                continue
            process = subprocess.run(command, cwd=root, env=environment)
            if process.returncode:
                raise RuntimeError('child exit=' + str(process.returncode))
            completed += 1
        except (OSError, ValueError, TypeError, KeyError, RuntimeError) as exc:
            failed += 1
            print('FAILED ' + model.name + ': ' + str(exc), file=sys.stderr, flush=True)
            if args.stop_on_error:
                break
        except KeyboardInterrupt:
            print('Interrupted: remaining models were not started', file=sys.stderr, flush=True)
            return 130
    print(f'SUMMARY selected={len(models)} completed={completed} skipped={skipped} failed={failed} dry_run={args.dry_run}', flush=True)
    return 1 if failed else 0


if __name__ == '__main__':
    raise SystemExit(main())
