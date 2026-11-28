#!/usr/bin/env python3
"""facebook__mask2former-swin-tiny-coco-instance: standalone data loading, inference, timing and COCO export.

Model modules are embedded as readable source at the end of this file.
No project Python source is loaded from weights/ or another model directory.
"""
from __future__ import annotations

# Standalone stages: standard-library bootstrap before model dependencies.
import sys as _standalone_sys
import os as _standalone_os

# Standalone, explicit activation of existing packages; no venv files are edited.
def _activate_project_venv(argv, project_root):
    import argparse
    import os
    import sys
    from pathlib import Path
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    parser.add_argument('--venv')
    option, _ = parser.parse_known_args(argv)
    if not option.venv:
        return None
    if not sys.flags.no_site:
        raise ValueError('--venv requires the matching Python with -I -S -B (see environment/README.md)')
    supplied = Path(option.venv).expanduser()
    project = Path(project_root).resolve()
    candidate = supplied if supplied.is_absolute() else project / supplied
    environment = candidate.resolve()
    if not environment.is_relative_to(project) or environment == project:
        raise ValueError('Environment must belong to this project copy')
    if any(part.is_symlink() for part in (candidate, *candidate.parents)):
        raise ValueError('Environment path must not use symlinks')
    values = {}
    for line in (environment / 'pyvenv.cfg').read_text().splitlines():
        key, sep, value = line.partition('=')
        if sep:
            values[key.strip().lower()] = value.strip()
    version = tuple(int(part) for part in values['version'].split('.')[:2])
    if version != sys.version_info[:2] or values.get('include-system-site-packages', '').lower() != 'false':
        raise ValueError('Requested venv must match this Python version and exclude system site-packages')
    packages = environment / 'lib' / ('python%d.%d' % sys.version_info[:2]) / 'site-packages'
    if not packages.is_dir():
        raise ValueError('Missing site-packages: ' + str(packages))
    # Do not execute .pth launch hooks or activate scripts containing old paths.
    # Keep Python's standard library before legacy backport packages (e.g. typing).
    sys.path.append(str(packages))
    sys.prefix = sys.exec_prefix = str(environment)
    sys.dont_write_bytecode = True
    os.environ['STANDALONE_VENV_READY'] = str(environment)
    os.environ['VIRTUAL_ENV'] = str(environment)
    os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
    os.environ.setdefault('PYDANTIC_DISABLE_PLUGINS', '__all__')
    return environment

def _prepare_model_runtime(project_root, model_id, stage='full'):
    """Retain the successful model-specific package selection, using only this copy."""
    import os
    import sys
    from pathlib import Path
    if stage == 'export':
        os.environ['CUDA_VISIBLE_DEVICES']=''
        return
    if '--help' in sys.argv or '-h' in sys.argv or '--dry-run' in sys.argv or '--check' in sys.argv:
        return
    if model_id.startswith(('ibm-granite__','IBM__')) or model_id=='shi-labs__oneformer_ade20k_dinat_large':
        for optional in ('transformer_engine','transformer_engine_extensions','msamp','flash_attn','flash_attn_2_cuda'):
            sys.modules[optional]=None
    if model_id=='shi-labs__oneformer_ade20k_dinat_large':
        if 'transformers' in sys.modules or 'natten' in sys.modules:
            raise RuntimeError('DiNAT compatibility must be configured before Transformers/NATTEN imports')
        alternate=Path(project_root)/'venv_tf/lib'/('python%d.%d'%sys.version_info[:2])/'site-packages'
        if not alternate.is_dir():raise ValueError('DiNAT requires the documented alternate Transformers/NATTEN environment')
        import numpy
        import torch
        if torch.cuda.is_available():
            torch.cuda.init()
            torch.backends.cudnn.version()
        main_site=str(Path(project_root)/'venv/lib'/('python%d.%d'%sys.version_info[:2])/'site-packages')
        position=sys.path.index(main_site) if main_site in sys.path else len(sys.path)
        sys.path.insert(position,str(alternate))


def _prepare_library_loader(environment, script):
    """Set package CUDA libraries before exec; copied venv binaries need not execute."""
    import os
    import sys
    from pathlib import Path
    if environment is None or any(flag in sys.argv for flag in ('--help','-h','--dry-run','--check')):
        return
    environment=Path(environment).resolve()
    marker=str(environment)
    if os.environ.get('SEGMENTATION_LIBRARY_ENV')==marker:return
    packages=environment/'lib'/('python%d.%d'%sys.version_info[:2])/'site-packages'
    libraries=sorted((packages/'nvidia').glob('*/lib'))
    if (packages/'torch/lib').is_dir():libraries.append(packages/'torch/lib')
    updated=os.environ.copy()
    for key in ('PYTHONHOME','PYTHONPATH','LD_PRELOAD','SEGMENTATION_TF_VENV','STANDALONE_VENV_READY'):
        updated.pop(key,None)
    updated['SEGMENTATION_LIBRARY_ENV']=marker
    updated['LD_LIBRARY_PATH']=os.pathsep.join(str(p) for p in libraries)
    updated['PYTHONDONTWRITEBYTECODE']='1'
    os.execve(sys.executable,[sys.executable,'-I','-S','-B',str(Path(script).resolve()),*sys.argv[1:]],updated)

_standalone_runtime_ns = {"__name__": "_standalone_stages", "__file__": __file__,
    "INFERENCE_ENGINE_SOURCE": '#!/usr/bin/env python3\n"""Stage 1: sequential inference to lossless prediction caches, without COCO/PNG export.\n\nExamples (use the activated venv\'s Python):\n  python run_all_inference_only.py --device auto\n  python run_all_inference_only.py --model CIDAS__clipseg-rd64-refined --device cuda\n  python run_all_inference_only.py --start-from ChongCong__Medical-SAM3\n  python run_all_inference_only.py --model apple__deeplabv3-mobilevit-xx-small --limit 2\n  python run_all_inference_only.py --cache-root /local-ssd/prediction_cache\n  python run_all_export.py --cache-root /local-ssd/prediction_cache\n  python run_all_evaluation.py --environment current\n\nOnly this script and run_all_export.py are new; existing model code is imported,\nnot rewritten. Each model has its own child process. No batch-size change is made.\nCompleted matching caches are reused; --force reruns a whole model. Incomplete\nmodels restart from sample zero (random prompt state cannot simply be skipped).\nDefault cache: Model/<name>/result/prediction_cache/. Arrays are uncompressed,\nlossless NPY members in ZIP shards (no pickle), avoiding thousands of small files.\nUse local SSD for --cache-root AND --dataset-root when the project is on FUSE.\nUncompressed caches need free disk space; retain them until CPU export succeeds.\nThe original dataset must remain unchanged between stages. Manifests are checked,\nbut replacing an image/GT in place without changing its manifest is not detected.\nResume checks shard sizes and ZIP directory fingerprints; payload CRC is checked\nwhile exporting. Use --force after changing weights or hardware. Do not run the\nold inference script on the same model concurrently: it clears result/.\n"""\nfrom __future__ import annotations\n\nimport argparse\nimport base64\nfrom contextlib import contextmanager, redirect_stderr, redirect_stdout\nfrom datetime import datetime, timezone\nimport fcntl\nimport hashlib\nimport importlib.util\nimport json\nimport os\nfrom pathlib import Path\nimport random\nimport shlex\nimport shutil\nimport signal\nimport subprocess\nimport sys\nimport tempfile\nimport time\nimport traceback\nimport uuid\nimport zipfile\n\nsys.dont_write_bytecode = True\nSCHEMA = \'deferred-predictions/v1\'\nMETRICS = (\'mIoU\', \'IoU\', \'Mask AP@[.50:.95]\', \'AP50\', \'AP75\')\n\n\ndef read_json(path):\n    def reject(value):\n        raise ValueError(f\'Non-finite JSON value: {value}\')\n    value = json.loads(Path(path).read_text(encoding=\'utf-8\'), parse_constant=reject)\n    if not isinstance(value, dict):\n        raise ValueError(f\'Expected JSON object: {path}\')\n    return value\n\n\ndef atomic_json(path, value):\n    path = Path(path)\n    path.parent.mkdir(parents=True, exist_ok=True)\n    temporary = None\n    try:\n        with tempfile.NamedTemporaryFile(mode=\'w\', encoding=\'utf-8\', dir=path.parent,\n                                         prefix=\'.\' + path.name, delete=False) as stream:\n            temporary = Path(stream.name)\n            json.dump(value, stream, ensure_ascii=False, allow_nan=False, indent=2)\n            stream.write(\'\\n\')\n        temporary.replace(path)\n    finally:\n        if temporary is not None:\n            temporary.unlink(missing_ok=True)\n\n\ndef sha256(path):\n    digest = hashlib.sha256()\n    with Path(path).open(\'rb\') as stream:\n        for block in iter(lambda: stream.read(1024 * 1024), b\'\'):\n            digest.update(block)\n    return digest.hexdigest()\n\n\ndef positive(value):\n    number = int(value)\n    if number < 1:\n        raise argparse.ArgumentTypeError(\'must be positive\')\n    return number\n\n\ndef nonnegative(value):\n    number = int(value)\n    if number < 0:\n        raise argparse.ArgumentTypeError(\'must be nonnegative\')\n    return number\n\n\ndef base_parser(description):\n    parser = argparse.ArgumentParser(description=description,\n                                     formatter_class=argparse.RawDescriptionHelpFormatter)\n    parser.add_argument(\'--project-root\', type=Path, default=Path(__file__).resolve().parent)\n    selection = parser.add_mutually_exclusive_group()\n    selection.add_argument(\'--model\', action=\'append\', help=\'Exact model directory; repeatable\')\n    selection.add_argument(\'--start-from\', help=\'Include this model and later names in sorted order\')\n    parser.add_argument(\'--dataset\', help=\'Select a supported dataset; otherwise use model default\')\n    parser.add_argument(\'--dataset-root\', type=Path, help=\'Prepared DataSet root, preferably local SSD\')\n    parser.add_argument(\'--cache-root\', type=Path, help=\'Cache root/<model>; default model/result/prediction_cache\')\n    parser.add_argument(\'--environment\', choices=[\'current\', \'auto\'], default=\'current\',\n                        help=\'Default current Python; auto allows configured existing Python\')\n    parser.add_argument(\'--threads\', type=positive, default=4)\n    parser.add_argument(\'--force\', action=\'store_true\')\n    parser.add_argument(\'--dry-run\', action=\'store_true\')\n    parser.add_argument(\'--stop-on-error\', action=\'store_true\')\n    parser.add_argument(\'--_worker\', help=argparse.SUPPRESS)\n    return parser\n\n\ndef resolve_paths(args):\n    for name in (\'project_root\', \'dataset_root\', \'cache_root\', \'bundle_root\'):\n        value = getattr(args, name, None)\n        if value is not None:\n            setattr(args, name, value.expanduser().resolve())\n    if args._worker and Path(args._worker).name != args._worker:\n        raise ValueError(\'Worker model must be a directory name\')\n\n\ndef _state_signature(model):\n    paths = [stage_result(model) / name for name in (\'results.json\', \'prediction_cache_state.json\')]\n    return tuple(sha256(path) if path.is_file() else None for path in paths)\n\n\ndef _record_launch_failure(model, stage, signature, error):\n    """Invalidate an old success when Python could not enter the worker at all."""\n    try:\n        with model_lock(model):\n            if _state_signature(model) != signature:\n                return  # The worker (or a different run) already recorded its own state.\n            run_id = \'launch_failed_\' + uuid.uuid4().hex\n            failure = dict(status=\'failed_launch\', model_id=model.name, run_id=run_id,\n                           stage=stage, error=str(error), metrics=dict.fromkeys(METRICS))\n            atomic_json(stage_result(model) / \'results.json\', failure)\n            atomic_json(stage_result(model) / \'last_attempt.json\', failure)\n            if stage == \'inference\':\n                atomic_json(stage_result(model) / \'prediction_cache_state.json\',\n                            dict(status=\'failed\', run_id=run_id, error=str(error)))\n    except (OSError, ValueError, RuntimeError) as exc:\n        print(f\'Could not record launch failure for {model.name}: {exc}\', file=sys.stderr)\n\n\ndef launch_batch(args, parser, script, stage):\n    root = args.project_root / \'Model\'\n    if not root.is_dir():\n        parser.error(f\'Model directory not found: {root}\')\n    available = {p.name: p for p in sorted(root.iterdir())\n                 if p.is_dir() and (p / \'inference.py\').is_file()}\n    requested = list(dict.fromkeys(args.model or available))\n    if args.start_from:\n        if args.start_from not in available:\n            parser.error(f\'Unknown start model: {args.start_from}\')\n        requested = requested[requested.index(args.start_from):]\n    if any(name not in available for name in requested):\n        parser.error(\'Unknown model(s): \' + \', \'.join(set(requested) - available.keys()))\n    failures, selected = [], 0\n    for name in requested:\n        signature = None\n        try:\n            signature = _state_signature(available[name])\n            config = read_json(available[name] / \'weights\' / \'model.json\')\n            if config.get(\'model_id\') != name:\n                raise ValueError(\'model_id differs from directory name\')\n            if args.dataset and args.dataset not in {d[\'name\'] for d in config[\'datasets\']}:\n                continue\n            selected += 1\n            preferred = config.get(\'python\' if stage == \'inference\' else \'evaluation_python\')\n            executable = (preferred if args.environment == \'auto\' and preferred\n                          and Path(preferred).is_file() else sys.executable)\n            command = [executable, str(script.resolve()), \'--_worker\', name]\n            for key, value in vars(args).items():\n                if key in {\'_worker\', \'model\', \'start_from\', \'dry_run\', \'stop_on_error\'} or value is None:\n                    continue\n                flag = \'--\' + key.replace(\'_\', \'-\')\n                if isinstance(value, bool):\n                    if value:\n                        command.append(flag)\n                elif isinstance(value, list):\n                    for item in value:\n                        command.extend([flag, str(item)])\n                else:\n                    command.extend([flag, str(value)])\n            print(f\'[{stage}] {name}: {shlex.join(command)}\', flush=True)\n            if args.dry_run:\n                continue\n            environment = os.environ.copy()\n            environment[\'PYTHONDONTWRITEBYTECODE\'] = \'1\'\n            for marker in (\'MODEL_RUNTIME_SELECTED\', \'UNIFIED_EVALUATION_REEXEC\'):\n                environment.pop(marker, None)\n            if stage == \'export\':\n                environment[\'CUDA_VISIBLE_DEVICES\'] = \'\'\n            elif args.environment == \'auto\':\n                preload = config.get(\'runtime_preload\')\n                if preload and Path(preload).is_file():\n                    environment[\'LD_PRELOAD\'] = \':\'.join(dict.fromkeys(filter(None, [\n                        preload, *environment.get(\'LD_PRELOAD\', \'\').split(\':\')])))\n            child = subprocess.Popen(command, cwd=args.project_root, env=environment,\n                                     start_new_session=True)\n            try:\n                code = child.wait()\n            except KeyboardInterrupt:\n                if child.poll() is None:\n                    try:\n                        os.killpg(child.pid, signal.SIGINT)\n                        child.wait(timeout=5)\n                    except subprocess.TimeoutExpired:\n                        os.killpg(child.pid, signal.SIGKILL)\n                        child.wait()\n                    except ProcessLookupError:\n                        pass\n                _record_launch_failure(available[name], stage, signature, \'Worker interrupted\')\n                return 130\n            if code:\n                _record_launch_failure(available[name], stage, signature, f\'Worker exit={code}\')\n                failures.append(name)\n                print(f\'FAILED {name} (exit={code})\', file=sys.stderr, flush=True)\n                if args.stop_on_error:\n                    break\n        except (OSError, ValueError, KeyError) as exc:\n            if signature is not None and not args.dry_run:\n                _record_launch_failure(available[name], stage, signature, exc)\n            failures.append(name)\n            print(f\'FAILED {name}: {exc}\', file=sys.stderr, flush=True)\n            if args.stop_on_error:\n                break\n    if not selected and not failures:\n        parser.error(\'No models match the selection\')\n    print(f\'{stage}: selected={selected}, failed={len(failures)}\' + (\' (dry run)\' if args.dry_run else \'\'))\n    return 1 if failures else 0\n\n\ndef cache_directory(model, args):\n    return (args.cache_root / model.name if args.cache_root else stage_result(model) / \'prediction_cache\')\n\n\n@contextmanager\ndef model_lock(model):\n    result = stage_result(model)\n    if result.is_symlink():\n        raise ValueError(f\'Result directory must not be a symlink: {result}\')\n    result.mkdir(parents=True, exist_ok=True)\n    descriptor = os.open(result / \'.two_stage.lock\', os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)\n    try:\n        try:\n            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)\n        except BlockingIOError as exc:\n            raise RuntimeError(f\'Another two-stage worker is using {model.name}\') from exc\n        yield\n    finally:\n        os.close(descriptor)\n\n\ndef setup_environment(model, args, cpu=False):\n    cache = stage_result(model) / \'.cache\'\n    if cache.is_symlink():\n        raise ValueError(\'Result runtime cache must not be a symlink\')\n    for key, sub in [(\'HF_MODULES_CACHE\', \'hf_modules\'), (\'HF_HOME\', \'huggingface\'),\n                     (\'MPLCONFIGDIR\', \'matplotlib\'), (\'YOLO_CONFIG_DIR\', \'ultralytics\')]:\n        target = cache / sub\n        target.mkdir(parents=True, exist_ok=True)\n        os.environ[key] = str(target)\n    os.environ.update(HF_HUB_OFFLINE=\'1\', TRANSFORMERS_OFFLINE=\'1\',\n                      YOLO_OFFLINE=\'true\', YOLO_AUTOINSTALL=\'false\',\n                      OMP_NUM_THREADS=str(args.threads), MKL_NUM_THREADS=str(args.threads))\n    if cpu:\n        os.environ[\'CUDA_VISIBLE_DEVICES\'] = \'\'\n\n\ndef load_model_module(model):\n    name = \'_deferred_model_\' + hashlib.sha256(str(model).encode()).hexdigest()[:16]\n    spec = importlib.util.spec_from_file_location(name, model / \'inference.py\')\n    module = importlib.util.module_from_spec(spec)\n    sys.modules[name] = module\n    spec.loader.exec_module(module)\n    return module\n\n\ndef _zip_index(archive):\n    rows = [(info.filename, info.file_size, info.CRC) for info in archive.infolist()]\n    if len({row[0] for row in rows}) != len(rows):\n        raise ValueError(\'Duplicate archive members\')\n    return hashlib.sha256(json.dumps(rows, separators=(\',\', \':\')).encode()).hexdigest()\n\n\nclass CacheWriter:\n    """Bound memory to one prediction; write a few large, uncompressed ZIP shards."""\n\n    def __init__(self, directory, shard_bytes=256 * 1024 * 1024):\n        self.directory, self.shard_bytes = Path(directory), shard_bytes\n        self.archive, self.shards = None, []\n        self.count, self.shard_count, self.array_number = 0, 0, 0\n        self.sample_ids = set()\n\n    def __enter__(self):\n        self.directory.mkdir(parents=True, exist_ok=True)\n        return self\n\n    def _encode(self, value, prefix):\n        import numpy as np\n        if isinstance(value, np.ndarray):\n            if value.dtype.hasobject:\n                raise TypeError(\'Object arrays cannot be cached without pickle\')\n            member = f\'{prefix}/array_{self.array_number:06d}.npy\'\n            self.array_number += 1\n            with self.archive.open(member, \'w\', force_zip64=True) as stream:\n                np.save(stream, value, allow_pickle=False)\n            return {\'type\': \'array\', \'member\': member}\n        if isinstance(value, np.generic):\n            return self._encode(value.item(), prefix)\n        if isinstance(value, dict):\n            return {\'type\': \'dict\', \'items\': [[self._encode(k, prefix), self._encode(v, prefix)]\n                                             for k, v in value.items()]}\n        if isinstance(value, (list, tuple)):\n            return {\'type\': \'tuple\' if isinstance(value, tuple) else \'list\',\n                    \'items\': [self._encode(item, prefix) for item in value]}\n        if isinstance(value, bytes):\n            return {\'type\': \'bytes\', \'value\': base64.b64encode(value).decode(\'ascii\')}\n        if isinstance(value, Path):\n            return str(value)\n        if value is None or isinstance(value, (bool, int, float, str)):\n            return value\n        raise TypeError(f\'Unsupported prediction cache value: {type(value).__name__}\')\n\n    def add(self, record):\n        if record[\'index\'] != self.count or record[\'sample_id\'] in self.sample_ids:\n            raise ValueError(\'Cache records must have unique IDs and contiguous indices\')\n        if self.archive is None:\n            self.path = self.directory / f\'predictions_{len(self.shards):05d}.zip\'\n            self.archive = zipfile.ZipFile(self.path, \'x\', compression=zipfile.ZIP_STORED,\n                                           allowZip64=True)\n            self.shard_count = 0\n        prefix = f\'samples/{self.count:08d}\'\n        self.array_number = 0\n        encoded = self._encode(record, prefix)\n        self.archive.writestr(prefix + \'/record.json\', json.dumps(encoded, allow_nan=False))\n        self.count += 1\n        self.shard_count += 1\n        self.sample_ids.add(record[\'sample_id\'])\n        if self.archive.fp.tell() >= self.shard_bytes:\n            self._close_shard()\n\n    def _close_shard(self):\n        if self.archive is not None:\n            fingerprint = _zip_index(self.archive)\n            self.archive.close()\n            self.archive = None\n            self.shards.append({\'name\': self.path.name, \'bytes\': self.path.stat().st_size,\n                                \'count\': self.shard_count, \'index_sha256\': fingerprint})\n\n    def finish(self):\n        self._close_shard()\n        return self.shards\n\n    def __exit__(self, *exc):\n        self._close_shard()\n\n\ndef read_cache(cache):\n    cache = Path(cache)\n    manifest = read_json(cache / \'manifest.json\')\n    if manifest.get(\'schema_version\') != SCHEMA or manifest.get(\'status\') != \'complete\':\n        raise ValueError(\'Prediction cache is not complete or has an unknown schema\')\n    count = manifest.get(\'dataset\', {}).get(\'count\')\n    if type(count) is not int or count < 1:\n        raise ValueError(\'Invalid cache sample count\')\n    shards = manifest.get(\'shards\')\n    if not isinstance(shards, list) or not shards:\n        raise ValueError(\'Missing prediction shards\')\n    observed = 0\n    for index, shard in enumerate(shards):\n        name = f\'predictions_{index:05d}.zip\'\n        path = cache / name\n        if shard.get(\'name\') != name or path.is_symlink() or not path.is_file():\n            raise ValueError(f\'Missing or invalid prediction shard: {name}\')\n        if path.stat().st_size != shard.get(\'bytes\'):\n            raise ValueError(f\'Prediction shard size changed: {name}\')\n        if type(shard.get(\'count\')) is not int or shard[\'count\'] < 1:\n            raise ValueError(\'Invalid shard count\')\n        with zipfile.ZipFile(path) as archive:\n            if _zip_index(archive) != shard.get(\'index_sha256\'):\n                raise ValueError(f\'Prediction shard index changed: {name}\')\n            records = [entry for entry in archive.namelist() if entry.endswith(\'/record.json\')]\n            expected = [f\'samples/{i:08d}/record.json\'\n                        for i in range(observed, observed + shard[\'count\'])]\n            if records != expected:\n                raise ValueError(\'Missing or unordered prediction records\')\n        observed += shard[\'count\']\n    if observed != count:\n        raise ValueError(\'Prediction cache coverage is incomplete\')\n    return manifest\n\n\ndef _decode(archive, value, prefix, consumed):\n    import numpy as np\n    if not isinstance(value, dict):\n        return value\n    kind = value.get(\'type\')\n    if kind == \'array\':\n        member = value[\'member\']\n        if not member.startswith(prefix + \'/array_\') or not member.endswith(\'.npy\') or member in consumed:\n            raise ValueError(\'Invalid or duplicate cached array reference\')\n        consumed.add(member)\n        with archive.open(member) as stream:\n            array = np.load(stream, allow_pickle=False)\n            if stream.read(1):\n                raise ValueError(\'Trailing bytes in cached NPY array\')\n        return array\n    if kind == \'dict\':\n        return {_decode(archive, k, prefix, consumed): _decode(archive, v, prefix, consumed)\n                for k, v in value[\'items\']}\n    if kind in (\'tuple\', \'list\'):\n        items = [_decode(archive, item, prefix, consumed) for item in value[\'items\']]\n        return tuple(items) if kind == \'tuple\' else items\n    if kind == \'bytes\':\n        return base64.b64decode(value[\'value\'], validate=True)\n    raise ValueError(f\'Unknown cached value type: {kind}\')\n\n\ndef iter_records(cache, manifest):\n    count, identifiers = 0, set()\n    for shard in manifest[\'shards\']:\n        with zipfile.ZipFile(Path(cache) / shard[\'name\']) as archive:\n            if _zip_index(archive) != shard[\'index_sha256\']:\n                raise ValueError(\'Cache shard index changed during export\')\n            consumed = set()\n            for _ in range(shard[\'count\']):\n                prefix = f\'samples/{count:08d}\'\n                name = prefix + \'/record.json\'\n                consumed.add(name)\n                value = _decode(archive, json.loads(archive.read(name)), prefix, consumed)\n                if value[\'index\'] != count or value[\'sample_id\'] in identifiers:\n                    raise ValueError(\'Cached prediction indices/IDs are inconsistent\')\n                identifiers.add(value[\'sample_id\'])\n                yield value\n                count += 1\n            if set(archive.namelist()) != consumed:\n                raise ValueError(\'Unreferenced cache members\')\n    if count != manifest[\'dataset\'][\'count\']:\n        raise ValueError(\'Prediction cache sample count mismatch\')\n\n\nclass CacheOutput:\n    def __init__(self, destination, model_id):\n        self.destination, self.model_id = destination, model_id\n\n    def __enter__(self):\n        if self.destination.is_symlink():\n            raise ValueError(\'Prediction cache must not be a symlink\')\n        self.destination.parent.mkdir(parents=True, exist_ok=True)\n        self.container = Path(tempfile.mkdtemp(prefix=\'.prediction-cache-\', dir=self.destination.parent))\n        self.staging = self.container / \'staging\'\n        self.staging.mkdir()\n        return self\n\n    def publish(self):\n        previous = self.container / \'previous\'\n        if self.destination.exists():\n            existing = read_json(self.destination / \'manifest.json\')\n            if existing.get(\'schema_version\') != SCHEMA or existing.get(\'model_id\') != self.model_id:\n                raise ValueError(\'Refusing to replace unrelated cache data\')\n            self.destination.rename(previous)\n        try:\n            self.staging.rename(self.destination)\n        except BaseException:\n            if previous.exists():\n                previous.rename(self.destination)\n            raise\n\n    def __exit__(self, *exc):\n        # Keep previous data recoverable if rollback itself was interrupted.\n        if (self.container / \'previous\').exists() and not self.destination.exists():\n            return\n        shutil.rmtree(self.container)\n\n\ndef make_request(model, config, args):\n    entries = config[\'datasets\']\n    selection = next((d for d in entries if d[\'name\'] == args.dataset), None) if args.dataset else entries[0]\n    if selection is None:\n        raise ValueError(f\'Unknown dataset {args.dataset}\')\n    dataset_root = (args.dataset_root or model / config[\'dataset_root\']).resolve()\n    data = dataset_root if (dataset_root / \'evaluation_samples.jsonl\').is_file() else dataset_root / selection[\'name\']\n    request = dict(config_sha256=sha256(model / \'weights\' / \'model.json\'),\n                   inference_sha256=sha256(model / \'inference.py\'),\n                   source_manifest_sha256=sha256(data / \'evaluation_samples.jsonl\'),\n                   dataset_manifest_sha256=sha256(data / \'dataset_manifest.json\')\n                   if (data / \'dataset_manifest.json\').is_file() else None,\n                   dataset_root=str(dataset_root), dataset=selection[\'name\'],\n                   split=args.split or selection.get(\'split\'),\n                   seed=args.seed if args.seed is not None else config.get(\'seed\', 42),\n                   precision=args.precision or (\'bf16\' if config.get(\'adapter\') == \'medical_sam3\' else \'fp32\'),\n                   device=args.device, threads=args.threads, warmup=args.warmup, limit=args.limit,\n                   environment=args.environment)\n    effective = {**config, **selection.get(\'protocol\', {}), \'cpu_threads\': args.threads}\n    return request, effective\n\n\ndef choose_device(config, requested):\n    if requested != \'auto\':\n        return requested\n    if config.get(\'adapter\') == \'gguf\':\n        return \'cpu\'\n    if config.get(\'adapter\') in {\'keras_pets\', \'teeth\'}:\n        import tensorflow as tf\n        return \'cuda\' if tf.config.list_physical_devices(\'GPU\') else \'cpu\'\n    try:\n        import torch\n        return \'cuda\' if torch.cuda.is_available() else \'cpu\'\n    except ImportError:\n        return \'cpu\'\n\n\ndef seed_all(seed):\n    import numpy as np\n    random.seed(seed)\n    np.random.seed(seed)\n    if \'torch\' in sys.modules:\n        sys.modules[\'torch\'].manual_seed(seed)\n\n\ndef restore_cache_status(model, cache, manifest):\n    """A reused cache must not leave another inference run selected for evaluation."""\n    result = stage_result(model)\n    try:\n        current = read_json(result / \'results.json\')\n        metadata = current.get(\'metadata\') or {}\n        same_run = (current.get(\'run_id\') or metadata.get(\'run_id\')) == manifest[\'run_id\']\n        same_model = (current.get(\'model_id\') or metadata.get(\'model_id\') or model.name) == model.name\n    except (OSError, ValueError, AttributeError):\n        current, same_run, same_model = {}, False, False\n    if not (same_run and same_model and current.get(\'status\') in {\'cached\', \'complete\', \'exported\'}):\n        record = dict(status=\'cached\', model_id=model.name, run_id=manifest[\'run_id\'],\n                      prediction_cache=str(cache), metrics=dict.fromkeys(METRICS),\n                      environment=manifest[\'environment\'], timing=manifest[\'timing\'],\n                      scope=manifest[\'dataset\'][\'scope\'],\n                      next_step=\'inference.py --stage export, then evaluate.py\')\n        atomic_json(result / \'results.json\', record)\n        atomic_json(result / \'last_attempt.json\', record)\n        atomic_json(result / \'environment.json\', manifest[\'environment\'])\n        atomic_json(result / \'timing.json\', manifest[\'timing\'])\n    atomic_json(result / \'prediction_cache_state.json\',\n                dict(status=\'complete\', run_id=manifest[\'run_id\'], cache=str(cache)))\n\n\ndef inference_worker(args):\n    model = args.project_root / \'Model\' / args._worker\n    with model_lock(model):\n        result = stage_result(model)\n        cache = cache_directory(model, args)\n        run_id = datetime.now(timezone.utc).strftime(\'%Y%m%dT%H%M%S_%fZ\') + \'_\' + uuid.uuid4().hex[:6]\n        started = time.perf_counter()\n        state = {\'status\': \'running\', \'run_id\': run_id, \'cache\': str(cache)}\n        timed_adapter = None\n        try:\n            config = read_json(model / \'weights\' / \'model.json\')\n            request, effective = make_request(model, config, args)\n            try:\n                previous = read_cache(cache)\n                marker = read_json(result / \'prediction_cache_state.json\') if (result / \'prediction_cache_state.json\').exists() else {}\n                reusable = (previous[\'model_id\'] == model.name and previous[\'request\'] == request\n                            and (not marker or (marker.get(\'status\') == \'complete\'\n                                 and marker.get(\'run_id\') == previous[\'run_id\'])))\n            except (OSError, ValueError, KeyError, zipfile.BadZipFile):\n                reusable = False\n            if reusable and not args.force:\n                restore_cache_status(model, cache, previous)\n                print(f\'SKIP {model.name}: matching complete prediction cache (no model loaded)\', flush=True)\n                return 0\n            atomic_json(result / \'prediction_cache_state.json\', state)\n            atomic_json(result / \'results.json\', {\'status\': \'inferring\', \'model_id\': model.name,\n                                                 \'run_id\': run_id, \'metrics\': dict.fromkeys(METRICS)})\n            setup_environment(model, args)\n            module = load_model_module(model)\n            with (result / \'inference_only.log\').open(\'w\', encoding=\'utf-8\') as log, \\\n                    redirect_stdout(module.LogStream(sys.stdout, log)), \\\n                    redirect_stderr(module.LogStream(sys.stderr, log)):\n                print(f\'[runtime] {sys.executable}\', flush=True)\n                seed_all(request[\'seed\'])\n                device = choose_device(config, request[\'device\'])\n                if device == \'cpu\' and request[\'precision\'] != \'fp32\':\n                    raise ValueError(\'CPU requires fp32; specify --precision fp32\')\n                module.configure_threads(args.threads)\n                seed_all(request[\'seed\'])\n                environment = module.capture_environment(device, request[\'precision\'])\n                environment[\'actual_device\'] = device\n                atomic_json(result / \'environment.json\', environment)\n                dataset = module.load_dataset(Path(request[\'dataset_root\']), request[\'dataset\'], request[\'split\'], effective)\n                count = min(len(dataset), args.limit) if args.limit else len(dataset)\n                effective.update(dataset_name=request[\'dataset\'], categories=dataset.categories)\n                source = (model / config[\'source_root\']).resolve()\n                atomic_json(result / \'run_config.json\', {\'config\': effective, \'arguments\': request,\n                            \'seed\': request[\'seed\'], \'source_root\': str(source), \'dataset_root\': request[\'dataset_root\']})\n                begin = time.perf_counter()\n                adapter = module.create_adapter(effective, source, device, request[\'precision\'])\n                if hasattr(module, \'_ForwardTimedAdapter\'):\n                    timed_adapter = module._ForwardTimedAdapter(adapter, effective, device)\n                    adapter = timed_adapter\n                module.configure_threads(args.threads)\n                model_load = time.perf_counter() - begin\n                begin = time.perf_counter()\n                first = dataset.load(0)\n                data_seconds = time.perf_counter() - begin\n                begin = time.perf_counter()\n                for _ in range(args.warmup):\n                    adapter.predict(first)\n                    module._pipeline_synchronize(device)\n                warmup_seconds = time.perf_counter() - begin\n                if timed_adapter is not None:\n                    timed_adapter.reset_forward_timing()\n                seed_all(request[\'seed\'])\n                pipelines, forwards, cache_seconds = [], [], 0.0\n                dataset_info = dict(name=request[\'dataset\'], splits=dataset.splits,\n                                    scope=\'subset\' if count < len(dataset) else \'full\',\n                                    count=count, full_count=len(dataset), categories=dataset.categories,\n                                    task=dataset.task, ap_supported=dataset.ap_supported,\n                                    ap_reason=dataset.ap_reason, manifest_status=dataset.manifest.get(\'status\'))\n                with CacheOutput(cache, model.name) as output:\n                    with CacheWriter(output.staging, args.shard_mib * 1024 * 1024) as writer:\n                        for index in range(count):\n                            begin = time.perf_counter()\n                            sample = first if index == 0 else dataset.load(index)\n                            data_seconds += time.perf_counter() - begin\n                            module._pipeline_synchronize(device)\n                            begin = time.perf_counter()\n                            prediction = adapter.predict(sample)\n                            module._pipeline_synchronize(device)\n                            pipelines.append(time.perf_counter() - begin)\n                            forwards.append(prediction.forward_seconds)\n                            begin = time.perf_counter()\n                            writer.add({\'index\': index, \'sample_id\': sample.sample_id,\n                                        \'pipeline_seconds\': pipelines[-1],\n                                        \'prediction\': {\'semantic\': prediction.semantic,\n                                                       \'instances\': prediction.instances,\n                                                       \'forward_seconds\': prediction.forward_seconds,\n                                                       \'metadata\': prediction.metadata}})\n                            cache_seconds += time.perf_counter() - begin\n                            if index == 0 or (index + 1) % 25 == 0 or index + 1 == count:\n                                print(f\'[{model.name}] inferred {index + 1}/{count}; \'\n                                      f\'pipeline={sum(pipelines):.1f}s data={data_seconds:.1f}s \'\n                                      f\'cache={cache_seconds:.1f}s\', flush=True)\n                        begin = time.perf_counter()\n                        shards = writer.finish()\n                        cache_seconds += time.perf_counter() - begin\n                    timing = module.timing_summary(forwards, pipelines, count)\n                    if timed_adapter is not None:\n                        timing.update(timed_adapter.forward_timing_metadata(count))\n                    timing.update(warmup_samples=args.warmup, warmup_seconds=warmup_seconds,\n                                  model_load_seconds=model_load, data_loading_seconds=data_seconds,\n                                  prediction_cache_seconds=cache_seconds, batch_size_source_samples=1,\n                                  inference_and_cache_seconds=time.perf_counter() - started)\n                    environment = module.capture_environment(device, request[\'precision\'])\n                    environment[\'actual_device\'] = device\n                    if timed_adapter is not None:\n                        environment[\'backend_execution\'] = timed_adapter.forward_execution_metadata()\n                    environment[\'adapter_execution\'] = {key: module.json_metadata(prediction.metadata[key])\n                        for key in (\'actual_precision\', \'precision\', \'providers\', \'runtime\', \'backend\', \'timing_scope\', \'forward_scope\')\n                        if key in prediction.metadata}\n                    manifest = dict(schema_version=SCHEMA, status=\'complete\', model_id=model.name,\n                                    run_id=run_id, request=request, config=effective, dataset=dataset_info,\n                                    environment=environment, timing=timing, shards=shards)\n                    atomic_json(output.staging / \'manifest.json\', manifest)\n                    read_cache(output.staging)\n                    output.publish()\n                atomic_json(result / \'environment.json\', environment)\n                atomic_json(result / \'timing.json\', timing)\n                state[\'status\'] = \'complete\'\n                atomic_json(result / \'prediction_cache_state.json\', state)\n                completed = dict(status=\'cached\', model_id=model.name, run_id=run_id,\n                                 prediction_cache=str(cache), metrics=dict.fromkeys(METRICS),\n                                 environment=environment, timing=timing, scope=dataset_info[\'scope\'],\n                                 next_step=\'inference.py --stage export, then evaluate.py\')\n                atomic_json(result / \'results.json\', completed)\n                atomic_json(result / \'last_attempt.json\', completed)\n                for path in (result / \'inference_time.json\', stage_result(model) / \'inference_time.json\'):\n                    module.update_timing_index(path, {\'model_id\': model.name, \'run_id\': run_id,\n                        \'status\': \'cached\', \'dataset\': request[\'dataset\'], \'scope\': dataset_info[\'scope\'],\n                        \'timing\': timing, \'run_directory\': str(result)})\n                (result / \'inference_only_error.log\').unlink(missing_ok=True)\n                print(f\'CACHED {model.name}: {count} samples -> {cache}\', flush=True)\n            return 0\n        except BaseException as exc:\n            state.update(status=\'failed\', error=str(exc))\n            atomic_json(result / \'prediction_cache_state.json\', state)\n            failure = dict(status=\'failed_inference\', model_id=model.name, run_id=run_id,\n                           metrics=dict.fromkeys(METRICS), error=str(exc), error_type=type(exc).__name__)\n            atomic_json(result / \'results.json\', failure)\n            atomic_json(result / \'last_attempt.json\', failure)\n            (result / \'inference_only_error.log\').write_text(traceback.format_exc(), encoding=\'utf-8\')\n            print(f\'FAILED {model.name}: {exc}\\nDetails: {result / "inference_only_error.log"}\', file=sys.stderr)\n            return 130 if isinstance(exc, KeyboardInterrupt) else 1\n        finally:\n            if timed_adapter is not None:\n                timed_adapter.close()\n\n\ndef main():\n    parser = base_parser(__doc__)\n    parser.add_argument(\'--device\', default=\'auto\', help=\'auto, cpu, cuda, or cuda:N\')\n    parser.add_argument(\'--precision\', choices=[\'fp32\', \'fp16\', \'bf16\'])\n    parser.add_argument(\'--limit\', type=positive)\n    parser.add_argument(\'--split\', action=\'append\')\n    parser.add_argument(\'--seed\', type=int)\n    parser.add_argument(\'--warmup\', type=nonnegative, default=1)\n    parser.add_argument(\'--shard-mib\', type=positive, default=256,\n                        help=\'Rotate cache after a whole sample reaches this size (default 256 MiB)\')\n    args = parser.parse_args()\n    resolve_paths(args)\n    if args._worker:\n        return inference_worker(args)\n    return launch_batch(args, parser, Path(__file__), \'inference\')\n\n\n\n',
    "EXPORT_ENGINE_SOURCE": '#!/usr/bin/env python3\n"""Export completed prediction caches serially on the CPU, without inference.\n\nExamples::\n\n    python run_all_export.py\n    python run_all_export.py --model MODEL --bundle-root /data/Dataset_Coco\n    python run_all_export.py --cache-root /data/predictions --dataset-root /data/DataSet\n    python run_all_export.py --prediction-images labels\n    python run_all_export.py --export-mode full\n\nRun run_all_inference_only.py first. This step still needs the original prepared\ndatasets: it reloads each image and target, checks the dataset manifests, and\npasses cached predictions through the model\'s original COCO exporter.\nThe default --export-mode json writes only the four evaluation JSON payloads\nand their manifest. --prediction-images none|labels|preview|both (default none)\noptionally adds prediction masks/previews with a separate checked image index.\nLabels preserve native category IDs or individual instance masks; previews mark\nignored pixels magenta. No original images, ground-truth pictures or probability\nNPZ files are written in JSON mode. Existing probability outputs are retained.\n--export-mode full retains the original complete image export and requires\n--prediction-images none. Previously completed full bundles are preserved and\nreused by default; --force rebuilds from the same validated prediction cache,\nwithout inference. New bundles are reused only for matching export options.\nWorkers hide CUDA devices and never load model weights or run evaluation.\nThe resulting bundles can be evaluated with the unchanged run_all_evaluation.py.\nInference environment and timing describe the original inference process;\nexport_environment and export_seconds describe this separate CPU process.\n\nModels remain serial. Within one model, --io-workers (default 2) overlaps file\nwrites, and --verify-workers (default 2) overlaps complete image readback checks.\n--queue-mib (default 64) bounds pending encoded image bytes; the current image,\nsource arrays and each worker\'s 1 MiB copy buffer are additional memory. A single\nimage exceeding that budget is written synchronously after draining the queue.\nAll writes and full integrity checks finish before the bundle is published.\nInterrupted models restart from sample zero; this is not sample-level resume.\n"""\nfrom __future__ import annotations\n\nfrom collections import deque\nfrom concurrent.futures import ThreadPoolExecutor\nimport copy\nimport hashlib\nimport io\nimport json\nimport math\nimport os\nfrom pathlib import Path\nimport platform\nimport sys\nimport tarfile\nimport time\nimport traceback\n\nsys.dont_write_bytecode = True\n\n\n\n\nMETRICS = (\'mIoU\', \'IoU\', \'Mask AP@[.50:.95]\', \'AP50\', \'AP75\')\n\n\ndef _export_options(args):\n    mode = getattr(args, \'export_mode\', \'json\')\n    pictures = getattr(args, \'prediction_images\', \'none\')\n    if mode not in (\'json\', \'full\') or pictures not in (\'none\', \'labels\', \'preview\', \'both\'):\n        raise ValueError(\'Unknown export mode or prediction image selection\')\n    if mode == \'full\' and pictures != \'none\':\n        raise ValueError(\'--export-mode full already includes all pictures; use --prediction-images none\')\n    return mode, pictures\n\n\ndef _write_encoded(path, data, digest):\n    started = time.perf_counter()\n    with path.open(\'wb\') as stream:\n        stream.write(data)\n    return digest, len(data), time.perf_counter() - started\n\n\ndef _copy_original(source, destination, member_name=None):\n    """Copy exact source bytes, hashing the same stream without a readback."""\n    started = time.perf_counter()\n    digest, count = hashlib.sha256(), 0\n\n    def copy_stream(stream):\n        nonlocal count\n        with destination.open(\'wb\') as output:\n            for block in iter(lambda: stream.read(1024 * 1024), b\'\'):\n                digest.update(block)\n                output.write(block)\n                count += len(block)\n\n    if member_name:\n        with tarfile.open(source) as archive:\n            member = archive.getmember(member_name)\n            if not member.isfile():\n                raise ValueError(f\'Original image archive member is not a regular file: {member_name}\')\n            with archive.extractfile(member) as stream:\n                copy_stream(stream)\n    else:\n        with source.open(\'rb\') as stream:\n            copy_stream(stream)\n    return digest.hexdigest(), count, time.perf_counter() - started\n\n\nclass _ExportIO:\n    """Bounded file I/O; only the calling thread changes bundle metadata/IDs.\n\n    Enter inside BundleOutput, so workers finish before staging cleanup, even\n    when a producer, write or validation fails. Model files are never changed.\n    """\n\n    def __init__(self, module, output, args):\n        self.module, self.output = module, Path(output)\n        self.workers = getattr(args, \'io_workers\', 2)\n        self.byte_limit = getattr(args, \'queue_mib\', 64) * 1024 * 1024\n        self.file_limit = max(2, self.workers * 2)\n        self.pending, self.pending_bytes = deque(), 0\n        self.directories = set()\n        self.original_class = module.ImageArtifacts\n        self.executor = None\n        self.stats = {\n            \'io_workers\': self.workers, \'pending_bytes_limit\': self.byte_limit,\n            \'pending_files_limit\': self.file_limit, \'peak_pending_bytes\': 0,\n            \'files_written\': 0, \'bytes_written\': 0,\n            \'encoding_seconds\': 0.0, \'queue_wait_seconds\': 0.0,\n            \'write_worker_seconds_sum\': 0.0,\n        }\n\n    def _directory(self, path):\n        if path not in self.directories:\n            path.mkdir(parents=True, exist_ok=True)\n            self.directories.add(path)\n\n    def _record(self, result, hashes, relative):\n        digest, size, seconds = result\n        hashes[relative] = digest\n        self.stats[\'files_written\'] += 1\n        self.stats[\'bytes_written\'] += size\n        self.stats[\'write_worker_seconds_sum\'] += seconds\n\n    def _drain_one(self):\n        future, size, hashes, relative = self.pending.popleft()\n        started = time.perf_counter()\n        try:\n            self._record(future.result(), hashes, relative)\n        finally:\n            self.pending_bytes -= size\n            self.stats[\'queue_wait_seconds\'] += time.perf_counter() - started\n\n    def submit(self, function, arguments, hashes, relative, size=0):\n        while self.pending and self.pending[0][0].done():\n            self._drain_one()\n        if size > self.byte_limit:\n            self.flush()\n            self._record(function(*arguments), hashes, relative)\n            return\n        while self.pending and (len(self.pending) >= self.file_limit\n                                or self.pending_bytes + size > self.byte_limit):\n            self._drain_one()\n        future = self.executor.submit(function, *arguments)\n        self.pending.append((future, size, hashes, relative))\n        self.pending_bytes += size\n        self.stats[\'peak_pending_bytes\'] = max(self.stats[\'peak_pending_bytes\'], self.pending_bytes)\n\n    def flush(self):\n        failure = None\n        while self.pending:\n            try:\n                self._drain_one()\n            except Exception as exc:\n                if failure is None:\n                    failure = exc\n        if failure is not None:\n            raise failure\n\n    def __enter__(self):\n        owner = self\n\n        class BufferedImageArtifacts(self.original_class):\n            def _save(self, relative, image):\n                path = self.output / relative\n                owner._directory(path.parent)\n                started = time.perf_counter()\n                image_format = owner.module.Image.registered_extensions()[path.suffix.lower()]\n                with io.BytesIO() as buffer:\n                    image.save(buffer, format=image_format)\n                    data = buffer.getvalue()\n                digest = hashlib.sha256(data).hexdigest()\n                owner.stats[\'encoding_seconds\'] += time.perf_counter() - started\n                owner.submit(_write_encoded, (path, data, digest), self.hashes, relative, len(data))\n                return relative\n\n            def _originals(self, image_id, image, metadata):\n                assets = [a for a in metadata.get(\'source_assets\', []) if a.get(\'role\') in {\n                    \'image\', \'image_sequence\', \'image_and_mask_float32\', \'ed_image\', \'es_image\'}]\n                root = Path(metadata.get(\'dataset_root\', \'.\'))\n                if not assets and metadata.get(\'image_path\'):\n                    assets = [{\'role\': \'image\', \'path\': metadata[\'image_path\']}]\n                key = json.dumps([str(root), assets], sort_keys=True) if assets else str(image_id)\n                if key in self._originals_cache:\n                    return self._originals_cache[key]\n                result = []\n                source_key = hashlib.sha256(key.encode()).hexdigest()[:16]\n                for index, asset in enumerate(assets):\n                    source = root / asset[\'path\']\n                    member_name = asset.get(\'archive_member\')\n                    filename = Path(member_name or asset[\'path\']).name\n                    suffix = \'.nii.gz\' if filename.lower().endswith(\'.nii.gz\') else Path(filename).suffix\n                    if len(suffix.encode()) > 16:\n                        suffix = \'.bin\'\n                    relative = f\'images/original/{source_key}_{index}{suffix}\'\n                    path = self.output / relative\n                    owner._directory(path.parent)\n                    owner.submit(_copy_original, (source, path, member_name), self.hashes, relative)\n                    result.append({\'path\': relative, \'role\': asset[\'role\'], \'source_path\': str(source),\n                                   \'original_filename\': filename, \'archive_member\': member_name,\n                                   \'copy\': \'original bytes\'})\n                if not result:\n                    relative = self._save(f\'images/original/{image_id:08d}.png\', image)\n                    result.append({\'path\': relative, \'copy\': \'decoded input preview; source file was not provided\'})\n                self._originals_cache[key] = result\n                return result\n\n        self.executor = ThreadPoolExecutor(max_workers=self.workers, thread_name_prefix=\'export-write\')\n        self.module.ImageArtifacts = BufferedImageArtifacts\n        return self\n\n    def __exit__(self, exception_type, exception, traceback):\n        try:\n            if exception_type is None:\n                self.flush()\n        finally:\n            try:\n                self.executor.shutdown(wait=True, cancel_futures=True)\n            finally:\n                self.module.ImageArtifacts = self.original_class\n        return False\n\n\nclass _PredictionImages:\n    """Optional attachments, independent of the model\'s complete image bundle."""\n\n    def __init__(self, module, writer, selection):\n        owner = self\n        self.writer, self.selection = writer, selection\n        self.has_binary_mask = False\n\n        class SelectedMasks(module.ImageArtifacts):\n            def _save(self, relative, image):\n                parts = Path(relative).parts\n                labels = (\'labels\' in parts or \'instances\' in parts or\n                          (owner.has_binary_mask and \'preview\' not in parts))\n                if (labels and selection not in (\'labels\', \'both\') or\n                        not labels and selection not in (\'preview\', \'both\')):\n                    return None\n                return super()._save(relative, image)\n\n        manifest = writer.manifest\n        self.artifacts = SelectedMasks(writer.output, manifest[\'categories\'],\n                                       manifest[\'task\'], manifest[\'foreground_id\'])\n        self.index = {\'version\': 1, \'selection\': selection, \'images\': [],\n                      \'sha256\': self.artifacts.hashes,\n                      \'category_colors\': self.artifacts.description()[\'category_colors\'],\n                      \'valid_pixels\': \'Use each JSON image valid_mask RLE; previews mark ignored pixels magenta.\'}\n        manifest[\'prediction_images\'] = self.index\n\n    def add(self, unit):\n        import numpy as np\n\n        image = self.writer.images[-1]\n        valid = unit.get(\'valid_mask\')\n        if valid is None:\n            valid = np.ones((image[\'height\'], image[\'width\']), bool)\n        else:\n            valid = np.asarray(valid, bool)\n        instances = unit.get(\'pred_instances\') or []\n        if self.writer.manifest[\'class_agnostic\']:\n            category = self.writer.manifest[\'categories\'][0][\'id\']\n            instances = [{**item, \'category_id\': category} for item in instances]\n        self.has_binary_mask = self.artifacts.task == \'binary\' and unit[\'prediction\'] is not None\n        artifacts = self.artifacts._mask_images(\n            \'predictions\', f\'{image["id"]:08d}\', unit[\'prediction\'], instances, valid)\n        artifacts = {key: value for key, value in artifacts.items() if value is not None}\n        if self.selection == \'preview\':\n            for key in (\'instances\', \'encoding\', \'instance_preview_note\'):\n                artifacts.pop(key, None)\n        self.index[\'images\'].append({\n            \'image_id\': image[\'id\'], \'sample_id\': image[\'sample_id\'],\n            \'width\': image[\'width\'], \'height\': image[\'height\'], \'artifacts\': artifacts,\n        })\n\n\ndef _verify_artifact_files(path, hashes, workers, dimensions=None, module=None):\n    resolved_root = path.resolve()\n\n    def check_image(name, expected):\n        if not isinstance(name, str) or not name or not isinstance(expected, str):\n            raise ValueError(\'invalid image artifact hash entry\')\n        candidate = path / name\n        if Path(name).is_absolute() or not candidate.resolve().is_relative_to(resolved_root):\n            raise ValueError(f\'invalid image artifact path: {name}\')\n        if not candidate.is_file() or sha256(candidate) != expected:\n            raise ValueError(f\'image artifact missing or hash mismatch: {name}\')\n        if dimensions is not None:\n            with module.Image.open(candidate) as image:\n                if image.size != dimensions[name]:\n                    raise ValueError(f\'prediction image dimensions differ: {name}\')\n                image.verify()\n\n    executor = ThreadPoolExecutor(max_workers=workers, thread_name_prefix=\'export-verify\')\n    pending = deque()\n    try:\n        for name, expected in hashes.items():\n            pending.append(executor.submit(check_image, name, expected))\n            if len(pending) >= max(2, workers * 2):\n                pending.popleft().result()\n        while pending:\n            pending.popleft().result()\n    finally:\n        executor.shutdown(wait=True, cancel_futures=True)\n\n\ndef _validate_prediction_images(module, path, manifest, payload, workers):\n    metadata = manifest.get(\'metadata\', {})\n    mode, selection = metadata.get(\'export_mode\'), metadata.get(\'prediction_images\', \'none\')\n    index = manifest.get(\'prediction_images\')\n    if mode is not None and mode not in (\'json\', \'full\'):\n        raise ValueError(\'invalid bundle export mode\')\n    if selection not in (\'none\', \'labels\', \'preview\', \'both\'):\n        raise ValueError(\'invalid prediction image selection\')\n    if mode == \'json\' and \'image_artifacts_sha256\' in manifest:\n        raise ValueError(\'JSON export must not contain full image artifacts\')\n    if mode == \'full\' and \'image_artifacts_sha256\' not in manifest:\n        raise ValueError(\'Full export is missing full image artifact hashes\')\n    if selection == \'none\':\n        if index is not None:\n            raise ValueError(\'unexpected prediction image attachments\')\n        return\n    if mode != \'json\' or not isinstance(index, dict) or index.get(\'version\') != 1 or index.get(\'selection\') != selection:\n        raise ValueError(\'prediction image index is missing or inconsistent\')\n    images = payload[\'ground_truth.json\'][\'images\']\n    rows, hashes = index.get(\'images\'), index.get(\'sha256\')\n    if not isinstance(rows, list) or len(rows) != len(images) or not isinstance(hashes, dict):\n        raise ValueError(\'prediction image coverage is incomplete\')\n    semantic_ids = {entry[\'image_id\'] for entry in payload[\'semantic_predictions.json\']}\n    instances = {}\n    for entry in payload[\'predictions.json\']:\n        instances.setdefault(entry[\'image_id\'], []).append(entry)\n    dimensions, resolved_paths = {}, set()\n    for image, row in zip(images, rows):\n        if not isinstance(row, dict) or any(row.get(key) != image[source] for key, source in\n                [(\'image_id\', \'id\'), (\'sample_id\', \'sample_id\'), (\'width\', \'width\'), (\'height\', \'height\')]):\n            raise ValueError(\'prediction image identity or dimensions differ\')\n        artifacts = row.get(\'artifacts\')\n        native = instances.get(image[\'id\'], [])\n        has_labels = image[\'id\'] in semantic_ids and manifest[\'task\'] != \'instance\'\n        available = has_labels or bool(native) or manifest[\'task\'] == \'instance\'\n        if not isinstance(artifacts, dict) or artifacts.get(\'available\') is not available:\n            raise ValueError(\'prediction mask availability differs from JSON\')\n        expected = set()\n        if available and selection in (\'preview\', \'both\'):\n            expected.add(\'preview\')\n        if selection in (\'labels\', \'both\'):\n            if has_labels:\n                expected.add(\'mask\' if manifest[\'task\'] == \'binary\' else \'labels\')\n            if native or manifest[\'task\'] == \'instance\':\n                expected.add(\'instances\')\n        if set(artifacts) & {\'mask\', \'labels\', \'preview\', \'instances\', \'layout_preview\'} != expected:\n            raise ValueError(\'prediction image selection coverage differs\')\n        references = [artifacts[key] for key in expected - {\'instances\'}]\n        if \'instances\' in expected:\n            masks = artifacts[\'instances\']\n            if not isinstance(masks, list) or len(masks) != len(native):\n                raise ValueError(\'prediction instance image coverage differs\')\n            for number, (mask, annotation) in enumerate(zip(masks, native), 1):\n                if not isinstance(mask, dict) or mask.get(\'instance_index\') != number or any(\n                        mask.get(key) != annotation[key] for key in (\'category_id\', \'score\')):\n                    raise ValueError(\'prediction instance image identity differs\')\n                references.append(mask.get(\'path\'))\n        for name in references:\n            if (not isinstance(name, str) or not name.startswith(\'predictions/\') or\n                    Path(name).as_posix() != name or \'..\' in Path(name).parts or\n                    (path / name).resolve() in resolved_paths):\n                raise ValueError(\'invalid or duplicate prediction image path\')\n            resolved_paths.add((path / name).resolve())\n            dimensions[name] = (image[\'width\'], image[\'height\'])\n    if set(hashes) != set(dimensions):\n        raise ValueError(\'prediction image hashes do not match image references\')\n    _verify_artifact_files(path, hashes, workers, dimensions, module)\n\n\ndef _validate_bundle(module, path, workers=2):\n    """Same complete bundle checks, with bounded parallel image hash reads."""\n    path = Path(path)\n    manifest = json.loads((path / \'manifest.json\').read_text(encoding=\'utf-8\'))\n    if manifest.get(\'schema_version\') != module.SCHEMA or manifest.get(\'complete\') is not True:\n        raise ValueError(\'unsupported or incomplete bundle manifest\')\n    required = {\'ground_truth.json\', \'predictions.json\', \'semantic_ground_truth.json\', \'semantic_predictions.json\'}\n    if set(manifest.get(\'payload_sha256\', {})) != required:\n        raise ValueError(\'incomplete payload hash manifest\')\n    payload = {}\n    for name, expected in manifest[\'payload_sha256\'].items():\n        data = (path / name).read_bytes()\n        if hashlib.sha256(data).hexdigest() != expected:\n            raise ValueError(f\'payload hash mismatch: {name}\')\n        text = data.decode(\'utf-8\')\n        del data\n        payload[name] = json.loads(text)\n        del text\n    gt = payload[\'ground_truth.json\']\n    if \'image_artifacts_sha256\' in manifest:\n        hashes, references = manifest[\'image_artifacts_sha256\'], set()\n        for item in gt[\'images\']:\n            artifacts = item.get(\'artifacts\')\n            if not artifacts or item[\'file_name\'] != artifacts.get(\'image\'):\n                raise ValueError(\'image artifact coverage is incomplete\')\n            references.update([artifacts[\'image\'], artifacts[\'valid_mask\']])\n            references.update(x[\'path\'] for x in artifacts[\'originals\'])\n            for role in (\'ground_truth\', \'prediction\'):\n                masks = artifacts[role]\n                references.update(masks[key] for key in (\'mask\', \'labels\', \'preview\', \'layout_preview\') if key in masks)\n                references.update(x[\'path\'] for x in masks.get(\'instances\', []))\n        if set(hashes) != references:\n            raise ValueError(\'image artifact hash manifest does not match image references\')\n        _verify_artifact_files(path, hashes, workers)\n    if gt[\'categories\'] != manifest[\'categories\']:\n        raise ValueError(\'manifest and ground-truth categories differ\')\n    images = {im[\'id\']: im for im in gt[\'images\']}\n    sample_ids = [x[\'sample_id\'] for x in gt[\'images\']]\n    if len(images) != len(gt[\'images\']) or len(set(sample_ids)) != len(sample_ids):\n        raise ValueError(\'duplicate images or samples\')\n    if sample_ids != manifest[\'coverage_sample_ids\'] or len(images) != manifest[\'expected_samples\'] or len(images) != manifest[\'evaluated_samples\']:\n        raise ValueError(\'coverage mismatch in bundle\')\n    for im in images.values():\n        if module.decode_mask(im[\'valid_mask\']).shape != (im[\'height\'], im[\'width\']):\n            raise ValueError(\'valid mask shape mismatch\')\n    for name in required:\n        entries = payload[name][\'annotations\'] if name == \'ground_truth.json\' else payload[name]\n        for entry in entries:\n            if entry[\'image_id\'] not in images:\n                raise ValueError(\'annotation references an unknown image\')\n            im = images[entry[\'image_id\']]\n            if module.decode_mask(entry[\'segmentation\']).shape != (im[\'height\'], im[\'width\']):\n                raise ValueError(\'annotation mask shape mismatch\')\n            if name == \'predictions.json\' and not math.isfinite(entry[\'score\']):\n                raise ValueError(\'nonfinite prediction score\')\n    _validate_prediction_images(module, path, manifest, payload, workers)\n    return manifest, payload\n\n\nclass _ExportLog:\n    def __init__(self, original, stream):\n        self.original, self.stream = original, stream\n\n    def write(self, text):\n        self.stream.write(text)\n        self.stream.flush()\n        return self.original.write(text)\n\n    def flush(self):\n        self.stream.flush()\n        self.original.flush()\n\n    def __getattr__(self, name):\n        return getattr(self.original, name)\n\n\ndef _current_result(model):\n    try:\n        return read_json(stage_result(model) / \'results.json\')\n    except (OSError, ValueError):\n        return {}\n\n\ndef _run_id(result):\n    metadata = result.get(\'metadata\')\n    return result.get(\'run_id\') or (metadata.get(\'run_id\') if isinstance(metadata, dict) else None)\n\n\ndef _validate_cache_identity(model, cache, manifest, args):\n    if manifest.get(\'model_id\') != model.name:\n        raise ValueError(\'Prediction cache belongs to a different model\')\n    if not isinstance(manifest.get(\'run_id\'), str) or not manifest[\'run_id\']:\n        raise ValueError(\'Prediction cache has no inference run ID\')\n    marker_path = stage_result(model) / \'prediction_cache_state.json\'\n    if marker_path.exists():\n        marker = read_json(marker_path)\n        if marker.get(\'status\') != \'complete\' or marker.get(\'run_id\') != manifest[\'run_id\']:\n            raise ValueError(\'Latest inference attempt does not match this completed prediction cache\')\n    request = manifest[\'request\']\n    for field, path in (\n        (\'config_sha256\', model / \'weights\' / \'model.json\'),\n        (\'inference_sha256\', model / \'inference.py\'),\n    ):\n        if request.get(field) != sha256(path):\n            raise ValueError(f\'Prediction cache is stale: {field} differs\')\n    dataset_name = manifest[\'dataset\'][\'name\']\n    if request.get(\'dataset\') != dataset_name:\n        raise ValueError(\'Prediction cache dataset identity is inconsistent\')\n    if args.dataset is not None and args.dataset != dataset_name:\n        raise ValueError(\'Requested dataset differs from the prediction cache\')\n\n\ndef _load_verified_dataset(module, manifest, args):\n    request, expected = manifest[\'request\'], manifest[\'dataset\']\n    dataset_root = Path(args.dataset_root or request[\'dataset_root\']).expanduser().resolve()\n    config = copy.deepcopy(manifest[\'config\'])\n    dataset = module.load_dataset(dataset_root, expected[\'name\'], expected[\'splits\'], config)\n    checks = {\n        \'name\': dataset.name,\n        \'splits\': dataset.splits,\n        \'full_count\': len(dataset),\n        \'categories\': dataset.categories,\n        \'task\': dataset.task,\n        \'ap_supported\': dataset.ap_supported,\n        \'ap_reason\': dataset.ap_reason,\n        \'manifest_status\': dataset.manifest.get(\'status\'),\n    }\n    for field, actual in checks.items():\n        if expected.get(field) != actual:\n            raise ValueError(f\'Dataset differs from cached inference: {field}\')\n    count = expected.get(\'count\')\n    if type(count) is not int or count < 1 or count > len(dataset):\n        raise ValueError(\'Prediction cache has an invalid selected sample count\')\n    if expected.get(\'scope\') != (\'subset\' if count < len(dataset) else \'full\'):\n        raise ValueError(\'Prediction cache scope does not match its sample count\')\n    if request.get(\'source_manifest_sha256\') != sha256(dataset.root / \'evaluation_samples.jsonl\'):\n        raise ValueError(\'Dataset sample manifest changed since inference\')\n    dataset_manifest = dataset.root / \'dataset_manifest.json\'\n    current_hash = sha256(dataset_manifest) if dataset_manifest.is_file() else None\n    if request.get(\'dataset_manifest_sha256\') != current_hash:\n        raise ValueError(\'Dataset metadata manifest changed since inference\')\n    return dataset, dataset_root, config\n\n\ndef _reusable_bundle(module, model, manifest, destination, args):\n    if args.force:\n        return False\n    result = _current_result(model)\n    value = result.get(\'bundle\')\n    path = Path(value) if isinstance(value, str) and value else None\n    if path is not None:\n        path = (path if path.is_absolute() else model / path).resolve()\n    if (result.get(\'status\') not in (\'complete\', \'exported\') or\n            _run_id(result) != manifest[\'run_id\'] or\n            result.get(\'model_id\') not in (None, model.name) or path != destination.resolve()):\n        # A failed/interrupted result record does not authorize replacing a\n        # completed full bundle from this inference run with the new default.\n        try:\n            published = read_json(destination / \'manifest.json\')\n            identity = published.get(\'metadata\', {})\n            preserve = (published.get(\'complete\') is True and\n                        \'image_artifacts_sha256\' in published and\n                        identity.get(\'model_id\') == model.name and\n                        identity.get(\'run_id\') == manifest[\'run_id\'])\n        except (OSError, ValueError, AttributeError):\n            preserve = False\n        if preserve:\n            _validate_bundle(module, destination, getattr(args, \'verify_workers\', 2))\n            raise ValueError(\'A complete full bundle already exists for this inference run, but \'\n                             \'results.json is inconsistent; refusing to replace it. Restore its \'\n                             \'result by evaluating the bundle, or explicitly rebuild with --force.\')\n        return False\n    try:\n        bundle_manifest, _ = _validate_bundle(module, path, getattr(args, \'verify_workers\', 2))\n    except (OSError, ValueError, KeyError, TypeError):\n        return False\n    metadata = bundle_manifest.get(\'metadata\', {})\n    if metadata.get(\'model_id\') != model.name or metadata.get(\'run_id\') != manifest[\'run_id\']:\n        return False\n    requested_mode, requested_pictures = _export_options(args)\n    actual_mode = metadata.get(\'export_mode\')\n    legacy_full = actual_mode is None and \'image_artifacts_sha256\' in bundle_manifest\n    if not legacy_full and (actual_mode != requested_mode or\n                           metadata.get(\'prediction_images\', \'none\') != requested_pictures):\n        return False\n    print(f\'[{model.name}] reusable bundle: requested mode={requested_mode}, \'\n          f\'prediction_images={requested_pictures}; actual mode={actual_mode or "legacy full"}, \'\n          f\'prediction_images={metadata.get("prediction_images", "all (legacy full)")}\', flush=True)\n    return True\n\n\ndef _export_environment(args):\n    mode, pictures = _export_options(args)\n    return {\n        \'actual_device\': \'cpu\', \'python\': sys.version, \'executable\': sys.executable,\n        \'platform\': platform.platform(), \'machine\': platform.machine(),\n        \'cpu_count\': os.cpu_count(), \'threads\': args.threads,\n        \'io_workers\': getattr(args, \'io_workers\', 2),\n        \'verify_workers\': getattr(args, \'verify_workers\', 2),\n        \'queue_mib\': getattr(args, \'queue_mib\', 64),\n        \'export_mode\': mode, \'prediction_images\': pictures,\n        \'environment_overrides\': {\'CUDA_VISIBLE_DEVICES\': os.environ.get(\'CUDA_VISIBLE_DEVICES\', \'\')},\n    }\n\n\ndef _write_export(module, model, cache, manifest, dataset, config, destination, args, started):\n    import numpy as np\n\n    run_dir = stage_result(model)\n    info, request = manifest[\'dataset\'], manifest[\'request\']\n    environment = copy.deepcopy(manifest[\'environment\'])\n    timing = copy.deepcopy(manifest[\'timing\'])\n    export_environment = _export_environment(args)\n    mode, pictures = _export_options(args)\n    metadata = {\n        \'model_id\': model.name, \'dataset\': info[\'name\'], \'splits\': info[\'splits\'],\n        \'scope\': info[\'scope\'], \'selected_source_samples\': info[\'count\'],\n        \'full_source_samples\': info[\'full_count\'],\n        \'source_manifest_sha256\': request[\'source_manifest_sha256\'],\n        \'dataset_status\': info[\'manifest_status\'], \'protocol\': module.json_metadata(config),\n        \'seed\': request[\'seed\'], \'run_id\': manifest[\'run_id\'],\n        \'environment\': environment, \'timing\': timing,\n        \'export_environment\': export_environment,\n        \'export_mode\': mode, \'prediction_images\': pictures,\n    }\n    samples, units_count, writer, prediction_images = [], 0, None, None\n    dataset_loading_seconds = 0.0\n    probability_outputs = set()\n    native_outputs = run_dir / \'native_outputs\'\n    if mode == \'full\' and native_outputs.is_symlink():\n        raise ValueError(\'Native probability output directory must not be a symlink\')\n    print(f\'[{model.name}] export mode={mode}, prediction_images={pictures}\', flush=True)\n    with module.BundleOutput(destination, model.name) as output:\n        with _ExportIO(module, output.staging, args) as export_io:\n            for expected_index, record in enumerate(iter_records(cache, manifest)):\n                if expected_index >= info[\'count\'] or record.get(\'index\') != expected_index:\n                    raise ValueError(\'Prediction cache sample indices are not consecutive\')\n                load_started = time.perf_counter()\n                sample = dataset.load(expected_index)\n                dataset_loading_seconds += time.perf_counter() - load_started\n                if record.get(\'sample_id\') != sample.sample_id:\n                    raise ValueError(f\'Prediction cache sample ID differs at index {expected_index}\')\n                prediction = module.Prediction(**record[\'prediction\'])\n                probability = prediction.metadata.pop(\'native_probability\', None)\n                if mode == \'full\' and probability is not None:\n                    native_outputs.mkdir(parents=True, exist_ok=True)\n                    probability_path = native_outputs / f\'{expected_index:06d}_probability.npz\'\n                    np.savez_compressed(probability_path, probability=probability)\n                    probability_outputs.add(probability_path.name)\n                    prediction.metadata[\'native_probability_path\'] = os.path.relpath(args.result_dir / \'native_outputs\' / probability_path.name, model)\n                    prediction.metadata[\'native_probability_path_base\'] = \'model\'\n                del probability\n                if writer is None:\n                    ap_supported = dataset.ap_supported and prediction.metadata.get(\'ap_supported\', config.get(\'ap_supported\', True))\n                    ap_reason = dataset.ap_reason if not dataset.ap_supported else prediction.metadata.get(\'ap_reason\', config.get(\'ap_reason\'))\n                    task = dataset.task\n                    if prediction.metadata.get(\'native_task\') == \'bounding_box_detection\' or (\n                        sample.target is None and task != \'instance\' and prediction.semantic is None\n                    ):\n                        task = \'unsupported\'\n                    metadata[\'iou_reason\'] = prediction.metadata.get(\'iou_reason\') or prediction.metadata.get(\'metrics_unavailable_reason\')\n                    writer = module.BundleWriter(\n                        output.staging, categories=dataset.categories, task=task,\n                        expected_samples=info[\'count\'], ap_supported=ap_supported,\n                        ap_reason=ap_reason, class_agnostic=config.get(\'class_agnostic\', False),\n                        metadata=metadata,\n                    )\n                    if mode == \'json\' and pictures != \'none\':\n                        prediction_images = _PredictionImages(module, writer, pictures)\n                for unit in module.evaluation_units(sample, prediction):\n                    if mode == \'json\':\n                        unit.pop(\'image\', None)\n                    writer.add(**unit)\n                    if prediction_images is not None:\n                        prediction_images.add(unit)\n                    units_count += 1\n                samples.append({\n                    \'sample_id\': sample.sample_id, \'forward_seconds\': prediction.forward_seconds,\n                    \'pipeline_seconds\': record[\'pipeline_seconds\'],\n                    \'metadata\': module.json_metadata(prediction.metadata),\n                })\n                if expected_index == 0 or (expected_index + 1) % 25 == 0 or expected_index + 1 == info[\'count\']:\n                    print(f\'[{model.name}] export {expected_index + 1}/{info["count"]} {info["scope"]}\', flush=True)\n            if writer is None or len(samples) != info[\'count\']:\n                raise ValueError(\'Prediction cache sample count differs from completed inference\')\n            writer.manifest[\'expected_samples\'] = units_count\n            writer.manifest[\'metadata\'][\'evaluated_2d_units\'] = units_count\n            timing[\'evaluated_2d_units\'] = units_count\n            writer.manifest[\'metadata\'][\'export_seconds\'] = time.perf_counter() - started\n            print(f\'[{model.name}] waiting for pending image writes before final validation\', flush=True)\n            export_io.flush()\n            staged = writer.finalize()\n            print(f\'[{model.name}] validating complete bundle with {getattr(args, "verify_workers", 2)} read workers\', flush=True)\n            validation_started = time.perf_counter()\n            _validate_bundle(module, staged, getattr(args, \'verify_workers\', 2))\n            writer.manifest[\'metadata\'][\'export_io\'] = {\n                **export_io.stats, \'dataset_loading_seconds\': dataset_loading_seconds,\n                \'validation_seconds\': time.perf_counter() - validation_started,\n                \'verify_workers\': getattr(args, \'verify_workers\', 2),\n            }\n            # Include image/JSON serialization and integrity validation in export time.\n            export_seconds = time.perf_counter() - started\n            writer.manifest[\'metadata\'][\'export_seconds\'] = export_seconds\n            atomic_json(staged / \'manifest.json\', writer.manifest)\n            bundle = output.publish()\n    # Retire only numbered probability files owned by prior exports, after success.\n    if mode == \'full\' and native_outputs.is_dir():\n        for path in native_outputs.glob(\'*_probability.npz\'):\n            if path.name.removesuffix(\'_probability.npz\').isdigit() and path.name not in probability_outputs:\n                path.unlink()\n    result = {\n        \'status\': \'exported\', \'metrics\': dict.fromkeys(METRICS), \'bundle\': str(bundle),\n        \'metadata\': writer.manifest[\'metadata\'], \'model_id\': model.name,\n        \'run_id\': manifest[\'run_id\'], \'environment\': environment, \'timing\': timing,\n        \'export_environment\': export_environment, \'export_seconds\': export_seconds,\n    }\n    atomic_json(run_dir / \'environment.json\', environment)\n    atomic_json(run_dir / \'timing.json\', timing)\n    atomic_json(run_dir / \'export_environment.json\', export_environment)\n    atomic_json(run_dir / \'samples.json\', samples)\n    atomic_json(run_dir / \'bundle.json\', {\'path\': os.path.relpath(bundle, run_dir)})\n    atomic_json(run_dir / \'results.json\', result)\n    atomic_json(run_dir / \'last_attempt.json\', result)\n    record = {\n        \'model_id\': model.name, \'run_id\': manifest[\'run_id\'], \'status\': \'exported\',\n        \'dataset\': info[\'name\'], \'scope\': info[\'scope\'], \'timing\': timing,\n        \'export_seconds\': export_seconds, \'run_directory\': str(run_dir),\n    }\n    module.update_timing_index(run_dir / \'inference_time.json\', record)\n    module.update_timing_index(stage_result(model) / \'inference_time.json\', record)\n    (run_dir / \'export_error.log\').unlink(missing_ok=True)\n    print(f\'[{model.name}] exported {units_count} evaluation units to {bundle}\', flush=True)\n\n\ndef export_worker(args):\n    model = args.project_root / \'Model\' / args._worker\n    os.environ[\'CUDA_VISIBLE_DEVICES\'] = \'\'\n    with model_lock(model):\n        manifest = None\n        run_dir = stage_result(model)\n        log = (run_dir / \'export.log\').open(\'a\', encoding=\'utf-8\')\n        stdout, stderr = sys.stdout, sys.stderr\n        sys.stdout, sys.stderr = _ExportLog(stdout, log), _ExportLog(stderr, log)\n        try:\n            setup_environment(model, args, cpu=True)\n            run_dir.mkdir(parents=True, exist_ok=True)\n            config = read_json(model / \'weights\' / \'model.json\')\n            cache = cache_directory(model, args)\n            manifest = read_cache(cache)\n            _validate_cache_identity(model, cache, manifest, args)\n            if config.get(\'model_id\') != model.name:\n                raise ValueError(\'Model configuration has a different model ID\')\n            current = _current_result(model)\n            if args.force or _run_id(current) != manifest[\'run_id\']:\n                atomic_json(run_dir / \'results.json\', {\n                    \'status\': \'exporting\', \'model_id\': model.name, \'run_id\': manifest[\'run_id\'],\n                    \'metrics\': dict.fromkeys(METRICS),\n                })\n            started = time.perf_counter()\n            module = load_model_module(model)\n            dataset, dataset_root, effective_config = _load_verified_dataset(module, manifest, args)\n            destination = stage_result(model) / \'coco_bundle\'\n            if _reusable_bundle(module, model, manifest, destination, args):\n                print(f\'[{model.name}] reusing validated bundle for inference {manifest["run_id"]}\', flush=True)\n                return 0\n            atomic_json(run_dir / \'results.json\', {\n                \'status\': \'exporting\', \'model_id\': model.name, \'run_id\': manifest[\'run_id\'],\n                \'metrics\': dict.fromkeys(METRICS),\n            })\n            atomic_json(run_dir / \'run_config.json\', {\n                \'config\': effective_config, \'arguments\': manifest[\'request\'],\n                \'seed\': manifest[\'request\'][\'seed\'],\n                \'source_root\': str((model / config[\'source_root\']).resolve()),\n                \'dataset_root\': str(dataset_root), \'prediction_cache\': str(cache),\n                \'inference_run_id\': manifest[\'run_id\'],\n            })\n            _write_export(module, model, cache, manifest, dataset, effective_config, destination, args, started)\n            return 0\n        except BaseException as exc:\n            run_dir.mkdir(parents=True, exist_ok=True)\n            failure = {\n                \'status\': \'failed_export\', \'model_id\': model.name,\n                \'run_id\': manifest.get(\'run_id\') if isinstance(manifest, dict) else None,\n                \'metrics\': dict.fromkeys(METRICS), \'error_type\': type(exc).__name__,\n                \'error\': str(exc), \'run_directory\': str(run_dir),\n            }\n            if isinstance(manifest, dict):\n                failure.update(environment=manifest.get(\'environment\'), timing=manifest.get(\'timing\'))\n            atomic_json(run_dir / \'results.json\', failure)\n            atomic_json(run_dir / \'last_attempt.json\', failure)\n            (run_dir / \'export_error.log\').write_text(traceback.format_exc(), encoding=\'utf-8\')\n            print(f\'[{model.name}] FAILED export: {exc}\', file=sys.stderr, flush=True)\n            return 130 if isinstance(exc, KeyboardInterrupt) else 1\n        finally:\n            sys.stdout, sys.stderr = stdout, stderr\n            log.close()\n\n\ndef main():\n    parser = base_parser(__doc__)\n    parser.add_argument(\'--bundle-root\', type=Path, help=\'Exported package root; default: PROJECT_ROOT/Dataset_Coco\')\n    parser.add_argument(\'--export-mode\', choices=(\'json\', \'full\'), default=\'json\',\n                        help=\'json: evaluation JSON only (default); full: original complete image export\')\n    parser.add_argument(\'--prediction-images\', choices=(\'none\', \'labels\', \'preview\', \'both\'), default=\'none\',\n                        help=\'Optional prediction pictures in JSON mode; default none\')\n    parser.add_argument(\'--io-workers\', type=positive, default=2,\n                        help=\'Concurrent file writes within one model; default 2 (models stay serial)\')\n    parser.add_argument(\'--verify-workers\', type=positive, default=2,\n                        help=\'Concurrent full image readback checks; default 2\')\n    parser.add_argument(\'--queue-mib\', type=positive, default=64,\n                        help=\'Pending encoded image byte budget in MiB; default 64, plus current image and source arrays\')\n    args = parser.parse_args()\n    try:\n        _export_options(args)\n    except ValueError as exc:\n        parser.error(str(exc))\n    resolve_paths(args)\n    if args.bundle_root is not None:\n        args.bundle_root = args.bundle_root.expanduser().resolve()\n    if args._worker:\n        return export_worker(args)\n    return launch_batch(args, parser, Path(__file__), \'export\')\n\n\n\n',
    "EXPORT_IMPORTS": ['atomic_json', 'base_parser', 'cache_directory', 'iter_records', 'launch_batch', 'load_model_module', 'model_lock', 'read_cache', 'read_json', 'resolve_paths', 'setup_environment', 'sha256', 'positive'],
    "CACHE_COMPAT_SOURCE": '"""Verify explicit migration attestations without rewriting historical records.\n\nThis stdlib-only block may be embedded in each inference.py. A True result\nauthorizes only the requested historical-cache operation. It does not claim\nthat the current environment or a newly executed model reproduces old results.\nExisting shard/CRC, model/run-state, sample-ID and dataset checks remain required.\n"""\nimport hashlib\nimport json\nfrom pathlib import Path\n\n\ndef _cache_migration_digest(path):\n    digest = hashlib.sha256()\n    with Path(path).open(\'rb\') as stream:\n        for block in iter(lambda: stream.read(1024 * 1024), b\'\'):\n            digest.update(block)\n    return digest.hexdigest()\n\n\ndef _cache_migration_json(path):\n    def reject(value):\n        raise ValueError(\'Non-finite migration JSON: \' + value)\n    value = json.loads(Path(path).read_text(encoding=\'utf-8\'), parse_constant=reject)\n    if not isinstance(value, dict):\n        raise ValueError(\'Migration evidence must be a JSON object: \' + str(path))\n    return value\n\n\ndef _cache_migration_equal(left, right):\n    # JSON booleans must not compare equal to integer seed/count fields.\n    return json.dumps(left, sort_keys=True, allow_nan=False) == json.dumps(right, sort_keys=True, allow_nan=False)\n\n\ndef verify_cache_migration(model: Path, manifest: dict, request: dict | None = None) -> bool:\n    """Check an attested migration; invalid evidence raises ValueError.\n\n    Return False for absent/unsupported attestations, operations not approved\n    by the attestation, or a different requested inference configuration.\n    request=None asks to consume historical predictions for CPU export;\n    request=<make_request result> asks to skip inference using those predictions.\n    """\n    model = Path(model).resolve()\n    provenance_path = model / \'result\' / \'migration_provenance.json\'\n    if not provenance_path.exists() and not provenance_path.is_symlink():\n        return False\n    try:\n        if provenance_path.is_symlink():\n            raise ValueError(\'Migration provenance must not be a symlink\')\n        provenance = _cache_migration_json(provenance_path)\n        if provenance.get(\'schema_version\') != \'segmentation-cache-migration/v1\':\n            return False\n        uses = provenance[\'allowed_uses\']\n        if (not isinstance(uses, list) or len(set(uses)) != len(uses)\n                or any(value not in {\'export\', \'inference_resume\'} for value in uses)):\n            raise ValueError(\'Invalid migration allowed_uses\')\n        cache_manifest = model / \'result\' / \'prediction_cache\' / \'manifest.json\'\n        if cache_manifest.is_symlink():\n            raise ValueError(\'Attested prediction manifest must not be a symlink\')\n        if _cache_migration_digest(cache_manifest) != provenance[\'original_cache_manifest_sha256\']:\n            raise ValueError(\'Historical prediction manifest changed after migration attestation\')\n        original = _cache_migration_json(cache_manifest)\n        if not _cache_migration_equal(original, manifest):\n            raise ValueError(\'Supplied prediction manifest differs from attested cache\')\n        if (original.get(\'schema_version\') != \'deferred-predictions/v1\'\n                or original.get(\'status\') != \'complete\'\n                or original.get(\'model_id\') != model.name\n                or provenance.get(\'model_id\') != model.name\n                or not isinstance(original.get(\'run_id\'), str) or not original[\'run_id\']\n                or provenance.get(\'run_id\') != original[\'run_id\']):\n            raise ValueError(\'Migration model/run identity differs from historical cache\')\n        old_request = original[\'request\']\n        if not _cache_migration_equal(provenance[\'original_request\'], old_request):\n            raise ValueError(\'Attested request differs from historical prediction request\')\n        current_files = provenance[\'current_files\']\n        if set(current_files) != {\'inference.py\', \'weights/model.json\'}:\n            raise ValueError(\'Migration current_files must bind exactly code and configuration\')\n        for relative, expected in current_files.items():\n            path = model / relative\n            if path.is_symlink() or _cache_migration_digest(path) != expected:\n                raise ValueError(\'Migrated code/configuration differs from attestation: \' + relative)\n\n        evidence_root = model / \'result\' / \'migration_evidence\'\n        if (model / \'result\').is_symlink() or evidence_root.is_symlink():\n            raise ValueError(\'Migration evidence directories must not be symlinks\')\n        evidence_root = evidence_root.resolve()\n        evidence = {}\n        for record in provenance[\'evidence\']:\n            relative = Path(record[\'path\'])\n            path = model / relative\n            if (relative.is_absolute() or \'..\' in relative.parts or path.is_symlink()\n                    or not path.resolve().is_relative_to(evidence_root)):\n                raise ValueError(\'Migration evidence must remain inside result/migration_evidence\')\n            if _cache_migration_digest(path) != record[\'sha256\']:\n                raise ValueError(\'Migration evidence hash mismatch: \' + record[\'path\'])\n            role = record[\'role\']\n            if role in evidence:\n                raise ValueError(\'Duplicate migration evidence role: \' + role)\n            evidence[role] = (path, record[\'sha256\'])\n        required = {\'original_inference\', \'before_migration_inference\', \'original_config\', \'code_review\'}\n        if not required.issubset(evidence):\n            raise ValueError(\'Migration evidence is missing original source/configuration or review\')\n        chain = provenance[\'code_chain\']\n        original_inference = old_request[\'inference_sha256\']\n        before_inference = evidence[\'before_migration_inference\'][1]\n        if (chain[\'original_inference_sha256\'] != original_inference\n                or evidence[\'original_inference\'][1] != original_inference\n                or chain[\'before_migration_inference_sha256\'] != before_inference\n                or chain[\'review_sha256\'] != evidence[\'code_review\'][1]\n                or evidence[\'original_config\'][1] != old_request[\'config_sha256\']):\n            raise ValueError(\'Migration evidence does not connect to original cached code/configuration\')\n        review = _cache_migration_json(evidence[\'code_review\'][0])\n        if (review.get(\'schema_version\') != \'segmentation-cache-code-review/v1\'\n                or review.get(\'model_id\') != model.name\n                or review.get(\'original_inference_sha256\') != original_inference\n                or review.get(\'before_migration_inference_sha256\') != before_inference\n                or review.get(\'current_inference_sha256\') != current_files[\'inference.py\']\n                or review.get(\'export_data_definitions_equal\') is not True\n                or not isinstance(review.get(\'compared_definitions\'), list)\n                or not review[\'compared_definitions\']\n                or any(not isinstance(value, str) or not value for value in review[\'compared_definitions\'])):\n            raise ValueError(\'Migration code review does not bind unchanged dataset/export interpretation\')\n        if \'inference_resume\' in uses and review.get(\'inference_resume_approved\') is not True:\n            raise ValueError(\'Inference resume was not approved by the source compatibility review\')\n        if before_inference != original_inference:\n            prior = review.get(\'prior_timing_integration\')\n            if (not isinstance(prior, dict)\n                    or prior.get(\'kind\') != \'documented_forward_timing_integration\'\n                    or prior.get(\'old_sha256\') != original_inference\n                    or prior.get(\'new_sha256\') != before_inference\n                    or prior.get(\'native_sources_equal\') is not True\n                    or not isinstance(prior.get(\'changed_existing_definitions\'), list)):\n                raise ValueError(\'Preexisting code changes lack a linked timing integration review\')\n\n        old_config = _cache_migration_json(evidence[\'original_config\'][0])\n        config = _cache_migration_json(model / \'weights\' / \'model.json\')\n        relocatable = {\'dataset_root\', \'bundle_root\', \'python\', \'evaluation_python\', \'runtime_preload\'}\n        if not _cache_migration_equal(\n                {key: value for key, value in old_config.items() if key not in relocatable},\n                {key: value for key, value in config.items() if key not in relocatable}):\n            raise ValueError(\'Migration changed model configuration beyond approved runtime/output paths\')\n        dataset = provenance[\'dataset\']\n        if dataset.get(\'path_base\') != \'model\' or Path(dataset[\'root\']).is_absolute():\n            raise ValueError(\'Migrated dataset root must be relative to model\')\n        dataset_root = (model / dataset[\'root\']).resolve()\n        if (model / config[\'dataset_root\']).resolve() != dataset_root:\n            raise ValueError(\'Configured dataset root differs from the attested migration\')\n        dataset_name = original[\'dataset\'][\'name\']\n        if not isinstance(dataset_name, str) or Path(dataset_name).name != dataset_name:\n            raise ValueError(\'Invalid cached dataset name\')\n        if old_request[\'dataset\'] != dataset_name:\n            raise ValueError(\'Cached request/dataset identity is inconsistent\')\n        data = dataset_root if (dataset_root / \'evaluation_samples.jsonl\').is_file() else dataset_root / dataset_name\n        for field, relative in [(\'source_manifest_sha256\', \'evaluation_samples.jsonl\'),\n                                (\'dataset_manifest_sha256\', \'dataset_manifest.json\')]:\n            expected = old_request[field]\n            path = data / relative\n            actual = _cache_migration_digest(path) if path.is_file() else None\n            if (field == \'source_manifest_sha256\' and not isinstance(expected, str)) or actual != expected or dataset.get(field) != expected:\n                raise ValueError(\'Migrated dataset manifest differs from cached inference: \' + field)\n\n        use = \'export\' if request is None else \'inference_resume\'\n        if use not in uses:\n            return False\n        if request is not None:\n            expected_request = dict(old_request,\n                                    config_sha256=current_files[\'weights/model.json\'],\n                                    inference_sha256=current_files[\'inference.py\'],\n                                    dataset_root=str(dataset_root))\n            if not _cache_migration_equal(request, expected_request):\n                return False\n        return True\n    except (OSError, KeyError, TypeError, AttributeError, OverflowError, RuntimeError) as exc:\n        raise ValueError(\'Invalid or unavailable cache migration evidence: \' + str(exc)) from exc\n'}
exec(compile('"""Standalone orchestration; embedded verbatim by generate_models.py.\n\nModel mathematics and native adapters remain in the surrounding inference.py.\nThe original cache/export engines run in private namespaces and only write an\nunpublished attempt. Publication has a rollback journal and retains old files.\n"""\nimport argparse\nfrom contextlib import contextmanager, nullcontext\nimport copy\nimport fcntl\nimport hashlib\nimport json\nimport os\nfrom pathlib import Path\nimport shutil\nimport signal\nimport stat\nimport subprocess\nimport sys\nimport tempfile\nimport time\nimport traceback\nimport types\nimport uuid\n\n\ndef positive(value):\n    number=int(value)\n    if number<1:raise argparse.ArgumentTypeError(\'must be positive\')\n    return number\n\n\ndef nonnegative(value):\n    number=int(value)\n    if number<0:raise argparse.ArgumentTypeError(\'must be nonnegative\')\n    return number\n\n\ndef parser():\n    p=argparse.ArgumentParser(description=\'Independent model inference, CPU COCO export and optional evaluation\')\n    p.add_argument(\'--stage\',choices=(\'inference\',\'export\',\'full\'),default=\'full\')\n    p.add_argument(\'--dataset\')\n    p.add_argument(\'--dataset-root\',type=Path,help=\'Relative paths use the project root\')\n    p.add_argument(\'--result-dir\',type=Path,help=\'Isolated output directory under this model/result; relative to project root\')\n    p.add_argument(\'--environment\',choices=(\'current\',),default=\'current\')\n    p.add_argument(\'--venv\',type=Path,help=\'Explicit copied environment selected by the safe startup bootstrap\')\n    p.add_argument(\'--device\',default=\'auto\')\n    p.add_argument(\'--precision\',choices=(\'fp32\',\'fp16\',\'bf16\'))\n    p.add_argument(\'--limit\',type=positive)\n    p.add_argument(\'--split\',action=\'append\')\n    p.add_argument(\'--seed\',type=int)\n    p.add_argument(\'--warmup\',type=nonnegative,default=1)\n    p.add_argument(\'--threads\',type=positive,default=4)\n    p.add_argument(\'--shard-mib\',type=positive,default=256)\n    p.add_argument(\'--export-mode\',choices=(\'json\',\'full\'),default=\'json\')\n    p.add_argument(\'--prediction-images\',choices=(\'none\',\'labels\',\'preview\',\'both\'),default=\'none\')\n    p.add_argument(\'--io-workers\',type=positive,default=2)\n    p.add_argument(\'--verify-workers\',type=positive,default=2)\n    p.add_argument(\'--queue-mib\',type=positive,default=64)\n    p.add_argument(\'--force\',action=\'store_true\')\n    evaluation=p.add_mutually_exclusive_group()\n    evaluation.add_argument(\'--evaluate\',action=\'store_true\',help=\'Evaluate after exporting; implicit for full\')\n    evaluation.add_argument(\'--skip-evaluation\',action=\'store_true\')\n    p.add_argument(\'--dry-run\',action=\'store_true\',help=\'Validate arguments and output ownership without importing model dependencies or writing\')\n    p.add_argument(\'--check\',action=\'store_true\',help=\'Check configured dataset and model paths without loading weights\')\n    return p\n\n\ndef rooted(value,base):\n    path=Path(value).expanduser()\n    return (base/path if not path.is_absolute() else path).resolve()\n\n\ndef no_symlink_path(path):\n    path=Path(path).absolute()\n    for candidate in (path,*path.parents):\n        if candidate.is_symlink():raise ValueError(\'Output path contains a symlink: \'+str(candidate))\n    return path\n\n\ndef output_directory(model,value,project):\n    canonical=no_symlink_path(model/\'result\')\n    raw=canonical if value is None else Path(value).expanduser()\n    raw=raw if raw.is_absolute() else project/raw\n    no_symlink_path(raw)\n    target=raw.resolve()\n    if target!=canonical and not target.is_relative_to(canonical):\n        raise ValueError(\'--result-dir must stay inside this model/result\')\n    parts=target.relative_to(canonical).parts\n    if parts and parts[0] in {\'prediction_cache\',\'coco_bundle\',\'native_outputs\',\'migration_evidence\'}:\n        raise ValueError(\'--result-dir cannot select an existing artifact subtree\')\n    if any(part.startswith(\'.\') for part in parts):\n        raise ValueError(\'--result-dir cannot select reserved hidden transaction paths\')\n    if target.exists() and not target.is_dir():raise ValueError(\'Output directory is not a directory\')\n    return target\n\n\n@contextmanager\ndef lock_model(model):\n    result=no_symlink_path(model/\'result\')\n    result.mkdir(parents=True,exist_ok=True)\n    descriptor=os.open(result/\'.two_stage.lock\',os.O_CREAT|os.O_RDWR|os.O_NOFOLLOW,0o600)\n    try:\n        info=os.fstat(descriptor)\n        if not stat.S_ISREG(info.st_mode) or info.st_nlink!=1:raise ValueError(\'Unsafe model lock\')\n        try:fcntl.flock(descriptor,fcntl.LOCK_EX|fcntl.LOCK_NB)\n        except BlockingIOError as exc:raise RuntimeError(\'Another inference/export/evaluation is using \'+model.name) from exc\n        yield\n    finally:os.close(descriptor)\n\n\ndef engines(module,args,attempt):\n    common={\'__name__\':\'_standalone_cache_engine\',\'__file__\':str(module.__file__),\n            \'stage_result\':lambda model:attempt}\n    exec(compile(INFERENCE_ENGINE_SOURCE,str(module.__file__)+\':cache\',\'exec\'),common)\n    common[\'load_model_module\']=lambda model:module\n    common[\'model_lock\']=lambda model:nullcontext()\n    common[\'cache_directory\']=lambda model,options:options._cache_directory\n    export={\'__name__\':\'_standalone_export_engine\',\'__file__\':str(module.__file__),\n            \'stage_result\':lambda model:attempt}\n    for name in EXPORT_IMPORTS:export[name]=common[name]\n    exec(compile(EXPORT_ENGINE_SOURCE,str(module.__file__)+\':export\',\'exec\'),export)\n    export[\'_validate_cache_identity\']=lambda model,cache,manifest,options:validate_cache(common,model,cache,manifest,options)\n    return common,export\n\n\ndef compatibility(model,manifest,request=None):\n    if not CACHE_COMPAT_SOURCE:return False\n    namespace={\'__name__\':\'_standalone_cache_compat\',\'__file__\':str(model/\'inference.py\')}\n    exec(compile(CACHE_COMPAT_SOURCE,str(model/\'inference.py\')+\':cache_compat\',\'exec\'),namespace)\n    return namespace[\'verify_cache_migration\'](model,manifest,request=request)\n\n\ndef validate_cache(common,model,cache,manifest,args,request=None,allow_stale=False):\n    if manifest.get(\'model_id\')!=model.name or not isinstance(manifest.get(\'run_id\'),str) or not manifest[\'run_id\']:\n        raise ValueError(\'Prediction cache model/run identity is invalid\')\n    previous=manifest[\'request\']\n    current={\'config_sha256\':common[\'sha256\'](model/\'weights/model.json\'),\n             \'inference_sha256\':common[\'sha256\'](model/\'inference.py\')}\n    exact=all(previous.get(key)==value for key,value in current.items())\n    if request is not None:exact=exact and previous==request\n    if not exact and not compatibility(model,manifest,request):\n        if allow_stale:return False\n        raise ValueError(\'Prediction cache is stale or migration evidence is missing\')\n    if previous.get(\'dataset\')!=manifest[\'dataset\'][\'name\']:\n        raise ValueError(\'Prediction cache dataset identity is inconsistent\')\n    if args.dataset is not None and args.dataset!=manifest[\'dataset\'][\'name\']:\n        raise ValueError(\'Requested dataset differs from prediction cache\')\n    marker=cache.parent/\'prediction_cache_state.json\'\n    if marker.exists():\n        state=common[\'read_json\'](marker)\n        if state.get(\'status\')!=\'complete\' or state.get(\'run_id\')!=manifest[\'run_id\']:\n            raise ValueError(\'Prediction cache state does not match the completed run\')\n    return True\n\n\ndef normalize_records(attempt,target,model,cache):\n    """Only new run records are changed; cache requests and bundle evidence stay intact."""\n    canonical=model/\'result\'\n    def path_value(value):\n        if not isinstance(value,str):return value\n        try:path=Path(value)\n        except (TypeError,ValueError):return value\n        path=(path if path.is_absolute() else model/path).resolve()\n        if path.is_relative_to(attempt):\n            return os.path.relpath(target/path.relative_to(attempt),model)\n        if path==cache:return os.path.relpath(cache,model)\n        return value\n    def transform(value,key=None):\n        if isinstance(value,dict):return {k:transform(v,k) for k,v in value.items()}\n        if isinstance(value,list):return [transform(v,key) for v in value]\n        if key in {\'bundle\',\'prediction_cache\',\'cache\',\'run_directory\',\'native_probability_path\'}:return path_value(value)\n        return value\n    for path in attempt.iterdir():\n        if path.is_file() and path.suffix==\'.json\':\n            value=json.loads(path.read_text(encoding=\'utf-8\'))\n            if path.name==\'bundle.json\':value={\'path\':\'coco_bundle\'}\n            else:value=transform(value)\n            path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+\'\\n\',encoding=\'utf-8\')\n\n\ndef journal_write(path,value):\n    temporary=path.with_name(path.name+\'.tmp\')\n    with temporary.open(\'x\',encoding=\'utf-8\') as stream:\n        json.dump(value,stream,ensure_ascii=False,indent=2)\n        stream.write(\'\\n\')\n    os.replace(temporary,path)\n\n\ndef recover_publications(target):\n    """Revert interrupted multi-entry publication before consuming formal state."""\n    if not target.is_dir():return\n    for attempt in sorted(target.glob(\'.attempt-*\')):\n        if attempt.is_symlink():raise ValueError(\'Transaction attempt cannot be a symlink\')\n        path=attempt/\'.publication_journal.json\'\n        if not path.is_file():continue\n        if path.is_symlink():raise ValueError(\'Publication journal cannot be a symlink\')\n        value=json.loads(path.read_text(encoding=\'utf-8\'))\n        if value.get(\'status\') in {\'complete\',\'rolled_back\'}:continue\n        if value.get(\'status\')!=\'publishing\' or value.get(\'target\')!=str(target):\n            raise ValueError(\'Publication journal has an invalid target/status: \'+str(path))\n        history=Path(value[\'history\'])\n        if history.parent!=target or not history.name.startswith(\'.history-\') or history.is_symlink() or not history.is_dir():\n            raise ValueError(\'Publication history has an invalid path: \'+str(path))\n        entries=value[\'entries\']\n        names=[entry[\'name\'] for entry in entries]\n        if (len(names)!=len(set(names)) or any(not name or Path(name).name!=name or name.startswith(\'.\') for name in names)\n                or any(type(entry[\'existed\']) is not bool for entry in entries)):\n            raise ValueError(\'Publication journal has invalid entries\')\n        for entry in reversed(entries):\n            name=entry[\'name\']\n            staged,destination,previous=attempt/name,target/name,history/name\n            if any(candidate.is_symlink() for candidate in (staged,destination,previous)):\n                raise ValueError(\'Publication recovery encountered a symlink\')\n            if entry[\'existed\'] and previous.exists():\n                if destination.exists():\n                    if staged.exists():raise ValueError(\'Ambiguous publication recovery: \'+name)\n                    os.replace(destination,staged)\n                os.replace(previous,destination)\n            elif entry[\'existed\']:\n                if not (staged.exists() and destination.exists()):\n                    raise ValueError(\'Original publication entry is missing; inspect \'+str(history))\n            elif destination.exists():\n                if staged.exists():raise ValueError(\'Ambiguous new publication entry: \'+name)\n                os.replace(destination,staged)\n            elif not staged.exists():raise ValueError(\'Staged publication entry is missing: \'+name)\n        value[\'status\']=\'rolled_back\'\n        # An interrupted journal replacement may leave this exact owned temp.\n        pending=path.with_name(path.name+\'.tmp\')\n        if pending.exists():\n            if pending.is_symlink() or not pending.is_file():raise ValueError(\'Unsafe journal temporary\')\n            pending.unlink()\n        journal_write(path,value)\n\n\ndef publish(attempt,target):\n    """Publish validated entries, retaining every replaced entry for recovery."""\n    no_symlink_path(target)\n    target.mkdir(parents=True,exist_ok=True)\n    entries=[p for p in sorted(attempt.iterdir()) if not p.name.startswith(\'.\')]\n    for entry in entries:\n        if entry.is_symlink() or (target/entry.name).is_symlink():raise ValueError(\'Refusing symlink publication: \'+entry.name)\n    history=target/(\'.history-\'+uuid.uuid4().hex)\n    history.mkdir()\n    journal_path=attempt/\'.publication_journal.json\'\n    state={\'status\':\'publishing\',\'target\':str(target),\'history\':str(history),\n           \'entries\':[{\'name\':entry.name,\'existed\':(target/entry.name).exists()} for entry in entries]}\n    journal_write(journal_path,state)\n    try:\n        for entry,record in zip(entries,state[\'entries\']):\n            destination=target/entry.name\n            if record[\'existed\']:os.replace(destination,history/entry.name)\n            os.replace(entry,destination)\n        state[\'status\']=\'complete\'\n        journal_write(journal_path,state)\n    except BaseException:\n        recover_publications(target)\n        raise\n    return history\n\n\ndef evaluate(model,bundle,attempt,args,manifest):\n    output=attempt/\'evaluated_results.json\'\n    command=[sys.executable]\n    selected=os.environ.get(\'STANDALONE_VENV_READY\')\n    if selected:command.extend([\'-I\',\'-S\',\'-B\'])\n    command.extend([str(model/\'evaluate.py\'),\'--environment\',\'current\',\'--bundle\',str(bundle),\'--output\',str(output),\n                    \'--parent-lock-owner\',str(os.getpid())])\n    if selected:command.extend([\'--venv\',selected])\n    environment=dict(os.environ,SEGMENTATION_MODEL_LOCK=str(os.getpid())+\':\'+str(model))\n    completed=subprocess.run(command,cwd=args.project_root,env=environment,capture_output=True,text=True,encoding=\'utf-8\',errors=\'replace\')\n    (attempt/\'evaluation.log\').write_text(completed.stdout+completed.stderr,encoding=\'utf-8\')\n    if completed.returncode:raise RuntimeError(\'Evaluation failed with exit \'+str(completed.returncode))\n    result=json.loads(output.read_text(encoding=\'utf-8\'))\n    metadata=result.get(\'metadata\') or {}\n    if (result.get(\'status\')!=\'complete\' or (result.get(\'run_id\') or metadata.get(\'run_id\'))!=manifest[\'run_id\']\n        or (result.get(\'model_id\') or metadata.get(\'model_id\'))!=model.name):\n        raise ValueError(\'Evaluation result identity does not match staged inference\')\n    reported=rooted(result.get(\'bundle\',\'\'),model)\n    if reported!=bundle.resolve():raise ValueError(\'Evaluation returned a different bundle\')\n    os.replace(output,attempt/\'results.json\')\n    shutil.copyfile(attempt/\'results.json\',attempt/\'last_attempt.json\')\n\n\ndef merge_measurement_records(stored,reported,label):\n    """Keep complementary recorded context while rejecting shared-value conflicts."""\n    if not isinstance(stored,dict) or not isinstance(reported,dict):\n        raise ValueError(\'Measurement record must be an object: \'+label)\n    merged=copy.deepcopy(stored)\n    for key,value in reported.items():\n        if key in stored:\n            if isinstance(stored[key],dict) and isinstance(value,dict):\n                merged[key]=merge_measurement_records(stored[key],value,label+\'.\'+key)\n                continue\n            if json.dumps(stored[key],sort_keys=True,allow_nan=False)!=json.dumps(value,sort_keys=True,allow_nan=False):\n                raise ValueError(\'Prior measurement conflicts with selected result: \'+label+\'.\'+key)\n        merged[key]=copy.deepcopy(value)\n    return merged\n\n\ndef preserve_prior_measurement(common,export,module,model,target,manifest,cache):\n    """Keep already validated measurements for the exact same cached prediction run."""\n    current_path=target/\'results.json\'\n    if not current_path.is_file():return None\n    current=common[\'read_json\'](current_path)\n    metadata=current.get(\'metadata\') or {}\n    if (current.get(\'run_id\') or metadata.get(\'run_id\'))!=manifest[\'run_id\']:return None\n    bundle=target/\'coco_bundle\'\n    if not bundle.is_dir():return None\n    previous_bundle,_=export[\'_validate_bundle\'](module,bundle,2)\n    prior=previous_bundle.get(\'metadata\') or {}\n    if (current.get(\'status\')==\'cached\' and current.get(\'model_id\')==model.name\n            and current.get(\'run_id\')==manifest[\'run_id\']\n            and prior.get(\'model_id\')==model.name\n            and isinstance(prior.get(\'run_id\'),str) and prior[\'run_id\']\n            and prior[\'run_id\']!=manifest[\'run_id\']):\n        # Inference-only publication intentionally retains the preceding bundle.\n        # Its measurements describe another run and cannot accompany this cache.\n        return None\n    if (prior.get(\'run_id\')!=manifest[\'run_id\'] or prior.get(\'model_id\')!=model.name\n            or prior.get(\'source_manifest_sha256\')!=manifest[\'request\'][\'source_manifest_sha256\']):\n        raise ValueError(\'Prior exported bundle has different prediction identity\')\n    timing=common[\'read_json\'](target/\'timing.json\')\n    environment=common[\'read_json\'](target/\'environment.json\')\n    for record in (current,metadata):\n        if record.get(\'timing\') is not None:\n            timing=merge_measurement_records(timing,record[\'timing\'],\'timing\')\n        if record.get(\'environment\') is not None:\n            environment=merge_measurement_records(environment,record[\'environment\'],\'environment\')\n    digest=common[\'sha256\'](bundle/\'manifest.json\')\n    measured_digest=None\n    override_path=model/\'result/forward_timing.json\'\n    if override_path.is_file():\n        override=common[\'read_json\'](override_path)\n        if override.get(\'original_run_id\')==manifest[\'run_id\']:\n            if (override.get(\'schema_version\')!=\'segmentation-forward-timing-override/v1\'\n                    or override.get(\'model_id\')!=model.name\n                    or override.get(\'source_manifest_sha256\')!=manifest[\'request\'][\'source_manifest_sha256\']):\n                raise ValueError(\'Forward timing override is not bound to the prior bundle\')\n            measured_digest=override.get(\'bundle_manifest_sha256\')\n            if measured_digest!=digest:\n                ancestry=prior.get(\'measurement_preservation\') or {}\n                if (ancestry.get(\'kind\')!=\'same_cached_run_reexport\'\n                        or ancestry.get(\'measured_bundle_manifest_sha256\')!=measured_digest\n                        or ancestry.get(\'run_id\')!=manifest[\'run_id\']\n                        or ancestry.get(\'source_manifest_sha256\')!=manifest[\'request\'][\'source_manifest_sha256\']):\n                    raise ValueError(\'Reexport lost the measured bundle ancestry\')\n                originals=[path for path in target.glob(\'.history-*/coco_bundle/manifest.json\')\n                           if not path.is_symlink() and common[\'sha256\'](path)==measured_digest]\n                if not originals:raise ValueError(\'Measured original bundle is absent from retained history\')\n                original_bundle,_=export[\'_validate_bundle\'](module,originals[0].parent,2)\n                original_metadata=original_bundle.get(\'metadata\') or {}\n                if (original_metadata.get(\'run_id\')!=manifest[\'run_id\']\n                        or original_metadata.get(\'source_manifest_sha256\')!=manifest[\'request\'][\'source_manifest_sha256\']):\n                    raise ValueError(\'Retained measured bundle identity differs\')\n            if any(timing.get(k)!=v for k,v in override[\'forward_fields\'].items()):\n                raise ValueError(\'Prior timing differs from its verified forward fields\')\n    return {\'timing\':timing,\'environment\':environment,\n            \'provenance\':{\'kind\':\'same_cached_run_reexport\',\'run_id\':manifest[\'run_id\'],\n                          \'cache_manifest_sha256\':common[\'sha256\'](cache/\'manifest.json\'),\n                          \'previous_bundle_manifest_sha256\':digest,\n                          \'measured_bundle_manifest_sha256\':measured_digest,\n                          \'source_manifest_sha256\':manifest[\'request\'][\'source_manifest_sha256\']}}\n\n\ndef apply_prior_measurement(common,attempt,preserved):\n    for filename in (\'results.json\',\'last_attempt.json\',\'coco_bundle/manifest.json\'):\n        path=attempt/filename\n        value=common[\'read_json\'](path)\n        metadata=value.setdefault(\'metadata\',{})\n        metadata.update(timing=copy.deepcopy(preserved[\'timing\']),environment=copy.deepcopy(preserved[\'environment\']),\n                        measurement_preservation=preserved[\'provenance\'])\n        if filename!=\'coco_bundle/manifest.json\':\n            value.update(timing=copy.deepcopy(preserved[\'timing\']),environment=copy.deepcopy(preserved[\'environment\']))\n        common[\'atomic_json\'](path,value)\n    for filename in (\'timing.json\',\'environment.json\'):\n        common[\'atomic_json\'](attempt/filename,preserved[filename.removesuffix(\'.json\')])\n    index=common[\'read_json\'](attempt/\'inference_time.json\')\n    for record in index.get(\'runs\',[]):\n        if record.get(\'run_id\')==preserved[\'provenance\'][\'run_id\']:\n            record[\'timing\']=copy.deepcopy(preserved[\'timing\'])\n    common[\'atomic_json\'](attempt/\'inference_time.json\',index)\n\n\ndef main(module,argv=None):\n    args=parser().parse_args(argv)\n    model=Path(module.__file__).resolve().parent\n    project=model.parents[1]\n    attempt=None\n    old_termination=signal.getsignal(signal.SIGTERM)\n    def terminated(signum,frame):raise KeyboardInterrupt(\'Model stage received SIGTERM\')\n    signal.signal(signal.SIGTERM,terminated)\n    try:\n        args.project_root=project\n        args._worker=model.name\n        args.cache_root=args.bundle_root=None\n        args.result_dir=output_directory(model,args.result_dir,project)\n        if args.export_mode==\'full\' and args.prediction_images!=\'none\':\n            raise ValueError(\'Full export includes all images; --prediction-images must be none\')\n        if args.venv is not None and not args.dry_run:\n            selected=rooted(args.venv,project)\n            if os.environ.get(\'STANDALONE_VENV_READY\')!=str(selected):\n                raise ValueError(\'Explicit --venv requires the safe runtime bootstrap\')\n        if args.stage==\'inference\' and args.evaluate:raise ValueError(\'--evaluate requires export or full stage\')\n        config=json.loads((model/\'weights/model.json\').read_text(encoding=\'utf-8\'))\n        args.dataset_root=rooted(args.dataset_root,project) if args.dataset_root is not None else rooted(config.get(\'dataset_root\',\'../../../../dataset/segmentation\'),model)\n        if args.dry_run:\n            print(json.dumps({\'model_id\':model.name,\'stage\':args.stage,\'dataset_root\':str(args.dataset_root),\n                \'result_dir\':str(args.result_dir),\'bundle\':str(args.result_dir/\'coco_bundle\'),\n                \'environment\':\'current\',\'evaluation\':args.evaluate or (args.stage==\'full\' and not args.skip_evaluation),\n                \'execution_verified\':False},ensure_ascii=False,indent=2))\n            return 0\n        if args.check:\n            report=module.check_model(model,dataset_root=args.dataset_root)\n            report[\'runtime_exists\']=Path(sys.executable).is_file()\n            report[\'selected_executable\']=sys.executable\n            print(json.dumps(report,ensure_ascii=False,indent=2))\n            return 0 if report[\'paths_ready\'] else 2\n        with lock_model(model):\n            recover_publications(model/\'result\')\n            if args.result_dir!=model/\'result\':recover_publications(args.result_dir)\n            args.result_dir.mkdir(parents=True,exist_ok=True)\n            attempt=Path(tempfile.mkdtemp(prefix=\'.attempt-\',dir=args.result_dir))\n            common,export=engines(module,args,attempt)\n            canonical_cache=model/\'result/prediction_cache\'\n            existing_cache=args.result_dir/\'prediction_cache\'\n            if not existing_cache.is_dir():existing_cache=canonical_cache\n            manifest=None\n            reused=False\n            if args.stage!=\'export\' and not args.force and existing_cache.is_dir():\n                previous=common[\'read_cache\'](existing_cache)\n                request,_=common[\'make_request\'](model,config,args)\n                reused=validate_cache(common,model,existing_cache,previous,args,request,allow_stale=True)\n                if reused:manifest=previous\n            if args.stage!=\'export\' and not reused:\n                args._cache_directory=attempt/\'prediction_cache\'\n                if common[\'inference_worker\'](args):raise RuntimeError(\'Inference failed; details retained in \'+str(attempt))\n                manifest=common[\'read_cache\'](args._cache_directory)\n            else:\n                args._cache_directory=existing_cache\n                manifest=manifest or common[\'read_cache\'](existing_cache)\n                validate_cache(common,model,existing_cache,manifest,args)\n            if args.stage==\'inference\' and reused:\n                attempt.rmdir()\n                print(\'SKIP \'+model.name+\': matching verified prediction cache\')\n                return 0\n            if args.stage in {\'export\',\'full\'}:\n                preserved=preserve_prior_measurement(common,export,module,model,args.result_dir,manifest,args._cache_directory)\n                if export[\'export_worker\'](args):raise RuntimeError(\'Export failed; details retained in \'+str(attempt))\n                if preserved:apply_prior_measurement(common,attempt,preserved)\n                export[\'_validate_bundle\'](module,attempt/\'coco_bundle\',args.verify_workers)\n                if args.evaluate or (args.stage==\'full\' and not args.skip_evaluation):\n                    evaluate(model,attempt/\'coco_bundle\',attempt,args,manifest)\n            for record_name in (\'results.json\',\'last_attempt.json\'):\n                record_path=attempt/record_name\n                if record_path.is_file():\n                    record=common[\'read_json\'](record_path)\n                    record[\'request\']=manifest[\'request\']\n                    common[\'atomic_json\'](record_path,record)\n            normalize_records(attempt,args.result_dir,model,args._cache_directory)\n            if args.stage in {\'export\',\'full\'} and preserved:\n                for filename in (\'timing.json\',\'environment.json\'):\n                    (attempt/filename).unlink(missing_ok=True)\n            history=publish(attempt,args.result_dir)\n            print(json.dumps({\'status\':\'complete\',\'stage\':args.stage,\'model_id\':model.name,\n                \'run_id\':manifest[\'run_id\'],\'result_dir\':str(args.result_dir),\'previous_outputs\':str(history)},ensure_ascii=False))\n            return 0\n    except BaseException as error:\n        if isinstance(error,SystemExit):raise\n        if attempt is not None and attempt.is_dir():\n            (attempt/\'attempt_failure.json\').write_text(json.dumps({\'status\':\'failed\',\'error\':str(error),\n                \'error_type\':type(error).__name__,\'traceback\':traceback.format_exc()},ensure_ascii=False,indent=2)+\'\\n\',encoding=\'utf-8\')\n        print(\'FAILED \'+model.name+\': \'+str(error),file=sys.stderr)\n        return 130 if isinstance(error,KeyboardInterrupt) else 1\n    finally:\n        signal.signal(signal.SIGTERM,old_termination)\n', __file__ + ":stages", "exec"), _standalone_runtime_ns)
if __name__ == "__main__":
    if any(value in _standalone_sys.argv[1:] for value in ("--help", "-h", "--dry-run")):
        raise SystemExit(_standalone_runtime_ns["main"](_standalone_sys.modules[__name__]))
    if "--stage=export" in _standalone_sys.argv[1:] or any(_standalone_sys.argv[i:i+2] == ["--stage", "export"] for i in range(1, len(_standalone_sys.argv)-1)):
        _standalone_os.environ["CUDA_VISIBLE_DEVICES"] = ""
    _standalone_project = __import__("pathlib").Path(__file__).resolve().parents[2]
    _standalone_selected_env = _activate_project_venv(_standalone_sys.argv[1:], _standalone_project)
    _prepare_library_loader(_standalone_selected_env, __file__)
    _standalone_stage = _standalone_runtime_ns["parser"]().parse_args().stage
    _prepare_model_runtime(_standalone_project, __import__("pathlib").Path(__file__).resolve().parent.name, _standalone_stage)

"""Independent inference data, export, runtime and validation code."""
import json
import os
from pathlib import Path
import sys

sys.dont_write_bytecode = True

def result_directory(model_dir):
    """Return the single owned directory for this model's runtime outputs."""
    directory = Path(model_dir).resolve() / 'result'
    if directory.is_symlink():
        raise ValueError(f'Result directory must not be a symlink: {directory}')
    directory.mkdir(parents=True, exist_ok=True)
    return directory

def prepare_runtime(mode, script):
    model = Path(script).resolve().parent
    config = json.loads((model / 'weights' / 'model.json').read_text())
    args = sys.argv[1:]
    current = '--environment=current' in args
    if '--environment' in args:
        index = args.index('--environment')
        current = index + 1 < len(args) and args[index + 1] == 'current'
    cache = result_directory(model) / '.cache'
    if cache.is_symlink():
        raise ValueError(f'Result cache must not be a symlink: {cache}')
    for key, sub in [('HF_MODULES_CACHE', 'hf_modules'), ('HF_HOME', 'huggingface'), ('MPLCONFIGDIR', 'matplotlib'), ('YOLO_CONFIG_DIR', 'ultralytics')]:
        target = cache / sub
        target.mkdir(parents=True, exist_ok=True)
        os.environ[key] = str(target)
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    os.environ['YOLO_OFFLINE'] = 'true'
    os.environ['YOLO_AUTOINSTALL'] = 'false'
    os.environ.setdefault('OMP_NUM_THREADS', '4')
    os.environ.setdefault('MKL_NUM_THREADS', '4')
    preferred = config.get('python' if mode == 'inference' else 'evaluation_python')
    marker = str(model) + ':' + mode
    if preferred and Path(preferred).is_file() and (not current) and (os.environ.get('MODEL_RUNTIME_SELECTED') != marker):
        environment = os.environ.copy()
        environment['MODEL_RUNTIME_SELECTED'] = marker
        preload = config.get('runtime_preload') if mode == 'inference' else None
        if preload and Path(preload).is_file():
            environment['LD_PRELOAD'] = ':'.join(dict.fromkeys(filter(None, [preload, *environment.get('LD_PRELOAD', '').split(':')])))
        if Path(preferred).absolute() != Path(sys.executable).absolute() or preload:
            print(f'[runtime] {preferred}', flush=True)
            os.execve(preferred, [preferred, str(Path(script).resolve()), *args], environment)
    return model

def evaluate_in_configured_environment(model_dir, bundle):
    """Evaluate an unpublished bundle with this model's evaluation interpreter.

    A temporary output keeps existing results unchanged until the caller commits
    the bundle. Child logs are replayed into the caller's normal run log.
    """
    import subprocess
    import tempfile
    model_dir = Path(model_dir).resolve()
    bundle = Path(bundle).resolve()
    environment = os.environ.copy()
    for marker in ('MODEL_RUNTIME_SELECTED', 'UNIFIED_EVALUATION_REEXEC'):
        environment.pop(marker, None)
    with tempfile.TemporaryDirectory(prefix='.evaluation-', dir=result_directory(model_dir)) as temporary:
        output = Path(temporary) / 'results.json'
        command = [sys.executable, str(model_dir / 'evaluate.py'), '--environment', 'auto', '--bundle', str(bundle), '--output', str(output)]
        completed = subprocess.run(command, cwd=model_dir, env=environment, capture_output=True, text=True, encoding='utf-8', errors='replace')
        if completed.stdout:
            print(completed.stdout, end='' if completed.stdout.endswith('\n') else '\n', flush=True)
        if completed.stderr:
            print(completed.stderr, end='' if completed.stderr.endswith('\n') else '\n', file=sys.stderr, flush=True)
        if completed.returncode:
            detail = (completed.stderr or completed.stdout).strip()
            raise RuntimeError(f'Configured evaluation failed (exit {completed.returncode}): {detail}')
        try:
            result = json.loads(output.read_text(encoding='utf-8'))
        except (OSError, ValueError) as exc:
            raise RuntimeError('Configured evaluation did not produce a valid result JSON') from exc
        if not isinstance(result, dict) or result.get('status') != 'complete':
            raise RuntimeError('Configured evaluation did not report complete status')
        if not isinstance(result.get('bundle'), str) or Path(result['bundle']).resolve() != bundle:
            raise RuntimeError('Configured evaluation returned results for a different bundle')
        return result

from dataclasses import dataclass, field
from typing import Any

@dataclass
class Sample:
    sample_id: str
    image: Any
    target: Any = None
    instances: list = field(default_factory=list)
    valid_mask: Any = None
    metadata: dict = field(default_factory=dict)

@dataclass
class Prediction:
    semantic: Any = None
    instances: list = field(default_factory=list)
    forward_seconds: float | None = None
    metadata: dict = field(default_factory=dict)
from collections import Counter, defaultdict
import csv
import hashlib
import io
import json
from pathlib import Path
import tarfile
import numpy as np
from PIL import Image, ImageDraw
DATASET_NAMES = ('ADE20K', 'ATR', 'AnimeSeg', 'BUSI', 'CAMUS', 'CASIA2', 'CHASE_DB1', 'COCO', 'CVC-ClinicDB', 'CelebAMask-HQ', 'Cityscapes', 'DF2023', 'DIS5K', 'DocLayNet', 'Fashion', 'MSD-Task09-Spleen', 'Manga109', 'OCTA', 'OmniDocBench', 'OpenEarthMap', 'Oxford-IIIT-Pet', 'P3M-10k', 'PCB-Defect', 'PPM-100', 'PhraseCut', 'Pothole', 'Real5-OmniDocBench', 'SemanticDrone', 'Teeth', 'ToonOut', 'UKI-Flood', 'VOC2012')
CITY_IDS = [7, 8, 11, 12, 13, 17, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 31, 32, 33]
CITY_NAMES = ['road', 'sidewalk', 'building', 'wall', 'fence', 'pole', 'traffic light', 'traffic sign', 'vegetation', 'terrain', 'sky', 'person', 'rider', 'car', 'truck', 'bus', 'train', 'motorcycle', 'bicycle']
ATR_NAMES = ['background', 'hat', 'hair', 'sunglasses', 'upper-clothes', 'skirt', 'pants', 'dress', 'belt', 'left-shoe', 'right-shoe', 'face', 'left-leg', 'right-leg', 'left-arm', 'right-arm', 'bag', 'scarf']
COMMON_NAMES = ['background', 'face', 'hair', 'top', 'dress', 'skirt', 'pants', 'belt', 'bag', 'hat', 'scarf', 'glasses', 'arms', 'legs', 'feet']
ATR_COMMON = np.asarray([0, 9, 2, 11, 3, 5, 6, 4, 7, 14, 14, 1, 13, 13, 12, 12, 8, 10])
FACE_PARTS = ['skin', 'nose', 'eye_g', 'l_eye', 'r_eye', 'l_brow', 'r_brow', 'l_ear', 'r_ear', 'mouth', 'u_lip', 'l_lip', 'hair', 'hat', 'ear_r', 'neck_l', 'neck', 'cloth']
COCO_DATASETS = {'COCO', 'PCB-Defect', 'Pothole', 'Manga109', 'DocLayNet'}
ALPHA_DATASETS = {'P3M-10k', 'PPM-100', 'AnimeSeg', 'Teeth'}

def _dataset_categories(names):
    return [{'id': index, 'name': name} for index, name in enumerate(names)]

def _eligible_box_annotations(records):
    """Native SAM prompt eligibility; official instance GT remains unfiltered."""
    counts = dict.fromkeys(('eligible', 'crowd', 'zero_area', 'invalid_box', 'missing_segmentation'), 0)
    selected = []
    for annotation in records:
        box = annotation.get('bbox')
        if int(annotation.get('iscrowd', 0)) != 0:
            reason = 'crowd'
        elif float(annotation.get('area', 0)) <= 0:
            reason = 'zero_area'
        elif not isinstance(box, list) or len(box) != 4 or float(box[2]) <= 0 or (float(box[3]) <= 0):
            reason = 'invalid_box'
        elif not annotation.get('segmentation'):
            reason = 'missing_segmentation'
        else:
            reason = 'eligible'
            selected.append(annotation)
        counts[reason] += 1
    return (selected, counts)

def _rle_mask(segmentation, height, width):
    """Decode native COCO compressed/uncompressed RLE without a lossy contour."""
    if tuple(segmentation['size']) != (height, width):
        raise ValueError('COCO RLE size differs from image dimensions')
    counts = segmentation['counts']
    if isinstance(counts, (str, bytes)):
        encoded = counts.decode('ascii') if isinstance(counts, bytes) else counts
        counts, position = ([], 0)
        while position < len(encoded):
            value, shift, more = (0, 0, True)
            while more:
                if position >= len(encoded):
                    raise ValueError('Truncated COCO RLE')
                code = ord(encoded[position]) - 48
                position += 1
                value |= (code & 31) << 5 * shift
                more = bool(code & 32)
                shift += 1
                if not more and code & 16:
                    value |= -1 << 5 * shift
            if len(counts) > 2:
                value += counts[-2]
            counts.append(value)
    if any((int(n) < 0 for n in counts)) or sum(counts) != height * width:
        raise ValueError('Invalid COCO RLE counts')
    flat = np.zeros(height * width, dtype=bool)
    offset = 0
    for index, count in enumerate(counts):
        if index % 2:
            flat[offset:offset + int(count)] = True
        offset += int(count)
    return flat.reshape((height, width), order='F')

def _coco_mask(segmentation, height, width):
    if isinstance(segmentation, dict):
        return _rle_mask(segmentation, height, width)
    if not isinstance(segmentation, list) or not segmentation:
        raise ValueError('Instance annotation has no native segmentation')
    try:
        from pycocotools import mask as mask_api
    except ImportError as exc:
        raise RuntimeError('COCO polygon rasterization requires pycocotools; install Evaluation requirements') from exc
    rles = mask_api.frPyObjects(segmentation, height, width)
    return mask_api.decode(mask_api.merge(rles)).astype(bool)

class ManifestDataset:

    def __init__(self, root, name, split=None, config=None):
        if name not in DATASET_NAMES:
            raise ValueError(f'Unsupported dataset: {name}')
        self.name = name
        supplied = Path(root).expanduser().resolve()
        if supplied.is_file():
            supplied = supplied.parent
        self.root = supplied if (supplied / 'evaluation_samples.jsonl').is_file() else supplied / name
        self.config = dict(config or {})
        self.view = self.config.get('label_view', 'native')
        allowed_views = {'native'}
        if name == 'ATR':
            allowed_views |= {'atr_common15', 'atr_human24_common15'}
        if name == 'COCO':
            allowed_views |= {'instances', 'panoptic_semantic'}
        if name == 'OpenEarthMap':
            allowed_views.add('vegetation_binary')
        if self.view not in allowed_views:
            raise ValueError(f'Unknown label_view {self.view!r} for {name}')
        manifest_path = self.root / 'dataset_manifest.json'
        self.manifest = json.loads(manifest_path.read_text()) if manifest_path.is_file() else {}
        if self.manifest.get('dataset_id', name) != name:
            raise ValueError(f"Requested dataset_id {name!r} differs from manifest {self.manifest.get('dataset_id')!r}")
        manifest_bytes = (self.root / 'evaluation_samples.jsonl').read_bytes()
        expected_digest = self.manifest.get('sample_manifest_sha256')
        if expected_digest is not None and hashlib.sha256(manifest_bytes).hexdigest() != expected_digest:
            raise ValueError('sample_manifest_sha256 differs from evaluation_samples.jsonl')
        rows = [json.loads(line) for line in manifest_bytes.decode('utf-8').splitlines() if line.strip()]
        if 'sample_count' in self.manifest and self.manifest['sample_count'] != len(rows):
            raise ValueError(f"sample_count mismatch: manifest={self.manifest['sample_count']} actual={len(rows)}")
        if 'splits' in self.manifest:
            observed_splits = dict(Counter((row['split'] for row in rows)))
            expected_splits = {item['split']: item['sample_count'] for item in self.manifest['splits']}
            if len(expected_splits) != len(self.manifest['splits']) or observed_splits != expected_splits:
                raise ValueError(f'split sample counts mismatch: manifest={expected_splits} actual={observed_splits}')
        if len({row['sample_id'] for row in rows}) != len(rows):
            raise ValueError('Duplicate sample_id in dataset manifest')
        chosen_splits = {split} if isinstance(split, str) else set(split or [])
        self.samples = [row for row in rows if not chosen_splits or row['split'] in chosen_splits]
        available = {row['split'] for row in rows}
        if not self.samples or chosen_splits - available:
            raise ValueError(f'Empty or unavailable split {split!r}; available splits: {sorted(available)}')
        self.splits = sorted({row['split'] for row in self.samples})
        self._json_cache = {}
        self._coco_indices = {}
        self.task = 'binary'
        self.ap_supported = False
        self.ap_reason = 'Native annotations do not define individual instances; mask AP is not applicable.'
        self.categories = _dataset_categories(['background', 'foreground'])
        self._setup_semantics()

    def __len__(self):
        return len(self.samples)

    def _json(self, relative_path):
        path = self.root / relative_path
        if relative_path not in self._json_cache:
            self._json_cache[relative_path] = json.loads(path.read_text(encoding='utf-8-sig'))
        return self._json_cache[relative_path]

    def _setup_semantics(self):
        name = self.name
        names = None
        if name == 'ADE20K':
            names = [f'ade20k_{i + 1}' for i in range(150)]
        elif name == 'ATR':
            names = COMMON_NAMES if self.view != 'native' else ATR_NAMES
        elif name == 'CelebAMask-HQ':
            names = ['background', *FACE_PARTS]
        elif name == 'Cityscapes':
            names = CITY_NAMES
        elif name == 'Fashion':
            labels = self._json('id2label.json')
            self.categories = [{'id': int(key), 'name': value} for key, value in labels.items()]
            self.task = 'semantic'
        elif name == 'VOC2012':
            names = ['background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']
        elif name == 'Oxford-IIIT-Pet':
            names = ['pet', 'background', 'border']
        elif name == 'CAMUS':
            names = ['background', 'lv_endo', 'myocardium', 'left_atrium']
        elif name == 'OpenEarthMap' and self.view == 'native':
            names = ['unlabelled', 'bareland', 'rangeland', 'developed_space', 'road', 'tree', 'water', 'agriculture_land', 'building']
        elif name == 'SemanticDrone':
            with (self.root / 'source_metadata/class_dict.csv').open(newline='', encoding='utf-8-sig') as stream:
                reader = csv.DictReader(stream, skipinitialspace=True)
                self._palette = [(row['name'].strip(), tuple((int(row[key]) for key in ('r', 'g', 'b')))) for row in reader]
            names = [row[0] for row in self._palette]
        if names is not None:
            self.categories, self.task = (_dataset_categories(names), 'semantic')
        if name in COCO_DATASETS:
            self.task = 'panoptic' if self.view == 'panoptic_semantic' else 'instance'
            role = 'panoptic_annotations' if self.task == 'panoptic' else 'instance_annotations'
            row = self.samples[0]
            assets = [a for a in row['assets'] if a['path'].endswith('.json')]
            asset = next((a for a in assets if a['role'] == role), assets[0])
            self.annotation_path = asset['path']
            data = self._json(self.annotation_path)
            self.categories = [dict(category) for category in data['categories']]
            if name == 'Manga109':
                self.categories = [c for c in self.categories if c['name'] == 'bubble']
            self._instance_category_ids = {category['id'] for category in self.categories}
            self.ap_supported = name != 'DocLayNet'
            self.ap_reason = '' if self.ap_supported else 'DocLayNet polygons are layout boxes, not pixel instance masks.'
            if self.config.get('class_agnostic'):
                if self.task != 'instance':
                    raise ValueError('class_agnostic requires an instance label view')
                self.categories = [{'id': 1, 'name': 'foreground'}]
        if name in {'OmniDocBench', 'Real5-OmniDocBench'}:
            self.task = 'instance'
            self.ap_reason = 'Document layout polygons describe page regions; a pixel instance mask benchmark is not defined.'
            self.categories = []

    def _open_image(self, asset, mode=None):
        path = self.root / asset['path']
        if asset.get('archive_member'):
            with tarfile.open(path) as archive:
                stream = archive.extractfile(asset['archive_member'])
                if stream is None:
                    raise FileNotFoundError(asset['archive_member'])
                source = io.BytesIO(stream.read())
        else:
            source = path
        with Image.open(source) as opened:
            return opened.convert(mode) if mode else opened.copy()

    def _volume(self, asset):
        try:
            import SimpleITK as sitk
        except ImportError as exc:
            raise RuntimeError('NIfTI datasets require SimpleITK') from exc
        image = sitk.ReadImage(str(self.root / asset['path']))
        array = sitk.GetArrayFromImage(image)
        return (array, {'spacing_zyx': list(image.GetSpacing()[::-1]), 'origin': list(image.GetOrigin()), 'direction': list(image.GetDirection()), 'axis_order': 'zyx', 'native_dimension': image.GetDimension()})

    def load(self, index):
        row = self.samples[index]
        by_role = defaultdict(list)
        metadata = dict(row.get('metadata', {}))
        metadata.update(dataset_id=self.name, split=row['split'], source_sample_id=row.get('source_sample_id'), dataset_root=str(self.root), label_view=self.view, source_assets=row['assets'])
        if self.config.get('text_prompt') is not None:
            metadata.update(text_prompt=str(self.config['text_prompt']), prompt_source='protocol_fixed_text')
        for asset in row['assets']:
            by_role[asset['role']].append(asset)
            metadata.setdefault(f"{asset['role']}_path", str(self.root / asset['path']))
        if self.name in {'OCTA', 'MSD-Task09-Spleen', 'CAMUS'}:
            sample = self._load_medical(row, by_role, metadata)
        elif self.name == 'UKI-Flood':
            sample = self._load_uki(row, by_role, metadata)
        elif 'image_and_mask_float32' in by_role:
            with np.load(self.root / by_role['image_and_mask_float32'][0]['path'], allow_pickle=False) as archive:
                image, alpha = (archive['image'].copy(), archive['mask'].squeeze(-1).copy())
            metadata['alpha'] = alpha
            metadata['image_encoding'] = 'float32_rgb_hwc_0_1'
            sample = Sample(row['sample_id'], image, (alpha >= self.config.get('gt_threshold', 0.5)).astype(np.uint8), metadata=metadata)
        else:
            image = self._open_image(by_role['image'][0], 'RGB')
            sample = Sample(row['sample_id'], image, metadata=metadata)
            if self.name in COCO_DATASETS:
                self._load_coco(sample, row, by_role)
            elif self.name in {'OmniDocBench', 'Real5-OmniDocBench'}:
                record = self._json(by_role['annotation'][0]['path'])[metadata['annotation_record_index']]
                metadata['document_annotation'] = record
                metadata['annotation_coordinate_size'] = [record['page_info']['height'], record['page_info']['width']]
            elif self.name == 'PhraseCut':
                record = self._json(by_role['annotation'][0]['path'])[metadata['annotation_record_index']]
                canvas = Image.new('1', image.size)
                draw = ImageDraw.Draw(canvas)
                for instance_polygons in record['Polygons']:
                    for polygon in instance_polygons:
                        if len(polygon) >= 2:
                            draw.polygon([(int(x), int(y)) for x, y in polygon], fill=1)
                sample.target = np.asarray(canvas, dtype=np.uint8)
                metadata.update(text_prompt=record['phrase'], prompt_source='dataset_human_phrase', query_id=record['task_id'])
            elif self.name == 'CelebAMask-HQ':
                parts = {Path(a['path']).stem.split('_', 1)[1]: a for a in by_role['component_mask']}
                sample.target = None
                for class_id, part in enumerate(FACE_PARTS, start=1):
                    if part in parts:
                        component = np.asarray(self._open_image(parts[part]))
                        if component.ndim == 3:
                            component = component[..., 0]
                        if sample.target is None:
                            sample.target = np.zeros(component.shape, np.uint8)
                        sample.target[component != 0] = class_id
                if sample.target is None:
                    raise ValueError('No face component masks found')
            elif self.name == 'CASIA2' and metadata.get('label') == 0:
                sample.target = np.zeros((image.height, image.width), np.uint8)
            else:
                mask_asset = (by_role['mask'] or by_role['ground_truth'])[0]
                source = self._open_image(mask_asset)
                metadata['mask_path'] = str(self.root / mask_asset['path'])
                if self.name in {'CASIA2', 'DF2023'}:
                    if source.size != image.size:
                        metadata['native_mask_size'] = list(source.size)
                        metadata['gt_transform'] = 'native_VAAS_nearest_to_image; source alignment unverified'
                        source = source.resize(image.size, Image.Resampling.NEAREST)
                    raw = np.asarray(source.convert('L'))
                elif self.task == 'binary':
                    raw = np.asarray(source.convert('L'))
                else:
                    raw = np.asarray(source)
                sample.target, sample.valid_mask = self._target(raw, metadata)
        if sample.target is not None:
            if sample.valid_mask is None:
                sample.valid_mask = np.ones(sample.target.shape, dtype=bool)
            if self.config.get('gt_size') is not None:
                if sample.target.ndim != 2:
                    raise ValueError('gt_size is only supported for 2D annotation domains')
                height, width = map(int, self.config['gt_size'])
                metadata['native_gt_shape'] = list(sample.target.shape)
                sample.target = np.asarray(Image.fromarray(sample.target.astype(np.int32)).resize((width, height), Image.Resampling.NEAREST))
                sample.valid_mask = np.asarray(Image.fromarray(sample.valid_mask).resize((width, height), Image.Resampling.NEAREST))
                if 'alpha' in metadata:
                    metadata['alpha'] = np.asarray(Image.fromarray(metadata['alpha']).resize((width, height), Image.Resampling.BILINEAR))
                metadata['gt_transform'] = 'explicit_gt_size_nearest_labels'
            if self.config.get('prompt') == 'gt_box':
                ys, xs = np.where((sample.target != 0) & sample.valid_mask)
                exclusive = bool(self.config.get('box_max_exclusive', False))
                metadata['boxes_xyxy'] = [[int(xs.min()), int(ys.min()), int(xs.max()) + int(exclusive), int(ys.max()) + int(exclusive)]] if xs.size else []
                metadata['box_max_exclusive'] = exclusive
                metadata['prompt_source'] = 'ground_truth_oracle'
        return sample

    def _target(self, raw, metadata):
        name = self.name
        valid = np.ones(raw.shape[:2], dtype=bool)
        if name == 'ADE20K':
            if not np.isin(raw, [*range(151), 255]).all():
                raise ValueError('Unknown ADE20K GT label outside 0..150 and void255')
            valid = (raw >= 1) & (raw <= 150)
            target = np.where(valid, raw.astype(np.int16) - 1, 0)
        elif name == 'Cityscapes':
            target, valid = (np.zeros(raw.shape, np.uint8), np.isin(raw, CITY_IDS))
            for index, label_id in enumerate(CITY_IDS):
                target[raw == label_id] = index
        elif name == 'Oxford-IIIT-Pet':
            if not np.isin(raw, [1, 2, 3]).all():
                raise ValueError('Oxford trimaps must contain IDs 1, 2 and 3')
            target = raw - 1
        elif name == 'ATR':
            valid = raw != 255
            if np.any(valid & (raw >= 18)):
                raise ValueError('Unknown ATR label')
            target = np.where(valid, raw, 0)
            if self.view != 'native':
                if self.view == 'atr_human24_common15':
                    valid &= raw != 8
                target = np.where(valid, ATR_COMMON[target], 0)
        elif name == 'SemanticDrone':
            if raw.ndim == 2:
                target, valid = (raw, raw != 255)
            else:
                packed = raw[..., 0].astype(np.uint32) * 65536 + raw[..., 1].astype(np.uint32) * 256 + raw[..., 2]
                target = np.full(packed.shape, -1, np.int16)
                for index, (_, color) in enumerate(self._palette):
                    target[packed == color[0] * 65536 + color[1] * 256 + color[2]] = index
                if np.any(target < 0):
                    raise ValueError('Unknown SemanticDrone mask RGB color')
        elif name == 'Fashion':
            target, valid = (raw, raw != 0)
        elif name == 'VOC2012':
            target, valid = (np.where(raw != 255, raw, 0), raw != 255)
        elif name == 'OpenEarthMap':
            valid = raw != 0
            if self.view == 'vegetation_binary':
                target = np.isin(raw, [2, 5, 7]).astype(np.uint8)
            else:
                target = raw
        else:
            if name in ALPHA_DATASETS:
                alpha = raw.astype(np.float32) / 255.0
                metadata['alpha'] = alpha
                threshold = float(self.config.get('gt_threshold', 0.5))
                target = alpha >= threshold
            else:
                threshold = self.config.get('gt_threshold', 30 if name in {'CASIA2', 'DF2023'} else 0 if name == 'BUSI' else 127)
                target = raw > threshold
            metadata['gt_threshold'] = threshold
        target = np.asarray(target, dtype=np.int16)
        category_ids = [category['id'] for category in self.categories]
        if not np.isin(target[valid], category_ids).all():
            raise ValueError(f'GT labels outside declared {name} categories')
        return (target, valid)

    def _coco_index(self, annotation_path):
        data = self._json(annotation_path)
        if annotation_path not in self._coco_indices:
            annotations = defaultdict(list)
            for annotation in data['annotations']:
                annotations[annotation['image_id']].append(annotation)
            self._coco_indices[annotation_path] = ({im['id']: im for im in data['images']}, annotations)
        return self._coco_indices[annotation_path]

    def _load_coco(self, sample, row, by_role):
        images, annotations = self._coco_index(self.annotation_path)
        image_id = row['metadata'].get('annotation_image_id', row['metadata'].get('image_id'))
        if image_id is None:
            image_id = int(row['source_sample_id'])
        info = images[image_id]
        sample.metadata.update(annotation_image_id=image_id, native_annotations_path=str(self.root / self.annotation_path))
        if (sample.image.width, sample.image.height) != (info['width'], info['height']):
            raise ValueError('Image size differs from COCO annotation')
        records = annotations[image_id]
        sample.metadata['annotations'] = records
        if self.name == 'DocLayNet':
            return
        if self.task == 'panoptic':
            rgb = np.asarray(self._open_image(by_role['ground_truth'][0], 'RGB'), dtype=np.int64)
            ids = rgb[..., 0] + 256 * rgb[..., 1] + 65536 * rgb[..., 2]
            target = np.zeros(ids.shape, np.int16)
            valid = np.zeros(ids.shape, bool)
            for segment in records[0]['segments_info']:
                mask = ids == segment['id']
                target[mask] = segment['category_id']
                valid[mask] = True
            sample.target, sample.valid_mask = (target, valid)
            sample.metadata.update(panoptic_ids=ids, segments_info=records[0]['segments_info'])
            if not by_role['instance_annotations']:
                raise ValueError('COCO panoptic evaluation also requires original instance_annotations for AP')
            instance_path = by_role['instance_annotations'][0]['path']
            instance_images, instance_annotations = self._coco_index(instance_path)
            instance_info = instance_images[image_id]
            if (instance_info['width'], instance_info['height']) != sample.image.size:
                raise ValueError('COCO instance image geometry differs from panoptic geometry')
            records = instance_annotations[image_id]
        else:
            instance_path = self.annotation_path
        sample.metadata['native_instances_annotations_path'] = str(self.root / instance_path)
        allowed = self._instance_category_ids
        for annotation in records:
            if annotation['category_id'] not in allowed:
                continue
            sample.instances.append({'mask': _coco_mask(annotation['segmentation'], info['height'], info['width']), 'category_id': 1 if self.config.get('class_agnostic') else annotation['category_id'], 'iscrowd': annotation.get('iscrowd', 0), 'annotation_id': annotation['id'], 'bbox': annotation.get('bbox')})
        if self.config.get('class_agnostic'):
            sample.metadata.update(class_agnostic=True, coco_use_cats=0)
        if self.config.get('prompt') == 'gt_box':
            prompts = sample.instances
            if self.config.get('sam_box_filter'):
                selected, counts = _eligible_box_annotations(records)
                selected_ids = {a['id'] for a in selected}
                prompts = [a for a in sample.instances if a['annotation_id'] in selected_ids]
                sample.metadata.update(prompt_annotations=selected, sam_box_filter_counts=counts, instance_gt_protocol='official_unfiltered_instances', prompt_protocol='native_eligible_box_prompt_annotations')
            sample.metadata['boxes_xyxy'] = [[a['bbox'][0], a['bbox'][1], a['bbox'][0] + a['bbox'][2], a['bbox'][1] + a['bbox'][3]] for a in prompts]
            sample.metadata['prompt_annotation_ids'] = [a['annotation_id'] for a in prompts]
            sample.metadata['prompt_category_ids'] = [a['category_id'] for a in prompts]
            sample.metadata['prompt_source'] = 'ground_truth_oracle'

    def _load_medical(self, row, by_role, metadata):
        if self.name == 'CAMUS':
            frames, geometry = self._volume(by_role['image_sequence'][0])
            masks, _ = self._volume(by_role['mask_sequence'][0])
            ed_image, _ = self._volume(by_role['ed_image'][0])
            es_image, _ = self._volume(by_role['es_image'][0])
            ed_mask, _ = self._volume(by_role['ed_mask'][0])
            es_mask, _ = self._volume(by_role['es_mask'][0])
            if frames.shape != masks.shape:
                raise ValueError('CAMUS sequence image/GT geometry mismatch')
            ed_matches = [i for i, frame in enumerate(frames) if np.array_equal(frame, ed_image.squeeze())]
            es_matches = [i for i, frame in enumerate(frames) if np.array_equal(frame, es_image.squeeze())]
            if len(ed_matches) != 1 or len(es_matches) != 1 or {ed_matches[0], es_matches[0]} != {0, len(frames) - 1}:
                raise ValueError('CAMUS endpoints do not uniquely match sequence ends')
            order = list(range(len(frames))) if ed_matches[0] == 0 else list(range(len(frames) - 1, -1, -1))
            frames, masks = (frames[order], masks[order].copy())
            masks[0], masks[-1] = (ed_mask.squeeze(), es_mask.squeeze())
            valid = np.zeros(masks.shape, bool)
            role = self.config.get('gt_role', 'endpoints')
            if role == 'endpoints':
                valid[[0, -1]] = True
            elif role == 'es':
                valid[-1] = True
            elif role == 'sequence':
                valid[:] = True
            else:
                raise ValueError('CAMUS gt_role must be endpoints, es or sequence')
            metadata.update(geometry, axis_order='time_y_x', original_indices=order, gt_role=role, spacing_yx_mm=geometry['spacing_zyx'][-2:])
            if self.config.get('prompt') == 'ed_mask':
                metadata.update(prompt_masks=masks[0].copy(), prompt_frame_index=0, prompt_source='ground_truth_oracle')
            return Sample(row['sample_id'], frames, masks.astype(np.int16), valid_mask=valid, metadata=metadata)
        image, geometry = self._volume(by_role['image'][0])
        target, target_geometry = self._volume(by_role['mask'][0])
        if image.shape != target.shape or geometry != target_geometry:
            raise ValueError('NIfTI image/GT geometry mismatch')
        metadata.update(geometry)
        return Sample(row['sample_id'], image, (target == 1 if self.name == 'MSD-Task09-Spleen' else target > 0).astype(np.uint8), metadata=metadata)

    def _load_uki(self, row, by_role, metadata):
        image_path, mask_path = (self.root / by_role[role][0]['path'] for role in ('image', 'mask'))
        try:
            import rasterio
        except ImportError:
            import tifffile
            with tifffile.TiffFile(image_path) as source:
                image = source.asarray()
                metadata['geospatial'] = str(source.geotiff_metadata)
                metadata['tiff_axes'] = source.series[0].axes
            target = tifffile.imread(mask_path).squeeze()
            if image.shape[-1] == 9:
                image = image.transpose(2, 0, 1)
        else:
            with rasterio.open(image_path) as source:
                image = source.read(masked=True).filled(0)
                metadata['geospatial'] = {'crs': str(source.crs), 'transform': list(source.transform), 'nodata': source.nodata}
            with rasterio.open(mask_path) as source:
                target = source.read(1)
        if image.ndim != 3 or image.shape[0] != 9 or image.shape[1:] != target.shape:
            raise ValueError('UKI requires nine native bands aligned with label geometry')
        valid = np.isin(target, [0, 1])
        metadata.update(native_band_count=9, axis_order='cyx')
        return Sample(row['sample_id'], image, np.where(valid, target, 0).astype(np.uint8), valid_mask=valid, metadata=metadata)

def load_dataset(root, name, split=None, config=None):
    return ManifestDataset(root, name, split=split, config=config)
import hashlib
import json
from pathlib import Path
import shutil
import tarfile
import numpy as np
from PIL import Image, ImageDraw

def image_for_unit(sample, prediction, index):
    """Select the actual evaluation view, including medical slices and crops."""
    exact = prediction.metadata.get('evaluation_image')
    value = np.asarray(sample.image if exact is None else exact)
    display = {'source': 'sample.image' if exact is None else 'prediction.evaluation_image'}
    if exact is not None:
        display['transform'] = prediction.metadata.get('evaluation_image_transform', 'explicit evaluation image')
    axis = sample.metadata.get('axis_order')
    if index:
        value = value[index]
        display.update(axis_order=axis, slice_index=list(index))
        if sample.metadata.get('original_indices'):
            display['original_frame_index'] = sample.metadata['original_indices'][index[0]]
    elif axis == 'cyx' and exact is None:
        bands = [4, 3, 2] if sample.metadata.get('dataset_id') == 'UKI-Flood' else list(range(min(3, value.shape[0])))
        value = np.moveaxis(value[bands], 0, -1)
        display.update(axis_order=axis, band_indices=bands, purpose='multiband image preview')
        if sample.metadata.get('dataset_id') == 'UKI-Flood':
            display.update(band_names=['RED', 'GREEN', 'BLUE'], value_range=[0, 10000])
    return (value, display)

def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()

def _color(label):
    value = int(label) + 1 if int(label) < 0 else int(label)
    color = [0, 0, 0]
    for shift in range(8):
        for channel in range(3):
            color[channel] |= (value >> channel & 1) << 7 - shift
        value >>= 3
    return color

def _display_image(value, display):
    array = np.asarray(value)
    if array.ndim == 3 and array.shape[-1] == 1:
        array = array[..., 0]
    if array.ndim not in (2, 3) or (array.ndim == 3 and array.shape[-1] not in (3, 4)):
        raise ValueError(f'Image preview requires HW or HWC image: {array.shape}')
    details = dict(display or {})
    details.update(input_shape=list(array.shape), input_dtype=str(array.dtype))
    if array.dtype != np.uint8:
        finite = np.isfinite(array)
        finite_values = array[finite]
        if 'value_range' in details:
            low, high = map(float, details['value_range'])
        elif finite_values.size and finite_values.min() >= 0 and (finite_values.max() <= 1):
            low, high = (0.0, 1.0)
        else:
            low = float(finite_values.min()) if finite_values.size else 0.0
            high = float(finite_values.max()) if finite_values.size else 0.0
        safe = np.where(finite, array, low).astype(np.float64)
        array = np.rint(np.clip((safe - low) / (high - low), 0, 1) * 255).astype(np.uint8) if high > low else np.zeros(array.shape, np.uint8)
        details.update(value_range=[low, high], conversion='linear display scaling to uint8; original data preserved separately', nonfinite_pixels=int((~finite).sum()))
    return (Image.fromarray(array).convert('RGB'), details)

class ImageArtifacts:

    def __init__(self, output, categories, task, foreground_id=1):
        self.output = Path(output)
        self.categories = categories
        self.task = task
        self.foreground_id = foreground_id
        self.hashes = {}
        self._originals_cache = {}

    def _record(self, relative):
        self.hashes[relative] = _sha256(self.output / relative)
        return relative

    def _save(self, relative, image):
        path = self.output / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        image.save(path)
        return self._record(relative)

    def _originals(self, image_id, image, metadata):
        assets = [a for a in metadata.get('source_assets', []) if a.get('role') in {'image', 'image_sequence', 'image_and_mask_float32', 'ed_image', 'es_image'}]
        root = Path(metadata.get('dataset_root', '.'))
        if not assets and metadata.get('image_path'):
            assets = [{'role': 'image', 'path': metadata['image_path']}]
        key = json.dumps([str(root), assets], sort_keys=True) if assets else str(image_id)
        if key in self._originals_cache:
            return self._originals_cache[key]
        result = []
        source_key = hashlib.sha256(key.encode()).hexdigest()[:16]
        for index, asset in enumerate(assets):
            source = root / asset['path']
            member_name = asset.get('archive_member')
            filename = Path(member_name or asset['path']).name
            suffix = '.nii.gz' if filename.lower().endswith('.nii.gz') else Path(filename).suffix
            if len(suffix.encode()) > 16:
                suffix = '.bin'
            relative = f'images/original/{source_key}_{index}{suffix}'
            path = self.output / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            if member_name:
                with tarfile.open(source) as archive:
                    member = archive.getmember(member_name)
                    if not member.isfile():
                        raise ValueError(f'Original image archive member is not a regular file: {member_name}')
                    with archive.extractfile(member) as stream, path.open('wb') as out:
                        shutil.copyfileobj(stream, out)
            else:
                shutil.copyfile(source, path)
            self._record(relative)
            result.append({'path': relative, 'role': asset['role'], 'source_path': str(source), 'original_filename': filename, 'archive_member': member_name, 'copy': 'original bytes'})
        if not result:
            relative = self._save(f'images/original/{image_id:08d}.png', image)
            result.append({'path': relative, 'copy': 'decoded input preview; source file was not provided'})
        self._originals_cache[key] = result
        return result

    def _mask_images(self, role, stem, labels, instances, valid):
        if self.task == 'instance':
            labels = None
        available = labels is not None or self.task == 'instance' or bool(instances)
        if not available:
            return {'available': False, 'reason': 'No native pixel mask is defined for this sample/task.'}
        result = {'available': True}
        preview = np.zeros((*valid.shape, 3), np.uint8)
        if labels is not None:
            labels = np.asarray(labels)
            if self.task == 'binary':
                mask = (labels == self.foreground_id).astype(np.uint8) * 255
                result['mask'] = self._save(f'{role}/{stem}.png', Image.fromarray(mask))
                result['encoding'] = {'background': 0, 'foreground': 255, 'foreground_category_id': self.foreground_id}
                preview[:] = mask[..., None]
            else:
                for category in np.unique(labels):
                    preview[labels == category] = _color(category)
                if np.all(np.isfinite(labels)) and np.all(labels == labels.astype(np.int64)) and (labels.min() >= 0) and (labels.max() <= 65535):
                    label_image = Image.fromarray(labels.astype(np.uint16))
                    name = f'{role}/labels/{stem}.png'
                elif labels.min() >= -2147483648 and labels.max() <= 2147483647:
                    label_image = Image.fromarray(labels.astype(np.int32))
                    name = f'{role}/labels/{stem}.tiff'
                else:
                    raise ValueError('Class labels must fit a lossless integer image')
                result['labels'] = self._save(name, label_image)
                result['encoding'] = 'labels store original category IDs; preview uses the manifest palette'
        instance_images = []
        for index, item in enumerate(instances, 1):
            mask = np.asarray(item['mask'], bool)
            record = {'path': self._save(f'{role}/instances/{stem}_{index:04d}.png', Image.fromarray(mask.astype(np.uint8) * 255)), 'instance_index': index, 'category_id': int(item['category_id'])}
            for key in ('score', 'iscrowd'):
                if key in item:
                    record[key] = float(item[key]) if key == 'score' else int(item[key])
            if 'annotation_id' in item:
                record['source_annotation_id'] = int(item['annotation_id'])
            instance_images.append(record)
            if labels is None:
                preview[mask] = _color(index)
        if instances or self.task == 'instance':
            result['instances'] = instance_images
            result['instance_preview_note'] = 'Later instances cover earlier ones in preview only; separate PNGs and COCO RLE retain overlaps.'
        preview[~valid] = [255, 0, 255]
        result['preview'] = self._save(f'{role}/preview/{stem}.png' if 'mask' in result else f'{role}/{stem}.png', Image.fromarray(preview))
        return result

    def _layout_preview(self, role, stem, original_view, output_size, metadata):
        """Draw native document outlines only when their coordinate frame is known."""
        dataset = metadata.get('dataset_id')
        if role == 'prediction':
            records = metadata.get('raw_predictions')
            kind = 'native_prediction'
        elif dataset == 'DocLayNet':
            records = metadata.get('annotations')
            kind = 'coco_xywh'
        else:
            document = metadata.get('document_annotation')
            if not isinstance(document, dict):
                return {'layout_preview_reason': 'No native layout annotations are available.'}
            coordinate_size = metadata.get('annotation_coordinate_size')
            actual_size = [original_view.height, original_view.width]
            if coordinate_size != actual_size:
                return {'layout_preview_reason': f'Annotation coordinate size {coordinate_size} differs from image {actual_size}; no verified alignment transform is available.'}
            if dataset == 'Real5-OmniDocBench':
                return {'layout_preview_reason': 'Real5 annotation coordinates refer to the original page; no verified deformation transform is available even when dimensions match.'}
            if dataset != 'OmniDocBench':
                return {'layout_preview_reason': 'Native annotation coordinate convention is not defined for this dataset.'}
            records = document.get('layout_dets')
            kind = 'omni_polygon'
        if not isinstance(records, list):
            return {'layout_preview_reason': 'Native layout records are unavailable or have an unsupported format.'}
        canvas = original_view.copy()
        draw = ImageDraw.Draw(canvas)
        drawn = skipped = ignored = 0
        for record in records:
            if not isinstance(record, dict):
                skipped += 1
                continue
            if kind == 'coco_xywh':
                values = record.get('bbox')
                coordinate_kind = 'xywh'
            elif kind == 'omni_polygon':
                values = record.get('poly')
                coordinate_kind = 'polygon'
            else:
                values = record.get('coordinate', record.get('poly'))
                coordinate_kind = 'xyxy' if 'coordinate' in record and isinstance(values, list) and (len(values) == 4) else 'polygon'
            if not isinstance(values, list) or any((not isinstance(x, (int, float, np.number)) or isinstance(x, (bool, np.bool_)) for x in values)):
                skipped += 1
                continue
            values = np.asarray(values, dtype=np.float64)
            if values.ndim != 1 or not np.isfinite(values).all():
                skipped += 1
                continue
            if coordinate_kind in {'xywh', 'xyxy'} and len(values) == 4:
                x1, y1, a, b = values
                x2, y2 = (x1 + a, y1 + b) if coordinate_kind == 'xywh' else (a, b)
                if x2 <= x1 or y2 <= y1:
                    skipped += 1
                    continue
                points = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
            elif coordinate_kind == 'polygon' and len(values) >= 6 and (len(values) % 2 == 0):
                points = list(zip(values[::2], values[1::2]))
            else:
                skipped += 1
                continue
            is_ignored = bool(record.get('ignore', False))
            color = (128, 128, 128) if is_ignored else (0, 160, 0) if role == 'ground_truth' else (220, 0, 0)
            draw.line(points + [points[0]], fill=color, width=max(1, min(canvas.size) // 300))
            drawn += 1
            ignored += int(is_ignored)
        if records and (not drawn):
            return {'layout_preview_reason': f'All {len(records)} layout records have unsupported or invalid coordinates; no inferred boxes were drawn.', 'layout_preview_annotations_skipped': skipped}
        if canvas.size != output_size:
            canvas = canvas.resize(output_size, Image.Resampling.BILINEAR)
        directory = 'ground_truth' if role == 'ground_truth' else 'predictions'
        return {'layout_preview': self._save(f'{directory}/layout/{stem}.png', canvas), 'layout_preview_note': 'Native layout outlines only, not segmentation masks. Drawn in original image coordinates, then resized only for display; ignored annotations are gray.', 'layout_preview_annotations_drawn': drawn, 'layout_preview_annotations_skipped': skipped, 'layout_preview_ignored_annotations': ignored}

    def export(self, image_id, image, target, prediction, valid, gt_instances, pred_instances, metadata):
        metadata = metadata or {}
        valid = np.asarray(valid, bool)
        preview, display = _display_image(image, metadata.get('image_display'))
        original_view = preview
        originals = self._originals(image_id, preview, metadata)
        height, width = valid.shape
        display['evaluation_shape'] = [height, width]
        if preview.size != (width, height):
            display['geometry'] = {'operation': 'resize preview to evaluation grid', 'interpolation': 'bilinear', 'from_size': list(preview.size), 'to_size': [width, height]}
            preview = preview.resize((width, height), Image.Resampling.BILINEAR)
        else:
            display['geometry'] = {'operation': 'unchanged'}
        stem = f'{image_id:08d}'
        result = {'image': self._save(f'images/{stem}.png', preview), 'originals': originals, 'image_display': display, 'valid_mask': self._save(f'valid_masks/{stem}.png', Image.fromarray(valid.astype(np.uint8) * 255)), 'valid_encoding': '255=evaluated, 0=ignored; preview marks ignored pixels magenta', 'ground_truth': self._mask_images('ground_truth', stem, target, gt_instances, valid), 'prediction': self._mask_images('predictions', stem, prediction, pred_instances, valid)}
        if self.task == 'unsupported':
            for role in ('ground_truth', 'prediction'):
                if not result[role]['available']:
                    result[role].update(self._layout_preview(role, stem, original_view, (width, height), metadata))
        return result

    def description(self):
        return {'version': 1, 'authoritative_metrics': 'JSON/RLE; pictures are exported from the same final masks', 'category_colors': [{'id': c['id'], 'name': c.get('name', ''), 'rgb': _color(c['id'])} for c in self.categories], 'source_images': 'images/original contains byte copies when source assets are provided', 'evaluation_images': 'images/*.png matches COCO image width/height; conversions recorded per image', 'mask_images': 'binary 0/255; semantic labels preserve IDs; each instance has its own 0/255 PNG', 'layout_previews': 'Native bbox/polygon outlines where the coordinate frame is known; mask availability and segmentation metrics remain unchanged.'}
import operator
import numpy as np

def encode_rle(mask) -> dict:
    """Encode a 2D mask; zero pixels are background, nonzero foreground."""
    array = np.asarray(mask)
    if array.ndim != 2:
        raise ValueError(f'COCO masks must be 2D, received shape {array.shape}')
    flat = np.asarray(array, dtype=bool).ravel(order='F')
    if flat.size:
        changes = np.flatnonzero(flat[1:] != flat[:-1]) + 1
        runs = np.diff(np.concatenate(([0], changes, [flat.size]))).tolist()
        if flat[0]:
            runs.insert(0, 0)
    else:
        runs = [0]
    characters = []
    for index, run in enumerate(runs):
        value = int(run) - (int(runs[index - 2]) if index > 2 else 0)
        while True:
            group = value & 31
            value >>= 5
            more = value != (-1 if group & 16 else 0)
            characters.append(chr((group | (32 if more else 0)) + 48))
            if not more:
                break
    return {'size': [int(array.shape[0]), int(array.shape[1])], 'counts': ''.join(characters)}

def _integer(value, role):
    try:
        if isinstance(value, (bool, np.bool_)):
            raise TypeError
        result = operator.index(value)
    except TypeError as exc:
        raise ValueError(f'{role} must be an integer') from exc
    if result < 0:
        raise ValueError(f'{role} must be nonnegative')
    return int(result)

def _compressed_runs(text):
    if isinstance(text, bytes):
        try:
            text = text.decode('ascii')
        except UnicodeDecodeError as exc:
            raise ValueError('COCO RLE counts must contain ASCII characters') from exc
    runs = []
    position = 0
    while position < len(text):
        value, shift = (0, 0)
        while True:
            if position == len(text):
                raise ValueError('Truncated COCO RLE integer')
            group = ord(text[position]) - 48
            position += 1
            if not 0 <= group <= 63:
                raise ValueError('Invalid character in COCO RLE counts')
            value |= (group & 31) << shift
            shift += 5
            if not group & 32:
                if group & 16:
                    value |= -1 << shift
                break
        if len(runs) > 2:
            value += runs[-2]
        if value < 0:
            raise ValueError('COCO RLE contains a negative run length')
        runs.append(value)
    return runs

def decode_rle(rle) -> np.ndarray:
    """Decode compressed string/bytes or uncompressed counts to a bool mask."""
    try:
        size, counts = (rle['size'], rle['counts'])
        if len(size) != 2:
            raise ValueError('COCO RLE size must contain height and width')
    except (KeyError, TypeError) as exc:
        raise ValueError('COCO RLE requires size and counts') from exc
    height = _integer(size[0], 'height')
    width = _integer(size[1], 'width')
    if isinstance(counts, (str, bytes)):
        runs = _compressed_runs(counts)
    elif isinstance(counts, (list, tuple, np.ndarray)):
        runs = [_integer(value, 'run length') for value in counts]
    else:
        raise ValueError('COCO RLE counts must be a string, bytes, or integer sequence')
    total = height * width
    if sum(runs) != total:
        raise ValueError('COCO RLE run lengths do not match the declared image size')
    flat = np.zeros(total, dtype=bool)
    offset = 0
    for index, length in enumerate(runs):
        if index & 1:
            flat[offset:offset + length] = True
        offset += length
    return flat.reshape((height, width), order='F')
import contextlib
import hashlib
import io
import json
import math
import os
from pathlib import Path
import tempfile
import numpy as np
METRIC_KEYS = ('mIoU', 'IoU', 'Mask AP@[.50:.95]', 'AP50', 'AP75')
REGION_AP_DEFINITION = 'class_region_ap_v1'
SCHEMA = 'segmentation-coco-bundle/v1'

def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.name + '.', dir=path.parent)
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as stream:
            json.dump(value, stream, ensure_ascii=False, allow_nan=False, indent=2)
            stream.write('\n')
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)

def sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()

def encode_mask(mask):
    return encode_rle(mask)

def decode_mask(rle):
    return decode_rle(rle)

def _annotation(mask, image_id, category_id, annotation_id=None, score=None, iscrowd=0):
    rle = encode_mask(mask)
    result = {'image_id': image_id, 'category_id': int(category_id), 'segmentation': rle}
    if annotation_id is not None:
        ys, xs = np.nonzero(np.asarray(mask, bool))
        bbox = [int(xs.min()), int(ys.min()), int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1)] if len(xs) else [0, 0, 0, 0]
        result.update(id=annotation_id, area=int(len(xs)), bbox=bbox, iscrowd=int(iscrowd))
    if score is not None:
        if not math.isfinite(float(score)):
            raise ValueError('instance score must be finite')
        result['score'] = float(score)
    return result

class BundleWriter:

    def __init__(self, output, *, categories, task, expected_samples, ap_supported=False, ap_reason=None, metadata=None, class_agnostic=False, foreground_id=1):
        self.output = Path(output)
        if self.output.exists() and any(self.output.iterdir()):
            raise FileExistsError(f'Bundle output already contains files: {self.output}')
        ids = [int(c['id']) for c in categories]
        if len(set(ids)) != len(ids):
            raise ValueError('duplicate category IDs')
        if class_agnostic and len(ids) != 1:
            raise ValueError('class-agnostic bundles require one foreground category')
        if task not in {'binary', 'semantic', 'instance', 'panoptic', 'unsupported'}:
            raise ValueError(f'Unknown task: {task}')
        self.manifest = {'schema_version': SCHEMA, 'task': task, 'categories': categories, 'expected_samples': int(expected_samples), 'ap_supported': bool(ap_supported), 'ap_reason': ap_reason, 'class_agnostic': bool(class_agnostic), 'foreground_id': ids[0] if len(ids) == 1 else int(foreground_id), 'metadata': metadata or {}}
        self.manifest['evaluation_protocols'] = {'primary_ap': REGION_AP_DEFINITION if task != 'unsupported' else 'native_bbox_v1', 'ap_supported_field_means': 'native_instance_ap_supported', 'native_instance_ap_supported': bool(ap_supported)}
        self.images, self.gt, self.pred, self.semantic_gt, self.semantic_pred = ([], [], [], [], [])
        self.sample_ids = set()
        self.image_artifacts = None

    def add(self, sample_id, target, prediction, *, gt_instances=None, pred_instances=None, valid_mask=None, shape=None, metadata=None, image=None):
        input_image = image
        if str(sample_id) in self.sample_ids:
            raise ValueError(f'duplicate sample ID: {sample_id}')
        gt_instances, pred_instances = (gt_instances or [], pred_instances or [])
        if self.manifest['class_agnostic']:
            cid = self.manifest['categories'][0]['id']
            gt_instances = [{**x, 'category_id': cid} for x in gt_instances]
            pred_instances = [{**x, 'category_id': cid} for x in pred_instances]
        arrays = [np.asarray(x) for x in (target, prediction, valid_mask) if x is not None]
        arrays += [np.asarray(x['mask']) for x in gt_instances + pred_instances]
        if shape is None:
            if not arrays:
                raise ValueError('shape is required for a sample without masks')
            shape = arrays[0].shape
        shape = tuple(shape)
        if len(shape) != 2 or any((a.shape != shape for a in arrays)):
            raise ValueError(f'inconsistent mask shape, expected {shape}: {[a.shape for a in arrays]}')
        height, width = shape
        valid = np.ones(shape, bool) if valid_mask is None else np.asarray(valid_mask, bool)
        if target is not None and prediction is None:
            raise ValueError('semantic prediction missing for a ground truth sample')
        category_ids = {int(c['id']) for c in self.manifest['categories']}
        if target is not None:
            unexpected = set(np.unique(np.asarray(target)[valid]).tolist()) - category_ids
            if unexpected:
                raise ValueError(f'ground truth contains unknown valid categories: {unexpected}')
        for instance in gt_instances + pred_instances:
            if int(instance['category_id']) not in category_ids and (not self.manifest['class_agnostic']):
                raise ValueError(f"unknown instance category: {instance['category_id']}")
        for instance in pred_instances:
            if 'score' not in instance or not math.isfinite(float(instance['score'])):
                raise ValueError('instance score must be finite')
        image_id = len(self.images) + 1
        image = {'id': image_id, 'file_name': str(sample_id), 'width': width, 'height': height, 'sample_id': str(sample_id), 'valid_mask': encode_mask(valid), 'metadata': metadata or {}}
        new_gt = [_annotation(x['mask'], image_id, x['category_id'], len(self.gt) + i + 1, iscrowd=x.get('iscrowd', 0)) for i, x in enumerate(gt_instances)]
        new_pred = [_annotation(x['mask'], image_id, x['category_id'], score=x['score']) for x in pred_instances]
        sg, sp = ([], [])
        if self.manifest['task'] == 'instance':
            for entries, destination in [(gt_instances, sg), (pred_instances, sp)]:
                for cid in sorted({int(x['category_id']) for x in entries}):
                    union = np.zeros(shape, bool)
                    for item in entries:
                        if int(item['category_id']) == cid:
                            union |= np.asarray(item['mask'], bool)
                    destination.append(_annotation(union, image_id, cid))
        else:
            for array, destination in [(target, sg), (prediction, sp)]:
                if array is not None:
                    array = np.asarray(array)
                    for cid in np.unique(array):
                        destination.append(_annotation(array == cid, image_id, int(cid)))
        image['semantic_available'] = self.manifest['task'] == 'instance' or (target is not None and prediction is not None)
        if input_image is not None:
            if self.image_artifacts is None:
                self.image_artifacts = ImageArtifacts(self.output, self.manifest['categories'], self.manifest['task'], self.manifest['foreground_id'])
            artifacts = self.image_artifacts.export(image_id, input_image, target, prediction, valid, gt_instances, pred_instances, metadata)
            image.update(file_name=artifacts['image'], artifacts=artifacts)
        self.images.append(image)
        self.gt.extend(new_gt)
        self.pred.extend(new_pred)
        self.semantic_gt.extend(sg)
        self.semantic_pred.extend(sp)
        self.sample_ids.add(str(sample_id))

    def finalize(self):
        if len(self.images) != self.manifest['expected_samples'] or not self.images:
            raise ValueError(f"coverage mismatch: {len(self.images)} / {self.manifest['expected_samples']}")
        self.output.mkdir(parents=True, exist_ok=True)
        payloads = {'ground_truth.json': {'info': {'description': 'Native instance ground truth'}, 'images': self.images, 'annotations': self.gt, 'categories': self.manifest['categories']}, 'predictions.json': self.pred, 'semantic_ground_truth.json': self.semantic_gt, 'semantic_predictions.json': self.semantic_pred}
        hashes = {}
        for name, payload in payloads.items():
            atomic_json(self.output / name, payload)
            hashes[name] = sha256(self.output / name)
        self.manifest.update(evaluated_samples=len(self.images), payload_sha256=hashes, coverage_sample_ids=[x['sample_id'] for x in self.images], complete=True)
        if self.image_artifacts is not None:
            if any(('artifacts' not in item for item in self.images)):
                raise ValueError('image artifacts must cover every evaluation unit')
            self.manifest.update(image_artifacts_sha256=self.image_artifacts.hashes, image_artifacts=self.image_artifacts.description())
        atomic_json(self.output / 'manifest.json', self.manifest)
        return self.output

def validate_bundle(path):
    path = Path(path)
    manifest = json.loads((path / 'manifest.json').read_text(encoding='utf-8'))
    if manifest.get('schema_version') != SCHEMA or manifest.get('complete') is not True:
        raise ValueError('unsupported or incomplete bundle manifest')
    required = {'ground_truth.json', 'predictions.json', 'semantic_ground_truth.json', 'semantic_predictions.json'}
    if set(manifest.get('payload_sha256', {})) != required:
        raise ValueError('incomplete payload hash manifest')
    for name, expected in manifest['payload_sha256'].items():
        if sha256(path / name) != expected:
            raise ValueError(f'payload hash mismatch: {name}')
    payload = {name: json.loads((path / name).read_text(encoding='utf-8')) for name in required}
    gt = payload['ground_truth.json']
    if 'image_artifacts_sha256' in manifest:
        hashes = manifest['image_artifacts_sha256']
        references = set()
        for item in gt['images']:
            artifacts = item.get('artifacts')
            if not artifacts or item['file_name'] != artifacts.get('image'):
                raise ValueError('image artifact coverage is incomplete')
            references.update([artifacts['image'], artifacts['valid_mask']])
            references.update((x['path'] for x in artifacts['originals']))
            for role in ('ground_truth', 'prediction'):
                masks = artifacts[role]
                references.update((masks[key] for key in ('mask', 'labels', 'preview', 'layout_preview') if key in masks))
                references.update((x['path'] for x in masks.get('instances', [])))
        if set(hashes) != references:
            raise ValueError('image artifact hash manifest does not match image references')
        for name, expected in hashes.items():
            candidate = path / name
            if Path(name).is_absolute() or not candidate.resolve().is_relative_to(path.resolve()):
                raise ValueError(f'invalid image artifact path: {name}')
            if not candidate.is_file() or sha256(candidate) != expected:
                raise ValueError(f'image artifact missing or hash mismatch: {name}')
    if gt['categories'] != manifest['categories']:
        raise ValueError('manifest and ground-truth categories differ')
    images = {im['id']: im for im in gt['images']}
    sample_ids = [x['sample_id'] for x in gt['images']]
    if len(images) != len(gt['images']) or len(set(sample_ids)) != len(sample_ids):
        raise ValueError('duplicate images or samples')
    if sample_ids != manifest['coverage_sample_ids'] or len(images) != manifest['expected_samples'] or len(images) != manifest['evaluated_samples']:
        raise ValueError('coverage mismatch in bundle')
    for im in images.values():
        if decode_mask(im['valid_mask']).shape != (im['height'], im['width']):
            raise ValueError('valid mask shape mismatch')
    for name in required:
        entries = payload[name]['annotations'] if name == 'ground_truth.json' else payload[name]
        for entry in entries:
            if entry['image_id'] not in images:
                raise ValueError('annotation references an unknown image')
            im = images[entry['image_id']]
            if decode_mask(entry['segmentation']).shape != (im['height'], im['width']):
                raise ValueError('annotation mask shape mismatch')
            if name == 'predictions.json' and (not math.isfinite(entry['score'])):
                raise ValueError('nonfinite prediction score')
    return (manifest, payload)
import argparse
from datetime import datetime, timezone
import importlib
import json
import os
from pathlib import Path
import random
import sys
import time
import traceback
import uuid
import fcntl
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import numpy as np

class LogStream:
    """Keep normal CLI output while recording Python/native-library warnings."""

    def __init__(self, original, log):
        self.original, self.log = (original, log)

    def write(self, text):
        self.log.write(text)
        self.log.flush()
        return self.original.write(text)

    def flush(self):
        self.log.flush()
        self.original.flush()

    def __getattr__(self, name):
        return getattr(self.original, name)

def json_metadata(value):
    if isinstance(value, np.ndarray):
        return {'array_shape': list(value.shape), 'array_dtype': str(value.dtype)}
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): json_metadata(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_metadata(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    return str(value)

def evaluation_units(sample, prediction):
    target = prediction.metadata.get('evaluation_target', sample.target)
    valid = prediction.metadata.get('evaluation_valid_mask', sample.valid_mask)
    semantic = prediction.semantic
    metadata = json_metadata({**sample.metadata, **prediction.metadata})
    if target is not None:
        target = np.asarray(target)
        shape = target.shape
    elif semantic is not None:
        shape = np.asarray(semantic).shape
    elif sample.instances:
        shape = np.asarray(sample.instances[0]['mask']).shape
    elif prediction.instances:
        shape = np.asarray(prediction.instances[0]['mask']).shape
    elif hasattr(sample.image, 'size') and isinstance(sample.image.size, tuple):
        shape = (sample.image.height, sample.image.width)
    else:
        shape = np.asarray(sample.image).shape[:2]
    if len(shape) < 2:
        raise ValueError(f'Cannot determine evaluation image shape: {shape}')
    if any((dimension <= 0 for dimension in shape)):
        raise ValueError(f'All evaluation dimensions must be positive: {shape}')
    if semantic is not None and np.asarray(semantic).shape != shape:
        raise ValueError(f'prediction/GT shape mismatch: {np.asarray(semantic).shape} vs {shape}')
    if valid is not None and np.asarray(valid).shape != shape:
        raise ValueError('valid-mask / evaluation geometry mismatch')
    if len(shape) > 2 and (sample.instances or prediction.instances):
        raise ValueError('3D instance AP is not defined by the 2D COCO protocol')
    for index in np.ndindex(shape[:-2]):
        suffix = '/slice_' + '_'.join(map(str, index)) if index else ''
        image, display = image_for_unit(sample, prediction, index)
        unit = dict(sample_id=sample.sample_id + suffix, target=None if target is None else np.asarray(target)[index], prediction=None if semantic is None else np.asarray(semantic)[index], valid_mask=None if valid is None else np.asarray(valid)[index], gt_instances=sample.instances, pred_instances=prediction.instances, shape=shape[-2:], image=image, metadata={**metadata, 'source_sample_id': sample.sample_id, 'slice_index': list(index), 'image_display': display})
        yield unit

def timing_summary(forward, pipeline, sample_count):
    valid = len(forward) == sample_count and all((x is not None and np.isfinite(x) and (x >= 0) for x in forward))
    total = float(sum(forward)) if valid else None
    return {'sample_count': sample_count, 'total_forward_seconds': total, 'mean_forward_ms': 1000 * total / sample_count if total is not None and sample_count else None, 'total_pipeline_seconds': float(sum(pipeline)), 'mean_pipeline_ms': 1000 * sum(pipeline) / sample_count if sample_count else None, 'forward_timed_sample_count': sum((x is not None for x in forward)), 'forward_time_reason': None if valid else 'Backend did not expose isolated forward timing for every sample.', 'pipeline_scope': 'preprocessing + synchronized inference + postprocessing; excludes data loading and export'}

def capture_environment(device, precision):
    packages = {}
    for name in ['numpy', 'Pillow', 'pycocotools', 'torch', 'torchvision', 'transformers', 'onnxruntime', 'onnxruntime-gpu', 'tensorflow', 'ultralytics', 'monai', 'paddlepaddle-gpu']:
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            pass
    try:
        gpu = subprocess.run(['nvidia-smi', '--query-gpu=name,driver_version,memory.total', '--format=csv,noheader'], capture_output=True, text=True, timeout=10)
        nvidia = {'available': gpu.returncode == 0, 'output': (gpu.stdout + gpu.stderr).strip()}
    except (OSError, subprocess.TimeoutExpired) as exc:
        nvidia = {'available': False, 'output': str(exc)}
    result = {'python': sys.version, 'executable': sys.executable, 'platform': platform.platform(), 'machine': platform.machine(), 'processor': platform.processor(), 'cpu_count': os.cpu_count(), 'packages': packages, 'requested_device': device, 'precision': precision, 'nvidia_smi': nvidia, 'environment_overrides': {k: os.environ[k] for k in ('CUDA_VISIBLE_DEVICES', 'LD_PRELOAD', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'MPLCONFIGDIR', 'HF_MODULES_CACHE', 'YOLO_CONFIG_DIR') if k in os.environ}}
    torch = sys.modules.get('torch')
    if torch is not None:
        result['torch_cuda_version'] = torch.version.cuda
        result['torch_cuda_available'] = torch.cuda.is_available()
        result['torch_num_threads'] = torch.get_num_threads()
        if torch.cuda.is_available():
            result['gpu_name'] = torch.cuda.get_device_name(device if str(device).startswith('cuda') else None)
    return result

def _pipeline_synchronize(device):
    torch = sys.modules.get('torch')
    if str(device).startswith('cuda') and torch is not None and torch.cuda.is_available():
        torch.cuda.synchronize(device)

def configure_threads(count):
    os.environ['OMP_NUM_THREADS'] = str(count)
    os.environ['MKL_NUM_THREADS'] = str(count)
    torch = sys.modules.get('torch')
    if torch is not None:
        torch.set_num_threads(count)

def update_timing_index(path, record):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with (path.parent / (path.name + '.lock')).open('a') as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        data = json.loads(path.read_text()) if path.exists() else {'schema_version': 'inference-time/v1', 'runs': [], 'latest': {}}
        data['runs'] = [r for r in data['runs'] if r['model_id'] != record['model_id']]
        data['runs'].append(record)
        data['latest'][record['model_id']] = record['run_id']
        atomic_json(path, data)
import fcntl
import json
import os
from pathlib import Path
import shutil
import tempfile

class BundleOutput:

    def __init__(self, destination, model_id):
        requested = Path(destination).expanduser().absolute()
        if requested.is_symlink():
            raise ValueError(f'Bundle destination must not be a symlink: {requested}')
        self.destination = requested.resolve()
        if not self.destination.name:
            raise ValueError('A filesystem root cannot be a bundle destination')
        if not isinstance(model_id, str) or not model_id:
            raise ValueError('model_id must be a nonempty string')
        self.model_id = model_id
        self.staging = None
        self._container = None
        self._entered = False
        self._closed = False
        self._publish_called = False
        self._published = False
        self._preserve_container = False

    def __enter__(self):
        if self._entered or self._closed:
            raise RuntimeError('BundleOutput context has already been entered')
        self.destination.parent.mkdir(parents=True, exist_ok=True)
        self._container = Path(tempfile.mkdtemp(prefix=f'.{self.destination.name}.', dir=self.destination.parent))
        self.staging = self._container / 'staging'
        try:
            self.staging.mkdir()
        except BaseException:
            shutil.rmtree(self._container)
            raise
        self._entered = True
        return self

    def _check_destination(self):
        destination = self.destination
        if destination.is_symlink():
            raise ValueError(f'Bundle destination must not be a symlink: {destination}')
        if not destination.exists():
            return
        if not destination.is_dir():
            raise FileExistsError(f'Bundle destination is not a directory: {destination}')
        if next(destination.iterdir(), None) is None:
            return
        manifest_path = destination / 'manifest.json'
        if manifest_path.is_symlink():
            raise ValueError(f'Existing bundle manifest must not be a symlink: {manifest_path}')
        try:
            manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
        except (OSError, ValueError) as exc:
            raise ValueError(f'Refusing to replace non-bundle data: {destination}') from exc
        metadata = manifest.get('metadata') if isinstance(manifest, dict) else None
        if not isinstance(metadata, dict) or metadata.get('model_id') != self.model_id or manifest.get('complete') is not True:
            raise ValueError(f'Existing destination is not a complete bundle owned by {self.model_id}: {destination}')

    def publish(self):
        """Replace the current owned bundle, restoring it if installation fails."""
        if not self._entered or self._closed:
            raise RuntimeError('publish requires an active BundleOutput context')
        if self._publish_called:
            raise RuntimeError('publish has already been called for this BundleOutput')
        self._publish_called = True
        if self.staging.is_symlink() or not self.staging.is_dir():
            raise ValueError('Bundle staging must be a real directory')
        previous = self._container / 'previous'
        descriptor = os.open(self.destination.parent, os.O_RDONLY | os.O_DIRECTORY)
        locked = False
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX)
            locked = True
            self._check_destination()
            try:
                if self.destination.exists():
                    os.rename(self.destination, previous)
                os.rename(self.staging, self.destination)
            except BaseException:
                if not self.staging.exists() and self.destination.is_dir():
                    self._published = True
                elif previous.exists():
                    try:
                        os.rename(previous, self.destination)
                    except BaseException as rollback_error:
                        self._preserve_container = True
                        raise RuntimeError(f'Bundle rollback failed; previous bundle retained at {previous}') from rollback_error
                raise
            self._published = True
            return self.destination
        finally:
            try:
                if locked:
                    fcntl.flock(descriptor, fcntl.LOCK_UN)
            finally:
                os.close(descriptor)

    def __exit__(self, exception_type, exception, traceback):
        self._closed = True
        if self._container is not None and (not self._preserve_container):
            shutil.rmtree(self._container)
        return False




import hashlib
import json
from pathlib import Path

def check_model(model_dir, dataset_root=None, source_root=None):
    model_dir = Path(model_dir).resolve()
    evaluation_root = model_dir
    config = json.loads((model_dir / 'weights' / 'model.json').read_text())
    data_root = Path(dataset_root or model_dir / config['dataset_root']).resolve()
    source_root = Path(source_root or model_dir / config['source_root']).resolve()
    weight = (source_root / config['weight_path']).resolve()
    failures = []
    if not source_root.is_relative_to(evaluation_root):
        failures.append(f'Native code root is outside this model: {source_root}')
    if not weight.is_relative_to(evaluation_root):
        failures.append(f'Weight path is outside this model: {weight}')
    if config.get('source_entry'):
        entry = Path(config['source_entry'])
        embedded = globals().get('_EMBEDDED_SOURCES', {})
        if entry.is_absolute() or '..' in entry.parts:
            failures.append(f'Native entry is outside this model: {entry}')
        elif entry.as_posix() not in embedded:
            failures.append(f'Missing embedded native entry: {entry.as_posix()}')
    if not weight.exists():
        failures.append(f'Missing weight path: {weight}')
    payload_extensions = {'.pt', '.pth', '.bin', '.safetensors', '.onnx', '.gguf', '.h5', '.ckpt', '.pb', '.mlmodel', '.pdiparams', '.pdparams'}
    payloads = [weight] if weight.is_file() else [p for p in weight.rglob('*') if p.is_file() and p.suffix in payload_extensions and ('.cache' not in p.parts)] if weight.exists() else []
    if weight.exists() and (not payloads):
        failures.append(f'Weight directory has no model payload: {weight}')
    if config['adapter'] == 'coreml':
        package = weight if weight.suffix == '.mlpackage' else weight / config.get('package', 'DETRResnet50SemanticSegmentationF32.mlpackage')
        specification = package / 'Data/com.apple.CoreML/model.mlmodel'
        if not specification.is_file():
            failures.append(f'Missing selected CoreML package model specification: {specification}')
        elif b'@model_path/weights/weight.bin' in specification.read_bytes() and (not (specification.parent / 'weights/weight.bin').is_file()):
            failures.append(f"Selected CoreML package references missing external weights: {specification.parent / 'weights/weight.bin'}")
    datasets = []
    for entry in config['datasets']:
        root = data_root / entry['name']
        path = root / 'dataset_manifest.json'
        if not path.is_file() or not (root / 'evaluation_samples.jsonl').is_file():
            failures.append(f'Missing prepared dataset: {root}')
            continue
        manifest = json.loads(path.read_text())
        available = {s['split']: s['sample_count'] for s in manifest['splits']}
        selected = entry.get('split', list(available))
        if isinstance(selected, str):
            selected = [selected]
        if set(selected) - set(available):
            failures.append(f"Unavailable split: {entry['name']} {selected}")
        datasets.append({'name': entry['name'], 'splits': selected, 'sample_count': sum((available.get(s, 0) for s in selected)), 'status': manifest.get('status'), 'optional': entry.get('optional', False)})
    hashes = {name: hashlib.sha256((model_dir / name).read_bytes()).hexdigest() for name in ['inference.py', 'evaluate.py']}
    return {'model_id': config['model_id'], 'paths_ready': not failures, 'failures': failures, 'known_blocker': config.get('known_blocker'), 'weight_path': str(weight), 'weight_exists': weight.exists(), 'weight_payload_count': len(payloads), 'weight_payload_bytes': sum((p.stat().st_size for p in payloads)), 'adapter': config['adapter'], 'datasets': datasets, 'entry_sha256': hashes, 'runtime_exists': Path(config['python']).is_file(), 'execution_verified': False}

# Native architecture modules are stored as readable Python source below.
# This loader preserves their separate namespaces without extracting .py files.
import importlib.abc
import importlib.util
import importlib.machinery
import importlib
import linecache
import hashlib
import sys
import types
from pathlib import Path

def install_embedded(root, sources):
    root = Path(root).resolve()
    files = {str(root / name): source for name, source in sources.items()}
    directories = set()
    for filename in files:
        directories.update(str(p) for p in Path(filename).parents if p == root or root in p.parents)

    class SourceLoader(importlib.abc.Loader):
        def __init__(self, filename):
            self.filename = filename
        def create_module(self, spec):
            return None
        def get_filename(self, fullname):
            return self.filename
        def get_source(self, fullname):
            return files[self.filename]
        def is_package(self, fullname):
            return self.filename.endswith('/__init__.py')
        def get_code(self, fullname):
            source = self.get_source(fullname)
            linecache.cache[self.filename] = (len(source), None, source.splitlines(True), self.filename)
            return compile(source, self.filename, 'exec')
        def exec_module(self, module):
            module.__file__ = self.filename
            module.__dict__.update(load_embedded_path=load_path, load_native=load_native)
            exec(self.get_code(module.__name__), module.__dict__)

    class SourceFinder(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path=None, target=None):
            name = fullname.rsplit('.', 1)[-1]
            namespaces = []
            for entry in sys.path if path is None else path:
                base = Path(entry or '.').resolve() / name
                package, plain = str(base / '__init__.py'), str(base.with_suffix('.py'))
                for filename, is_package in ((package, True), (plain, False)):
                    if filename in files:
                        spec = importlib.util.spec_from_loader(fullname, SourceLoader(filename), origin=filename, is_package=is_package)
                        spec.has_location = True
                        if is_package:
                            spec.submodule_search_locations = [str(base)]
                        return spec
                if str(base) in directories:
                    namespaces.append(str(base))
            if namespaces:
                spec = importlib.machinery.ModuleSpec(fullname, loader=None, is_package=True)
                spec.submodule_search_locations = namespaces
                return spec
            return None

    def load_path(path, name=None):
        filename = str(Path(path).resolve())
        if filename not in files:
            raise FileNotFoundError(f'No embedded Python source: {filename}')
        if name is None:
            directory = str(Path(filename).parent)
            package_name = '_embedded_' + hashlib.sha256(directory.encode()).hexdigest()[:16]
            if package_name not in sys.modules:
                package = types.ModuleType(package_name)
                package.__path__ = [directory]
                package.__package__ = package_name
                package.__spec__ = importlib.machinery.ModuleSpec(package_name, loader=None, is_package=True)
                package.__spec__.submodule_search_locations = [directory]
                sys.modules[package_name] = package
            name = package_name + '.' + Path(filename).stem
        existing = sys.modules.get(name)
        if existing is not None and getattr(existing, '__file__', None) == filename:
            return existing
        is_package = filename.endswith('/__init__.py')
        spec = importlib.util.spec_from_loader(name, SourceLoader(filename), origin=filename, is_package=is_package)
        spec.has_location = True
        if is_package:
            spec.submodule_search_locations = [str(Path(filename).parent)]
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        try:
            spec.loader.exec_module(module)
        except BaseException:
            sys.modules.pop(name, None)
            raise
        return module

    def load_native(source_root, relative_file):
        project_root = Path(source_root).resolve()
        if project_root != root and root not in project_root.parents:
            raise ValueError('Native source root is outside this model resource directory')
        relative = Path(relative_file)
        path = (project_root / relative).resolve()
        if relative.is_absolute() or '..' in relative.parts or project_root not in path.parents:
            raise ValueError('Embedded source must be relative to the model resource directory')
        if str(path) not in files:
            raise FileNotFoundError(f'No embedded Python source: {path}')
        project = project_root / Path(*relative.parts[:2]) if len(relative.parts) > 2 and relative.parts[0] == 'shared' else project_root
        local = path.relative_to(project)
        local_names = {Path(key).relative_to(project).parts[0].removesuffix('.py') for key in files if project in Path(key).parents}
        local_names.update(('utils', 'weights'))
        for key, module in list(sys.modules.items()):
            if key.split('.', 1)[0] not in local_names:
                continue
            module_file = getattr(module, '__file__', None)
            module_paths = getattr(module, '__path__', ())
            belongs = module_file and (project == Path(module_file).resolve() or project in Path(module_file).resolve().parents)
            belongs = belongs or any(project == Path(p).resolve() or project in Path(p).resolve().parents for p in module_paths)
            if not belongs:
                sys.modules.pop(key, None)
        for item in (str(path.parent), str(project)):
            if item not in sys.path:
                sys.path.insert(0, item)
        if len(local.parts) > 1 and local.parts[0] in {'utils', 'weights'}:
            return importlib.import_module('.'.join(local.with_suffix('').parts))
        return load_path(path)

    sys.meta_path.insert(0, SourceFinder())
    load_path.load_native = load_native
    return load_path

import time
from pathlib import Path
import numpy as np
from PIL import Image, ImageOps
ADE_VERSIONS = {'Intel__dpt-large-ade': 'dpt_large', 'facebook__mask2former-swin-large-ade-semantic': 'mask2former_swin_large', 'facebook__maskformer-swin-base-ade': 'maskformer_swin_base', 'facebook__maskformer-swin-large-ade': 'maskformer_swin_large', 'microsoft__beit-base-finetuned-ade-640-640': 'beit_base', 'microsoft__beit-large-finetuned-ade-640-640': 'beit_large', 'openmmlab__upernet-convnext-small': 'upernet_convnext_small', **{f'nvidia__segformer-b{n}-finetuned-ade-{s}-{s}': f'b{n}' for n, s in [(0, 512), (1, 512), (3, 512), (5, 640)]}, **{f'shi-labs__oneformer_ade20k_{n}': f'oneformer_{n}' for n in ('swin_large', 'swin_tiny', 'dinat_large')}}

def resolve_precision(requested: str, device: str, architecture: str='') -> str:
    if requested not in {'auto', 'fp16', 'fp32'}:
        raise ValueError(f'Unsupported precision: {requested}')
    cuda = str(device).startswith('cuda')
    if requested == 'fp16' and (not cuda or architecture == 'maskformer'):
        raise ValueError('fp16 is unavailable for CPU or native MaskFormer postprocessing')
    if requested == 'auto':
        return 'fp16' if cuda and architecture not in {'maskformer', 'fashion_segformer'} else 'fp32'
    return requested

def as_image(value) -> Image.Image:
    if isinstance(value, Image.Image):
        return value.convert('RGB')
    if isinstance(value, (str, Path)):
        with Image.open(value) as opened:
            return opened.convert('RGB')
    return Image.fromarray(np.asarray(value)).convert('RGB')

def _nearest(array, shape):
    array = np.asarray(array)
    if array.shape == tuple(shape):
        return array.copy()
    return np.asarray(Image.fromarray(array.astype(np.int32), mode='I').resize((int(shape[1]), int(shape[0])), Image.Resampling.NEAREST))

def synchronize(device):
    if str(device).startswith('cuda'):
        import torch
        torch.cuda.synchronize(device)

def bind_native_synchronization(native):
    """Keep native timing boundaries while honoring indexed CUDA devices.

    Original helpers only recognize the literal ``cuda`` and synchronize the
    default device. Replace their module-local hooks in this worker process;
    the source files and native model/preprocessing/postprocessing stay intact.
    """
    for name in ('_synchronize', 'synchronize_device'):
        if callable(getattr(native, name, None)):
            setattr(native, name, synchronize)

#!/usr/bin/env python3
"""facebook__mask2former-swin-tiny-coco-instance: complete inference/data/export/evaluation pipeline."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))

import json
import time
from pathlib import Path
import numpy as np
FACEBOOK_VERSIONS = {'facebook__detr-resnet-50-panoptic': 'detr_r50', 'facebook__detr-resnet-101-panoptic': 'detr_r101', 'facebook__mask2former-swin-base-coco-panoptic': 'mask2former_base', 'facebook__mask2former-swin-large-coco-panoptic': 'mask2former_large', 'facebook__mask2former-swin-tiny-coco-panoptic': 'mask2former_tiny', 'facebook__mask2former-swin-tiny-coco-instance': 'mask2former_tiny_instance', 'facebook__maskformer-swin-base-coco': 'maskformer_base', 'facebook__maskformer-swin-large-coco': 'maskformer_large'}

def panoptic_semantic(segment_ids, segments):
    result = np.full(np.asarray(segment_ids).shape, -1, dtype=np.int32)
    for segment in segments:
        result[np.asarray(segment_ids) == int(segment['id'])] = int(segment['category_id'])
    return result

def instance_semantic(instances, shape):
    result = np.zeros(shape, dtype=np.int32)
    for item in sorted(instances, key=lambda row: row['score']):
        result[np.asarray(item['mask'], dtype=bool)] = int(item['category_id'])
    return result

def _categories(config, source_root):
    value = config.get('categories')
    if value:
        values = value.values() if isinstance(value, dict) else value
        return {int(item['id']): {'isthing': 1, **item} for item in values}
    explicit = config.get('panoptic_annotation')
    candidates = [Path(explicit)] if explicit else []
    data = Path(source_root).resolve().parents[2] / 'DataSet'
    candidates += sorted(data.rglob('panoptic_val2017.json')) if data.exists() else []
    for path in candidates:
        if path.is_file():
            payload = json.loads(path.read_text())
            return {int(item['id']): item for item in payload['categories']}
    raise FileNotFoundError('COCO panoptic categories required: set categories or panoptic_annotation, or provide Evaluation/DataSet/**/panoptic_val2017.json')

def _decode_instances(records):
    from pycocotools import mask as coco_mask
    return [{'mask': coco_mask.decode(item['segmentation']).astype(bool), 'category_id': int(item['category_id']), 'score': float(item['score'])} for item in records]

def rf_instances(result, valid_categories):
    if not result['segments_info']:
        return []
    instances = []
    for segment, mask in zip(result['segments_info'], result['segmentation'], strict=True):
        category = int(segment['label_id'])
        if category not in valid_categories:
            continue
        if hasattr(mask, 'detach'):
            mask = mask.detach().cpu().numpy()
        instances.append({'mask': np.asarray(mask, dtype=bool), 'category_id': category, 'score': float(segment['score'])})
    return instances

class CocoAdapter:

    def __init__(self, config, source_root, device, precision):
        import torch
        self.config = config
        self.root = Path(source_root)
        self.device = device
        self.group = config['adapter_group']
        self.source = config['source_entry']
        self.weights = self.root / config['weight_path']
        self.native = load_native(self.root, self.source)
        if self.group == 'facebook_coco_shared':
            bind_native_synchronization(self.native)
            registry = load_native(self.root, 'utils/model_registry.py')
            self.data = load_native(self.root, 'utils/coco_panoptic_data.py')
            self.spec = registry.resolve_model_spec(FACEBOOK_VERSIONS[config['model_id']], self.root)
            self.precision = resolve_precision(precision, device, self.spec.family)
            self.categories = _categories(config, self.root)
            self.processor, self.model = self.native.load_local_model(self.spec, self.weights, device, self.precision, registry.find_weight_file(self.weights))
            self.predictor = self.native.build_panoptic_predictor(self.processor, self.model, spec=self.spec, categories=self.categories, device=device, precision=self.precision)
        elif self.group == 'eomt_coco':
            self.precision = resolve_precision(precision, device)
            self.categories = _categories(config, self.root)
            self.processor, self.model, _ = self.native._load_offline_model(self.weights, device, self.precision)
            data = load_native(self.root, 'utils/coco_panoptic.py')
            self.mapping = data.build_label_to_category_id(self.model.config.id2label, list(self.categories.values()))
            self.stuff = data.stuff_label_ids(self.mapping, list(self.categories.values()))
        elif self.group == 'rf_detr_coco':
            from transformers import RfDetrImageProcessor, RfDetrForInstanceSegmentation
            self.precision = resolve_precision(precision, device)
            self.processor = RfDetrImageProcessor.from_pretrained(self.weights, local_files_only=True)
            self.model = RfDetrForInstanceSegmentation.from_pretrained(self.weights, local_files_only=True)
            self.valid_categories = self.native.valid_coco_category_ids(self.model.config.id2label)
        else:
            raise ValueError(f'Unknown COCO adapter: {self.group}')
        self.dtype = torch.float16 if self.precision == 'fp16' else torch.float32
        self.model.to(device=device, dtype=self.dtype).eval()

    def predict(self, sample):
        import torch
        image = as_image(sample.image)
        shape = (image.height, image.width)
        meta = {'source_file': str((self.root / self.source).resolve()), 'native_protocol': self.group, 'timing_scope': 'model_forward', 'precision': self.precision, 'ap_supported': True}
        if self.group == 'facebook_coco_shared':
            image_id = sample.metadata.get('image_id', 0)
            native_sample = self.data.CocoPanopticSample(int(image_id), str(sample.sample_id), Path('.'), Path('.'), {})
            result, timings = self.predictor(native_sample, image)
            instances = _decode_instances(result.instance_predictions)
            semantic = instance_semantic(instances, shape) if self.spec.task == 'instance' else panoptic_semantic(result.segmentation, result.segments_info)
            meta['prediction_kind'] = 'instance' if self.spec.task == 'instance' else 'panoptic'
            return Prediction(semantic, instances, timings['forward'] / 1000, meta)
        encoded = self.processor(images=image, return_tensors='pt')
        if self.group == 'eomt_coco':
            inputs = self.native.move_inputs_to_device(encoded, self.device, self.dtype)
        else:
            inputs = {key: value.to(device=self.device, dtype=self.dtype if value.is_floating_point() else value.dtype) for key, value in encoded.items()}
        synchronize(self.device)
        started = time.perf_counter()
        with torch.inference_mode():
            output = self.model(**inputs)
        synchronize(self.device)
        seconds = time.perf_counter() - started
        if self.group == 'eomt_coco':
            ids, segments = self.native.postprocess_prediction(processor=self.processor, outputs=output, target_size=shape, threshold=0.8, mask_threshold=0.5, overlap_mask_area_threshold=0.8, stuff_classes=self.stuff)
            converted, instances = ([], [])
            scores_available = True
            for segment in segments:
                category_id = self.mapping[int(segment['label_id'])]
                converted.append({**segment, 'category_id': category_id})
                if self.categories[category_id].get('isthing', 0):
                    if 'score' not in segment:
                        scores_available = False
                    else:
                        instances.append({'mask': ids == int(segment['id']), 'category_id': category_id, 'score': float(segment['score'])})
            meta.update(ap_supported=scores_available, prediction_kind='panoptic', instance_protocol='native panoptic thing segments with native confidence')
            return Prediction(panoptic_semantic(ids, converted), instances, seconds, meta)
        result = self.processor.post_process_instance_segmentation(output, threshold=0.0, mask_threshold=0.5, target_sizes=[shape], top_k=100, return_coco_annotation=False, return_binary_maps=True)[0]
        instances = rf_instances(result, self.valid_categories)
        meta['prediction_kind'] = 'instance'
        return Prediction(instance_semantic(instances, shape), instances, seconds, meta)

import argparse
from datetime import datetime, timezone
import importlib
import json
import os
from pathlib import Path
import random
import sys
import time
import traceback
import uuid

def _common_parser(description):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('--environment', choices=['auto', 'current'], default='auto', help='Use configured existing environment, or keep this Python')
    return parser

def reset_result_directory(model_dir):
    """Replace the preceding run's outputs; keep reusable runtime caches."""
    directory = result_directory(model_dir)
    for path in directory.iterdir():
        if path.name == '.cache':
            if path.is_symlink():
                raise ValueError(f'Result cache must not be a symlink: {path}')
            continue
        if path.is_symlink() or not path.is_dir():
            path.unlink()
        else:
            shutil.rmtree(path)
    return directory

def inference_main(model_dir, adapter_factory=None):
    parser = _common_parser('Run the local model, export a COCO bundle, then evaluate it')
    parser.add_argument('--dataset-root', type=Path)
    parser.add_argument('--bundle-root', type=Path, help='COCO package root; default Segmantation/Dataset_Coco')
    parser.add_argument('--dataset', help='Choose an entry from model.json datasets')
    parser.add_argument('--split', action='append', help='Select a prepared split; repeat for several splits')
    parser.add_argument('--device', default='auto', help='auto, cpu, cuda, or cuda:N')
    parser.add_argument('--precision', default='fp32', choices=['fp32', 'fp16', 'bf16'])
    parser.add_argument('--limit', type=int, help='Smoke-test subset; results explicitly marked subset')
    parser.add_argument('--warmup', type=int, default=1)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--seed', type=int)
    parser.add_argument('--output', type=Path, help='Fixed output directory; only this model/result is accepted')
    parser.add_argument('--skip-evaluation', action='store_true', help='Export only; evaluate.py can run independently')
    parser.add_argument('--check', action='store_true', help='Check paths and prepared split without loading the model')
    args = parser.parse_args()
    if args.output is not None and args.output.expanduser().resolve() != (model_dir / 'result').resolve():
        parser.error('--output must be this model/result directory; runs are overwritten in place')
    if args.limit is not None and args.limit < 1:
        parser.error('--limit must be positive')
    if args.warmup < 0 or args.threads < 1:
        parser.error('--warmup must be nonnegative and --threads positive')
    config = json.loads((model_dir / 'weights' / 'model.json').read_text())
    args.source_root = None
    if args.check:
        report = check_model(model_dir, dataset_root=args.dataset_root, source_root=args.source_root)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0 if report['paths_ready'] else 2
    return _run(model_dir, config, args, adapter_factory=adapter_factory)

def _run(model_dir, config, args, adapter_factory=None):
    import numpy as np
    from contextlib import ExitStack
    evaluation_root = model_dir.parents[1]
    source_root = (model_dir / config['source_root']).resolve()
    dataset_root = (args.dataset_root or model_dir / config['dataset_root']).resolve()
    entries = config['datasets']
    selection = next((x for x in entries if x['name'] == args.dataset), None) if args.dataset else entries[0]
    if selection is None:
        raise ValueError(f"Unknown dataset {args.dataset}; available: {[x['name'] for x in entries]}")
    effective_config = {**config, **selection.get('protocol', {})}
    effective_config['cpu_threads'] = args.threads
    run_id = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ') + '_' + uuid.uuid4().hex[:6]
    run_dir = reset_result_directory(model_dir)
    seed = args.seed if args.seed is not None else config.get('seed', 42)
    random.seed(seed)
    np.random.seed(seed)
    device = args.device
    environment = capture_environment(device, args.precision)
    atomic_json(run_dir / 'environment.json', environment)
    atomic_json(run_dir / 'run_config.json', {'config': effective_config, 'arguments': json_metadata(vars(args)), 'seed': seed, 'source_root': str(source_root), 'dataset_root': str(dataset_root)})
    pipeline, forwards = ([], [])
    started = time.perf_counter()
    log = (run_dir / 'run.log').open('w', encoding='utf-8')
    stdout, stderr = (sys.stdout, sys.stderr)
    sys.stdout, sys.stderr = (LogStream(stdout, log), LogStream(stderr, log))
    bundle_outputs = ExitStack()
    try:
        if device == 'auto':
            try:
                import torch
                device = 'cuda' if torch.cuda.is_available() else 'cpu'
            except ImportError:
                device = 'cpu'
        if device == 'cpu' and args.precision != 'fp32':
            raise ValueError('CPU runs require fp32; use CUDA to request fp16/bf16')
        configure_threads(args.threads)
        if 'torch' in sys.modules:
            sys.modules['torch'].set_num_threads(args.threads)
            sys.modules['torch'].manual_seed(seed)
        dataset = load_dataset(dataset_root, selection['name'], args.split or selection.get('split'), effective_config)
        count = min(len(dataset), args.limit) if args.limit else len(dataset)
        scope = 'subset' if count < len(dataset) else 'full'
        effective_config['dataset_name'] = selection['name']
        effective_config['categories'] = dataset.categories
        if adapter_factory is None:
            adapter_factory = create_adapter
        load_start = time.perf_counter()
        adapter = adapter_factory(effective_config, source_root, device, args.precision)
        configure_threads(args.threads)
        model_load_seconds = time.perf_counter() - load_start
        first = dataset.load(0)
        for _ in range(args.warmup):
            adapter.predict(first)
            _pipeline_synchronize(device)
        random.seed(seed)
        np.random.seed(seed)
        if 'torch' in sys.modules:
            sys.modules['torch'].manual_seed(seed)
        writer = None
        units_count = 0
        source_samples = []
        annotations_dir = run_dir / 'native_outputs'
        for index in range(count):
            sample = first if index == 0 else dataset.load(index)
            _pipeline_synchronize(device)
            begin = time.perf_counter()
            prediction = adapter.predict(sample)
            _pipeline_synchronize(device)
            pipeline.append(time.perf_counter() - begin)
            forwards.append(prediction.forward_seconds)
            if prediction.metadata.get('native_probability') is not None:
                annotations_dir.mkdir(exist_ok=True)
                probability_path = annotations_dir / f'{index:06d}_probability.npz'
                np.savez_compressed(probability_path, probability=prediction.metadata.pop('native_probability'))
                prediction.metadata['native_probability_path'] = str(probability_path)
            if writer is None:
                ap_supported = dataset.ap_supported and prediction.metadata.get('ap_supported', config.get('ap_supported', True))
                ap_reason = dataset.ap_reason if not dataset.ap_supported else prediction.metadata.get('ap_reason', config.get('ap_reason'))
                task = dataset.task
                if prediction.metadata.get('native_task') == 'bounding_box_detection' or (sample.target is None and task != 'instance' and (prediction.semantic is None)):
                    task = 'unsupported'
                bundle_path = (args.bundle_root or evaluation_root / 'Dataset_Coco') / config['model_id']
                bundle_output = bundle_outputs.enter_context(BundleOutput(bundle_path, config['model_id']))
                writer = BundleWriter(bundle_output.staging, categories=dataset.categories, task=task, expected_samples=count, ap_supported=ap_supported, ap_reason=ap_reason, class_agnostic=config.get('class_agnostic', False), metadata={'model_id': config['model_id'], 'dataset': selection['name'], 'splits': dataset.splits, 'scope': scope, 'selected_source_samples': count, 'full_source_samples': len(dataset), 'source_manifest_sha256': sha256(dataset.root / 'evaluation_samples.jsonl'), 'dataset_status': dataset.manifest.get('status'), 'protocol': json_metadata(effective_config), 'seed': seed, 'run_id': run_id, 'iou_reason': prediction.metadata.get('iou_reason') or prediction.metadata.get('metrics_unavailable_reason')})
            for unit in evaluation_units(sample, prediction):
                writer.add(**unit)
                units_count += 1
            source_samples.append({'sample_id': sample.sample_id, 'forward_seconds': prediction.forward_seconds, 'pipeline_seconds': pipeline[-1], 'metadata': json_metadata(prediction.metadata)})
            if index == 0 or (index + 1) % 25 == 0 or index + 1 == count:
                print(f"[{config['model_id']}] {index + 1}/{count} {scope}", flush=True)
        timing = timing_summary(forwards, pipeline, count)
        timing.update(warmup_samples=args.warmup, model_load_seconds=model_load_seconds, evaluated_2d_units=units_count, inference_and_export_seconds=time.perf_counter() - started, batch_size_source_samples=1)
        environment = capture_environment(device, args.precision)
        environment['actual_device'] = device
        environment['adapter_execution'] = {key: json_metadata(prediction.metadata[key]) for key in ('actual_precision', 'precision', 'providers', 'runtime', 'backend', 'timing_scope', 'forward_scope') if key in prediction.metadata}
        writer.manifest['expected_samples'] = units_count
        writer.manifest['metadata'].update(environment=environment, timing=timing)
        bundle = writer.finalize()
        atomic_json(run_dir / 'environment.json', environment)
        atomic_json(run_dir / 'timing.json', timing)
        atomic_json(run_dir / 'samples.json', source_samples)
        if args.skip_evaluation:
            validate_bundle(bundle)
            result = {'status': 'exported', 'metrics': dict.fromkeys(METRIC_KEYS), 'bundle': str(bundle), 'metadata': writer.manifest['metadata']}
        else:
            result = evaluate_in_configured_environment(model_dir, bundle)
        bundle = bundle_output.publish()
        result['bundle'] = str(bundle)
        atomic_json(run_dir / 'bundle.json', {'path': os.path.relpath(bundle, run_dir)})
        result.update(model_id=config['model_id'], run_id=run_id, environment=environment, timing=timing)
        atomic_json(run_dir / 'results.json', result)
        atomic_json(run_dir / 'last_attempt.json', result)
        update_timing_index(run_dir / 'inference_time.json', {'model_id': config['model_id'], 'run_id': run_id, 'status': result['status'], 'dataset': selection['name'], 'scope': scope, 'timing': timing, 'run_directory': str(run_dir)})
        update_timing_index(evaluation_root / 'inference_time.json', {'model_id': config['model_id'], 'run_id': run_id, 'status': result['status'], 'dataset': selection['name'], 'scope': scope, 'timing': timing, 'run_directory': str(run_dir)})
        print(json.dumps({'status': result['status'], 'metrics': result['metrics'], 'scope': scope, 'results': str(run_dir / 'results.json')}, ensure_ascii=False, indent=2))
        return 0
    except Exception as exc:
        failure = {'status': 'failed', 'model_id': config['model_id'], 'run_id': run_id, 'metrics': dict.fromkeys(METRIC_KEYS), 'error_type': type(exc).__name__, 'error': str(exc), 'completed_source_samples': len(pipeline), 'environment': environment, 'run_directory': str(run_dir)}
        atomic_json(run_dir / 'results.json', failure)
        atomic_json(run_dir / 'last_attempt.json', failure)
        (run_dir / 'error.log').write_text(traceback.format_exc())
        update_timing_index(run_dir / 'inference_time.json', {'model_id': config['model_id'], 'run_id': run_id, 'status': 'failed', 'error': str(exc), 'run_directory': str(run_dir)})
        update_timing_index(evaluation_root / 'inference_time.json', {'model_id': config['model_id'], 'run_id': run_id, 'status': 'failed', 'error': str(exc), 'run_directory': str(run_dir)})
        print(f"FAILED: {exc}\nDetails: {run_dir / 'error.log'}", file=sys.stderr)
        return 1
    finally:
        sys.stdout, sys.stderr = (stdout, stderr)
        try:
            bundle_outputs.close()
        finally:
            log.close()

def create_adapter(config, source_root, device, precision):
    return CocoAdapter(config, source_root, device, precision)


# ==================== Native model definitions ====================
_EMBEDDED_SOURCES = {}

# Native module: utils/__init__.py
_EMBEDDED_SOURCES['utils/__init__.py'] = '''"""Shared local COCO Panoptic evaluation utilities."""
'''

# Native module: utils/coco_instance_eval.py
_EMBEDDED_SOURCES['utils/coco_instance_eval.py'] = '''from __future__ import annotations

import contextlib
import copy
import io
import json
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np


COCO_AP_STAT_NAMES = (
    "AP",
    "AP50",
    "AP75",
    "AP_small",
    "AP_medium",
    "AP_large",
    "AR_1",
    "AR_10",
    "AR_100",
    "AR_small",
    "AR_medium",
    "AR_large",
)


def find_coco_instance_annotation(coco_root: Path) -> Path:
    """Locate the single COCO 2017 instance validation annotation file."""
    matches = sorted(Path(coco_root).rglob("instances_val2017.json"))
    if len(matches) != 1:
        raise FileNotFoundError(
            f"COCO 必须包含唯一的 instances_val2017.json，找到：{matches}"
        )
    return matches[0]


def _to_numpy(value: object, *, name: str) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    array = np.asarray(value)
    if array.size == 0:
        raise ValueError(f"DETR 输出 {name} 不能为空")
    return array


def _output_value(outputs: object, name: str) -> object:
    if isinstance(outputs, Mapping):
        try:
            return outputs[name]
        except KeyError as exc:
            raise ValueError(f"DETR 输出缺少 {name}") from exc
    try:
        return getattr(outputs, name)
    except AttributeError as exc:
        raise ValueError(f"DETR 输出缺少 {name}") from exc


def _softmax(values: np.ndarray) -> np.ndarray:
    shifted = values - np.max(values, axis=-1, keepdims=True)
    exponentiated = np.exp(shifted)
    return exponentiated / np.sum(exponentiated, axis=-1, keepdims=True)


def _resize_mask_logits(mask_logits: np.ndarray, target_size: tuple[int, int]) -> np.ndarray:
    import torch
    import torch.nn.functional as functional

    height, width = target_size
    resized = functional.interpolate(
        torch.as_tensor(mask_logits, dtype=torch.float32).unsqueeze(0).unsqueeze(0),
        size=(height, width),
        mode="bilinear",
        align_corners=False,
    )[0, 0]
    return resized.numpy()


def _encode_binary_mask(mask: np.ndarray) -> dict[str, object]:
    try:
        from pycocotools import mask as mask_utils
    except ImportError as exc:
        raise RuntimeError("缺少 pycocotools，无法编码 COCO 实例 mask") from exc
    encoded = mask_utils.encode(np.asfortranarray(mask.astype(np.uint8)))
    counts = encoded["counts"]
    return {"size": [int(size) for size in encoded["size"]], "counts": counts.decode("ascii") if isinstance(counts, bytes) else str(counts)}


def universal_instance_result_to_coco_instances(
    result: dict[str, object],
    *,
    image_id: int,
    target_size: tuple[int, int],
    label_to_category_id: Mapping[int, int],
    instance_category_ids: set[int],
) -> list[dict[str, object]]:
    """Convert a Mask2Former/MaskFormer instance post-process result to COCO rows."""
    height, width = (int(target_size[0]), int(target_size[1]))
    segmentation = _to_numpy(result.get("segmentation"), name="instance segmentation")
    if segmentation.shape != (height, width):
        raise ValueError(
            "通用实例分割 mask 尺寸必须匹配 target_size："
            f"mask={segmentation.shape} target_size={(height, width)}"
        )
    converted: list[dict[str, object]] = []
    for raw_segment in result.get("segments_info", []):
        segment = dict(raw_segment)
        label_id = int(segment["label_id"])
        try:
            category_id = int(label_to_category_id[label_id])
        except KeyError as exc:
            raise ValueError(f"实例输出 label_id={label_id} 无法映射到 COCO category") from exc
        if category_id not in instance_category_ids:
            continue
        segment_id = int(segment["id"])
        binary_mask = segmentation == segment_id
        if not np.any(binary_mask):
            continue
        ys, xs = np.nonzero(binary_mask)
        x0, x1 = int(xs.min()), int(xs.max()) + 1
        y0, y1 = int(ys.min()), int(ys.max()) + 1
        converted.append({
            "image_id": int(image_id),
            "category_id": category_id,
            "bbox": [float(x0), float(y0), float(x1 - x0), float(y1 - y0)],
            "score": float(segment.get("score", 1.0)),
            "segmentation": _encode_binary_mask(binary_mask),
        })
    return converted


def universal_outputs_to_coco_instances(
    outputs: object,
    *,
    image_id: int,
    target_size: tuple[int, int],
    label_to_category_id: Mapping[int, int],
    instance_category_ids: set[int],
) -> list[dict[str, object]]:
    """Apply the official MaskFormer top-k instance inference at original resolution."""
    import torch
    import torch.nn.functional as functional

    height, width = (int(target_size[0]), int(target_size[1]))
    class_logits = _output_value(outputs, "class_queries_logits")
    mask_logits = _output_value(outputs, "masks_queries_logits")
    if not isinstance(class_logits, torch.Tensor) or not isinstance(mask_logits, torch.Tensor):
        class_logits = torch.as_tensor(class_logits)
        mask_logits = torch.as_tensor(mask_logits)
    if class_logits.ndim != 3 or mask_logits.ndim != 4 or class_logits.shape[0] != 1 or mask_logits.shape[0] != 1:
        raise ValueError("MaskFormer 输出必须是 [1,Q,C+1] 与 [1,Q,H,W]")
    query_count = int(class_logits.shape[1])
    class_count = int(class_logits.shape[2] - 1)
    scores = functional.softmax(class_logits[0].float(), dim=-1)[:, :-1]
    labels = torch.arange(class_count, device=scores.device).repeat(query_count)
    scores_per_image, topk_indices = scores.flatten().topk(query_count, sorted=False)
    labels_per_image = labels[topk_indices]
    query_indices = torch.div(topk_indices, class_count, rounding_mode="floor")
    resized_masks = functional.interpolate(
        mask_logits.float(), size=(height, width), mode="bilinear", align_corners=False
    )[0, query_indices]
    binary_masks = resized_masks > 0
    mask_scores = (
        resized_masks.sigmoid().flatten(1) * binary_masks.flatten(1)
    ).sum(1) / (binary_masks.flatten(1).sum(1) + 1e-6)
    final_scores = scores_per_image * mask_scores

    converted: list[dict[str, object]] = []
    for label, score, binary_mask_tensor in zip(
        labels_per_image, final_scores, binary_masks, strict=True
    ):
        label_id = int(label.item())
        category_id = int(label_to_category_id[label_id])
        if category_id not in instance_category_ids:
            continue
        binary_mask = binary_mask_tensor.detach().cpu().numpy().astype(bool, copy=False)
        if not np.any(binary_mask):
            continue
        ys, xs = np.nonzero(binary_mask)
        x0, x1 = int(xs.min()), int(xs.max()) + 1
        y0, y1 = int(ys.min()), int(ys.max()) + 1
        converted.append({
            "image_id": int(image_id),
            "category_id": category_id,
            "bbox": [float(x0), float(y0), float(x1 - x0), float(y1 - y0)],
            "score": float(score.item()),
            "segmentation": _encode_binary_mask(binary_mask),
        })
    return converted


def mask_to_boundary(mask: np.ndarray, *, dilation_ratio: float = 0.02) -> np.ndarray:
    """Convert a binary mask to the Boundary-IoU foreground band used for Boundary AP."""
    binary = np.asarray(mask, dtype=bool)
    if binary.ndim != 2:
        raise ValueError(f"实例 mask 必须为二维，实际为：{binary.shape}")
    height, width = binary.shape
    if height == 0 or width == 0:
        raise ValueError("实例 mask 不能为空")
    iterations = max(1, int(round(dilation_ratio * float((height * height + width * width) ** 0.5))))
    eroded = binary.copy()
    for _ in range(iterations):
        padded = np.pad(eroded, 1, mode="constant", constant_values=False)
        neighbours = [
            padded[row:row + height, column:column + width]
            for row in range(3) for column in range(3)
        ]
        eroded = np.logical_and.reduce(neighbours)
    return binary & ~eroded


def detr_outputs_to_coco_instances(
    outputs: object,
    *,
    image_id: int,
    target_size: tuple[int, int],
    category_ids: set[int],
) -> list[dict[str, object]]:
    """Convert one DETR segmentation forward output into official COCO result rows."""
    height, width = (int(target_size[0]), int(target_size[1]))
    if height <= 0 or width <= 0:
        raise ValueError(f"target_size 必须为正数，实际为：{target_size}")
    logits = _to_numpy(_output_value(outputs, "logits"), name="logits")
    boxes = _to_numpy(_output_value(outputs, "pred_boxes"), name="pred_boxes")
    masks = _to_numpy(_output_value(outputs, "pred_masks"), name="pred_masks")
    if logits.ndim != 3 or boxes.ndim != 3 or masks.ndim != 4:
        raise ValueError(
            "DETR 输出形状必须分别为 [1,Q,C]、[1,Q,4]、[1,Q,H,W]，实际为："
            f"{logits.shape}、{boxes.shape}、{masks.shape}"
        )
    if logits.shape[0] != 1 or boxes.shape[0] != 1 or masks.shape[0] != 1:
        raise ValueError("实例评测一次仅支持一张图片的 DETR 输出")
    query_count = logits.shape[1]
    if boxes.shape[1] != query_count or masks.shape[1] != query_count or boxes.shape[2] != 4:
        raise ValueError("DETR logits、boxes 与 masks 的 query 维度不一致")
    if logits.shape[2] < 2:
        raise ValueError("DETR logits 必须至少包含一个类别与 no-object 类别")

    probabilities = _softmax(logits[0].astype(np.float64, copy=False))[:, :-1]
    labels = probabilities.argmax(axis=1)
    scores = probabilities[np.arange(query_count), labels]
    converted: list[dict[str, object]] = []
    for query_index, (label, score) in enumerate(zip(labels, scores, strict=True)):
        category_id = int(label)
        if category_id not in category_ids:
            continue
        center_x, center_y, box_width, box_height = boxes[0, query_index].astype(np.float64, copy=False)
        x0 = float(np.clip((center_x - box_width / 2.0) * width, 0.0, width))
        y0 = float(np.clip((center_y - box_height / 2.0) * height, 0.0, height))
        x1 = float(np.clip((center_x + box_width / 2.0) * width, 0.0, width))
        y1 = float(np.clip((center_y + box_height / 2.0) * height, 0.0, height))
        if x1 <= x0 or y1 <= y0:
            continue
        mask = _resize_mask_logits(masks[0, query_index], (height, width)) > 0.0
        if not np.any(mask):
            continue
        converted.append(
            {
                "image_id": int(image_id),
                "category_id": category_id,
                "bbox": [x0, y0, x1 - x0, y1 - y0],
                "score": float(score),
                "segmentation": _encode_binary_mask(mask),
            }
        )
    return converted


def _metric_group(evaluator: object) -> dict[str, float]:
    stats = np.asarray(getattr(evaluator, "stats"), dtype=np.float64)
    if stats.shape != (len(COCO_AP_STAT_NAMES),):
        raise RuntimeError(f"COCOeval 返回了异常 stats 形状：{stats.shape}")
    return {name: float(value) for name, value in zip(COCO_AP_STAT_NAMES, stats, strict=True)}


def evaluate_with_official_coco_api(
    *,
    annotation_path: Path,
    predictions: Sequence[dict[str, object]],
    output_dir: Path,
    image_ids: Sequence[int] | None = None,
) -> tuple[dict[str, dict[str, float]], Path]:
    """Persist detection records and evaluate them with the official COCO API."""
    try:
        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval
    except ImportError as exc:
        raise RuntimeError("缺少 pycocotools，无法计算 COCO bbox/segm AP") from exc
    source = Path(annotation_path)
    if not source.is_file():
        raise FileNotFoundError(f"COCO instances 标注不存在：{source}")
    destination = Path(output_dir) / "coco_instance_predictions.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(list(predictions), ensure_ascii=False), encoding="utf-8")
    with contextlib.redirect_stdout(io.StringIO()):
        ground_truth = COCO(str(source))
        detections = ground_truth.loadRes(str(destination))
        metrics: dict[str, dict[str, float]] = {}
        for iou_type in ("bbox", "segm"):
            evaluator = COCOeval(ground_truth, detections, iouType=iou_type)
            if image_ids is not None:
                evaluator.params.imgIds = sorted({int(image_id) for image_id in image_ids})
            evaluator.evaluate()
            evaluator.accumulate()
            evaluator.summarize()
            metrics[iou_type] = _metric_group(evaluator)
    return metrics, destination


def evaluate_with_boundary_coco_api(
    *,
    annotation_path: Path,
    predictions: Sequence[dict[str, object]],
    output_dir: Path,
    image_ids: Sequence[int] | None = None,
) -> tuple[dict[str, float], Path]:
    """Evaluate Boundary AP by running COCOeval on official Boundary-IoU masks."""
    try:
        from pycocotools import mask as mask_utils
        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval
    except ImportError as exc:
        raise RuntimeError("缺少 pycocotools，无法计算 COCO Boundary AP") from exc
    source = Path(annotation_path)
    if not source.is_file():
        raise FileNotFoundError(f"COCO instances 标注不存在：{source}")
    with contextlib.redirect_stdout(io.StringIO()):
        source_coco = COCO(str(source))
    boundary_dataset = copy.deepcopy(source_coco.dataset)
    selected_image_ids = (
        None if image_ids is None else {int(image_id) for image_id in image_ids}
    )
    if selected_image_ids is not None:
        boundary_dataset["images"] = [
            image for image in boundary_dataset.get("images", [])
            if int(image["id"]) in selected_image_ids
        ]
    boundary_annotations: list[dict[str, object]] = []
    for annotation in source_coco.dataset.get("annotations", []):
        if (
            selected_image_ids is not None
            and int(annotation["image_id"]) not in selected_image_ids
        ):
            continue
        converted = dict(annotation)
        converted["segmentation"] = _encode_binary_mask(mask_to_boundary(source_coco.annToMask(annotation)))
        boundary_annotations.append(converted)
    boundary_dataset["annotations"] = boundary_annotations
    boundary_ground_truth = COCO()
    boundary_ground_truth.dataset = boundary_dataset
    boundary_ground_truth.createIndex()
    boundary_predictions: list[dict[str, object]] = []
    for prediction in predictions:
        converted = dict(prediction)
        converted["segmentation"] = _encode_binary_mask(mask_to_boundary(mask_utils.decode(prediction["segmentation"])))
        boundary_predictions.append(converted)
    destination = Path(output_dir) / "coco_boundary_predictions.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(boundary_predictions, ensure_ascii=False), encoding="utf-8")
    with contextlib.redirect_stdout(io.StringIO()):
        detections = boundary_ground_truth.loadRes(str(destination))
        evaluator = COCOeval(boundary_ground_truth, detections, iouType="segm")
        if image_ids is not None:
            evaluator.params.imgIds = sorted({int(image_id) for image_id in image_ids})
        evaluator.evaluate()
        evaluator.accumulate()
        evaluator.summarize()
    return _metric_group(evaluator), destination
'''

# Native module: utils/coco_panoptic_data.py
_EMBEDDED_SOURCES['utils/coco_panoptic_data.py'] = '''from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


EXPECTED_VAL_COUNT = 5000


@dataclass(frozen=True)
class CocoPanopticSample:
    image_id: int
    file_name: str
    image_path: Path
    gt_mask_path: Path
    annotation: dict[str, object]


def _normalized_category_name(name: object) -> str:
    normalized = " ".join(str(name).replace("_", " ").strip().casefold().split())
    return {
        "aeroplane": "airplane",
        "motorbike": "motorcycle",
        "sofa": "couch",
        "pottedplant": "potted plant",
        "diningtable": "dining table",
        "tvmonitor": "tv",
    }.get(normalized, normalized)


def _find_annotation_path(coco_root: Path) -> Path:
    matches = sorted(Path(coco_root).rglob("panoptic_val2017.json"))
    if len(matches) != 1:
        raise FileNotFoundError(
            f"COCO 必须包含唯一的 panoptic_val2017.json，找到：{matches}"
        )
    return matches[0]


def _resolve_mask_dir(coco_root: Path, first_mask_name: str) -> Path:
    candidates = sorted({path.parent for path in Path(coco_root).rglob(first_mask_name)})
    if len(candidates) != 1:
        raise FileNotFoundError(
            f"无法唯一定位 COCO panoptic GT 目录（{first_mask_name}）：{candidates}"
        )
    return candidates[0]


def load_coco_panoptic_manifest(
    coco_root: Path,
) -> tuple[list[CocoPanopticSample], dict[int, dict[str, object]], Path, Path]:
    root = Path(coco_root)
    image_dir = root / "val2017"
    if not image_dir.is_dir():
        raise FileNotFoundError(f"COCO val2017 图片目录不存在：{image_dir}")
    annotation_path = _find_annotation_path(root)
    payload = json.loads(annotation_path.read_text(encoding="utf-8"))
    images = payload.get("images")
    annotations = payload.get("annotations")
    categories_raw = payload.get("categories")
    if not all(isinstance(value, list) and value for value in (images, annotations, categories_raw)):
        raise ValueError(f"COCO Panoptic JSON 结构不完整：{annotation_path}")
    image_by_id = {int(item["id"]): item for item in images}
    annotation_by_id = {int(item["image_id"]): item for item in annotations}
    if set(image_by_id) != set(annotation_by_id):
        raise ValueError("COCO Panoptic images 与 annotations 的 image_id 不一致")
    categories = {int(item["id"]): dict(item) for item in categories_raw}
    if not categories:
        raise ValueError("COCO Panoptic categories 为空")
    first_mask_name = str(annotations[0]["file_name"])
    mask_dir = _resolve_mask_dir(root, first_mask_name)
    manifest: list[CocoPanopticSample] = []
    for image_id in sorted(image_by_id):
        image_info = image_by_id[image_id]
        annotation = annotation_by_id[image_id]
        file_name = str(image_info["file_name"])
        image_path = image_dir / file_name
        gt_mask_path = mask_dir / str(annotation["file_name"])
        if not image_path.is_file() or not gt_mask_path.is_file():
            raise FileNotFoundError(
                f"COCO image/GT 缺失：image={image_path}，panoptic={gt_mask_path}"
            )
        manifest.append(
            CocoPanopticSample(image_id, file_name, image_path, gt_mask_path, dict(annotation))
        )
    return manifest, categories, annotation_path, mask_dir


def build_label_to_category_id(
    id2label: dict[int, str],
    categories: dict[int, dict[str, object]],
) -> dict[int, int]:
    category_by_name = {
        _normalized_category_name(category["name"]): category_id
        for category_id, category in categories.items()
    }
    mapping: dict[int, int] = {}
    missing: list[str] = []
    for label_id, label_name in id2label.items():
        category_id = category_by_name.get(_normalized_category_name(label_name))
        if category_id is None:
            missing.append(f"{label_id}:{label_name}")
        else:
            mapping[int(label_id)] = int(category_id)
    if missing:
        raise ValueError("模型标签无法映射到 COCO category：" + ", ".join(missing[:10]))
    return mapping


def panoptic_id_to_rgb(segment_ids: np.ndarray) -> np.ndarray:
    ids = np.asarray(segment_ids, dtype=np.int64)
    if ids.ndim != 2 or np.any(ids < 0) or np.any(ids >= 256 ** 3):
        raise ValueError("panoptic segment ID 必须是二维且位于 0..16777215")
    return np.stack((ids % 256, (ids // 256) % 256, ids // (256 ** 2)), axis=-1).astype(np.uint8)


def rgb_to_panoptic_id(rgb: np.ndarray) -> np.ndarray:
    array = np.asarray(rgb, dtype=np.uint8)
    if array.ndim != 3 or array.shape[2] != 3:
        raise ValueError(f"panoptic RGB 必须是 [H,W,3]，实际为：{array.shape}")
    return (
        array[..., 0].astype(np.int64)
        + 256 * array[..., 1].astype(np.int64)
        + (256 ** 2) * array[..., 2].astype(np.int64)
    )
'''

# Native module: utils/coco_panoptic_eval.py
_EMBEDDED_SOURCES['utils/coco_panoptic_eval.py'] = '''from __future__ import annotations

import json
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
from PIL import Image

from .coco_panoptic_data import CocoPanopticSample, panoptic_id_to_rgb


@dataclass(frozen=True)
class PanopticPrediction:
    segmentation: np.ndarray
    segments_info: list[dict[str, object]]
    instance_predictions: list[dict[str, object]] = field(default_factory=list)


def _to_numpy_segmentation(value: object) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    segmentation = np.asarray(value, dtype=np.int64)
    if segmentation.ndim != 2:
        raise ValueError(f"预测 panoptic mask 必须是二维，实际为：{segmentation.shape}")
    # Transformers uses -1 for void pixels while COCO PNG reserves ID 0 for void.
    segmentation = segmentation.copy()
    segmentation[segmentation < 0] = 0
    return segmentation


def _category_id_for_label(
    label_id: int,
    family: str,
    categories: dict[int, dict[str, object]],
    label_to_category_id: dict[int, int],
) -> int:
    if family == "detr":
        category_id = label_id
    else:
        try:
            category_id = label_to_category_id[label_id]
        except KeyError as exc:
            raise ValueError(f"{family} 输出了无法映射到 COCO 的 label_id={label_id}") from exc
    if category_id not in categories:
        raise ValueError(f"预测 category_id={category_id} 不属于 COCO Panoptic categories")
    return category_id


def normalize_panoptic_result(
    result: dict[str, object],
    *,
    family: str,
    categories: dict[int, dict[str, object]],
    label_to_category_id: dict[int, int],
) -> PanopticPrediction:
    """Convert Transformers post-processing output to COCO Panoptic API fields."""
    segmentation = _to_numpy_segmentation(result["segmentation"])
    converted: list[dict[str, object]] = []
    for raw_segment in result.get("segments_info", []):
        segment = dict(raw_segment)
        segment_id = int(segment["id"])
        if segment_id <= 0:
            raise ValueError(f"预测 segment id 必须为正数，实际为：{segment_id}")
        label_id = int(segment.get("label_id", segment.get("category_id")))
        category_id = _category_id_for_label(label_id, family, categories, label_to_category_id)
        area = int(np.count_nonzero(segmentation == segment_id))
        if area == 0:
            continue
        converted.append(
            {
                "id": segment_id,
                "category_id": category_id,
                "isthing": bool(categories[category_id].get("isthing", 0)),
                "area": area,
            }
        )
    expected_ids = {int(item["id"]) for item in converted}
    observed_ids = set(np.unique(segmentation).tolist()) - {0}
    if observed_ids != expected_ids:
        raise ValueError(
            "预测 mask 与 segments_info 不一致："
            f"mask={sorted(observed_ids)}，segments={sorted(expected_ids)}"
        )
    return PanopticPrediction(segmentation=segmentation, segments_info=converted)


def _subset_ground_truth_payload(
    samples: Iterable[CocoPanopticSample],
    categories: dict[int, dict[str, object]],
) -> dict[str, object]:
    selected = list(samples)
    return {
        "images": [
            {"id": sample.image_id, "file_name": sample.file_name}
            for sample in selected
        ],
        "annotations": [dict(sample.annotation) for sample in selected],
        "categories": [dict(category) for _, category in sorted(categories.items())],
    }


def evaluate_with_official_panoptic_api(
    *,
    samples: Iterable[CocoPanopticSample],
    predictions: dict[int, PanopticPrediction],
    categories: dict[int, dict[str, object]],
    pq_compute_fn: Callable[..., dict[str, object]] | None = None,
) -> dict[str, object]:
    """Use temporary COCO artifacts, then delegate PQ/SQ/RQ to PanopticAPI."""
    selected = list(samples)
    missing = sorted(sample.image_id for sample in selected if sample.image_id not in predictions)
    if missing:
        raise ValueError(f"缺少 {len(missing)} 个预测，首个 image_id={missing[0]}")
    if pq_compute_fn is None:
        try:
            from panopticapi.evaluation import pq_compute
        except ImportError as exc:
            raise RuntimeError(
                "缺少官方 PanopticAPI；请执行："
                "conda run -n nvidia-inf pip install git+https://github.com/cocodataset/panopticapi.git"
            ) from exc
        pq_compute_fn = pq_compute
    gt_folder = selected[0].gt_mask_path.parent if selected else None
    if gt_folder is None:
        raise ValueError("评测样本不能为空")
    with tempfile.TemporaryDirectory(prefix="panoptic_metrics_") as temporary:
        temporary_root = Path(temporary)
        prediction_dir = temporary_root / "predictions"
        prediction_dir.mkdir()
        annotations: list[dict[str, object]] = []
        for sample in selected:
            prediction = predictions[sample.image_id]
            prediction_file_name = Path(str(sample.annotation["file_name"])).name
            Image.fromarray(panoptic_id_to_rgb(prediction.segmentation)).save(
                prediction_dir / prediction_file_name
            )
            annotations.append(
                {
                    "image_id": sample.image_id,
                    "file_name": prediction_file_name,
                    "segments_info": prediction.segments_info,
                }
            )
        gt_json = temporary_root / "ground_truth_subset.json"
        prediction_json = prediction_dir / "predictions.json"
        gt_json.write_text(json.dumps(_subset_ground_truth_payload(selected, categories)), encoding="utf-8")
        prediction_json.write_text(json.dumps({"annotations": annotations}), encoding="utf-8")
        metrics = pq_compute_fn(
            str(gt_json), str(prediction_json), gt_folder=str(gt_folder), pred_folder=str(prediction_dir)
        )
    return dict(metrics)
'''

# Native module: utils/coco_panoptic_runner.py
_EMBEDDED_SOURCES['utils/coco_panoptic_runner.py'] = '''from __future__ import annotations

import time
from pathlib import Path
from typing import Callable

import numpy as np
from PIL import Image
from tqdm import tqdm

from .coco_panoptic_data import CocoPanopticSample, build_label_to_category_id
from .coco_panoptic_eval import PanopticPrediction, normalize_panoptic_result
from .coco_instance_eval import (
    detr_outputs_to_coco_instances,
    universal_outputs_to_coco_instances,
)
from .model_registry import ModelSpec


PanopticPredictor = Callable[[CocoPanopticSample, Image.Image], tuple[PanopticPrediction, dict[str, float]]]


def load_local_model(
    spec: ModelSpec,
    model_path: Path,
    device: str,
    precision: str,
    weight_path: Path,
) -> tuple[object, object]:
    import torch
    from transformers import (
        DetrForSegmentation,
        DetrImageProcessor,
        Mask2FormerForUniversalSegmentation,
        MaskFormerForInstanceSegmentation,
        AutoImageProcessor,
    )

    path = str(model_path)
    dtype = torch.float16 if precision == "fp16" else torch.float32
    common = {
        "local_files_only": True,
        "torch_dtype": dtype,
        "use_safetensors": weight_path.name == "model.safetensors",
    }
    if spec.family == "detr":
        processor = DetrImageProcessor.from_pretrained(path, local_files_only=True)
        model = DetrForSegmentation.from_pretrained(path, **common)
    elif spec.family == "mask2former":
        processor = AutoImageProcessor.from_pretrained(path, local_files_only=True)
        model = Mask2FormerForUniversalSegmentation.from_pretrained(path, **common)
    elif spec.family == "maskformer":
        processor = AutoImageProcessor.from_pretrained(path, local_files_only=True)
        model = MaskFormerForInstanceSegmentation.from_pretrained(path, **common)
    else:
        raise ValueError(f"不支持的 COCO Panoptic 模型族：{spec.family}")
    return processor, model.to(device).eval()


def _synchronize(device: str) -> None:
    if device == "cuda":
        import torch

        torch.cuda.synchronize()


def _post_process_universal_panoptic(
    outputs: object,
    *,
    target_size: tuple[int, int],
    object_threshold: float,
    mask_threshold: float,
    overlap_threshold: float,
    label_ids_to_fuse: set[int],
) -> dict[str, object]:
    """Mask2Former/MaskFormer official panoptic inference at original resolution."""
    import torch
    import torch.nn.functional as functional
    from transformers.models.mask2former.image_processing_mask2former import (
        compute_segments,
        remove_low_and_no_objects,
    )

    class_logits = outputs.class_queries_logits
    mask_logits = functional.interpolate(
        outputs.masks_queries_logits.float(),
        size=target_size,
        mode="bilinear",
        align_corners=False,
    )
    number_of_labels = int(class_logits.shape[-1] - 1)
    mask_probs = mask_logits.sigmoid()[0]
    pred_scores, pred_labels = functional.softmax(class_logits.float(), dim=-1).max(-1)
    mask_probs, scores, labels = remove_low_and_no_objects(
        mask_probs, pred_scores[0], pred_labels[0], object_threshold, number_of_labels
    )
    if mask_probs.shape[0] == 0:
        return {
            "segmentation": torch.full(target_size, -1, dtype=torch.int64),
            "segments_info": [],
        }
    segmentation, segments = compute_segments(
        mask_probs=mask_probs,
        pred_scores=scores,
        pred_labels=labels,
        mask_threshold=mask_threshold,
        overlap_mask_area_threshold=overlap_threshold,
        label_ids_to_fuse=label_ids_to_fuse,
        target_size=None,
    )
    return {"segmentation": segmentation, "segments_info": segments}


def build_panoptic_predictor(
    processor: object,
    model: object,
    *,
    spec: ModelSpec,
    categories: dict[int, dict[str, object]],
    device: str,
    precision: str,
) -> PanopticPredictor:
    import torch

    if precision not in {"fp16", "fp32"}:
        raise ValueError(f"precision 必须是 fp16/fp32，实际为：{precision}")
    id2label = {int(key): str(value) for key, value in model.config.id2label.items()}
    label_to_category_id = (
        {} if spec.family == "detr" else build_label_to_category_id(id2label, categories)
    )
    instance_category_ids = {
        int(category_id)
        for category_id, category in categories.items()
        if bool(category.get("isthing", 0))
    }
    stuff_label_ids = {
        label_id
        for label_id, category_id in label_to_category_id.items()
        if not bool(categories[category_id].get("isthing", 0))
    }
    target_dtype = torch.float16 if precision == "fp16" else torch.float32

    def predict(sample: CocoPanopticSample, image: Image.Image) -> tuple[PanopticPrediction, dict[str, float]]:
        pipeline_started = time.perf_counter()
        preprocess_started = time.perf_counter()
        processor_kwargs: dict[str, object] = {}
        if spec.short_edge is not None and spec.max_edge is not None:
            processor_kwargs["size"] = {
                "shortest_edge": spec.short_edge,
                "longest_edge": spec.max_edge,
            }
        encoded = processor(images=image, return_tensors="pt", **processor_kwargs)
        preprocess_ms = (time.perf_counter() - preprocess_started) * 1000.0
        if not hasattr(encoded, "items"):
            raise TypeError("图像处理器输出必须是 tensor 字典")

        _synchronize(device)
        transfer_started = time.perf_counter()
        model_inputs: dict[str, torch.Tensor] = {}
        for name, value in encoded.items():
            if not isinstance(value, torch.Tensor):
                raise TypeError(f"图像处理器输出 {name} 必须是 Tensor")
            dtype = target_dtype if value.is_floating_point() else value.dtype
            model_inputs[name] = value.to(device=device, dtype=dtype)
        _synchronize(device)
        host_to_device_ms = (time.perf_counter() - transfer_started) * 1000.0

        _synchronize(device)
        forward_started = time.perf_counter()
        with torch.inference_mode():
            output = model(**model_inputs)
        _synchronize(device)
        forward_ms = (time.perf_counter() - forward_started) * 1000.0

        postprocess_started = time.perf_counter()
        if spec.family in {"mask2former", "maskformer"} and spec.task == "panoptic":
            raw = _post_process_universal_panoptic(
                output,
                target_size=(image.height, image.width),
                object_threshold=spec.object_mask_threshold,
                mask_threshold=spec.mask_threshold,
                overlap_threshold=spec.overlap_mask_area_threshold,
                label_ids_to_fuse=stuff_label_ids,
            )
            prediction = normalize_panoptic_result(
                raw,
                family=spec.family,
                categories=categories,
                label_to_category_id=label_to_category_id,
            )
        elif spec.task == "instance":
            prediction = PanopticPrediction(
                segmentation=np.zeros((image.height, image.width), dtype=np.int64),
                segments_info=[],
            )
        else:
            raw = processor.post_process_panoptic_segmentation(
                output, target_sizes=[(image.height, image.width)]
            )[0]
            prediction = normalize_panoptic_result(
                raw,
                family=spec.family,
                categories=categories,
                label_to_category_id=label_to_category_id,
            )
        if spec.family == "detr":
            instance_predictions = detr_outputs_to_coco_instances(
                output,
                image_id=sample.image_id,
                target_size=(image.height, image.width),
                category_ids=instance_category_ids,
            )
        elif spec.family in {"mask2former", "maskformer"}:
            instance_predictions = universal_outputs_to_coco_instances(
                output,
                image_id=sample.image_id,
                target_size=(image.height, image.width),
                label_to_category_id=label_to_category_id,
                instance_category_ids=instance_category_ids,
            )
        else:
            instance_predictions = []
        prediction = PanopticPrediction(
            segmentation=prediction.segmentation,
            segments_info=prediction.segments_info,
            instance_predictions=instance_predictions,
        )
        _synchronize(device)
        postprocess_ms = (time.perf_counter() - postprocess_started) * 1000.0
        return prediction, {
            "preprocess": preprocess_ms,
            "host_to_device": host_to_device_ms,
            "forward": forward_ms,
            "postprocess": postprocess_ms,
            "pipeline": (time.perf_counter() - pipeline_started) * 1000.0,
        }

    return predict


def collect_instance_predictions(
    predictions: dict[int, PanopticPrediction],
) -> list[dict[str, object]]:
    """Flatten successful image predictions into COCOeval's result-list format."""
    return [
        record
        for image_id in sorted(predictions)
        for record in predictions[image_id].instance_predictions
    ]


def run_panoptic_predictions(
    *,
    samples: list[CocoPanopticSample],
    predictor: PanopticPredictor,
    mode: str,
    repeats: int,
) -> tuple[dict[int, PanopticPrediction], list[dict[str, object]], list[dict[str, str]], dict[str, list[float]]]:
    if mode not in {"single", "dataset"}:
        raise ValueError(f"未知评测模式：{mode}")
    if not samples:
        raise ValueError("COCO Panoptic 样本不能为空")
    if repeats <= 0:
        raise ValueError(f"repeats 必须为正整数，实际为：{repeats}")
    predictions: dict[int, PanopticPrediction] = {}
    outputs: list[dict[str, object]] = []
    errors: list[dict[str, str]] = []
    timings: dict[str, list[float]] = {}
    for sample in tqdm(samples, total=len(samples), desc="COCO Panoptic 评测推理", unit="image"):
        try:
            decode_started = time.perf_counter()
            with Image.open(sample.image_path) as opened:
                image = opened.convert("RGB")
            decode_ms = (time.perf_counter() - decode_started) * 1000.0
            prediction: PanopticPrediction | None = None
            sample_timings: dict[str, float] = {}
            for _ in range(repeats if mode == "single" else 1):
                prediction, sample_timings = predictor(sample, image)
                for stage, elapsed_ms in sample_timings.items():
                    timings.setdefault(stage, []).append(float(elapsed_ms))
            if prediction is None:
                raise RuntimeError("Panoptic predictor 未返回预测结果")
            predictions[sample.image_id] = prediction
            outputs.append(
                {
                    "image_id": sample.image_id,
                    "image_file": sample.file_name,
                    "original_size": [image.width, image.height],
                    "image_decode_ms": decode_ms,
                    "segment_count": len(prediction.segments_info),
                    **{f"{stage}_ms": elapsed_ms for stage, elapsed_ms in sample_timings.items()},
                }
            )
        except Exception as exc:
            errors.append({
                "image_id": str(sample.image_id),
                "image_file": sample.file_name,
                "error": f"{type(exc).__name__}: {exc}",
            })
    return predictions, outputs, errors, timings


def is_complete_evaluation(
    *,
    mode: str,
    limit: int | None,
    succeeded: int,
    expected: int,
    failures: int,
) -> bool:
    return mode == "dataset" and limit is None and failures == 0 and succeeded == expected
'''

# Native module: utils/model_registry.py
_EMBEDDED_SOURCES['utils/model_registry.py'] = '''from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ModelSpec:
    version: str
    model_id: str
    family: str
    default_path: Path
    task: str = "panoptic"
    short_edge: int | None = None
    max_edge: int | None = None
    object_mask_threshold: float = 0.5
    mask_threshold: float = 0.5
    overlap_mask_area_threshold: float = 0.8


def resolve_model_spec(version: str, project_dir: Path) -> ModelSpec:
    if version != 'mask2former_tiny_instance':
        raise ValueError("This model directory only contains its own native model specification")
    return ModelSpec(version='mask2former_tiny_instance', model_id='facebook/mask2former-swin-tiny-coco-instance', family='mask2former', task='instance', short_edge=800, max_edge=1333, object_mask_threshold=0.0, mask_threshold=0.5, overlap_mask_area_threshold=0.8, default_path=Path(project_dir).resolve() / 'weights')



def find_weight_file(model_path: Path) -> Path:
    for filename in ("model.safetensors", "pytorch_model.bin"):
        candidate = Path(model_path) / filename
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"模型目录缺少 model.safetensors/pytorch_model.bin：{model_path}")
'''

load_embedded_path = install_embedded(Path(__file__).resolve().parent / "weights", _EMBEDDED_SOURCES)
load_native = load_embedded_path.load_native



if __name__ == "__main__":
    raise SystemExit(_standalone_runtime_ns["main"](_standalone_sys.modules[__name__]))
