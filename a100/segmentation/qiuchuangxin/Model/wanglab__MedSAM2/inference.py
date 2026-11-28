#!/usr/bin/env python3
"""wanglab__MedSAM2: standalone data loading, inference, timing and COCO export.

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

import ctypes
import importlib
import sys
import time
from pathlib import Path
import numpy as np
from PIL import Image

def rgb_image(value):
    if isinstance(value, Image.Image):
        return value.convert('RGB')
    if isinstance(value, (str, Path)):
        with Image.open(value) as image:
            return image.convert('RGB')
    array = np.asarray(value)
    if array.ndim != 3 or array.shape[-1] != 3:
        raise ValueError(f'Expected RGB HWC image, got {array.shape}')
    if np.issubdtype(array.dtype, np.floating):
        if array.min() < 0 or array.max() > 1:
            raise ValueError('Floating RGB input must be in [0, 1]')
        array = np.rint(array * 255).astype(np.uint8)
    return Image.fromarray(array.astype(np.uint8), 'RGB')

def binary_prediction(probability, threshold, seconds, **metadata):
    probability = np.asarray(probability, dtype=np.float32)
    if probability.ndim < 2 or not np.isfinite(probability).all():
        raise ValueError('Native probability must be a finite image or volume')
    return Prediction(semantic=(probability >= threshold).astype(np.uint8), forward_seconds=float(seconds), metadata={'native_probability': probability, 'binary_threshold': float(threshold), 'ap_supported': False, 'ap_reason': 'native_model_has_no_instance_predictions', 'instance_ap_reason': 'native_model_has_no_instance_predictions', **metadata})

def preprocess_onnx(image, family):
    if family == 'modnet':
        w, h = image.size
        h2, w2 = (int(512 * h / min(w, h)), int(512 * w / min(w, h)))
        size = (max(32, w2 - w2 % 32), max(32, h2 - h2 % 32))
        mean, std = ([0.5] * 3, [0.5] * 3)
    else:
        size = (1024, 1024)
        mean, std = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    x = np.asarray(image.resize(size, Image.Resampling.BILINEAR), np.float32) / 255
    x = (x - np.asarray(mean, np.float32)) / np.asarray(std, np.float32)
    return np.ascontiguousarray(x.transpose(2, 0, 1)[None])

def postprocess_onnx(output, original_size, family):
    result = np.asarray(output).squeeze()
    if result.ndim != 2:
        raise ValueError(f'Expected one ONNX foreground plane, got {result.shape}')
    if family in {'birefnet', 'birefnet-lite'}:
        result = 1 / (1 + np.exp(-np.clip(result, -80, 80)))
    return np.asarray(Image.fromarray((np.clip(result, 0, 1) * 255).astype(np.uint8)).resize(original_size, Image.Resampling.BILINEAR))

def native(source_root, relative):
    return load_native(Path(source_root), relative)

def torch_device(device, precision):
    import torch
    if str(device).startswith('cuda') and (not torch.cuda.is_available()):
        raise RuntimeError('CUDA requested but unavailable; select cpu explicitly')
    precision = ('fp16' if str(device).startswith('cuda') else 'fp32') if precision == 'auto' else precision
    if str(device).startswith('cpu') and precision != 'fp32':
        raise ValueError('This native CPU adapter requires fp32')
    dtypes = {'fp16': torch.float16, 'fp32': torch.float32, 'bf16': torch.bfloat16}
    if precision not in dtypes:
        raise ValueError(f'Unsupported torch precision {precision}')
    return (torch.device(device), dtypes[precision])

def timed_forward(function, device):
    import torch
    if str(device).startswith('cuda'):
        torch.cuda.synchronize(device)
    begin = time.perf_counter()
    with torch.inference_mode():
        result = function()
    if str(device).startswith('cuda'):
        torch.cuda.synchronize(device)
    return (result, time.perf_counter() - begin)

#!/usr/bin/env python3
"""wanglab__MedSAM2: complete inference/data/export/evaluation pipeline."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))

import tempfile
import time
from pathlib import Path
import numpy as np

def restore_volume_axes(array, axis_order):
    array = np.asarray(array)
    if array.ndim != 3:
        raise ValueError(f'Expected three-dimensional volume, got {array.shape}')
    if axis_order == 'zyx':
        return np.transpose(array, (2, 1, 0))
    if axis_order == 'xyz':
        return array
    raise ValueError('Volume axis_order must explicitly be xyz or zyx')

def native_video_frames(value):
    frames = np.asarray(value)
    if frames.ndim != 3 or not np.isfinite(frames).all() or frames.min() < 0 or (frames.max() > 255):
        raise ValueError('MedSAM2 requires finite T,H,W video intensities in [0,255]')
    return np.asarray(frames, dtype=np.uint8)

class MedSAM2Adapter:

    def __init__(self, config, source_root, device, precision):
        self.config = config
        self.device, _ = torch_device(device, 'fp32')
        project = Path(source_root)
        self.module = native(source_root, 'MedSAM2CAMUSStart.py')
        self.predictor = self.module.build_official_predictor(project / 'test-code/source/MedSAM2-main', project / config['weight_path'], str(self.device))

    def predict(self, sample):
        frames = native_video_frames(sample.image)
        metadata = sample.metadata
        if 'prompt_masks' not in metadata or int(metadata.get('prompt_frame_index', -1)) != 0:
            raise ValueError('MedSAM2 requires declared prompt_masks at ED frame zero; set prompt=ed_mask in dataset config')
        prompt = np.asarray(metadata['prompt_masks'])
        if prompt.shape != frames.shape[1:]:
            raise ValueError('ED prompt mask geometry does not match video')
        masks = np.zeros(frames.shape, dtype=np.uint8)
        masks[0] = prompt
        order = tuple(metadata.get('original_indices', range(len(frames))))
        loaded = self.module.LoadedCAMUSCase(patient_id=str(metadata.get('patient_id', sample.sample_id)), view=str(metadata.get('view', 'unknown')), frames=frames, masks=masks, original_indices=order, ed_original_index=order[0], es_original_index=order[-1], spacing_yx_mm=tuple(metadata.get('spacing_yx_mm', (1.0, 1.0))))
        with tempfile.TemporaryDirectory(prefix='medsam2-video-') as directory:
            self.module.write_jpeg_video(frames, Path(directory))
            outputs, seconds, peak = self.module.infer_video(self.predictor, loaded, Path(directory), self.device)
        semantic = np.stack([outputs[index] for index in range(len(frames))])
        if semantic.shape != frames.shape:
            raise ValueError('MedSAM2 output geometry does not match input video')
        return Prediction(semantic=semantic, forward_seconds=None, metadata={'native_pipeline_seconds': seconds, 'timing_reason': 'Original infer_video includes state initialization, mask prompts and temporal propagation', 'actual_precision': 'bfloat16_autocast' if self.device.type == 'cuda' else 'float32', 'axis_order': 'time_y_x', 'prompt_source': metadata.get('prompt_source', 'explicit_mask'), 'oracle_assisted': metadata.get('prompt_source') == 'ground_truth_oracle', 'prompt_frame_index': 0, 'original_indices': list(order), 'peak_memory_bytes': peak, 'instance_ap_reason': 'native_temporal_protocol_reports_semantic_structure_masks'})

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
    timed_adapter = None
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
        timed_adapter = _ForwardTimedAdapter(adapter, effective_config, device)
        adapter = timed_adapter
        configure_threads(args.threads)
        model_load_seconds = time.perf_counter() - load_start
        first = dataset.load(0)
        warmup_started = time.perf_counter()
        for _ in range(args.warmup):
            adapter.predict(first)
            _pipeline_synchronize(device)
        warmup_seconds = time.perf_counter() - warmup_started
        timed_adapter.reset_forward_timing()
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
        timing.update(timed_adapter.forward_timing_metadata(count))
        timing.update(warmup_seconds=warmup_seconds, scope=scope, expected_source_samples=len(dataset), measured_source_samples=count)
        timing.update(warmup_samples=args.warmup, model_load_seconds=model_load_seconds, evaluated_2d_units=units_count, inference_and_export_seconds=time.perf_counter() - started, batch_size_source_samples=1)
        environment = capture_environment(device, args.precision)
        environment['actual_device'] = device
        environment['backend_execution'] = timed_adapter.forward_execution_metadata()
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
            try:
                if timed_adapter is not None:
                    timed_adapter.close()
            finally:
                log.close()

def create_adapter(config, source_root, device, precision):
    return MedSAM2Adapter(config, source_root, device, precision)


# ==================== Integrated forward timing ====================
import math
import time


class ForwardMeter:
    """Synchronized, non-overlapping model computation; no external host I/O."""

    def __init__(self, device, sync=None):
        self.device = device
        self.sync = sync
        self.seconds = 0.0
        self.calls = 0
        self.depth = 0

    def synchronize(self):
        if self.sync is not None:
            self.sync()
        elif str(self.device).startswith('cuda'):
            import torch
            torch.cuda.synchronize(self.device)

    def record(self, seconds):
        if not math.isfinite(seconds) or seconds <= 0:
            raise ValueError('Forward duration must be finite and positive')
        self.seconds += float(seconds)
        self.calls += 1

    def measure(self, function, synchronize=None):
        if self.depth:
            return function()
        sync = synchronize or self.synchronize
        sync()
        self.depth += 1
        try:
            begin = time.perf_counter()
            output = function()
            sync()
            seconds = time.perf_counter() - begin
        finally:
            self.depth -= 1
        self.record(seconds)
        return output

    def reset(self):
        if self.depth:
            raise RuntimeError('Cannot reset an active timer')
        self.seconds, self.calls = 0.0, 0


class _ForwardTimedAdapter:
    """Attach live forward measurements to the original prediction objects."""

    def __init__(self, adapter, config, device):
        self.adapter = adapter
        self.meter = ForwardMeter(device)
        self.runner = _create_forward_runner(adapter, config, device, self.meter)
        self.rows = []

    def predict(self, sample):
        before_seconds, before_calls = self.meter.seconds, self.meter.calls
        prediction = self.runner(sample)
        seconds = self.meter.seconds - before_seconds
        calls = self.meter.calls - before_calls
        if calls < 1 or not math.isfinite(seconds) or seconds <= 0:
            raise RuntimeError('Every source sample requires valid isolated forward timing')
        prediction.forward_seconds = float(seconds)
        details = {
            'forward_calls': calls,
            'component_calls': dict(getattr(self.runner, 'last_call_counts', {})),
            'window_count': int(getattr(self.runner, 'last_window_count', 0)),
            'source_sample_unit': self.runner.sample_unit,
            'forward_scope': self.runner.timing_scope,
        }
        prediction.metadata['forward_scope'] = self.runner.timing_scope
        prediction.metadata['forward_timing'] = details
        self.rows.append({'forward_seconds': seconds, **details})
        return prediction

    def reset_forward_timing(self):
        if hasattr(self.runner, 'reset'):
            self.runner.reset()
        self.meter.reset()
        self.rows.clear()

    def forward_timing_metadata(self, expected_count):
        if not self.rows or len(self.rows) != expected_count:
            raise ValueError('Forward timed source sample count does not match inference coverage')
        total = sum(row['forward_seconds'] for row in self.rows)
        return {
            'total_forward_seconds': total,
            'mean_forward_ms': 1000 * total / len(self.rows),
            'sample_count': len(self.rows),
            'timed_source_samples': len(self.rows),
            'forward_timed_sample_count': len(self.rows),
            'forward_call_count': sum(row['forward_calls'] for row in self.rows),
            'source_sample_unit': self.runner.sample_unit,
            'forward_scope': self.runner.timing_scope,
            'boundaries': list(self.runner.boundary_names),
            'warmup_excluded': True,
            'backend_initialization_forwards_excluded': int(getattr(self.runner, 'backend_initialization_forwards', 0)),
            'forward_time_reason': None,
            'forward_timing_version': 'inline-forward/v1',
            'timing_definition': 'synchronized model forward sum / timed source samples; excludes external preprocessing, input upload/output download, postprocessing and export',
        }

    def forward_execution_metadata(self):
        return dict(getattr(self.runner, 'execution_metadata', {}))

    def close(self):
        if hasattr(self.runner, 'close'):
            self.runner.close()

# Reviewed PyTorch forward boundaries, embedded into each inference script.
# No model or project imports are required; torch is imported only at setup.
# Patches are local to one prediction and restored on success and failure.


def _torch_timing_member(root, path, optional=False):
    value = root
    for name in path.split("."):
        value = getattr(value, name, None)
        if value is None:
            if optional:
                return None
            raise RuntimeError("Required PyTorch timing boundary is missing: " + path)
    return value


class _TorchTimingRunner:
    def __init__(self, adapter, boundaries, meter, scope, sample_unit, execution_defaults):
        self.adapter = adapter
        self.boundaries = boundaries
        self.meter = meter
        self.timing_scope = scope
        self.sample_unit = sample_unit
        self.boundary_names = [row[2] for row in boundaries]
        self.last_call_counts = {}
        self.last_window_count = 0
        self._execution_defaults = dict(execution_defaults)
        self.execution_metadata = dict(execution_defaults)
        self._inside = False
        self._running = False

    def _wrap(self, original, label, windows):
        def measured(*args, **kwargs):
            # SAM3 text embeddings, or a child used by another selected stage,
            # may run inside an already selected parent. Count only the parent.
            if self._inside:
                return original(*args, **kwargs)
            window_count = 0
            if windows:
                value = kwargs.get("input_images", args[0] if args else None)
                if value is None or not hasattr(value, "shape") or len(value.shape) < 1:
                    raise RuntimeError("Cannot determine sliding-window batch cardinality")
                window_count = int(value.shape[0])
                if window_count < 1:
                    raise RuntimeError("Empty sliding-window model input")
            self._inside = True
            try:
                # The shared meter synchronizes CUDA before starting and before
                # stopping. Device transfers happen in the caller, outside here.
                result = self.meter.measure(lambda: original(*args, **kwargs))
            finally:
                self._inside = False
            self.last_call_counts[label] = self.last_call_counts.get(label, 0) + 1
            self.last_window_count += window_count
            return result
        return measured

    def __call__(self, sample):
        if self._running:
            raise RuntimeError("PyTorch timing runner does not support concurrent predictions")
        self.reset()
        self._running = True
        restore = []
        try:
            for owner, method, label, windows in self.boundaries:
                original = getattr(owner, method)
                # Preserve exact instance state, including compiled forward
                # overrides. Avoid leaving bound methods in the instance dict.
                existed = method in vars(owner)
                prior = vars(owner).get(method)
                restore.append((owner, method, existed, prior))
                setattr(owner, method, self._wrap(original, label, windows))
            # The native adapter owns inference_mode/no_grad/autocast. Adding
            # an outer context here would also change preprocessing and output
            # tensor semantics; timing must preserve its existing behavior.
            output = self.adapter.predict(sample)
            if not self.last_call_counts:
                raise RuntimeError("No reviewed model boundary executed for this source sample")
            # Native adapters declare actual autocast separately from requested
            # precision. Copy only execution fields; probability tensors, masks,
            # and the old pipeline timing scope must not enter this report.
            prediction_metadata = getattr(output, "metadata", {})
            if isinstance(prediction_metadata, dict):
                for key in ("actual_precision", "precision", "runtime", "backend"):
                    if key in prediction_metadata:
                        self.execution_metadata[key] = prediction_metadata[key]
            return output
        finally:
            for owner, method, existed, prior in reversed(restore):
                if existed:
                    setattr(owner, method, prior)
                else:
                    delattr(owner, method)
            self._inside = False
            self._running = False

    def reset(self):
        if self._running:
            raise RuntimeError("Cannot reset an active PyTorch prediction")
        # Native SAM3 creates a fresh state per image. MedSAM2 infer_video creates
        # and resets its state per video, including when propagation raises.
        self.last_call_counts = {}
        self.last_window_count = 0
        self.execution_metadata = dict(self._execution_defaults)

    def close(self):
        if self._running:
            raise RuntimeError("Cannot close an active PyTorch prediction")


def _create_forward_runner(adapter, config, device, meter):
    """Return a callable(sample); unknown or changed native layouts fail closed.

    `last_call_counts` counts nonoverlapping model calls for the latest source
    sample. `last_window_count` counts actual windows, including batches, for 3D
    models. Neither changes the denominator: one dataset sample is one original
    image, volume, or sequence. Warmup counters can be discarded using reset().

    The existing adapter handles prompts, recurrent state, preprocessing and
    postprocessing. It returns its original Prediction without modification.
    The enclosing inference loop owns meter resets and timing metadata export.
    """
    kind = config.get("adapter")
    if kind not in {"medical_sam3", "vaas", "nv_ct", "medsam2", "vessel", "anime"}:
        raise ValueError("Unsupported PyTorch timing adapter: " + str(kind))
    import torch
    boundaries = []
    seen = set()

    def add(path, method="forward", optional=False, windows=False):
        owner = _torch_timing_member(adapter, path, optional=optional)
        if owner is None:
            return
        if not isinstance(owner, torch.nn.Module) or not callable(getattr(owner, method, None)):
            raise RuntimeError("Expected torch module computation boundary: " + path + "." + method)
        identity = (id(owner), method)
        if identity not in seen:
            seen.add(identity)
            boundaries.append((owner, method, path + "." + method, windows))

    sample_unit = "image"
    scope = "model_forward"
    if kind in {"anime", "vessel", "nv_ct"}:
        # Anime's native predict performs an extra sigmoid after network(); only
        # model.forward is measured. Sliding-window extraction, weighting,
        # accumulation and inverse geometry are likewise outside model.forward.
        add("model", windows=kind in {"vessel", "nv_ct"})
        if kind in {"vessel", "nv_ct"}:
            sample_unit = "volume"
            scope = "sum_sliding_window_model_forwards_per_source_volume"
    elif kind == "vaas":
        add("px")
        add("fx")
        scope = "sum_segformer_and_vit_model_forwards"
    elif kind == "medical_sam3":
        add("predictor.model.backbone", "forward_image")
        add("predictor.model", "forward_grounding")
        # VETextEncoder.forward includes CPU tokenization and .to(device).
        # Its neural children isolate all learned text computation. The token
        # embedding is called both independently and within encoder.forward;
        # the runner's nesting guard prevents counting the latter twice.
        add("predictor.model.backbone.language_backbone.encoder")
        add("predictor.model.backbone.language_backbone.encoder.token_embedding")
        add("predictor.model.backbone.language_backbone.resizer")
        # Processor.set_image may project the SAM2 high-resolution features.
        if _torch_timing_member(adapter, "predictor.model.inst_interactive_predictor", optional=True) is not None:
            add("predictor.model.inst_interactive_predictor.model.sam_mask_decoder.conv_s0")
            add("predictor.model.inst_interactive_predictor.model.sam_mask_decoder.conv_s1")
        scope = "sum_image_text_grounding_model_component_forwards"
    else:  # medsam2; predictor.forward intentionally raises NotImplementedError.
        add("predictor", "forward_image")
        for component in ("sam_prompt_encoder", "sam_mask_decoder", "memory_attention", "memory_encoder"):
            add("predictor." + component)
        # Dense positional encoding is evaluated outside prompt_encoder.forward.
        add("predictor.sam_prompt_encoder", "get_dense_pe")
        for component in ("obj_ptr_proj", "obj_ptr_tpos_proj", "mask_downsample"):
            add("predictor." + component, optional=True)
        sample_unit = "sequence"
        scope = "sum_backbone_prompt_decoder_memory_model_component_forwards_per_sequence"
    return _TorchTimingRunner(adapter, boundaries, meter, scope, sample_unit, {
        "torch_version": str(torch.__version__),
        "runtime": "pytorch",
        "device": str(getattr(adapter, "device", device)),
    })
# ==================== End integrated forward timing ====================

# ==================== Native model definitions ====================
_EMBEDDED_SOURCES = {}

# Native module: MedSAM2CAMUSStart.py
_EMBEDDED_SOURCES['MedSAM2CAMUSStart.py'] = '''# singlefile-special-source-transform-v1
#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import sys
import tempfile
import time
from contextlib import nullcontext
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from PIL import Image

from medsam2_camus import (
    CAMUS_CLASS_NAMES,
    METRIC_NAMES,
    CAMUSCase,
    LoadedCAMUSCase,
    aggregate_metric_rows,
    build_case_manifest,
    build_frame_metric_rows,
    combine_object_masks,
    load_camus_case,
    resolve_dataset_root,
    write_jpeg_video,
)


SCRIPT_ROOT = Path(__file__).resolve().parent
DEFAULT_DATASET_ROOT = SCRIPT_ROOT.parents[2] / "DataSet" / "CAMUS"
DEFAULT_SOURCE_ROOT = SCRIPT_ROOT / "test-code" / "source" / "MedSAM2-main"
DEFAULT_CHECKPOINT = SCRIPT_ROOT / "weights" / "MedSAM2_US_Heart.pt"
DEFAULT_CONFIG_NAME = "configs/sam2.1_hiera_t512.yaml"
EXPECTED_TEST_PATIENTS = 50
EXPECTED_TEST_VIDEOS = 100
EXPECTED_TEST_FRAMES = 1968
SCORE_THRESHOLD = 0.0


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate the official MedSAM2 heart-ultrasound checkpoint on the "
            "official CAMUS 50-patient test split using an ED mask prompt."
        )
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=DEFAULT_DATASET_ROOT,
        help="CAMUS directory containing CAMUS_public (or CAMUS_public itself)",
    )
    parser.add_argument(
        "--source-root",
        type=Path,
        default=DEFAULT_SOURCE_ROOT,
        help="extracted official MedSAM2 source directory",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_CHECKPOINT,
        help="MedSAM2_US_Heart.pt checkpoint path",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="result directory; defaults to results/camus_<UTC timestamp>",
    )
    parser.add_argument(
        "--work-dir",
        type=Path,
        default=None,
        help="parent for temporary per-video JPEG directories; defaults to system temp",
    )
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument(
        "--save-predictions",
        action="store_true",
        help="save per-frame predicted label maps in addition to metrics",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="validate every CAMUS test sequence without loading the model",
    )
    parser.add_argument(
        "--max-videos",
        type=int,
        default=None,
        help="evaluate only the first N videos for a smoke run; omitted means all 100",
    )
    args = parser.parse_args(argv)
    if args.max_videos is not None and args.max_videos <= 0:
        parser.error("--max-videos must be positive")
    return args


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def validate_model_paths(source_root: Path, checkpoint: Path) -> tuple[Path, Path]:
    source = source_root.expanduser().resolve()
    weight = checkpoint.expanduser().resolve()
    missing = []
    # sam2.build_sam is embedded; only physical configuration and weights are checked here.
    config_path = source / "sam2" / DEFAULT_CONFIG_NAME
    if not config_path.is_file():
        missing.append(config_path)
    if not weight.is_file():
        missing.append(weight)
    if missing:
        raise FileNotFoundError(
            "required MedSAM2 files are missing:\\n" + "\\n".join(str(path) for path in missing)
        )
    return source, weight


def validate_device(device: str) -> torch.device:
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested but torch.cuda.is_available() is false; "
            "activate a CUDA-enabled environment or pass --device cpu explicitly"
        )
    return torch.device(device)


def validate_dataset(cases: Sequence[CAMUSCase]) -> dict[str, Any]:
    total_frames = 0
    reversed_videos = 0
    frame_counts: list[int] = []
    for index, case in enumerate(cases, start=1):
        loaded = load_camus_case(case)
        frame_count = int(loaded.frames.shape[0])
        total_frames += frame_count
        frame_counts.append(frame_count)
        reversed_videos += int(loaded.ed_original_index > loaded.es_original_index)
        print(
            f"validate {index:03d}/{len(cases):03d} "
            f"{case.patient_id} {case.view}: {frame_count} frames",
            flush=True,
        )
    return {
        "patient_count": len({case.patient_id for case in cases}),
        "video_count": len(cases),
        "frame_count": total_frames,
        "minimum_frames_per_video": min(frame_counts),
        "maximum_frames_per_video": max(frame_counts),
        "reversed_video_count": reversed_videos,
    }


def build_official_predictor(source_root: Path, checkpoint: Path, device: str):
    source_text = str(source_root)
    if source_text not in sys.path:
        sys.path.insert(0, source_text)
    from sam2.build_sam import build_sam2_video_predictor

    return build_sam2_video_predictor(
        config_file=DEFAULT_CONFIG_NAME,
        ckpt_path=str(checkpoint),
        device=device,
        apply_postprocessing=False,
        hydra_overrides_extra=["++model.non_overlap_masks=true"],
    )


def infer_video(
    predictor: Any,
    loaded: LoadedCAMUSCase,
    video_dir: Path,
    device: torch.device,
) -> tuple[dict[int, np.ndarray], float, int | None]:
    for class_id in CAMUS_CLASS_NAMES:
        if not np.any(loaded.masks[0] == class_id):
            raise ValueError(
                f"ED prompt is missing class {class_id} for {loaded.patient_id} {loaded.view}"
            )

    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
        autocast_context = torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    else:
        autocast_context = nullcontext()

    started = time.perf_counter()
    predictions: dict[int, np.ndarray] = {}
    inference_state = None
    try:
        with torch.inference_mode(), autocast_context:
            inference_state = predictor.init_state(
                video_path=str(video_dir),
                offload_video_to_cpu=False,
                offload_state_to_cpu=False,
                async_loading_frames=False,
            )
            for class_id in CAMUS_CLASS_NAMES:
                predictor.add_new_mask(
                    inference_state=inference_state,
                    frame_idx=0,
                    obj_id=class_id,
                    mask=loaded.masks[0] == class_id,
                )
            for frame_index, object_ids, mask_logits in predictor.propagate_in_video(
                inference_state
            ):
                predictions[int(frame_index)] = combine_object_masks(
                    object_ids, mask_logits, score_threshold=SCORE_THRESHOLD
                )
    finally:
        if inference_state is not None:
            predictor.reset_state(inference_state)
        del inference_state

    if device.type == "cuda":
        torch.cuda.synchronize(device)
        peak_memory = int(torch.cuda.max_memory_allocated(device))
    else:
        peak_memory = None
    elapsed_seconds = time.perf_counter() - started
    expected_indices = set(range(loaded.frames.shape[0]))
    if set(predictions) != expected_indices:
        raise RuntimeError(
            f"MedSAM2 returned incomplete frames for {loaded.patient_id} {loaded.view}: "
            f"expected={sorted(expected_indices)}, got={sorted(predictions)}"
        )
    return predictions, elapsed_seconds, peak_memory


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        if math.isnan(number):
            return None
        if math.isinf(number):
            return "Infinity" if number > 0 else "-Infinity"
        return number
    return value


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    temporary_path = path.with_name(f".{path.name}.tmp")
    temporary_path.write_text(
        json.dumps(_json_safe(payload), ensure_ascii=False, indent=2, allow_nan=False) + "\\n",
        encoding="utf-8",
    )
    os.replace(temporary_path, path)


def atomic_write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    if not rows:
        return
    temporary_path = path.with_name(f".{path.name}.tmp")
    with temporary_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary_path, path)


def save_predictions(
    output_dir: Path,
    loaded: LoadedCAMUSCase,
    predictions: dict[int, np.ndarray],
) -> None:
    case_dir = output_dir / "predictions" / f"{loaded.patient_id}_{loaded.view}"
    case_dir.mkdir(parents=True, exist_ok=True)
    for inference_index, prediction in sorted(predictions.items()):
        original_index = loaded.original_indices[inference_index]
        path = case_dir / f"inference_{inference_index:03d}_original_{original_index:03d}.png"
        Image.fromarray(np.asarray(prediction, dtype=np.uint8)).save(path)


def make_summary(
    all_rows: Sequence[dict[str, Any]],
    case_rows: Sequence[dict[str, Any]],
    selected_video_count: int,
    official_video_count: int,
    status: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    es_rows = [row for row in all_rows if row["phase"] == "ES"]
    all_aggregation = aggregate_metric_rows(all_rows)
    es_aggregation = aggregate_metric_rows(es_rows)
    per_video_rows = [
        {"scope": "primary_es", **row} for row in es_aggregation["per_video"]
    ] + [
        {"scope": "secondary_all_non_prompt", **row}
        for row in all_aggregation["per_video"]
    ]
    summary = {
        "status": status,
        "protocol_name": "MedSAM2 official VOS behavior + CAMUS official test split",
        "official_medsam2_camus_benchmark": False,
        "processed_video_count": len(case_rows),
        "selected_video_count": selected_video_count,
        "official_test_video_count": official_video_count,
        "complete_official_split": (
            status == "complete"
            and len(case_rows) == official_video_count
            and selected_video_count == official_video_count
        ),
        "primary_es": {
            "description": "ES only; ED ground-truth prompt excluded",
            "per_class": es_aggregation["per_class"],
            "macro": es_aggregation["macro"],
        },
        "secondary_all_non_prompt": {
            "description": "all frames after the ED ground-truth prompt",
            "per_class": all_aggregation["per_class"],
            "macro": all_aggregation["macro"],
        },
        "runtime": {
            "total_inference_seconds": float(
                sum(float(row["inference_seconds"]) for row in case_rows)
            ),
            "mean_inference_seconds_per_video": float(
                np.mean([float(row["inference_seconds"]) for row in case_rows])
            ),
            "maximum_peak_gpu_memory_bytes": max(
                (
                    int(row["peak_gpu_memory_bytes"])
                    for row in case_rows
                    if row["peak_gpu_memory_bytes"] not in (None, "")
                ),
                default=None,
            ),
        },
    }
    return summary, per_video_rows


def write_result_artifacts(
    output_dir: Path,
    all_rows: Sequence[dict[str, Any]],
    case_rows: Sequence[dict[str, Any]],
    selected_video_count: int,
    official_video_count: int,
    status: str,
) -> dict[str, Any]:
    summary, per_video_rows = make_summary(
        all_rows,
        case_rows,
        selected_video_count,
        official_video_count,
        status,
    )
    atomic_write_csv(output_dir / "per_frame_metrics.csv", all_rows)
    atomic_write_csv(output_dir / "per_video_metrics.csv", per_video_rows)
    atomic_write_csv(output_dir / "case_runtime.csv", case_rows)
    atomic_write_json(output_dir / "summary.json", summary)
    return summary


def print_metric_block(title: str, block: dict[str, Any]) -> None:
    macro = block["macro"]
    print(title)
    print(
        "  Macro: "
        + ", ".join(
            f"{name}={macro[name]:.6f}" if math.isfinite(float(macro[name])) else f"{name}=inf"
            for name in METRIC_NAMES
        )
    )
    for row in block["per_class"]:
        values = ", ".join(
            f"{name}={row[name]:.6f}" if math.isfinite(float(row[name])) else f"{name}=inf"
            for name in METRIC_NAMES
        )
        print(f"  class {row['class_id']} ({row['class_name']}): {values}")


def run(args: argparse.Namespace) -> int:
    dataset_root = resolve_dataset_root(args.dataset_root)
    source_root, checkpoint = validate_model_paths(args.source_root, args.checkpoint)
    cases = build_case_manifest(dataset_root)
    if len(cases) != EXPECTED_TEST_VIDEOS:
        raise RuntimeError(
            f"official CAMUS split should produce {EXPECTED_TEST_VIDEOS} videos, got {len(cases)}"
        )

    if args.validate_only:
        stats = validate_dataset(cases)
        expected = {
            "patient_count": EXPECTED_TEST_PATIENTS,
            "video_count": EXPECTED_TEST_VIDEOS,
            "frame_count": EXPECTED_TEST_FRAMES,
        }
        mismatches = {
            key: (stats[key], expected_value)
            for key, expected_value in expected.items()
            if stats[key] != expected_value
        }
        if mismatches:
            raise RuntimeError(f"CAMUS dataset does not match expected official split: {mismatches}")
        print(json.dumps(stats, ensure_ascii=False, indent=2))
        print("CAMUS official test split validation passed; model was not loaded.")
        return 0

    device = validate_device(args.device)
    selected_cases = cases[: args.max_videos] if args.max_videos else cases
    if args.work_dir is not None:
        args.work_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else SCRIPT_ROOT / "results" / f"camus_{timestamp}"
    )
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    run_config = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_root": str(dataset_root),
        "split_file": str(dataset_root / "database_split" / "subgroup_testing.txt"),
        "source_root": str(source_root),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
        "sam2_config": DEFAULT_CONFIG_NAME,
        "device": str(device),
        "score_threshold": SCORE_THRESHOLD,
        "apply_postprocessing": False,
        "non_overlap_masks": True,
        "prompt": "complete ED masks for CAMUS labels 1, 2, and 3",
        "primary_scope": "ES only, excluding ED prompt",
        "secondary_scope": "all non-prompt frames from ED toward ES",
        "jpeg_conversion": {
            "quality": 95,
            "subsampling": 0,
            "rgb_channels": "replicated grayscale",
        },
        "official_test_patient_count": EXPECTED_TEST_PATIENTS,
        "official_test_video_count": EXPECTED_TEST_VIDEOS,
        "selected_video_count": len(selected_cases),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
    }
    atomic_write_json(output_dir / "run_config.json", run_config)

    print(f"Loading official MedSAM2 predictor on {device} ...", flush=True)
    predictor = build_official_predictor(source_root, checkpoint, args.device)
    all_rows: list[dict[str, Any]] = []
    case_rows: list[dict[str, Any]] = []
    try:
        for case_index, case in enumerate(selected_cases, start=1):
            loaded = load_camus_case(case)
            print(
                f"[{case_index:03d}/{len(selected_cases):03d}] "
                f"{case.patient_id} {case.view}, frames={loaded.frames.shape[0]}, "
                f"ED original={loaded.ed_original_index}, ES original={loaded.es_original_index}",
                flush=True,
            )
            with tempfile.TemporaryDirectory(
                prefix=f"medsam2_{case.patient_id}_{case.view}_",
                dir=str(args.work_dir) if args.work_dir is not None else None,
            ) as temporary_dir:
                video_dir = Path(temporary_dir) / "frames"
                write_jpeg_video(loaded.frames, video_dir)
                predictions, elapsed_seconds, peak_memory = infer_video(
                    predictor, loaded, video_dir, device
                )

            rows = build_frame_metric_rows(
                patient_id=loaded.patient_id,
                view=loaded.view,
                predictions=predictions,
                targets=loaded.masks,
                original_indices=loaded.original_indices,
                spacing_yx_mm=loaded.spacing_yx_mm,
            )
            all_rows.extend(rows)
            case_rows.append(
                {
                    "patient_id": loaded.patient_id,
                    "view": loaded.view,
                    "frame_count": int(loaded.frames.shape[0]),
                    "scored_frame_count": int(loaded.frames.shape[0] - 1),
                    "ed_original_index": loaded.ed_original_index,
                    "es_original_index": loaded.es_original_index,
                    "inference_seconds": elapsed_seconds,
                    "peak_gpu_memory_bytes": peak_memory,
                }
            )
            if args.save_predictions:
                save_predictions(output_dir, loaded, predictions)
            write_result_artifacts(
                output_dir,
                all_rows,
                case_rows,
                len(selected_cases),
                len(cases),
                status="running",
            )
    except BaseException:
        if all_rows:
            write_result_artifacts(
                output_dir,
                all_rows,
                case_rows,
                len(selected_cases),
                len(cases),
                status="interrupted",
            )
        raise

    summary = write_result_artifacts(
        output_dir,
        all_rows,
        case_rows,
        len(selected_cases),
        len(cases),
        status="complete",
    )
    print_metric_block("Primary ES metrics", summary["primary_es"])
    print_metric_block(
        "Secondary all-non-prompt-frame metrics", summary["secondary_all_non_prompt"]
    )
    print(f"Results: {output_dir}")
    if not summary["complete_official_split"]:
        print("NOTE: --max-videos was used; this is not a complete official-split result.")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
'''

# Native module: medsam2_camus.py
_EMBEDDED_SOURCES['medsam2_camus.py'] = '''from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
from PIL import Image
from scipy.ndimage import binary_erosion, distance_transform_edt


CAMUS_CLASS_NAMES = {
    1: "lv_endo",
    2: "lv_epi",
    3: "left_atrium",
}
CAMUS_STRUCTURE_LABELS = {
    1: (1,),
    2: (1, 2),
    3: (3,),
}
METRIC_NAMES = ("dice", "mad_mm", "hd_mm", "iou", "hd95_mm")
VIEWS = ("2CH", "4CH")


@dataclass(frozen=True)
class OrientedSequence:
    frames: np.ndarray
    masks: np.ndarray
    original_indices: tuple[int, ...]
    ed_original_index: int
    es_original_index: int


@dataclass(frozen=True)
class CAMUSCase:
    patient_id: str
    view: str
    image_sequence_path: Path
    mask_sequence_path: Path
    ed_image_path: Path
    ed_mask_path: Path
    es_image_path: Path
    es_mask_path: Path


@dataclass(frozen=True)
class LoadedCAMUSCase:
    patient_id: str
    view: str
    frames: np.ndarray
    masks: np.ndarray
    original_indices: tuple[int, ...]
    ed_original_index: int
    es_original_index: int
    spacing_yx_mm: tuple[float, float]


def resolve_dataset_root(path: Path | str) -> Path:
    root = Path(path).expanduser().resolve()
    candidates = (root, root / "CAMUS_public")
    for candidate in candidates:
        if (candidate / "database_nifti").is_dir() and (
            candidate / "database_split" / "subgroup_testing.txt"
        ).is_file():
            return candidate
    raise FileNotFoundError(
        "could not find database_nifti and "
        f"database_split/subgroup_testing.txt under {root}"
    )


def load_official_test_ids(
    dataset_root: Path | str, expected_count: int = 50
) -> list[str]:
    split_path = Path(dataset_root) / "database_split" / "subgroup_testing.txt"
    if not split_path.is_file():
        raise FileNotFoundError(f"official CAMUS test split is missing: {split_path}")
    patient_ids = [
        line.strip()
        for line in split_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(set(patient_ids)) != len(patient_ids):
        raise ValueError(f"official CAMUS test split contains duplicate IDs: {split_path}")
    if len(patient_ids) != expected_count:
        raise ValueError(
            f"official CAMUS test split must contain {expected_count} patients, "
            f"found {len(patient_ids)} in {split_path}"
        )
    invalid = [patient_id for patient_id in patient_ids if not patient_id.startswith("patient")]
    if invalid:
        raise ValueError(f"invalid CAMUS patient IDs in test split: {invalid[:5]}")
    return patient_ids


def build_case_manifest(dataset_root: Path | str) -> list[CAMUSCase]:
    root = resolve_dataset_root(dataset_root)
    cases: list[CAMUSCase] = []
    missing: list[Path] = []
    for patient_id in load_official_test_ids(root):
        patient_dir = root / "database_nifti" / patient_id
        for view in VIEWS:
            prefix = patient_dir / f"{patient_id}_{view}"
            case = CAMUSCase(
                patient_id=patient_id,
                view=view,
                image_sequence_path=Path(f"{prefix}_half_sequence.nii.gz"),
                mask_sequence_path=Path(f"{prefix}_half_sequence_gt.nii.gz"),
                ed_image_path=Path(f"{prefix}_ED.nii.gz"),
                ed_mask_path=Path(f"{prefix}_ED_gt.nii.gz"),
                es_image_path=Path(f"{prefix}_ES.nii.gz"),
                es_mask_path=Path(f"{prefix}_ES_gt.nii.gz"),
            )
            for required_path in (
                case.image_sequence_path,
                case.mask_sequence_path,
                case.ed_image_path,
                case.ed_mask_path,
                case.es_image_path,
                case.es_mask_path,
            ):
                if not required_path.is_file():
                    missing.append(required_path)
            cases.append(case)
    if missing:
        preview = "\\n".join(str(path) for path in missing[:10])
        raise FileNotFoundError(
            f"CAMUS test data is incomplete; {len(missing)} required files are missing:\\n"
            f"{preview}"
        )
    return cases


def _matching_frame_index(frames: np.ndarray, endpoint: np.ndarray, name: str) -> int:
    if endpoint.ndim == 3 and endpoint.shape[0] == 1:
        endpoint = endpoint[0]
    if endpoint.shape != frames.shape[1:]:
        raise ValueError(
            f"{name} shape {endpoint.shape} does not match frame shape {frames.shape[1:]}"
        )
    matches = [index for index, frame in enumerate(frames) if np.array_equal(frame, endpoint)]
    if len(matches) != 1:
        raise ValueError(f"{name} must match exactly one sequence frame, found {matches}")
    return matches[0]


def orient_ed_to_es(
    frames: np.ndarray,
    masks: np.ndarray,
    ed_image: np.ndarray,
    es_image: np.ndarray,
) -> OrientedSequence:
    frame_array = np.asarray(frames)
    mask_array = np.asarray(masks)
    if frame_array.ndim != 3 or mask_array.ndim != 3:
        raise ValueError("CAMUS image and mask sequences must have shape (T, H, W)")
    if frame_array.shape != mask_array.shape:
        raise ValueError(
            f"image/mask sequence shape mismatch: {frame_array.shape} vs {mask_array.shape}"
        )
    if frame_array.shape[0] < 2:
        raise ValueError("CAMUS sequence must contain at least two frames")

    ed_index = _matching_frame_index(frame_array, np.asarray(ed_image), "ED image")
    es_index = _matching_frame_index(frame_array, np.asarray(es_image), "ES image")
    endpoints = {0, frame_array.shape[0] - 1}
    if {ed_index, es_index} != endpoints:
        raise ValueError(
            "ED and ES images must match opposite sequence endpoints; "
            f"found ED={ed_index}, ES={es_index}, frames={frame_array.shape[0]}"
        )

    if ed_index == 0:
        order = tuple(range(frame_array.shape[0]))
    else:
        order = tuple(range(frame_array.shape[0] - 1, -1, -1))
    return OrientedSequence(
        frames=frame_array[list(order)],
        masks=mask_array[list(order)],
        original_indices=order,
        ed_original_index=ed_index,
        es_original_index=es_index,
    )


def _read_sitk_array(path: Path) -> tuple[np.ndarray, tuple[float, ...]]:
    try:
        import SimpleITK as sitk
    except ImportError as error:
        raise RuntimeError("SimpleITK is required to read CAMUS NIfTI files") from error
    image = sitk.ReadImage(str(path))
    return sitk.GetArrayFromImage(image), tuple(float(value) for value in image.GetSpacing())


def _endpoint_2d(array: np.ndarray, name: str) -> np.ndarray:
    endpoint = np.asarray(array)
    if endpoint.ndim == 2:
        return endpoint
    if endpoint.ndim == 3 and endpoint.shape[0] == 1:
        return endpoint[0]
    raise ValueError(f"{name} must be a 2D image, got shape {endpoint.shape}")


def load_camus_case(case: CAMUSCase) -> LoadedCAMUSCase:
    frames, spacing_xyz = _read_sitk_array(case.image_sequence_path)
    masks, mask_spacing_xyz = _read_sitk_array(case.mask_sequence_path)
    ed_image, _ = _read_sitk_array(case.ed_image_path)
    es_image, _ = _read_sitk_array(case.es_image_path)
    ed_mask, _ = _read_sitk_array(case.ed_mask_path)
    es_mask, _ = _read_sitk_array(case.es_mask_path)

    frames = np.asarray(frames)
    masks = np.asarray(masks)
    oriented = orient_ed_to_es(frames, masks, ed_image, es_image)
    endpoint_ed_mask = _endpoint_2d(ed_mask, "ED mask")
    endpoint_es_mask = _endpoint_2d(es_mask, "ES mask")
    if endpoint_ed_mask.shape != oriented.masks.shape[1:]:
        raise ValueError(
            f"ED mask shape mismatch for {case.patient_id} {case.view}: "
            f"{endpoint_ed_mask.shape} vs {oriented.masks.shape[1:]}"
        )
    if endpoint_es_mask.shape != oriented.masks.shape[1:]:
        raise ValueError(
            f"ES mask shape mismatch for {case.patient_id} {case.view}: "
            f"{endpoint_es_mask.shape} vs {oriented.masks.shape[1:]}"
        )
    # The dense half-sequence labels are not pixel-identical to CAMUS's separately
    # released manual endpoint labels. Use the official endpoint files for the ED
    # prompt and primary ES score, retaining dense labels only for intermediate frames.
    oriented_masks = np.asarray(oriented.masks).copy()
    oriented_masks[0] = endpoint_ed_mask
    oriented_masks[-1] = endpoint_es_mask

    labels = set(int(value) for value in np.unique(oriented_masks))
    unexpected = sorted(labels - {0, *CAMUS_CLASS_NAMES})
    if unexpected:
        raise ValueError(
            f"unexpected CAMUS labels for {case.patient_id} {case.view}: {unexpected}"
        )
    if tuple(spacing_xyz[:2]) != tuple(mask_spacing_xyz[:2]):
        raise ValueError(
            f"image/mask spacing mismatch for {case.patient_id} {case.view}: "
            f"{spacing_xyz[:2]} vs {mask_spacing_xyz[:2]}"
        )
    return LoadedCAMUSCase(
        patient_id=case.patient_id,
        view=case.view,
        frames=np.asarray(oriented.frames, dtype=np.uint8),
        masks=np.asarray(oriented_masks, dtype=np.uint8),
        original_indices=oriented.original_indices,
        ed_original_index=oriented.ed_original_index,
        es_original_index=oriented.es_original_index,
        spacing_yx_mm=(spacing_xyz[1], spacing_xyz[0]),
    )


def combine_object_masks(
    object_ids: Sequence[int], mask_logits: Any, score_threshold: float = 0.0
) -> np.ndarray:
    if hasattr(mask_logits, "detach"):
        logits = mask_logits.detach().float().cpu().numpy()
    else:
        logits = np.asarray(mask_logits)
    if logits.ndim == 4 and logits.shape[1] == 1:
        logits = logits[:, 0]
    if logits.ndim != 3:
        raise ValueError(f"mask logits must have shape (N, 1, H, W) or (N, H, W), got {logits.shape}")
    ids = [int(object_id) for object_id in object_ids]
    if len(ids) != logits.shape[0]:
        raise ValueError(f"object ID/logit count mismatch: {len(ids)} vs {logits.shape[0]}")
    if len(set(ids)) != len(ids) or any(object_id <= 0 or object_id > 255 for object_id in ids):
        raise ValueError(f"object IDs must be unique integers in [1, 255], got {ids}")

    per_object = {
        object_id: logits[index] > score_threshold
        for index, object_id in enumerate(ids)
    }
    output = np.zeros(logits.shape[1:], dtype=np.uint8)
    # This is the same ordering used by official medsam2_infer_video.put_per_obj_mask.
    for object_id in sorted(per_object, reverse=True):
        output[per_object[object_id]] = object_id
    return output


def write_jpeg_video(frames: np.ndarray, video_dir: Path | str) -> list[Path]:
    frame_array = np.asarray(frames)
    if frame_array.ndim != 3 or frame_array.shape[0] == 0:
        raise ValueError(f"frames must have non-empty shape (T, H, W), got {frame_array.shape}")
    if frame_array.dtype != np.uint8:
        raise ValueError(f"JPEG input frames must be uint8, got {frame_array.dtype}")
    output_dir = Path(video_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    existing = list(output_dir.glob("*.jpg")) + list(output_dir.glob("*.jpeg"))
    if existing:
        raise FileExistsError(f"video directory already contains JPEG frames: {output_dir}")

    paths: list[Path] = []
    for frame_index, frame in enumerate(frame_array):
        output_path = output_dir / f"{frame_index:05d}.jpg"
        Image.fromarray(frame).convert("RGB").save(
            output_path,
            format="JPEG",
            quality=95,
            subsampling=0,
            optimize=False,
            progressive=False,
        )
        paths.append(output_path)
    return paths


def _surface(mask: np.ndarray) -> np.ndarray:
    eroded = binary_erosion(mask, structure=np.ones((3, 3), dtype=bool), border_value=0)
    return mask & ~eroded


def binary_metrics(
    prediction: np.ndarray,
    target: np.ndarray,
    spacing_yx: tuple[float, float],
) -> dict[str, float]:
    pred = np.asarray(prediction, dtype=bool)
    truth = np.asarray(target, dtype=bool)
    if pred.shape != truth.shape or pred.ndim != 2:
        raise ValueError(
            f"prediction and target must have the same 2D shape, got {pred.shape} and {truth.shape}"
        )
    if len(spacing_yx) != 2 or any(value <= 0 for value in spacing_yx):
        raise ValueError(f"spacing_yx must contain two positive values, got {spacing_yx}")

    pred_count = int(pred.sum())
    truth_count = int(truth.sum())
    if pred_count == 0 and truth_count == 0:
        return {
            "dice": 1.0,
            "iou": 1.0,
            "mad_mm": 0.0,
            "hd_mm": 0.0,
            "hd95_mm": 0.0,
        }
    if pred_count == 0 or truth_count == 0:
        return {
            "dice": 0.0,
            "iou": 0.0,
            "mad_mm": float("inf"),
            "hd_mm": float("inf"),
            "hd95_mm": float("inf"),
        }

    intersection = int(np.logical_and(pred, truth).sum())
    union = pred_count + truth_count - intersection
    pred_surface = _surface(pred)
    truth_surface = _surface(truth)
    distance_to_truth = distance_transform_edt(~truth_surface, sampling=spacing_yx)
    distance_to_pred = distance_transform_edt(~pred_surface, sampling=spacing_yx)
    surface_distances = np.concatenate(
        (distance_to_truth[pred_surface], distance_to_pred[truth_surface])
    )
    return {
        "dice": float(2.0 * intersection / (pred_count + truth_count)),
        "iou": float(intersection / union),
        "mad_mm": float(np.mean(surface_distances)),
        "hd_mm": float(np.max(surface_distances)),
        "hd95_mm": float(np.percentile(surface_distances, 95)),
    }


def build_frame_metric_rows(
    patient_id: str,
    view: str,
    predictions: dict[int, np.ndarray],
    targets: np.ndarray,
    original_indices: Sequence[int],
    spacing_yx_mm: tuple[float, float],
) -> list[dict[str, Any]]:
    target_array = np.asarray(targets)
    if target_array.ndim != 3:
        raise ValueError(f"targets must have shape (T, H, W), got {target_array.shape}")
    if len(original_indices) != target_array.shape[0]:
        raise ValueError(
            "original index count does not match target frames: "
            f"{len(original_indices)} vs {target_array.shape[0]}"
        )
    expected_indices = set(range(target_array.shape[0]))
    if set(predictions) != expected_indices:
        missing = sorted(expected_indices - set(predictions))
        extra = sorted(set(predictions) - expected_indices)
        raise ValueError(f"prediction frame indices mismatch: missing={missing}, extra={extra}")

    rows: list[dict[str, Any]] = []
    for inference_index in range(1, target_array.shape[0]):
        prediction = np.asarray(predictions[inference_index])
        target = target_array[inference_index]
        if prediction.shape != target.shape:
            raise ValueError(
                f"prediction/target shape mismatch at frame {inference_index}: "
                f"{prediction.shape} vs {target.shape}"
            )
        phase = "ES" if inference_index == target_array.shape[0] - 1 else "intermediate"
        for class_id, class_name in CAMUS_CLASS_NAMES.items():
            structure_labels = CAMUS_STRUCTURE_LABELS[class_id]
            pred_binary = np.isin(prediction, structure_labels)
            target_binary = np.isin(target, structure_labels)
            row: dict[str, Any] = {
                "patient_id": patient_id,
                "view": view,
                "inference_frame_index": inference_index,
                "original_frame_index": int(original_indices[inference_index]),
                "phase": phase,
                "class_id": class_id,
                "class_name": class_name,
                "target_pixels": int(target_binary.sum()),
                "predicted_pixels": int(pred_binary.sum()),
            }
            row.update(binary_metrics(pred_binary, target_binary, spacing_yx_mm))
            rows.append(row)
    return rows


def _mean_metric(rows: Iterable[dict[str, Any]], metric_name: str) -> float:
    values = [float(row[metric_name]) for row in rows]
    if not values:
        raise ValueError("cannot aggregate an empty metric group")
    return float(np.mean(np.asarray(values, dtype=np.float64)))


def aggregate_metric_rows(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("cannot aggregate an empty metric row list")
    grouped: dict[tuple[str, str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["patient_id"]), str(row["view"]), int(row["class_id"]))].append(row)

    per_video: list[dict[str, Any]] = []
    for (patient_id, view, class_id), group in sorted(grouped.items()):
        record: dict[str, Any] = {
            "patient_id": patient_id,
            "view": view,
            "class_id": class_id,
            "class_name": CAMUS_CLASS_NAMES.get(class_id, f"class_{class_id}"),
            "frame_count": len(group),
        }
        record.update({name: _mean_metric(group, name) for name in METRIC_NAMES})
        per_video.append(record)

    class_groups: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in per_video:
        class_groups[int(row["class_id"])].append(row)
    per_class: list[dict[str, Any]] = []
    for class_id, group in sorted(class_groups.items()):
        record = {
            "class_id": class_id,
            "class_name": CAMUS_CLASS_NAMES.get(class_id, f"class_{class_id}"),
            "video_count": len(group),
        }
        record.update({name: _mean_metric(group, name) for name in METRIC_NAMES})
        per_class.append(record)

    macro = {name: _mean_metric(per_class, name) for name in METRIC_NAMES}
    macro["class_count"] = len(per_class)
    macro["video_count"] = len({(row["patient_id"], row["view"]) for row in rows})
    macro["frame_row_count"] = len(rows)
    return {"per_video": per_video, "per_class": per_class, "macro": macro}
'''

# Native module: test-code/source/MedSAM2-main/efficient_track_anything/__init__.py
_EMBEDDED_SOURCES['test-code/source/MedSAM2-main/efficient_track_anything/__init__.py'] = '''# singlefile-special-source-transform-v1
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from pathlib import Path
from hydra import initialize_config_dir
from hydra.core.global_hydra import GlobalHydra

if not GlobalHydra.instance().is_initialized():
    initialize_config_dir(config_dir=str(Path(__file__).resolve().parent), version_base="1.2")
'''

# Native module: test-code/source/MedSAM2-main/efficient_track_anything/modeling/__init__.py
_EMBEDDED_SOURCES['test-code/source/MedSAM2-main/efficient_track_anything/modeling/__init__.py'] = '''# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
'''

# Native module: test-code/source/MedSAM2-main/efficient_track_anything/modeling/efficienttam_utils.py
_EMBEDDED_SOURCES['test-code/source/MedSAM2-main/efficient_track_anything/modeling/efficienttam_utils.py'] = '''# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import copy
from typing import Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from efficient_track_anything.utils.misc import mask_to_box


def select_closest_cond_frames(frame_idx, cond_frame_outputs, max_cond_frame_num):
    """
    Select up to `max_cond_frame_num` conditioning frames from `cond_frame_outputs`
    that are temporally closest to the current frame at `frame_idx`. Here, we take
    - a) the closest conditioning frame before `frame_idx` (if any);
    - b) the closest conditioning frame after `frame_idx` (if any);
    - c) any other temporally closest conditioning frames until reaching a total
         of `max_cond_frame_num` conditioning frames.

    Outputs:
    - selected_outputs: selected items (keys & values) from `cond_frame_outputs`.
    - unselected_outputs: items (keys & values) not selected in `cond_frame_outputs`.
    """
    if max_cond_frame_num == -1 or len(cond_frame_outputs) <= max_cond_frame_num:
        selected_outputs = cond_frame_outputs
        unselected_outputs = {}
    else:
        assert max_cond_frame_num >= 2, "we should allow using 2+ conditioning frames"
        selected_outputs = {}

        # the closest conditioning frame before `frame_idx` (if any)
        idx_before = max((t for t in cond_frame_outputs if t < frame_idx), default=None)
        if idx_before is not None:
            selected_outputs[idx_before] = cond_frame_outputs[idx_before]

        # the closest conditioning frame after `frame_idx` (if any)
        idx_after = min((t for t in cond_frame_outputs if t >= frame_idx), default=None)
        if idx_after is not None:
            selected_outputs[idx_after] = cond_frame_outputs[idx_after]

        # add other temporally closest conditioning frames until reaching a total
        # of `max_cond_frame_num` conditioning frames.
        num_remain = max_cond_frame_num - len(selected_outputs)
        inds_remain = sorted(
            (t for t in cond_frame_outputs if t not in selected_outputs),
            key=lambda x: abs(x - frame_idx),
        )[:num_remain]
        selected_outputs.update((t, cond_frame_outputs[t]) for t in inds_remain)
        unselected_outputs = {
            t: v for t, v in cond_frame_outputs.items() if t not in selected_outputs
        }

    return selected_outputs, unselected_outputs


def get_1d_sine_pe(pos_inds, dim, temperature=10000):
    """
    Get 1D sine positional embedding as in the original Transformer paper.
    """
    pe_dim = dim // 2
    dim_t = torch.arange(pe_dim, dtype=torch.float32, device=pos_inds.device)
    dim_t = temperature ** (2 * (dim_t // 2) / pe_dim)

    pos_embed = pos_inds.unsqueeze(-1) / dim_t
    pos_embed = torch.cat([pos_embed.sin(), pos_embed.cos()], dim=-1)
    return pos_embed


def get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(f"activation should be relu/gelu, not {activation}.")


def get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


class DropPath(nn.Module):
    # adapted from https://github.com/huggingface/pytorch-image-models/blob/main/timm/layers/drop.py
    def __init__(self, drop_prob=0.0, scale_by_keep=True):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x):
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
        if keep_prob > 0.0 and self.scale_by_keep:
            random_tensor.div_(keep_prob)
        return x * random_tensor


# Lightly adapted from
# https://github.com/facebookresearch/MaskFormer/blob/main/mask_former/modeling/transformer/transformer_predictor.py # noqa
class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int,
        activation: nn.Module = nn.ReLU,
        sigmoid_output: bool = False,
    ) -> None:
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )
        self.sigmoid_output = sigmoid_output
        self.act = activation()

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = self.act(layer(x)) if i < self.num_layers - 1 else layer(x)
        if self.sigmoid_output:
            x = F.sigmoid(x)
        return x


# From https://github.com/facebookresearch/detectron2/blob/main/detectron2/layers/batch_norm.py # noqa
# Itself from https://github.com/facebookresearch/ConvNeXt/blob/d1fa8f6fef0a165b27399986cc2bdacc92777e40/models/convnext.py#L119  # noqa
class LayerNorm2d(nn.Module):
    def __init__(self, num_channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x


def sample_box_points(
    masks: torch.Tensor,
    noise: float = 0.1,  # SAM default
    noise_bound: int = 20,  # SAM default
    top_left_label: int = 2,
    bottom_right_label: int = 3,
) -> Tuple[np.array, np.array]:
    """
    Sample a noised version of the top left and bottom right corners of a given `bbox`

    Inputs:
    - masks: [B, 1, H,W] boxes, dtype=torch.Tensor
    - noise: noise as a fraction of box width and height, dtype=float
    - noise_bound: maximum amount of noise (in pure pixesl), dtype=int

    Returns:
    - box_coords: [B, num_pt, 2], contains (x, y) coordinates of top left and bottom right box corners, dtype=torch.float
    - box_labels: [B, num_pt], label 2 is reserverd for top left and 3 for bottom right corners, dtype=torch.int32
    """
    device = masks.device
    box_coords = mask_to_box(masks)
    B, _, H, W = masks.shape
    box_labels = torch.tensor(
        [top_left_label, bottom_right_label], dtype=torch.int, device=device
    ).repeat(B)
    if noise > 0.0:
        if not isinstance(noise_bound, torch.Tensor):
            noise_bound = torch.tensor(noise_bound, device=device)
        bbox_w = box_coords[..., 2] - box_coords[..., 0]
        bbox_h = box_coords[..., 3] - box_coords[..., 1]
        max_dx = torch.min(bbox_w * noise, noise_bound)
        max_dy = torch.min(bbox_h * noise, noise_bound)
        box_noise = 2 * torch.rand(B, 1, 4, device=device) - 1
        box_noise = box_noise * torch.stack((max_dx, max_dy, max_dx, max_dy), dim=-1)

        box_coords = box_coords + box_noise
        img_bounds = (
            torch.tensor([W, H, W, H], device=device) - 1
        )  # uncentered pixel coords
        box_coords.clamp_(torch.zeros_like(img_bounds), img_bounds)  # In place clamping

    box_coords = box_coords.reshape(-1, 2, 2)  # always 2 points
    box_labels = box_labels.reshape(-1, 2)
    return box_coords, box_labels


def sample_random_points_from_errors(gt_masks, pred_masks, num_pt=1):
    """
    Sample `num_pt` random points (along with their labels) independently from the error regions.

    Inputs:
    - gt_masks: [B, 1, H_im, W_im] masks, dtype=torch.bool
    - pred_masks: [B, 1, H_im, W_im] masks, dtype=torch.bool or None
    - num_pt: int, number of points to sample independently for each of the B error maps

    Outputs:
    - points: [B, num_pt, 2], dtype=torch.float, contains (x, y) coordinates of each sampled point
    - labels: [B, num_pt], dtype=torch.int32, where 1 means positive clicks and 0 means
      negative clicks
    """
    if pred_masks is None:  # if pred_masks is not provided, treat it as empty
        pred_masks = torch.zeros_like(gt_masks)
    assert gt_masks.dtype == torch.bool and gt_masks.size(1) == 1
    assert pred_masks.dtype == torch.bool and pred_masks.shape == gt_masks.shape
    assert num_pt >= 0

    B, _, H_im, W_im = gt_masks.shape
    device = gt_masks.device

    # false positive region, a new point sampled in this region should have
    # negative label to correct the FP error
    fp_masks = ~gt_masks & pred_masks
    # false negative region, a new point sampled in this region should have
    # positive label to correct the FN error
    fn_masks = gt_masks & ~pred_masks
    # whether the prediction completely match the ground-truth on each mask
    all_correct = torch.all((gt_masks == pred_masks).flatten(2), dim=2)
    all_correct = all_correct[..., None, None]

    # channel 0 is FP map, while channel 1 is FN map
    pts_noise = torch.rand(B, num_pt, H_im, W_im, 2, device=device)
    # sample a negative new click from FP region or a positive new click
    # from FN region, depend on where the maximum falls,
    # and in case the predictions are all correct (no FP or FN), we just
    # sample a negative click from the background region
    pts_noise[..., 0] *= fp_masks | (all_correct & ~gt_masks)
    pts_noise[..., 1] *= fn_masks
    pts_idx = pts_noise.flatten(2).argmax(dim=2)
    labels = (pts_idx % 2).to(torch.int32)
    pts_idx = pts_idx // 2
    pts_x = pts_idx % W_im
    pts_y = pts_idx // W_im
    points = torch.stack([pts_x, pts_y], dim=2).to(torch.float)
    return points, labels


def sample_one_point_from_error_center(gt_masks, pred_masks, padding=True):
    """
    Sample 1 random point (along with its label) from the center of each error region,
    that is, the point with the largest distance to the boundary of each error region.
    This is the RITM sampling method from https://github.com/saic-vul/ritm_interactive_segmentation/blob/master/isegm/inference/clicker.py

    Inputs:
    - gt_masks: [B, 1, H_im, W_im] masks, dtype=torch.bool
    - pred_masks: [B, 1, H_im, W_im] masks, dtype=torch.bool or None
    - padding: if True, pad with boundary of 1 px for distance transform

    Outputs:
    - points: [B, 1, 2], dtype=torch.float, contains (x, y) coordinates of each sampled point
    - labels: [B, 1], dtype=torch.int32, where 1 means positive clicks and 0 means negative clicks
    """
    import cv2

    if pred_masks is None:
        pred_masks = torch.zeros_like(gt_masks)
    assert gt_masks.dtype == torch.bool and gt_masks.size(1) == 1
    assert pred_masks.dtype == torch.bool and pred_masks.shape == gt_masks.shape

    B, _, _, W_im = gt_masks.shape
    device = gt_masks.device

    # false positive region, a new point sampled in this region should have
    # negative label to correct the FP error
    fp_masks = ~gt_masks & pred_masks
    # false negative region, a new point sampled in this region should have
    # positive label to correct the FN error
    fn_masks = gt_masks & ~pred_masks

    fp_masks = fp_masks.cpu().numpy()
    fn_masks = fn_masks.cpu().numpy()
    points = torch.zeros(B, 1, 2, dtype=torch.float)
    labels = torch.ones(B, 1, dtype=torch.int32)
    for b in range(B):
        fn_mask = fn_masks[b, 0]
        fp_mask = fp_masks[b, 0]
        if padding:
            fn_mask = np.pad(fn_mask, ((1, 1), (1, 1)), "constant")
            fp_mask = np.pad(fp_mask, ((1, 1), (1, 1)), "constant")
        # compute the distance of each point in FN/FP region to its boundary
        fn_mask_dt = cv2.distanceTransform(fn_mask.astype(np.uint8), cv2.DIST_L2, 0)
        fp_mask_dt = cv2.distanceTransform(fp_mask.astype(np.uint8), cv2.DIST_L2, 0)
        if padding:
            fn_mask_dt = fn_mask_dt[1:-1, 1:-1]
            fp_mask_dt = fp_mask_dt[1:-1, 1:-1]

        # take the point in FN/FP region with the largest distance to its boundary
        fn_mask_dt_flat = fn_mask_dt.reshape(-1)
        fp_mask_dt_flat = fp_mask_dt.reshape(-1)
        fn_argmax = np.argmax(fn_mask_dt_flat)
        fp_argmax = np.argmax(fp_mask_dt_flat)
        is_positive = fn_mask_dt_flat[fn_argmax] > fp_mask_dt_flat[fp_argmax]
        pt_idx = fn_argmax if is_positive else fp_argmax
        points[b, 0, 0] = pt_idx % W_im  # x
        points[b, 0, 1] = pt_idx // W_im  # y
        labels[b, 0] = int(is_positive)

    points = points.to(device)
    labels = labels.to(device)
    return points, labels


def get_next_point(gt_masks, pred_masks, method):
    if method == "uniform":
        return sample_random_points_from_errors(gt_masks, pred_masks)
    elif method == "center":
        return sample_one_point_from_error_center(gt_masks, pred_masks)
    else:
        raise ValueError(f"unknown sampling method {method}")


class LayerScale(nn.Module):
    def __init__(
        self,
        dim: int,
        init_values: Union[float, torch.Tensor] = 1e-5,
        inplace: bool = False,
    ) -> None:
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.mul_(self.gamma) if self.inplace else x * self.gamma
'''

# Native module: test-code/source/MedSAM2-main/efficient_track_anything/utils/__init__.py
_EMBEDDED_SOURCES['test-code/source/MedSAM2-main/efficient_track_anything/utils/__init__.py'] = '''# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
'''

# Native module: test-code/source/MedSAM2-main/efficient_track_anything/utils/misc.py
_EMBEDDED_SOURCES['test-code/source/MedSAM2-main/efficient_track_anything/utils/misc.py'] = '''# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import warnings
from threading import Thread

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm


def get_sdpa_settings():
    if torch.cuda.is_available():
        old_gpu = torch.cuda.get_device_properties(0).major < 7
        # only use Flash Attention on Ampere (8.0) or newer GPUs
        use_flash_attn = torch.cuda.get_device_properties(0).major >= 8
        if not use_flash_attn:
            warnings.warn(
                "Flash Attention is disabled as it requires a GPU with Ampere (8.0) CUDA capability.",
                category=UserWarning,
                stacklevel=2,
            )
        # keep math kernel for PyTorch versions before 2.2 (Flash Attention v2 is only
        # available on PyTorch 2.2+, while Flash Attention v1 cannot handle all cases)
        pytorch_version = tuple(int(v) for v in torch.__version__.split(".")[:2])
        if pytorch_version < (2, 2):
            warnings.warn(
                f"You are using PyTorch {torch.__version__} without Flash Attention v2 support. "
                "Consider upgrading to PyTorch 2.2+ for Flash Attention v2 (which could be faster).",
                category=UserWarning,
                stacklevel=2,
            )
        math_kernel_on = pytorch_version < (2, 2) or not use_flash_attn
    else:
        old_gpu = True
        use_flash_attn = False
        math_kernel_on = True

    return old_gpu, use_flash_attn, math_kernel_on


def get_connected_components(mask):
    """
    Get the connected components (8-connectivity) of binary masks of shape (N, 1, H, W).

    Inputs:
    - mask: A binary mask tensor of shape (N, 1, H, W), where 1 is foreground and 0 is
            background.

    Outputs:
    - labels: A tensor of shape (N, 1, H, W) containing the connected component labels
              for foreground pixels and 0 for background pixels.
    - counts: A tensor of shape (N, 1, H, W) containing the area of the connected
              components for foreground pixels and 0 for background pixels.
    """
    from efficient_track_anything import _C

    return _C.get_connected_componnets(mask.to(torch.uint8).contiguous())


def mask_to_box(masks: torch.Tensor):
    """
    compute bounding box given an input mask

    Inputs:
    - masks: [B, 1, H, W] masks, dtype=torch.Tensor

    Returns:
    - box_coords: [B, 1, 4], contains (x, y) coordinates of top left and bottom right box corners, dtype=torch.Tensor
    """
    B, _, h, w = masks.shape
    device = masks.device
    xs = torch.arange(w, device=device, dtype=torch.int32)
    ys = torch.arange(h, device=device, dtype=torch.int32)
    grid_xs, grid_ys = torch.meshgrid(xs, ys, indexing="xy")
    grid_xs = grid_xs[None, None, ...].expand(B, 1, h, w)
    grid_ys = grid_ys[None, None, ...].expand(B, 1, h, w)
    min_xs, _ = torch.min(torch.where(masks, grid_xs, w).flatten(-2), dim=-1)
    max_xs, _ = torch.max(torch.where(masks, grid_xs, -1).flatten(-2), dim=-1)
    min_ys, _ = torch.min(torch.where(masks, grid_ys, h).flatten(-2), dim=-1)
    max_ys, _ = torch.max(torch.where(masks, grid_ys, -1).flatten(-2), dim=-1)
    bbox_coords = torch.stack((min_xs, min_ys, max_xs, max_ys), dim=-1)

    return bbox_coords


def _load_img_as_tensor(img_path, image_size):
    img_pil = Image.open(img_path)
    img_np = np.array(img_pil.convert("RGB").resize((image_size, image_size)))
    if img_np.dtype == np.uint8:  # np.uint8 is expected for JPEG images
        img_np = img_np / 255.0
    else:
        raise RuntimeError(f"Unknown image dtype: {img_np.dtype} on {img_path}")
    img = torch.from_numpy(img_np).permute(2, 0, 1)
    video_width, video_height = img_pil.size  # the original video size
    return img, video_height, video_width


class AsyncVideoFrameLoader:
    """
    A list of video frames to be load asynchronously without blocking session start.
    """

    def __init__(
        self,
        img_paths,
        image_size,
        offload_video_to_cpu,
        img_mean,
        img_std,
        compute_device,
    ):
        self.img_paths = img_paths
        self.image_size = image_size
        self.offload_video_to_cpu = offload_video_to_cpu
        self.img_mean = img_mean
        self.img_std = img_std
        # items in `self.images` will be loaded asynchronously
        self.images = [None] * len(img_paths)
        # catch and raise any exceptions in the async loading thread
        self.exception = None
        # video_height and video_width be filled when loading the first image
        self.video_height = None
        self.video_width = None
        self.compute_device = compute_device

        # load the first frame to fill video_height and video_width and also
        # to cache it (since it's most likely where the user will click)
        self.__getitem__(0)

        # load the rest of frames asynchronously without blocking the session start
        def _load_frames():
            try:
                for n in tqdm(range(len(self.images)), desc="frame loading (JPEG)"):
                    self.__getitem__(n)
            except Exception as e:
                self.exception = e

        self.thread = Thread(target=_load_frames, daemon=True)
        self.thread.start()

    def __getitem__(self, index):
        if self.exception is not None:
            raise RuntimeError("Failure in frame loading thread") from self.exception

        img = self.images[index]
        if img is not None:
            return img

        img, video_height, video_width = _load_img_as_tensor(
            self.img_paths[index], self.image_size
        )
        self.video_height = video_height
        self.video_width = video_width
        # normalize by mean and std
        img -= self.img_mean
        img /= self.img_std
        if not self.offload_video_to_cpu:
            img = img.to(self.compute_device, non_blocking=True)
        self.images[index] = img
        return img

    def __len__(self):
        return len(self.images)


def load_video_frames(
    video_path,
    image_size,
    offload_video_to_cpu,
    img_mean=(0.485, 0.456, 0.406),
    img_std=(0.229, 0.224, 0.225),
    async_loading_frames=False,
    compute_device=torch.device("cuda"),
):
    """
    Load the video frames from video_path. The frames are resized to image_size as in
    the model and are loaded to GPU if offload_video_to_cpu=False. This is used by the demo.
    """
    is_bytes = isinstance(video_path, bytes)
    is_str = isinstance(video_path, str)
    is_mp4_path = is_str and os.path.splitext(video_path)[-1] in [".mp4", ".MP4"]
    if is_bytes or is_mp4_path:
        return load_video_frames_from_video_file(
            video_path=video_path,
            image_size=image_size,
            offload_video_to_cpu=offload_video_to_cpu,
            img_mean=img_mean,
            img_std=img_std,
            compute_device=compute_device,
        )
    elif is_str and os.path.isdir(video_path):
        return load_video_frames_from_jpg_images(
            video_path=video_path,
            image_size=image_size,
            offload_video_to_cpu=offload_video_to_cpu,
            img_mean=img_mean,
            img_std=img_std,
            async_loading_frames=async_loading_frames,
            compute_device=compute_device,
        )
    else:
        raise NotImplementedError(
            "Only MP4 video and JPEG folder are supported at this moment"
        )


def load_video_frames_from_jpg_images(
    video_path,
    image_size,
    offload_video_to_cpu,
    img_mean=(0.485, 0.456, 0.406),
    img_std=(0.229, 0.224, 0.225),
    async_loading_frames=False,
    compute_device=torch.device("cuda"),
):
    """
    Load the video frames from a directory of JPEG files ("<frame_index>.jpg" format).

    The frames are resized to image_size x image_size and are loaded to GPU if
    `offload_video_to_cpu` is `False` and to CPU if `offload_video_to_cpu` is `True`.

    You can load a frame asynchronously by setting `async_loading_frames` to `True`.
    """
    if isinstance(video_path, str) and os.path.isdir(video_path):
        jpg_folder = video_path
    else:
        raise NotImplementedError(
            "Only JPEG frames are supported at this moment. For video files, you may use "
            "ffmpeg (https://ffmpeg.org/) to extract frames into a folder of JPEG files, such as \\n"
            "```\\n"
            "ffmpeg -i <your_video>.mp4 -q:v 2 -start_number 0 <output_dir>/'%05d.jpg'\\n"
            "```\\n"
            "where `-q:v` generates high-quality JPEG frames and `-start_number 0` asks "
            "ffmpeg to start the JPEG file from 00000.jpg."
        )

    frame_names = [
        p
        for p in os.listdir(jpg_folder)
        if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG"]
    ]
    frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))
    num_frames = len(frame_names)
    if num_frames == 0:
        raise RuntimeError(f"no images found in {jpg_folder}")
    img_paths = [os.path.join(jpg_folder, frame_name) for frame_name in frame_names]
    img_mean = torch.tensor(img_mean, dtype=torch.float32)[:, None, None]
    img_std = torch.tensor(img_std, dtype=torch.float32)[:, None, None]

    if async_loading_frames:
        lazy_images = AsyncVideoFrameLoader(
            img_paths,
            image_size,
            offload_video_to_cpu,
            img_mean,
            img_std,
            compute_device,
        )
        return lazy_images, lazy_images.video_height, lazy_images.video_width

    images = torch.zeros(num_frames, 3, image_size, image_size, dtype=torch.float32)
    for n, img_path in enumerate(tqdm(img_paths, desc="frame loading (JPEG)")):
        images[n], video_height, video_width = _load_img_as_tensor(img_path, image_size)
    if not offload_video_to_cpu:
        images = images.to(compute_device)
        img_mean = img_mean.to(compute_device)
        img_std = img_std.to(compute_device)
    # normalize by mean and std
    images -= img_mean
    images /= img_std
    return images, video_height, video_width


def load_video_frames_from_video_file(
    video_path,
    image_size,
    offload_video_to_cpu,
    img_mean=(0.485, 0.456, 0.406),
    img_std=(0.229, 0.224, 0.225),
    compute_device=torch.device("cuda"),
):
    """Load the video frames from a video file."""
    import decord

    img_mean = torch.tensor(img_mean, dtype=torch.float32)[:, None, None]
    img_std = torch.tensor(img_std, dtype=torch.float32)[:, None, None]
    # Get the original video height and width
    decord.bridge.set_bridge("torch")
    video_height, video_width, _ = decord.VideoReader(video_path).next().shape
    # Iterate over all frames in the video
    images = []
    for frame in decord.VideoReader(video_path, width=image_size, height=image_size):
        images.append(frame.permute(2, 0, 1))

    images = torch.stack(images, dim=0).float() / 255.0
    if not offload_video_to_cpu:
        images = images.to(compute_device)
        img_mean = img_mean.to(compute_device)
        img_std = img_std.to(compute_device)
    # normalize by mean and std
    images -= img_mean
    images /= img_std
    return images, video_height, video_width


def fill_holes_in_mask_scores(mask, max_area):
    """
    A post processor to fill small holes in mask scores with area under `max_area`.
    """
    # Holes are those connected components in background with area <= self.max_area
    # (background regions are those with mask scores <= 0)
    assert max_area > 0, "max_area must be positive"

    input_mask = mask
    try:
        labels, areas = get_connected_components(mask <= 0)
        is_hole = (labels > 0) & (areas <= max_area)
        # We fill holes with a small positive mask score (0.1) to change them to foreground.
        mask = torch.where(is_hole, 0.1, mask)
    except Exception as e:
        # Following SAM 2, skip the post-processing step on removing small holes if the CUDA kernel fails
        warnings.warn(
            f"{e}\\n\\nSkipping the post-processing step due to the error above. You can "
            "still use Efficient Track Anything and it's OK to ignore the error above.",
            category=UserWarning,
            stacklevel=2,
        )
        mask = input_mask

    return mask


def concat_points(old_point_inputs, new_points, new_labels):
    """Add new points and labels to previous point inputs (add at the end)."""
    if old_point_inputs is None:
        points, labels = new_points, new_labels
    else:
        points = torch.cat([old_point_inputs["point_coords"], new_points], dim=1)
        labels = torch.cat([old_point_inputs["point_labels"], new_labels], dim=1)

    return {"point_coords": points, "point_labels": labels}
'''

# Native module: test-code/source/MedSAM2-main/sam2/__init__.py
_EMBEDDED_SOURCES['test-code/source/MedSAM2-main/sam2/__init__.py'] = '''# singlefile-special-source-transform-v1
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from pathlib import Path
from hydra import initialize_config_dir
from hydra.core.global_hydra import GlobalHydra

if not GlobalHydra.instance().is_initialized():
    initialize_config_dir(config_dir=str(Path(__file__).resolve().parent), version_base="1.2")
'''

# Native module: test-code/source/MedSAM2-main/sam2/build_sam.py
_EMBEDDED_SOURCES['test-code/source/MedSAM2-main/sam2/build_sam.py'] = '''# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging

import torch
from hydra import compose
from hydra.utils import instantiate
from omegaconf import OmegaConf

HF_MODEL_ID_TO_FILENAMES = {
    "facebook/sam2-hiera-tiny": (
        "configs/sam2/sam2_hiera_t.yaml",
        "sam2_hiera_tiny.pt",
    ),
    "facebook/sam2-hiera-small": (
        "configs/sam2/sam2_hiera_s.yaml",
        "sam2_hiera_small.pt",
    ),
    "facebook/sam2-hiera-base-plus": (
        "configs/sam2/sam2_hiera_b+.yaml",
        "sam2_hiera_base_plus.pt",
    ),
    "facebook/sam2-hiera-large": (
        "configs/sam2/sam2_hiera_l.yaml",
        "sam2_hiera_large.pt",
    ),
    "facebook/sam2.1-hiera-tiny": (
        "configs/sam2.1/sam2.1_hiera_t.yaml",
        "sam2.1_hiera_tiny.pt",
    ),
    "facebook/sam2.1-hiera-small": (
        "configs/sam2.1/sam2.1_hiera_s.yaml",
        "sam2.1_hiera_small.pt",
    ),
    "facebook/sam2.1-hiera-base-plus": (
        "configs/sam2.1/sam2.1_hiera_b+.yaml",
        "sam2.1_hiera_base_plus.pt",
    ),
    "facebook/sam2.1-hiera-large": (
        "configs/sam2.1/sam2.1_hiera_l.yaml",
        "sam2.1_hiera_large.pt",
    ),
}


def get_best_available_device():
    """
    Get the best available device in the order: CUDA, MPS, CPU
    Returns: device string for torch.device
    """
    if torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"


def build_sam2(
    config_file,
    ckpt_path=None,
    device=None,
    mode="eval",
    hydra_overrides_extra=[],
    apply_postprocessing=True,
    **kwargs,
):
    # Use the provided device or get the best available one
    device = device or get_best_available_device()
    logging.info(f"Using device: {device}")

    if apply_postprocessing:
        hydra_overrides_extra = hydra_overrides_extra.copy()
        hydra_overrides_extra += [
            # dynamically fall back to multi-mask if the single mask is not stable
            "++model.sam_mask_decoder_extra_args.dynamic_multimask_via_stability=true",
            "++model.sam_mask_decoder_extra_args.dynamic_multimask_stability_delta=0.05",
            "++model.sam_mask_decoder_extra_args.dynamic_multimask_stability_thresh=0.98",
        ]
    # Read config and init model
    cfg = compose(config_name=config_file, overrides=hydra_overrides_extra)
    OmegaConf.resolve(cfg)
    model = instantiate(cfg.model, _recursive_=True)
    _load_checkpoint(model, ckpt_path)
    model = model.to(device)
    if mode == "eval":
        model.eval()
    return model


def build_sam2_video_predictor(
    config_file,
    ckpt_path=None,
    device=None,
    mode="eval",
    hydra_overrides_extra=[],
    apply_postprocessing=True,
    **kwargs,
):
    # Use the provided device or get the best available one
    device = device or get_best_available_device()
    logging.info(f"Using device: {device}")

    hydra_overrides = [
        "++model._target_=sam2.sam2_video_predictor.SAM2VideoPredictor",
    ]
    if apply_postprocessing:
        hydra_overrides_extra = hydra_overrides_extra.copy()
        hydra_overrides_extra += [
            # dynamically fall back to multi-mask if the single mask is not stable
            "++model.sam_mask_decoder_extra_args.dynamic_multimask_via_stability=true",
            "++model.sam_mask_decoder_extra_args.dynamic_multimask_stability_delta=0.05",
            "++model.sam_mask_decoder_extra_args.dynamic_multimask_stability_thresh=0.98",
            # the sigmoid mask logits on interacted frames with clicks in the memory encoder so that the encoded masks are exactly as what users see from clicking
            "++model.binarize_mask_from_pts_for_mem_enc=true",
            # fill small holes in the low-res masks up to `fill_hole_area` (before resizing them to the original video resolution)
            "++model.fill_hole_area=8",
        ]
    hydra_overrides.extend(hydra_overrides_extra)

    # Read config and init model
    cfg = compose(config_name=config_file, overrides=hydra_overrides)
    OmegaConf.resolve(cfg)
    model = instantiate(cfg.model, _recursive_=True)
    _load_checkpoint(model, ckpt_path)
    model = model.to(device)
    if mode == "eval":
        model.eval()
    return model

def build_sam2_video_predictor_npz(
    config_file,
    ckpt_path=None,
    device=None,
    mode="eval",
    hydra_overrides_extra=[],
    apply_postprocessing=True,
    **kwargs,
):
    # Use the provided device or get the best available one
    device = device or get_best_available_device()
    logging.info(f"Using device: {device}")

    hydra_overrides = [
        "++model._target_=sam2.sam2_video_predictor_npz.SAM2VideoPredictorNPZ",
    ]
    if apply_postprocessing:
        hydra_overrides_extra = hydra_overrides_extra.copy()
        hydra_overrides_extra += [
            # dynamically fall back to multi-mask if the single mask is not stable
            "++model.sam_mask_decoder_extra_args.dynamic_multimask_via_stability=true",
            "++model.sam_mask_decoder_extra_args.dynamic_multimask_stability_delta=0.05",
            "++model.sam_mask_decoder_extra_args.dynamic_multimask_stability_thresh=0.98",
            # the sigmoid mask logits on interacted frames with clicks in the memory encoder so that the encoded masks are exactly as what users see from clicking
            "++model.binarize_mask_from_pts_for_mem_enc=true",
            # fill small holes in the low-res masks up to `fill_hole_area` (before resizing them to the original video resolution)
            "++model.fill_hole_area=8",
        ]
    hydra_overrides.extend(hydra_overrides_extra)

    # Read config and init model
    cfg = compose(config_name=config_file, overrides=hydra_overrides)
    OmegaConf.resolve(cfg)
    model = instantiate(cfg.model, _recursive_=True)
    _load_checkpoint(model, ckpt_path)
    model = model.to(device)
    if mode == "eval":
        model.eval()
    return model



def _hf_download(model_id):
    from huggingface_hub import hf_hub_download

    config_name, checkpoint_name = HF_MODEL_ID_TO_FILENAMES[model_id]
    ckpt_path = hf_hub_download(repo_id=model_id, filename=checkpoint_name)
    return config_name, ckpt_path


def build_sam2_hf(model_id, **kwargs):
    config_name, ckpt_path = _hf_download(model_id)
    return build_sam2(config_file=config_name, ckpt_path=ckpt_path, **kwargs)


def build_sam2_video_predictor_hf(model_id, **kwargs):
    config_name, ckpt_path = _hf_download(model_id)
    return build_sam2_video_predictor(
        config_file=config_name, ckpt_path=ckpt_path, **kwargs
    )


def _load_checkpoint(model, ckpt_path):
    if ckpt_path is not None:
        sd = torch.load(ckpt_path, map_location="cpu", weights_only=True)["model"]
        missing_keys, unexpected_keys = model.load_state_dict(sd)
        if missing_keys:
            logging.error(missing_keys)
            raise RuntimeError()
        if unexpected_keys:
            logging.error(unexpected_keys)
            raise RuntimeError()
        logging.info("Loaded checkpoint sucessfully")
'''

# Native module: test-code/source/MedSAM2-main/sam2/modeling/__init__.py
_EMBEDDED_SOURCES['test-code/source/MedSAM2-main/sam2/modeling/__init__.py'] = '''# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
'''

# Native module: test-code/source/MedSAM2-main/sam2/modeling/backbones/__init__.py
_EMBEDDED_SOURCES['test-code/source/MedSAM2-main/sam2/modeling/backbones/__init__.py'] = '''# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
'''

# Native module: test-code/source/MedSAM2-main/sam2/modeling/backbones/hieradet.py
_EMBEDDED_SOURCES['test-code/source/MedSAM2-main/sam2/modeling/backbones/hieradet.py'] = '''# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging
from functools import partial
from typing import List, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from iopath.common.file_io import g_pathmgr

from sam2.modeling.backbones.utils import (
    PatchEmbed,
    window_partition,
    window_unpartition,
)

from sam2.modeling.sam2_utils import DropPath, MLP


def do_pool(x: torch.Tensor, pool: nn.Module, norm: nn.Module = None) -> torch.Tensor:
    if pool is None:
        return x
    # (B, H, W, C) -> (B, C, H, W)
    x = x.permute(0, 3, 1, 2)
    x = pool(x)
    # (B, C, H', W') -> (B, H', W', C)
    x = x.permute(0, 2, 3, 1)
    if norm:
        x = norm(x)

    return x


class MultiScaleAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        dim_out: int,
        num_heads: int,
        q_pool: nn.Module = None,
    ):
        super().__init__()

        self.dim = dim
        self.dim_out = dim_out
        self.num_heads = num_heads
        self.q_pool = q_pool
        self.qkv = nn.Linear(dim, dim_out * 3)
        self.proj = nn.Linear(dim_out, dim_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, H, W, _ = x.shape
        # qkv with shape (B, H * W, 3, nHead, C)
        qkv = self.qkv(x).reshape(B, H * W, 3, self.num_heads, -1)
        # q, k, v with shape (B, H * W, nheads, C)
        q, k, v = torch.unbind(qkv, 2)

        # Q pooling (for downsample at stage changes)
        if self.q_pool:
            q = do_pool(q.reshape(B, H, W, -1), self.q_pool)
            H, W = q.shape[1:3]  # downsampled shape
            q = q.reshape(B, H * W, self.num_heads, -1)

        # Torch's SDPA expects [B, nheads, H*W, C] so we transpose
        x = F.scaled_dot_product_attention(
            q.transpose(1, 2),
            k.transpose(1, 2),
            v.transpose(1, 2),
        )
        # Transpose back
        x = x.transpose(1, 2)
        x = x.reshape(B, H, W, -1)

        x = self.proj(x)

        return x


class MultiScaleBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        dim_out: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        drop_path: float = 0.0,
        norm_layer: Union[nn.Module, str] = "LayerNorm",
        q_stride: Tuple[int, int] = None,
        act_layer: nn.Module = nn.GELU,
        window_size: int = 0,
    ):
        super().__init__()

        if isinstance(norm_layer, str):
            norm_layer = partial(getattr(nn, norm_layer), eps=1e-6)

        self.dim = dim
        self.dim_out = dim_out
        self.norm1 = norm_layer(dim)

        self.window_size = window_size

        self.pool, self.q_stride = None, q_stride
        if self.q_stride:
            self.pool = nn.MaxPool2d(
                kernel_size=q_stride, stride=q_stride, ceil_mode=False
            )

        self.attn = MultiScaleAttention(
            dim,
            dim_out,
            num_heads=num_heads,
            q_pool=self.pool,
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.norm2 = norm_layer(dim_out)
        self.mlp = MLP(
            dim_out,
            int(dim_out * mlp_ratio),
            dim_out,
            num_layers=2,
            activation=act_layer,
        )

        if dim != dim_out:
            self.proj = nn.Linear(dim, dim_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x  # B, H, W, C
        x = self.norm1(x)

        # Skip connection
        if self.dim != self.dim_out:
            shortcut = do_pool(self.proj(x), self.pool)

        # Window partition
        window_size = self.window_size
        if window_size > 0:
            H, W = x.shape[1], x.shape[2]
            x, pad_hw = window_partition(x, window_size)

        # Window Attention + Q Pooling (if stage change)
        x = self.attn(x)
        if self.q_stride:
            # Shapes have changed due to Q pooling
            window_size = self.window_size // self.q_stride[0]
            H, W = shortcut.shape[1:3]

            pad_h = (window_size - H % window_size) % window_size
            pad_w = (window_size - W % window_size) % window_size
            pad_hw = (H + pad_h, W + pad_w)

        # Reverse window partition
        if self.window_size > 0:
            x = window_unpartition(x, window_size, pad_hw, (H, W))

        x = shortcut + self.drop_path(x)
        # MLP
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class Hiera(nn.Module):
    """
    Reference: https://arxiv.org/abs/2306.00989
    """

    def __init__(
        self,
        embed_dim: int = 96,  # initial embed dim
        num_heads: int = 1,  # initial number of heads
        drop_path_rate: float = 0.0,  # stochastic depth
        q_pool: int = 3,  # number of q_pool stages
        q_stride: Tuple[int, int] = (2, 2),  # downsample stride bet. stages
        stages: Tuple[int, ...] = (2, 3, 16, 3),  # blocks per stage
        dim_mul: float = 2.0,  # dim_mul factor at stage shift
        head_mul: float = 2.0,  # head_mul factor at stage shift
        window_pos_embed_bkg_spatial_size: Tuple[int, int] = (14, 14),
        # window size per stage, when not using global att.
        window_spec: Tuple[int, ...] = (
            8,
            4,
            14,
            7,
        ),
        # global attn in these blocks
        global_att_blocks: Tuple[int, ...] = (
            12,
            16,
            20,
        ),
        weights_path=None,
        return_interm_layers=True,  # return feats from every stage
    ):
        super().__init__()

        assert len(stages) == len(window_spec)
        self.window_spec = window_spec

        depth = sum(stages)
        self.q_stride = q_stride
        self.stage_ends = [sum(stages[:i]) - 1 for i in range(1, len(stages) + 1)]
        assert 0 <= q_pool <= len(self.stage_ends[:-1])
        self.q_pool_blocks = [x + 1 for x in self.stage_ends[:-1]][:q_pool]
        self.return_interm_layers = return_interm_layers

        self.patch_embed = PatchEmbed(
            embed_dim=embed_dim,
        )
        # Which blocks have global att?
        self.global_att_blocks = global_att_blocks

        # Windowed positional embedding (https://arxiv.org/abs/2311.05613)
        self.window_pos_embed_bkg_spatial_size = window_pos_embed_bkg_spatial_size
        self.pos_embed = nn.Parameter(
            torch.zeros(1, embed_dim, *self.window_pos_embed_bkg_spatial_size)
        )
        self.pos_embed_window = nn.Parameter(
            torch.zeros(1, embed_dim, self.window_spec[0], self.window_spec[0])
        )

        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, depth)
        ]  # stochastic depth decay rule

        cur_stage = 1
        self.blocks = nn.ModuleList()

        for i in range(depth):
            dim_out = embed_dim
            # lags by a block, so first block of
            # next stage uses an initial window size
            # of previous stage and final window size of current stage
            window_size = self.window_spec[cur_stage - 1]

            if self.global_att_blocks is not None:
                window_size = 0 if i in self.global_att_blocks else window_size

            if i - 1 in self.stage_ends:
                dim_out = int(embed_dim * dim_mul)
                num_heads = int(num_heads * head_mul)
                cur_stage += 1

            block = MultiScaleBlock(
                dim=embed_dim,
                dim_out=dim_out,
                num_heads=num_heads,
                drop_path=dpr[i],
                q_stride=self.q_stride if i in self.q_pool_blocks else None,
                window_size=window_size,
            )

            embed_dim = dim_out
            self.blocks.append(block)

        self.channel_list = (
            [self.blocks[i].dim_out for i in self.stage_ends[::-1]]
            if return_interm_layers
            else [self.blocks[-1].dim_out]
        )

        if weights_path is not None:
            with g_pathmgr.open(weights_path, "rb") as f:
                chkpt = torch.load(f, map_location="cpu")
            logging.info("loading Hiera", self.load_state_dict(chkpt, strict=False))

    def _get_pos_embed(self, hw: Tuple[int, int]) -> torch.Tensor:
        h, w = hw
        window_embed = self.pos_embed_window
        pos_embed = F.interpolate(self.pos_embed, size=(h, w), mode="bicubic")
        pos_embed = pos_embed + window_embed.tile(
            [x // y for x, y in zip(pos_embed.shape, window_embed.shape)]
        )
        pos_embed = pos_embed.permute(0, 2, 3, 1)
        return pos_embed

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        x = self.patch_embed(x)
        # x: (B, H, W, C)

        # Add pos embed
        x = x + self._get_pos_embed(x.shape[1:3])

        outputs = []
        for i, blk in enumerate(self.blocks):
            x = blk(x)
            if (i == self.stage_ends[-1]) or (
                i in self.stage_ends and self.return_interm_layers
            ):
                feats = x.permute(0, 3, 1, 2)
                outputs.append(feats)

        return outputs

    def get_layer_id(self, layer_name):
        # https://github.com/microsoft/unilm/blob/master/beit/optim_factory.py#L33
        num_layers = self.get_num_layers()

        if layer_name.find("rel_pos") != -1:
            return num_layers + 1
        elif layer_name.find("pos_embed") != -1:
            return 0
        elif layer_name.find("patch_embed") != -1:
            return 0
        elif layer_name.find("blocks") != -1:
            return int(layer_name.split("blocks")[1].split(".")[1]) + 1
        else:
            return num_layers + 1

    def get_num_layers(self) -> int:
        return len(self.blocks)
'''

# Native module: test-code/source/MedSAM2-main/sam2/modeling/backbones/image_encoder.py
_EMBEDDED_SOURCES['test-code/source/MedSAM2-main/sam2/modeling/backbones/image_encoder.py'] = '''# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from efficient_track_anything.modeling.efficienttam_utils import LayerNorm2d



class ImageEncoder(nn.Module):
    def __init__(
        self,
        trunk: nn.Module,
        neck: nn.Module,
        scalp: int = 0,
    ):
        super().__init__()
        self.trunk = trunk
        self.neck = neck
        self.scalp = scalp
        assert (
            self.trunk.channel_list == self.neck.backbone_channel_list
        ), f"Channel dims of trunk and neck do not match. Trunk: {self.trunk.channel_list}, neck: {self.neck.backbone_channel_list}"

    def forward(self, sample: torch.Tensor):
        # Forward through backbone
        features, pos = self.neck(self.trunk(sample))
        if self.scalp > 0:
            # Discard the lowest resolution features
            features, pos = features[: -self.scalp], pos[: -self.scalp]

        src = features[-1]
        output = {
            "vision_features": src,
            "vision_pos_enc": pos,
            "backbone_fpn": features,
        }
        return output


class FpnNeck(nn.Module):
    """
    A modified variant of Feature Pyramid Network (FPN) neck
    (we remove output conv and also do bicubic interpolation similar to ViT
    pos embed interpolation)
    """

    def __init__(
        self,
        position_encoding: nn.Module,
        d_model: int,
        backbone_channel_list: List[int],
        kernel_size: int = 1,
        stride: int = 1,
        padding: int = 0,
        fpn_interp_model: str = "bilinear",
        fuse_type: str = "sum",
        fpn_top_down_levels: Optional[List[int]] = None,
    ):
        """Initialize the neck
        :param trunk: the backbone
        :param position_encoding: the positional encoding to use
        :param d_model: the dimension of the model
        :param neck_norm: the normalization to use
        """
        super().__init__()
        self.position_encoding = position_encoding
        self.convs = nn.ModuleList()
        self.backbone_channel_list = backbone_channel_list
        self.d_model = d_model
        for dim in backbone_channel_list:
            current = nn.Sequential()
            current.add_module(
                "conv",
                nn.Conv2d(
                    in_channels=dim,
                    out_channels=d_model,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                ),
            )

            self.convs.append(current)
        self.fpn_interp_model = fpn_interp_model
        assert fuse_type in ["sum", "avg"]
        self.fuse_type = fuse_type

        # levels to have top-down features in its outputs
        # e.g. if fpn_top_down_levels is [2, 3], then only outputs of level 2 and 3
        # have top-down propagation, while outputs of level 0 and level 1 have only
        # lateral features from the same backbone level.
        if fpn_top_down_levels is None:
            # default is to have top-down features on all levels
            fpn_top_down_levels = range(len(self.convs))
        self.fpn_top_down_levels = list(fpn_top_down_levels)

    def forward(self, xs: List[torch.Tensor]):

        out = [None] * len(self.convs)
        pos = [None] * len(self.convs)
        assert len(xs) == len(self.convs)
        # fpn forward pass
        # see https://github.com/facebookresearch/detectron2/blob/main/detectron2/modeling/backbone/fpn.py
        prev_features = None
        # forward in top-down order (from low to high resolution)
        n = len(self.convs) - 1
        for i in range(n, -1, -1):
            x = xs[i]
            lateral_features = self.convs[n - i](x)
            if i in self.fpn_top_down_levels and prev_features is not None:
                top_down_features = F.interpolate(
                    prev_features.to(dtype=torch.float32),
                    scale_factor=2.0,
                    mode=self.fpn_interp_model,
                    align_corners=(
                        None if self.fpn_interp_model == "nearest" else False
                    ),
                    antialias=False,
                )
                prev_features = lateral_features + top_down_features
                if self.fuse_type == "avg":
                    prev_features /= 2
            else:
                prev_features = lateral_features
            x_out = prev_features
            out[i] = x_out
            pos[i] = self.position_encoding(x_out).to(x_out.dtype)

        return out, pos


class ViTDetNeck(nn.Module):
    def __init__(
        self,
        position_encoding: nn.Module,
        d_model: int,
        backbone_channel_list: List[int],
        kernel_size: int = 1,
        stride: int = 1,
        padding: int = 0,
        neck_norm=None,
    ):
        """Initialize the neck

        :param trunk: the backbone
        :param position_encoding: the positional encoding to use
        :param d_model: the dimension of the model
        :param neck_norm: the normalization to use
        """
        super().__init__()
        self.backbone_channel_list = backbone_channel_list
        self.position_encoding = position_encoding
        self.convs = nn.ModuleList()
        self.d_model = d_model
        use_bias = neck_norm is None
        for dim in self.backbone_channel_list:
            current = nn.Sequential()
            current.add_module(
                "conv_1x1",
                nn.Conv2d(
                    in_channels=dim,
                    out_channels=d_model,
                    kernel_size=1,
                    bias=use_bias,
                ),
            )
            if neck_norm is not None:
                current.add_module("norm_0", LayerNorm2d(d_model))
            current.add_module(
                "conv_3x3",
                nn.Conv2d(
                    in_channels=d_model,
                    out_channels=d_model,
                    kernel_size=3,
                    padding=1,
                    bias=use_bias,
                ),
            )
            if neck_norm is not None:
                current.add_module("norm_1", LayerNorm2d(d_model))
            self.convs.append(current)

    def forward(self, xs: List[torch.Tensor]):
        out = [None] * len(self.convs)
        pos = [None] * len(self.convs)
        assert len(xs) == len(self.convs)

        x = xs[0]
        x_out = self.convs[0](x)
        out[0] = x_out
        pos[0] = self.position_encoding(x_out).to(x_out.dtype)

        return out, pos


'''

# Native module: test-code/source/MedSAM2-main/sam2/modeling/backbones/utils.py
_EMBEDDED_SOURCES['test-code/source/MedSAM2-main/sam2/modeling/backbones/utils.py'] = '''# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Some utilities for backbones, in particular for windowing"""

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

import math

def window_partition(x, window_size):
    """
    Partition into non-overlapping windows with padding if needed.
    Args:
        x (tensor): input tokens with [B, H, W, C].
        window_size (int): window size.
    Returns:
        windows: windows after partition with [B * num_windows, window_size, window_size, C].
        (Hp, Wp): padded height and width before partition
    """
    B, H, W, C = x.shape

    pad_h = (window_size - H % window_size) % window_size
    pad_w = (window_size - W % window_size) % window_size
    if pad_h > 0 or pad_w > 0:
        x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))
    Hp, Wp = H + pad_h, W + pad_w

    x = x.view(B, Hp // window_size, window_size, Wp // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).reshape(-1, window_size, window_size, C)
    return windows, (Hp, Wp)


def window_unpartition(windows, window_size, pad_hw, hw):
    """
    Window unpartition into original sequences and removing padding.
    Args:
        x (tensor): input tokens with [B * num_windows, window_size, window_size, C].
        window_size (int): window size.
        pad_hw (Tuple): padded height and width (Hp, Wp).
        hw (Tuple): original height and width (H, W) before padding.
    Returns:
        x: unpartitioned sequences with [B, H, W, C].
    """
    Hp, Wp = pad_hw
    H, W = hw
    B = windows.shape[0] // (Hp * Wp // window_size // window_size)
    x = windows.reshape(
        B, Hp // window_size, Wp // window_size, window_size, window_size, -1
    )
    x = x.permute(0, 1, 3, 2, 4, 5).reshape(B, Hp, Wp, -1)

    if Hp > H or Wp > W:
        x = x[:, :H, :W, :]
    return x


class PatchEmbed(nn.Module):
    """
    Image to Patch Embedding.
    """

    def __init__(
        self,
        kernel_size: Tuple[int, ...] = (7, 7),
        stride: Tuple[int, ...] = (4, 4),
        padding: Tuple[int, ...] = (3, 3),
        in_chans: int = 3,
        embed_dim: int = 768,
    ):
        """
        Args:
            kernel_size (Tuple): kernel size of the projection layer.
            stride (Tuple): stride of the projection layer.
            padding (Tuple): padding size of the projection layer.
            in_chans (int): Number of input image channels.
            embed_dim (int):  embed_dim (int): Patch embedding dimension.
        """
        super().__init__()
        self.proj = nn.Conv2d(
            in_chans, embed_dim, kernel_size=kernel_size, stride=stride, padding=padding
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        # B C H W -> B H W C
        x = x.permute(0, 2, 3, 1)
        return x


def get_abs_pos(abs_pos, has_cls_token, hw):
    """
    Calculate absolute positional embeddings. If needed, resize embeddings and remove cls_token
        dimension for the original embeddings.
    Args:
        abs_pos (Tensor): absolute positional embeddings with (1, num_position, C).
        has_cls_token (bool): If true, has 1 embedding in abs_pos for cls token.
        hw (Tuple): size of input image tokens.
    Returns:
        Absolute positional embeddings after processing with shape (1, H, W, C)
    """
    h, w = hw
    if has_cls_token:
        abs_pos = abs_pos[:, 1:]
    xy_num = abs_pos.shape[1]
    size = int(math.sqrt(xy_num))
    assert size * size == xy_num

    if size != h or size != w:
        interpolate_mode = "bicubic"
        if not torch.cuda.is_available() and torch.mps.is_available():
            # bicubic is not supported on torch mps
            interpolate_mode = "bilinear"
        new_abs_pos = F.interpolate(
            abs_pos.reshape(1, size, size, -1).permute(0, 3, 1, 2),
            size=(h, w),
            mode=interpolate_mode,
            align_corners=False,
        )
        return new_abs_pos.permute(0, 2, 3, 1)
    else:
        return abs_pos.reshape(1, h, w, -1)
'''

# Native module: test-code/source/MedSAM2-main/sam2/modeling/backbones/vitdet.py
_EMBEDDED_SOURCES['test-code/source/MedSAM2-main/sam2/modeling/backbones/vitdet.py'] = '''"""ViTDet backbone adapted from Detectron2"""

from functools import partial
from typing import List, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from sam2.modeling.backbones.utils import (
    get_abs_pos,
    PatchEmbed,
    window_partition,
    window_unpartition,
)

from sam2.modeling.efficienttam_utils import (
    DropPath,
    LayerScale,
    MLP,
)


class Attention(nn.Module):
    """Multi-head Attention block with relative position embeddings."""

    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=True,
        use_rel_pos=False,
        rel_pos_zero_init=True,
        input_size=None,
    ):
        """
        Args:
            dim (int): Number of input channels.
            num_heads (int): Number of attention heads.
            qkv_bias (bool:  If True, add a learnable bias to query, key, value.
            rel_pos (bool): If True, add relative positional embeddings to the attention map.
            rel_pos_zero_init (bool): If True, zero initialize relative positional parameters.
            input_size (int or None): Input resolution for calculating the relative positional
                parameter size.
            attn_type: Type of attention operation, e.g. "vanilla", "vanilla-xformer".
        """
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

        self.use_rel_pos = use_rel_pos

    def forward(self, x):
        B, H, W, _ = x.shape
        # qkv with shape (3, B, nHead, H * W, C)
        qkv = (
            self.qkv(x).reshape(B, H * W, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        )
        # q, k, v with shape (B * nHead, H * W, C)
        q, k, v = qkv.reshape(3, B * self.num_heads, H * W, -1).unbind(0)

        q = q.view(B, self.num_heads, H * W, -1)
        k = k.view(B, self.num_heads, H * W, -1)
        v = v.view(B, self.num_heads, H * W, -1)

        x = F.scaled_dot_product_attention(q, k, v)

        x = (
            x.view(B, self.num_heads, H, W, -1)
            .permute(0, 2, 3, 1, 4)
            .reshape(B, H, W, -1)
        )
        x = self.proj(x)

        return x


class Block(nn.Module):
    """Transformer blocks with support of window attention"""

    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=True,
        drop_path=0.0,
        norm_layer=nn.LayerNorm,
        act_layer=nn.GELU,
        use_rel_pos=False,
        rel_pos_zero_init=True,
        window_size=0,
        input_size=None,
        dropout=0.0,
        init_values=None,
    ):
        """
        Args:
            dim (int): Number of input channels.
            num_heads (int): Number of attention heads in each ViT block.
            mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
            qkv_bias (bool): If True, add a learnable bias to query, key, value.
            drop_path (float): Stochastic depth rate.
            norm_layer (nn.Module): Normalization layer.
            act_layer (nn.Module): Activation layer.
            use_rel_pos (bool): If True, add relative positional embeddings to the attention map.
            rel_pos_zero_init (bool): If True, zero initialize relative positional parameters.
            window_size (int): Window size for window attention blocks. If it equals 0, then not
                use window attention.
            input_size (int or None): Input resolution for calculating the relative positional
                parameter size.
            dropout (float): Dropout rate.
        """
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            use_rel_pos=use_rel_pos,
            rel_pos_zero_init=rel_pos_zero_init,
            input_size=input_size if window_size == 0 else (window_size, window_size),
        )
        self.ls1 = (
            LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.norm2 = norm_layer(dim)
        self.mlp = MLP(
            dim,
            int(dim * mlp_ratio),
            dim,
            num_layers=2,
            activation=act_layer,
        )
        self.ls2 = (
            LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        )
        self.dropout = nn.Dropout(dropout)
        self.window_size = window_size

    def forward(self, x):
        shortcut = x
        x = self.norm1(x)
        # Window partition
        if self.window_size > 0:
            H, W = x.shape[1], x.shape[2]
            x, pad_hw = window_partition(x, self.window_size)

        x = self.ls1(self.attn(x))
        # Reverse window partition
        if self.window_size > 0:
            x = window_unpartition(x, self.window_size, pad_hw, (H, W))

        x = shortcut + self.dropout(self.drop_path(x))
        x = x + self.dropout(self.drop_path(self.ls2(self.mlp(self.norm2(x)))))

        return x


class ViT(nn.Module):
    """
    This module implements Vision Transformer (ViT) backbone in :paper:`vitdet`.
    "Exploring Plain Vision Transformer Backbones for Object Detection",
    https://arxiv.org/abs/2203.16527
    """

    def __init__(
        self,
        img_size=1024,
        patch_size=16,
        in_chans=3,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=True,
        drop_path_rate=0.0,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        act_layer=nn.GELU,
        use_abs_pos=True,
        use_rel_pos=False,
        rel_pos_zero_init=True,
        window_size=14,
        window_block_indexes=(0, 1, 3, 4, 6, 7, 9, 10),
        use_act_checkpoint=False,
        pretrain_img_size=224,
        pretrain_use_cls_token=True,
        dropout=0.0,
        weights_path=None,
        return_interm_layers=False,
        init_values=None,
    ):
        """
        Args:
            img_size (int): Input image size. Only relevant for rel pos.
            patch_size (int): Patch size.
            in_chans (int): Number of input image channels.
            embed_dim (int): Patch embedding dimension.
            depth (int): Depth of ViT.
            num_heads (int): Number of attention heads in each ViT block.
            mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
            qkv_bias (bool): If True, add a learnable bias to query, key, value.
            drop_path_rate (float): Stochastic depth rate.
            norm_layer (nn.Module): Normalization layer.
            act_layer (nn.Module): Activation layer.
            use_abs_pos (bool): If True, use absolute positional embeddings.
            use_rel_pos (bool): If True, add relative positional embeddings to the attention map.
            rel_pos_zero_init (bool): If True, zero initialize relative positional parameters.
            window_size (int): Window size for window attention blocks.
            window_block_indexes (list): Indexes for blocks using window attention.
            residual_block_indexes (list): Indexes for blocks using conv propagation.
            use_act_checkpoint (bool): If True, use activation checkpointing.
            pretrain_img_size (int): input image size for pretraining models.
            pretrain_use_cls_token (bool): If True, pretrainig models use class token.
            dropout (float): Dropout rate. Applied in residual blocks of attn, mlp and inside the mlp.
            path (str or None): Path to the pretrained weights.
            return_interm_layers (bool): Whether to return intermediate layers (all global attention blocks).
            freezing (BackboneFreezingType): Type of freezing.
        """
        super().__init__()
        self.pretrain_use_cls_token = pretrain_use_cls_token

        self.patch_embed = PatchEmbed(
            kernel_size=(patch_size, patch_size),
            stride=(patch_size, patch_size),
            padding=(0, 0),
            in_chans=in_chans,
            embed_dim=embed_dim,
        )

        if use_abs_pos:
            # Initialize absolute positional embedding with pretrain image size.
            num_patches = (pretrain_img_size // patch_size) * (
                pretrain_img_size // patch_size
            )
            num_positions = (num_patches + 1) if pretrain_use_cls_token else num_patches
            self.pos_embed = nn.Parameter(torch.zeros(1, num_positions, embed_dim))
        else:
            self.pos_embed = None

        # stochastic depth decay rule
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]

        self.blocks = nn.ModuleList()
        self.full_attn_ids = []
        cur_stage = 1
        for i in range(depth):
            block = Block(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop_path=dpr[i],
                norm_layer=norm_layer,
                act_layer=act_layer,
                use_rel_pos=use_rel_pos,
                rel_pos_zero_init=rel_pos_zero_init,
                window_size=window_size if i in window_block_indexes else 0,
                input_size=(img_size // patch_size, img_size // patch_size),
                dropout=dropout,
                init_values=init_values,
            )
            if i not in window_block_indexes:
                self.full_attn_ids.append(i)
                cur_stage += 1

            self.blocks.append(block)

        self.return_interm_layers = return_interm_layers
        self.channel_list = (
            [embed_dim] * len(self.full_attn_ids)
            if return_interm_layers
            else [embed_dim]
        )

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:

        x = self.patch_embed(x)
        if self.pos_embed is not None:
            x = x + get_abs_pos(
                self.pos_embed, self.pretrain_use_cls_token, (x.shape[1], x.shape[2])
            )

        outputs = []
        for i, blk in enumerate(self.blocks):
            x = blk(x)
            if (i == self.full_attn_ids[-1]) or (
                self.return_interm_layers and i in self.full_attn_ids
            ):
                feats = x.permute(0, 3, 1, 2)
                outputs.append(feats)

        return outputs
    
    def get_num_layers(self):
        return len(self.blocks)
    
    def get_layer_id(self, layer_name):
        # https://github.com/microsoft/unilm/blob/master/beit/optim_factory.py#L33
        num_layers = self.get_num_layers()

        if layer_name.find("rel_pos") != -1:
            return num_layers + 1
        elif layer_name.find("pos_embed") != -1:
            return 0
        elif layer_name.find("patch_embed") != -1:
            return 0
        elif layer_name.find("blocks") != -1:
            return int(layer_name.split("blocks")[1].split(".")[1]) + 1
        else:
            return num_layers + 1
'''

# Native module: test-code/source/MedSAM2-main/sam2/modeling/efficienttam_base.py
_EMBEDDED_SOURCES['test-code/source/MedSAM2-main/sam2/modeling/efficienttam_base.py'] = '''# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.distributed
import torch.nn.functional as F
from sam2.modeling.efficienttam_utils import (
    get_1d_sine_pe,
    MLP,
    select_closest_cond_frames,
)

from sam2.modeling.sam.mask_decoder import MaskDecoder
from sam2.modeling.sam.prompt_encoder import PromptEncoder
from sam2.modeling.sam.transformer import TwoWayTransformer

from torch.nn.init import trunc_normal_

# a large negative value as a placeholder score for missing objects
NO_OBJ_SCORE = -1024.0


class EfficientTAMBase(torch.nn.Module):
    def __init__(
        self,
        image_encoder,
        memory_attention,
        memory_encoder,
        num_maskmem=7,  # default 1 input frame + 6 previous frames
        image_size=512,
        backbone_stride=16,  # stride of the image backbone output
        sigmoid_scale_for_mem_enc=1.0,  # scale factor for mask sigmoid prob
        sigmoid_bias_for_mem_enc=0.0,  # bias factor for mask sigmoid prob
        # During evaluation, whether to binarize the sigmoid mask logits on interacted frames with clicks
        binarize_mask_from_pts_for_mem_enc=False,
        use_mask_input_as_output_without_sam=False,  # on frames with mask input, whether to directly output the input mask without using a SAM prompt encoder + mask decoder
        # The maximum number of conditioning frames to participate in the memory attention (-1 means no limit; if there are more conditioning frames than this limit,
        # we only cross-attend to the temporally closest `max_cond_frames_in_attn` conditioning frames in the encoder when tracking each frame). This gives the model
        # a temporal locality when handling a large number of annotated frames (since closer frames should be more important) and also avoids GPU OOM.
        max_cond_frames_in_attn=-1,
        # on the first frame, whether to directly add the no-memory embedding to the image feature
        # (instead of using the transformer encoder)
        directly_add_no_mem_embed=False,
        # whether to use high-resolution feature maps in the SAM mask decoder
        use_high_res_features_in_sam=False,
        # whether to output multiple (3) masks for the first click on initial conditioning frames
        multimask_output_in_sam=False,
        # the minimum and maximum number of clicks to use multimask_output_in_sam (only relevant when `multimask_output_in_sam=True`;
        # default is 1 for both, meaning that only the first click gives multimask output; also note that a box counts as two points)
        multimask_min_pt_num=1,
        multimask_max_pt_num=1,
        # whether to also use multimask output for tracking (not just for the first click on initial conditioning frames; only relevant when `multimask_output_in_sam=True`)
        multimask_output_for_tracking=False,
        # Whether to use multimask tokens for obj ptr; Only relevant when both
        # use_obj_ptrs_in_encoder=True and multimask_output_for_tracking=True
        use_multimask_token_for_obj_ptr: bool = False,
        # whether to use sigmoid to restrict ious prediction to [0-1]
        iou_prediction_use_sigmoid=False,
        # The memory bank's temporal stride during evaluation (i.e. the `r` parameter in XMem and Cutie; XMem and Cutie use r=5).
        # For r>1, the (self.num_maskmem - 1) non-conditioning memory frames consist of
        # (self.num_maskmem - 2) nearest frames from every r-th frames, plus the last frame.
        memory_temporal_stride_for_eval=1,
        # whether to apply non-overlapping constraints on the object masks in the memory encoder during evaluation (to avoid/alleviate superposing masks)
        non_overlap_masks_for_mem_enc=False,
        # whether to cross-attend to object pointers from other frames (based on SAM output tokens) in the encoder
        use_obj_ptrs_in_encoder=False,
        # the maximum number of object pointers from other frames in encoder cross attention (only relevant when `use_obj_ptrs_in_encoder=True`)
        max_obj_ptrs_in_encoder=16,
        # whether to add temporal positional encoding to the object pointers in the encoder (only relevant when `use_obj_ptrs_in_encoder=True`)
        add_tpos_enc_to_obj_ptrs=True,
        # whether to add an extra linear projection layer for the temporal positional encoding in the object pointers to avoid potential interference
        # with spatial positional encoding (only relevant when both `use_obj_ptrs_in_encoder=True` and `add_tpos_enc_to_obj_ptrs=True`)
        proj_tpos_enc_in_obj_ptrs=False,
        # whether to use signed distance (instead of unsigned absolute distance) in the temporal positional encoding in the object pointers
        # (only relevant when both `use_obj_ptrs_in_encoder=True` and `add_tpos_enc_to_obj_ptrs=True`)
        use_signed_tpos_enc_to_obj_ptrs=False,
        # whether to only attend to object pointers in the past (before the current frame) in the encoder during evaluation
        # (only relevant when `use_obj_ptrs_in_encoder=True`; this might avoid pointer information too far in the future to distract the initial tracking)
        only_obj_ptrs_in_the_past_for_eval=False,
        # Whether to predict if there is an object in the frame
        pred_obj_scores: bool = False,
        # Whether to use an MLP to predict object scores
        pred_obj_scores_mlp: bool = False,
        # Only relevant if pred_obj_scores=True and use_obj_ptrs_in_encoder=True;
        # Whether to have a fixed no obj pointer when there is no object present
        # or to use it as an additive embedding with obj_ptr produced by decoder
        fixed_no_obj_ptr: bool = False,
        # Soft no object, i.e. mix in no_obj_ptr softly,
        # hope to make recovery easier if there is a mistake and mitigate accumulation of errors
        soft_no_obj_ptr: bool = False,
        use_mlp_for_obj_ptr_proj: bool = False,
        # add no obj embedding to spatial frames
        no_obj_embed_spatial: bool = False,
        # extra arguments used to construct the SAM mask decoder; if not None, it should be a dict of kwargs to be passed into `MaskDecoder` class.
        sam_mask_decoder_extra_args=None,
        compile_image_encoder: bool = False,
    ):
        super().__init__()

        # Part 1: the image backbone
        self.image_encoder = image_encoder
        self.use_high_res_features_in_sam = False
        self.num_feature_levels = 1
        self.use_obj_ptrs_in_encoder = use_obj_ptrs_in_encoder
        self.max_obj_ptrs_in_encoder = max_obj_ptrs_in_encoder
        if use_obj_ptrs_in_encoder:
            # A conv layer to downsample the mask prompt to stride 4 (the same stride as
            # low-res SAM mask logits) and to change its scales from 0~1 to SAM logit scale,
            # so that it can be fed into the SAM mask decoder to generate a pointer.
            self.mask_downsample = torch.nn.Conv2d(1, 1, kernel_size=4, stride=4)
        self.add_tpos_enc_to_obj_ptrs = add_tpos_enc_to_obj_ptrs
        if proj_tpos_enc_in_obj_ptrs:
            assert add_tpos_enc_to_obj_ptrs  # these options need to be used together
        self.proj_tpos_enc_in_obj_ptrs = proj_tpos_enc_in_obj_ptrs
        self.use_signed_tpos_enc_to_obj_ptrs = use_signed_tpos_enc_to_obj_ptrs
        self.only_obj_ptrs_in_the_past_for_eval = only_obj_ptrs_in_the_past_for_eval

        # Part 2: memory attention to condition current frame's visual features
        # with memories (and obj ptrs) from past frames
        self.memory_attention = memory_attention
        self.hidden_dim = image_encoder.neck.d_model

        # Part 3: memory encoder for the previous frame's outputs
        self.memory_encoder = memory_encoder
        self.mem_dim = self.hidden_dim
        if hasattr(self.memory_encoder, "out_proj") and hasattr(
            self.memory_encoder.out_proj, "weight"
        ):
            # if there is compression of memories along channel dim
            self.mem_dim = self.memory_encoder.out_proj.weight.shape[0]
        self.num_maskmem = num_maskmem  # Number of memories accessible
        # Temporal encoding of the memories
        self.maskmem_tpos_enc = torch.nn.Parameter(
            torch.zeros(num_maskmem, 1, 1, self.mem_dim)
        )
        trunc_normal_(self.maskmem_tpos_enc, std=0.02)
        # a single token to indicate no memory embedding from previous frames
        self.no_mem_embed = torch.nn.Parameter(torch.zeros(1, 1, self.hidden_dim))
        self.no_mem_pos_enc = torch.nn.Parameter(torch.zeros(1, 1, self.hidden_dim))
        trunc_normal_(self.no_mem_embed, std=0.02)
        trunc_normal_(self.no_mem_pos_enc, std=0.02)
        self.directly_add_no_mem_embed = directly_add_no_mem_embed
        # Apply sigmoid to the output raw mask logits (to turn them from
        # range (-inf, +inf) to range (0, 1)) before feeding them into the memory encoder
        self.sigmoid_scale_for_mem_enc = sigmoid_scale_for_mem_enc
        self.sigmoid_bias_for_mem_enc = sigmoid_bias_for_mem_enc
        self.binarize_mask_from_pts_for_mem_enc = binarize_mask_from_pts_for_mem_enc
        self.non_overlap_masks_for_mem_enc = non_overlap_masks_for_mem_enc
        self.memory_temporal_stride_for_eval = memory_temporal_stride_for_eval
        # On frames with mask input, whether to directly output the input mask without
        # using a SAM prompt encoder + mask decoder
        self.use_mask_input_as_output_without_sam = use_mask_input_as_output_without_sam
        self.multimask_output_in_sam = multimask_output_in_sam
        self.multimask_min_pt_num = multimask_min_pt_num
        self.multimask_max_pt_num = multimask_max_pt_num
        self.multimask_output_for_tracking = multimask_output_for_tracking
        self.use_multimask_token_for_obj_ptr = use_multimask_token_for_obj_ptr
        self.iou_prediction_use_sigmoid = iou_prediction_use_sigmoid

        # Part 4: SAM-style prompt encoder (for both mask and point inputs)
        # and SAM-style mask decoder for the final mask output
        self.image_size = image_size
        self.backbone_stride = backbone_stride
        self.sam_mask_decoder_extra_args = sam_mask_decoder_extra_args
        self.pred_obj_scores = pred_obj_scores
        self.pred_obj_scores_mlp = pred_obj_scores_mlp
        self.fixed_no_obj_ptr = fixed_no_obj_ptr
        self.soft_no_obj_ptr = soft_no_obj_ptr
        if self.fixed_no_obj_ptr:
            assert self.pred_obj_scores
            assert self.use_obj_ptrs_in_encoder
        if self.pred_obj_scores and self.use_obj_ptrs_in_encoder:
            self.no_obj_ptr = torch.nn.Parameter(torch.zeros(1, self.hidden_dim))
            trunc_normal_(self.no_obj_ptr, std=0.02)
        self.use_mlp_for_obj_ptr_proj = use_mlp_for_obj_ptr_proj
        self.no_obj_embed_spatial = None
        if no_obj_embed_spatial:
            self.no_obj_embed_spatial = torch.nn.Parameter(torch.zeros(1, self.mem_dim))
            trunc_normal_(self.no_obj_embed_spatial, std=0.02)

        self._build_sam_heads()
        self.max_cond_frames_in_attn = max_cond_frames_in_attn

        # Model compilation
        if compile_image_encoder:
            # Compile the forward function (not the full module) to allow loading checkpoints.
            print(
                "Image encoder compilation is enabled. First forward pass will be slow."
            )
            self.image_encoder.forward = torch.compile(
                self.image_encoder.forward,
                mode="max-autotune",
                fullgraph=True,
                dynamic=False,
            )

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(self, *args, **kwargs):
        raise NotImplementedError(
            "Please use the corresponding methods in EfficientTAMVideoPredictor for inference"
        )

    def _build_sam_heads(self):
        """Build SAM-style prompt encoder and mask decoder."""
        self.sam_prompt_embed_dim = self.hidden_dim
        self.sam_image_embedding_size = self.image_size // self.backbone_stride

        # build PromptEncoder and MaskDecoder from SAM
        # (their hyperparameters like `mask_in_chans=16` are from SAM code)
        self.sam_prompt_encoder = PromptEncoder(
            embed_dim=self.sam_prompt_embed_dim,
            image_embedding_size=(
                self.sam_image_embedding_size,
                self.sam_image_embedding_size,
            ),
            input_image_size=(self.image_size, self.image_size),
            mask_in_chans=16,
        )
        self.sam_mask_decoder = MaskDecoder(
            num_multimask_outputs=3,
            transformer=TwoWayTransformer(
                depth=2,
                embedding_dim=self.sam_prompt_embed_dim,
                mlp_dim=2048,
                num_heads=8,
            ),
            transformer_dim=self.sam_prompt_embed_dim,
            iou_head_depth=3,
            iou_head_hidden_dim=256,
            use_high_res_features=self.use_high_res_features_in_sam,
            iou_prediction_use_sigmoid=self.iou_prediction_use_sigmoid,
            pred_obj_scores=self.pred_obj_scores,
            pred_obj_scores_mlp=self.pred_obj_scores_mlp,
            use_multimask_token_for_obj_ptr=self.use_multimask_token_for_obj_ptr,
            **(self.sam_mask_decoder_extra_args or {}),
        )
        if self.use_obj_ptrs_in_encoder:
            # a linear projection on SAM output tokens to turn them into object pointers
            self.obj_ptr_proj = torch.nn.Linear(self.hidden_dim, self.hidden_dim)
            if self.use_mlp_for_obj_ptr_proj:
                self.obj_ptr_proj = MLP(
                    self.hidden_dim, self.hidden_dim, self.hidden_dim, 3
                )
        else:
            self.obj_ptr_proj = torch.nn.Identity()
        if self.proj_tpos_enc_in_obj_ptrs:
            # a linear projection on temporal positional encoding in object pointers to
            # avoid potential interference with spatial positional encoding
            self.obj_ptr_tpos_proj = torch.nn.Linear(self.hidden_dim, self.mem_dim)
        else:
            self.obj_ptr_tpos_proj = torch.nn.Identity()

    def _forward_sam_heads(
        self,
        backbone_features,
        point_inputs=None,
        mask_inputs=None,
        high_res_features=None,
        multimask_output=False,
    ):
        """
        Forward SAM prompt encoders and mask heads.

        Inputs:
        - backbone_features: image features of [B, C, H, W] shape
        - point_inputs: a dictionary with "point_coords" and "point_labels", where
          1) "point_coords" has [B, P, 2] shape and float32 dtype and contains the
             absolute pixel-unit coordinate in (x, y) format of the P input points
          2) "point_labels" has shape [B, P] and int32 dtype, where 1 means
             positive clicks, 0 means negative clicks, and -1 means padding
        - mask_inputs: a mask of [B, 1, H*16, W*16] shape, float or bool, with the
          same spatial size as the image.
        - high_res_features: either 1) None or 2) or a list of length 2 containing
          two feature maps of [B, C, 4*H, 4*W] and [B, C, 2*H, 2*W] shapes respectively,
          which will be used as high-resolution feature maps for SAM decoder.
        - multimask_output: if it's True, we output 3 candidate masks and their 3
          corresponding IoU estimates, and if it's False, we output only 1 mask and
          its corresponding IoU estimate.

        Outputs:
        - low_res_multimasks: [B, M, H*4, W*4] shape (where M = 3 if
          `multimask_output=True` and M = 1 if `multimask_output=False`), the SAM
          output mask logits (before sigmoid) for the low-resolution masks, with 4x
          the resolution (1/4 stride) of the input backbone_features.
        - high_res_multimasks: [B, M, H*16, W*16] shape (where M = 3
          if `multimask_output=True` and M = 1 if `multimask_output=False`),
          upsampled from the low-resolution masks, with shape size as the image
          (stride is 1 pixel).
        - ious, [B, M] shape, where (where M = 3 if `multimask_output=True` and M = 1
          if `multimask_output=False`), the estimated IoU of each output mask.
        - low_res_masks: [B, 1, H*4, W*4] shape, the best mask in `low_res_multimasks`.
          If `multimask_output=True`, it's the mask with the highest IoU estimate.
          If `multimask_output=False`, it's the same as `low_res_multimasks`.
        - high_res_masks: [B, 1, H*16, W*16] shape, the best mask in `high_res_multimasks`.
          If `multimask_output=True`, it's the mask with the highest IoU estimate.
          If `multimask_output=False`, it's the same as `high_res_multimasks`.
        - obj_ptr: [B, C] shape, the object pointer vector for the output mask, extracted
          based on the output token from the SAM mask decoder.
        """
        B = backbone_features.size(0)
        device = backbone_features.device
        assert backbone_features.size(1) == self.sam_prompt_embed_dim
        assert backbone_features.size(2) == self.sam_image_embedding_size
        assert backbone_features.size(3) == self.sam_image_embedding_size

        # a) Handle point prompts
        if point_inputs is not None:
            sam_point_coords = point_inputs["point_coords"]
            sam_point_labels = point_inputs["point_labels"]
            assert sam_point_coords.size(0) == B and sam_point_labels.size(0) == B
        else:
            # If no points are provide, pad with an empty point (with label -1)
            sam_point_coords = torch.zeros(B, 1, 2, device=device)
            sam_point_labels = -torch.ones(B, 1, dtype=torch.int32, device=device)

        # b) Handle mask prompts
        if mask_inputs is not None:
            # If mask_inputs is provided, downsize it into low-res mask input if needed
            # and feed it as a dense mask prompt into the SAM mask encoder
            assert len(mask_inputs.shape) == 4 and mask_inputs.shape[:2] == (B, 1)
            if mask_inputs.shape[-2:] != self.sam_prompt_encoder.mask_input_size:
                sam_mask_prompt = F.interpolate(
                    mask_inputs.float(),
                    size=self.sam_prompt_encoder.mask_input_size,
                    align_corners=False,
                    mode="bilinear",
                    antialias=True,  # use antialias for downsampling
                )
            else:
                sam_mask_prompt = mask_inputs
        else:
            # Otherwise, simply feed None (and SAM's prompt encoder will add
            # a learned `no_mask_embed` to indicate no mask input in this case).
            sam_mask_prompt = None

        sparse_embeddings, dense_embeddings = self.sam_prompt_encoder(
            points=(sam_point_coords, sam_point_labels),
            boxes=None,
            masks=sam_mask_prompt,
        )
        (
            low_res_multimasks,
            ious,
            sam_output_tokens,
            object_score_logits,
        ) = self.sam_mask_decoder(
            image_embeddings=backbone_features,
            image_pe=self.sam_prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=multimask_output,
            repeat_image=False,  # the image is already batched
            high_res_features=high_res_features,
        )
        if self.pred_obj_scores:
            is_obj_appearing = object_score_logits > 0

            # Mask used for spatial memories is always a *hard* choice between obj and no obj,
            # consistent with the actual mask prediction
            low_res_multimasks = torch.where(
                is_obj_appearing[:, None, None],
                low_res_multimasks,
                NO_OBJ_SCORE,
            )

        # convert masks from possibly bfloat16 (or float16) to float32
        # (older PyTorch versions before 2.1 don't support `interpolate` on bf16)
        low_res_multimasks = low_res_multimasks.float()
        high_res_multimasks = F.interpolate(
            low_res_multimasks,
            size=(self.image_size, self.image_size),
            mode="bilinear",
            align_corners=False,
        )

        sam_output_token = sam_output_tokens[:, 0]
        if multimask_output:
            # take the best mask prediction (with the highest IoU estimation)
            best_iou_inds = torch.argmax(ious, dim=-1)
            batch_inds = torch.arange(B, device=device)
            low_res_masks = low_res_multimasks[batch_inds, best_iou_inds].unsqueeze(1)
            high_res_masks = high_res_multimasks[batch_inds, best_iou_inds].unsqueeze(1)
            if sam_output_tokens.size(1) > 1:
                sam_output_token = sam_output_tokens[batch_inds, best_iou_inds]
        else:
            low_res_masks, high_res_masks = low_res_multimasks, high_res_multimasks

        # Extract object pointer from the SAM output token (with occlusion handling)
        obj_ptr = self.obj_ptr_proj(sam_output_token)
        if self.pred_obj_scores:
            # Allow *soft* no obj ptr, unlike for masks
            if self.soft_no_obj_ptr:
                lambda_is_obj_appearing = object_score_logits.sigmoid()
            else:
                lambda_is_obj_appearing = is_obj_appearing.float()

            if self.fixed_no_obj_ptr:
                obj_ptr = lambda_is_obj_appearing * obj_ptr
            obj_ptr = obj_ptr + (1 - lambda_is_obj_appearing) * self.no_obj_ptr

        return (
            low_res_multimasks,
            high_res_multimasks,
            ious,
            low_res_masks,
            high_res_masks,
            obj_ptr,
            object_score_logits,
        )

    def _use_mask_as_output(self, backbone_features, high_res_features, mask_inputs):
        """
        Directly turn binary `mask_inputs` into a output mask logits without using SAM.
        (same input and output shapes as in _forward_sam_heads above).
        """
        # Use -10/+10 as logits for neg/pos pixels (very close to 0/1 in prob after sigmoid).
        out_scale, out_bias = 20.0, -10.0  # sigmoid(-10.0)=4.5398e-05
        mask_inputs_float = mask_inputs.float()
        high_res_masks = mask_inputs_float * out_scale + out_bias
        low_res_masks = F.interpolate(
            high_res_masks,
            size=(high_res_masks.size(-2) // 4, high_res_masks.size(-1) // 4),
            align_corners=False,
            mode="bilinear",
            antialias=True,  # use antialias for downsampling
        )
        # a dummy IoU prediction of all 1's under mask input
        ious = mask_inputs.new_ones(mask_inputs.size(0), 1).float()
        if not self.use_obj_ptrs_in_encoder:
            # all zeros as a dummy object pointer (of shape [B, C])
            obj_ptr = torch.zeros(
                mask_inputs.size(0), self.hidden_dim, device=mask_inputs.device
            )
        else:
            # produce an object pointer using the SAM decoder from the mask input
            _, _, _, _, _, obj_ptr, _ = self._forward_sam_heads(
                backbone_features=backbone_features,
                mask_inputs=self.mask_downsample(mask_inputs_float),
                high_res_features=high_res_features,
            )
        # In this method, we are treating mask_input as output, e.g. using it directly to create spatial mem;
        # Below, we follow the same design axiom to use mask_input to decide if obj appears or not instead of relying
        # on the object_scores from the SAM decoder.
        is_obj_appearing = torch.any(mask_inputs.flatten(1).float() > 0.0, dim=1)
        is_obj_appearing = is_obj_appearing[..., None]
        lambda_is_obj_appearing = is_obj_appearing.float()
        object_score_logits = out_scale * lambda_is_obj_appearing + out_bias
        if self.pred_obj_scores:
            if self.fixed_no_obj_ptr:
                obj_ptr = lambda_is_obj_appearing * obj_ptr
            obj_ptr = obj_ptr + (1 - lambda_is_obj_appearing) * self.no_obj_ptr

        return (
            low_res_masks,
            high_res_masks,
            ious,
            low_res_masks,
            high_res_masks,
            obj_ptr,
            object_score_logits,
        )

    def forward_image(self, img_batch: torch.Tensor):
        """Get the image feature on the input batch."""
        backbone_out = self.image_encoder(img_batch)
        if self.use_high_res_features_in_sam:
            # precompute projected level 0 and level 1 features in SAM decoder
            # to avoid running it again on every SAM click
            backbone_out["backbone_fpn"][0] = self.sam_mask_decoder.conv_s0(
                backbone_out["backbone_fpn"][0]
            )
            backbone_out["backbone_fpn"][1] = self.sam_mask_decoder.conv_s1(
                backbone_out["backbone_fpn"][1]
            )
        return backbone_out

    def _prepare_backbone_features(self, backbone_out):
        """Prepare and flatten visual features."""
        backbone_out = backbone_out.copy()
        assert len(backbone_out["backbone_fpn"]) == len(backbone_out["vision_pos_enc"])
        assert len(backbone_out["backbone_fpn"]) >= self.num_feature_levels

        feature_maps = backbone_out["backbone_fpn"][-self.num_feature_levels :]
        vision_pos_embeds = backbone_out["vision_pos_enc"][-self.num_feature_levels :]

        feat_sizes = [(x.shape[-2], x.shape[-1]) for x in vision_pos_embeds]
        # flatten NxCxHxW to HWxNxC
        vision_feats = [x.flatten(2).permute(2, 0, 1) for x in feature_maps]
        vision_pos_embeds = [x.flatten(2).permute(2, 0, 1) for x in vision_pos_embeds]

        return backbone_out, vision_feats, vision_pos_embeds, feat_sizes

    def _prepare_memory_conditioned_features(
        self,
        frame_idx,
        is_init_cond_frame,
        current_vision_feats,
        current_vision_pos_embeds,
        feat_sizes,
        output_dict,
        num_frames,
        track_in_reverse=False,  # tracking in reverse time order (for demo usage)
    ):
        """Fuse the current frame's visual feature map with previous memory."""
        B = current_vision_feats[-1].size(1)  # batch size on this frame
        C = self.hidden_dim
        H, W = feat_sizes[-1]  # top-level (lowest-resolution) feature size
        device = current_vision_feats[-1].device
        # The case of `self.num_maskmem == 0` below is primarily used for reproducing SAM on images.
        # In this case, we skip the fusion with any memory.
        if self.num_maskmem == 0:  # Disable memory and skip fusion
            pix_feat = current_vision_feats[-1].permute(1, 2, 0).view(B, C, H, W)
            return pix_feat

        num_obj_ptr_tokens = 0
        tpos_sign_mul = -1 if track_in_reverse else 1
        # Step 1: condition the visual features of the current frame on previous memories
        if not is_init_cond_frame:
            # Retrieve the memories encoded with the maskmem backbone
            to_cat_memory, to_cat_memory_pos_embed = [], []
            # Add conditioning frames's output first (all cond frames have t_pos=0 for
            # when getting temporal positional embedding below)
            assert len(output_dict["cond_frame_outputs"]) > 0
            # Select a maximum number of temporally closest cond frames for cross attention
            cond_outputs = output_dict["cond_frame_outputs"]
            selected_cond_outputs, unselected_cond_outputs = select_closest_cond_frames(
                frame_idx, cond_outputs, self.max_cond_frames_in_attn
            )
            t_pos_and_prevs = [(0, out) for out in selected_cond_outputs.values()]
            # Add last (self.num_maskmem - 1) frames before current frame for non-conditioning memory
            # the earliest one has t_pos=1 and the latest one has t_pos=self.num_maskmem-1
            # We also allow taking the memory frame non-consecutively (with stride>1), in which case
            # we take (self.num_maskmem - 2) frames among every stride-th frames plus the last frame.
            stride = 1 if self.training else self.memory_temporal_stride_for_eval
            for t_pos in range(1, self.num_maskmem):
                t_rel = self.num_maskmem - t_pos  # how many frames before current frame
                if t_rel == 1:
                    # for t_rel == 1, we take the last frame (regardless of r)
                    if not track_in_reverse:
                        # the frame immediately before this frame (i.e. frame_idx - 1)
                        prev_frame_idx = frame_idx - t_rel
                    else:
                        # the frame immediately after this frame (i.e. frame_idx + 1)
                        prev_frame_idx = frame_idx + t_rel
                else:
                    # for t_rel >= 2, we take the memory frame from every r-th frames
                    if not track_in_reverse:
                        # first find the nearest frame among every r-th frames before this frame
                        # for r=1, this would be (frame_idx - 2)
                        prev_frame_idx = ((frame_idx - 2) // stride) * stride
                        # then seek further among every r-th frames
                        prev_frame_idx = prev_frame_idx - (t_rel - 2) * stride
                    else:
                        # first find the nearest frame among every r-th frames after this frame
                        # for r=1, this would be (frame_idx + 2)
                        prev_frame_idx = -(-(frame_idx + 2) // stride) * stride
                        # then seek further among every r-th frames
                        prev_frame_idx = prev_frame_idx + (t_rel - 2) * stride
                out = output_dict["non_cond_frame_outputs"].get(prev_frame_idx, None)
                if out is None:
                    # If an unselected conditioning frame is among the last (self.num_maskmem - 1)
                    # frames, we still attend to it as if it's a non-conditioning frame.
                    out = unselected_cond_outputs.get(prev_frame_idx, None)
                t_pos_and_prevs.append((t_pos, out))

            for t_pos, prev in t_pos_and_prevs:
                if prev is None:
                    continue  # skip padding frames
                # "maskmem_features" might have been offloaded to CPU in demo use cases,
                # so we load it back to GPU (it's a no-op if it's already on GPU).
                feats = prev["maskmem_features"].to(device, non_blocking=True)
                to_cat_memory.append(feats.flatten(2).permute(2, 0, 1))
                # Spatial positional encoding (it might have been offloaded to CPU in eval)
                maskmem_enc = prev["maskmem_pos_enc"][-1].to(device)
                maskmem_enc = maskmem_enc.flatten(2).permute(2, 0, 1)
                # Temporal positional encoding
                maskmem_enc = (
                    maskmem_enc + self.maskmem_tpos_enc[self.num_maskmem - t_pos - 1]
                )
                to_cat_memory_pos_embed.append(maskmem_enc)

            # Construct the list of past object pointers
            if self.use_obj_ptrs_in_encoder:
                max_obj_ptrs_in_encoder = min(num_frames, self.max_obj_ptrs_in_encoder)
                # First add those object pointers from selected conditioning frames
                # (optionally, only include object pointers in the past during evaluation)
                if not self.training and self.only_obj_ptrs_in_the_past_for_eval:
                    ptr_cond_outputs = {
                        t: out
                        for t, out in selected_cond_outputs.items()
                        if (t >= frame_idx if track_in_reverse else t <= frame_idx)
                    }
                else:
                    ptr_cond_outputs = selected_cond_outputs
                pos_and_ptrs = [
                    # Temporal pos encoding contains how far away each pointer is from current frame
                    (
                        (
                            (frame_idx - t) * tpos_sign_mul
                            if self.use_signed_tpos_enc_to_obj_ptrs
                            else abs(frame_idx - t)
                        ),
                        out["obj_ptr"],
                    )
                    for t, out in ptr_cond_outputs.items()
                ]
                # Add up to (max_obj_ptrs_in_encoder - 1) non-conditioning frames before current frame
                for t_diff in range(1, max_obj_ptrs_in_encoder):
                    t = frame_idx + t_diff if track_in_reverse else frame_idx - t_diff
                    if t < 0 or (num_frames is not None and t >= num_frames):
                        break
                    out = output_dict["non_cond_frame_outputs"].get(
                        t, unselected_cond_outputs.get(t, None)
                    )
                    if out is not None:
                        pos_and_ptrs.append((t_diff, out["obj_ptr"]))
                # If we have at least one object pointer, add them to the across attention
                if len(pos_and_ptrs) > 0:
                    pos_list, ptrs_list = zip(*pos_and_ptrs)
                    # stack object pointers along dim=0 into [ptr_seq_len, B, C] shape
                    obj_ptrs = torch.stack(ptrs_list, dim=0)
                    # a temporal positional embedding based on how far each object pointer is from
                    # the current frame (sine embedding normalized by the max pointer num).
                    if self.add_tpos_enc_to_obj_ptrs:
                        t_diff_max = max_obj_ptrs_in_encoder - 1
                        tpos_dim = C if self.proj_tpos_enc_in_obj_ptrs else self.mem_dim
                        obj_pos = torch.tensor(pos_list).to(
                            device=device, non_blocking=True
                        )
                        obj_pos = get_1d_sine_pe(obj_pos / t_diff_max, dim=tpos_dim)
                        obj_pos = self.obj_ptr_tpos_proj(obj_pos)
                        obj_pos = obj_pos.unsqueeze(1).expand(-1, B, self.mem_dim)
                    else:
                        obj_pos = obj_ptrs.new_zeros(len(pos_list), B, self.mem_dim)
                    if self.mem_dim < C:
                        # split a pointer into (C // self.mem_dim) tokens for self.mem_dim < C
                        obj_ptrs = obj_ptrs.reshape(
                            -1, B, C // self.mem_dim, self.mem_dim
                        )
                        obj_ptrs = obj_ptrs.permute(0, 2, 1, 3).flatten(0, 1)
                        obj_pos = obj_pos.repeat_interleave(C // self.mem_dim, dim=0)
                    to_cat_memory.append(obj_ptrs)
                    to_cat_memory_pos_embed.append(obj_pos)
                    num_obj_ptr_tokens = obj_ptrs.shape[0]
                else:
                    num_obj_ptr_tokens = 0
        else:
            # for initial conditioning frames, encode them without using any previous memory
            if self.directly_add_no_mem_embed:
                # directly add no-mem embedding (instead of using the transformer encoder)
                pix_feat_with_mem = current_vision_feats[-1] + self.no_mem_embed
                pix_feat_with_mem = pix_feat_with_mem.permute(1, 2, 0).view(B, C, H, W)
                return pix_feat_with_mem

            # Use a dummy token on the first frame (to avoid empty memory input to tranformer encoder)
            to_cat_memory = [self.no_mem_embed.expand(1, B, self.mem_dim)]
            to_cat_memory_pos_embed = [self.no_mem_pos_enc.expand(1, B, self.mem_dim)]

        # Step 2: Concatenate the memories and forward through the transformer encoder
        memory = torch.cat(to_cat_memory, dim=0)
        memory_pos_embed = torch.cat(to_cat_memory_pos_embed, dim=0)

        pix_feat_with_mem = self.memory_attention(
            curr=current_vision_feats,
            curr_pos=current_vision_pos_embeds,
            memory=memory,
            memory_pos=memory_pos_embed,
            num_obj_ptr_tokens=num_obj_ptr_tokens,
        )
        # reshape the output (HW)BC => BCHW
        pix_feat_with_mem = pix_feat_with_mem.permute(1, 2, 0).view(B, C, H, W)
        return pix_feat_with_mem

    def _encode_new_memory(
        self,
        current_vision_feats,
        feat_sizes,
        pred_masks_high_res,
        object_score_logits,
        is_mask_from_pts,
    ):
        """Encode the current image and its prediction into a memory feature."""
        B = current_vision_feats[-1].size(1)  # batch size on this frame
        C = self.hidden_dim
        H, W = feat_sizes[-1]  # top-level (lowest-resolution) feature size
        # top-level feature, (HW)BC => BCHW
        pix_feat = current_vision_feats[-1].permute(1, 2, 0).view(B, C, H, W)
        if self.non_overlap_masks_for_mem_enc and not self.training:
            # optionally, apply non-overlapping constraints to the masks (it's applied
            # in the batch dimension and should only be used during eval, where all
            # the objects come from the same video under batch size 1).
            pred_masks_high_res = self._apply_non_overlapping_constraints(
                pred_masks_high_res
            )
        # scale the raw mask logits with a temperature before applying sigmoid
        binarize = self.binarize_mask_from_pts_for_mem_enc and is_mask_from_pts
        if binarize and not self.training:
            mask_for_mem = (pred_masks_high_res > 0).float()
        else:
            # apply sigmoid on the raw mask logits to turn them into range (0, 1)
            mask_for_mem = torch.sigmoid(pred_masks_high_res)
        # apply scale and bias terms to the sigmoid probabilities
        if self.sigmoid_scale_for_mem_enc != 1.0:
            mask_for_mem = mask_for_mem * self.sigmoid_scale_for_mem_enc
        if self.sigmoid_bias_for_mem_enc != 0.0:
            mask_for_mem = mask_for_mem + self.sigmoid_bias_for_mem_enc
        maskmem_out = self.memory_encoder(
            pix_feat, mask_for_mem, skip_mask_sigmoid=True  # sigmoid already applied
        )
        maskmem_features = maskmem_out["vision_features"]
        maskmem_pos_enc = maskmem_out["vision_pos_enc"]
        # add a no-object embedding to the spatial memory to indicate that the frame
        # is predicted to be occluded (i.e. no object is appearing in the frame)
        if self.no_obj_embed_spatial is not None:
            is_obj_appearing = (object_score_logits > 0).float()
            maskmem_features += (
                1 - is_obj_appearing[..., None, None]
            ) * self.no_obj_embed_spatial[..., None, None].expand(
                *maskmem_features.shape
            )

        return maskmem_features, maskmem_pos_enc

    def _track_step(
        self,
        frame_idx,
        is_init_cond_frame,
        current_vision_feats,
        current_vision_pos_embeds,
        feat_sizes,
        point_inputs,
        mask_inputs,
        output_dict,
        num_frames,
        track_in_reverse,
        prev_sam_mask_logits,
    ):
        current_out = {"point_inputs": point_inputs, "mask_inputs": mask_inputs}
        # High-resolution feature maps for the SAM head, reshape (HW)BC => BCHW
        if len(current_vision_feats) > 1:
            high_res_features = [
                x.permute(1, 2, 0).view(x.size(1), x.size(2), *s)
                for x, s in zip(current_vision_feats[:-1], feat_sizes[:-1])
            ]
        else:
            high_res_features = None
        if mask_inputs is not None and self.use_mask_input_as_output_without_sam:
            # When use_mask_input_as_output_without_sam=True, we directly output the mask input
            # (see it as a GT mask) without using a SAM prompt encoder + mask decoder.
            pix_feat = current_vision_feats[-1].permute(1, 2, 0)
            pix_feat = pix_feat.view(-1, self.hidden_dim, *feat_sizes[-1])
            sam_outputs = self._use_mask_as_output(
                pix_feat, high_res_features, mask_inputs
            )
        else:
            # fused the visual feature with previous memory features in the memory bank
            pix_feat = self._prepare_memory_conditioned_features(
                frame_idx=frame_idx,
                is_init_cond_frame=is_init_cond_frame,
                current_vision_feats=current_vision_feats[-1:],
                current_vision_pos_embeds=current_vision_pos_embeds[-1:],
                feat_sizes=feat_sizes[-1:],
                output_dict=output_dict,
                num_frames=num_frames,
                track_in_reverse=track_in_reverse,
            )
            # apply SAM-style segmentation head
            # here we might feed previously predicted low-res SAM mask logits into the SAM mask decoder,
            # e.g. in demo where such logits come from earlier interaction instead of correction sampling
            # (in this case, any `mask_inputs` shouldn't reach here as they are sent to _use_mask_as_output instead)
            if prev_sam_mask_logits is not None:
                assert point_inputs is not None and mask_inputs is None
                mask_inputs = prev_sam_mask_logits
            multimask_output = self._use_multimask(is_init_cond_frame, point_inputs)
            sam_outputs = self._forward_sam_heads(
                backbone_features=pix_feat,
                point_inputs=point_inputs,
                mask_inputs=mask_inputs,
                high_res_features=high_res_features,
                multimask_output=multimask_output,
            )

        return current_out, sam_outputs, high_res_features, pix_feat

    def _encode_memory_in_output(
        self,
        current_vision_feats,
        feat_sizes,
        point_inputs,
        run_mem_encoder,
        high_res_masks,
        object_score_logits,
        current_out,
    ):
        if run_mem_encoder and self.num_maskmem > 0:
            high_res_masks_for_mem_enc = high_res_masks
            maskmem_features, maskmem_pos_enc = self._encode_new_memory(
                current_vision_feats=current_vision_feats,
                feat_sizes=feat_sizes,
                pred_masks_high_res=high_res_masks_for_mem_enc,
                object_score_logits=object_score_logits,
                is_mask_from_pts=(point_inputs is not None),
            )
            current_out["maskmem_features"] = maskmem_features
            current_out["maskmem_pos_enc"] = maskmem_pos_enc
        else:
            current_out["maskmem_features"] = None
            current_out["maskmem_pos_enc"] = None

    def track_step(
        self,
        frame_idx,
        is_init_cond_frame,
        current_vision_feats,
        current_vision_pos_embeds,
        feat_sizes,
        point_inputs,
        mask_inputs,
        output_dict,
        num_frames,
        track_in_reverse=False,  # tracking in reverse time order (for demo usage)
        # Whether to run the memory encoder on the predicted masks. Sometimes we might want
        # to skip the memory encoder with `run_mem_encoder=False`. For example,
        # in demo we might call `track_step` multiple times for each user click,
        # and only encode the memory when the user finalizes their clicks. And in ablation
        # settings like SAM training on static images, we don't need the memory encoder.
        run_mem_encoder=True,
        # The previously predicted SAM mask logits (which can be fed together with new clicks in demo).
        prev_sam_mask_logits=None,
    ):
        current_out, sam_outputs, _, _ = self._track_step(
            frame_idx,
            is_init_cond_frame,
            current_vision_feats,
            current_vision_pos_embeds,
            feat_sizes,
            point_inputs,
            mask_inputs,
            output_dict,
            num_frames,
            track_in_reverse,
            prev_sam_mask_logits,
        )

        (
            _,
            _,
            _,
            low_res_masks,
            high_res_masks,
            obj_ptr,
            object_score_logits,
        ) = sam_outputs

        current_out["pred_masks"] = low_res_masks
        current_out["pred_masks_high_res"] = high_res_masks
        current_out["obj_ptr"] = obj_ptr
        if not self.training:
            # Only add this in inference (to avoid unused param in activation checkpointing;
            # it's mainly used in the demo to encode spatial memories w/ consolidated masks)
            current_out["object_score_logits"] = object_score_logits

        # Finally run the memory encoder on the predicted mask to encode
        # it into a new memory feature (that can be used in future frames)
        self._encode_memory_in_output(
            current_vision_feats,
            feat_sizes,
            point_inputs,
            run_mem_encoder,
            high_res_masks,
            object_score_logits,
            current_out,
        )

        return current_out

    def _use_multimask(self, is_init_cond_frame, point_inputs):
        """Whether to use multimask output in the SAM head."""
        num_pts = 0 if point_inputs is None else point_inputs["point_labels"].size(1)
        multimask_output = (
            self.multimask_output_in_sam
            and (is_init_cond_frame or self.multimask_output_for_tracking)
            and (self.multimask_min_pt_num <= num_pts <= self.multimask_max_pt_num)
        )
        return multimask_output

    def _apply_non_overlapping_constraints(self, pred_masks):
        """
        Apply non-overlapping constraints to the object scores in pred_masks. Here we
        keep only the highest scoring object at each spatial location in pred_masks.
        """
        batch_size = pred_masks.size(0)
        if batch_size == 1:
            return pred_masks

        device = pred_masks.device
        # "max_obj_inds": object index of the object with the highest score at each location
        max_obj_inds = torch.argmax(pred_masks, dim=0, keepdim=True)
        # "batch_obj_inds": object index of each object slice (along dim 0) in `pred_masks`
        batch_obj_inds = torch.arange(batch_size, device=device)[:, None, None, None]
        keep = max_obj_inds == batch_obj_inds
        # suppress overlapping regions' scores below -10.0 so that the foreground regions
        # don't overlap (here sigmoid(-10.0)=4.5398e-05)
        pred_masks = torch.where(keep, pred_masks, torch.clamp(pred_masks, max=-10.0))
        return pred_masks
'''

# Native module: test-code/source/MedSAM2-main/sam2/modeling/efficienttam_utils.py
_EMBEDDED_SOURCES['test-code/source/MedSAM2-main/sam2/modeling/efficienttam_utils.py'] = '''# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import copy
from typing import Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from sam2.utils.misc import mask_to_box


def select_closest_cond_frames(frame_idx, cond_frame_outputs, max_cond_frame_num):
    """
    Select up to `max_cond_frame_num` conditioning frames from `cond_frame_outputs`
    that are temporally closest to the current frame at `frame_idx`. Here, we take
    - a) the closest conditioning frame before `frame_idx` (if any);
    - b) the closest conditioning frame after `frame_idx` (if any);
    - c) any other temporally closest conditioning frames until reaching a total
         of `max_cond_frame_num` conditioning frames.

    Outputs:
    - selected_outputs: selected items (keys & values) from `cond_frame_outputs`.
    - unselected_outputs: items (keys & values) not selected in `cond_frame_outputs`.
    """
    if max_cond_frame_num == -1 or len(cond_frame_outputs) <= max_cond_frame_num:
        selected_outputs = cond_frame_outputs
        unselected_outputs = {}
    else:
        assert max_cond_frame_num >= 2, "we should allow using 2+ conditioning frames"
        selected_outputs = {}

        # the closest conditioning frame before `frame_idx` (if any)
        idx_before = max((t for t in cond_frame_outputs if t < frame_idx), default=None)
        if idx_before is not None:
            selected_outputs[idx_before] = cond_frame_outputs[idx_before]

        # the closest conditioning frame after `frame_idx` (if any)
        idx_after = min((t for t in cond_frame_outputs if t >= frame_idx), default=None)
        if idx_after is not None:
            selected_outputs[idx_after] = cond_frame_outputs[idx_after]

        # add other temporally closest conditioning frames until reaching a total
        # of `max_cond_frame_num` conditioning frames.
        num_remain = max_cond_frame_num - len(selected_outputs)
        inds_remain = sorted(
            (t for t in cond_frame_outputs if t not in selected_outputs),
            key=lambda x: abs(x - frame_idx),
        )[:num_remain]
        selected_outputs.update((t, cond_frame_outputs[t]) for t in inds_remain)
        unselected_outputs = {
            t: v for t, v in cond_frame_outputs.items() if t not in selected_outputs
        }

    return selected_outputs, unselected_outputs


def get_1d_sine_pe(pos_inds, dim, temperature=10000):
    """
    Get 1D sine positional embedding as in the original Transformer paper.
    """
    pe_dim = dim // 2
    dim_t = torch.arange(pe_dim, dtype=torch.float32, device=pos_inds.device)
    dim_t = temperature ** (2 * (dim_t // 2) / pe_dim)

    pos_embed = pos_inds.unsqueeze(-1) / dim_t
    pos_embed = torch.cat([pos_embed.sin(), pos_embed.cos()], dim=-1)
    return pos_embed


def get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(f"activation should be relu/gelu, not {activation}.")


def get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


class DropPath(nn.Module):
    # adapted from https://github.com/huggingface/pytorch-image-models/blob/main/timm/layers/drop.py
    def __init__(self, drop_prob=0.0, scale_by_keep=True):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x):
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
        if keep_prob > 0.0 and self.scale_by_keep:
            random_tensor.div_(keep_prob)
        return x * random_tensor


# Lightly adapted from
# https://github.com/facebookresearch/MaskFormer/blob/main/mask_former/modeling/transformer/transformer_predictor.py # noqa
class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int,
        activation: nn.Module = nn.ReLU,
        sigmoid_output: bool = False,
    ) -> None:
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )
        self.sigmoid_output = sigmoid_output
        self.act = activation()

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = self.act(layer(x)) if i < self.num_layers - 1 else layer(x)
        if self.sigmoid_output:
            x = F.sigmoid(x)
        return x


# From https://github.com/facebookresearch/detectron2/blob/main/detectron2/layers/batch_norm.py # noqa
# Itself from https://github.com/facebookresearch/ConvNeXt/blob/d1fa8f6fef0a165b27399986cc2bdacc92777e40/models/convnext.py#L119  # noqa
class LayerNorm2d(nn.Module):
    def __init__(self, num_channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x


def sample_box_points(
    masks: torch.Tensor,
    noise: float = 0.1,  # SAM default
    noise_bound: int = 10,  # SAM default
    top_left_label: int = 2,
    bottom_right_label: int = 3,
) -> Tuple[np.array, np.array]:
    """
    Sample a noised version of the top left and bottom right corners of a given `bbox`

    Inputs:
    - masks: [B, 1, H,W] boxes, dtype=torch.Tensor
    - noise: noise as a fraction of box width and height, dtype=float
    - noise_bound: maximum amount of noise (in pure pixesl), dtype=int

    Returns:
    - box_coords: [B, num_pt, 2], contains (x, y) coordinates of top left and bottom right box corners, dtype=torch.float
    - box_labels: [B, num_pt], label 2 is reserverd for top left and 3 for bottom right corners, dtype=torch.int32
    """
    device = masks.device
    box_coords = mask_to_box(masks)
    B, _, H, W = masks.shape
    box_labels = torch.tensor(
        [top_left_label, bottom_right_label], dtype=torch.int, device=device
    ).repeat(B)
    if noise > 0.0:
        if not isinstance(noise_bound, torch.Tensor):
            noise_bound = torch.tensor(noise_bound, device=device)
        bbox_w = box_coords[..., 2] - box_coords[..., 0]
        bbox_h = box_coords[..., 3] - box_coords[..., 1]
        max_dx = torch.min(bbox_w * noise, noise_bound)
        max_dy = torch.min(bbox_h * noise, noise_bound)
        box_noise = 2 * torch.rand(B, 1, 4, device=device) - 1
        box_noise = box_noise * torch.stack((max_dx, max_dy, max_dx, max_dy), dim=-1)

        box_coords = box_coords + box_noise
        img_bounds = (
            torch.tensor([W, H, W, H], device=device) - 1
        )  # uncentered pixel coords
        box_coords.clamp_(torch.zeros_like(img_bounds), img_bounds)  # In place clamping

    box_coords = box_coords.reshape(-1, 2, 2)  # always 2 points
    box_labels = box_labels.reshape(-1, 2)
    return box_coords, box_labels


def sample_random_points_from_errors(gt_masks, pred_masks, num_pt=1):
    """
    Sample `num_pt` random points (along with their labels) independently from the error regions.

    Inputs:
    - gt_masks: [B, 1, H_im, W_im] masks, dtype=torch.bool
    - pred_masks: [B, 1, H_im, W_im] masks, dtype=torch.bool or None
    - num_pt: int, number of points to sample independently for each of the B error maps

    Outputs:
    - points: [B, num_pt, 2], dtype=torch.float, contains (x, y) coordinates of each sampled point
    - labels: [B, num_pt], dtype=torch.int32, where 1 means positive clicks and 0 means
      negative clicks
    """
    if pred_masks is None:  # if pred_masks is not provided, treat it as empty
        pred_masks = torch.zeros_like(gt_masks)
    assert gt_masks.dtype == torch.bool and gt_masks.size(1) == 1
    assert pred_masks.dtype == torch.bool and pred_masks.shape == gt_masks.shape
    assert num_pt >= 0

    B, _, H_im, W_im = gt_masks.shape
    device = gt_masks.device

    # false positive region, a new point sampled in this region should have
    # negative label to correct the FP error
    fp_masks = ~gt_masks & pred_masks
    # false negative region, a new point sampled in this region should have
    # positive label to correct the FN error
    fn_masks = gt_masks & ~pred_masks
    # whether the prediction completely match the ground-truth on each mask
    all_correct = torch.all((gt_masks == pred_masks).flatten(2), dim=2)
    all_correct = all_correct[..., None, None]

    # channel 0 is FP map, while channel 1 is FN map
    pts_noise = torch.rand(B, num_pt, H_im, W_im, 2, device=device)
    # sample a negative new click from FP region or a positive new click
    # from FN region, depend on where the maximum falls,
    # and in case the predictions are all correct (no FP or FN), we just
    # sample a negative click from the background region
    pts_noise[..., 0] *= fp_masks | (all_correct & ~gt_masks)
    pts_noise[..., 1] *= fn_masks
    pts_idx = pts_noise.flatten(2).argmax(dim=2)
    labels = (pts_idx % 2).to(torch.int32)
    pts_idx = pts_idx // 2
    pts_x = pts_idx % W_im
    pts_y = pts_idx // W_im
    points = torch.stack([pts_x, pts_y], dim=2).to(torch.float)
    return points, labels


def sample_one_point_from_error_center(gt_masks, pred_masks, padding=True):
    """
    Sample 1 random point (along with its label) from the center of each error region,
    that is, the point with the largest distance to the boundary of each error region.
    This is the RITM sampling method from https://github.com/saic-vul/ritm_interactive_segmentation/blob/master/isegm/inference/clicker.py

    Inputs:
    - gt_masks: [B, 1, H_im, W_im] masks, dtype=torch.bool
    - pred_masks: [B, 1, H_im, W_im] masks, dtype=torch.bool or None
    - padding: if True, pad with boundary of 1 px for distance transform

    Outputs:
    - points: [B, 1, 2], dtype=torch.float, contains (x, y) coordinates of each sampled point
    - labels: [B, 1], dtype=torch.int32, where 1 means positive clicks and 0 means negative clicks
    """
    import cv2

    if pred_masks is None:
        pred_masks = torch.zeros_like(gt_masks)
    assert gt_masks.dtype == torch.bool and gt_masks.size(1) == 1
    assert pred_masks.dtype == torch.bool and pred_masks.shape == gt_masks.shape

    B, _, _, W_im = gt_masks.shape
    device = gt_masks.device

    # false positive region, a new point sampled in this region should have
    # negative label to correct the FP error
    fp_masks = ~gt_masks & pred_masks
    # false negative region, a new point sampled in this region should have
    # positive label to correct the FN error
    fn_masks = gt_masks & ~pred_masks

    fp_masks = fp_masks.cpu().numpy()
    fn_masks = fn_masks.cpu().numpy()
    points = torch.zeros(B, 1, 2, dtype=torch.float)
    labels = torch.ones(B, 1, dtype=torch.int32)
    for b in range(B):
        fn_mask = fn_masks[b, 0]
        fp_mask = fp_masks[b, 0]
        if padding:
            fn_mask = np.pad(fn_mask, ((1, 1), (1, 1)), "constant")
            fp_mask = np.pad(fp_mask, ((1, 1), (1, 1)), "constant")
        # compute the distance of each point in FN/FP region to its boundary
        fn_mask_dt = cv2.distanceTransform(fn_mask.astype(np.uint8), cv2.DIST_L2, 0)
        fp_mask_dt = cv2.distanceTransform(fp_mask.astype(np.uint8), cv2.DIST_L2, 0)
        if padding:
            fn_mask_dt = fn_mask_dt[1:-1, 1:-1]
            fp_mask_dt = fp_mask_dt[1:-1, 1:-1]

        # take the point in FN/FP region with the largest distance to its boundary
        fn_mask_dt_flat = fn_mask_dt.reshape(-1)
        fp_mask_dt_flat = fp_mask_dt.reshape(-1)
        fn_argmax = np.argmax(fn_mask_dt_flat)
        fp_argmax = np.argmax(fp_mask_dt_flat)
        is_positive = fn_mask_dt_flat[fn_argmax] > fp_mask_dt_flat[fp_argmax]
        pt_idx = fn_argmax if is_positive else fp_argmax
        points[b, 0, 0] = pt_idx % W_im  # x
        points[b, 0, 1] = pt_idx // W_im  # y
        labels[b, 0] = int(is_positive)

    points = points.to(device)
    labels = labels.to(device)
    return points, labels


def get_next_point(gt_masks, pred_masks, method):
    if method == "uniform":
        return sample_random_points_from_errors(gt_masks, pred_masks)
    elif method == "center":
        return sample_one_point_from_error_center(gt_masks, pred_masks)
    else:
        raise ValueError(f"unknown sampling method {method}")


class LayerScale(nn.Module):
    def __init__(
        self,
        dim: int,
        init_values: Union[float, torch.Tensor] = 1e-5,
        inplace: bool = False,
    ) -> None:
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.mul_(self.gamma) if self.inplace else x * self.gamma
'''

# Native module: test-code/source/MedSAM2-main/sam2/modeling/memory_attention.py
_EMBEDDED_SOURCES['test-code/source/MedSAM2-main/sam2/modeling/memory_attention.py'] = '''# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional

import torch
from torch import nn, Tensor

from sam2.modeling.sam.transformer import RoPEAttention

from sam2.modeling.sam2_utils import get_activation_fn, get_clones


class MemoryAttentionLayer(nn.Module):

    def __init__(
        self,
        activation: str,
        cross_attention: nn.Module,
        d_model: int,
        dim_feedforward: int,
        dropout: float,
        pos_enc_at_attn: bool,
        pos_enc_at_cross_attn_keys: bool,
        pos_enc_at_cross_attn_queries: bool,
        self_attention: nn.Module,
    ):
        super().__init__()
        self.d_model = d_model
        self.dim_feedforward = dim_feedforward
        self.dropout_value = dropout
        self.self_attn = self_attention
        self.cross_attn_image = cross_attention

        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation_str = activation
        self.activation = get_activation_fn(activation)

        # Where to add pos enc
        self.pos_enc_at_attn = pos_enc_at_attn
        self.pos_enc_at_cross_attn_queries = pos_enc_at_cross_attn_queries
        self.pos_enc_at_cross_attn_keys = pos_enc_at_cross_attn_keys

    def _forward_sa(self, tgt, query_pos):
        # Self-Attention
        tgt2 = self.norm1(tgt)
        q = k = tgt2 + query_pos if self.pos_enc_at_attn else tgt2
        tgt2 = self.self_attn(q, k, v=tgt2)
        tgt = tgt + self.dropout1(tgt2)
        return tgt

    def _forward_ca(self, tgt, memory, query_pos, pos, num_k_exclude_rope=0):
        kwds = {}
        if num_k_exclude_rope > 0:
            assert isinstance(self.cross_attn_image, RoPEAttention)
            kwds = {"num_k_exclude_rope": num_k_exclude_rope}

        # Cross-Attention
        tgt2 = self.norm2(tgt)
        tgt2 = self.cross_attn_image(
            q=tgt2 + query_pos if self.pos_enc_at_cross_attn_queries else tgt2,
            k=memory + pos if self.pos_enc_at_cross_attn_keys else memory,
            v=memory,
            **kwds,
        )
        tgt = tgt + self.dropout2(tgt2)
        return tgt

    def forward(
        self,
        tgt,
        memory,
        pos: Optional[Tensor] = None,
        query_pos: Optional[Tensor] = None,
        num_k_exclude_rope: int = 0,
    ) -> torch.Tensor:

        # Self-Attn, Cross-Attn
        tgt = self._forward_sa(tgt, query_pos)
        tgt = self._forward_ca(tgt, memory, query_pos, pos, num_k_exclude_rope)
        # MLP
        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)
        return tgt


class MemoryAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        pos_enc_at_input: bool,
        layer: nn.Module,
        num_layers: int,
        batch_first: bool = True,  # Do layers expect batch first input?
    ):
        super().__init__()
        self.d_model = d_model
        self.layers = get_clones(layer, num_layers)
        self.num_layers = num_layers
        self.norm = nn.LayerNorm(d_model)
        self.pos_enc_at_input = pos_enc_at_input
        self.batch_first = batch_first

    def forward(
        self,
        curr: torch.Tensor,  # self-attention inputs
        memory: torch.Tensor,  # cross-attention inputs
        curr_pos: Optional[Tensor] = None,  # pos_enc for self-attention inputs
        memory_pos: Optional[Tensor] = None,  # pos_enc for cross-attention inputs
        num_obj_ptr_tokens: int = 0,  # number of object pointer *tokens*
    ):
        if isinstance(curr, list):
            assert isinstance(curr_pos, list)
            assert len(curr) == len(curr_pos) == 1
            curr, curr_pos = (
                curr[0],
                curr_pos[0],
            )

        assert (
            curr.shape[1] == memory.shape[1]
        ), "Batch size must be the same for curr and memory"

        output = curr
        if self.pos_enc_at_input and curr_pos is not None:
            output = output + 0.1 * curr_pos

        if self.batch_first:
            # Convert to batch first
            output = output.transpose(0, 1)
            curr_pos = curr_pos.transpose(0, 1)
            memory = memory.transpose(0, 1)
            memory_pos = memory_pos.transpose(0, 1)

        for layer in self.layers:
            kwds = {}
            if isinstance(layer.cross_attn_image, RoPEAttention):
                kwds = {"num_k_exclude_rope": num_obj_ptr_tokens}

            output = layer(
                tgt=output,
                memory=memory,
                pos=memory_pos,
                query_pos=curr_pos,
                **kwds,
            )
        normed_output = self.norm(output)

        if self.batch_first:
            # Convert back to seq first
            normed_output = normed_output.transpose(0, 1)
            curr_pos = curr_pos.transpose(0, 1)

        return normed_output
'''

# Native module: test-code/source/MedSAM2-main/sam2/modeling/memory_encoder.py
_EMBEDDED_SOURCES['test-code/source/MedSAM2-main/sam2/modeling/memory_encoder.py'] = '''# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from sam2.modeling.sam2_utils import DropPath, get_clones, LayerNorm2d


class MaskDownSampler(nn.Module):
    """
    Progressively downsample a mask by total_stride, each time by stride.
    Note that LayerNorm is applied per *token*, like in ViT.

    With each downsample (by a factor stride**2), channel capacity increases by the same factor.
    In the end, we linearly project to embed_dim channels.
    """

    def __init__(
        self,
        embed_dim=256,
        kernel_size=4,
        stride=4,
        padding=0,
        total_stride=16,
        activation=nn.GELU,
    ):
        super().__init__()
        num_layers = int(math.log2(total_stride) // math.log2(stride))
        assert stride**num_layers == total_stride
        self.encoder = nn.Sequential()
        mask_in_chans, mask_out_chans = 1, 1
        for _ in range(num_layers):
            mask_out_chans = mask_in_chans * (stride**2)
            self.encoder.append(
                nn.Conv2d(
                    mask_in_chans,
                    mask_out_chans,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                )
            )
            self.encoder.append(LayerNorm2d(mask_out_chans))
            self.encoder.append(activation())
            mask_in_chans = mask_out_chans

        self.encoder.append(nn.Conv2d(mask_out_chans, embed_dim, kernel_size=1))

    def forward(self, x):
        return self.encoder(x)


# Lightly adapted from ConvNext (https://github.com/facebookresearch/ConvNeXt)
class CXBlock(nn.Module):
    r"""ConvNeXt Block. There are two equivalent implementations:
    (1) DwConv -> LayerNorm (channels_first) -> 1x1 Conv -> GELU -> 1x1 Conv; all in (N, C, H, W)
    (2) DwConv -> Permute to (N, H, W, C); LayerNorm (channels_last) -> Linear -> GELU -> Linear; Permute back
    We use (2) as we find it slightly faster in PyTorch

    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
    """

    def __init__(
        self,
        dim,
        kernel_size=7,
        padding=3,
        drop_path=0.0,
        layer_scale_init_value=1e-6,
        use_dwconv=True,
    ):
        super().__init__()
        self.dwconv = nn.Conv2d(
            dim,
            dim,
            kernel_size=kernel_size,
            padding=padding,
            groups=dim if use_dwconv else 1,
        )  # depthwise conv
        self.norm = LayerNorm2d(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(
            dim, 4 * dim
        )  # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = (
            nn.Parameter(layer_scale_init_value * torch.ones((dim)), requires_grad=True)
            if layer_scale_init_value > 0
            else None
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = self.norm(x)
        x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)

        x = input + self.drop_path(x)
        return x


class Fuser(nn.Module):
    def __init__(self, layer, num_layers, dim=None, input_projection=False):
        super().__init__()
        self.proj = nn.Identity()
        self.layers = get_clones(layer, num_layers)

        if input_projection:
            assert dim is not None
            self.proj = nn.Conv2d(dim, dim, kernel_size=1)

    def forward(self, x):
        # normally x: (N, C, H, W)
        x = self.proj(x)
        for layer in self.layers:
            x = layer(x)
        return x


class MemoryEncoder(nn.Module):
    def __init__(
        self,
        out_dim,
        mask_downsampler,
        fuser,
        position_encoding,
        in_dim=256,  # in_dim of pix_feats
    ):
        super().__init__()

        self.mask_downsampler = mask_downsampler

        self.pix_feat_proj = nn.Conv2d(in_dim, in_dim, kernel_size=1)
        self.fuser = fuser
        self.position_encoding = position_encoding
        self.out_proj = nn.Identity()
        if out_dim != in_dim:
            self.out_proj = nn.Conv2d(in_dim, out_dim, kernel_size=1)

    def forward(
        self,
        pix_feat: torch.Tensor,
        masks: torch.Tensor,
        skip_mask_sigmoid: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        ## Process masks
        # sigmoid, so that less domain shift from gt masks which are bool
        if not skip_mask_sigmoid:
            masks = F.sigmoid(masks)
        masks = self.mask_downsampler(masks)

        ## Fuse pix_feats and downsampled masks
        # in case the visual features are on CPU, cast them to CUDA
        pix_feat = pix_feat.to(masks.device)

        x = self.pix_feat_proj(pix_feat)
        x = x + masks
        x = self.fuser(x)
        x = self.out_proj(x)

        pos = self.position_encoding(x).to(x.dtype)

        return {"vision_features": x, "vision_pos_enc": [pos]}
'''

# Native module: test-code/source/MedSAM2-main/sam2/modeling/position_encoding.py
_EMBEDDED_SOURCES['test-code/source/MedSAM2-main/sam2/modeling/position_encoding.py'] = '''# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import math
from typing import Any, Optional, Tuple

import numpy as np

import torch
from torch import nn


class PositionEmbeddingSine(nn.Module):
    """
    This is a more standard version of the position embedding, very similar to the one
    used by the Attention Is All You Need paper, generalized to work on images.
    """

    def __init__(
        self,
        num_pos_feats,
        temperature: int = 10000,
        normalize: bool = True,
        scale: Optional[float] = None,
    ):
        super().__init__()
        assert num_pos_feats % 2 == 0, "Expecting even model width"
        self.num_pos_feats = num_pos_feats // 2
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

        self.cache = {}

    def _encode_xy(self, x, y):
        # The positions are expected to be normalized
        assert len(x) == len(y) and x.ndim == y.ndim == 1
        x_embed = x * self.scale
        y_embed = y * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        pos_x = x_embed[:, None] / dim_t
        pos_y = y_embed[:, None] / dim_t
        pos_x = torch.stack(
            (pos_x[:, 0::2].sin(), pos_x[:, 1::2].cos()), dim=2
        ).flatten(1)
        pos_y = torch.stack(
            (pos_y[:, 0::2].sin(), pos_y[:, 1::2].cos()), dim=2
        ).flatten(1)
        return pos_x, pos_y

    @torch.no_grad()
    def encode_boxes(self, x, y, w, h):
        pos_x, pos_y = self._encode_xy(x, y)
        pos = torch.cat((pos_y, pos_x, h[:, None], w[:, None]), dim=1)
        return pos

    encode = encode_boxes  # Backwards compatibility

    @torch.no_grad()
    def encode_points(self, x, y, labels):
        (bx, nx), (by, ny), (bl, nl) = x.shape, y.shape, labels.shape
        assert bx == by and nx == ny and bx == bl and nx == nl
        pos_x, pos_y = self._encode_xy(x.flatten(), y.flatten())
        pos_x, pos_y = pos_x.reshape(bx, nx, -1), pos_y.reshape(by, ny, -1)
        pos = torch.cat((pos_y, pos_x, labels[:, :, None]), dim=2)
        return pos

    @torch.no_grad()
    def forward(self, x: torch.Tensor):
        cache_key = (x.shape[-2], x.shape[-1])
        if cache_key in self.cache:
            return self.cache[cache_key][None].repeat(x.shape[0], 1, 1, 1)
        y_embed = (
            torch.arange(1, x.shape[-2] + 1, dtype=torch.float32, device=x.device)
            .view(1, -1, 1)
            .repeat(x.shape[0], 1, x.shape[-1])
        )
        x_embed = (
            torch.arange(1, x.shape[-1] + 1, dtype=torch.float32, device=x.device)
            .view(1, 1, -1)
            .repeat(x.shape[0], x.shape[-2], 1)
        )

        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack(
            (pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4
        ).flatten(3)
        pos_y = torch.stack(
            (pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4
        ).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        self.cache[cache_key] = pos[0]
        return pos


class PositionEmbeddingRandom(nn.Module):
    """
    Positional encoding using random spatial frequencies.
    """

    def __init__(self, num_pos_feats: int = 64, scale: Optional[float] = None) -> None:
        super().__init__()
        if scale is None or scale <= 0.0:
            scale = 1.0
        self.register_buffer(
            "positional_encoding_gaussian_matrix",
            scale * torch.randn((2, num_pos_feats)),
        )

    def _pe_encoding(self, coords: torch.Tensor) -> torch.Tensor:
        """Positionally encode points that are normalized to [0,1]."""
        # assuming coords are in [0, 1]^2 square and have d_1 x ... x d_n x 2 shape
        coords = 2 * coords - 1
        coords = coords @ self.positional_encoding_gaussian_matrix
        coords = 2 * np.pi * coords
        # outputs d_1 x ... x d_n x C shape
        return torch.cat([torch.sin(coords), torch.cos(coords)], dim=-1)

    def forward(self, size: Tuple[int, int]) -> torch.Tensor:
        """Generate positional encoding for a grid of the specified size."""
        h, w = size
        device: Any = self.positional_encoding_gaussian_matrix.device
        grid = torch.ones((h, w), device=device, dtype=torch.float32)
        y_embed = grid.cumsum(dim=0) - 0.5
        x_embed = grid.cumsum(dim=1) - 0.5
        y_embed = y_embed / h
        x_embed = x_embed / w

        pe = self._pe_encoding(torch.stack([x_embed, y_embed], dim=-1))
        return pe.permute(2, 0, 1)  # C x H x W

    def forward_with_coords(
        self, coords_input: torch.Tensor, image_size: Tuple[int, int]
    ) -> torch.Tensor:
        """Positionally encode points that are not normalized to [0,1]."""
        coords = coords_input.clone()
        coords[:, :, 0] = coords[:, :, 0] / image_size[1]
        coords[:, :, 1] = coords[:, :, 1] / image_size[0]
        return self._pe_encoding(coords.to(torch.float))  # B x N x C


# Rotary Positional Encoding, adapted from:
# 1. https://github.com/meta-llama/codellama/blob/main/llama/model.py
# 2. https://github.com/naver-ai/rope-vit
# 3. https://github.com/lucidrains/rotary-embedding-torch


def init_t_xy(end_x: int, end_y: int):
    t = torch.arange(end_x * end_y, dtype=torch.float32)
    t_x = (t % end_x).float()
    t_y = torch.div(t, end_x, rounding_mode="floor").float()
    return t_x, t_y


def compute_axial_cis(dim: int, end_x: int, end_y: int, theta: float = 10000.0):
    freqs_x = 1.0 / (theta ** (torch.arange(0, dim, 4)[: (dim // 4)].float() / dim))
    freqs_y = 1.0 / (theta ** (torch.arange(0, dim, 4)[: (dim // 4)].float() / dim))

    t_x, t_y = init_t_xy(end_x, end_y)
    freqs_x = torch.outer(t_x, freqs_x)
    freqs_y = torch.outer(t_y, freqs_y)
    freqs_cis_x = torch.polar(torch.ones_like(freqs_x), freqs_x)
    freqs_cis_y = torch.polar(torch.ones_like(freqs_y), freqs_y)
    return torch.cat([freqs_cis_x, freqs_cis_y], dim=-1)


def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[-2], x.shape[-1])
    shape = [d if i >= ndim - 2 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)


def apply_rotary_enc(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
    repeat_freqs_k: bool = False,
):
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = (
        torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
        if xk.shape[-2] != 0
        else None
    )
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    if xk_ is None:
        # no keys to rotate, due to dropout
        return xq_out.type_as(xq).to(xq.device), xk
    # repeat freqs along seq_len dim to match k seq_len
    if repeat_freqs_k:
        r = xk_.shape[-2] // xq_.shape[-2]
        if freqs_cis.is_cuda:
            freqs_cis = freqs_cis.repeat(*([1] * (freqs_cis.ndim - 2)), r, 1)
        else:
            # torch.repeat on complex numbers may not be supported on non-CUDA devices
            # (freqs_cis has 4 dims and we repeat on dim 2) so we use expand + flatten
            freqs_cis = freqs_cis.unsqueeze(2).expand(-1, -1, r, -1, -1).flatten(2, 3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq).to(xq.device), xk_out.type_as(xk).to(xk.device)
'''

# Native module: test-code/source/MedSAM2-main/sam2/modeling/sam/__init__.py
_EMBEDDED_SOURCES['test-code/source/MedSAM2-main/sam2/modeling/sam/__init__.py'] = '''# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
'''

# Native module: test-code/source/MedSAM2-main/sam2/modeling/sam/mask_decoder.py
_EMBEDDED_SOURCES['test-code/source/MedSAM2-main/sam2/modeling/sam/mask_decoder.py'] = '''# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import List, Optional, Tuple, Type

import torch
from torch import nn

from sam2.modeling.sam2_utils import LayerNorm2d, MLP


class MaskDecoder(nn.Module):
    def __init__(
        self,
        *,
        transformer_dim: int,
        transformer: nn.Module,
        num_multimask_outputs: int = 3,
        activation: Type[nn.Module] = nn.GELU,
        iou_head_depth: int = 3,
        iou_head_hidden_dim: int = 256,
        use_high_res_features: bool = False,
        iou_prediction_use_sigmoid=False,
        dynamic_multimask_via_stability=False,
        dynamic_multimask_stability_delta=0.05,
        dynamic_multimask_stability_thresh=0.98,
        pred_obj_scores: bool = False,
        pred_obj_scores_mlp: bool = False,
        use_multimask_token_for_obj_ptr: bool = False,
    ) -> None:
        """
        Predicts masks given an image and prompt embeddings, using a
        transformer architecture.

        Arguments:
          transformer_dim (int): the channel dimension of the transformer
          transformer (nn.Module): the transformer used to predict masks
          num_multimask_outputs (int): the number of masks to predict
            when disambiguating masks
          activation (nn.Module): the type of activation to use when
            upscaling masks
          iou_head_depth (int): the depth of the MLP used to predict
            mask quality
          iou_head_hidden_dim (int): the hidden dimension of the MLP
            used to predict mask quality
        """
        super().__init__()
        self.transformer_dim = transformer_dim
        self.transformer = transformer

        self.num_multimask_outputs = num_multimask_outputs

        self.iou_token = nn.Embedding(1, transformer_dim)
        self.num_mask_tokens = num_multimask_outputs + 1
        self.mask_tokens = nn.Embedding(self.num_mask_tokens, transformer_dim)

        self.pred_obj_scores = pred_obj_scores
        if self.pred_obj_scores:
            self.obj_score_token = nn.Embedding(1, transformer_dim)
        self.use_multimask_token_for_obj_ptr = use_multimask_token_for_obj_ptr

        self.output_upscaling = nn.Sequential(
            nn.ConvTranspose2d(
                transformer_dim, transformer_dim // 4, kernel_size=2, stride=2
            ),
            LayerNorm2d(transformer_dim // 4),
            activation(),
            nn.ConvTranspose2d(
                transformer_dim // 4, transformer_dim // 8, kernel_size=2, stride=2
            ),
            activation(),
        )
        self.use_high_res_features = use_high_res_features
        if use_high_res_features:
            self.conv_s0 = nn.Conv2d(
                transformer_dim, transformer_dim // 8, kernel_size=1, stride=1
            )
            self.conv_s1 = nn.Conv2d(
                transformer_dim, transformer_dim // 4, kernel_size=1, stride=1
            )

        self.output_hypernetworks_mlps = nn.ModuleList(
            [
                MLP(transformer_dim, transformer_dim, transformer_dim // 8, 3)
                for i in range(self.num_mask_tokens)
            ]
        )

        self.iou_prediction_head = MLP(
            transformer_dim,
            iou_head_hidden_dim,
            self.num_mask_tokens,
            iou_head_depth,
            sigmoid_output=iou_prediction_use_sigmoid,
        )
        if self.pred_obj_scores:
            self.pred_obj_score_head = nn.Linear(transformer_dim, 1)
            if pred_obj_scores_mlp:
                self.pred_obj_score_head = MLP(transformer_dim, transformer_dim, 1, 3)

        # When outputting a single mask, optionally we can dynamically fall back to the best
        # multimask output token if the single mask output token gives low stability scores.
        self.dynamic_multimask_via_stability = dynamic_multimask_via_stability
        self.dynamic_multimask_stability_delta = dynamic_multimask_stability_delta
        self.dynamic_multimask_stability_thresh = dynamic_multimask_stability_thresh

    def forward(
        self,
        image_embeddings: torch.Tensor,
        image_pe: torch.Tensor,
        sparse_prompt_embeddings: torch.Tensor,
        dense_prompt_embeddings: torch.Tensor,
        multimask_output: bool,
        repeat_image: bool,
        high_res_features: Optional[List[torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict masks given image and prompt embeddings.

        Arguments:
          image_embeddings (torch.Tensor): the embeddings from the image encoder
          image_pe (torch.Tensor): positional encoding with the shape of image_embeddings
          sparse_prompt_embeddings (torch.Tensor): the embeddings of the points and boxes
          dense_prompt_embeddings (torch.Tensor): the embeddings of the mask inputs
          multimask_output (bool): Whether to return multiple masks or a single
            mask.

        Returns:
          torch.Tensor: batched predicted masks
          torch.Tensor: batched predictions of mask quality
          torch.Tensor: batched SAM token for mask output
        """
        masks, iou_pred, mask_tokens_out, object_score_logits = self.predict_masks(
            image_embeddings=image_embeddings,
            image_pe=image_pe,
            sparse_prompt_embeddings=sparse_prompt_embeddings,
            dense_prompt_embeddings=dense_prompt_embeddings,
            repeat_image=repeat_image,
            high_res_features=high_res_features,
        )

        # Select the correct mask or masks for output
        if multimask_output:
            masks = masks[:, 1:, :, :]
            iou_pred = iou_pred[:, 1:]
        elif self.dynamic_multimask_via_stability and not self.training:
            masks, iou_pred = self._dynamic_multimask_via_stability(masks, iou_pred)
        else:
            masks = masks[:, 0:1, :, :]
            iou_pred = iou_pred[:, 0:1]

        if multimask_output and self.use_multimask_token_for_obj_ptr:
            sam_tokens_out = mask_tokens_out[:, 1:]  # [b, 3, c] shape
        else:
            # Take the mask output token. Here we *always* use the token for single mask output.
            # At test time, even if we track after 1-click (and using multimask_output=True),
            # we still take the single mask token here. The rationale is that we always track
            # after multiple clicks during training, so the past tokens seen during training
            # are always the single mask token (and we'll let it be the object-memory token).
            sam_tokens_out = mask_tokens_out[:, 0:1]  # [b, 1, c] shape

        # Prepare output
        return masks, iou_pred, sam_tokens_out, object_score_logits

    def predict_masks(
        self,
        image_embeddings: torch.Tensor,
        image_pe: torch.Tensor,
        sparse_prompt_embeddings: torch.Tensor,
        dense_prompt_embeddings: torch.Tensor,
        repeat_image: bool,
        high_res_features: Optional[List[torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Predicts masks. See 'forward' for more details."""
        # Concatenate output tokens
        s = 0
        if self.pred_obj_scores:
            output_tokens = torch.cat(
                [
                    self.obj_score_token.weight,
                    self.iou_token.weight,
                    self.mask_tokens.weight,
                ],
                dim=0,
            )
            s = 1
        else:
            output_tokens = torch.cat(
                [self.iou_token.weight, self.mask_tokens.weight], dim=0
            )
        output_tokens = output_tokens.unsqueeze(0).expand(
            sparse_prompt_embeddings.size(0), -1, -1
        )
        tokens = torch.cat((output_tokens, sparse_prompt_embeddings), dim=1)

        # Expand per-image data in batch direction to be per-mask
        if repeat_image:
            src = torch.repeat_interleave(image_embeddings, tokens.shape[0], dim=0)
        else:
            assert image_embeddings.shape[0] == tokens.shape[0]
            src = image_embeddings
        src = src + dense_prompt_embeddings
        assert (
            image_pe.size(0) == 1
        ), "image_pe should have size 1 in batch dim (from `get_dense_pe()`)"
        pos_src = torch.repeat_interleave(image_pe, tokens.shape[0], dim=0)
        b, c, h, w = src.shape

        # Run the transformer
        hs, src = self.transformer(src, pos_src, tokens)
        iou_token_out = hs[:, s, :]
        mask_tokens_out = hs[:, s + 1 : (s + 1 + self.num_mask_tokens), :]

        # Upscale mask embeddings and predict masks using the mask tokens
        src = src.transpose(1, 2).view(b, c, h, w)
        if not self.use_high_res_features:
            upscaled_embedding = self.output_upscaling(src)
        else:
            dc1, ln1, act1, dc2, act2 = self.output_upscaling
            feat_s0, feat_s1 = high_res_features
            upscaled_embedding = act1(ln1(dc1(src) + feat_s1))
            upscaled_embedding = act2(dc2(upscaled_embedding) + feat_s0)

        hyper_in_list: List[torch.Tensor] = []
        for i in range(self.num_mask_tokens):
            hyper_in_list.append(
                self.output_hypernetworks_mlps[i](mask_tokens_out[:, i, :])
            )
        hyper_in = torch.stack(hyper_in_list, dim=1)
        b, c, h, w = upscaled_embedding.shape
        masks = (hyper_in @ upscaled_embedding.view(b, c, h * w)).view(b, -1, h, w)

        # Generate mask quality predictions
        iou_pred = self.iou_prediction_head(iou_token_out)
        if self.pred_obj_scores:
            assert s == 1
            object_score_logits = self.pred_obj_score_head(hs[:, 0, :])
        else:
            # Obj scores logits - default to 10.0, i.e. assuming the object is present, sigmoid(10)=1
            object_score_logits = 10.0 * iou_pred.new_ones(iou_pred.shape[0], 1)

        return masks, iou_pred, mask_tokens_out, object_score_logits

    def _get_stability_scores(self, mask_logits):
        """
        Compute stability scores of the mask logits based on the IoU between upper and
        lower thresholds.
        """
        mask_logits = mask_logits.flatten(-2)
        stability_delta = self.dynamic_multimask_stability_delta
        area_i = torch.sum(mask_logits > stability_delta, dim=-1).float()
        area_u = torch.sum(mask_logits > -stability_delta, dim=-1).float()
        stability_scores = torch.where(area_u > 0, area_i / area_u, 1.0)
        return stability_scores

    def _dynamic_multimask_via_stability(self, all_mask_logits, all_iou_scores):
        """
        When outputting a single mask, if the stability score from the current single-mask
        output (based on output token 0) falls below a threshold, we instead select from
        multi-mask outputs (based on output token 1~3) the mask with the highest predicted
        IoU score. This is intended to ensure a valid mask for both clicking and tracking.
        """
        # The best mask from multimask output tokens (1~3)
        multimask_logits = all_mask_logits[:, 1:, :, :]
        multimask_iou_scores = all_iou_scores[:, 1:]
        best_scores_inds = torch.argmax(multimask_iou_scores, dim=-1)
        batch_inds = torch.arange(
            multimask_iou_scores.size(0), device=all_iou_scores.device
        )
        best_multimask_logits = multimask_logits[batch_inds, best_scores_inds]
        best_multimask_logits = best_multimask_logits.unsqueeze(1)
        best_multimask_iou_scores = multimask_iou_scores[batch_inds, best_scores_inds]
        best_multimask_iou_scores = best_multimask_iou_scores.unsqueeze(1)

        # The mask from singlemask output token 0 and its stability score
        singlemask_logits = all_mask_logits[:, 0:1, :, :]
        singlemask_iou_scores = all_iou_scores[:, 0:1]
        stability_scores = self._get_stability_scores(singlemask_logits)
        is_stable = stability_scores >= self.dynamic_multimask_stability_thresh

        # Dynamically fall back to best multimask output upon low stability scores.
        mask_logits_out = torch.where(
            is_stable[..., None, None].expand_as(singlemask_logits),
            singlemask_logits,
            best_multimask_logits,
        )
        iou_scores_out = torch.where(
            is_stable.expand_as(singlemask_iou_scores),
            singlemask_iou_scores,
            best_multimask_iou_scores,
        )
        return mask_logits_out, iou_scores_out
'''

# Native module: test-code/source/MedSAM2-main/sam2/modeling/sam/prompt_encoder.py
_EMBEDDED_SOURCES['test-code/source/MedSAM2-main/sam2/modeling/sam/prompt_encoder.py'] = '''# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional, Tuple, Type

import torch
from torch import nn

from sam2.modeling.position_encoding import PositionEmbeddingRandom

from sam2.modeling.sam2_utils import LayerNorm2d


class PromptEncoder(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        image_embedding_size: Tuple[int, int],
        input_image_size: Tuple[int, int],
        mask_in_chans: int,
        activation: Type[nn.Module] = nn.GELU,
    ) -> None:
        """
        Encodes prompts for input to SAM's mask decoder.

        Arguments:
          embed_dim (int): The prompts' embedding dimension
          image_embedding_size (tuple(int, int)): The spatial size of the
            image embedding, as (H, W).
          input_image_size (int): The padded size of the image as input
            to the image encoder, as (H, W).
          mask_in_chans (int): The number of hidden channels used for
            encoding input masks.
          activation (nn.Module): The activation to use when encoding
            input masks.
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.input_image_size = input_image_size
        self.image_embedding_size = image_embedding_size
        self.pe_layer = PositionEmbeddingRandom(embed_dim // 2)

        self.num_point_embeddings: int = 4  # pos/neg point + 2 box corners
        point_embeddings = [
            nn.Embedding(1, embed_dim) for i in range(self.num_point_embeddings)
        ]
        self.point_embeddings = nn.ModuleList(point_embeddings)
        self.not_a_point_embed = nn.Embedding(1, embed_dim)

        self.mask_input_size = (
            4 * image_embedding_size[0],
            4 * image_embedding_size[1],
        )
        self.mask_downscaling = nn.Sequential(
            nn.Conv2d(1, mask_in_chans // 4, kernel_size=2, stride=2),
            LayerNorm2d(mask_in_chans // 4),
            activation(),
            nn.Conv2d(mask_in_chans // 4, mask_in_chans, kernel_size=2, stride=2),
            LayerNorm2d(mask_in_chans),
            activation(),
            nn.Conv2d(mask_in_chans, embed_dim, kernel_size=1),
        )
        self.no_mask_embed = nn.Embedding(1, embed_dim)

    def get_dense_pe(self) -> torch.Tensor:
        """
        Returns the positional encoding used to encode point prompts,
        applied to a dense set of points the shape of the image encoding.

        Returns:
          torch.Tensor: Positional encoding with shape
            1x(embed_dim)x(embedding_h)x(embedding_w)
        """
        return self.pe_layer(self.image_embedding_size).unsqueeze(0)

    def _embed_points(
        self,
        points: torch.Tensor,
        labels: torch.Tensor,
        pad: bool,
    ) -> torch.Tensor:
        """Embeds point prompts."""
        points = points + 0.5  # Shift to center of pixel
        if pad:
            padding_point = torch.zeros((points.shape[0], 1, 2), device=points.device)
            padding_label = -torch.ones((labels.shape[0], 1), device=labels.device)
            points = torch.cat([points, padding_point], dim=1)
            labels = torch.cat([labels, padding_label], dim=1)
        point_embedding = self.pe_layer.forward_with_coords(
            points, self.input_image_size
        )
        point_embedding[labels == -1] = 0.0
        point_embedding[labels == -1] += self.not_a_point_embed.weight
        point_embedding[labels == 0] += self.point_embeddings[0].weight
        point_embedding[labels == 1] += self.point_embeddings[1].weight
        point_embedding[labels == 2] += self.point_embeddings[2].weight
        point_embedding[labels == 3] += self.point_embeddings[3].weight
        return point_embedding

    def _embed_boxes(self, boxes: torch.Tensor) -> torch.Tensor:
        """Embeds box prompts."""
        boxes = boxes + 0.5  # Shift to center of pixel
        coords = boxes.reshape(-1, 2, 2)
        corner_embedding = self.pe_layer.forward_with_coords(
            coords, self.input_image_size
        )
        corner_embedding[:, 0, :] += self.point_embeddings[2].weight
        corner_embedding[:, 1, :] += self.point_embeddings[3].weight
        return corner_embedding

    def _embed_masks(self, masks: torch.Tensor) -> torch.Tensor:
        """Embeds mask inputs."""
        mask_embedding = self.mask_downscaling(masks)
        return mask_embedding

    def _get_batch_size(
        self,
        points: Optional[Tuple[torch.Tensor, torch.Tensor]],
        boxes: Optional[torch.Tensor],
        masks: Optional[torch.Tensor],
    ) -> int:
        """
        Gets the batch size of the output given the batch size of the input prompts.
        """
        if points is not None:
            return points[0].shape[0]
        elif boxes is not None:
            return boxes.shape[0]
        elif masks is not None:
            return masks.shape[0]
        else:
            return 1

    def _get_device(self) -> torch.device:
        return self.point_embeddings[0].weight.device

    def forward(
        self,
        points: Optional[Tuple[torch.Tensor, torch.Tensor]],
        boxes: Optional[torch.Tensor],
        masks: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Embeds different types of prompts, returning both sparse and dense
        embeddings.

        Arguments:
          points (tuple(torch.Tensor, torch.Tensor) or none): point coordinates
            and labels to embed.
          boxes (torch.Tensor or none): boxes to embed
          masks (torch.Tensor or none): masks to embed

        Returns:
          torch.Tensor: sparse embeddings for the points and boxes, with shape
            BxNx(embed_dim), where N is determined by the number of input points
            and boxes.
          torch.Tensor: dense embeddings for the masks, in the shape
            Bx(embed_dim)x(embed_H)x(embed_W)
        """
        bs = self._get_batch_size(points, boxes, masks)
        sparse_embeddings = torch.empty(
            (bs, 0, self.embed_dim), device=self._get_device()
        )
        if points is not None:
            coords, labels = points
            point_embeddings = self._embed_points(coords, labels, pad=(boxes is None))
            sparse_embeddings = torch.cat([sparse_embeddings, point_embeddings], dim=1)
        if boxes is not None:
            box_embeddings = self._embed_boxes(boxes)
            sparse_embeddings = torch.cat([sparse_embeddings, box_embeddings], dim=1)

        if masks is not None:
            dense_embeddings = self._embed_masks(masks)
        else:
            dense_embeddings = self.no_mask_embed.weight.reshape(1, -1, 1, 1).expand(
                bs, -1, self.image_embedding_size[0], self.image_embedding_size[1]
            )

        return sparse_embeddings, dense_embeddings
'''

# Native module: test-code/source/MedSAM2-main/sam2/modeling/sam/transformer.py
_EMBEDDED_SOURCES['test-code/source/MedSAM2-main/sam2/modeling/sam/transformer.py'] = '''# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import contextlib
import math
import warnings
from functools import partial
from typing import Tuple, Type

import torch
import torch.nn.functional as F
from torch import nn, Tensor

from sam2.modeling.position_encoding import apply_rotary_enc, compute_axial_cis
from sam2.modeling.sam2_utils import MLP
from sam2.utils.misc import get_sdpa_settings

warnings.simplefilter(action="ignore", category=FutureWarning)
# Check whether Flash Attention is available (and use it by default)
OLD_GPU, USE_FLASH_ATTN, MATH_KERNEL_ON = get_sdpa_settings()
# A fallback setting to allow all available kernels if Flash Attention fails
ALLOW_ALL_KERNELS = False


def sdp_kernel_context(dropout_p):
    """
    Get the context for the attention scaled dot-product kernel. We use Flash Attention
    by default, but fall back to all available kernels if Flash Attention fails.
    """
    if ALLOW_ALL_KERNELS:
        return contextlib.nullcontext()

    return torch.backends.cuda.sdp_kernel(
        enable_flash=USE_FLASH_ATTN,
        # if Flash attention kernel is off, then math kernel needs to be enabled
        enable_math=(OLD_GPU and dropout_p > 0.0) or MATH_KERNEL_ON,
        enable_mem_efficient=OLD_GPU,
    )


class TwoWayTransformer(nn.Module):
    def __init__(
        self,
        depth: int,
        embedding_dim: int,
        num_heads: int,
        mlp_dim: int,
        activation: Type[nn.Module] = nn.ReLU,
        attention_downsample_rate: int = 2,
    ) -> None:
        """
        A transformer decoder that attends to an input image using
        queries whose positional embedding is supplied.

        Args:
          depth (int): number of layers in the transformer
          embedding_dim (int): the channel dimension for the input embeddings
          num_heads (int): the number of heads for multihead attention. Must
            divide embedding_dim
          mlp_dim (int): the channel dimension internal to the MLP block
          activation (nn.Module): the activation to use in the MLP block
        """
        super().__init__()
        self.depth = depth
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.mlp_dim = mlp_dim
        self.layers = nn.ModuleList()

        for i in range(depth):
            self.layers.append(
                TwoWayAttentionBlock(
                    embedding_dim=embedding_dim,
                    num_heads=num_heads,
                    mlp_dim=mlp_dim,
                    activation=activation,
                    attention_downsample_rate=attention_downsample_rate,
                    skip_first_layer_pe=(i == 0),
                )
            )

        self.final_attn_token_to_image = Attention(
            embedding_dim, num_heads, downsample_rate=attention_downsample_rate
        )
        self.norm_final_attn = nn.LayerNorm(embedding_dim)

    def forward(
        self,
        image_embedding: Tensor,
        image_pe: Tensor,
        point_embedding: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """
        Args:
          image_embedding (torch.Tensor): image to attend to. Should be shape
            B x embedding_dim x h x w for any h and w.
          image_pe (torch.Tensor): the positional encoding to add to the image. Must
            have the same shape as image_embedding.
          point_embedding (torch.Tensor): the embedding to add to the query points.
            Must have shape B x N_points x embedding_dim for any N_points.

        Returns:
          torch.Tensor: the processed point_embedding
          torch.Tensor: the processed image_embedding
        """
        # BxCxHxW -> BxHWxC == B x N_image_tokens x C
        bs, c, h, w = image_embedding.shape
        image_embedding = image_embedding.flatten(2).permute(0, 2, 1)
        image_pe = image_pe.flatten(2).permute(0, 2, 1)

        # Prepare queries
        queries = point_embedding
        keys = image_embedding

        # Apply transformer blocks and final layernorm
        for layer in self.layers:
            queries, keys = layer(
                queries=queries,
                keys=keys,
                query_pe=point_embedding,
                key_pe=image_pe,
            )

        # Apply the final attention layer from the points to the image
        q = queries + point_embedding
        k = keys + image_pe
        attn_out = self.final_attn_token_to_image(q=q, k=k, v=keys)
        queries = queries + attn_out
        queries = self.norm_final_attn(queries)

        return queries, keys


class TwoWayAttentionBlock(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        mlp_dim: int = 2048,
        activation: Type[nn.Module] = nn.ReLU,
        attention_downsample_rate: int = 2,
        skip_first_layer_pe: bool = False,
    ) -> None:
        """
        A transformer block with four layers: (1) self-attention of sparse
        inputs, (2) cross attention of sparse inputs to dense inputs, (3) mlp
        block on sparse inputs, and (4) cross attention of dense inputs to sparse
        inputs.

        Arguments:
          embedding_dim (int): the channel dimension of the embeddings
          num_heads (int): the number of heads in the attention layers
          mlp_dim (int): the hidden dimension of the mlp block
          activation (nn.Module): the activation of the mlp block
          skip_first_layer_pe (bool): skip the PE on the first layer
        """
        super().__init__()
        self.self_attn = Attention(embedding_dim, num_heads)
        self.norm1 = nn.LayerNorm(embedding_dim)

        self.cross_attn_token_to_image = Attention(
            embedding_dim, num_heads, downsample_rate=attention_downsample_rate
        )
        self.norm2 = nn.LayerNorm(embedding_dim)

        self.mlp = MLP(
            embedding_dim, mlp_dim, embedding_dim, num_layers=2, activation=activation
        )
        self.norm3 = nn.LayerNorm(embedding_dim)

        self.norm4 = nn.LayerNorm(embedding_dim)
        self.cross_attn_image_to_token = Attention(
            embedding_dim, num_heads, downsample_rate=attention_downsample_rate
        )

        self.skip_first_layer_pe = skip_first_layer_pe

    def forward(
        self, queries: Tensor, keys: Tensor, query_pe: Tensor, key_pe: Tensor
    ) -> Tuple[Tensor, Tensor]:
        # Self attention block
        if self.skip_first_layer_pe:
            queries = self.self_attn(q=queries, k=queries, v=queries)
        else:
            q = queries + query_pe
            attn_out = self.self_attn(q=q, k=q, v=queries)
            queries = queries + attn_out
        queries = self.norm1(queries)

        # Cross attention block, tokens attending to image embedding
        q = queries + query_pe
        k = keys + key_pe
        attn_out = self.cross_attn_token_to_image(q=q, k=k, v=keys)
        queries = queries + attn_out
        queries = self.norm2(queries)

        # MLP block
        mlp_out = self.mlp(queries)
        queries = queries + mlp_out
        queries = self.norm3(queries)

        # Cross attention block, image embedding attending to tokens
        q = queries + query_pe
        k = keys + key_pe
        attn_out = self.cross_attn_image_to_token(q=k, k=q, v=queries)
        keys = keys + attn_out
        keys = self.norm4(keys)

        return queries, keys


class Attention(nn.Module):
    """
    An attention layer that allows for downscaling the size of the embedding
    after projection to queries, keys, and values.
    """

    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        downsample_rate: int = 1,
        dropout: float = 0.0,
        kv_in_dim: int = None,
    ) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.kv_in_dim = kv_in_dim if kv_in_dim is not None else embedding_dim
        self.internal_dim = embedding_dim // downsample_rate
        self.num_heads = num_heads
        assert (
            self.internal_dim % num_heads == 0
        ), "num_heads must divide embedding_dim."

        self.q_proj = nn.Linear(embedding_dim, self.internal_dim)
        self.k_proj = nn.Linear(self.kv_in_dim, self.internal_dim)
        self.v_proj = nn.Linear(self.kv_in_dim, self.internal_dim)
        self.out_proj = nn.Linear(self.internal_dim, embedding_dim)

        self.dropout_p = dropout

    def _separate_heads(self, x: Tensor, num_heads: int) -> Tensor:
        b, n, c = x.shape
        x = x.reshape(b, n, num_heads, c // num_heads)
        return x.transpose(1, 2)  # B x N_heads x N_tokens x C_per_head

    def _recombine_heads(self, x: Tensor) -> Tensor:
        b, n_heads, n_tokens, c_per_head = x.shape
        x = x.transpose(1, 2)
        return x.reshape(b, n_tokens, n_heads * c_per_head)  # B x N_tokens x C

    def forward(self, q: Tensor, k: Tensor, v: Tensor) -> Tensor:
        # Input projections
        q = self.q_proj(q)
        k = self.k_proj(k)
        v = self.v_proj(v)

        # Separate into heads
        q = self._separate_heads(q, self.num_heads)
        k = self._separate_heads(k, self.num_heads)
        v = self._separate_heads(v, self.num_heads)

        dropout_p = self.dropout_p if self.training else 0.0
        # Attention
        try:
            with sdp_kernel_context(dropout_p):
                out = F.scaled_dot_product_attention(q, k, v, dropout_p=dropout_p)
        except Exception as e:
            # Fall back to all kernels if the Flash attention kernel fails
            warnings.warn(
                f"Flash Attention kernel failed due to: {e}\\nFalling back to all available "
                f"kernels for scaled_dot_product_attention (which may have a slower speed).",
                category=UserWarning,
                stacklevel=2,
            )
            global ALLOW_ALL_KERNELS
            ALLOW_ALL_KERNELS = True
            out = F.scaled_dot_product_attention(q, k, v, dropout_p=dropout_p)

        out = self._recombine_heads(out)
        out = self.out_proj(out)

        return out


class RoPEAttention(Attention):
    """Attention with rotary position encoding."""

    def __init__(
        self,
        *args,
        rope_theta=10000.0,
        # whether to repeat q rope to match k length
        # this is needed for cross-attention to memories
        rope_k_repeat=False,
        feat_sizes=(32, 32),  # [w, h] for stride 16 feats at 512 resolution
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.compute_cis = partial(
            compute_axial_cis, dim=self.internal_dim // self.num_heads, theta=rope_theta
        )
        freqs_cis = self.compute_cis(end_x=feat_sizes[0], end_y=feat_sizes[1])
        self.freqs_cis = freqs_cis
        self.rope_k_repeat = rope_k_repeat

    def forward(
        self, q: Tensor, k: Tensor, v: Tensor, num_k_exclude_rope: int = 0
    ) -> Tensor:
        # Input projections
        q = self.q_proj(q)
        k = self.k_proj(k)
        v = self.v_proj(v)

        # Separate into heads
        q = self._separate_heads(q, self.num_heads)
        k = self._separate_heads(k, self.num_heads)
        v = self._separate_heads(v, self.num_heads)

        # Apply rotary position encoding
        w = h = math.sqrt(q.shape[-2])
        self.freqs_cis = self.freqs_cis.to(q.device)
        if self.freqs_cis.shape[0] != q.shape[-2]:
            self.freqs_cis = self.compute_cis(end_x=w, end_y=h).to(q.device)
        if q.shape[-2] != k.shape[-2]:
            assert self.rope_k_repeat

        num_k_rope = k.size(-2) - num_k_exclude_rope
        q, k[:, :, :num_k_rope] = apply_rotary_enc(
            q,
            k[:, :, :num_k_rope],
            freqs_cis=self.freqs_cis,
            repeat_freqs_k=self.rope_k_repeat,
        )

        dropout_p = self.dropout_p if self.training else 0.0
        # Attention
        try:
            with sdp_kernel_context(dropout_p):
                out = F.scaled_dot_product_attention(q, k, v, dropout_p=dropout_p)
        except Exception as e:
            # Fall back to all kernels if the Flash attention kernel fails
            warnings.warn(
                f"Flash Attention kernel failed due to: {e}\\nFalling back to all available "
                f"kernels for scaled_dot_product_attention (which may have a slower speed).",
                category=UserWarning,
                stacklevel=2,
            )
            global ALLOW_ALL_KERNELS
            ALLOW_ALL_KERNELS = True
            out = F.scaled_dot_product_attention(q, k, v, dropout_p=dropout_p)

        out = self._recombine_heads(out)
        out = self.out_proj(out)

        return out
'''

# Native module: test-code/source/MedSAM2-main/sam2/modeling/sam2_base.py
_EMBEDDED_SOURCES['test-code/source/MedSAM2-main/sam2/modeling/sam2_base.py'] = '''# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.distributed
import torch.nn.functional as F

from torch.nn.init import trunc_normal_

from sam2.modeling.sam.mask_decoder import MaskDecoder
from sam2.modeling.sam.prompt_encoder import PromptEncoder
from sam2.modeling.sam.transformer import TwoWayTransformer
from sam2.modeling.sam2_utils import get_1d_sine_pe, MLP, select_closest_cond_frames

# a large negative value as a placeholder score for missing objects
NO_OBJ_SCORE = -1024.0


class SAM2Base(torch.nn.Module):
    def __init__(
        self,
        image_encoder,
        memory_attention,
        memory_encoder,
        num_maskmem=7,  # default 1 input frame + 6 previous frames
        image_size=512,
        backbone_stride=16,  # stride of the image backbone output
        sigmoid_scale_for_mem_enc=1.0,  # scale factor for mask sigmoid prob
        sigmoid_bias_for_mem_enc=0.0,  # bias factor for mask sigmoid prob
        # During evaluation, whether to binarize the sigmoid mask logits on interacted frames with clicks
        binarize_mask_from_pts_for_mem_enc=False,
        use_mask_input_as_output_without_sam=False,  # on frames with mask input, whether to directly output the input mask without using a SAM prompt encoder + mask decoder
        # The maximum number of conditioning frames to participate in the memory attention (-1 means no limit; if there are more conditioning frames than this limit,
        # we only cross-attend to the temporally closest `max_cond_frames_in_attn` conditioning frames in the encoder when tracking each frame). This gives the model
        # a temporal locality when handling a large number of annotated frames (since closer frames should be more important) and also avoids GPU OOM.
        max_cond_frames_in_attn=-1,
        # on the first frame, whether to directly add the no-memory embedding to the image feature
        # (instead of using the transformer encoder)
        directly_add_no_mem_embed=False,
        # whether to use high-resolution feature maps in the SAM mask decoder
        use_high_res_features_in_sam=False,
        # whether to output multiple (3) masks for the first click on initial conditioning frames
        multimask_output_in_sam=False,
        # the minimum and maximum number of clicks to use multimask_output_in_sam (only relevant when `multimask_output_in_sam=True`;
        # default is 1 for both, meaning that only the first click gives multimask output; also note that a box counts as two points)
        multimask_min_pt_num=1,
        multimask_max_pt_num=1,
        # whether to also use multimask output for tracking (not just for the first click on initial conditioning frames; only relevant when `multimask_output_in_sam=True`)
        multimask_output_for_tracking=False,
        # Whether to use multimask tokens for obj ptr; Only relevant when both
        # use_obj_ptrs_in_encoder=True and multimask_output_for_tracking=True
        use_multimask_token_for_obj_ptr: bool = False,
        # whether to use sigmoid to restrict ious prediction to [0-1]
        iou_prediction_use_sigmoid=False,
        # The memory bank's temporal stride during evaluation (i.e. the `r` parameter in XMem and Cutie; XMem and Cutie use r=5).
        # For r>1, the (self.num_maskmem - 1) non-conditioning memory frames consist of
        # (self.num_maskmem - 2) nearest frames from every r-th frames, plus the last frame.
        memory_temporal_stride_for_eval=1,
        # whether to apply non-overlapping constraints on the object masks in the memory encoder during evaluation (to avoid/alleviate superposing masks)
        non_overlap_masks_for_mem_enc=False,
        # whether to cross-attend to object pointers from other frames (based on SAM output tokens) in the encoder
        use_obj_ptrs_in_encoder=False,
        # the maximum number of object pointers from other frames in encoder cross attention (only relevant when `use_obj_ptrs_in_encoder=True`)
        max_obj_ptrs_in_encoder=16,
        # whether to add temporal positional encoding to the object pointers in the encoder (only relevant when `use_obj_ptrs_in_encoder=True`)
        add_tpos_enc_to_obj_ptrs=True,
        # whether to add an extra linear projection layer for the temporal positional encoding in the object pointers to avoid potential interference
        # with spatial positional encoding (only relevant when both `use_obj_ptrs_in_encoder=True` and `add_tpos_enc_to_obj_ptrs=True`)
        proj_tpos_enc_in_obj_ptrs=False,
        # whether to use signed distance (instead of unsigned absolute distance) in the temporal positional encoding in the object pointers
        # (only relevant when both `use_obj_ptrs_in_encoder=True` and `add_tpos_enc_to_obj_ptrs=True`)
        use_signed_tpos_enc_to_obj_ptrs=False,
        # whether to only attend to object pointers in the past (before the current frame) in the encoder during evaluation
        # (only relevant when `use_obj_ptrs_in_encoder=True`; this might avoid pointer information too far in the future to distract the initial tracking)
        only_obj_ptrs_in_the_past_for_eval=False,
        # Whether to predict if there is an object in the frame
        pred_obj_scores: bool = False,
        # Whether to use an MLP to predict object scores
        pred_obj_scores_mlp: bool = False,
        # Only relevant if pred_obj_scores=True and use_obj_ptrs_in_encoder=True;
        # Whether to have a fixed no obj pointer when there is no object present
        # or to use it as an additive embedding with obj_ptr produced by decoder
        fixed_no_obj_ptr: bool = False,
        # Soft no object, i.e. mix in no_obj_ptr softly,
        # hope to make recovery easier if there is a mistake and mitigate accumulation of errors
        soft_no_obj_ptr: bool = False,
        use_mlp_for_obj_ptr_proj: bool = False,
        # add no obj embedding to spatial frames
        no_obj_embed_spatial: bool = False,
        # extra arguments used to construct the SAM mask decoder; if not None, it should be a dict of kwargs to be passed into `MaskDecoder` class.
        sam_mask_decoder_extra_args=None,
        compile_image_encoder: bool = False,
    ):
        super().__init__()

        # Part 1: the image backbone
        self.image_encoder = image_encoder
        # Use level 0, 1, 2 for high-res setting, or just level 2 for the default setting
        self.use_high_res_features_in_sam = use_high_res_features_in_sam
        self.num_feature_levels = 3 if use_high_res_features_in_sam else 1
        self.use_obj_ptrs_in_encoder = use_obj_ptrs_in_encoder
        self.max_obj_ptrs_in_encoder = max_obj_ptrs_in_encoder
        if use_obj_ptrs_in_encoder:
            # A conv layer to downsample the mask prompt to stride 4 (the same stride as
            # low-res SAM mask logits) and to change its scales from 0~1 to SAM logit scale,
            # so that it can be fed into the SAM mask decoder to generate a pointer.
            self.mask_downsample = torch.nn.Conv2d(1, 1, kernel_size=4, stride=4)
        self.add_tpos_enc_to_obj_ptrs = add_tpos_enc_to_obj_ptrs
        if proj_tpos_enc_in_obj_ptrs:
            assert add_tpos_enc_to_obj_ptrs  # these options need to be used together
        self.proj_tpos_enc_in_obj_ptrs = proj_tpos_enc_in_obj_ptrs
        self.use_signed_tpos_enc_to_obj_ptrs = use_signed_tpos_enc_to_obj_ptrs
        self.only_obj_ptrs_in_the_past_for_eval = only_obj_ptrs_in_the_past_for_eval

        # Part 2: memory attention to condition current frame's visual features
        # with memories (and obj ptrs) from past frames
        self.memory_attention = memory_attention
        self.hidden_dim = image_encoder.neck.d_model

        # Part 3: memory encoder for the previous frame's outputs
        self.memory_encoder = memory_encoder
        self.mem_dim = self.hidden_dim
        if hasattr(self.memory_encoder, "out_proj") and hasattr(
            self.memory_encoder.out_proj, "weight"
        ):
            # if there is compression of memories along channel dim
            self.mem_dim = self.memory_encoder.out_proj.weight.shape[0]
        self.num_maskmem = num_maskmem  # Number of memories accessible
        # Temporal encoding of the memories
        self.maskmem_tpos_enc = torch.nn.Parameter(
            torch.zeros(num_maskmem, 1, 1, self.mem_dim)
        )
        trunc_normal_(self.maskmem_tpos_enc, std=0.02)
        # a single token to indicate no memory embedding from previous frames
        self.no_mem_embed = torch.nn.Parameter(torch.zeros(1, 1, self.hidden_dim))
        self.no_mem_pos_enc = torch.nn.Parameter(torch.zeros(1, 1, self.hidden_dim))
        trunc_normal_(self.no_mem_embed, std=0.02)
        trunc_normal_(self.no_mem_pos_enc, std=0.02)
        self.directly_add_no_mem_embed = directly_add_no_mem_embed
        # Apply sigmoid to the output raw mask logits (to turn them from
        # range (-inf, +inf) to range (0, 1)) before feeding them into the memory encoder
        self.sigmoid_scale_for_mem_enc = sigmoid_scale_for_mem_enc
        self.sigmoid_bias_for_mem_enc = sigmoid_bias_for_mem_enc
        self.binarize_mask_from_pts_for_mem_enc = binarize_mask_from_pts_for_mem_enc
        self.non_overlap_masks_for_mem_enc = non_overlap_masks_for_mem_enc
        self.memory_temporal_stride_for_eval = memory_temporal_stride_for_eval
        # On frames with mask input, whether to directly output the input mask without
        # using a SAM prompt encoder + mask decoder
        self.use_mask_input_as_output_without_sam = use_mask_input_as_output_without_sam
        self.multimask_output_in_sam = multimask_output_in_sam
        self.multimask_min_pt_num = multimask_min_pt_num
        self.multimask_max_pt_num = multimask_max_pt_num
        self.multimask_output_for_tracking = multimask_output_for_tracking
        self.use_multimask_token_for_obj_ptr = use_multimask_token_for_obj_ptr
        self.iou_prediction_use_sigmoid = iou_prediction_use_sigmoid

        # Part 4: SAM-style prompt encoder (for both mask and point inputs)
        # and SAM-style mask decoder for the final mask output
        self.image_size = image_size
        self.backbone_stride = backbone_stride
        self.sam_mask_decoder_extra_args = sam_mask_decoder_extra_args
        self.pred_obj_scores = pred_obj_scores
        self.pred_obj_scores_mlp = pred_obj_scores_mlp
        self.fixed_no_obj_ptr = fixed_no_obj_ptr
        self.soft_no_obj_ptr = soft_no_obj_ptr
        if self.fixed_no_obj_ptr:
            assert self.pred_obj_scores
            assert self.use_obj_ptrs_in_encoder
        if self.pred_obj_scores and self.use_obj_ptrs_in_encoder:
            self.no_obj_ptr = torch.nn.Parameter(torch.zeros(1, self.hidden_dim))
            trunc_normal_(self.no_obj_ptr, std=0.02)
        self.use_mlp_for_obj_ptr_proj = use_mlp_for_obj_ptr_proj
        self.no_obj_embed_spatial = None
        if no_obj_embed_spatial:
            self.no_obj_embed_spatial = torch.nn.Parameter(torch.zeros(1, self.mem_dim))
            trunc_normal_(self.no_obj_embed_spatial, std=0.02)

        self._build_sam_heads()
        self.max_cond_frames_in_attn = max_cond_frames_in_attn

        # Model compilation
        if compile_image_encoder:
            # Compile the forward function (not the full module) to allow loading checkpoints.
            print(
                "Image encoder compilation is enabled. First forward pass will be slow."
            )
            self.image_encoder.forward = torch.compile(
                self.image_encoder.forward,
                mode="max-autotune",
                fullgraph=True,
                dynamic=False,
            )

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(self, *args, **kwargs):
        raise NotImplementedError(
            "Please use the corresponding methods in SAM2VideoPredictor for inference or SAM2Train for training/fine-tuning"
            "See notebooks/video_predictor_example.ipynb for an inference example."
        )

    def _build_sam_heads(self):
        """Build SAM-style prompt encoder and mask decoder."""
        self.sam_prompt_embed_dim = self.hidden_dim
        self.sam_image_embedding_size = self.image_size // self.backbone_stride

        # build PromptEncoder and MaskDecoder from SAM
        # (their hyperparameters like `mask_in_chans=16` are from SAM code)
        self.sam_prompt_encoder = PromptEncoder(
            embed_dim=self.sam_prompt_embed_dim,
            image_embedding_size=(
                self.sam_image_embedding_size,
                self.sam_image_embedding_size,
            ),
            input_image_size=(self.image_size, self.image_size),
            mask_in_chans=16,
        )
        self.sam_mask_decoder = MaskDecoder(
            num_multimask_outputs=3,
            transformer=TwoWayTransformer(
                depth=2,
                embedding_dim=self.sam_prompt_embed_dim,
                mlp_dim=2048,
                num_heads=8,
            ),
            transformer_dim=self.sam_prompt_embed_dim,
            iou_head_depth=3,
            iou_head_hidden_dim=256,
            use_high_res_features=self.use_high_res_features_in_sam,
            iou_prediction_use_sigmoid=self.iou_prediction_use_sigmoid,
            pred_obj_scores=self.pred_obj_scores,
            pred_obj_scores_mlp=self.pred_obj_scores_mlp,
            use_multimask_token_for_obj_ptr=self.use_multimask_token_for_obj_ptr,
            **(self.sam_mask_decoder_extra_args or {}),
        )
        if self.use_obj_ptrs_in_encoder:
            # a linear projection on SAM output tokens to turn them into object pointers
            self.obj_ptr_proj = torch.nn.Linear(self.hidden_dim, self.hidden_dim)
            if self.use_mlp_for_obj_ptr_proj:
                self.obj_ptr_proj = MLP(
                    self.hidden_dim, self.hidden_dim, self.hidden_dim, 3
                )
        else:
            self.obj_ptr_proj = torch.nn.Identity()
        if self.proj_tpos_enc_in_obj_ptrs:
            # a linear projection on temporal positional encoding in object pointers to
            # avoid potential interference with spatial positional encoding
            self.obj_ptr_tpos_proj = torch.nn.Linear(self.hidden_dim, self.mem_dim)
        else:
            self.obj_ptr_tpos_proj = torch.nn.Identity()

    def _forward_sam_heads(
        self,
        backbone_features,
        point_inputs=None,
        mask_inputs=None,
        high_res_features=None,
        multimask_output=False,
    ):
        """
        Forward SAM prompt encoders and mask heads.

        Inputs:
        - backbone_features: image features of [B, C, H, W] shape
        - point_inputs: a dictionary with "point_coords" and "point_labels", where
          1) "point_coords" has [B, P, 2] shape and float32 dtype and contains the
             absolute pixel-unit coordinate in (x, y) format of the P input points
          2) "point_labels" has shape [B, P] and int32 dtype, where 1 means
             positive clicks, 0 means negative clicks, and -1 means padding
        - mask_inputs: a mask of [B, 1, H*16, W*16] shape, float or bool, with the
          same spatial size as the image.
        - high_res_features: either 1) None or 2) or a list of length 2 containing
          two feature maps of [B, C, 4*H, 4*W] and [B, C, 2*H, 2*W] shapes respectively,
          which will be used as high-resolution feature maps for SAM decoder.
        - multimask_output: if it's True, we output 3 candidate masks and their 3
          corresponding IoU estimates, and if it's False, we output only 1 mask and
          its corresponding IoU estimate.

        Outputs:
        - low_res_multimasks: [B, M, H*4, W*4] shape (where M = 3 if
          `multimask_output=True` and M = 1 if `multimask_output=False`), the SAM
          output mask logits (before sigmoid) for the low-resolution masks, with 4x
          the resolution (1/4 stride) of the input backbone_features.
        - high_res_multimasks: [B, M, H*16, W*16] shape (where M = 3
          if `multimask_output=True` and M = 1 if `multimask_output=False`),
          upsampled from the low-resolution masks, with shape size as the image
          (stride is 1 pixel).
        - ious, [B, M] shape, where (where M = 3 if `multimask_output=True` and M = 1
          if `multimask_output=False`), the estimated IoU of each output mask.
        - low_res_masks: [B, 1, H*4, W*4] shape, the best mask in `low_res_multimasks`.
          If `multimask_output=True`, it's the mask with the highest IoU estimate.
          If `multimask_output=False`, it's the same as `low_res_multimasks`.
        - high_res_masks: [B, 1, H*16, W*16] shape, the best mask in `high_res_multimasks`.
          If `multimask_output=True`, it's the mask with the highest IoU estimate.
          If `multimask_output=False`, it's the same as `high_res_multimasks`.
        - obj_ptr: [B, C] shape, the object pointer vector for the output mask, extracted
          based on the output token from the SAM mask decoder.
        """
        B = backbone_features.size(0)
        device = backbone_features.device
        assert backbone_features.size(1) == self.sam_prompt_embed_dim
        assert backbone_features.size(2) == self.sam_image_embedding_size
        assert backbone_features.size(3) == self.sam_image_embedding_size

        # a) Handle point prompts
        if point_inputs is not None:
            sam_point_coords = point_inputs["point_coords"]
            sam_point_labels = point_inputs["point_labels"]
            assert sam_point_coords.size(0) == B and sam_point_labels.size(0) == B
        else:
            # If no points are provide, pad with an empty point (with label -1)
            sam_point_coords = torch.zeros(B, 1, 2, device=device)
            sam_point_labels = -torch.ones(B, 1, dtype=torch.int32, device=device)

        # b) Handle mask prompts
        if mask_inputs is not None:
            # If mask_inputs is provided, downsize it into low-res mask input if needed
            # and feed it as a dense mask prompt into the SAM mask encoder
            assert len(mask_inputs.shape) == 4 and mask_inputs.shape[:2] == (B, 1)
            if mask_inputs.shape[-2:] != self.sam_prompt_encoder.mask_input_size:
                sam_mask_prompt = F.interpolate(
                    mask_inputs.float(),
                    size=self.sam_prompt_encoder.mask_input_size,
                    align_corners=False,
                    mode="bilinear",
                    antialias=True,  # use antialias for downsampling
                )
            else:
                sam_mask_prompt = mask_inputs
        else:
            # Otherwise, simply feed None (and SAM's prompt encoder will add
            # a learned `no_mask_embed` to indicate no mask input in this case).
            sam_mask_prompt = None

        sparse_embeddings, dense_embeddings = self.sam_prompt_encoder(
            points=(sam_point_coords, sam_point_labels),
            boxes=None,
            masks=sam_mask_prompt,
        )
        (
            low_res_multimasks,
            ious,
            sam_output_tokens,
            object_score_logits,
        ) = self.sam_mask_decoder(
            image_embeddings=backbone_features,
            image_pe=self.sam_prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=multimask_output,
            repeat_image=False,  # the image is already batched
            high_res_features=high_res_features,
        )
        if self.pred_obj_scores:
            is_obj_appearing = object_score_logits > 0

            # Mask used for spatial memories is always a *hard* choice between obj and no obj,
            # consistent with the actual mask prediction
            low_res_multimasks = torch.where(
                is_obj_appearing[:, None, None],
                low_res_multimasks,
                NO_OBJ_SCORE,
            )

        # convert masks from possibly bfloat16 (or float16) to float32
        # (older PyTorch versions before 2.1 don't support `interpolate` on bf16)
        low_res_multimasks = low_res_multimasks.float()
        high_res_multimasks = F.interpolate(
            low_res_multimasks,
            size=(self.image_size, self.image_size),
            mode="bilinear",
            align_corners=False,
        )

        sam_output_token = sam_output_tokens[:, 0]
        if multimask_output:
            # take the best mask prediction (with the highest IoU estimation)
            best_iou_inds = torch.argmax(ious, dim=-1)
            batch_inds = torch.arange(B, device=device)
            low_res_masks = low_res_multimasks[batch_inds, best_iou_inds].unsqueeze(1)
            high_res_masks = high_res_multimasks[batch_inds, best_iou_inds].unsqueeze(1)
            if sam_output_tokens.size(1) > 1:
                sam_output_token = sam_output_tokens[batch_inds, best_iou_inds]
        else:
            low_res_masks, high_res_masks = low_res_multimasks, high_res_multimasks

        # Extract object pointer from the SAM output token (with occlusion handling)
        obj_ptr = self.obj_ptr_proj(sam_output_token)
        if self.pred_obj_scores:
            # Allow *soft* no obj ptr, unlike for masks
            if self.soft_no_obj_ptr:
                lambda_is_obj_appearing = object_score_logits.sigmoid()
            else:
                lambda_is_obj_appearing = is_obj_appearing.float()

            if self.fixed_no_obj_ptr:
                obj_ptr = lambda_is_obj_appearing * obj_ptr
            obj_ptr = obj_ptr + (1 - lambda_is_obj_appearing) * self.no_obj_ptr

        return (
            low_res_multimasks,
            high_res_multimasks,
            ious,
            low_res_masks,
            high_res_masks,
            obj_ptr,
            object_score_logits,
        )

    def _use_mask_as_output(self, backbone_features, high_res_features, mask_inputs):
        """
        Directly turn binary `mask_inputs` into a output mask logits without using SAM.
        (same input and output shapes as in _forward_sam_heads above).
        """
        # Use -10/+10 as logits for neg/pos pixels (very close to 0/1 in prob after sigmoid).
        out_scale, out_bias = 20.0, -10.0  # sigmoid(-10.0)=4.5398e-05
        mask_inputs_float = mask_inputs.float()
        high_res_masks = mask_inputs_float * out_scale + out_bias
        low_res_masks = F.interpolate(
            high_res_masks,
            size=(high_res_masks.size(-2) // 4, high_res_masks.size(-1) // 4),
            align_corners=False,
            mode="bilinear",
            antialias=True,  # use antialias for downsampling
        )
        # a dummy IoU prediction of all 1's under mask input
        ious = mask_inputs.new_ones(mask_inputs.size(0), 1).float()
        if not self.use_obj_ptrs_in_encoder:
            # all zeros as a dummy object pointer (of shape [B, C])
            obj_ptr = torch.zeros(
                mask_inputs.size(0), self.hidden_dim, device=mask_inputs.device
            )
        else:
            # produce an object pointer using the SAM decoder from the mask input
            _, _, _, _, _, obj_ptr, _ = self._forward_sam_heads(
                backbone_features=backbone_features,
                mask_inputs=self.mask_downsample(mask_inputs_float),
                high_res_features=high_res_features,
            )
        # In this method, we are treating mask_input as output, e.g. using it directly to create spatial mem;
        # Below, we follow the same design axiom to use mask_input to decide if obj appears or not instead of relying
        # on the object_scores from the SAM decoder.
        is_obj_appearing = torch.any(mask_inputs.flatten(1).float() > 0.0, dim=1)
        is_obj_appearing = is_obj_appearing[..., None]
        lambda_is_obj_appearing = is_obj_appearing.float()
        object_score_logits = out_scale * lambda_is_obj_appearing + out_bias
        if self.pred_obj_scores:
            if self.fixed_no_obj_ptr:
                obj_ptr = lambda_is_obj_appearing * obj_ptr
            obj_ptr = obj_ptr + (1 - lambda_is_obj_appearing) * self.no_obj_ptr

        return (
            low_res_masks,
            high_res_masks,
            ious,
            low_res_masks,
            high_res_masks,
            obj_ptr,
            object_score_logits,
        )

    def forward_image(self, img_batch: torch.Tensor):
        """Get the image feature on the input batch."""
        backbone_out = self.image_encoder(img_batch)
        if self.use_high_res_features_in_sam:
            # precompute projected level 0 and level 1 features in SAM decoder
            # to avoid running it again on every SAM click
            backbone_out["backbone_fpn"][0] = self.sam_mask_decoder.conv_s0(
                backbone_out["backbone_fpn"][0]
            )
            backbone_out["backbone_fpn"][1] = self.sam_mask_decoder.conv_s1(
                backbone_out["backbone_fpn"][1]
            )
        return backbone_out

    def _prepare_backbone_features(self, backbone_out):
        """Prepare and flatten visual features."""
        backbone_out = backbone_out.copy()
        assert len(backbone_out["backbone_fpn"]) == len(backbone_out["vision_pos_enc"])
        assert len(backbone_out["backbone_fpn"]) >= self.num_feature_levels

        feature_maps = backbone_out["backbone_fpn"][-self.num_feature_levels :]
        vision_pos_embeds = backbone_out["vision_pos_enc"][-self.num_feature_levels :]

        feat_sizes = [(x.shape[-2], x.shape[-1]) for x in vision_pos_embeds]
        # flatten NxCxHxW to HWxNxC
        vision_feats = [x.flatten(2).permute(2, 0, 1) for x in feature_maps]
        vision_pos_embeds = [x.flatten(2).permute(2, 0, 1) for x in vision_pos_embeds]

        return backbone_out, vision_feats, vision_pos_embeds, feat_sizes

    def _prepare_memory_conditioned_features(
        self,
        frame_idx,
        is_init_cond_frame,
        current_vision_feats,
        current_vision_pos_embeds,
        feat_sizes,
        output_dict,
        num_frames,
        track_in_reverse=False,  # tracking in reverse time order (for demo usage)
    ):
        """Fuse the current frame's visual feature map with previous memory."""
        B = current_vision_feats[-1].size(1)  # batch size on this frame
        C = self.hidden_dim
        H, W = feat_sizes[-1]  # top-level (lowest-resolution) feature size
        device = current_vision_feats[-1].device
        # The case of `self.num_maskmem == 0` below is primarily used for reproducing SAM on images.
        # In this case, we skip the fusion with any memory.
        if self.num_maskmem == 0:  # Disable memory and skip fusion
            pix_feat = current_vision_feats[-1].permute(1, 2, 0).view(B, C, H, W)
            return pix_feat

        num_obj_ptr_tokens = 0
        tpos_sign_mul = -1 if track_in_reverse else 1
        # Step 1: condition the visual features of the current frame on previous memories
        if not is_init_cond_frame:
            # Retrieve the memories encoded with the maskmem backbone
            to_cat_memory, to_cat_memory_pos_embed = [], []
            # Add conditioning frames's output first (all cond frames have t_pos=0 for
            # when getting temporal positional embedding below)
            assert len(output_dict["cond_frame_outputs"]) > 0
            # Select a maximum number of temporally closest cond frames for cross attention
            cond_outputs = output_dict["cond_frame_outputs"]
            selected_cond_outputs, unselected_cond_outputs = select_closest_cond_frames(
                frame_idx, cond_outputs, self.max_cond_frames_in_attn
            )
            t_pos_and_prevs = [(0, out) for out in selected_cond_outputs.values()]
            # Add last (self.num_maskmem - 1) frames before current frame for non-conditioning memory
            # the earliest one has t_pos=1 and the latest one has t_pos=self.num_maskmem-1
            # We also allow taking the memory frame non-consecutively (with stride>1), in which case
            # we take (self.num_maskmem - 2) frames among every stride-th frames plus the last frame.
            stride = 1 if self.training else self.memory_temporal_stride_for_eval
            for t_pos in range(1, self.num_maskmem):
                t_rel = self.num_maskmem - t_pos  # how many frames before current frame
                if t_rel == 1:
                    # for t_rel == 1, we take the last frame (regardless of r)
                    if not track_in_reverse:
                        # the frame immediately before this frame (i.e. frame_idx - 1)
                        prev_frame_idx = frame_idx - t_rel
                    else:
                        # the frame immediately after this frame (i.e. frame_idx + 1)
                        prev_frame_idx = frame_idx + t_rel
                else:
                    # for t_rel >= 2, we take the memory frame from every r-th frames
                    if not track_in_reverse:
                        # first find the nearest frame among every r-th frames before this frame
                        # for r=1, this would be (frame_idx - 2)
                        prev_frame_idx = ((frame_idx - 2) // stride) * stride
                        # then seek further among every r-th frames
                        prev_frame_idx = prev_frame_idx - (t_rel - 2) * stride
                    else:
                        # first find the nearest frame among every r-th frames after this frame
                        # for r=1, this would be (frame_idx + 2)
                        prev_frame_idx = -(-(frame_idx + 2) // stride) * stride
                        # then seek further among every r-th frames
                        prev_frame_idx = prev_frame_idx + (t_rel - 2) * stride
                out = output_dict["non_cond_frame_outputs"].get(prev_frame_idx, None)
                if out is None:
                    # If an unselected conditioning frame is among the last (self.num_maskmem - 1)
                    # frames, we still attend to it as if it's a non-conditioning frame.
                    out = unselected_cond_outputs.get(prev_frame_idx, None)
                t_pos_and_prevs.append((t_pos, out))

            for t_pos, prev in t_pos_and_prevs:
                if prev is None:
                    continue  # skip padding frames
                # "maskmem_features" might have been offloaded to CPU in demo use cases,
                # so we load it back to GPU (it's a no-op if it's already on GPU).
                feats = prev["maskmem_features"].to(device, non_blocking=True)
                to_cat_memory.append(feats.flatten(2).permute(2, 0, 1))
                # Spatial positional encoding (it might have been offloaded to CPU in eval)
                maskmem_enc = prev["maskmem_pos_enc"][-1].to(device)
                maskmem_enc = maskmem_enc.flatten(2).permute(2, 0, 1)
                # Temporal positional encoding
                maskmem_enc = (
                    maskmem_enc + self.maskmem_tpos_enc[self.num_maskmem - t_pos - 1]
                )
                to_cat_memory_pos_embed.append(maskmem_enc)

            # Construct the list of past object pointers
            if self.use_obj_ptrs_in_encoder:
                max_obj_ptrs_in_encoder = min(num_frames, self.max_obj_ptrs_in_encoder)
                # First add those object pointers from selected conditioning frames
                # (optionally, only include object pointers in the past during evaluation)
                if not self.training and self.only_obj_ptrs_in_the_past_for_eval:
                    ptr_cond_outputs = {
                        t: out
                        for t, out in selected_cond_outputs.items()
                        if (t >= frame_idx if track_in_reverse else t <= frame_idx)
                    }
                else:
                    ptr_cond_outputs = selected_cond_outputs
                pos_and_ptrs = [
                    # Temporal pos encoding contains how far away each pointer is from current frame
                    (
                        (
                            (frame_idx - t) * tpos_sign_mul
                            if self.use_signed_tpos_enc_to_obj_ptrs
                            else abs(frame_idx - t)
                        ),
                        out["obj_ptr"],
                    )
                    for t, out in ptr_cond_outputs.items()
                ]
                # Add up to (max_obj_ptrs_in_encoder - 1) non-conditioning frames before current frame
                for t_diff in range(1, max_obj_ptrs_in_encoder):
                    t = frame_idx + t_diff if track_in_reverse else frame_idx - t_diff
                    if t < 0 or (num_frames is not None and t >= num_frames):
                        break
                    out = output_dict["non_cond_frame_outputs"].get(
                        t, unselected_cond_outputs.get(t, None)
                    )
                    if out is not None:
                        pos_and_ptrs.append((t_diff, out["obj_ptr"]))
                # If we have at least one object pointer, add them to the across attention
                if len(pos_and_ptrs) > 0:
                    pos_list, ptrs_list = zip(*pos_and_ptrs)
                    # stack object pointers along dim=0 into [ptr_seq_len, B, C] shape
                    obj_ptrs = torch.stack(ptrs_list, dim=0)
                    # a temporal positional embedding based on how far each object pointer is from
                    # the current frame (sine embedding normalized by the max pointer num).
                    if self.add_tpos_enc_to_obj_ptrs:
                        t_diff_max = max_obj_ptrs_in_encoder - 1
                        tpos_dim = C if self.proj_tpos_enc_in_obj_ptrs else self.mem_dim
                        obj_pos = torch.tensor(pos_list, device=device)
                        obj_pos = get_1d_sine_pe(obj_pos / t_diff_max, dim=tpos_dim)
                        obj_pos = self.obj_ptr_tpos_proj(obj_pos)
                        obj_pos = obj_pos.unsqueeze(1).expand(-1, B, self.mem_dim)
                    else:
                        obj_pos = obj_ptrs.new_zeros(len(pos_list), B, self.mem_dim)
                    if self.mem_dim < C:
                        # split a pointer into (C // self.mem_dim) tokens for self.mem_dim < C
                        obj_ptrs = obj_ptrs.reshape(
                            -1, B, C // self.mem_dim, self.mem_dim
                        )
                        obj_ptrs = obj_ptrs.permute(0, 2, 1, 3).flatten(0, 1)
                        obj_pos = obj_pos.repeat_interleave(C // self.mem_dim, dim=0)
                    to_cat_memory.append(obj_ptrs)
                    to_cat_memory_pos_embed.append(obj_pos)
                    num_obj_ptr_tokens = obj_ptrs.shape[0]
                else:
                    num_obj_ptr_tokens = 0
        else:
            # for initial conditioning frames, encode them without using any previous memory
            if self.directly_add_no_mem_embed:
                # directly add no-mem embedding (instead of using the transformer encoder)
                pix_feat_with_mem = current_vision_feats[-1] + self.no_mem_embed
                pix_feat_with_mem = pix_feat_with_mem.permute(1, 2, 0).view(B, C, H, W)
                return pix_feat_with_mem

            # Use a dummy token on the first frame (to avoid empty memory input to tranformer encoder)
            to_cat_memory = [self.no_mem_embed.expand(1, B, self.mem_dim)]
            to_cat_memory_pos_embed = [self.no_mem_pos_enc.expand(1, B, self.mem_dim)]

        # Step 2: Concatenate the memories and forward through the transformer encoder
        memory = torch.cat(to_cat_memory, dim=0)
        memory_pos_embed = torch.cat(to_cat_memory_pos_embed, dim=0)

        pix_feat_with_mem = self.memory_attention(
            curr=current_vision_feats,
            curr_pos=current_vision_pos_embeds,
            memory=memory,
            memory_pos=memory_pos_embed,
            num_obj_ptr_tokens=num_obj_ptr_tokens,
        )
        # reshape the output (HW)BC => BCHW
        pix_feat_with_mem = pix_feat_with_mem.permute(1, 2, 0).view(B, C, H, W)
        return pix_feat_with_mem

    def _encode_new_memory(
        self,
        current_vision_feats,
        feat_sizes,
        pred_masks_high_res,
        object_score_logits,
        is_mask_from_pts,
    ):
        """Encode the current image and its prediction into a memory feature."""
        B = current_vision_feats[-1].size(1)  # batch size on this frame
        C = self.hidden_dim
        H, W = feat_sizes[-1]  # top-level (lowest-resolution) feature size
        # top-level feature, (HW)BC => BCHW
        pix_feat = current_vision_feats[-1].permute(1, 2, 0).view(B, C, H, W)
        if self.non_overlap_masks_for_mem_enc and not self.training:
            # optionally, apply non-overlapping constraints to the masks (it's applied
            # in the batch dimension and should only be used during eval, where all
            # the objects come from the same video under batch size 1).
            pred_masks_high_res = self._apply_non_overlapping_constraints(
                pred_masks_high_res
            )
        # scale the raw mask logits with a temperature before applying sigmoid
        binarize = self.binarize_mask_from_pts_for_mem_enc and is_mask_from_pts
        if binarize and not self.training:
            mask_for_mem = (pred_masks_high_res > 0).float()
        else:
            # apply sigmoid on the raw mask logits to turn them into range (0, 1)
            mask_for_mem = torch.sigmoid(pred_masks_high_res)
        # apply scale and bias terms to the sigmoid probabilities
        if self.sigmoid_scale_for_mem_enc != 1.0:
            mask_for_mem = mask_for_mem * self.sigmoid_scale_for_mem_enc
        if self.sigmoid_bias_for_mem_enc != 0.0:
            mask_for_mem = mask_for_mem + self.sigmoid_bias_for_mem_enc
        maskmem_out = self.memory_encoder(
            pix_feat, mask_for_mem, skip_mask_sigmoid=True  # sigmoid already applied
        )
        maskmem_features = maskmem_out["vision_features"]
        maskmem_pos_enc = maskmem_out["vision_pos_enc"]
        # add a no-object embedding to the spatial memory to indicate that the frame
        # is predicted to be occluded (i.e. no object is appearing in the frame)
        if self.no_obj_embed_spatial is not None:
            is_obj_appearing = (object_score_logits > 0).float()
            maskmem_features += (
                1 - is_obj_appearing[..., None, None]
            ) * self.no_obj_embed_spatial[..., None, None].expand(
                *maskmem_features.shape
            )

        return maskmem_features, maskmem_pos_enc

    def _track_step(
        self,
        frame_idx,
        is_init_cond_frame,
        current_vision_feats,
        current_vision_pos_embeds,
        feat_sizes,
        point_inputs,
        mask_inputs,
        output_dict,
        num_frames,
        track_in_reverse,
        prev_sam_mask_logits,
    ):
        current_out = {"point_inputs": point_inputs, "mask_inputs": mask_inputs}
        # High-resolution feature maps for the SAM head, reshape (HW)BC => BCHW
        if len(current_vision_feats) > 1:
            high_res_features = [
                x.permute(1, 2, 0).view(x.size(1), x.size(2), *s)
                for x, s in zip(current_vision_feats[:-1], feat_sizes[:-1])
            ]
        else:
            high_res_features = None
        if mask_inputs is not None and self.use_mask_input_as_output_without_sam:
            # When use_mask_input_as_output_without_sam=True, we directly output the mask input
            # (see it as a GT mask) without using a SAM prompt encoder + mask decoder.
            pix_feat = current_vision_feats[-1].permute(1, 2, 0)
            pix_feat = pix_feat.view(-1, self.hidden_dim, *feat_sizes[-1])
            sam_outputs = self._use_mask_as_output(
                pix_feat, high_res_features, mask_inputs
            )
        else:
            # fused the visual feature with previous memory features in the memory bank
            pix_feat = self._prepare_memory_conditioned_features(
                frame_idx=frame_idx,
                is_init_cond_frame=is_init_cond_frame,
                current_vision_feats=current_vision_feats[-1:],
                current_vision_pos_embeds=current_vision_pos_embeds[-1:],
                feat_sizes=feat_sizes[-1:],
                output_dict=output_dict,
                num_frames=num_frames,
                track_in_reverse=track_in_reverse,
            )
            # apply SAM-style segmentation head
            # here we might feed previously predicted low-res SAM mask logits into the SAM mask decoder,
            # e.g. in demo where such logits come from earlier interaction instead of correction sampling
            # (in this case, any `mask_inputs` shouldn't reach here as they are sent to _use_mask_as_output instead)
            if prev_sam_mask_logits is not None:
                assert point_inputs is not None and mask_inputs is None
                mask_inputs = prev_sam_mask_logits
            multimask_output = self._use_multimask(is_init_cond_frame, point_inputs)
            sam_outputs = self._forward_sam_heads(
                backbone_features=pix_feat,
                point_inputs=point_inputs,
                mask_inputs=mask_inputs,
                high_res_features=high_res_features,
                multimask_output=multimask_output,
            )

        return current_out, sam_outputs, high_res_features, pix_feat

    def _encode_memory_in_output(
        self,
        current_vision_feats,
        feat_sizes,
        point_inputs,
        run_mem_encoder,
        high_res_masks,
        object_score_logits,
        current_out,
    ):
        if run_mem_encoder and self.num_maskmem > 0:
            high_res_masks_for_mem_enc = high_res_masks
            maskmem_features, maskmem_pos_enc = self._encode_new_memory(
                current_vision_feats=current_vision_feats,
                feat_sizes=feat_sizes,
                pred_masks_high_res=high_res_masks_for_mem_enc,
                object_score_logits=object_score_logits,
                is_mask_from_pts=(point_inputs is not None),
            )
            current_out["maskmem_features"] = maskmem_features
            current_out["maskmem_pos_enc"] = maskmem_pos_enc
        else:
            current_out["maskmem_features"] = None
            current_out["maskmem_pos_enc"] = None

    def track_step(
        self,
        frame_idx,
        is_init_cond_frame,
        current_vision_feats,
        current_vision_pos_embeds,
        feat_sizes,
        point_inputs,
        mask_inputs,
        output_dict,
        num_frames,
        track_in_reverse=False,  # tracking in reverse time order (for demo usage)
        # Whether to run the memory encoder on the predicted masks. Sometimes we might want
        # to skip the memory encoder with `run_mem_encoder=False`. For example,
        # in demo we might call `track_step` multiple times for each user click,
        # and only encode the memory when the user finalizes their clicks. And in ablation
        # settings like SAM training on static images, we don't need the memory encoder.
        run_mem_encoder=True,
        # The previously predicted SAM mask logits (which can be fed together with new clicks in demo).
        prev_sam_mask_logits=None,
    ):
        current_out, sam_outputs, _, _ = self._track_step(
            frame_idx,
            is_init_cond_frame,
            current_vision_feats,
            current_vision_pos_embeds,
            feat_sizes,
            point_inputs,
            mask_inputs,
            output_dict,
            num_frames,
            track_in_reverse,
            prev_sam_mask_logits,
        )

        (
            _,
            _,
            _,
            low_res_masks,
            high_res_masks,
            obj_ptr,
            object_score_logits,
        ) = sam_outputs

        current_out["pred_masks"] = low_res_masks
        current_out["pred_masks_high_res"] = high_res_masks
        current_out["obj_ptr"] = obj_ptr
        if not self.training:
            # Only add this in inference (to avoid unused param in activation checkpointing;
            # it's mainly used in the demo to encode spatial memories w/ consolidated masks)
            current_out["object_score_logits"] = object_score_logits

        # Finally run the memory encoder on the predicted mask to encode
        # it into a new memory feature (that can be used in future frames)
        self._encode_memory_in_output(
            current_vision_feats,
            feat_sizes,
            point_inputs,
            run_mem_encoder,
            high_res_masks,
            object_score_logits,
            current_out,
        )

        return current_out

    def _use_multimask(self, is_init_cond_frame, point_inputs):
        """Whether to use multimask output in the SAM head."""
        num_pts = 0 if point_inputs is None else point_inputs["point_labels"].size(1)
        multimask_output = (
            self.multimask_output_in_sam
            and (is_init_cond_frame or self.multimask_output_for_tracking)
            and (self.multimask_min_pt_num <= num_pts <= self.multimask_max_pt_num)
        )
        return multimask_output

    def _apply_non_overlapping_constraints(self, pred_masks):
        """
        Apply non-overlapping constraints to the object scores in pred_masks. Here we
        keep only the highest scoring object at each spatial location in pred_masks.
        """
        batch_size = pred_masks.size(0)
        if batch_size == 1:
            return pred_masks

        device = pred_masks.device
        # "max_obj_inds": object index of the object with the highest score at each location
        max_obj_inds = torch.argmax(pred_masks, dim=0, keepdim=True)
        # "batch_obj_inds": object index of each object slice (along dim 0) in `pred_masks`
        batch_obj_inds = torch.arange(batch_size, device=device)[:, None, None, None]
        keep = max_obj_inds == batch_obj_inds
        # suppress overlapping regions' scores below -10.0 so that the foreground regions
        # don't overlap (here sigmoid(-10.0)=4.5398e-05)
        pred_masks = torch.where(keep, pred_masks, torch.clamp(pred_masks, max=-10.0))
        return pred_masks
'''

# Native module: test-code/source/MedSAM2-main/sam2/modeling/sam2_utils.py
_EMBEDDED_SOURCES['test-code/source/MedSAM2-main/sam2/modeling/sam2_utils.py'] = '''# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import copy
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from sam2.utils.misc import mask_to_box


def select_closest_cond_frames(frame_idx, cond_frame_outputs, max_cond_frame_num):
    """
    Select up to `max_cond_frame_num` conditioning frames from `cond_frame_outputs`
    that are temporally closest to the current frame at `frame_idx`. Here, we take
    - a) the closest conditioning frame before `frame_idx` (if any);
    - b) the closest conditioning frame after `frame_idx` (if any);
    - c) any other temporally closest conditioning frames until reaching a total
         of `max_cond_frame_num` conditioning frames.

    Outputs:
    - selected_outputs: selected items (keys & values) from `cond_frame_outputs`.
    - unselected_outputs: items (keys & values) not selected in `cond_frame_outputs`.
    """
    if max_cond_frame_num == -1 or len(cond_frame_outputs) <= max_cond_frame_num:
        selected_outputs = cond_frame_outputs
        unselected_outputs = {}
    else:
        assert max_cond_frame_num >= 2, "we should allow using 2+ conditioning frames"
        selected_outputs = {}

        # the closest conditioning frame before `frame_idx` (if any)
        idx_before = max((t for t in cond_frame_outputs if t < frame_idx), default=None)
        if idx_before is not None:
            selected_outputs[idx_before] = cond_frame_outputs[idx_before]

        # the closest conditioning frame after `frame_idx` (if any)
        idx_after = min((t for t in cond_frame_outputs if t >= frame_idx), default=None)
        if idx_after is not None:
            selected_outputs[idx_after] = cond_frame_outputs[idx_after]

        # add other temporally closest conditioning frames until reaching a total
        # of `max_cond_frame_num` conditioning frames.
        num_remain = max_cond_frame_num - len(selected_outputs)
        inds_remain = sorted(
            (t for t in cond_frame_outputs if t not in selected_outputs),
            key=lambda x: abs(x - frame_idx),
        )[:num_remain]
        selected_outputs.update((t, cond_frame_outputs[t]) for t in inds_remain)
        unselected_outputs = {
            t: v for t, v in cond_frame_outputs.items() if t not in selected_outputs
        }

    return selected_outputs, unselected_outputs


def get_1d_sine_pe(pos_inds, dim, temperature=10000):
    """
    Get 1D sine positional embedding as in the original Transformer paper.
    """
    pe_dim = dim // 2
    dim_t = torch.arange(pe_dim, dtype=torch.float32, device=pos_inds.device)
    dim_t = temperature ** (2 * (dim_t // 2) / pe_dim)

    pos_embed = pos_inds.unsqueeze(-1) / dim_t
    pos_embed = torch.cat([pos_embed.sin(), pos_embed.cos()], dim=-1)
    return pos_embed


def get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(f"activation should be relu/gelu, not {activation}.")


def get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


class DropPath(nn.Module):
    # adapted from https://github.com/huggingface/pytorch-image-models/blob/main/timm/layers/drop.py
    def __init__(self, drop_prob=0.0, scale_by_keep=True):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x):
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
        if keep_prob > 0.0 and self.scale_by_keep:
            random_tensor.div_(keep_prob)
        return x * random_tensor


# Lightly adapted from
# https://github.com/facebookresearch/MaskFormer/blob/main/mask_former/modeling/transformer/transformer_predictor.py # noqa
class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int,
        activation: nn.Module = nn.ReLU,
        sigmoid_output: bool = False,
    ) -> None:
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )
        self.sigmoid_output = sigmoid_output
        self.act = activation()

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = self.act(layer(x)) if i < self.num_layers - 1 else layer(x)
        if self.sigmoid_output:
            x = F.sigmoid(x)
        return x


# From https://github.com/facebookresearch/detectron2/blob/main/detectron2/layers/batch_norm.py # noqa
# Itself from https://github.com/facebookresearch/ConvNeXt/blob/d1fa8f6fef0a165b27399986cc2bdacc92777e40/models/convnext.py#L119  # noqa
class LayerNorm2d(nn.Module):
    def __init__(self, num_channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x


def sample_box_points(
    masks: torch.Tensor,
    noise: float = 0.1,  # SAM default
    noise_bound: int = 20,  # SAM default
    top_left_label: int = 2,
    bottom_right_label: int = 3,
) -> Tuple[np.array, np.array]:
    """
    Sample a noised version of the top left and bottom right corners of a given `bbox`

    Inputs:
    - masks: [B, 1, H,W] boxes, dtype=torch.Tensor
    - noise: noise as a fraction of box width and height, dtype=float
    - noise_bound: maximum amount of noise (in pure pixesl), dtype=int

    Returns:
    - box_coords: [B, num_pt, 2], contains (x, y) coordinates of top left and bottom right box corners, dtype=torch.float
    - box_labels: [B, num_pt], label 2 is reserverd for top left and 3 for bottom right corners, dtype=torch.int32
    """
    device = masks.device
    box_coords = mask_to_box(masks)
    B, _, H, W = masks.shape
    box_labels = torch.tensor(
        [top_left_label, bottom_right_label], dtype=torch.int, device=device
    ).repeat(B)
    if noise > 0.0:
        if not isinstance(noise_bound, torch.Tensor):
            noise_bound = torch.tensor(noise_bound, device=device)
        bbox_w = box_coords[..., 2] - box_coords[..., 0]
        bbox_h = box_coords[..., 3] - box_coords[..., 1]
        max_dx = torch.min(bbox_w * noise, noise_bound)
        max_dy = torch.min(bbox_h * noise, noise_bound)
        box_noise = 2 * torch.rand(B, 1, 4, device=device) - 1
        box_noise = box_noise * torch.stack((max_dx, max_dy, max_dx, max_dy), dim=-1)

        box_coords = box_coords + box_noise
        img_bounds = (
            torch.tensor([W, H, W, H], device=device) - 1
        )  # uncentered pixel coords
        box_coords.clamp_(torch.zeros_like(img_bounds), img_bounds)  # In place clamping

    box_coords = box_coords.reshape(-1, 2, 2)  # always 2 points
    box_labels = box_labels.reshape(-1, 2)
    return box_coords, box_labels


def sample_random_points_from_errors(gt_masks, pred_masks, num_pt=1):
    """
    Sample `num_pt` random points (along with their labels) independently from the error regions.

    Inputs:
    - gt_masks: [B, 1, H_im, W_im] masks, dtype=torch.bool
    - pred_masks: [B, 1, H_im, W_im] masks, dtype=torch.bool or None
    - num_pt: int, number of points to sample independently for each of the B error maps

    Outputs:
    - points: [B, num_pt, 2], dtype=torch.float, contains (x, y) coordinates of each sampled point
    - labels: [B, num_pt], dtype=torch.int32, where 1 means positive clicks and 0 means
      negative clicks
    """
    if pred_masks is None:  # if pred_masks is not provided, treat it as empty
        pred_masks = torch.zeros_like(gt_masks)
    assert gt_masks.dtype == torch.bool and gt_masks.size(1) == 1
    assert pred_masks.dtype == torch.bool and pred_masks.shape == gt_masks.shape
    assert num_pt >= 0

    B, _, H_im, W_im = gt_masks.shape
    device = gt_masks.device

    # false positive region, a new point sampled in this region should have
    # negative label to correct the FP error
    fp_masks = ~gt_masks & pred_masks
    # false negative region, a new point sampled in this region should have
    # positive label to correct the FN error
    fn_masks = gt_masks & ~pred_masks
    # whether the prediction completely match the ground-truth on each mask
    all_correct = torch.all((gt_masks == pred_masks).flatten(2), dim=2)
    all_correct = all_correct[..., None, None]

    # channel 0 is FP map, while channel 1 is FN map
    pts_noise = torch.rand(B, num_pt, H_im, W_im, 2, device=device)
    # sample a negative new click from FP region or a positive new click
    # from FN region, depend on where the maximum falls,
    # and in case the predictions are all correct (no FP or FN), we just
    # sample a negative click from the background region
    pts_noise[..., 0] *= fp_masks | (all_correct & ~gt_masks)
    pts_noise[..., 1] *= fn_masks
    pts_idx = pts_noise.flatten(2).argmax(dim=2)
    labels = (pts_idx % 2).to(torch.int32)
    pts_idx = pts_idx // 2
    pts_x = pts_idx % W_im
    pts_y = pts_idx // W_im
    points = torch.stack([pts_x, pts_y], dim=2).to(torch.float)
    return points, labels


def sample_one_point_from_error_center(gt_masks, pred_masks, padding=True):
    """
    Sample 1 random point (along with its label) from the center of each error region,
    that is, the point with the largest distance to the boundary of each error region.
    This is the RITM sampling method from https://github.com/saic-vul/ritm_interactive_segmentation/blob/master/isegm/inference/clicker.py

    Inputs:
    - gt_masks: [B, 1, H_im, W_im] masks, dtype=torch.bool
    - pred_masks: [B, 1, H_im, W_im] masks, dtype=torch.bool or None
    - padding: if True, pad with boundary of 1 px for distance transform

    Outputs:
    - points: [B, 1, 2], dtype=torch.float, contains (x, y) coordinates of each sampled point
    - labels: [B, 1], dtype=torch.int32, where 1 means positive clicks and 0 means negative clicks
    """
    import cv2

    if pred_masks is None:
        pred_masks = torch.zeros_like(gt_masks)
    assert gt_masks.dtype == torch.bool and gt_masks.size(1) == 1
    assert pred_masks.dtype == torch.bool and pred_masks.shape == gt_masks.shape

    B, _, _, W_im = gt_masks.shape
    device = gt_masks.device

    # false positive region, a new point sampled in this region should have
    # negative label to correct the FP error
    fp_masks = ~gt_masks & pred_masks
    # false negative region, a new point sampled in this region should have
    # positive label to correct the FN error
    fn_masks = gt_masks & ~pred_masks

    fp_masks = fp_masks.cpu().numpy()
    fn_masks = fn_masks.cpu().numpy()
    points = torch.zeros(B, 1, 2, dtype=torch.float)
    labels = torch.ones(B, 1, dtype=torch.int32)
    for b in range(B):
        fn_mask = fn_masks[b, 0]
        fp_mask = fp_masks[b, 0]
        if padding:
            fn_mask = np.pad(fn_mask, ((1, 1), (1, 1)), "constant")
            fp_mask = np.pad(fp_mask, ((1, 1), (1, 1)), "constant")
        # compute the distance of each point in FN/FP region to its boundary
        fn_mask_dt = cv2.distanceTransform(fn_mask.astype(np.uint8), cv2.DIST_L2, 0)
        fp_mask_dt = cv2.distanceTransform(fp_mask.astype(np.uint8), cv2.DIST_L2, 0)
        if padding:
            fn_mask_dt = fn_mask_dt[1:-1, 1:-1]
            fp_mask_dt = fp_mask_dt[1:-1, 1:-1]

        # take the point in FN/FP region with the largest distance to its boundary
        fn_mask_dt_flat = fn_mask_dt.reshape(-1)
        fp_mask_dt_flat = fp_mask_dt.reshape(-1)
        fn_argmax = np.argmax(fn_mask_dt_flat)
        fp_argmax = np.argmax(fp_mask_dt_flat)
        is_positive = fn_mask_dt_flat[fn_argmax] > fp_mask_dt_flat[fp_argmax]
        pt_idx = fn_argmax if is_positive else fp_argmax
        points[b, 0, 0] = pt_idx % W_im  # x
        points[b, 0, 1] = pt_idx // W_im  # y
        labels[b, 0] = int(is_positive)

    points = points.to(device)
    labels = labels.to(device)
    return points, labels


def get_next_point(gt_masks, pred_masks, method):
    if method == "uniform":
        return sample_random_points_from_errors(gt_masks, pred_masks)
    elif method == "center":
        return sample_one_point_from_error_center(gt_masks, pred_masks)
    else:
        raise ValueError(f"unknown sampling method {method}")
'''

# Native module: test-code/source/MedSAM2-main/sam2/sam2_image_predictor.py
_EMBEDDED_SOURCES['test-code/source/MedSAM2-main/sam2/sam2_image_predictor.py'] = '''# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging

from typing import List, Optional, Tuple, Union

import numpy as np
import torch
from PIL.Image import Image

from sam2.modeling.sam2_base import SAM2Base

from sam2.utils.transforms import SAM2Transforms


class SAM2ImagePredictor:
    def __init__(
        self,
        sam_model: SAM2Base,
        mask_threshold=0.0,
        max_hole_area=0.0,
        max_sprinkle_area=0.0,
        **kwargs,
    ) -> None:
        """
        Uses SAM-2 to calculate the image embedding for an image, and then
        allow repeated, efficient mask prediction given prompts.

        Arguments:
          sam_model (Sam-2): The model to use for mask prediction.
          mask_threshold (float): The threshold to use when converting mask logits
            to binary masks. Masks are thresholded at 0 by default.
          max_hole_area (int): If max_hole_area > 0, we fill small holes in up to
            the maximum area of max_hole_area in low_res_masks.
          max_sprinkle_area (int): If max_sprinkle_area > 0, we remove small sprinkles up to
            the maximum area of max_sprinkle_area in low_res_masks.
        """
        super().__init__()
        self.model = sam_model
        self._transforms = SAM2Transforms(
            resolution=self.model.image_size,
            mask_threshold=mask_threshold,
            max_hole_area=max_hole_area,
            max_sprinkle_area=max_sprinkle_area,
        )

        # Predictor state
        self._is_image_set = False
        self._features = None
        self._orig_hw = None
        # Whether the predictor is set for single image or a batch of images
        self._is_batch = False

        # Predictor config
        self.mask_threshold = mask_threshold

        # Spatial dim for backbone feature maps
        hires_size = self.model.image_size // 4
        self._bb_feat_sizes = [[hires_size // (2**k)]*2 for k in range(3)]
        # self._bb_feat_sizes = [
        #     (256, 256),
        #     (128, 128),
        #     (64, 64),
        # ]

    @classmethod
    def from_pretrained(cls, model_id: str, **kwargs) -> "SAM2ImagePredictor":
        """
        Load a pretrained model from the Hugging Face hub.

        Arguments:
          model_id (str): The Hugging Face repository ID.
          **kwargs: Additional arguments to pass to the model constructor.

        Returns:
          (SAM2ImagePredictor): The loaded model.
        """
        from sam2.build_sam import build_sam2_hf

        sam_model = build_sam2_hf(model_id, **kwargs)
        return cls(sam_model, **kwargs)

    @torch.no_grad()
    def set_image(
        self,
        image: Union[np.ndarray, Image],
    ) -> None:
        """
        Calculates the image embeddings for the provided image, allowing
        masks to be predicted with the 'predict' method.

        Arguments:
          image (np.ndarray or PIL Image): The input image to embed in RGB format. The image should be in HWC format if np.ndarray, or WHC format if PIL Image
          with pixel values in [0, 255].
          image_format (str): The color format of the image, in ['RGB', 'BGR'].
        """
        self.reset_predictor()
        # Transform the image to the form expected by the model
        if isinstance(image, np.ndarray):
            logging.info("For numpy array image, we assume (HxWxC) format")
            self._orig_hw = [image.shape[:2]]
        elif isinstance(image, Image):
            w, h = image.size
            self._orig_hw = [(h, w)]
        else:
            raise NotImplementedError("Image format not supported")

        input_image = self._transforms(image)
        input_image = input_image[None, ...].to(self.device)

        assert (
            len(input_image.shape) == 4 and input_image.shape[1] == 3
        ), f"input_image must be of size 1x3xHxW, got {input_image.shape}"
        logging.info("Computing image embeddings for the provided image...")
        backbone_out = self.model.forward_image(input_image)
        _, vision_feats, _, _ = self.model._prepare_backbone_features(backbone_out)
        # Add no_mem_embed, which is added to the lowest rest feat. map during training on videos
        if self.model.directly_add_no_mem_embed:
            vision_feats[-1] = vision_feats[-1] + self.model.no_mem_embed

        feats = [
            feat.permute(1, 2, 0).view(1, -1, *feat_size)
            for feat, feat_size in zip(vision_feats[::-1], self._bb_feat_sizes[::-1])
        ][::-1]
        self._features = {"image_embed": feats[-1], "high_res_feats": feats[:-1]}
        self._is_image_set = True
        logging.info("Image embeddings computed.")

    @torch.no_grad()
    def set_image_batch(
        self,
        image_list: List[Union[np.ndarray]],
    ) -> None:
        """
        Calculates the image embeddings for the provided image batch, allowing
        masks to be predicted with the 'predict_batch' method.

        Arguments:
          image_list (List[np.ndarray]): The input images to embed in RGB format. The image should be in HWC format if np.ndarray
          with pixel values in [0, 255].
        """
        self.reset_predictor()
        assert isinstance(image_list, list)
        self._orig_hw = []
        for image in image_list:
            assert isinstance(
                image, np.ndarray
            ), "Images are expected to be an np.ndarray in RGB format, and of shape  HWC"
            self._orig_hw.append(image.shape[:2])
        # Transform the image to the form expected by the model
        img_batch = self._transforms.forward_batch(image_list)
        img_batch = img_batch.to(self.device)
        batch_size = img_batch.shape[0]
        assert (
            len(img_batch.shape) == 4 and img_batch.shape[1] == 3
        ), f"img_batch must be of size Bx3xHxW, got {img_batch.shape}"
        logging.info("Computing image embeddings for the provided images...")
        backbone_out = self.model.forward_image(img_batch)
        _, vision_feats, _, _ = self.model._prepare_backbone_features(backbone_out)
        # Add no_mem_embed, which is added to the lowest rest feat. map during training on videos
        if self.model.directly_add_no_mem_embed:
            vision_feats[-1] = vision_feats[-1] + self.model.no_mem_embed

        feats = [
            feat.permute(1, 2, 0).view(batch_size, -1, *feat_size)
            for feat, feat_size in zip(vision_feats[::-1], self._bb_feat_sizes[::-1])
        ][::-1]
        self._features = {"image_embed": feats[-1], "high_res_feats": feats[:-1]}
        self._is_image_set = True
        self._is_batch = True
        logging.info("Image embeddings computed.")

    def predict_batch(
        self,
        point_coords_batch: List[np.ndarray] = None,
        point_labels_batch: List[np.ndarray] = None,
        box_batch: List[np.ndarray] = None,
        mask_input_batch: List[np.ndarray] = None,
        multimask_output: bool = True,
        return_logits: bool = False,
        normalize_coords=True,
    ) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
        """This function is very similar to predict(...), however it is used for batched mode, when the model is expected to generate predictions on multiple images.
        It returns a tuple of lists of masks, ious, and low_res_masks_logits.
        """
        assert self._is_batch, "This function should only be used when in batched mode"
        if not self._is_image_set:
            raise RuntimeError(
                "An image must be set with .set_image_batch(...) before mask prediction."
            )
        num_images = len(self._features["image_embed"])
        all_masks = []
        all_ious = []
        all_low_res_masks = []
        for img_idx in range(num_images):
            # Transform input prompts
            point_coords = (
                point_coords_batch[img_idx] if point_coords_batch is not None else None
            )
            point_labels = (
                point_labels_batch[img_idx] if point_labels_batch is not None else None
            )
            box = box_batch[img_idx] if box_batch is not None else None
            mask_input = (
                mask_input_batch[img_idx] if mask_input_batch is not None else None
            )
            mask_input, unnorm_coords, labels, unnorm_box = self._prep_prompts(
                point_coords,
                point_labels,
                box,
                mask_input,
                normalize_coords,
                img_idx=img_idx,
            )
            masks, iou_predictions, low_res_masks = self._predict(
                unnorm_coords,
                labels,
                unnorm_box,
                mask_input,
                multimask_output,
                return_logits=return_logits,
                img_idx=img_idx,
            )
            masks_np = masks.squeeze(0).float().detach().cpu().numpy()
            iou_predictions_np = (
                iou_predictions.squeeze(0).float().detach().cpu().numpy()
            )
            low_res_masks_np = low_res_masks.squeeze(0).float().detach().cpu().numpy()
            all_masks.append(masks_np)
            all_ious.append(iou_predictions_np)
            all_low_res_masks.append(low_res_masks_np)

        return all_masks, all_ious, all_low_res_masks

    def predict(
        self,
        point_coords: Optional[np.ndarray] = None,
        point_labels: Optional[np.ndarray] = None,
        box: Optional[np.ndarray] = None,
        mask_input: Optional[np.ndarray] = None,
        multimask_output: bool = True,
        return_logits: bool = False,
        normalize_coords=True,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Predict masks for the given input prompts, using the currently set image.

        Arguments:
          point_coords (np.ndarray or None): A Nx2 array of point prompts to the
            model. Each point is in (X,Y) in pixels.
          point_labels (np.ndarray or None): A length N array of labels for the
            point prompts. 1 indicates a foreground point and 0 indicates a
            background point.
          box (np.ndarray or None): A length 4 array given a box prompt to the
            model, in XYXY format.
          mask_input (np.ndarray): A low resolution mask input to the model, typically
            coming from a previous prediction iteration. Has form 1xHxW, where
            for SAM, H=W=256.
          multimask_output (bool): If true, the model will return three masks.
            For ambiguous input prompts (such as a single click), this will often
            produce better masks than a single prediction. If only a single
            mask is needed, the model's predicted quality score can be used
            to select the best mask. For non-ambiguous prompts, such as multiple
            input prompts, multimask_output=False can give better results.
          return_logits (bool): If true, returns un-thresholded masks logits
            instead of a binary mask.
          normalize_coords (bool): If true, the point coordinates will be normalized to the range [0,1] and point_coords is expected to be wrt. image dimensions.

        Returns:
          (np.ndarray): The output masks in CxHxW format, where C is the
            number of masks, and (H, W) is the original image size.
          (np.ndarray): An array of length C containing the model's
            predictions for the quality of each mask.
          (np.ndarray): An array of shape CxHxW, where C is the number
            of masks and H=W=256. These low resolution logits can be passed to
            a subsequent iteration as mask input.
        """
        if not self._is_image_set:
            raise RuntimeError(
                "An image must be set with .set_image(...) before mask prediction."
            )

        # Transform input prompts

        mask_input, unnorm_coords, labels, unnorm_box = self._prep_prompts(
            point_coords, point_labels, box, mask_input, normalize_coords
        )

        masks, iou_predictions, low_res_masks = self._predict(
            unnorm_coords,
            labels,
            unnorm_box,
            mask_input,
            multimask_output,
            return_logits=return_logits,
        )

        masks_np = masks.squeeze(0).float().detach().cpu().numpy()
        iou_predictions_np = iou_predictions.squeeze(0).float().detach().cpu().numpy()
        low_res_masks_np = low_res_masks.squeeze(0).float().detach().cpu().numpy()
        return masks_np, iou_predictions_np, low_res_masks_np

    def _prep_prompts(
        self, point_coords, point_labels, box, mask_logits, normalize_coords, img_idx=-1
    ):

        unnorm_coords, labels, unnorm_box, mask_input = None, None, None, None
        if point_coords is not None:
            assert (
                point_labels is not None
            ), "point_labels must be supplied if point_coords is supplied."
            point_coords = torch.as_tensor(
                point_coords, dtype=torch.float, device=self.device
            )
            unnorm_coords = self._transforms.transform_coords(
                point_coords, normalize=normalize_coords, orig_hw=self._orig_hw[img_idx]
            )
            labels = torch.as_tensor(point_labels, dtype=torch.int, device=self.device)
            if len(unnorm_coords.shape) == 2:
                unnorm_coords, labels = unnorm_coords[None, ...], labels[None, ...]
        if box is not None:
            box = torch.as_tensor(box, dtype=torch.float, device=self.device)
            unnorm_box = self._transforms.transform_boxes(
                box, normalize=normalize_coords, orig_hw=self._orig_hw[img_idx]
            )  # Bx2x2
        if mask_logits is not None:
            mask_input = torch.as_tensor(
                mask_logits, dtype=torch.float, device=self.device
            )
            if len(mask_input.shape) == 3:
                mask_input = mask_input[None, :, :, :]
        return mask_input, unnorm_coords, labels, unnorm_box

    @torch.no_grad()
    def _predict(
        self,
        point_coords: Optional[torch.Tensor],
        point_labels: Optional[torch.Tensor],
        boxes: Optional[torch.Tensor] = None,
        mask_input: Optional[torch.Tensor] = None,
        multimask_output: bool = True,
        return_logits: bool = False,
        img_idx: int = -1,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Predict masks for the given input prompts, using the currently set image.
        Input prompts are batched torch tensors and are expected to already be
        transformed to the input frame using SAM2Transforms.

        Arguments:
          point_coords (torch.Tensor or None): A BxNx2 array of point prompts to the
            model. Each point is in (X,Y) in pixels.
          point_labels (torch.Tensor or None): A BxN array of labels for the
            point prompts. 1 indicates a foreground point and 0 indicates a
            background point.
          boxes (np.ndarray or None): A Bx4 array given a box prompt to the
            model, in XYXY format.
          mask_input (np.ndarray): A low resolution mask input to the model, typically
            coming from a previous prediction iteration. Has form Bx1xHxW, where
            for SAM, H=W=256. Masks returned by a previous iteration of the
            predict method do not need further transformation.
          multimask_output (bool): If true, the model will return three masks.
            For ambiguous input prompts (such as a single click), this will often
            produce better masks than a single prediction. If only a single
            mask is needed, the model's predicted quality score can be used
            to select the best mask. For non-ambiguous prompts, such as multiple
            input prompts, multimask_output=False can give better results.
          return_logits (bool): If true, returns un-thresholded masks logits
            instead of a binary mask.

        Returns:
          (torch.Tensor): The output masks in BxCxHxW format, where C is the
            number of masks, and (H, W) is the original image size.
          (torch.Tensor): An array of shape BxC containing the model's
            predictions for the quality of each mask.
          (torch.Tensor): An array of shape BxCxHxW, where C is the number
            of masks and H=W=256. These low res logits can be passed to
            a subsequent iteration as mask input.
        """
        if not self._is_image_set:
            raise RuntimeError(
                "An image must be set with .set_image(...) before mask prediction."
            )

        if point_coords is not None:
            concat_points = (point_coords, point_labels)
        else:
            concat_points = None

        # Embed prompts
        if boxes is not None:
            box_coords = boxes.reshape(-1, 2, 2)
            box_labels = torch.tensor([[2, 3]], dtype=torch.int, device=boxes.device)
            box_labels = box_labels.repeat(boxes.size(0), 1)
            # we merge "boxes" and "points" into a single "concat_points" input (where
            # boxes are added at the beginning) to sam_prompt_encoder
            if concat_points is not None:
                concat_coords = torch.cat([box_coords, concat_points[0]], dim=1)
                concat_labels = torch.cat([box_labels, concat_points[1]], dim=1)
                concat_points = (concat_coords, concat_labels)
            else:
                concat_points = (box_coords, box_labels)

        sparse_embeddings, dense_embeddings = self.model.sam_prompt_encoder(
            points=concat_points,
            boxes=None,
            masks=mask_input,
        )

        # Predict masks
        batched_mode = (
            concat_points is not None and concat_points[0].shape[0] > 1
        )  # multi object prediction
        high_res_features = [
            feat_level[img_idx].unsqueeze(0)
            for feat_level in self._features["high_res_feats"]
        ]
        low_res_masks, iou_predictions, _, _ = self.model.sam_mask_decoder(
            image_embeddings=self._features["image_embed"][img_idx].unsqueeze(0),
            image_pe=self.model.sam_prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=multimask_output,
            repeat_image=batched_mode,
            high_res_features=high_res_features,
        )

        # Upscale the masks to the original image resolution
        masks = self._transforms.postprocess_masks(
            low_res_masks, self._orig_hw[img_idx]
        )
        low_res_masks = torch.clamp(low_res_masks, -32.0, 32.0)
        if not return_logits:
            masks = masks > self.mask_threshold

        return masks, iou_predictions, low_res_masks

    def get_image_embedding(self) -> torch.Tensor:
        """
        Returns the image embeddings for the currently set image, with
        shape 1xCxHxW, where C is the embedding dimension and (H,W) are
        the embedding spatial dimension of SAM (typically C=256, H=W=64).
        """
        if not self._is_image_set:
            raise RuntimeError(
                "An image must be set with .set_image(...) to generate an embedding."
            )
        assert (
            self._features is not None
        ), "Features must exist if an image has been set."
        return self._features["image_embed"]

    @property
    def device(self) -> torch.device:
        return self.model.device

    def reset_predictor(self) -> None:
        """
        Resets the image embeddings and other state variables.
        """
        self._is_image_set = False
        self._features = None
        self._orig_hw = None
        self._is_batch = False
'''

# Native module: test-code/source/MedSAM2-main/sam2/sam2_video_predictor.py
_EMBEDDED_SOURCES['test-code/source/MedSAM2-main/sam2/sam2_video_predictor.py'] = '''# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import warnings
from collections import OrderedDict

import torch

from tqdm import tqdm

from sam2.modeling.sam2_base import NO_OBJ_SCORE, SAM2Base
from sam2.utils.misc import concat_points, fill_holes_in_mask_scores, load_video_frames


class SAM2VideoPredictor(SAM2Base):
    """The predictor class to handle user interactions and manage inference states."""

    def __init__(
        self,
        fill_hole_area=0,
        # whether to apply non-overlapping constraints on the output object masks
        non_overlap_masks=False,
        # whether to clear non-conditioning memory of the surrounding frames (which may contain outdated information) after adding correction clicks;
        # note that this would only apply to *single-object tracking* unless `clear_non_cond_mem_for_multi_obj` is also set to True)
        clear_non_cond_mem_around_input=False,
        # whether to also clear non-conditioning memory of the surrounding frames (only effective when `clear_non_cond_mem_around_input` is True).
        clear_non_cond_mem_for_multi_obj=False,
        # if `add_all_frames_to_correct_as_cond` is True, we also append to the conditioning frame list any frame that receives a later correction click
        # if `add_all_frames_to_correct_as_cond` is False, we conditioning frame list to only use those initial conditioning frames
        add_all_frames_to_correct_as_cond=False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.fill_hole_area = fill_hole_area
        self.non_overlap_masks = non_overlap_masks
        self.clear_non_cond_mem_around_input = clear_non_cond_mem_around_input
        self.clear_non_cond_mem_for_multi_obj = clear_non_cond_mem_for_multi_obj
        self.add_all_frames_to_correct_as_cond = add_all_frames_to_correct_as_cond

    @torch.inference_mode()
    def init_state(
        self,
        video_path,
        offload_video_to_cpu=False,
        offload_state_to_cpu=False,
        async_loading_frames=False,
    ):
        """Initialize an inference state."""
        compute_device = self.device  # device of the model
        images, video_height, video_width = load_video_frames(
            video_path=video_path,
            image_size=self.image_size,
            offload_video_to_cpu=offload_video_to_cpu,
            async_loading_frames=async_loading_frames,
            compute_device=compute_device,
        )
        inference_state = {}
        inference_state["images"] = images
        inference_state["num_frames"] = len(images)
        # whether to offload the video frames to CPU memory
        # turning on this option saves the GPU memory with only a very small overhead
        inference_state["offload_video_to_cpu"] = offload_video_to_cpu
        # whether to offload the inference state to CPU memory
        # turning on this option saves the GPU memory at the cost of a lower tracking fps
        # (e.g. in a test case of 768x768 model, fps dropped from 27 to 24 when tracking one object
        # and from 24 to 21 when tracking two objects)
        inference_state["offload_state_to_cpu"] = offload_state_to_cpu
        # the original video height and width, used for resizing final output scores
        inference_state["video_height"] = video_height
        inference_state["video_width"] = video_width
        inference_state["device"] = compute_device
        if offload_state_to_cpu:
            inference_state["storage_device"] = torch.device("cpu")
        else:
            inference_state["storage_device"] = compute_device
        # inputs on each frame
        inference_state["point_inputs_per_obj"] = {}
        inference_state["mask_inputs_per_obj"] = {}
        # visual features on a small number of recently visited frames for quick interactions
        inference_state["cached_features"] = {}
        # values that don't change across frames (so we only need to hold one copy of them)
        inference_state["constants"] = {}
        # mapping between client-side object id and model-side object index
        inference_state["obj_id_to_idx"] = OrderedDict()
        inference_state["obj_idx_to_id"] = OrderedDict()
        inference_state["obj_ids"] = []
        # A storage to hold the model's tracking results and states on each frame
        inference_state["output_dict"] = {
            "cond_frame_outputs": {},  # dict containing {frame_idx: <out>}
            "non_cond_frame_outputs": {},  # dict containing {frame_idx: <out>}
        }
        # Slice (view) of each object tracking results, sharing the same memory with "output_dict"
        inference_state["output_dict_per_obj"] = {}
        # A temporary storage to hold new outputs when user interact with a frame
        # to add clicks or mask (it's merged into "output_dict" before propagation starts)
        inference_state["temp_output_dict_per_obj"] = {}
        # Frames that already holds consolidated outputs from click or mask inputs
        # (we directly use their consolidated outputs during tracking)
        inference_state["consolidated_frame_inds"] = {
            "cond_frame_outputs": set(),  # set containing frame indices
            "non_cond_frame_outputs": set(),  # set containing frame indices
        }
        # metadata for each tracking frame (e.g. which direction it's tracked)
        inference_state["tracking_has_started"] = False
        inference_state["frames_already_tracked"] = {}
        # Warm up the visual backbone and cache the image feature on frame 0
        self._get_image_feature(inference_state, frame_idx=0, batch_size=1)
        return inference_state

    @classmethod
    def from_pretrained(cls, model_id: str, **kwargs) -> "SAM2VideoPredictor":
        """
        Load a pretrained model from the Hugging Face hub.

        Arguments:
          model_id (str): The Hugging Face repository ID.
          **kwargs: Additional arguments to pass to the model constructor.

        Returns:
          (SAM2VideoPredictor): The loaded model.
        """
        from sam2.build_sam import build_sam2_video_predictor_hf

        sam_model = build_sam2_video_predictor_hf(model_id, **kwargs)
        return sam_model

    def _obj_id_to_idx(self, inference_state, obj_id):
        """Map client-side object id to model-side object index."""
        obj_idx = inference_state["obj_id_to_idx"].get(obj_id, None)
        if obj_idx is not None:
            return obj_idx

        # This is a new object id not sent to the server before. We only allow adding
        # new objects *before* the tracking starts.
        allow_new_object = not inference_state["tracking_has_started"]
        if allow_new_object:
            # get the next object slot
            obj_idx = len(inference_state["obj_id_to_idx"])
            inference_state["obj_id_to_idx"][obj_id] = obj_idx
            inference_state["obj_idx_to_id"][obj_idx] = obj_id
            inference_state["obj_ids"] = list(inference_state["obj_id_to_idx"])
            # set up input and output structures for this object
            inference_state["point_inputs_per_obj"][obj_idx] = {}
            inference_state["mask_inputs_per_obj"][obj_idx] = {}
            inference_state["output_dict_per_obj"][obj_idx] = {
                "cond_frame_outputs": {},  # dict containing {frame_idx: <out>}
                "non_cond_frame_outputs": {},  # dict containing {frame_idx: <out>}
            }
            inference_state["temp_output_dict_per_obj"][obj_idx] = {
                "cond_frame_outputs": {},  # dict containing {frame_idx: <out>}
                "non_cond_frame_outputs": {},  # dict containing {frame_idx: <out>}
            }
            return obj_idx
        else:
            raise RuntimeError(
                f"Cannot add new object id {obj_id} after tracking starts. "
                f"All existing object ids: {inference_state['obj_ids']}. "
                f"Please call 'reset_state' to restart from scratch."
            )

    def _obj_idx_to_id(self, inference_state, obj_idx):
        """Map model-side object index to client-side object id."""
        return inference_state["obj_idx_to_id"][obj_idx]

    def _get_obj_num(self, inference_state):
        """Get the total number of unique object ids received so far in this session."""
        return len(inference_state["obj_idx_to_id"])

    @torch.inference_mode()
    def add_new_points_or_box(
        self,
        inference_state,
        frame_idx,
        obj_id,
        points=None,
        labels=None,
        clear_old_points=True,
        normalize_coords=True,
        box=None,
    ):
        """Add new points to a frame."""
        obj_idx = self._obj_id_to_idx(inference_state, obj_id)
        point_inputs_per_frame = inference_state["point_inputs_per_obj"][obj_idx]
        mask_inputs_per_frame = inference_state["mask_inputs_per_obj"][obj_idx]

        if (points is not None) != (labels is not None):
            raise ValueError("points and labels must be provided together")
        if points is None and box is None:
            raise ValueError("at least one of points or box must be provided as input")

        if points is None:
            points = torch.zeros(0, 2, dtype=torch.float32)
        elif not isinstance(points, torch.Tensor):
            points = torch.tensor(points, dtype=torch.float32)
        if labels is None:
            labels = torch.zeros(0, dtype=torch.int32)
        elif not isinstance(labels, torch.Tensor):
            labels = torch.tensor(labels, dtype=torch.int32)
        if points.dim() == 2:
            points = points.unsqueeze(0)  # add batch dimension
        if labels.dim() == 1:
            labels = labels.unsqueeze(0)  # add batch dimension

        # If `box` is provided, we add it as the first two points with labels 2 and 3
        # along with the user-provided points (consistent with how SAM 2 is trained).
        if box is not None:
            if not clear_old_points:
                raise ValueError(
                    "cannot add box without clearing old points, since "
                    "box prompt must be provided before any point prompt "
                    "(please use clear_old_points=True instead)"
                )
            if inference_state["tracking_has_started"]:
                warnings.warn(
                    "You are adding a box after tracking starts. SAM 2 may not always be "
                    "able to incorporate a box prompt for *refinement*. If you intend to "
                    "use box prompt as an *initial* input before tracking, please call "
                    "'reset_state' on the inference state to restart from scratch.",
                    category=UserWarning,
                    stacklevel=2,
                )
            if not isinstance(box, torch.Tensor):
                box = torch.tensor(box, dtype=torch.float32, device=points.device)
            box_coords = box.reshape(1, 2, 2)
            box_labels = torch.tensor([2, 3], dtype=torch.int32, device=labels.device)
            box_labels = box_labels.reshape(1, 2)
            points = torch.cat([box_coords, points], dim=1)
            labels = torch.cat([box_labels, labels], dim=1)

        if normalize_coords:
            video_H = inference_state["video_height"]
            video_W = inference_state["video_width"]
            points = points / torch.tensor([video_W, video_H]).to(points.device)
        # scale the (normalized) coordinates by the model's internal image size
        points = points * self.image_size
        points = points.to(inference_state["device"])
        labels = labels.to(inference_state["device"])

        if not clear_old_points:
            point_inputs = point_inputs_per_frame.get(frame_idx, None)
        else:
            point_inputs = None
        point_inputs = concat_points(point_inputs, points, labels)

        point_inputs_per_frame[frame_idx] = point_inputs
        mask_inputs_per_frame.pop(frame_idx, None)
        # If this frame hasn't been tracked before, we treat it as an initial conditioning
        # frame, meaning that the inputs points are to generate segments on this frame without
        # using any memory from other frames, like in SAM. Otherwise (if it has been tracked),
        # the input points will be used to correct the already tracked masks.
        is_init_cond_frame = frame_idx not in inference_state["frames_already_tracked"]
        # whether to track in reverse time order
        if is_init_cond_frame:
            reverse = False
        else:
            reverse = inference_state["frames_already_tracked"][frame_idx]["reverse"]
        obj_output_dict = inference_state["output_dict_per_obj"][obj_idx]
        obj_temp_output_dict = inference_state["temp_output_dict_per_obj"][obj_idx]
        # Add a frame to conditioning output if it's an initial conditioning frame or
        # if the model sees all frames receiving clicks/mask as conditioning frames.
        is_cond = is_init_cond_frame or self.add_all_frames_to_correct_as_cond
        storage_key = "cond_frame_outputs" if is_cond else "non_cond_frame_outputs"

        # Get any previously predicted mask logits on this object and feed it along with
        # the new clicks into the SAM mask decoder.
        prev_sam_mask_logits = None
        # lookup temporary output dict first, which contains the most recent output
        # (if not found, then lookup conditioning and non-conditioning frame output)
        prev_out = obj_temp_output_dict[storage_key].get(frame_idx)
        if prev_out is None:
            prev_out = obj_output_dict["cond_frame_outputs"].get(frame_idx)
            if prev_out is None:
                prev_out = obj_output_dict["non_cond_frame_outputs"].get(frame_idx)

        if prev_out is not None and prev_out["pred_masks"] is not None:
            device = inference_state["device"]
            prev_sam_mask_logits = prev_out["pred_masks"].to(device, non_blocking=True)
            # Clamp the scale of prev_sam_mask_logits to avoid rare numerical issues.
            prev_sam_mask_logits = torch.clamp(prev_sam_mask_logits, -32.0, 32.0)
        current_out, _ = self._run_single_frame_inference(
            inference_state=inference_state,
            output_dict=obj_output_dict,  # run on the slice of a single object
            frame_idx=frame_idx,
            batch_size=1,  # run on the slice of a single object
            is_init_cond_frame=is_init_cond_frame,
            point_inputs=point_inputs,
            mask_inputs=None,
            reverse=reverse,
            # Skip the memory encoder when adding clicks or mask. We execute the memory encoder
            # at the beginning of `propagate_in_video` (after user finalize their clicks). This
            # allows us to enforce non-overlapping constraints on all objects before encoding
            # them into memory.
            run_mem_encoder=False,
            prev_sam_mask_logits=prev_sam_mask_logits,
        )
        # Add the output to the output dict (to be used as future memory)
        obj_temp_output_dict[storage_key][frame_idx] = current_out

        # Resize the output mask to the original video resolution
        obj_ids = inference_state["obj_ids"]
        consolidated_out = self._consolidate_temp_output_across_obj(
            inference_state,
            frame_idx,
            is_cond=is_cond,
            run_mem_encoder=False,
            consolidate_at_video_res=True,
        )
        _, video_res_masks = self._get_orig_video_res_output(
            inference_state, consolidated_out["pred_masks_video_res"]
        )
        return frame_idx, obj_ids, video_res_masks

    def add_new_points(self, *args, **kwargs):
        """Deprecated method. Please use `add_new_points_or_box` instead."""
        return self.add_new_points_or_box(*args, **kwargs)

    @torch.inference_mode()
    def add_new_mask(
        self,
        inference_state,
        frame_idx,
        obj_id,
        mask,
    ):
        """Add new mask to a frame."""
        obj_idx = self._obj_id_to_idx(inference_state, obj_id)
        point_inputs_per_frame = inference_state["point_inputs_per_obj"][obj_idx]
        mask_inputs_per_frame = inference_state["mask_inputs_per_obj"][obj_idx]

        if not isinstance(mask, torch.Tensor):
            mask = torch.tensor(mask, dtype=torch.bool)
        assert mask.dim() == 2
        mask_H, mask_W = mask.shape
        mask_inputs_orig = mask[None, None]  # add batch and channel dimension
        mask_inputs_orig = mask_inputs_orig.float().to(inference_state["device"])

        # resize the mask if it doesn't match the model's image size
        if mask_H != self.image_size or mask_W != self.image_size:
            mask_inputs = torch.nn.functional.interpolate(
                mask_inputs_orig,
                size=(self.image_size, self.image_size),
                align_corners=False,
                mode="bilinear",
                antialias=True,  # use antialias for downsampling
            )
            mask_inputs = (mask_inputs >= 0.5).float()
        else:
            mask_inputs = mask_inputs_orig

        mask_inputs_per_frame[frame_idx] = mask_inputs
        point_inputs_per_frame.pop(frame_idx, None)
        # If this frame hasn't been tracked before, we treat it as an initial conditioning
        # frame, meaning that the inputs points are to generate segments on this frame without
        # using any memory from other frames, like in SAM. Otherwise (if it has been tracked),
        # the input points will be used to correct the already tracked masks.
        is_init_cond_frame = frame_idx not in inference_state["frames_already_tracked"]
        # whether to track in reverse time order
        if is_init_cond_frame:
            reverse = False
        else:
            reverse = inference_state["frames_already_tracked"][frame_idx]["reverse"]
        obj_output_dict = inference_state["output_dict_per_obj"][obj_idx]
        obj_temp_output_dict = inference_state["temp_output_dict_per_obj"][obj_idx]
        # Add a frame to conditioning output if it's an initial conditioning frame or
        # if the model sees all frames receiving clicks/mask as conditioning frames.
        is_cond = is_init_cond_frame or self.add_all_frames_to_correct_as_cond
        storage_key = "cond_frame_outputs" if is_cond else "non_cond_frame_outputs"

        current_out, _ = self._run_single_frame_inference(
            inference_state=inference_state,
            output_dict=obj_output_dict,  # run on the slice of a single object
            frame_idx=frame_idx,
            batch_size=1,  # run on the slice of a single object
            is_init_cond_frame=is_init_cond_frame,
            point_inputs=None,
            mask_inputs=mask_inputs,
            reverse=reverse,
            # Skip the memory encoder when adding clicks or mask. We execute the memory encoder
            # at the beginning of `propagate_in_video` (after user finalize their clicks). This
            # allows us to enforce non-overlapping constraints on all objects before encoding
            # them into memory.
            run_mem_encoder=False,
        )
        # Add the output to the output dict (to be used as future memory)
        obj_temp_output_dict[storage_key][frame_idx] = current_out

        # Resize the output mask to the original video resolution
        obj_ids = inference_state["obj_ids"]
        consolidated_out = self._consolidate_temp_output_across_obj(
            inference_state,
            frame_idx,
            is_cond=is_cond,
            run_mem_encoder=False,
            consolidate_at_video_res=True,
        )
        _, video_res_masks = self._get_orig_video_res_output(
            inference_state, consolidated_out["pred_masks_video_res"]
        )
        return frame_idx, obj_ids, video_res_masks

    def _get_orig_video_res_output(self, inference_state, any_res_masks):
        """
        Resize the object scores to the original video resolution (video_res_masks)
        and apply non-overlapping constraints for final output.
        """
        device = inference_state["device"]
        video_H = inference_state["video_height"]
        video_W = inference_state["video_width"]
        any_res_masks = any_res_masks.to(device, non_blocking=True)
        if any_res_masks.shape[-2:] == (video_H, video_W):
            video_res_masks = any_res_masks
        else:
            video_res_masks = torch.nn.functional.interpolate(
                any_res_masks,
                size=(video_H, video_W),
                mode="bilinear",
                align_corners=False,
            )
        if self.non_overlap_masks:
            video_res_masks = self._apply_non_overlapping_constraints(video_res_masks)
        return any_res_masks, video_res_masks

    def _consolidate_temp_output_across_obj(
        self,
        inference_state,
        frame_idx,
        is_cond,
        run_mem_encoder,
        consolidate_at_video_res=False,
    ):
        """
        Consolidate the per-object temporary outputs in `temp_output_dict_per_obj` on
        a frame into a single output for all objects, including
        1) fill any missing objects either from `output_dict_per_obj` (if they exist in
           `output_dict_per_obj` for this frame) or leave them as placeholder values
           (if they don't exist in `output_dict_per_obj` for this frame);
        2) if specified, rerun memory encoder after apply non-overlapping constraints
           on the object scores.
        """
        batch_size = self._get_obj_num(inference_state)
        storage_key = "cond_frame_outputs" if is_cond else "non_cond_frame_outputs"
        # Optionally, we allow consolidating the temporary outputs at the original
        # video resolution (to provide a better editing experience for mask prompts).
        if consolidate_at_video_res:
            assert not run_mem_encoder, "memory encoder cannot run at video resolution"
            consolidated_H = inference_state["video_height"]
            consolidated_W = inference_state["video_width"]
            consolidated_mask_key = "pred_masks_video_res"
        else:
            consolidated_H = consolidated_W = self.image_size // 4
            consolidated_mask_key = "pred_masks"

        # Initialize `consolidated_out`. Its "maskmem_features" and "maskmem_pos_enc"
        # will be added when rerunning the memory encoder after applying non-overlapping
        # constraints to object scores. Its "pred_masks" are prefilled with a large
        # negative value (NO_OBJ_SCORE) to represent missing objects.
        consolidated_out = {
            "maskmem_features": None,
            "maskmem_pos_enc": None,
            consolidated_mask_key: torch.full(
                size=(batch_size, 1, consolidated_H, consolidated_W),
                fill_value=NO_OBJ_SCORE,
                dtype=torch.float32,
                device=inference_state["storage_device"],
            ),
            "obj_ptr": torch.full(
                size=(batch_size, self.hidden_dim),
                fill_value=NO_OBJ_SCORE,
                dtype=torch.float32,
                device=inference_state["device"],
            ),
            "object_score_logits": torch.full(
                size=(batch_size, 1),
                # default to 10.0 for object_score_logits, i.e. assuming the object is
                # present as sigmoid(10)=1, same as in `predict_masks` of `MaskDecoder`
                fill_value=10.0,
                dtype=torch.float32,
                device=inference_state["device"],
            ),
        }
        empty_mask_ptr = None
        for obj_idx in range(batch_size):
            obj_temp_output_dict = inference_state["temp_output_dict_per_obj"][obj_idx]
            obj_output_dict = inference_state["output_dict_per_obj"][obj_idx]
            out = obj_temp_output_dict[storage_key].get(frame_idx, None)
            # If the object doesn't appear in "temp_output_dict_per_obj" on this frame,
            # we fall back and look up its previous output in "output_dict_per_obj".
            # We look up both "cond_frame_outputs" and "non_cond_frame_outputs" in
            # "output_dict_per_obj" to find a previous output for this object.
            if out is None:
                out = obj_output_dict["cond_frame_outputs"].get(frame_idx, None)
            if out is None:
                out = obj_output_dict["non_cond_frame_outputs"].get(frame_idx, None)
            # If the object doesn't appear in "output_dict_per_obj" either, we skip it
            # and leave its mask scores to the default scores (i.e. the NO_OBJ_SCORE
            # placeholder above) and set its object pointer to be a dummy pointer.
            if out is None:
                # Fill in dummy object pointers for those objects without any inputs or
                # tracking outcomes on this frame (only do it under `run_mem_encoder=True`,
                # i.e. when we need to build the memory for tracking).
                if run_mem_encoder:
                    if empty_mask_ptr is None:
                        empty_mask_ptr = self._get_empty_mask_ptr(
                            inference_state, frame_idx
                        )
                    # fill object pointer with a dummy pointer (based on an empty mask)
                    consolidated_out["obj_ptr"][obj_idx : obj_idx + 1] = empty_mask_ptr
                continue
            # Add the temporary object output mask to consolidated output mask
            obj_mask = out["pred_masks"]
            consolidated_pred_masks = consolidated_out[consolidated_mask_key]
            if obj_mask.shape[-2:] == consolidated_pred_masks.shape[-2:]:
                consolidated_pred_masks[obj_idx : obj_idx + 1] = obj_mask
            else:
                # Resize first if temporary object mask has a different resolution
                resized_obj_mask = torch.nn.functional.interpolate(
                    obj_mask,
                    size=consolidated_pred_masks.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )
                consolidated_pred_masks[obj_idx : obj_idx + 1] = resized_obj_mask
            consolidated_out["obj_ptr"][obj_idx : obj_idx + 1] = out["obj_ptr"]
            consolidated_out["object_score_logits"][obj_idx : obj_idx + 1] = out[
                "object_score_logits"
            ]

        # Optionally, apply non-overlapping constraints on the consolidated scores
        # and rerun the memory encoder
        if run_mem_encoder:
            device = inference_state["device"]
            high_res_masks = torch.nn.functional.interpolate(
                consolidated_out["pred_masks"].to(device, non_blocking=True),
                size=(self.image_size, self.image_size),
                mode="bilinear",
                align_corners=False,
            )
            if self.non_overlap_masks_for_mem_enc:
                high_res_masks = self._apply_non_overlapping_constraints(high_res_masks)
            maskmem_features, maskmem_pos_enc = self._run_memory_encoder(
                inference_state=inference_state,
                frame_idx=frame_idx,
                batch_size=batch_size,
                high_res_masks=high_res_masks,
                object_score_logits=consolidated_out["object_score_logits"],
                is_mask_from_pts=True,  # these frames are what the user interacted with
            )
            consolidated_out["maskmem_features"] = maskmem_features
            consolidated_out["maskmem_pos_enc"] = maskmem_pos_enc

        return consolidated_out

    def _get_empty_mask_ptr(self, inference_state, frame_idx):
        """Get a dummy object pointer based on an empty mask on the current frame."""
        # A dummy (empty) mask with a single object
        batch_size = 1
        mask_inputs = torch.zeros(
            (batch_size, 1, self.image_size, self.image_size),
            dtype=torch.float32,
            device=inference_state["device"],
        )

        # Retrieve correct image features
        (
            _,
            _,
            current_vision_feats,
            current_vision_pos_embeds,
            feat_sizes,
        ) = self._get_image_feature(inference_state, frame_idx, batch_size)

        # Feed the empty mask and image feature above to get a dummy object pointer
        current_out = self.track_step(
            frame_idx=frame_idx,
            is_init_cond_frame=True,
            current_vision_feats=current_vision_feats,
            current_vision_pos_embeds=current_vision_pos_embeds,
            feat_sizes=feat_sizes,
            point_inputs=None,
            mask_inputs=mask_inputs,
            output_dict={},
            num_frames=inference_state["num_frames"],
            track_in_reverse=False,
            run_mem_encoder=False,
            prev_sam_mask_logits=None,
        )
        return current_out["obj_ptr"]

    @torch.inference_mode()
    def propagate_in_video_preflight(self, inference_state):
        """Prepare inference_state and consolidate temporary outputs before tracking."""
        # Tracking has started and we don't allow adding new objects until session is reset.
        inference_state["tracking_has_started"] = True
        batch_size = self._get_obj_num(inference_state)

        # Consolidate per-object temporary outputs in "temp_output_dict_per_obj" and
        # add them into "output_dict".
        temp_output_dict_per_obj = inference_state["temp_output_dict_per_obj"]
        output_dict = inference_state["output_dict"]
        # "consolidated_frame_inds" contains indices of those frames where consolidated
        # temporary outputs have been added (either in this call or any previous calls
        # to `propagate_in_video_preflight`).
        consolidated_frame_inds = inference_state["consolidated_frame_inds"]
        for is_cond in [False, True]:
            # Separately consolidate conditioning and non-conditioning temp outputs
            storage_key = "cond_frame_outputs" if is_cond else "non_cond_frame_outputs"
            # Find all the frames that contain temporary outputs for any objects
            # (these should be the frames that have just received clicks for mask inputs
            # via `add_new_points_or_box` or `add_new_mask`)
            temp_frame_inds = set()
            for obj_temp_output_dict in temp_output_dict_per_obj.values():
                temp_frame_inds.update(obj_temp_output_dict[storage_key].keys())
            consolidated_frame_inds[storage_key].update(temp_frame_inds)
            # consolidate the temporary output across all objects on this frame
            for frame_idx in temp_frame_inds:
                consolidated_out = self._consolidate_temp_output_across_obj(
                    inference_state, frame_idx, is_cond=is_cond, run_mem_encoder=True
                )
                # merge them into "output_dict" and also create per-object slices
                output_dict[storage_key][frame_idx] = consolidated_out
                self._add_output_per_object(
                    inference_state, frame_idx, consolidated_out, storage_key
                )
                clear_non_cond_mem = self.clear_non_cond_mem_around_input and (
                    self.clear_non_cond_mem_for_multi_obj or batch_size <= 1
                )
                if clear_non_cond_mem:
                    # clear non-conditioning memory of the surrounding frames
                    self._clear_non_cond_mem_around_input(inference_state, frame_idx)

            # clear temporary outputs in `temp_output_dict_per_obj`
            for obj_temp_output_dict in temp_output_dict_per_obj.values():
                obj_temp_output_dict[storage_key].clear()

        # edge case: if an output is added to "cond_frame_outputs", we remove any prior
        # output on the same frame in "non_cond_frame_outputs"
        for frame_idx in output_dict["cond_frame_outputs"]:
            output_dict["non_cond_frame_outputs"].pop(frame_idx, None)
        for obj_output_dict in inference_state["output_dict_per_obj"].values():
            for frame_idx in obj_output_dict["cond_frame_outputs"]:
                obj_output_dict["non_cond_frame_outputs"].pop(frame_idx, None)
        for frame_idx in consolidated_frame_inds["cond_frame_outputs"]:
            assert frame_idx in output_dict["cond_frame_outputs"]
            consolidated_frame_inds["non_cond_frame_outputs"].discard(frame_idx)

        # Make sure that the frame indices in "consolidated_frame_inds" are exactly those frames
        # with either points or mask inputs (which should be true under a correct workflow).
        all_consolidated_frame_inds = (
            consolidated_frame_inds["cond_frame_outputs"]
            | consolidated_frame_inds["non_cond_frame_outputs"]
        )
        input_frames_inds = set()
        for point_inputs_per_frame in inference_state["point_inputs_per_obj"].values():
            input_frames_inds.update(point_inputs_per_frame.keys())
        for mask_inputs_per_frame in inference_state["mask_inputs_per_obj"].values():
            input_frames_inds.update(mask_inputs_per_frame.keys())
        assert all_consolidated_frame_inds == input_frames_inds

    @torch.inference_mode()
    def propagate_in_video(
        self,
        inference_state,
        start_frame_idx=None,
        max_frame_num_to_track=None,
        reverse=False,
    ):
        """Propagate the input points across frames to track in the entire video."""
        self.propagate_in_video_preflight(inference_state)

        output_dict = inference_state["output_dict"]
        consolidated_frame_inds = inference_state["consolidated_frame_inds"]
        obj_ids = inference_state["obj_ids"]
        num_frames = inference_state["num_frames"]
        batch_size = self._get_obj_num(inference_state)
        if len(output_dict["cond_frame_outputs"]) == 0:
            raise RuntimeError("No points are provided; please add points first")
        clear_non_cond_mem = self.clear_non_cond_mem_around_input and (
            self.clear_non_cond_mem_for_multi_obj or batch_size <= 1
        )

        # set start index, end index, and processing order
        if start_frame_idx is None:
            # default: start from the earliest frame with input points
            start_frame_idx = min(output_dict["cond_frame_outputs"])
        if max_frame_num_to_track is None:
            # default: track all the frames in the video
            max_frame_num_to_track = num_frames
        if reverse:
            end_frame_idx = max(start_frame_idx - max_frame_num_to_track, 0)
            if start_frame_idx > 0:
                processing_order = range(start_frame_idx, end_frame_idx - 1, -1)
            else:
                processing_order = []  # skip reverse tracking if starting from frame 0
        else:
            end_frame_idx = min(
                start_frame_idx + max_frame_num_to_track, num_frames - 1
            )
            processing_order = range(start_frame_idx, end_frame_idx + 1)

        for frame_idx in tqdm(processing_order, desc="propagate in video"):
            # We skip those frames already in consolidated outputs (these are frames
            # that received input clicks or mask). Note that we cannot directly run
            # batched forward on them via `_run_single_frame_inference` because the
            # number of clicks on each object might be different.
            if frame_idx in consolidated_frame_inds["cond_frame_outputs"]:
                storage_key = "cond_frame_outputs"
                current_out = output_dict[storage_key][frame_idx]
                pred_masks = current_out["pred_masks"]
                if clear_non_cond_mem:
                    # clear non-conditioning memory of the surrounding frames
                    self._clear_non_cond_mem_around_input(inference_state, frame_idx)
            elif frame_idx in consolidated_frame_inds["non_cond_frame_outputs"]:
                storage_key = "non_cond_frame_outputs"
                current_out = output_dict[storage_key][frame_idx]
                pred_masks = current_out["pred_masks"]
            else:
                storage_key = "non_cond_frame_outputs"
                current_out, pred_masks = self._run_single_frame_inference(
                    inference_state=inference_state,
                    output_dict=output_dict,
                    frame_idx=frame_idx,
                    batch_size=batch_size,
                    is_init_cond_frame=False,
                    point_inputs=None,
                    mask_inputs=None,
                    reverse=reverse,
                    run_mem_encoder=True,
                )
                output_dict[storage_key][frame_idx] = current_out
            # Create slices of per-object outputs for subsequent interaction with each
            # individual object after tracking.
            self._add_output_per_object(
                inference_state, frame_idx, current_out, storage_key
            )
            inference_state["frames_already_tracked"][frame_idx] = {"reverse": reverse}

            # Resize the output mask to the original video resolution (we directly use
            # the mask scores on GPU for output to avoid any CPU conversion in between)
            _, video_res_masks = self._get_orig_video_res_output(
                inference_state, pred_masks
            )
            yield frame_idx, obj_ids, video_res_masks

    def _add_output_per_object(
        self, inference_state, frame_idx, current_out, storage_key
    ):
        """
        Split a multi-object output into per-object output slices and add them into
        `output_dict_per_obj`. The resulting slices share the same tensor storage.
        """
        maskmem_features = current_out["maskmem_features"]
        assert maskmem_features is None or isinstance(maskmem_features, torch.Tensor)

        maskmem_pos_enc = current_out["maskmem_pos_enc"]
        assert maskmem_pos_enc is None or isinstance(maskmem_pos_enc, list)

        output_dict_per_obj = inference_state["output_dict_per_obj"]
        for obj_idx, obj_output_dict in output_dict_per_obj.items():
            obj_slice = slice(obj_idx, obj_idx + 1)
            obj_out = {
                "maskmem_features": None,
                "maskmem_pos_enc": None,
                "pred_masks": current_out["pred_masks"][obj_slice],
                "obj_ptr": current_out["obj_ptr"][obj_slice],
                "object_score_logits": current_out["object_score_logits"][obj_slice],
            }
            if maskmem_features is not None:
                obj_out["maskmem_features"] = maskmem_features[obj_slice]
            if maskmem_pos_enc is not None:
                obj_out["maskmem_pos_enc"] = [x[obj_slice] for x in maskmem_pos_enc]
            obj_output_dict[storage_key][frame_idx] = obj_out

    @torch.inference_mode()
    def clear_all_prompts_in_frame(
        self, inference_state, frame_idx, obj_id, need_output=True
    ):
        """Remove all input points or mask in a specific frame for a given object."""
        obj_idx = self._obj_id_to_idx(inference_state, obj_id)

        # Clear the conditioning information on the given frame
        inference_state["point_inputs_per_obj"][obj_idx].pop(frame_idx, None)
        inference_state["mask_inputs_per_obj"][obj_idx].pop(frame_idx, None)

        temp_output_dict_per_obj = inference_state["temp_output_dict_per_obj"]
        temp_output_dict_per_obj[obj_idx]["cond_frame_outputs"].pop(frame_idx, None)
        temp_output_dict_per_obj[obj_idx]["non_cond_frame_outputs"].pop(frame_idx, None)

        # Check and see if there are still any inputs left on this frame
        batch_size = self._get_obj_num(inference_state)
        frame_has_input = False
        for obj_idx2 in range(batch_size):
            if frame_idx in inference_state["point_inputs_per_obj"][obj_idx2]:
                frame_has_input = True
                break
            if frame_idx in inference_state["mask_inputs_per_obj"][obj_idx2]:
                frame_has_input = True
                break

        # If this frame has no remaining inputs for any objects, we further clear its
        # conditioning frame status
        if not frame_has_input:
            output_dict = inference_state["output_dict"]
            consolidated_frame_inds = inference_state["consolidated_frame_inds"]
            consolidated_frame_inds["cond_frame_outputs"].discard(frame_idx)
            consolidated_frame_inds["non_cond_frame_outputs"].discard(frame_idx)
            # Remove the frame's conditioning output (possibly downgrading it to non-conditioning)
            out = output_dict["cond_frame_outputs"].pop(frame_idx, None)
            if out is not None:
                # The frame is not a conditioning frame anymore since it's not receiving inputs,
                # so we "downgrade" its output (if exists) to a non-conditioning frame output.
                output_dict["non_cond_frame_outputs"][frame_idx] = out
                inference_state["frames_already_tracked"].pop(frame_idx, None)
            # Similarly, do it for the sliced output on each object.
            for obj_idx2 in range(batch_size):
                obj_output_dict = inference_state["output_dict_per_obj"][obj_idx2]
                obj_out = obj_output_dict["cond_frame_outputs"].pop(frame_idx, None)
                if obj_out is not None:
                    obj_output_dict["non_cond_frame_outputs"][frame_idx] = obj_out

            # If all the conditioning frames have been removed, we also clear the tracking outputs
            if len(output_dict["cond_frame_outputs"]) == 0:
                self._reset_tracking_results(inference_state)

        if not need_output:
            return
        # Finally, output updated masks per object (after removing the inputs above)
        obj_ids = inference_state["obj_ids"]
        is_cond = any(
            frame_idx in obj_temp_output_dict["cond_frame_outputs"]
            for obj_temp_output_dict in temp_output_dict_per_obj.values()
        )
        consolidated_out = self._consolidate_temp_output_across_obj(
            inference_state,
            frame_idx,
            is_cond=is_cond,
            run_mem_encoder=False,
            consolidate_at_video_res=True,
        )
        _, video_res_masks = self._get_orig_video_res_output(
            inference_state, consolidated_out["pred_masks_video_res"]
        )
        return frame_idx, obj_ids, video_res_masks

    @torch.inference_mode()
    def reset_state(self, inference_state):
        """Remove all input points or mask in all frames throughout the video."""
        self._reset_tracking_results(inference_state)
        # Remove all object ids
        inference_state["obj_id_to_idx"].clear()
        inference_state["obj_idx_to_id"].clear()
        inference_state["obj_ids"].clear()
        inference_state["point_inputs_per_obj"].clear()
        inference_state["mask_inputs_per_obj"].clear()
        inference_state["output_dict_per_obj"].clear()
        inference_state["temp_output_dict_per_obj"].clear()

    def _reset_tracking_results(self, inference_state):
        """Reset all tracking inputs and results across the videos."""
        for v in inference_state["point_inputs_per_obj"].values():
            v.clear()
        for v in inference_state["mask_inputs_per_obj"].values():
            v.clear()
        for v in inference_state["output_dict_per_obj"].values():
            v["cond_frame_outputs"].clear()
            v["non_cond_frame_outputs"].clear()
        for v in inference_state["temp_output_dict_per_obj"].values():
            v["cond_frame_outputs"].clear()
            v["non_cond_frame_outputs"].clear()
        inference_state["output_dict"]["cond_frame_outputs"].clear()
        inference_state["output_dict"]["non_cond_frame_outputs"].clear()
        inference_state["consolidated_frame_inds"]["cond_frame_outputs"].clear()
        inference_state["consolidated_frame_inds"]["non_cond_frame_outputs"].clear()
        inference_state["tracking_has_started"] = False
        inference_state["frames_already_tracked"].clear()

    def _get_image_feature(self, inference_state, frame_idx, batch_size):
        """Compute the image features on a given frame."""
        # Look up in the cache first
        image, backbone_out = inference_state["cached_features"].get(
            frame_idx, (None, None)
        )
        if backbone_out is None:
            # Cache miss -- we will run inference on a single image
            device = inference_state["device"]
            image = inference_state["images"][frame_idx].to(device).float().unsqueeze(0)
            backbone_out = self.forward_image(image)
            # Cache the most recent frame's feature (for repeated interactions with
            # a frame; we can use an LRU cache for more frames in the future).
            inference_state["cached_features"] = {frame_idx: (image, backbone_out)}

        # expand the features to have the same dimension as the number of objects
        expanded_image = image.expand(batch_size, -1, -1, -1)
        expanded_backbone_out = {
            "backbone_fpn": backbone_out["backbone_fpn"].copy(),
            "vision_pos_enc": backbone_out["vision_pos_enc"].copy(),
        }
        for i, feat in enumerate(expanded_backbone_out["backbone_fpn"]):
            expanded_backbone_out["backbone_fpn"][i] = feat.expand(
                batch_size, -1, -1, -1
            )
        for i, pos in enumerate(expanded_backbone_out["vision_pos_enc"]):
            pos = pos.expand(batch_size, -1, -1, -1)
            expanded_backbone_out["vision_pos_enc"][i] = pos

        features = self._prepare_backbone_features(expanded_backbone_out)
        features = (expanded_image,) + features
        return features

    def _run_single_frame_inference(
        self,
        inference_state,
        output_dict,
        frame_idx,
        batch_size,
        is_init_cond_frame,
        point_inputs,
        mask_inputs,
        reverse,
        run_mem_encoder,
        prev_sam_mask_logits=None,
    ):
        """Run tracking on a single frame based on current inputs and previous memory."""
        # Retrieve correct image features
        (
            _,
            _,
            current_vision_feats,
            current_vision_pos_embeds,
            feat_sizes,
        ) = self._get_image_feature(inference_state, frame_idx, batch_size)

        # point and mask should not appear as input simultaneously on the same frame
        assert point_inputs is None or mask_inputs is None
        current_out = self.track_step(
            frame_idx=frame_idx,
            is_init_cond_frame=is_init_cond_frame,
            current_vision_feats=current_vision_feats,
            current_vision_pos_embeds=current_vision_pos_embeds,
            feat_sizes=feat_sizes,
            point_inputs=point_inputs,
            mask_inputs=mask_inputs,
            output_dict=output_dict,
            num_frames=inference_state["num_frames"],
            track_in_reverse=reverse,
            run_mem_encoder=run_mem_encoder,
            prev_sam_mask_logits=prev_sam_mask_logits,
        )

        # optionally offload the output to CPU memory to save GPU space
        storage_device = inference_state["storage_device"]
        maskmem_features = current_out["maskmem_features"]
        if maskmem_features is not None:
            maskmem_features = maskmem_features.to(torch.bfloat16)
            maskmem_features = maskmem_features.to(storage_device, non_blocking=True)
        pred_masks_gpu = current_out["pred_masks"]
        # potentially fill holes in the predicted masks
        if self.fill_hole_area > 0:
            pred_masks_gpu = fill_holes_in_mask_scores(
                pred_masks_gpu, self.fill_hole_area
            )
        pred_masks = pred_masks_gpu.to(storage_device, non_blocking=True)
        # "maskmem_pos_enc" is the same across frames, so we only need to store one copy of it
        maskmem_pos_enc = self._get_maskmem_pos_enc(inference_state, current_out)
        # object pointer is a small tensor, so we always keep it on GPU memory for fast access
        obj_ptr = current_out["obj_ptr"]
        object_score_logits = current_out["object_score_logits"]
        # make a compact version of this frame's output to reduce the state size
        compact_current_out = {
            "maskmem_features": maskmem_features,
            "maskmem_pos_enc": maskmem_pos_enc,
            "pred_masks": pred_masks,
            "obj_ptr": obj_ptr,
            "object_score_logits": object_score_logits,
        }
        return compact_current_out, pred_masks_gpu

    def _run_memory_encoder(
        self,
        inference_state,
        frame_idx,
        batch_size,
        high_res_masks,
        object_score_logits,
        is_mask_from_pts,
    ):
        """
        Run the memory encoder on `high_res_masks`. This is usually after applying
        non-overlapping constraints to object scores. Since their scores changed, their
        memory also need to be computed again with the memory encoder.
        """
        # Retrieve correct image features
        _, _, current_vision_feats, _, feat_sizes = self._get_image_feature(
            inference_state, frame_idx, batch_size
        )
        maskmem_features, maskmem_pos_enc = self._encode_new_memory(
            current_vision_feats=current_vision_feats,
            feat_sizes=feat_sizes,
            pred_masks_high_res=high_res_masks,
            object_score_logits=object_score_logits,
            is_mask_from_pts=is_mask_from_pts,
        )

        # optionally offload the output to CPU memory to save GPU space
        storage_device = inference_state["storage_device"]
        maskmem_features = maskmem_features.to(torch.bfloat16)
        maskmem_features = maskmem_features.to(storage_device, non_blocking=True)
        # "maskmem_pos_enc" is the same across frames, so we only need to store one copy of it
        maskmem_pos_enc = self._get_maskmem_pos_enc(
            inference_state, {"maskmem_pos_enc": maskmem_pos_enc}
        )
        return maskmem_features, maskmem_pos_enc

    def _get_maskmem_pos_enc(self, inference_state, current_out):
        """
        `maskmem_pos_enc` is the same across frames and objects, so we cache it as
        a constant in the inference session to reduce session storage size.
        """
        model_constants = inference_state["constants"]
        # "out_maskmem_pos_enc" should be either a list of tensors or None
        out_maskmem_pos_enc = current_out["maskmem_pos_enc"]
        if out_maskmem_pos_enc is not None:
            if "maskmem_pos_enc" not in model_constants:
                assert isinstance(out_maskmem_pos_enc, list)
                # only take the slice for one object, since it's same across objects
                maskmem_pos_enc = [x[0:1].clone() for x in out_maskmem_pos_enc]
                model_constants["maskmem_pos_enc"] = maskmem_pos_enc
            else:
                maskmem_pos_enc = model_constants["maskmem_pos_enc"]
            # expand the cached maskmem_pos_enc to the actual batch size
            batch_size = out_maskmem_pos_enc[0].size(0)
            expanded_maskmem_pos_enc = [
                x.expand(batch_size, -1, -1, -1) for x in maskmem_pos_enc
            ]
        else:
            expanded_maskmem_pos_enc = None
        return expanded_maskmem_pos_enc

    @torch.inference_mode()
    def remove_object(self, inference_state, obj_id, strict=False, need_output=True):
        """
        Remove an object id from the tracking state. If strict is True, we check whether
        the object id actually exists and raise an error if it doesn't exist.
        """
        old_obj_idx_to_rm = inference_state["obj_id_to_idx"].get(obj_id, None)
        updated_frames = []
        # Check whether this object_id to remove actually exists and possibly raise an error.
        if old_obj_idx_to_rm is None:
            if not strict:
                return inference_state["obj_ids"], updated_frames
            raise RuntimeError(
                f"Cannot remove object id {obj_id} as it doesn't exist. "
                f"All existing object ids: {inference_state['obj_ids']}."
            )

        # If this is the only remaining object id, we simply reset the state.
        if len(inference_state["obj_id_to_idx"]) == 1:
            self.reset_state(inference_state)
            return inference_state["obj_ids"], updated_frames

        # There are still remaining objects after removing this object id. In this case,
        # we need to delete the object storage from inference state tensors.
        # Step 0: clear the input on those frames where this object id has point or mask input
        # (note that this step is required as it might downgrade conditioning frames to
        # non-conditioning ones)
        obj_input_frames_inds = set()
        obj_input_frames_inds.update(
            inference_state["point_inputs_per_obj"][old_obj_idx_to_rm]
        )
        obj_input_frames_inds.update(
            inference_state["mask_inputs_per_obj"][old_obj_idx_to_rm]
        )
        for frame_idx in obj_input_frames_inds:
            self.clear_all_prompts_in_frame(
                inference_state, frame_idx, obj_id, need_output=False
            )

        # Step 1: Update the object id mapping (note that it must be done after Step 0,
        # since Step 0 still requires the old object id mappings in inference_state)
        old_obj_ids = inference_state["obj_ids"]
        old_obj_inds = list(range(len(old_obj_ids)))
        remain_old_obj_inds = old_obj_inds.copy()
        remain_old_obj_inds.remove(old_obj_idx_to_rm)
        new_obj_ids = [old_obj_ids[old_idx] for old_idx in remain_old_obj_inds]
        new_obj_inds = list(range(len(new_obj_ids)))
        # build new mappings
        old_idx_to_new_idx = dict(zip(remain_old_obj_inds, new_obj_inds))
        inference_state["obj_id_to_idx"] = dict(zip(new_obj_ids, new_obj_inds))
        inference_state["obj_idx_to_id"] = dict(zip(new_obj_inds, new_obj_ids))
        inference_state["obj_ids"] = new_obj_ids

        # Step 2: For per-object tensor storage, we shift their obj_idx in the dict keys.
        # (note that "consolidated_frame_inds" doesn't need to be updated in this step as
        # it's already handled in Step 0)
        def _map_keys(container):
            new_kvs = []
            for k in old_obj_inds:
                v = container.pop(k)
                if k in old_idx_to_new_idx:
                    new_kvs.append((old_idx_to_new_idx[k], v))
            container.update(new_kvs)

        _map_keys(inference_state["point_inputs_per_obj"])
        _map_keys(inference_state["mask_inputs_per_obj"])
        _map_keys(inference_state["output_dict_per_obj"])
        _map_keys(inference_state["temp_output_dict_per_obj"])

        # Step 3: For packed tensor storage, we index the remaining ids and rebuild the per-object slices.
        def _slice_state(output_dict, storage_key):
            for frame_idx, out in output_dict[storage_key].items():
                out["maskmem_features"] = out["maskmem_features"][remain_old_obj_inds]
                out["maskmem_pos_enc"] = [
                    x[remain_old_obj_inds] for x in out["maskmem_pos_enc"]
                ]
                # "maskmem_pos_enc" is the same across frames, so we only need to store one copy of it
                out["maskmem_pos_enc"] = self._get_maskmem_pos_enc(inference_state, out)
                out["pred_masks"] = out["pred_masks"][remain_old_obj_inds]
                out["obj_ptr"] = out["obj_ptr"][remain_old_obj_inds]
                out["object_score_logits"] = out["object_score_logits"][
                    remain_old_obj_inds
                ]
                # also update the per-object slices
                self._add_output_per_object(
                    inference_state, frame_idx, out, storage_key
                )

        _slice_state(inference_state["output_dict"], "cond_frame_outputs")
        _slice_state(inference_state["output_dict"], "non_cond_frame_outputs")

        # Step 4: Further collect the outputs on those frames in `obj_input_frames_inds`, which
        # could show an updated mask for objects previously occluded by the object being removed
        if need_output:
            temp_output_dict_per_obj = inference_state["temp_output_dict_per_obj"]
            for frame_idx in obj_input_frames_inds:
                is_cond = any(
                    frame_idx in obj_temp_output_dict["cond_frame_outputs"]
                    for obj_temp_output_dict in temp_output_dict_per_obj.values()
                )
                consolidated_out = self._consolidate_temp_output_across_obj(
                    inference_state,
                    frame_idx,
                    is_cond=is_cond,
                    run_mem_encoder=False,
                    consolidate_at_video_res=True,
                )
                _, video_res_masks = self._get_orig_video_res_output(
                    inference_state, consolidated_out["pred_masks_video_res"]
                )
                updated_frames.append((frame_idx, video_res_masks))

        return inference_state["obj_ids"], updated_frames

    def _clear_non_cond_mem_around_input(self, inference_state, frame_idx):
        """
        Remove the non-conditioning memory around the input frame. When users provide
        correction clicks, the surrounding frames' non-conditioning memories can still
        contain outdated object appearance information and could confuse the model.

        This method clears those non-conditioning memories surrounding the interacted
        frame to avoid giving the model both old and new information about the object.
        """
        r = self.memory_temporal_stride_for_eval
        frame_idx_begin = frame_idx - r * self.num_maskmem
        frame_idx_end = frame_idx + r * self.num_maskmem
        output_dict = inference_state["output_dict"]
        non_cond_frame_outputs = output_dict["non_cond_frame_outputs"]
        for t in range(frame_idx_begin, frame_idx_end + 1):
            non_cond_frame_outputs.pop(t, None)
            for obj_output_dict in inference_state["output_dict_per_obj"].values():
                obj_output_dict["non_cond_frame_outputs"].pop(t, None)
'''

# Native module: test-code/source/MedSAM2-main/sam2/sam2_video_predictor_npz.py
_EMBEDDED_SOURCES['test-code/source/MedSAM2-main/sam2/sam2_video_predictor_npz.py'] = '''# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import warnings
from collections import OrderedDict

import torch

from tqdm import tqdm

from sam2.modeling.sam2_base import NO_OBJ_SCORE, SAM2Base
from sam2.utils.misc import concat_points, fill_holes_in_mask_scores, load_video_frames


class SAM2VideoPredictorNPZ(SAM2Base):
    """The predictor class to handle user interactions and manage inference states."""

    def __init__(
        self,
        fill_hole_area=0,
        # whether to apply non-overlapping constraints on the output object masks
        non_overlap_masks=False,
        # whether to clear non-conditioning memory of the surrounding frames (which may contain outdated information) after adding correction clicks;
        # note that this would only apply to *single-object tracking* unless `clear_non_cond_mem_for_multi_obj` is also set to True)
        clear_non_cond_mem_around_input=False,
        # whether to also clear non-conditioning memory of the surrounding frames (only effective when `clear_non_cond_mem_around_input` is True).
        clear_non_cond_mem_for_multi_obj=False,
        # if `add_all_frames_to_correct_as_cond` is True, we also append to the conditioning frame list any frame that receives a later correction click
        # if `add_all_frames_to_correct_as_cond` is False, we conditioning frame list to only use those initial conditioning frames
        add_all_frames_to_correct_as_cond=False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.fill_hole_area = fill_hole_area
        self.non_overlap_masks = non_overlap_masks
        self.clear_non_cond_mem_around_input = clear_non_cond_mem_around_input
        self.clear_non_cond_mem_for_multi_obj = clear_non_cond_mem_for_multi_obj
        self.add_all_frames_to_correct_as_cond = add_all_frames_to_correct_as_cond

    @torch.inference_mode()
    def init_state(
        self,
        #video_path,
        # added below
        images,
        video_height,
        video_width,
        offload_video_to_cpu=False,
        offload_state_to_cpu=False,
        async_loading_frames=False,
    ):
        """Initialize an inference state."""
        compute_device = self.device  # device of the model
        # images, video_height, video_width = load_video_frames(
        #     video_path=video_path,
        #     image_size=self.image_size,
        #     offload_video_to_cpu=offload_video_to_cpu,
        #     async_loading_frames=async_loading_frames,
        #     compute_device=compute_device,
        # )
        inference_state = {}
        inference_state["images"] = images
        inference_state["num_frames"] = len(images)
        # whether to offload the video frames to CPU memory
        # turning on this option saves the GPU memory with only a very small overhead
        inference_state["offload_video_to_cpu"] = offload_video_to_cpu
        # whether to offload the inference state to CPU memory
        # turning on this option saves the GPU memory at the cost of a lower tracking fps
        # (e.g. in a test case of 768x768 model, fps dropped from 27 to 24 when tracking one object
        # and from 24 to 21 when tracking two objects)
        inference_state["offload_state_to_cpu"] = offload_state_to_cpu
        # the original video height and width, used for resizing final output scores
        inference_state["video_height"] = video_height
        inference_state["video_width"] = video_width
        inference_state["device"] = compute_device
        if offload_state_to_cpu:
            inference_state["storage_device"] = torch.device("cpu")
        else:
            inference_state["storage_device"] = compute_device
        # inputs on each frame
        inference_state["point_inputs_per_obj"] = {}
        inference_state["mask_inputs_per_obj"] = {}
        # visual features on a small number of recently visited frames for quick interactions
        inference_state["cached_features"] = {}
        # values that don't change across frames (so we only need to hold one copy of them)
        inference_state["constants"] = {}
        # mapping between client-side object id and model-side object index
        inference_state["obj_id_to_idx"] = OrderedDict()
        inference_state["obj_idx_to_id"] = OrderedDict()
        inference_state["obj_ids"] = []
        # A storage to hold the model's tracking results and states on each frame
        inference_state["output_dict"] = {
            "cond_frame_outputs": {},  # dict containing {frame_idx: <out>}
            "non_cond_frame_outputs": {},  # dict containing {frame_idx: <out>}
        }
        # Slice (view) of each object tracking results, sharing the same memory with "output_dict"
        inference_state["output_dict_per_obj"] = {}
        # A temporary storage to hold new outputs when user interact with a frame
        # to add clicks or mask (it's merged into "output_dict" before propagation starts)
        inference_state["temp_output_dict_per_obj"] = {}
        # Frames that already holds consolidated outputs from click or mask inputs
        # (we directly use their consolidated outputs during tracking)
        inference_state["consolidated_frame_inds"] = {
            "cond_frame_outputs": set(),  # set containing frame indices
            "non_cond_frame_outputs": set(),  # set containing frame indices
        }
        # metadata for each tracking frame (e.g. which direction it's tracked)
        inference_state["tracking_has_started"] = False
        inference_state["frames_already_tracked"] = {}
        # Warm up the visual backbone and cache the image feature on frame 0
        self._get_image_feature(inference_state, frame_idx=0, batch_size=1)
        return inference_state

    @classmethod
    def from_pretrained(cls, model_id: str, **kwargs) -> "SAM2VideoPredictor":
        """
        Load a pretrained model from the Hugging Face hub.

        Arguments:
          model_id (str): The Hugging Face repository ID.
          **kwargs: Additional arguments to pass to the model constructor.

        Returns:
          (SAM2VideoPredictor): The loaded model.
        """
        from sam2.build_sam import build_sam2_video_predictor_hf

        sam_model = build_sam2_video_predictor_hf(model_id, **kwargs)
        return sam_model

    def _obj_id_to_idx(self, inference_state, obj_id):
        """Map client-side object id to model-side object index."""
        obj_idx = inference_state["obj_id_to_idx"].get(obj_id, None)
        if obj_idx is not None:
            return obj_idx

        # This is a new object id not sent to the server before. We only allow adding
        # new objects *before* the tracking starts.
        allow_new_object = not inference_state["tracking_has_started"]
        if allow_new_object:
            # get the next object slot
            obj_idx = len(inference_state["obj_id_to_idx"])
            inference_state["obj_id_to_idx"][obj_id] = obj_idx
            inference_state["obj_idx_to_id"][obj_idx] = obj_id
            inference_state["obj_ids"] = list(inference_state["obj_id_to_idx"])
            # set up input and output structures for this object
            inference_state["point_inputs_per_obj"][obj_idx] = {}
            inference_state["mask_inputs_per_obj"][obj_idx] = {}
            inference_state["output_dict_per_obj"][obj_idx] = {
                "cond_frame_outputs": {},  # dict containing {frame_idx: <out>}
                "non_cond_frame_outputs": {},  # dict containing {frame_idx: <out>}
            }
            inference_state["temp_output_dict_per_obj"][obj_idx] = {
                "cond_frame_outputs": {},  # dict containing {frame_idx: <out>}
                "non_cond_frame_outputs": {},  # dict containing {frame_idx: <out>}
            }
            return obj_idx
        else:
            raise RuntimeError(
                f"Cannot add new object id {obj_id} after tracking starts. "
                f"All existing object ids: {inference_state['obj_ids']}. "
                f"Please call 'reset_state' to restart from scratch."
            )

    def _obj_idx_to_id(self, inference_state, obj_idx):
        """Map model-side object index to client-side object id."""
        return inference_state["obj_idx_to_id"][obj_idx]

    def _get_obj_num(self, inference_state):
        """Get the total number of unique object ids received so far in this session."""
        return len(inference_state["obj_idx_to_id"])

    @torch.inference_mode()
    def add_new_points_or_box(
        self,
        inference_state,
        frame_idx,
        obj_id,
        points=None,
        labels=None,
        clear_old_points=True,
        normalize_coords=True,
        box=None,
    ):
        """Add new points to a frame."""
        obj_idx = self._obj_id_to_idx(inference_state, obj_id)
        point_inputs_per_frame = inference_state["point_inputs_per_obj"][obj_idx]
        mask_inputs_per_frame = inference_state["mask_inputs_per_obj"][obj_idx]

        if (points is not None) != (labels is not None):
            raise ValueError("points and labels must be provided together")
        if points is None and box is None:
            raise ValueError("at least one of points or box must be provided as input")

        if points is None:
            points = torch.zeros(0, 2, dtype=torch.float32)
        elif not isinstance(points, torch.Tensor):
            points = torch.tensor(points, dtype=torch.float32)
        if labels is None:
            labels = torch.zeros(0, dtype=torch.int32)
        elif not isinstance(labels, torch.Tensor):
            labels = torch.tensor(labels, dtype=torch.int32)
        if points.dim() == 2:
            points = points.unsqueeze(0)  # add batch dimension
        if labels.dim() == 1:
            labels = labels.unsqueeze(0)  # add batch dimension

        # If `box` is provided, we add it as the first two points with labels 2 and 3
        # along with the user-provided points (consistent with how SAM 2 is trained).
        if box is not None:
            if not clear_old_points:
                raise ValueError(
                    "cannot add box without clearing old points, since "
                    "box prompt must be provided before any point prompt "
                    "(please use clear_old_points=True instead)"
                )
            if inference_state["tracking_has_started"]:
                warnings.warn(
                    "You are adding a box after tracking starts. SAM 2 may not always be "
                    "able to incorporate a box prompt for *refinement*. If you intend to "
                    "use box prompt as an *initial* input before tracking, please call "
                    "'reset_state' on the inference state to restart from scratch.",
                    category=UserWarning,
                    stacklevel=2,
                )
            if not isinstance(box, torch.Tensor):
                box = torch.tensor(box, dtype=torch.float32, device=points.device)
            box_coords = box.reshape(1, 2, 2)
            box_labels = torch.tensor([2, 3], dtype=torch.int32, device=labels.device)
            box_labels = box_labels.reshape(1, 2)
            points = torch.cat([box_coords, points], dim=1)
            labels = torch.cat([box_labels, labels], dim=1)

        if normalize_coords:
            video_H = inference_state["video_height"]
            video_W = inference_state["video_width"]
            points = points / torch.tensor([video_W, video_H]).to(points.device)
        # scale the (normalized) coordinates by the model's internal image size
        points = points * self.image_size
        points = points.to(inference_state["device"])
        labels = labels.to(inference_state["device"])

        if not clear_old_points:
            point_inputs = point_inputs_per_frame.get(frame_idx, None)
        else:
            point_inputs = None
        point_inputs = concat_points(point_inputs, points, labels)

        point_inputs_per_frame[frame_idx] = point_inputs
        mask_inputs_per_frame.pop(frame_idx, None)
        # If this frame hasn't been tracked before, we treat it as an initial conditioning
        # frame, meaning that the inputs points are to generate segments on this frame without
        # using any memory from other frames, like in SAM. Otherwise (if it has been tracked),
        # the input points will be used to correct the already tracked masks.
        is_init_cond_frame = frame_idx not in inference_state["frames_already_tracked"]
        # whether to track in reverse time order
        if is_init_cond_frame:
            reverse = False
        else:
            reverse = inference_state["frames_already_tracked"][frame_idx]["reverse"]
        obj_output_dict = inference_state["output_dict_per_obj"][obj_idx]
        obj_temp_output_dict = inference_state["temp_output_dict_per_obj"][obj_idx]
        # Add a frame to conditioning output if it's an initial conditioning frame or
        # if the model sees all frames receiving clicks/mask as conditioning frames.
        is_cond = is_init_cond_frame or self.add_all_frames_to_correct_as_cond
        storage_key = "cond_frame_outputs" if is_cond else "non_cond_frame_outputs"

        # Get any previously predicted mask logits on this object and feed it along with
        # the new clicks into the SAM mask decoder.
        prev_sam_mask_logits = None
        # lookup temporary output dict first, which contains the most recent output
        # (if not found, then lookup conditioning and non-conditioning frame output)
        prev_out = obj_temp_output_dict[storage_key].get(frame_idx)
        if prev_out is None:
            prev_out = obj_output_dict["cond_frame_outputs"].get(frame_idx)
            if prev_out is None:
                prev_out = obj_output_dict["non_cond_frame_outputs"].get(frame_idx)

        if prev_out is not None and prev_out["pred_masks"] is not None:
            device = inference_state["device"]
            prev_sam_mask_logits = prev_out["pred_masks"].to(device, non_blocking=True)
            # Clamp the scale of prev_sam_mask_logits to avoid rare numerical issues.
            prev_sam_mask_logits = torch.clamp(prev_sam_mask_logits, -32.0, 32.0)
        current_out, _ = self._run_single_frame_inference(
            inference_state=inference_state,
            output_dict=obj_output_dict,  # run on the slice of a single object
            frame_idx=frame_idx,
            batch_size=1,  # run on the slice of a single object
            is_init_cond_frame=is_init_cond_frame,
            point_inputs=point_inputs,
            mask_inputs=None,
            reverse=reverse,
            # Skip the memory encoder when adding clicks or mask. We execute the memory encoder
            # at the beginning of `propagate_in_video` (after user finalize their clicks). This
            # allows us to enforce non-overlapping constraints on all objects before encoding
            # them into memory.
            run_mem_encoder=False,
            prev_sam_mask_logits=prev_sam_mask_logits,
        )
        # Add the output to the output dict (to be used as future memory)
        obj_temp_output_dict[storage_key][frame_idx] = current_out

        # Resize the output mask to the original video resolution
        obj_ids = inference_state["obj_ids"]
        consolidated_out = self._consolidate_temp_output_across_obj(
            inference_state,
            frame_idx,
            is_cond=is_cond,
            run_mem_encoder=False,
            consolidate_at_video_res=True,
        )
        _, video_res_masks = self._get_orig_video_res_output(
            inference_state, consolidated_out["pred_masks_video_res"]
        )
        return frame_idx, obj_ids, video_res_masks

    def add_new_points(self, *args, **kwargs):
        """Deprecated method. Please use `add_new_points_or_box` instead."""
        return self.add_new_points_or_box(*args, **kwargs)

    @torch.inference_mode()
    def add_new_mask(
        self,
        inference_state,
        frame_idx,
        obj_id,
        mask,
    ):
        """Add new mask to a frame."""
        obj_idx = self._obj_id_to_idx(inference_state, obj_id)
        point_inputs_per_frame = inference_state["point_inputs_per_obj"][obj_idx]
        mask_inputs_per_frame = inference_state["mask_inputs_per_obj"][obj_idx]

        if not isinstance(mask, torch.Tensor):
            mask = torch.tensor(mask, dtype=torch.bool)
        assert mask.dim() == 2
        mask_H, mask_W = mask.shape
        mask_inputs_orig = mask[None, None]  # add batch and channel dimension
        mask_inputs_orig = mask_inputs_orig.float().to(inference_state["device"])

        # resize the mask if it doesn't match the model's image size
        if mask_H != self.image_size or mask_W != self.image_size:
            mask_inputs = torch.nn.functional.interpolate(
                mask_inputs_orig,
                size=(self.image_size, self.image_size),
                align_corners=False,
                mode="bilinear",
                antialias=True,  # use antialias for downsampling
            )
            mask_inputs = (mask_inputs >= 0.5).float()
        else:
            mask_inputs = mask_inputs_orig

        mask_inputs_per_frame[frame_idx] = mask_inputs
        point_inputs_per_frame.pop(frame_idx, None)
        # If this frame hasn't been tracked before, we treat it as an initial conditioning
        # frame, meaning that the inputs points are to generate segments on this frame without
        # using any memory from other frames, like in SAM. Otherwise (if it has been tracked),
        # the input points will be used to correct the already tracked masks.
        is_init_cond_frame = frame_idx not in inference_state["frames_already_tracked"]
        # whether to track in reverse time order
        if is_init_cond_frame:
            reverse = False
        else:
            reverse = inference_state["frames_already_tracked"][frame_idx]["reverse"]
        obj_output_dict = inference_state["output_dict_per_obj"][obj_idx]
        obj_temp_output_dict = inference_state["temp_output_dict_per_obj"][obj_idx]
        # Add a frame to conditioning output if it's an initial conditioning frame or
        # if the model sees all frames receiving clicks/mask as conditioning frames.
        is_cond = is_init_cond_frame or self.add_all_frames_to_correct_as_cond
        storage_key = "cond_frame_outputs" if is_cond else "non_cond_frame_outputs"

        current_out, _ = self._run_single_frame_inference(
            inference_state=inference_state,
            output_dict=obj_output_dict,  # run on the slice of a single object
            frame_idx=frame_idx,
            batch_size=1,  # run on the slice of a single object
            is_init_cond_frame=is_init_cond_frame,
            point_inputs=None,
            mask_inputs=mask_inputs,
            reverse=reverse,
            # Skip the memory encoder when adding clicks or mask. We execute the memory encoder
            # at the beginning of `propagate_in_video` (after user finalize their clicks). This
            # allows us to enforce non-overlapping constraints on all objects before encoding
            # them into memory.
            run_mem_encoder=False,
        )
        # Add the output to the output dict (to be used as future memory)
        obj_temp_output_dict[storage_key][frame_idx] = current_out

        # Resize the output mask to the original video resolution
        obj_ids = inference_state["obj_ids"]
        consolidated_out = self._consolidate_temp_output_across_obj(
            inference_state,
            frame_idx,
            is_cond=is_cond,
            run_mem_encoder=False,
            consolidate_at_video_res=True,
        )
        _, video_res_masks = self._get_orig_video_res_output(
            inference_state, consolidated_out["pred_masks_video_res"]
        )
        return frame_idx, obj_ids, video_res_masks

    def _get_orig_video_res_output(self, inference_state, any_res_masks):
        """
        Resize the object scores to the original video resolution (video_res_masks)
        and apply non-overlapping constraints for final output.
        """
        device = inference_state["device"]
        video_H = inference_state["video_height"]
        video_W = inference_state["video_width"]
        any_res_masks = any_res_masks.to(device, non_blocking=True)
        if any_res_masks.shape[-2:] == (video_H, video_W):
            video_res_masks = any_res_masks
        else:
            video_res_masks = torch.nn.functional.interpolate(
                any_res_masks,
                size=(video_H, video_W),
                mode="bilinear",
                align_corners=False,
            )
        if self.non_overlap_masks:
            video_res_masks = self._apply_non_overlapping_constraints(video_res_masks)
        return any_res_masks, video_res_masks

    def _consolidate_temp_output_across_obj(
        self,
        inference_state,
        frame_idx,
        is_cond,
        run_mem_encoder,
        consolidate_at_video_res=False,
    ):
        """
        Consolidate the per-object temporary outputs in `temp_output_dict_per_obj` on
        a frame into a single output for all objects, including
        1) fill any missing objects either from `output_dict_per_obj` (if they exist in
           `output_dict_per_obj` for this frame) or leave them as placeholder values
           (if they don't exist in `output_dict_per_obj` for this frame);
        2) if specified, rerun memory encoder after apply non-overlapping constraints
           on the object scores.
        """
        batch_size = self._get_obj_num(inference_state)
        storage_key = "cond_frame_outputs" if is_cond else "non_cond_frame_outputs"
        # Optionally, we allow consolidating the temporary outputs at the original
        # video resolution (to provide a better editing experience for mask prompts).
        if consolidate_at_video_res:
            assert not run_mem_encoder, "memory encoder cannot run at video resolution"
            consolidated_H = inference_state["video_height"]
            consolidated_W = inference_state["video_width"]
            consolidated_mask_key = "pred_masks_video_res"
        else:
            consolidated_H = consolidated_W = self.image_size // 4
            consolidated_mask_key = "pred_masks"

        # Initialize `consolidated_out`. Its "maskmem_features" and "maskmem_pos_enc"
        # will be added when rerunning the memory encoder after applying non-overlapping
        # constraints to object scores. Its "pred_masks" are prefilled with a large
        # negative value (NO_OBJ_SCORE) to represent missing objects.
        consolidated_out = {
            "maskmem_features": None,
            "maskmem_pos_enc": None,
            consolidated_mask_key: torch.full(
                size=(batch_size, 1, consolidated_H, consolidated_W),
                fill_value=NO_OBJ_SCORE,
                dtype=torch.float32,
                device=inference_state["storage_device"],
            ),
            "obj_ptr": torch.full(
                size=(batch_size, self.hidden_dim),
                fill_value=NO_OBJ_SCORE,
                dtype=torch.float32,
                device=inference_state["device"],
            ),
            "object_score_logits": torch.full(
                size=(batch_size, 1),
                # default to 10.0 for object_score_logits, i.e. assuming the object is
                # present as sigmoid(10)=1, same as in `predict_masks` of `MaskDecoder`
                fill_value=10.0,
                dtype=torch.float32,
                device=inference_state["device"],
            ),
        }
        empty_mask_ptr = None
        for obj_idx in range(batch_size):
            obj_temp_output_dict = inference_state["temp_output_dict_per_obj"][obj_idx]
            obj_output_dict = inference_state["output_dict_per_obj"][obj_idx]
            out = obj_temp_output_dict[storage_key].get(frame_idx, None)
            # If the object doesn't appear in "temp_output_dict_per_obj" on this frame,
            # we fall back and look up its previous output in "output_dict_per_obj".
            # We look up both "cond_frame_outputs" and "non_cond_frame_outputs" in
            # "output_dict_per_obj" to find a previous output for this object.
            if out is None:
                out = obj_output_dict["cond_frame_outputs"].get(frame_idx, None)
            if out is None:
                out = obj_output_dict["non_cond_frame_outputs"].get(frame_idx, None)
            # If the object doesn't appear in "output_dict_per_obj" either, we skip it
            # and leave its mask scores to the default scores (i.e. the NO_OBJ_SCORE
            # placeholder above) and set its object pointer to be a dummy pointer.
            if out is None:
                # Fill in dummy object pointers for those objects without any inputs or
                # tracking outcomes on this frame (only do it under `run_mem_encoder=True`,
                # i.e. when we need to build the memory for tracking).
                if run_mem_encoder:
                    if empty_mask_ptr is None:
                        empty_mask_ptr = self._get_empty_mask_ptr(
                            inference_state, frame_idx
                        )
                    # fill object pointer with a dummy pointer (based on an empty mask)
                    consolidated_out["obj_ptr"][obj_idx : obj_idx + 1] = empty_mask_ptr
                continue
            # Add the temporary object output mask to consolidated output mask
            obj_mask = out["pred_masks"]
            consolidated_pred_masks = consolidated_out[consolidated_mask_key]
            if obj_mask.shape[-2:] == consolidated_pred_masks.shape[-2:]:
                consolidated_pred_masks[obj_idx : obj_idx + 1] = obj_mask
            else:
                # Resize first if temporary object mask has a different resolution
                resized_obj_mask = torch.nn.functional.interpolate(
                    obj_mask,
                    size=consolidated_pred_masks.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )
                consolidated_pred_masks[obj_idx : obj_idx + 1] = resized_obj_mask
            consolidated_out["obj_ptr"][obj_idx : obj_idx + 1] = out["obj_ptr"]
            consolidated_out["object_score_logits"][obj_idx : obj_idx + 1] = out[
                "object_score_logits"
            ]

        # Optionally, apply non-overlapping constraints on the consolidated scores
        # and rerun the memory encoder
        if run_mem_encoder:
            device = inference_state["device"]
            high_res_masks = torch.nn.functional.interpolate(
                consolidated_out["pred_masks"].to(device, non_blocking=True),
                size=(self.image_size, self.image_size),
                mode="bilinear",
                align_corners=False,
            )
            if self.non_overlap_masks_for_mem_enc:
                high_res_masks = self._apply_non_overlapping_constraints(high_res_masks)
            maskmem_features, maskmem_pos_enc = self._run_memory_encoder(
                inference_state=inference_state,
                frame_idx=frame_idx,
                batch_size=batch_size,
                high_res_masks=high_res_masks,
                object_score_logits=consolidated_out["object_score_logits"],
                is_mask_from_pts=True,  # these frames are what the user interacted with
            )
            consolidated_out["maskmem_features"] = maskmem_features
            consolidated_out["maskmem_pos_enc"] = maskmem_pos_enc

        return consolidated_out

    def _get_empty_mask_ptr(self, inference_state, frame_idx):
        """Get a dummy object pointer based on an empty mask on the current frame."""
        # A dummy (empty) mask with a single object
        batch_size = 1
        mask_inputs = torch.zeros(
            (batch_size, 1, self.image_size, self.image_size),
            dtype=torch.float32,
            device=inference_state["device"],
        )

        # Retrieve correct image features
        (
            _,
            _,
            current_vision_feats,
            current_vision_pos_embeds,
            feat_sizes,
        ) = self._get_image_feature(inference_state, frame_idx, batch_size)

        # Feed the empty mask and image feature above to get a dummy object pointer
        current_out = self.track_step(
            frame_idx=frame_idx,
            is_init_cond_frame=True,
            current_vision_feats=current_vision_feats,
            current_vision_pos_embeds=current_vision_pos_embeds,
            feat_sizes=feat_sizes,
            point_inputs=None,
            mask_inputs=mask_inputs,
            output_dict={},
            num_frames=inference_state["num_frames"],
            track_in_reverse=False,
            run_mem_encoder=False,
            prev_sam_mask_logits=None,
        )
        return current_out["obj_ptr"]

    @torch.inference_mode()
    def propagate_in_video_preflight(self, inference_state):
        """Prepare inference_state and consolidate temporary outputs before tracking."""
        # Tracking has started and we don't allow adding new objects until session is reset.
        inference_state["tracking_has_started"] = True
        batch_size = self._get_obj_num(inference_state)

        # Consolidate per-object temporary outputs in "temp_output_dict_per_obj" and
        # add them into "output_dict".
        temp_output_dict_per_obj = inference_state["temp_output_dict_per_obj"]
        output_dict = inference_state["output_dict"]
        # "consolidated_frame_inds" contains indices of those frames where consolidated
        # temporary outputs have been added (either in this call or any previous calls
        # to `propagate_in_video_preflight`).
        consolidated_frame_inds = inference_state["consolidated_frame_inds"]
        for is_cond in [False, True]:
            # Separately consolidate conditioning and non-conditioning temp outputs
            storage_key = "cond_frame_outputs" if is_cond else "non_cond_frame_outputs"
            # Find all the frames that contain temporary outputs for any objects
            # (these should be the frames that have just received clicks for mask inputs
            # via `add_new_points_or_box` or `add_new_mask`)
            temp_frame_inds = set()
            for obj_temp_output_dict in temp_output_dict_per_obj.values():
                temp_frame_inds.update(obj_temp_output_dict[storage_key].keys())
            consolidated_frame_inds[storage_key].update(temp_frame_inds)
            # consolidate the temporary output across all objects on this frame
            for frame_idx in temp_frame_inds:
                consolidated_out = self._consolidate_temp_output_across_obj(
                    inference_state, frame_idx, is_cond=is_cond, run_mem_encoder=True
                )
                # merge them into "output_dict" and also create per-object slices
                output_dict[storage_key][frame_idx] = consolidated_out
                self._add_output_per_object(
                    inference_state, frame_idx, consolidated_out, storage_key
                )
                clear_non_cond_mem = self.clear_non_cond_mem_around_input and (
                    self.clear_non_cond_mem_for_multi_obj or batch_size <= 1
                )
                if clear_non_cond_mem:
                    # clear non-conditioning memory of the surrounding frames
                    self._clear_non_cond_mem_around_input(inference_state, frame_idx)

            # clear temporary outputs in `temp_output_dict_per_obj`
            for obj_temp_output_dict in temp_output_dict_per_obj.values():
                obj_temp_output_dict[storage_key].clear()

        # edge case: if an output is added to "cond_frame_outputs", we remove any prior
        # output on the same frame in "non_cond_frame_outputs"
        for frame_idx in output_dict["cond_frame_outputs"]:
            output_dict["non_cond_frame_outputs"].pop(frame_idx, None)
        for obj_output_dict in inference_state["output_dict_per_obj"].values():
            for frame_idx in obj_output_dict["cond_frame_outputs"]:
                obj_output_dict["non_cond_frame_outputs"].pop(frame_idx, None)
        for frame_idx in consolidated_frame_inds["cond_frame_outputs"]:
            assert frame_idx in output_dict["cond_frame_outputs"]
            consolidated_frame_inds["non_cond_frame_outputs"].discard(frame_idx)

        # Make sure that the frame indices in "consolidated_frame_inds" are exactly those frames
        # with either points or mask inputs (which should be true under a correct workflow).
        all_consolidated_frame_inds = (
            consolidated_frame_inds["cond_frame_outputs"]
            | consolidated_frame_inds["non_cond_frame_outputs"]
        )
        input_frames_inds = set()
        for point_inputs_per_frame in inference_state["point_inputs_per_obj"].values():
            input_frames_inds.update(point_inputs_per_frame.keys())
        for mask_inputs_per_frame in inference_state["mask_inputs_per_obj"].values():
            input_frames_inds.update(mask_inputs_per_frame.keys())
        assert all_consolidated_frame_inds == input_frames_inds

    @torch.inference_mode()
    def propagate_in_video(
        self,
        inference_state,
        start_frame_idx=None,
        max_frame_num_to_track=None,
        reverse=False,
    ):
        """Propagate the input points across frames to track in the entire video."""
        self.propagate_in_video_preflight(inference_state)

        output_dict = inference_state["output_dict"]
        consolidated_frame_inds = inference_state["consolidated_frame_inds"]
        obj_ids = inference_state["obj_ids"]
        num_frames = inference_state["num_frames"]
        batch_size = self._get_obj_num(inference_state)
        if len(output_dict["cond_frame_outputs"]) == 0:
            raise RuntimeError("No points are provided; please add points first")
        clear_non_cond_mem = self.clear_non_cond_mem_around_input and (
            self.clear_non_cond_mem_for_multi_obj or batch_size <= 1
        )

        # set start index, end index, and processing order
        if start_frame_idx is None:
            # default: start from the earliest frame with input points
            start_frame_idx = min(output_dict["cond_frame_outputs"])
        if max_frame_num_to_track is None:
            # default: track all the frames in the video
            max_frame_num_to_track = num_frames
        if reverse:
            end_frame_idx = max(start_frame_idx - max_frame_num_to_track, 0)
            if start_frame_idx > 0:
                processing_order = range(start_frame_idx, end_frame_idx - 1, -1)
            else:
                processing_order = []  # skip reverse tracking if starting from frame 0
        else:
            end_frame_idx = min(
                start_frame_idx + max_frame_num_to_track, num_frames - 1
            )
            processing_order = range(start_frame_idx, end_frame_idx + 1)

        for frame_idx in tqdm(processing_order, desc="propagate in video"):
            # We skip those frames already in consolidated outputs (these are frames
            # that received input clicks or mask). Note that we cannot directly run
            # batched forward on them via `_run_single_frame_inference` because the
            # number of clicks on each object might be different.
            if frame_idx in consolidated_frame_inds["cond_frame_outputs"]:
                storage_key = "cond_frame_outputs"
                current_out = output_dict[storage_key][frame_idx]
                pred_masks = current_out["pred_masks"]
                if clear_non_cond_mem:
                    # clear non-conditioning memory of the surrounding frames
                    self._clear_non_cond_mem_around_input(inference_state, frame_idx)
            elif frame_idx in consolidated_frame_inds["non_cond_frame_outputs"]:
                storage_key = "non_cond_frame_outputs"
                current_out = output_dict[storage_key][frame_idx]
                pred_masks = current_out["pred_masks"]
            else:
                storage_key = "non_cond_frame_outputs"
                current_out, pred_masks = self._run_single_frame_inference(
                    inference_state=inference_state,
                    output_dict=output_dict,
                    frame_idx=frame_idx,
                    batch_size=batch_size,
                    is_init_cond_frame=False,
                    point_inputs=None,
                    mask_inputs=None,
                    reverse=reverse,
                    run_mem_encoder=True,
                )
                output_dict[storage_key][frame_idx] = current_out
            # Create slices of per-object outputs for subsequent interaction with each
            # individual object after tracking.
            self._add_output_per_object(
                inference_state, frame_idx, current_out, storage_key
            )
            inference_state["frames_already_tracked"][frame_idx] = {"reverse": reverse}

            # Resize the output mask to the original video resolution (we directly use
            # the mask scores on GPU for output to avoid any CPU conversion in between)
            _, video_res_masks = self._get_orig_video_res_output(
                inference_state, pred_masks
            )
            yield frame_idx, obj_ids, video_res_masks

    def _add_output_per_object(
        self, inference_state, frame_idx, current_out, storage_key
    ):
        """
        Split a multi-object output into per-object output slices and add them into
        `output_dict_per_obj`. The resulting slices share the same tensor storage.
        """
        maskmem_features = current_out["maskmem_features"]
        assert maskmem_features is None or isinstance(maskmem_features, torch.Tensor)

        maskmem_pos_enc = current_out["maskmem_pos_enc"]
        assert maskmem_pos_enc is None or isinstance(maskmem_pos_enc, list)

        output_dict_per_obj = inference_state["output_dict_per_obj"]
        for obj_idx, obj_output_dict in output_dict_per_obj.items():
            obj_slice = slice(obj_idx, obj_idx + 1)
            obj_out = {
                "maskmem_features": None,
                "maskmem_pos_enc": None,
                "pred_masks": current_out["pred_masks"][obj_slice],
                "obj_ptr": current_out["obj_ptr"][obj_slice],
                "object_score_logits": current_out["object_score_logits"][obj_slice],
            }
            if maskmem_features is not None:
                obj_out["maskmem_features"] = maskmem_features[obj_slice]
            if maskmem_pos_enc is not None:
                obj_out["maskmem_pos_enc"] = [x[obj_slice] for x in maskmem_pos_enc]
            obj_output_dict[storage_key][frame_idx] = obj_out

    @torch.inference_mode()
    def clear_all_prompts_in_frame(
        self, inference_state, frame_idx, obj_id, need_output=True
    ):
        """Remove all input points or mask in a specific frame for a given object."""
        obj_idx = self._obj_id_to_idx(inference_state, obj_id)

        # Clear the conditioning information on the given frame
        inference_state["point_inputs_per_obj"][obj_idx].pop(frame_idx, None)
        inference_state["mask_inputs_per_obj"][obj_idx].pop(frame_idx, None)

        temp_output_dict_per_obj = inference_state["temp_output_dict_per_obj"]
        temp_output_dict_per_obj[obj_idx]["cond_frame_outputs"].pop(frame_idx, None)
        temp_output_dict_per_obj[obj_idx]["non_cond_frame_outputs"].pop(frame_idx, None)

        # Check and see if there are still any inputs left on this frame
        batch_size = self._get_obj_num(inference_state)
        frame_has_input = False
        for obj_idx2 in range(batch_size):
            if frame_idx in inference_state["point_inputs_per_obj"][obj_idx2]:
                frame_has_input = True
                break
            if frame_idx in inference_state["mask_inputs_per_obj"][obj_idx2]:
                frame_has_input = True
                break

        # If this frame has no remaining inputs for any objects, we further clear its
        # conditioning frame status
        if not frame_has_input:
            output_dict = inference_state["output_dict"]
            consolidated_frame_inds = inference_state["consolidated_frame_inds"]
            consolidated_frame_inds["cond_frame_outputs"].discard(frame_idx)
            consolidated_frame_inds["non_cond_frame_outputs"].discard(frame_idx)
            # Remove the frame's conditioning output (possibly downgrading it to non-conditioning)
            out = output_dict["cond_frame_outputs"].pop(frame_idx, None)
            if out is not None:
                # The frame is not a conditioning frame anymore since it's not receiving inputs,
                # so we "downgrade" its output (if exists) to a non-conditioning frame output.
                output_dict["non_cond_frame_outputs"][frame_idx] = out
                inference_state["frames_already_tracked"].pop(frame_idx, None)
            # Similarly, do it for the sliced output on each object.
            for obj_idx2 in range(batch_size):
                obj_output_dict = inference_state["output_dict_per_obj"][obj_idx2]
                obj_out = obj_output_dict["cond_frame_outputs"].pop(frame_idx, None)
                if obj_out is not None:
                    obj_output_dict["non_cond_frame_outputs"][frame_idx] = obj_out

            # If all the conditioning frames have been removed, we also clear the tracking outputs
            if len(output_dict["cond_frame_outputs"]) == 0:
                self._reset_tracking_results(inference_state)

        if not need_output:
            return
        # Finally, output updated masks per object (after removing the inputs above)
        obj_ids = inference_state["obj_ids"]
        is_cond = any(
            frame_idx in obj_temp_output_dict["cond_frame_outputs"]
            for obj_temp_output_dict in temp_output_dict_per_obj.values()
        )
        consolidated_out = self._consolidate_temp_output_across_obj(
            inference_state,
            frame_idx,
            is_cond=is_cond,
            run_mem_encoder=False,
            consolidate_at_video_res=True,
        )
        _, video_res_masks = self._get_orig_video_res_output(
            inference_state, consolidated_out["pred_masks_video_res"]
        )
        return frame_idx, obj_ids, video_res_masks

    @torch.inference_mode()
    def reset_state(self, inference_state):
        """Remove all input points or mask in all frames throughout the video."""
        self._reset_tracking_results(inference_state)
        # Remove all object ids
        inference_state["obj_id_to_idx"].clear()
        inference_state["obj_idx_to_id"].clear()
        inference_state["obj_ids"].clear()
        inference_state["point_inputs_per_obj"].clear()
        inference_state["mask_inputs_per_obj"].clear()
        inference_state["output_dict_per_obj"].clear()
        inference_state["temp_output_dict_per_obj"].clear()

    def _reset_tracking_results(self, inference_state):
        """Reset all tracking inputs and results across the videos."""
        for v in inference_state["point_inputs_per_obj"].values():
            v.clear()
        for v in inference_state["mask_inputs_per_obj"].values():
            v.clear()
        for v in inference_state["output_dict_per_obj"].values():
            v["cond_frame_outputs"].clear()
            v["non_cond_frame_outputs"].clear()
        for v in inference_state["temp_output_dict_per_obj"].values():
            v["cond_frame_outputs"].clear()
            v["non_cond_frame_outputs"].clear()
        inference_state["output_dict"]["cond_frame_outputs"].clear()
        inference_state["output_dict"]["non_cond_frame_outputs"].clear()
        inference_state["consolidated_frame_inds"]["cond_frame_outputs"].clear()
        inference_state["consolidated_frame_inds"]["non_cond_frame_outputs"].clear()
        inference_state["tracking_has_started"] = False
        inference_state["frames_already_tracked"].clear()

    def _get_image_feature(self, inference_state, frame_idx, batch_size):
        """Compute the image features on a given frame."""
        # Look up in the cache first
        image, backbone_out = inference_state["cached_features"].get(
            frame_idx, (None, None)
        )
        if backbone_out is None:
            # Cache miss -- we will run inference on a single image
            device = inference_state["device"]
            image = inference_state["images"][frame_idx].to(device).float().unsqueeze(0)
            backbone_out = self.forward_image(image)
            # Cache the most recent frame's feature (for repeated interactions with
            # a frame; we can use an LRU cache for more frames in the future).
            inference_state["cached_features"] = {frame_idx: (image, backbone_out)}

        # expand the features to have the same dimension as the number of objects
        expanded_image = image.expand(batch_size, -1, -1, -1)
        expanded_backbone_out = {
            "backbone_fpn": backbone_out["backbone_fpn"].copy(),
            "vision_pos_enc": backbone_out["vision_pos_enc"].copy(),
        }
        for i, feat in enumerate(expanded_backbone_out["backbone_fpn"]):
            expanded_backbone_out["backbone_fpn"][i] = feat.expand(
                batch_size, -1, -1, -1
            )
        for i, pos in enumerate(expanded_backbone_out["vision_pos_enc"]):
            pos = pos.expand(batch_size, -1, -1, -1)
            expanded_backbone_out["vision_pos_enc"][i] = pos

        features = self._prepare_backbone_features(expanded_backbone_out)
        features = (expanded_image,) + features
        return features

    def _run_single_frame_inference(
        self,
        inference_state,
        output_dict,
        frame_idx,
        batch_size,
        is_init_cond_frame,
        point_inputs,
        mask_inputs,
        reverse,
        run_mem_encoder,
        prev_sam_mask_logits=None,
    ):
        """Run tracking on a single frame based on current inputs and previous memory."""
        # Retrieve correct image features
        (
            _,
            _,
            current_vision_feats,
            current_vision_pos_embeds,
            feat_sizes,
        ) = self._get_image_feature(inference_state, frame_idx, batch_size)

        # point and mask should not appear as input simultaneously on the same frame
        assert point_inputs is None or mask_inputs is None
        current_out = self.track_step(
            frame_idx=frame_idx,
            is_init_cond_frame=is_init_cond_frame,
            current_vision_feats=current_vision_feats,
            current_vision_pos_embeds=current_vision_pos_embeds,
            feat_sizes=feat_sizes,
            point_inputs=point_inputs,
            mask_inputs=mask_inputs,
            output_dict=output_dict,
            num_frames=inference_state["num_frames"],
            track_in_reverse=reverse,
            run_mem_encoder=run_mem_encoder,
            prev_sam_mask_logits=prev_sam_mask_logits,
        )

        # optionally offload the output to CPU memory to save GPU space
        storage_device = inference_state["storage_device"]
        maskmem_features = current_out["maskmem_features"]
        if maskmem_features is not None:
            maskmem_features = maskmem_features.to(torch.bfloat16)
            maskmem_features = maskmem_features.to(storage_device, non_blocking=True)
        pred_masks_gpu = current_out["pred_masks"]
        # potentially fill holes in the predicted masks
        if self.fill_hole_area > 0:
            pred_masks_gpu = fill_holes_in_mask_scores(
                pred_masks_gpu, self.fill_hole_area
            )
        pred_masks = pred_masks_gpu.to(storage_device, non_blocking=True)
        # "maskmem_pos_enc" is the same across frames, so we only need to store one copy of it
        maskmem_pos_enc = self._get_maskmem_pos_enc(inference_state, current_out)
        # object pointer is a small tensor, so we always keep it on GPU memory for fast access
        obj_ptr = current_out["obj_ptr"]
        object_score_logits = current_out["object_score_logits"]
        # make a compact version of this frame's output to reduce the state size
        compact_current_out = {
            "maskmem_features": maskmem_features,
            "maskmem_pos_enc": maskmem_pos_enc,
            "pred_masks": pred_masks,
            "obj_ptr": obj_ptr,
            "object_score_logits": object_score_logits,
        }
        return compact_current_out, pred_masks_gpu

    def _run_memory_encoder(
        self,
        inference_state,
        frame_idx,
        batch_size,
        high_res_masks,
        object_score_logits,
        is_mask_from_pts,
    ):
        """
        Run the memory encoder on `high_res_masks`. This is usually after applying
        non-overlapping constraints to object scores. Since their scores changed, their
        memory also need to be computed again with the memory encoder.
        """
        # Retrieve correct image features
        _, _, current_vision_feats, _, feat_sizes = self._get_image_feature(
            inference_state, frame_idx, batch_size
        )
        maskmem_features, maskmem_pos_enc = self._encode_new_memory(
            current_vision_feats=current_vision_feats,
            feat_sizes=feat_sizes,
            pred_masks_high_res=high_res_masks,
            object_score_logits=object_score_logits,
            is_mask_from_pts=is_mask_from_pts,
        )

        # optionally offload the output to CPU memory to save GPU space
        storage_device = inference_state["storage_device"]
        maskmem_features = maskmem_features.to(torch.bfloat16)
        maskmem_features = maskmem_features.to(storage_device, non_blocking=True)
        # "maskmem_pos_enc" is the same across frames, so we only need to store one copy of it
        maskmem_pos_enc = self._get_maskmem_pos_enc(
            inference_state, {"maskmem_pos_enc": maskmem_pos_enc}
        )
        return maskmem_features, maskmem_pos_enc

    def _get_maskmem_pos_enc(self, inference_state, current_out):
        """
        `maskmem_pos_enc` is the same across frames and objects, so we cache it as
        a constant in the inference session to reduce session storage size.
        """
        model_constants = inference_state["constants"]
        # "out_maskmem_pos_enc" should be either a list of tensors or None
        out_maskmem_pos_enc = current_out["maskmem_pos_enc"]
        if out_maskmem_pos_enc is not None:
            if "maskmem_pos_enc" not in model_constants:
                assert isinstance(out_maskmem_pos_enc, list)
                # only take the slice for one object, since it's same across objects
                maskmem_pos_enc = [x[0:1].clone() for x in out_maskmem_pos_enc]
                model_constants["maskmem_pos_enc"] = maskmem_pos_enc
            else:
                maskmem_pos_enc = model_constants["maskmem_pos_enc"]
            # expand the cached maskmem_pos_enc to the actual batch size
            batch_size = out_maskmem_pos_enc[0].size(0)
            expanded_maskmem_pos_enc = [
                x.expand(batch_size, -1, -1, -1) for x in maskmem_pos_enc
            ]
        else:
            expanded_maskmem_pos_enc = None
        return expanded_maskmem_pos_enc

    @torch.inference_mode()
    def remove_object(self, inference_state, obj_id, strict=False, need_output=True):
        """
        Remove an object id from the tracking state. If strict is True, we check whether
        the object id actually exists and raise an error if it doesn't exist.
        """
        old_obj_idx_to_rm = inference_state["obj_id_to_idx"].get(obj_id, None)
        updated_frames = []
        # Check whether this object_id to remove actually exists and possibly raise an error.
        if old_obj_idx_to_rm is None:
            if not strict:
                return inference_state["obj_ids"], updated_frames
            raise RuntimeError(
                f"Cannot remove object id {obj_id} as it doesn't exist. "
                f"All existing object ids: {inference_state['obj_ids']}."
            )

        # If this is the only remaining object id, we simply reset the state.
        if len(inference_state["obj_id_to_idx"]) == 1:
            self.reset_state(inference_state)
            return inference_state["obj_ids"], updated_frames

        # There are still remaining objects after removing this object id. In this case,
        # we need to delete the object storage from inference state tensors.
        # Step 0: clear the input on those frames where this object id has point or mask input
        # (note that this step is required as it might downgrade conditioning frames to
        # non-conditioning ones)
        obj_input_frames_inds = set()
        obj_input_frames_inds.update(
            inference_state["point_inputs_per_obj"][old_obj_idx_to_rm]
        )
        obj_input_frames_inds.update(
            inference_state["mask_inputs_per_obj"][old_obj_idx_to_rm]
        )
        for frame_idx in obj_input_frames_inds:
            self.clear_all_prompts_in_frame(
                inference_state, frame_idx, obj_id, need_output=False
            )

        # Step 1: Update the object id mapping (note that it must be done after Step 0,
        # since Step 0 still requires the old object id mappings in inference_state)
        old_obj_ids = inference_state["obj_ids"]
        old_obj_inds = list(range(len(old_obj_ids)))
        remain_old_obj_inds = old_obj_inds.copy()
        remain_old_obj_inds.remove(old_obj_idx_to_rm)
        new_obj_ids = [old_obj_ids[old_idx] for old_idx in remain_old_obj_inds]
        new_obj_inds = list(range(len(new_obj_ids)))
        # build new mappings
        old_idx_to_new_idx = dict(zip(remain_old_obj_inds, new_obj_inds))
        inference_state["obj_id_to_idx"] = dict(zip(new_obj_ids, new_obj_inds))
        inference_state["obj_idx_to_id"] = dict(zip(new_obj_inds, new_obj_ids))
        inference_state["obj_ids"] = new_obj_ids

        # Step 2: For per-object tensor storage, we shift their obj_idx in the dict keys.
        # (note that "consolidated_frame_inds" doesn't need to be updated in this step as
        # it's already handled in Step 0)
        def _map_keys(container):
            new_kvs = []
            for k in old_obj_inds:
                v = container.pop(k)
                if k in old_idx_to_new_idx:
                    new_kvs.append((old_idx_to_new_idx[k], v))
            container.update(new_kvs)

        _map_keys(inference_state["point_inputs_per_obj"])
        _map_keys(inference_state["mask_inputs_per_obj"])
        _map_keys(inference_state["output_dict_per_obj"])
        _map_keys(inference_state["temp_output_dict_per_obj"])

        # Step 3: For packed tensor storage, we index the remaining ids and rebuild the per-object slices.
        def _slice_state(output_dict, storage_key):
            for frame_idx, out in output_dict[storage_key].items():
                out["maskmem_features"] = out["maskmem_features"][remain_old_obj_inds]
                out["maskmem_pos_enc"] = [
                    x[remain_old_obj_inds] for x in out["maskmem_pos_enc"]
                ]
                # "maskmem_pos_enc" is the same across frames, so we only need to store one copy of it
                out["maskmem_pos_enc"] = self._get_maskmem_pos_enc(inference_state, out)
                out["pred_masks"] = out["pred_masks"][remain_old_obj_inds]
                out["obj_ptr"] = out["obj_ptr"][remain_old_obj_inds]
                out["object_score_logits"] = out["object_score_logits"][
                    remain_old_obj_inds
                ]
                # also update the per-object slices
                self._add_output_per_object(
                    inference_state, frame_idx, out, storage_key
                )

        _slice_state(inference_state["output_dict"], "cond_frame_outputs")
        _slice_state(inference_state["output_dict"], "non_cond_frame_outputs")

        # Step 4: Further collect the outputs on those frames in `obj_input_frames_inds`, which
        # could show an updated mask for objects previously occluded by the object being removed
        if need_output:
            temp_output_dict_per_obj = inference_state["temp_output_dict_per_obj"]
            for frame_idx in obj_input_frames_inds:
                is_cond = any(
                    frame_idx in obj_temp_output_dict["cond_frame_outputs"]
                    for obj_temp_output_dict in temp_output_dict_per_obj.values()
                )
                consolidated_out = self._consolidate_temp_output_across_obj(
                    inference_state,
                    frame_idx,
                    is_cond=is_cond,
                    run_mem_encoder=False,
                    consolidate_at_video_res=True,
                )
                _, video_res_masks = self._get_orig_video_res_output(
                    inference_state, consolidated_out["pred_masks_video_res"]
                )
                updated_frames.append((frame_idx, video_res_masks))

        return inference_state["obj_ids"], updated_frames

    def _clear_non_cond_mem_around_input(self, inference_state, frame_idx):
        """
        Remove the non-conditioning memory around the input frame. When users provide
        correction clicks, the surrounding frames' non-conditioning memories can still
        contain outdated object appearance information and could confuse the model.

        This method clears those non-conditioning memories surrounding the interacted
        frame to avoid giving the model both old and new information about the object.
        """
        r = self.memory_temporal_stride_for_eval
        frame_idx_begin = frame_idx - r * self.num_maskmem
        frame_idx_end = frame_idx + r * self.num_maskmem
        output_dict = inference_state["output_dict"]
        non_cond_frame_outputs = output_dict["non_cond_frame_outputs"]
        for t in range(frame_idx_begin, frame_idx_end + 1):
            non_cond_frame_outputs.pop(t, None)
            for obj_output_dict in inference_state["output_dict_per_obj"].values():
                obj_output_dict["non_cond_frame_outputs"].pop(t, None)

'''

# Native module: test-code/source/MedSAM2-main/sam2/sam2_video_trainer.py
_EMBEDDED_SOURCES['test-code/source/MedSAM2-main/sam2/sam2_video_trainer.py'] = '''import unittest

import torch
import torch.nn as nn
import torch.nn.functional as F
from sam2.modeling.sam2_base import SAM2Base
from sam2.build_sam import build_sam2_video_predictor
from sam2.utils.transforms import SAM2Transforms
from sam2.modeling.sam2_utils import get_1d_sine_pe

class SAM2VideoTrainer(nn.Module):
    """
    SAM2VideoTrainer is a PyTorch module for training a video segmentation model using SAM2.
    Attributes:
        device (torch.device): The device to run the model on.
        model (nn.Module): The SAM2 video predictor model.
        num_feature_levels (int): Number of feature levels in the model.
        memory_size (int): Size of the memory for storing features.
        _transforms (SAM2Transforms): Transformations applied to the input data.
        _bb_feat_sizes (list): Spatial dimensions for backbone feature maps.
        num_maskmem (int): Number of mask memory features.
        sam_point_coords (torch.Tensor): Placeholder for SAM point coordinates.
        sam_point_labels (torch.Tensor): Placeholder for SAM point labels.
        _orig_hw (list): Original height and width of the input frames.
        maskmem_features (list): List of mask memory features.
        maskmem_pos_enc (list): List of mask memory positional encodings.
        batch_size (int): Batch size of the input data.
        obj_ptrs (list): List of object pointers.
    """

    def __init__(self, model_cfg, sam2_checkpoint, device, memory_size=7, mask_threshold=0.5, use_mask_threshold=False):
        """
        Initializes the SAM2VideoTrainer class.

        Args:
            model_cfg (dict): Configuration dictionary for the model.
            sam2_checkpoint (str): Path to the SAM2 checkpoint file.
            device (torch.device): The device to run the model on (e.g., 'cpu' or 'cuda').
            memory_size (int, optional): Size of the memory. Defaults to 7.
            mask_threshold (float, optional): Threshold for mask prediction. Defaults to 0.5.
            use_mask_threshold (bool, optional): Flag to use mask thresholding. Defaults to False.

        Attributes:
            device (torch.device): The device to run the model on.
            model (SAM2VideoPredictor): The SAM2 video predictor model.
            num_feature_levels (int): Number of feature levels in the model.
            memory_size (int): Size of the memory.
            _transforms (SAM2Transforms): Transformations applied to the input data.
            _bb_feat_sizes (list): Spatial dimensions for backbone feature maps.
            num_maskmem (int): Number of mask memories.
            sam_point_coords (torch.Tensor): Tensor for SAM point coordinates.
            sam_point_labels (torch.Tensor): Tensor for SAM point labels.
            mask_threshold (float): Threshold for mask prediction.
            use_mask_threshold (bool): Flag to use mask thresholding.
        """
        super().__init__()
        self.device = device
        self.model = build_sam2_video_predictor(
            model_cfg, sam2_checkpoint, device=self.device, mode="train"
        )
        self.model.train()
        self.num_feature_levels = self.model.num_feature_levels

        self.num_maskmem = 7
        self.memory_size = (
            memory_size if memory_size <= self.num_maskmem else self.num_maskmem
        )

        self._transforms = SAM2Transforms(
            resolution=self.model.image_size,
            mask_threshold=0.5,
            max_hole_area=0,
            max_sprinkle_area=0,
        )

        # Spatial dim for backbone feature maps
        self._bb_feat_sizes = [
            (256, 256),
            (128, 128),
            (64, 64),
        ]

        self.sam_point_coords = torch.zeros(1, 1, 2, device=device)
        self.sam_point_labels = -torch.ones(1, 1, dtype=torch.int32, device=device)
        self.mask_threshold = mask_threshold
        self.use_mask_threshold = use_mask_threshold

        self.init_state()

    def init_state(self):
        """
        Initializes the state variables for the video trainer.

        This method sets the initial state of various attributes used in the video
        training process. It resets the original height and width, mask memory
        features, mask memory positional encoding, batch size, and object pointers
        to their default values.

        Attributes:
            _orig_hw (tuple or None): Original height and width of the video frames.
            maskmem_features (Any or None): Features related to mask memory.
            maskmem_pos_enc (Any or None): Positional encoding for mask memory.
            batch_size (int or None): Size of the batch for training.
            obj_ptrs (list): List of object pointers used in the training process.
        """
        self._orig_hw = None
        self.maskmem_features = None
        self.maskmem_pos_enc = None
        self.batch_size = None
        self.current_frame_idx = 0
        self.obj_ptrs = []
        self.num_frames = 0

    def reset_state(self):
        """
        Resets the state of the video trainer.

        This method clears the internal state variables, setting them to their initial values:
        - `_orig_hw`: Set to None. Represents the original height and width.
        - `maskmem_features`: Set to None. Represents the mask memory features.
        - `maskmem_pos_enc`: Set to None. Represents the mask memory positional encoding.
        - `batch_size`: Set to None. Represents the batch size.
        - `obj_ptrs`: Set to an empty list. Represents the object pointers.
        """
        self._orig_hw = None
        self.maskmem_features = None
        self.maskmem_pos_enc = None
        self.batch_size = None
        self.current_frame_idx = 0
        self.obj_ptrs = []
        self.num_frames = 0

    def forward(self, videos, bboxes, labels=None):
        """
        Forward pass for processing video frames and predicting masks, logits, and IoUs.

        Args:
            videos (torch.Tensor): A tensor of shape (batch_size, num_frames, C, H, W) representing the input video frames.
            bboxes (torch.Tensor): A tensor of shape (batch_size, 4) representing the bounding boxes for the first frame.
            labels (torch.Tensor, optional): A tensor of shape (batch_size, num_frames, H, W) representing the ground truth masks for each frame. Defaults to None.

        Returns:
            tuple: A tuple containing:
            - all_masks (list of torch.Tensor): A list of tensors representing the predicted masks for each frame.
            - all_logits (list of torch.Tensor): A list of tensors representing the predicted logits for each frame.
            - all_ious (list of torch.Tensor): A list of tensors representing the predicted IoUs for each frame.
        """
        self.init_state()
        batch_size, num_frames, C, H, W = videos.shape
        self.num_frames = num_frames
        self._orig_hw = [H, W]
        self.batch_size = batch_size

        # Extract features for all frames in the video
        videos = videos.view(batch_size * num_frames, C, H, W)
        features = self.model.forward_image(videos)  # Extract features for all frames
        features = {
            k: (
                v.view(batch_size, num_frames, *v.shape[1:])
                if not isinstance(v, list)
                else ([_v.view(batch_size, num_frames, *_v.shape[1:]) for _v in v])
            )
            for k, v in features.items()
        }
        frame_features = self.preprocess_frame_features(
            features, batch_size, num_frames
        )

        # Process the first frame with bounding boxes as prompts
        first_frame_features = frame_features[0]
        first_frame_bbox = bboxes.view(batch_size, 4)

        # Predict the first frame masks and IoUs
        first_frame_masks, first_frame_logits, first_frame_ious, object_score_logits = (
            self._predict_first_frame(first_frame_features, first_frame_bbox)
        )

        # Initialize memory with first frame predictions
        prev_pred_mask = first_frame_masks if labels is None else labels[:, 0]
        memory = self._initialize_memory(first_frame_features, prev_pred_mask, object_score_logits)

        # Process remaining frames
        all_masks, all_logits, all_ious = (
            [first_frame_masks],
            [first_frame_logits],
            [first_frame_ious],
        )
        for t in range(1, num_frames):
            self.current_frame_idx = t
            frame_feature = frame_features[t]
            masks, logits, ious, object_score_logits = self._predict_frame(
                frame_feature, memory, prev_pred_mask
            )
            all_masks.append(masks)
            all_logits.append(logits)
            all_ious.append(ious)
            if t < num_frames - 1:
                prev_pred_mask = masks if labels is None else labels[:, t]
                memory = self._update_memory(frame_feature, prev_pred_mask, memory, object_score_logits)

        self.reset_state()
        return all_masks, all_logits, all_ious

    def normalize_bbox(self, bbox):
        """
        Normalize the given bounding box coordinates.

        This method transforms the bounding box coordinates to a normalized form
        based on the original height and width of the image.

        Args:
            bbox (list or ndarray): The bounding box coordinates to be normalized.

        Returns:
            list or ndarray: The normalized bounding box coordinates.
        """
        unnorm_bbox = self._transforms.transform_boxes(
            bbox, normalize=True, orig_hw=self._orig_hw
        )
        return unnorm_bbox

    def _get_points_placeholder(self, batch_size=None):
        """
        Generates a placeholder for point coordinates and labels.

        Args:
            batch_size (int, optional): The size of the batch. If not provided,
                        defaults to the instance's batch_size attribute.

        Returns:
            tuple: A tuple containing:
            - torch.Tensor: Expanded point coordinates tensor of shape (batch_size, -1, -1).
            - torch.Tensor: Expanded point labels tensor of shape (batch_size, -1).
        """
        batch_size = self.batch_size if batch_size is None else batch_size
        points_placeholder = (
            self.sam_point_coords.expand(batch_size, -1, -1),
            self.sam_point_labels.expand(batch_size, -1),
        )
        return points_placeholder

    def unbind_frame_features(self, frame_features, num_frames):
        """
        Unbind image features from the model.
        """
        keys = frame_features.keys()
        unbinded_frame_features = []
        for frame_idx in range(num_frames):
            frame_feature = {}
            for k in keys:
                frame_feature[k] = (
                    frame_features[k][:, frame_idx]
                    if not isinstance(frame_features[k], list)
                    else [v[:, frame_idx] for v in frame_features[k]]
                )
            unbinded_frame_features.append(frame_feature)
        return unbinded_frame_features

    def preprocess_frame_features(self, frame_features, batch_size, num_frames):
        """
        Preprocess frame features.
        """
        frame_features = self.unbind_frame_features(frame_features, num_frames)
        preprocessed_frame_features = []
        for frame_idx, frame_feature in enumerate(frame_features):
            feature_maps = frame_feature["backbone_fpn"][-self.num_feature_levels :]
            # flatten NxCxHxW to HWxNxC
            vision_feats = [x.flatten(2).permute(2, 0, 1) for x in feature_maps]
            if (
                frame_idx == 0 and self.model.directly_add_no_mem_embed
            ):  # Add no memory embedding
                vision_feats[-1] = vision_feats[-1] + self.model.no_mem_embed
            # HWxNxC to NxCxHxW
            feats = [
                feat.permute(1, 2, 0).view(batch_size, -1, *feat_size)
                for feat, feat_size in zip(
                    vision_feats[::-1], self._bb_feat_sizes[::-1]
                )
            ][::-1]
            _features = {
                "image_embed": feats[-1],
                "high_res_feats": feats[:-1],
                "backbone_fpn": frame_feature["backbone_fpn"][
                    -self.num_feature_levels :
                ],
                "vision_pos_enc": frame_feature["vision_pos_enc"][
                    -self.num_feature_levels :
                ],
            }
            preprocessed_frame_features.append(_features)
        return preprocessed_frame_features

    def _embed_bbox(self, bbox):
        """
        Embed bounding boxes.
        """
        bbox = self.normalize_bbox(bbox)
        box_coords = bbox.reshape(-1, 2, 2)
        box_labels = torch.tensor([[2, 3]], dtype=torch.int, device=bbox.device)
        box_labels = box_labels.repeat(bbox.size(0), 1)
        concat_points = (box_coords, box_labels)
        sparse_embeddings, dense_embeddings = self.model.sam_prompt_encoder(
            points=concat_points, boxes=None, masks=None
        )
        return sparse_embeddings, dense_embeddings

    def _predict_first_frame(self, features, bbox):
        """
        Predict masks and IoUs for the first frame.
        """
        sparse_embeddings, dense_embeddings = self._embed_bbox(bbox)

        low_res_masks, ious, sam_output_tokens, object_score_logits = (
            self.model.sam_mask_decoder(
                image_embeddings=features["image_embed"],
                image_pe=self.model.sam_prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=False,
                repeat_image=False,
                high_res_features=features["high_res_feats"],
            )
        )

        sam_output_token = sam_output_tokens[:, -1]
        obj_ptr = self.model.obj_ptr_proj(sam_output_token)
        self.obj_ptrs.append(obj_ptr)
        pred_mask, pred_logit = self._postprocess_masks(low_res_masks)
        return pred_mask, pred_logit, ious[:, -1], object_score_logits

    def _postprocess_masks(self, logits, size=None):
        """
        Perform post-processing on output masks.
        """
        size = self._orig_hw if size is None else size
        logits = F.interpolate(logits, size, mode="bilinear", align_corners=False)
        logits = logits[:, -1].unsqueeze(1)
        masks = torch.sigmoid(logits)
        if self.use_mask_threshold:
            masks = (masks > self.mask_threshold).float()
        return masks, logits

    def _extract_memory_features(self, features, masks, object_score_logits):
        """
        Extracts memory features from the given features and masks.

        Args:
            features (dict): A dictionary containing feature maps from the backbone FPN.
            masks (Tensor): A tensor representing the masks to be used by the memory encoder.

        Returns:
            dict: A dictionary containing:
            - "vision_features" (Tensor): The vision features extracted and processed by the memory encoder.
            - "vision_pos_enc" (Tensor): The positional encoding of the vision features.
        """
        pix_feat = features["backbone_fpn"][-1]
        maskmem_out = self.model.memory_encoder(
            pix_feat, masks, skip_mask_sigmoid=True  # sigmoid already applied
        )
        maskmem_features = maskmem_out["vision_features"]
        
        if self.model.no_obj_embed_spatial is not None:
            is_obj_appearing = (object_score_logits > 0).float()
            maskmem_features += (
                1 - is_obj_appearing[..., None, None]
            ) * self.model.no_obj_embed_spatial[..., None, None].expand(
                *maskmem_features.shape
            )
        maskmem_features = maskmem_features.flatten(2).permute(2, 0, 1)
        maskmem_pos_enc = maskmem_out["vision_pos_enc"][-1].flatten(2).permute(2, 0, 1)
        return {"vision_features": maskmem_features, "vision_pos_enc": maskmem_pos_enc}

    def _initialize_memory(self, features, masks, object_score_logits):
        """
        Initialize memory for the first frame.
        """
        maskmem_out = self._extract_memory_features(features, masks, object_score_logits)
        self.maskmem_features = [maskmem_out["vision_features"]]
        self.maskmem_pos_enc = [maskmem_out["vision_pos_enc"]]
        return self.maskmem_features, self.maskmem_pos_enc

    def _update_memory(self, features, masks, memory=None, object_score_logits=None):
        """
        Update memory with new frame data.
        """
        if memory is None:
            maskmem_features, maskmem_pos_enc = (
                self.maskmem_features,
                self.maskmem_pos_enc,
            )
        else:
            maskmem_features, maskmem_pos_enc = memory

        maskmem_out = self._extract_memory_features(features, masks, object_score_logits)
        maskmem_features.append(maskmem_out["vision_features"])
        maskmem_pos_enc.append(maskmem_out["vision_pos_enc"])
        if len(maskmem_features) > self.memory_size:
            self.maskmem_features = maskmem_features[-self.memory_size :]
            self.maskmem_pos_enc = maskmem_pos_enc[-self.memory_size :]
        return maskmem_features, maskmem_pos_enc

    def _prepare_memory(self, memory):
        """
        Prepare memory for the current frame.
        """
        if memory is None:
            maskmem_features, maskmem_pos_enc = (
                self.maskmem_features,
                self.maskmem_pos_enc,
            )
        else:
            maskmem_features, maskmem_pos_enc = memory
        for idx in range(len(maskmem_pos_enc)):
            rel_pos = len(maskmem_pos_enc) - idx
            maskmem_pos_enc[idx] = (
                maskmem_pos_enc[idx] + self.model.maskmem_tpos_enc[rel_pos - 1]
            )
        obj_ptrs = torch.stack(self.obj_ptrs, dim=0)
        
        if self.model.add_tpos_enc_to_obj_ptrs:
            max_obj_ptrs_in_encoder = self.num_frames
            pos_list = [self.current_frame_idx]
            t_diff_max = max_obj_ptrs_in_encoder - 1
            tpos_dim = self.model.hidden_dim if self.model.proj_tpos_enc_in_obj_ptrs else self.model.mem_dim
            obj_pos = torch.tensor(pos_list, device=obj_ptrs.device)
            obj_pos = get_1d_sine_pe(obj_pos / t_diff_max, dim=tpos_dim)
            obj_pos = self.model.obj_ptr_tpos_proj(obj_pos)
            obj_pos = obj_pos.unsqueeze(1).expand(-1, self.batch_size, self.model.mem_dim)
        else:
            obj_pos = obj_ptrs.new_zeros(
                len(self.obj_ptrs), self.batch_size, self.model.mem_dim
            )
        C = self.model.hidden_dim
        if self.model.mem_dim < C:
            # split a pointer into (C // self.mem_dim) tokens for self.mem_dim < C
            obj_ptrs = obj_ptrs.reshape(
                -1, self.batch_size, C // self.model.mem_dim, self.model.mem_dim
            )
            obj_ptrs = obj_ptrs.permute(0, 2, 1, 3).flatten(0, 1)
            obj_pos = obj_pos.repeat_interleave(C // self.model.mem_dim, dim=0)
        num_obj_ptr_tokens = obj_ptrs.shape[0]
        memory = torch.cat(maskmem_features + [obj_ptrs], dim=0)
        memory_pos_embed = torch.cat(maskmem_pos_enc + [obj_pos], dim=0)
        return memory, memory_pos_embed, num_obj_ptr_tokens

    def _predict_frame(self, features, memory, prev_mask=None):
        """
        Predict masks and IoUs for subsequent frames using memory.
        """
        memory, memory_pos_embed, num_obj_ptr_tokens = self._prepare_memory(memory)

        current_vision_feats = [
            x.flatten(2).permute(2, 0, 1) for x in features["backbone_fpn"]
        ]
        current_vision_pos_embeds = [
            x.flatten(2).permute(2, 0, 1) for x in features["vision_pos_enc"]
        ]
        pix_feat_with_mem = self.model.memory_attention(
            curr=current_vision_feats[-1:],
            curr_pos=current_vision_pos_embeds[-1:],
            memory=memory,
            memory_pos=memory_pos_embed,
            num_obj_ptr_tokens=num_obj_ptr_tokens,
        )
        pix_feat_with_mem = pix_feat_with_mem.permute(1, 2, 0).view(
            *features["backbone_fpn"][-1].shape
        )
        sparse_embeddings, dense_embeddings = self.model.sam_prompt_encoder(
            points=self._get_points_placeholder(),
            boxes=None,
            masks=None,
        )
        low_res_masks, ious, _, object_score_logits = self.model.sam_mask_decoder(
            image_embeddings=pix_feat_with_mem,
            image_pe=self.model.sam_prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=False,
            repeat_image=False,
            high_res_features=features["high_res_feats"],
        )

        pred_mask, pred_logit = self._postprocess_masks(low_res_masks)
        return pred_mask, pred_logit, ious[:, -1], object_score_logits


class TestSAM2VideoTrainer(unittest.TestCase):
    def setUp(self):
        # Initialize parameters
        sam2_checkpoint = "./checkpoints/sam2_hiera_tiny.pt"
        model_cfg = "sam2_hiera_t.yaml"
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Create an instance of SAM2VideoTrainer
        self.trainer = SAM2VideoTrainer(model_cfg, sam2_checkpoint, device=self.device)

        # Define input video and bounding boxes
        self.batch_size = 2
        self.num_frames = 2
        self.C = 3
        self.H = 1024
        self.W = 1024

        # Create random video data and bounding boxes
        self.videos = torch.randn(
            self.batch_size, self.num_frames, self.C, self.H, self.W
        ).to(self.device)
        self.masks = torch.zeros(
            self.batch_size, self.num_frames, 1, self.H, self.W
        ).to(self.device)
        self.bboxes = torch.tensor(
            [[100, 100, 200, 200], [150, 150, 250, 250]], dtype=torch.float32
        ).to(self.device)

    def test_forward(self):
        # Execute the forward method
        # masks, ious = self.trainer(self.videos, self.bboxes, self.masks)
        masks, ious = self.trainer(self.videos, self.bboxes, None)

        print("Masks shape:", masks[0].shape)
        print("IoUs shape:", ious[0].shape)

        print("Masks:", masks)
        print("IoUs:", ious)


if __name__ == "__main__":
    unittest.main()
'''

# Native module: test-code/source/MedSAM2-main/sam2/utils/__init__.py
_EMBEDDED_SOURCES['test-code/source/MedSAM2-main/sam2/utils/__init__.py'] = '''# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
'''

# Native module: test-code/source/MedSAM2-main/sam2/utils/amg.py
_EMBEDDED_SOURCES['test-code/source/MedSAM2-main/sam2/utils/amg.py'] = '''# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import math
from copy import deepcopy
from itertools import product
from typing import Any, Dict, Generator, ItemsView, List, Tuple

import numpy as np
import torch

# Very lightly adapted from https://github.com/facebookresearch/segment-anything/blob/main/segment_anything/utils/amg.py


class MaskData:
    """
    A structure for storing masks and their related data in batched format.
    Implements basic filtering and concatenation.
    """

    def __init__(self, **kwargs) -> None:
        for v in kwargs.values():
            assert isinstance(
                v, (list, np.ndarray, torch.Tensor)
            ), "MaskData only supports list, numpy arrays, and torch tensors."
        self._stats = dict(**kwargs)

    def __setitem__(self, key: str, item: Any) -> None:
        assert isinstance(
            item, (list, np.ndarray, torch.Tensor)
        ), "MaskData only supports list, numpy arrays, and torch tensors."
        self._stats[key] = item

    def __delitem__(self, key: str) -> None:
        del self._stats[key]

    def __getitem__(self, key: str) -> Any:
        return self._stats[key]

    def items(self) -> ItemsView[str, Any]:
        return self._stats.items()

    def filter(self, keep: torch.Tensor) -> None:
        for k, v in self._stats.items():
            if v is None:
                self._stats[k] = None
            elif isinstance(v, torch.Tensor):
                self._stats[k] = v[torch.as_tensor(keep, device=v.device)]
            elif isinstance(v, np.ndarray):
                self._stats[k] = v[keep.detach().cpu().numpy()]
            elif isinstance(v, list) and keep.dtype == torch.bool:
                self._stats[k] = [a for i, a in enumerate(v) if keep[i]]
            elif isinstance(v, list):
                self._stats[k] = [v[i] for i in keep]
            else:
                raise TypeError(f"MaskData key {k} has an unsupported type {type(v)}.")

    def cat(self, new_stats: "MaskData") -> None:
        for k, v in new_stats.items():
            if k not in self._stats or self._stats[k] is None:
                self._stats[k] = deepcopy(v)
            elif isinstance(v, torch.Tensor):
                self._stats[k] = torch.cat([self._stats[k], v], dim=0)
            elif isinstance(v, np.ndarray):
                self._stats[k] = np.concatenate([self._stats[k], v], axis=0)
            elif isinstance(v, list):
                self._stats[k] = self._stats[k] + deepcopy(v)
            else:
                raise TypeError(f"MaskData key {k} has an unsupported type {type(v)}.")

    def to_numpy(self) -> None:
        for k, v in self._stats.items():
            if isinstance(v, torch.Tensor):
                self._stats[k] = v.float().detach().cpu().numpy()


def is_box_near_crop_edge(
    boxes: torch.Tensor, crop_box: List[int], orig_box: List[int], atol: float = 20.0
) -> torch.Tensor:
    """Filter masks at the edge of a crop, but not at the edge of the original image."""
    crop_box_torch = torch.as_tensor(crop_box, dtype=torch.float, device=boxes.device)
    orig_box_torch = torch.as_tensor(orig_box, dtype=torch.float, device=boxes.device)
    boxes = uncrop_boxes_xyxy(boxes, crop_box).float()
    near_crop_edge = torch.isclose(boxes, crop_box_torch[None, :], atol=atol, rtol=0)
    near_image_edge = torch.isclose(boxes, orig_box_torch[None, :], atol=atol, rtol=0)
    near_crop_edge = torch.logical_and(near_crop_edge, ~near_image_edge)
    return torch.any(near_crop_edge, dim=1)


def box_xyxy_to_xywh(box_xyxy: torch.Tensor) -> torch.Tensor:
    box_xywh = deepcopy(box_xyxy)
    box_xywh[2] = box_xywh[2] - box_xywh[0]
    box_xywh[3] = box_xywh[3] - box_xywh[1]
    return box_xywh


def batch_iterator(batch_size: int, *args) -> Generator[List[Any], None, None]:
    assert len(args) > 0 and all(
        len(a) == len(args[0]) for a in args
    ), "Batched iteration must have inputs of all the same size."
    n_batches = len(args[0]) // batch_size + int(len(args[0]) % batch_size != 0)
    for b in range(n_batches):
        yield [arg[b * batch_size : (b + 1) * batch_size] for arg in args]


def mask_to_rle_pytorch(tensor: torch.Tensor) -> List[Dict[str, Any]]:
    """
    Encodes masks to an uncompressed RLE, in the format expected by
    pycoco tools.
    """
    # Put in fortran order and flatten h,w
    b, h, w = tensor.shape
    tensor = tensor.permute(0, 2, 1).flatten(1)

    # Compute change indices
    diff = tensor[:, 1:] ^ tensor[:, :-1]
    change_indices = diff.nonzero()

    # Encode run length
    out = []
    for i in range(b):
        cur_idxs = change_indices[change_indices[:, 0] == i, 1]
        cur_idxs = torch.cat(
            [
                torch.tensor([0], dtype=cur_idxs.dtype, device=cur_idxs.device),
                cur_idxs + 1,
                torch.tensor([h * w], dtype=cur_idxs.dtype, device=cur_idxs.device),
            ]
        )
        btw_idxs = cur_idxs[1:] - cur_idxs[:-1]
        counts = [] if tensor[i, 0] == 0 else [0]
        counts.extend(btw_idxs.detach().cpu().tolist())
        out.append({"size": [h, w], "counts": counts})
    return out


def rle_to_mask(rle: Dict[str, Any]) -> np.ndarray:
    """Compute a binary mask from an uncompressed RLE."""
    h, w = rle["size"]
    mask = np.empty(h * w, dtype=bool)
    idx = 0
    parity = False
    for count in rle["counts"]:
        mask[idx : idx + count] = parity
        idx += count
        parity ^= True
    mask = mask.reshape(w, h)
    return mask.transpose()  # Put in C order


def area_from_rle(rle: Dict[str, Any]) -> int:
    return sum(rle["counts"][1::2])


def calculate_stability_score(
    masks: torch.Tensor, mask_threshold: float, threshold_offset: float
) -> torch.Tensor:
    """
    Computes the stability score for a batch of masks. The stability
    score is the IoU between the binary masks obtained by thresholding
    the predicted mask logits at high and low values.
    """
    # One mask is always contained inside the other.
    # Save memory by preventing unnecessary cast to torch.int64
    intersections = (
        (masks > (mask_threshold + threshold_offset))
        .sum(-1, dtype=torch.int16)
        .sum(-1, dtype=torch.int32)
    )
    unions = (
        (masks > (mask_threshold - threshold_offset))
        .sum(-1, dtype=torch.int16)
        .sum(-1, dtype=torch.int32)
    )
    return intersections / unions


def build_point_grid(n_per_side: int) -> np.ndarray:
    """Generates a 2D grid of points evenly spaced in [0,1]x[0,1]."""
    offset = 1 / (2 * n_per_side)
    points_one_side = np.linspace(offset, 1 - offset, n_per_side)
    points_x = np.tile(points_one_side[None, :], (n_per_side, 1))
    points_y = np.tile(points_one_side[:, None], (1, n_per_side))
    points = np.stack([points_x, points_y], axis=-1).reshape(-1, 2)
    return points


def build_all_layer_point_grids(
    n_per_side: int, n_layers: int, scale_per_layer: int
) -> List[np.ndarray]:
    """Generates point grids for all crop layers."""
    points_by_layer = []
    for i in range(n_layers + 1):
        n_points = int(n_per_side / (scale_per_layer**i))
        points_by_layer.append(build_point_grid(n_points))
    return points_by_layer


def generate_crop_boxes(
    im_size: Tuple[int, ...], n_layers: int, overlap_ratio: float
) -> Tuple[List[List[int]], List[int]]:
    """
    Generates a list of crop boxes of different sizes. Each layer
    has (2**i)**2 boxes for the ith layer.
    """
    crop_boxes, layer_idxs = [], []
    im_h, im_w = im_size
    short_side = min(im_h, im_w)

    # Original image
    crop_boxes.append([0, 0, im_w, im_h])
    layer_idxs.append(0)

    def crop_len(orig_len, n_crops, overlap):
        return int(math.ceil((overlap * (n_crops - 1) + orig_len) / n_crops))

    for i_layer in range(n_layers):
        n_crops_per_side = 2 ** (i_layer + 1)
        overlap = int(overlap_ratio * short_side * (2 / n_crops_per_side))

        crop_w = crop_len(im_w, n_crops_per_side, overlap)
        crop_h = crop_len(im_h, n_crops_per_side, overlap)

        crop_box_x0 = [int((crop_w - overlap) * i) for i in range(n_crops_per_side)]
        crop_box_y0 = [int((crop_h - overlap) * i) for i in range(n_crops_per_side)]

        # Crops in XYWH format
        for x0, y0 in product(crop_box_x0, crop_box_y0):
            box = [x0, y0, min(x0 + crop_w, im_w), min(y0 + crop_h, im_h)]
            crop_boxes.append(box)
            layer_idxs.append(i_layer + 1)

    return crop_boxes, layer_idxs


def uncrop_boxes_xyxy(boxes: torch.Tensor, crop_box: List[int]) -> torch.Tensor:
    x0, y0, _, _ = crop_box
    offset = torch.tensor([[x0, y0, x0, y0]], device=boxes.device)
    # Check if boxes has a channel dimension
    if len(boxes.shape) == 3:
        offset = offset.unsqueeze(1)
    return boxes + offset


def uncrop_points(points: torch.Tensor, crop_box: List[int]) -> torch.Tensor:
    x0, y0, _, _ = crop_box
    offset = torch.tensor([[x0, y0]], device=points.device)
    # Check if points has a channel dimension
    if len(points.shape) == 3:
        offset = offset.unsqueeze(1)
    return points + offset


def uncrop_masks(
    masks: torch.Tensor, crop_box: List[int], orig_h: int, orig_w: int
) -> torch.Tensor:
    x0, y0, x1, y1 = crop_box
    if x0 == 0 and y0 == 0 and x1 == orig_w and y1 == orig_h:
        return masks
    # Coordinate transform masks
    pad_x, pad_y = orig_w - (x1 - x0), orig_h - (y1 - y0)
    pad = (x0, pad_x - x0, y0, pad_y - y0)
    return torch.nn.functional.pad(masks, pad, value=0)


def remove_small_regions(
    mask: np.ndarray, area_thresh: float, mode: str
) -> Tuple[np.ndarray, bool]:
    """
    Removes small disconnected regions and holes in a mask. Returns the
    mask and an indicator of if the mask has been modified.
    """
    import cv2  # type: ignore

    assert mode in ["holes", "islands"]
    correct_holes = mode == "holes"
    working_mask = (correct_holes ^ mask).astype(np.uint8)
    n_labels, regions, stats, _ = cv2.connectedComponentsWithStats(working_mask, 8)
    sizes = stats[:, -1][1:]  # Row 0 is background label
    small_regions = [i + 1 for i, s in enumerate(sizes) if s < area_thresh]
    if len(small_regions) == 0:
        return mask, False
    fill_labels = [0] + small_regions
    if not correct_holes:
        fill_labels = [i for i in range(n_labels) if i not in fill_labels]
        # If every region is below threshold, keep largest
        if len(fill_labels) == 0:
            fill_labels = [int(np.argmax(sizes)) + 1]
    mask = np.isin(regions, fill_labels)
    return mask, True


def coco_encode_rle(uncompressed_rle: Dict[str, Any]) -> Dict[str, Any]:
    from pycocotools import mask as mask_utils  # type: ignore

    h, w = uncompressed_rle["size"]
    rle = mask_utils.frPyObjects(uncompressed_rle, h, w)
    rle["counts"] = rle["counts"].decode("utf-8")  # Necessary to serialize with json
    return rle


def batched_mask_to_box(masks: torch.Tensor) -> torch.Tensor:
    """
    Calculates boxes in XYXY format around masks. Return [0,0,0,0] for
    an empty mask. For input shape C1xC2x...xHxW, the output shape is C1xC2x...x4.
    """
    # torch.max below raises an error on empty inputs, just skip in this case
    if torch.numel(masks) == 0:
        return torch.zeros(*masks.shape[:-2], 4, device=masks.device)

    # Normalize shape to CxHxW
    shape = masks.shape
    h, w = shape[-2:]
    if len(shape) > 2:
        masks = masks.flatten(0, -3)
    else:
        masks = masks.unsqueeze(0)

    # Get top and bottom edges
    in_height, _ = torch.max(masks, dim=-1)
    in_height_coords = in_height * torch.arange(h, device=in_height.device)[None, :]
    bottom_edges, _ = torch.max(in_height_coords, dim=-1)
    in_height_coords = in_height_coords + h * (~in_height)
    top_edges, _ = torch.min(in_height_coords, dim=-1)

    # Get left and right edges
    in_width, _ = torch.max(masks, dim=-2)
    in_width_coords = in_width * torch.arange(w, device=in_width.device)[None, :]
    right_edges, _ = torch.max(in_width_coords, dim=-1)
    in_width_coords = in_width_coords + w * (~in_width)
    left_edges, _ = torch.min(in_width_coords, dim=-1)

    # If the mask is empty the right edge will be to the left of the left edge.
    # Replace these boxes with [0, 0, 0, 0]
    empty_filter = (right_edges < left_edges) | (bottom_edges < top_edges)
    out = torch.stack([left_edges, top_edges, right_edges, bottom_edges], dim=-1)
    out = out * (~empty_filter).unsqueeze(-1)

    # Return to original shape
    if len(shape) > 2:
        out = out.reshape(*shape[:-2], 4)
    else:
        out = out[0]

    return out
'''

# Native module: test-code/source/MedSAM2-main/sam2/utils/misc.py
_EMBEDDED_SOURCES['test-code/source/MedSAM2-main/sam2/utils/misc.py'] = '''# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import warnings
from threading import Thread

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm


def get_sdpa_settings():
    if torch.cuda.is_available():
        old_gpu = torch.cuda.get_device_properties(0).major < 7
        # only use Flash Attention on Ampere (8.0) or newer GPUs
        use_flash_attn = torch.cuda.get_device_properties(0).major >= 8
        if not use_flash_attn:
            warnings.warn(
                "Flash Attention is disabled as it requires a GPU with Ampere (8.0) CUDA capability.",
                category=UserWarning,
                stacklevel=2,
            )
        # keep math kernel for PyTorch versions before 2.2 (Flash Attention v2 is only
        # available on PyTorch 2.2+, while Flash Attention v1 cannot handle all cases)
        pytorch_version = tuple(int(v) for v in torch.__version__.split(".")[:2])
        if pytorch_version < (2, 2):
            warnings.warn(
                f"You are using PyTorch {torch.__version__} without Flash Attention v2 support. "
                "Consider upgrading to PyTorch 2.2+ for Flash Attention v2 (which could be faster).",
                category=UserWarning,
                stacklevel=2,
            )
        math_kernel_on = pytorch_version < (2, 2) or not use_flash_attn
    else:
        old_gpu = True
        use_flash_attn = False
        math_kernel_on = True

    return old_gpu, use_flash_attn, math_kernel_on


def get_connected_components(mask):
    """
    Get the connected components (8-connectivity) of binary masks of shape (N, 1, H, W).

    Inputs:
    - mask: A binary mask tensor of shape (N, 1, H, W), where 1 is foreground and 0 is
            background.

    Outputs:
    - labels: A tensor of shape (N, 1, H, W) containing the connected component labels
              for foreground pixels and 0 for background pixels.
    - counts: A tensor of shape (N, 1, H, W) containing the area of the connected
              components for foreground pixels and 0 for background pixels.
    """
    from sam2 import _C

    return _C.get_connected_componnets(mask.to(torch.uint8).contiguous())


def mask_to_box(masks: torch.Tensor):
    """
    compute bounding box given an input mask

    Inputs:
    - masks: [B, 1, H, W] masks, dtype=torch.Tensor

    Returns:
    - box_coords: [B, 1, 4], contains (x, y) coordinates of top left and bottom right box corners, dtype=torch.Tensor
    """
    B, _, h, w = masks.shape
    device = masks.device
    xs = torch.arange(w, device=device, dtype=torch.int32)
    ys = torch.arange(h, device=device, dtype=torch.int32)
    grid_xs, grid_ys = torch.meshgrid(xs, ys, indexing="xy")
    grid_xs = grid_xs[None, None, ...].expand(B, 1, h, w)
    grid_ys = grid_ys[None, None, ...].expand(B, 1, h, w)
    min_xs, _ = torch.min(torch.where(masks, grid_xs, w).flatten(-2), dim=-1)
    max_xs, _ = torch.max(torch.where(masks, grid_xs, -1).flatten(-2), dim=-1)
    min_ys, _ = torch.min(torch.where(masks, grid_ys, h).flatten(-2), dim=-1)
    max_ys, _ = torch.max(torch.where(masks, grid_ys, -1).flatten(-2), dim=-1)
    bbox_coords = torch.stack((min_xs, min_ys, max_xs, max_ys), dim=-1)

    return bbox_coords


def _load_img_as_tensor(img_path, image_size):
    img_pil = Image.open(img_path)
    img_np = np.array(img_pil.convert("RGB").resize((image_size, image_size)))
    if img_np.dtype == np.uint8:  # np.uint8 is expected for JPEG images
        img_np = img_np / 255.0
    else:
        raise RuntimeError(f"Unknown image dtype: {img_np.dtype} on {img_path}")
    img = torch.from_numpy(img_np).permute(2, 0, 1)
    video_width, video_height = img_pil.size  # the original video size
    return img, video_height, video_width


class AsyncVideoFrameLoader:
    """
    A list of video frames to be load asynchronously without blocking session start.
    """

    def __init__(
        self,
        img_paths,
        image_size,
        offload_video_to_cpu,
        img_mean,
        img_std,
        compute_device,
    ):
        self.img_paths = img_paths
        self.image_size = image_size
        self.offload_video_to_cpu = offload_video_to_cpu
        self.img_mean = img_mean
        self.img_std = img_std
        # items in `self.images` will be loaded asynchronously
        self.images = [None] * len(img_paths)
        # catch and raise any exceptions in the async loading thread
        self.exception = None
        # video_height and video_width be filled when loading the first image
        self.video_height = None
        self.video_width = None
        self.compute_device = compute_device

        # load the first frame to fill video_height and video_width and also
        # to cache it (since it's most likely where the user will click)
        self.__getitem__(0)

        # load the rest of frames asynchronously without blocking the session start
        def _load_frames():
            try:
                for n in tqdm(range(len(self.images)), desc="frame loading (JPEG)"):
                    self.__getitem__(n)
            except Exception as e:
                self.exception = e

        self.thread = Thread(target=_load_frames, daemon=True)
        self.thread.start()

    def __getitem__(self, index):
        if self.exception is not None:
            raise RuntimeError("Failure in frame loading thread") from self.exception

        img = self.images[index]
        if img is not None:
            return img

        img, video_height, video_width = _load_img_as_tensor(
            self.img_paths[index], self.image_size
        )
        self.video_height = video_height
        self.video_width = video_width
        # normalize by mean and std
        img -= self.img_mean
        img /= self.img_std
        if not self.offload_video_to_cpu:
            img = img.to(self.compute_device, non_blocking=True)
        self.images[index] = img
        return img

    def __len__(self):
        return len(self.images)


def load_video_frames(
    video_path,
    image_size,
    offload_video_to_cpu,
    img_mean=(0.485, 0.456, 0.406),
    img_std=(0.229, 0.224, 0.225),
    async_loading_frames=False,
    compute_device=torch.device("cuda"),
):
    """
    Load the video frames from video_path. The frames are resized to image_size as in
    the model and are loaded to GPU if offload_video_to_cpu=False. This is used by the demo.
    """
    is_bytes = isinstance(video_path, bytes)
    is_str = isinstance(video_path, str)
    is_mp4_path = is_str and os.path.splitext(video_path)[-1] in [".mp4", ".MP4"]
    if is_bytes or is_mp4_path:
        return load_video_frames_from_video_file(
            video_path=video_path,
            image_size=image_size,
            offload_video_to_cpu=offload_video_to_cpu,
            img_mean=img_mean,
            img_std=img_std,
            compute_device=compute_device,
        )
    elif is_str and os.path.isdir(video_path):
        return load_video_frames_from_jpg_images(
            video_path=video_path,
            image_size=image_size,
            offload_video_to_cpu=offload_video_to_cpu,
            img_mean=img_mean,
            img_std=img_std,
            async_loading_frames=async_loading_frames,
            compute_device=compute_device,
        )
    else:
        raise NotImplementedError(
            "Only MP4 video and JPEG folder are supported at this moment"
        )


def load_video_frames_from_jpg_images(
    video_path,
    image_size,
    offload_video_to_cpu,
    img_mean=(0.485, 0.456, 0.406),
    img_std=(0.229, 0.224, 0.225),
    async_loading_frames=False,
    compute_device=torch.device("cuda"),
):
    """
    Load the video frames from a directory of JPEG files ("<frame_index>.jpg" format).

    The frames are resized to image_size x image_size and are loaded to GPU if
    `offload_video_to_cpu` is `False` and to CPU if `offload_video_to_cpu` is `True`.

    You can load a frame asynchronously by setting `async_loading_frames` to `True`.
    """
    if isinstance(video_path, str) and os.path.isdir(video_path):
        jpg_folder = video_path
    else:
        raise NotImplementedError(
            "Only JPEG frames are supported at this moment. For video files, you may use "
            "ffmpeg (https://ffmpeg.org/) to extract frames into a folder of JPEG files, such as \\n"
            "```\\n"
            "ffmpeg -i <your_video>.mp4 -q:v 2 -start_number 0 <output_dir>/'%05d.jpg'\\n"
            "```\\n"
            "where `-q:v` generates high-quality JPEG frames and `-start_number 0` asks "
            "ffmpeg to start the JPEG file from 00000.jpg."
        )

    frame_names = [
        p
        for p in os.listdir(jpg_folder)
        if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG"]
    ]
    frame_names.sort()
    num_frames = len(frame_names)
    if num_frames == 0:
        raise RuntimeError(f"no images found in {jpg_folder}")
    img_paths = [os.path.join(jpg_folder, frame_name) for frame_name in frame_names]
    img_mean = torch.tensor(img_mean, dtype=torch.float32)[:, None, None]
    img_std = torch.tensor(img_std, dtype=torch.float32)[:, None, None]

    if async_loading_frames:
        lazy_images = AsyncVideoFrameLoader(
            img_paths,
            image_size,
            offload_video_to_cpu,
            img_mean,
            img_std,
            compute_device,
        )
        return lazy_images, lazy_images.video_height, lazy_images.video_width

    images = torch.zeros(num_frames, 3, image_size, image_size, dtype=torch.float32)
    for n, img_path in enumerate(tqdm(img_paths, desc="frame loading (JPEG)")):
        images[n], video_height, video_width = _load_img_as_tensor(img_path, image_size)
    if not offload_video_to_cpu:
        images = images.to(compute_device)
        img_mean = img_mean.to(compute_device)
        img_std = img_std.to(compute_device)
    # normalize by mean and std
    images -= img_mean
    images /= img_std
    return images, video_height, video_width


def load_video_frames_from_video_file(
    video_path,
    image_size,
    offload_video_to_cpu,
    img_mean=(0.485, 0.456, 0.406),
    img_std=(0.229, 0.224, 0.225),
    compute_device=torch.device("cuda"),
):
    """Load the video frames from a video file."""
    import decord

    img_mean = torch.tensor(img_mean, dtype=torch.float32)[:, None, None]
    img_std = torch.tensor(img_std, dtype=torch.float32)[:, None, None]
    # Get the original video height and width
    decord.bridge.set_bridge("torch")
    video_height, video_width, _ = decord.VideoReader(video_path).next().shape
    # Iterate over all frames in the video
    images = []
    for frame in decord.VideoReader(video_path, width=image_size, height=image_size):
        images.append(frame.permute(2, 0, 1))

    images = torch.stack(images, dim=0).float() / 255.0
    if not offload_video_to_cpu:
        images = images.to(compute_device)
        img_mean = img_mean.to(compute_device)
        img_std = img_std.to(compute_device)
    # normalize by mean and std
    images -= img_mean
    images /= img_std
    return images, video_height, video_width


def fill_holes_in_mask_scores(mask, max_area):
    """
    A post processor to fill small holes in mask scores with area under `max_area`.
    """
    # Holes are those connected components in background with area <= self.max_area
    # (background regions are those with mask scores <= 0)
    assert max_area > 0, "max_area must be positive"

    input_mask = mask
    try:
        labels, areas = get_connected_components(mask <= 0)
        is_hole = (labels > 0) & (areas <= max_area)
        # We fill holes with a small positive mask score (0.1) to change them to foreground.
        mask = torch.where(is_hole, 0.1, mask)
    except Exception as e:
        # Skip the post-processing step on removing small holes if the CUDA kernel fails
        warnings.warn(
            f"{e}\\n\\nSkipping the post-processing step due to the error above. You can "
            "still use SAM 2 and it's OK to ignore the error above, although some post-processing "
            "functionality may be limited (which doesn't affect the results in most cases; see "
            "https://github.com/facebookresearch/sam2/blob/main/INSTALL.md).",
            category=UserWarning,
            stacklevel=2,
        )
        mask = input_mask

    return mask


def concat_points(old_point_inputs, new_points, new_labels):
    """Add new points and labels to previous point inputs (add at the end)."""
    if old_point_inputs is None:
        points, labels = new_points, new_labels
    else:
        points = torch.cat([old_point_inputs["point_coords"], new_points], dim=1)
        labels = torch.cat([old_point_inputs["point_labels"], new_labels], dim=1)

    return {"point_coords": points, "point_labels": labels}
'''

# Native module: test-code/source/MedSAM2-main/sam2/utils/transforms.py
_EMBEDDED_SOURCES['test-code/source/MedSAM2-main/sam2/utils/transforms.py'] = '''# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import Normalize, Resize, ToTensor


class SAM2Transforms(nn.Module):
    def __init__(
        self, resolution, mask_threshold, max_hole_area=0.0, max_sprinkle_area=0.0
    ):
        """
        Transforms for SAM2.
        """
        super().__init__()
        self.resolution = resolution
        self.mask_threshold = mask_threshold
        self.max_hole_area = max_hole_area
        self.max_sprinkle_area = max_sprinkle_area
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]
        self.to_tensor = ToTensor()
        self.transforms = torch.jit.script(
            nn.Sequential(
                Resize((self.resolution, self.resolution)),
                Normalize(self.mean, self.std),
            )
        )

    def __call__(self, x):
        x = self.to_tensor(x)
        return self.transforms(x)

    def forward_batch(self, img_list):
        img_batch = [self.transforms(self.to_tensor(img)) for img in img_list]
        img_batch = torch.stack(img_batch, dim=0)
        return img_batch

    def transform_coords(
        self, coords: torch.Tensor, normalize=False, orig_hw=None
    ) -> torch.Tensor:
        """
        Expects a torch tensor with length 2 in the last dimension. The coordinates can be in absolute image or normalized coordinates,
        If the coords are in absolute image coordinates, normalize should be set to True and original image size is required.

        Returns
            Un-normalized coordinates in the range of [0, 1] which is expected by the SAM2 model.
        """
        if normalize:
            assert orig_hw is not None
            h, w = orig_hw
            coords = coords.clone()
            coords[..., 0] = coords[..., 0] / w
            coords[..., 1] = coords[..., 1] / h

        coords = coords * self.resolution  # unnormalize coords
        return coords

    def transform_boxes(
        self, boxes: torch.Tensor, normalize=False, orig_hw=None
    ) -> torch.Tensor:
        """
        Expects a tensor of shape Bx4. The coordinates can be in absolute image or normalized coordinates,
        if the coords are in absolute image coordinates, normalize should be set to True and original image size is required.
        """
        boxes = self.transform_coords(boxes.reshape(-1, 2, 2), normalize, orig_hw)
        return boxes

    def postprocess_masks(self, masks: torch.Tensor, orig_hw) -> torch.Tensor:
        """
        Perform PostProcessing on output masks.
        """
        from sam2.utils.misc import get_connected_components

        masks = masks.float()
        input_masks = masks
        mask_flat = masks.flatten(0, 1).unsqueeze(1)  # flatten as 1-channel image
        try:
            if self.max_hole_area > 0:
                # Holes are those connected components in background with area <= self.fill_hole_area
                # (background regions are those with mask scores <= self.mask_threshold)
                labels, areas = get_connected_components(
                    mask_flat <= self.mask_threshold
                )
                is_hole = (labels > 0) & (areas <= self.max_hole_area)
                is_hole = is_hole.reshape_as(masks)
                # We fill holes with a small positive mask score (10.0) to change them to foreground.
                masks = torch.where(is_hole, self.mask_threshold + 10.0, masks)

            if self.max_sprinkle_area > 0:
                labels, areas = get_connected_components(
                    mask_flat > self.mask_threshold
                )
                is_hole = (labels > 0) & (areas <= self.max_sprinkle_area)
                is_hole = is_hole.reshape_as(masks)
                # We fill holes with negative mask score (-10.0) to change them to background.
                masks = torch.where(is_hole, self.mask_threshold - 10.0, masks)
        except Exception as e:
            # Skip the post-processing step if the CUDA kernel fails
            warnings.warn(
                f"{e}\\n\\nSkipping the post-processing step due to the error above. You can "
                "still use SAM 2 and it's OK to ignore the error above, although some post-processing "
                "functionality may be limited (which doesn't affect the results in most cases; see "
                "https://github.com/facebookresearch/sam2/blob/main/INSTALL.md).",
                category=UserWarning,
                stacklevel=2,
            )
            masks = input_masks

        masks = F.interpolate(masks, orig_hw, mode="bilinear", align_corners=False)
        return masks
'''

# Native module: test-code/source/MedSAM2-main/setup.py
_EMBEDDED_SOURCES['test-code/source/MedSAM2-main/setup.py'] = '''# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import os

from setuptools import find_packages, setup

# Package metadata
NAME = "MedSAM2"
VERSION = "1.0"
DESCRIPTION = "MedSAM2 was adapted from SAM2 (https://github.com/facebookresearch/sam2) for medical image segmentation."
URL = "https://github.com/bowang-lab/MedSAM2"
AUTHOR = "WangLab"
AUTHOR_EMAIL = "medseg20s@gmail.com"
LICENSE = "Apache 2.0"

# Read the contents of README file
with open("README.md", "r", encoding="utf-8") as f:
    LONG_DESCRIPTION = f.read()

# Required dependencies
REQUIRED_PACKAGES = [
    "torch>=2.5.1",
    "torchvision>=0.20.1",
    "numpy>=2.0.1",
    "tqdm>=4.66.5",
    "hydra-core>=1.3.2",
    "iopath>=0.1.10",
    "pillow>=10.4.0",
    "SimpleITK>=2.4.0",
]

EXTRA_PACKAGES = {
    "notebooks": [
        "matplotlib>=3.9.1",
        "jupyter>=1.0.0",
        "opencv-python>=4.10.0",
        "eva-decord>=0.6.1",
    ],
    "interactive-demo": [
        "Flask>=3.0.3",
        "Flask-Cors>=5.0.0",
        "av>=13.0.0",
        "dataclasses-json>=0.6.7",
        "eva-decord>=0.6.1",
        "gunicorn>=23.0.0",
        "imagesize>=1.4.1",
        "pycocotools>=2.0.8",
        "strawberry-graphql>=0.239.2",
    ],
    "dev": [
        "matplotlib>=3.9.1",
        "jupyter>=1.0.0",
        "black==24.2.0",
        "usort==1.0.2",
        "ufmt==2.0.0b2",
        "fvcore>=0.1.5.post20221221",
        "pandas>=2.2.3",
        "scikit-image>=0.24.0",
        "tensorboard>=2.17.0",
        "pycocotools>=2.0.8",
        "tensordict>=0.5.0",
        "opencv-python>=4.10.0",
        "submitit>=1.5.1",
    ],
}

# By default, we also build the SAM 2 CUDA extension.
# You may turn off CUDA build with `export SAM2_BUILD_CUDA=0`.
BUILD_CUDA = os.getenv("SAM2_BUILD_CUDA", "1") == "1"
# By default, we allow SAM 2 installation to proceed even with build errors.
# You may force stopping on errors with `export SAM2_BUILD_ALLOW_ERRORS=0`.
BUILD_ALLOW_ERRORS = os.getenv("SAM2_BUILD_ALLOW_ERRORS", "1") == "1"

# Catch and skip errors during extension building and print a warning message
# (note that this message only shows up under verbose build mode
# "pip install -v -e ." or "python setup.py build_ext -v")
CUDA_ERROR_MSG = (
    "{}\\n\\n"
    "Failed to build the SAM 2 CUDA extension due to the error above. "
    "You can still use SAM 2 and it's OK to ignore the error above, although some "
    "post-processing functionality may be limited (which doesn't affect the results in most cases; "
    "(see https://github.com/facebookresearch/sam2/blob/main/INSTALL.md).\\n"
)


def get_extensions():
    if not BUILD_CUDA:
        return []

    try:
        from torch.utils.cpp_extension import CUDAExtension

        srcs = ["sam2/csrc/connected_components.cu"]
        compile_args = {
            "cxx": [],
            "nvcc": [
                "-DCUDA_HAS_FP16=1",
                "-D__CUDA_NO_HALF_OPERATORS__",
                "-D__CUDA_NO_HALF_CONVERSIONS__",
                "-D__CUDA_NO_HALF2_OPERATORS__",
            ],
        }
        ext_modules = [CUDAExtension("sam2._C", srcs, extra_compile_args=compile_args)]
    except Exception as e:
        if BUILD_ALLOW_ERRORS:
            print(CUDA_ERROR_MSG.format(e))
            ext_modules = []
        else:
            raise e

    return ext_modules


try:
    from torch.utils.cpp_extension import BuildExtension

    class BuildExtensionIgnoreErrors(BuildExtension):

        def finalize_options(self):
            try:
                super().finalize_options()
            except Exception as e:
                print(CUDA_ERROR_MSG.format(e))
                self.extensions = []

        def build_extensions(self):
            try:
                super().build_extensions()
            except Exception as e:
                print(CUDA_ERROR_MSG.format(e))
                self.extensions = []

        def get_ext_filename(self, ext_name):
            try:
                return super().get_ext_filename(ext_name)
            except Exception as e:
                print(CUDA_ERROR_MSG.format(e))
                self.extensions = []
                return "_C.so"

    cmdclass = {
        "build_ext": (
            BuildExtensionIgnoreErrors.with_options(no_python_abi_suffix=True)
            if BUILD_ALLOW_ERRORS
            else BuildExtension.with_options(no_python_abi_suffix=True)
        )
    }
except Exception as e:
    cmdclass = {}
    if BUILD_ALLOW_ERRORS:
        print(CUDA_ERROR_MSG.format(e))
    else:
        raise e


# Setup configuration
setup(
    name=NAME,
    version=VERSION,
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    long_description_content_type="text/markdown",
    url=URL,
    author=AUTHOR,
    author_email=AUTHOR_EMAIL,
    license=LICENSE,
    packages=find_packages(exclude="notebooks"),
    include_package_data=True,
    install_requires=REQUIRED_PACKAGES,
    extras_require=EXTRA_PACKAGES,
    python_requires=">=3.10.0",
    ext_modules=get_extensions(),
    cmdclass=cmdclass,
)
'''

load_embedded_path = install_embedded(Path(__file__).resolve().parent / "weights", _EMBEDDED_SOURCES)
load_native = load_embedded_path.load_native



if __name__ == "__main__":
    raise SystemExit(_standalone_runtime_ns["main"](_standalone_sys.modules[__name__]))
