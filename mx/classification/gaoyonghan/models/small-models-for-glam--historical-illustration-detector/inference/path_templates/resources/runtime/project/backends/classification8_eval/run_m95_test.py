#!/usr/bin/env python3
"""M95: test-only downloader and inference-only entry to AutoTrain's evaluator."""
import argparse
import ast
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import hashlib
import importlib.metadata
import json
from pathlib import Path
import shutil
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
SPEC = json.loads((HERE / 'models_m95.json').read_text())
DATA = Path(SPEC['dataset_root'])


def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def check_environment():
    versions = {name: importlib.metadata.version(name) for name in SPEC['versions']}
    if versions != SPEC['versions']:
        raise RuntimeError(f'环境与冻结配置不同: {versions}')
    return versions


def file_label(path):
    parts = Path(path).parts
    if len(parts) != 5 or parts[:2] != ('data', 'test') or parts[2] not in [f'set-{i:03d}' for i in range(4)]:
        raise ValueError(f'非预期test路径，停止: {path}')
    return SPEC['class_to_idx'][parts[3]]


def verify_file(row):
    path = DATA / row['path']
    if not path.is_file() or path.stat().st_size != row['size'] or sha(path) != row['sha256']:
        raise ValueError(f'文件缺失或校验不符，请重试下载: {row["path"]}')


def download():
    from huggingface_hub import HfApi, get_token, get_hf_file_metadata, hf_hub_download, hf_hub_url
    from huggingface_hub.utils import GatedRepoError, HfHubHTTPError
    token = get_token()
    probe = 'data/test/set-000/artificial/--JxQ-avxAl_5eBkwvCJ0u3vZ_s.jpg'
    try:
        get_hf_file_metadata(hf_hub_url(SPEC['dataset_id'], probe, repo_type='dataset', revision=SPEC['dataset_revision']), token=token)
    except (GatedRepoError, HfHubHTTPError) as exc:
        if getattr(exc.response, 'status_code', None) in (401, 403):
            raise SystemExit('HF权限不足：请先在haywoodsloan/ai-images页面申请获批，再在终端登录同一获批账号。未下载任何训练数据。') from None
        raise
    marker = DATA / 'test_manifest.json'
    if marker.exists():
        manifest = json.loads(marker.read_text())
        if manifest['revision'] != SPEC['dataset_revision'] or manifest['repo_id'] != SPEC['dataset_id']:
            raise ValueError('已有数据清单版本不符，不覆盖')
    else:
        files = []
        api = HfApi(endpoint='https://huggingface.co', token=token)
        for entry in api.list_repo_tree(SPEC['dataset_id'], path_in_repo='data/test', recursive=True,
                                       repo_type='dataset', revision=SPEC['dataset_revision']):
            if not hasattr(entry, 'lfs'):
                continue
            target = file_label(entry.path)
            if entry.lfs is None or len(entry.lfs.sha256) != 64 or set(entry.lfs.sha256) == {'*'}:
                raise ValueError(f'未取得真实LFS校验值: {entry.path}')
            files.append(dict(path=entry.path, target=target, size=entry.size, sha256=entry.lfs.sha256))
            if len(files) % 10000 == 0:
                print(f'已列出 {len(files)} 个test文件', flush=True)
        files.sort(key=lambda x: x['path'])
        if not files or {Path(r['path']).parts[2] for r in files} != {f'set-{i:03d}' for i in range(4)}:
            raise ValueError('test分片不完整')
        if len({r['path'] for r in files}) != len(files) or {r['target'] for r in files} != {0, 1}:
            raise ValueError('test文件重复或类别不全')
        manifest = dict(repo_id=SPEC['dataset_id'], revision=SPEC['dataset_revision'], split='test', files=files)
        save(marker, manifest)
    files = manifest['files']
    print(f"仅下载test: {len(files)}张，{sum(r['size'] for r in files)/1024**3:.2f} GiB，类别计数{dict(Counter(r['target'] for r in files))}", flush=True)
    missing_bytes = sum(r['size'] for r in files if not (DATA / r['path']).is_file())
    if shutil.disk_usage(DATA).free < missing_bytes + 1024**3:
        raise RuntimeError('磁盘剩余空间不足，停止下载')
    def fetch(row):
        file_label(row['path'])
        try:
            verify_file(row)
            return
        except ValueError:
            pass
        existing = (DATA / row['path']).exists()
        hf_hub_download(SPEC['dataset_id'], row['path'], repo_type='dataset', revision=SPEC['dataset_revision'],
                        local_dir=DATA, token=token, force_download=existing)
        verify_file(row)
    with ThreadPoolExecutor(max_workers=4) as pool:
        for i, _ in enumerate(pool.map(fetch, files), 1):
            if i % 100 == 0 or i == len(files):
                print(f'[test下载及校验] {i}/{len(files)}', flush=True)
    print('完整test下载并校验完成；没有下载train。', flush=True)


def official_functions():
    import albumentations as A
    import numpy as np
    import torch
    from sklearn import metrics
    root = HERE / 'm95_official'
    for name, expected in SPEC['source_sha256'].items():
        if sha(root / name) != expected:
            raise ValueError(f'官方源码变化: {name}')
    ns = dict(A=A, np=np, torch=torch, metrics=metrics)
    exec(compile((root / 'dataset.py').read_text(), str(root / 'dataset.py'), 'exec'), ns)
    tree = ast.parse((root / 'utils.py').read_text())
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef)
             and n.name in ['process_data', '_binary_classification_metrics']]
    if len(nodes) != 2:
        raise ValueError('官方函数缺失')
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(root / 'utils.py'), 'exec'), ns)
    return ns


def prepare(out):
    from datasets import load_dataset, Image
    from transformers import AutoImageProcessor, AutoModelForImageClassification
    versions = check_environment()
    ns = official_functions()
    params = json.loads((HERE / 'm95_official/training_params.json').read_text())
    if params['data_path'] != SPEC['dataset_id'] or params['valid_split'] != 'test' or params['mixed_precision'] is not None:
        raise ValueError('官方训练参数与预期不同')
    manifest = json.loads((DATA / 'test_manifest.json').read_text())
    if manifest['revision'] != SPEC['dataset_revision'] or manifest['repo_id'] != SPEC['dataset_id']:
        raise ValueError('数据清单版本不符')
    rows = manifest['files']
    for i, row in enumerate(rows, 1):
        if row['target'] != file_label(row['path']):
            raise ValueError('类别映射不符')
        verify_file(row)
        if i % 5000 == 0:
            print(f'[test预检] {i}/{len(rows)}', flush=True)
    if not rows or len({r['path'] for r in rows}) != len(rows) or {r['target'] for r in rows} != {0, 1}:
        raise ValueError('test清单无效')
    if {Path(r['path']).parts[2] for r in rows} != {f'set-{i:03d}' for i in range(4)}:
        raise ValueError('test四个分片不完整')
    ds = load_dataset('imagefolder', data_files={'test': [str(DATA / r['path']) for r in rows]},
                      split='test', cache_dir=str(out / 'dataset_cache'))
    if ds.features['label'].names != ['artificial', 'real'] or len(ds) != len(rows):
        raise ValueError('原生ImageFolder识别的标签或样本数量不符')
    undecoded = ds.cast_column('image', Image(decode=False))
    mapping = {str(DATA / r['path']): r for r in rows}
    ordered = [mapping[item['image']['path']] for item in undecoded]
    if len({r['path'] for r in ordered}) != len(rows) or list(ds['label']) != [r['target'] for r in ordered]:
        raise ValueError('数据加载后样本或标签不符')
    original = json.loads((HERE / 'models.json').read_text())
    spec = next(m for m in original['models'] if m['sequence'] == 95)
    model_root = Path(original['model_root']) / spec['model_id']
    for name, expected in spec['files'].items():
        if sha(model_root / name) != expected['sha256']:
            raise ValueError(f'原权重文件变化: {name}')
    processor = AutoImageProcessor.from_pretrained(model_root, local_files_only=True)
    model, info = AutoModelForImageClassification.from_pretrained(model_root, local_files_only=True, output_loading_info=True)
    if any(info[k] for k in ['missing_keys', 'unexpected_keys', 'mismatched_keys', 'error_msgs']):
        raise ValueError(f'模型未严格加载: {info}')
    if model.config.label2id != SPEC['class_to_idx']:
        raise ValueError('模型标签映射不符')
    _, valid = ns['process_data']([], ds, processor, SimpleNamespace(**params))
    # Decode and apply original validation transforms on one sample of each class, no forward.
    for label in (0, 1):
        item = valid[next(i for i, r in enumerate(ordered) if r['target'] == label)]
        if tuple(item['pixel_values'].shape) != (3, 256, 256):
            raise ValueError('原生预处理尺寸不符')
    print(f'预检通过: {len(ds)}张test；权重严格加载；原生预处理。未执行模型前向。', flush=True)
    return ns, model, valid, ordered, params, versions, spec


def evaluate(ns, model, valid, rows, params, out, indices):
    import numpy as np
    import torch
    from torch.utils.data import Subset
    from transformers import Trainer, TrainingArguments
    out.mkdir()
    selected = [rows[i] for i in indices]
    args = TrainingArguments(output_dir=str(out), per_device_eval_batch_size=2 * params['batch_size'],
        report_to=[], dataloader_num_workers=0, fp16=False, bf16=False,
        eval_accumulation_steps=1)
    trainer = Trainer(model=model, args=args, eval_dataset=Subset(valid, indices),
                      compute_metrics=ns['_binary_classification_metrics'])
    predictions = []
    with (out / 'predictions.jsonl').open('w') as f:
        def record(_model, _inputs, output):
            logits = output.logits.detach().float().cpu().numpy()
            if logits.shape[1:] != (2,) or not np.isfinite(logits).all():
                raise ValueError('模型输出无效')
            for scores in logits:
                row = selected[len(predictions)]
                predictions.append(int(scores.argmax()))
                f.write(json.dumps(dict(row, logits=scores.tolist(), prediction=predictions[-1])) + '\n')
            f.flush()
            save(out / 'status.json', dict(state='running', processed=len(predictions), images=len(selected)))
        hook = model.register_forward_hook(record)
        try:
            result = trainer.evaluate()
        finally:
            hook.remove()
    correct = sum(p == r['target'] for p, r in zip(predictions, selected))
    if len(predictions) != len(selected) or abs(result['eval_accuracy'] - correct / len(selected)) > 1e-10:
        raise ValueError('原生指标与逐图计数不一致')
    result.update(images=len(selected), correct=correct, predictions_sha256=sha(out / 'predictions.jsonl'))
    save(out / 'metrics.json', result)
    save(out / 'status.json', dict(state='complete', images=len(selected)))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--download', action='store_true')
    parser.add_argument('--check-environment', action='store_true')
    parser.add_argument('--preflight', action='store_true')
    parser.add_argument('--max-images', type=int, default=0)
    parser.add_argument('--output-root', type=Path, default=Path('__MODEL_ROOT__/resources/runtime/assets/workspace/results/m95_author_test_v1'))
    args = parser.parse_args()
    if args.download:
        return download()
    if args.check_environment:
        print(check_environment())
        return
    if args.max_images not in (0, 10):
        parser.error('MAX_IMAGES仅支持0（完整test）或10（冒烟）')
    ns, model, valid, rows, params, versions, weights = prepare(args.output_root)
    if args.preflight:
        return
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA不可用，不回退CPU')
    out = args.output_root / ('smoke' if args.max_images else 'full') / datetime.now().strftime('run_%Y%m%d_%H%M%S_%f')
    out.mkdir(parents=True)
    for name in ['run_m95_test.py', 'run_m95_test.sh', 'models_m95.json', 'models.json', 'README_m95.md']:
        shutil.copy2(HERE / name, out / name)
    shutil.copytree(HERE / 'm95_official', out / 'm95_official')
    save(out / 'test_manifest.json', rows)
    save(out / 'protocol.json', dict(SPEC, versions=versions, training_params=params, weight_files=weights['files'],
        test_manifest_sha256=sha(DATA / 'test_manifest.json'), cuda=torch.version.cuda, gpu=torch.cuda.get_device_name(0),
        backend_flags=dict(matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,
                           cudnn_allow_tf32=torch.backends.cudnn.allow_tf32),
        source_snapshot_sha256={str(p.relative_to(out)): sha(p) for p in out.rglob('*') if p.is_file()},
        full_test=not bool(args.max_images), official_exact_reproduction=False))
    save(out / 'status.json', dict(state='running'))
    try:
        smoke = sorted([i for label in (0, 1) for i in [j for j,r in enumerate(rows) if r['target'] == label][:5]])
        if len(smoke) != 10:
            raise ValueError('不足两类各5张，停止')
        evaluate(ns, model, valid, rows, params, out / 'smoke_check', smoke)
        if not args.max_images:
            result = evaluate(ns, model, valid, rows, params, out / 'evaluation', list(range(len(rows))))
            result.update(dataset=SPEC['dataset_id'], split='test', dataset_revision=SPEC['dataset_revision'],
                          official_reference=SPEC['official_reference'], official_exact_reproduction=False)
            save(out / 'metrics.json', result)
            print(json.dumps(result, ensure_ascii=False, indent=2))
        save(out / 'status.json', dict(state='partial_complete' if args.max_images else 'complete',
                                      images=10 if args.max_images else len(rows)))
        print(f'结果: {out}', flush=True)
    except Exception as exc:
        save(out / 'status.json', dict(state='failed', error=f'{type(exc).__name__}: {exc}'))
        raise


if __name__ == '__main__':
    main()
