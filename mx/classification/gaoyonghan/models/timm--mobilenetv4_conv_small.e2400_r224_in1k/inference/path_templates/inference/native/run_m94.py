#!/usr/bin/env python3
"""M94: native timm model, author transforms/split/eval_violence, published local weights."""
import argparse
import ast
from collections import defaultdict
from datetime import datetime
import hashlib
import importlib.metadata
import inspect
import json
from pathlib import Path
import shutil
import subprocess
import time

HERE = Path(__file__).resolve().parent


def file_sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def save(path, value):
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    tmp.replace(path)


def source_namespace(source, spec, device):
    import torch
    from torchvision import transforms
    if file_sha(source) != spec['source_sha256']:
        raise ValueError('作者源码哈希不符')
    cells = json.loads(source.read_text())['cells']
    ns = {'torch': torch, 'transforms': transforms, 'device': device}
    # Execute only the two original preprocessing assignments and the original evaluator.
    nodes = [n for n in ast.parse(''.join(cells[8]['source'])).body if isinstance(n, ast.Assign)
             and any(isinstance(t, ast.Name) and t.id in ['img_size', 'data_transforms'] for t in n.targets)]
    if len(nodes) != 2:
        raise ValueError('作者预处理代码结构变化')
    nodes += [n for n in ast.parse(''.join(cells[19]['source'])).body if isinstance(n, ast.FunctionDef) and n.name == 'eval_violence']
    if len(nodes) != 3:
        raise ValueError('作者eval_violence缺失')
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(source), 'exec'), ns)
    return ns, ''.join(cells[14]['source'])


def prepare(spec, source):
    import torch
    import timm
    from PIL import Image
    from torchvision.datasets import ImageFolder
    from safetensors.torch import load_file
    versions = {k: importlib.metadata.version(k) for k in spec['versions']}
    if versions != spec['versions']:
        raise ValueError(f'依赖版本变化：{versions}')
    root = Path(spec['model_path'])
    for name, expected in spec['files_sha256'].items():
        if file_sha(root / name) != expected:
            raise ValueError(f'模型文件变化：{name}')
    if file_sha(spec['zip_path']) != spec['zip_sha256']:
        raise ValueError('数据压缩包变化')
    ns, split_source = source_namespace(source, spec, 'cpu')
    ds = ImageFolder(spec['dataset_root'], ns['data_transforms'])
    if len(ds) != spec['images'] or ds.class_to_idx != spec['class_to_idx']:
        raise ValueError('图像数量或ImageFolder类别顺序不符')
    ns.update(image_datasets=ds, dataset_sizes=len(ds))
    exec(compile(split_source, str(source), 'exec'), ns)  # Unchanged author random_split cell.
    splits = {name: ns[key].indices for name, key in [('train', 'train_dataset'), ('valid', 'val_dataset'), ('test', 'test_dataset')]}
    for name, indices in splits.items():
        if digest(indices) != spec['split_indices_sha256'][name]:
            raise ValueError(f'作者划分恢复不符：{name}')
    split_of = {i: name for name, indices in splits.items() for i in indices}
    rows, groups = [], defaultdict(list)
    print('核对全部11073张图片、作者划分及重复图；不删除、不重标。', flush=True)
    for i, (path, target) in enumerate(ds.samples):
        with Image.open(path) as image:
            image.convert('RGB').load()
        row = {'index': i, 'path': str(Path(path).relative_to(ds.root)), 'target': target,
               'sha256': file_sha(path), 'split': split_of[i]}
        rows.append(row)
        groups[row['sha256']].append(row)
    if digest(rows) != spec['full_manifest_sha256']:
        raise ValueError('图片字节、标签或排序与已审核清单不符')
    duplicates = [group for group in groups.values() if len(group) > 1]
    test_rows = [rows[i] for i in splits['test']]
    if len(test_rows) != ns['val_len'] or len(test_rows) != 1660:
        raise ValueError('原函数val_len分母与test大小不一致，不能直接计分')
    model = timm.create_model('vit_base_patch16_224', pretrained=False)
    model.head = torch.nn.Linear(model.head.in_features, 2)
    state = load_file(str(root / 'model.safetensors'), device='cpu')
    model.load_state_dict(state, strict=True)
    del state
    print('预检通过：原生timm权重严格加载；test=1660（非暴力771、暴力889）。未执行模型前向。', flush=True)
    return ns, model, rows, test_rows, splits, duplicates


def evaluate(ns, model, test_rows, out, limit=0):
    import torch
    from torch.utils.data import DataLoader, Subset
    chosen = test_rows[:limit] if limit else test_rows
    out.mkdir()
    loader = DataLoader(Subset(ns['image_datasets'], [r['index'] for r in chosen]), batch_size=64, shuffle=False)
    predictions = []
    started = time.monotonic()
    with (out / 'predictions.jsonl').open('w') as handle:
        def record(_module, _inputs, logits):
            if logits.ndim != 2 or logits.shape[1] != 2 or not torch.isfinite(logits).all():
                raise ValueError('原生模型输出无效')
            for pred, values in zip(logits.argmax(1).cpu().tolist(), logits.detach().cpu().tolist()):
                row = chosen[len(predictions)]
                handle.write(json.dumps(dict(row, source_dataset='abdulmananraja/real-life-violence-situations',
                                             prediction=pred, logits=values)) + '\n')
                predictions.append(pred)
            handle.flush()
            save(out / 'status.json', {'state': 'running', 'processed': len(predictions), 'images': len(chosen)})
            print(f'[m94] {len(predictions)}/{len(chosen)}', flush=True)
        hook = model.register_forward_hook(record)
        try:
            # Uses the exact author loop, argmax, correct-count sum and val_len denominator.
            native = float(ns['eval_violence'](model, loader).item())
        finally:
            hook.remove()
    correct = sum(pred == row['target'] for pred, row in zip(predictions, chosen))
    if len(predictions) != len(chosen) or abs(native - correct / ns['val_len']) > 1e-6:
        raise ValueError('原生返回值与逐图计数不一致')
    full = len(chosen) == len(test_rows)
    save(out / 'status.json', {'state': 'complete' if full else 'partial_complete', 'processed': len(predictions)})
    # The original denominator stays 1660 even for a smoke subset; no smoke metric is published.
    return {'images': len(chosen), 'correct': correct, 'native_accuracy_percent': native * 100,
            'exact_accuracy_percent': correct / len(chosen) * 100,
            'elapsed_inference_s': time.monotonic() - started, 'predictions_sha256': file_sha(out / 'predictions.jsonl')}


def main():
    spec = json.loads((HERE / 'models_m94.json').read_text())
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--describe', action='store_true')
    parser.add_argument('--preflight', action='store_true')
    parser.add_argument('--max-images', type=int, default=0)
    parser.add_argument('--output-root', type=Path, default=Path('__MODEL_ROOT__/resources/runtime/assets/workspace/results/m94_native_v1'))
    args = parser.parse_args()
    if args.describe:
        print(json.dumps(spec, ensure_ascii=False, indent=2))
        return
    if not 0 <= args.max_images <= 1660:
        parser.error('MAX_IMAGES应为0到1660；0表示完整test')
    source = args.output_root / 'assets' / 'author.ipynb'
    source.parent.mkdir(parents=True, exist_ok=True)
    if not source.exists():
        part = source.with_suffix('.ipynb.part')
        bundled = HERE / 'official_m94.ipynb'
        if bundled.exists():
            if file_sha(bundled) != spec['source_sha256']:
                raise ValueError('随脚本保存的作者源码校验失败')
            shutil.copy2(bundled, part)
            print('使用随脚本保存的作者原始笔记本，无需连接GitHub。', flush=True)
        else:
            subprocess.run(['curl', '--http1.1', '--fail', '--location', '--retry', '3', '--connect-timeout', '30',
                            '--max-time', '180', spec['source_url'], '-o', str(part)], check=True)
        if file_sha(part) != spec['source_sha256']:
            raise ValueError('下载源码校验失败，保留文件停止')
        part.replace(source)
    ns, model, rows, test_rows, splits, duplicates = prepare(spec, source)
    if args.preflight:
        return
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA不可用，不回退CPU推理')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    ns['device'] = torch.device('cuda:0')
    full = args.max_images in (0, 1660)
    out = args.output_root / ('full' if full else 'smoke') / datetime.now().strftime('run_%Y%m%d_%H%M%S_%f')
    out.mkdir(parents=True)
    for name in ['run_m94.py', 'run_m94.sh', 'models_m94.json', 'README_m94.md']:
        shutil.copy2(HERE / name, out / name)
    shutil.copy2(source, out / source.name)
    protocol = dict(spec, python=__import__('platform').python_version(), cuda=torch.version.cuda,
        device=torch.cuda.get_device_name(0), model_parameter_count=sum(p.numel() for p in model.parameters()),
        model_source_sha256=file_sha(inspect.getfile(type(model))),
        snapshot_sha256={p.name: file_sha(p) for p in out.iterdir() if p.is_file()})
    fingerprint = digest(protocol)
    protocol['protocol_fingerprint'] = fingerprint
    save(out / 'protocol.json', protocol)
    save(out / 'full_data_manifest.json', rows)
    save(out / 'test_manifest.json', test_rows)
    save(out / 'split_indices.json', splits)
    save(out / 'duplicate_audit.json', duplicates)
    save(out / 'status.json', {'state': 'running', 'protocol_fingerprint': fingerprint})
    try:
        model = model.to(ns['device']).eval()
        evaluate(ns, model, test_rows, out / 'smoke_check', min(args.max_images or 10, 10))
        print('原生10张以内冒烟通过；该子集不报告accuracy。', flush=True)
        if full or args.max_images > 10:
            result = evaluate(ns, model, test_rows, out / 'evaluation', 0 if full else args.max_images)
            if full:
                result.update(model_number=94, dataset=spec['dataset'], dataset_relation=spec['dataset_relation'],
                    fidelity=spec['fidelity'], protocol_fingerprint=fingerprint, official_references=spec['official_references'],
                    officially_comparable=False, official_difference_pp=None, data_audit=spec['data_audit'])
                save(out / 'metrics.json', result)
                print(f"原生accuracy={result['native_accuracy_percent']:.4f}%（{result['correct']}/1660）", flush=True)
        status = {'state': 'complete' if full else 'partial_complete', 'images': 1660 if full else args.max_images,
                  'protocol_fingerprint': fingerprint, 'officially_comparable': False}
        save(out / 'status.json', status)
        save(out / 'audit_summary.json', dict(status, model_number=94, data_audit=spec['data_audit']))
        print(f'结果：{out}', flush=True)
    except Exception as exc:
        save(out / 'status.json', {'state': 'failed', 'error': f'{type(exc).__name__}: {exc}', 'protocol_fingerprint': fingerprint})
        raise


if __name__ == '__main__':
    main()
