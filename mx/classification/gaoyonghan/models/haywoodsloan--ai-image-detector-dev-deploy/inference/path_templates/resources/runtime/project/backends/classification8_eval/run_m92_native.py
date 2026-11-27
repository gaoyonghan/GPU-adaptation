#!/usr/bin/env python3
"""Run untouched HuggingPics validation classes with local M92 weights and frozen test data."""
import argparse
import ast
import hashlib
import importlib.metadata
import io
import json
from datetime import datetime
from pathlib import Path
import shutil
import subprocess
import time
import zipfile

from run_m92 import HERE, audit, save, sha256


def native_classes(path, spec):
    import torch
    import pytorch_lightning as pl
    from torchmetrics import Accuracy
    if sha256(path) != spec['source_sha256']:
        raise ValueError('HuggingPics源码哈希不符')
    nodes = []
    for cell in json.loads(path.read_text())['cells']:
        source = ''.join(cell.get('source', []))
        if cell['cell_type'] == 'code' and any(f'class {name}' in source for name in spec['native_classes']):
            nodes.extend(n for n in ast.parse(source).body if isinstance(n, ast.ClassDef) and n.name in spec['native_classes'])
    if sorted(n.name for n in nodes) != sorted(spec['native_classes']):
        raise ValueError('原生类缺失或重复')
    namespace = {'__name__': __name__, 'torch': torch, 'pl': pl, 'Accuracy': Accuracy}
    # Execute only the two unchanged class definitions, never installation/training/upload cells.
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec'), namespace)
    return namespace['Classifier'], namespace['ImageClassificationCollator']


def validate(model, rows, processor, classes, data_zip, out, spec, accelerator='gpu'):
    import torch
    import pytorch_lightning as pl
    from torch.utils.data import DataLoader, Dataset
    from PIL import Image
    NativeClassifier, NativeCollator = classes
    out.mkdir()
    predictions = []
    wrapper = NativeClassifier(model)
    started = time.monotonic()
    with zipfile.ZipFile(data_zip) as z, (out / 'predictions.jsonl').open('w') as handle:
        class Images(Dataset):
            def __len__(self):
                return len(rows)

            def __getitem__(self, i):
                row = rows[i]
                payload = z.read(row['member'])
                if hashlib.sha256(payload).hexdigest() != row['sha256']:
                    raise ValueError('测试图片变化')
                with Image.open(io.BytesIO(payload)) as im:
                    return im.convert('RGB'), row['target']

        def record(_module, _inputs, kwargs, output):
            logits = output.logits
            if logits.ndim != 2 or logits.shape[1] != 5 or not torch.isfinite(logits).all():
                raise ValueError('原生前向结果异常')
            for target, pred in zip(kwargs['labels'].cpu().tolist(), logits.argmax(1).cpu().tolist()):
                row = rows[len(predictions)]
                if row['target'] != target:
                    raise ValueError('原生验证标签/顺序与清单不一致')
                predictions.append(pred)
                handle.write(json.dumps(dict(row, prediction=pred)) + '\n')
            if len(predictions) % 200 == 0 or len(predictions) == len(rows):
                handle.flush()
                print(f'[m92 native] {len(predictions)}/{len(rows)}', flush=True)
            save(out / 'status.json', {'state': 'running', 'processed': len(predictions), 'images': len(rows)})

        loader = DataLoader(Images(), batch_size=spec['batch_size'], shuffle=False,
                            num_workers=spec['num_workers'], collate_fn=NativeCollator(processor))
        hook = wrapper.register_forward_hook(record, with_kwargs=True)
        try:
            trainer = pl.Trainer(accelerator=accelerator, devices=1,
                precision=spec['precision'] if accelerator == 'gpu' else '32-true', max_epochs=4,
                logger=False, enable_checkpointing=False, enable_model_summary=False, enable_progress_bar=False)
            values = trainer.validate(wrapper, dataloaders=loader, verbose=False)[0]
        finally:
            hook.remove()
    if len(predictions) != len(rows):
        raise ValueError('原生验证未覆盖完整样本')
    exact = sum(p == r['target'] for p, r in zip(predictions, rows)) / len(rows)
    if abs(values['val_acc'] - exact) > 1e-6:
        raise ValueError(f'原生val_acc与逐图复算不符：{values["val_acc"]} vs {exact}')
    result = dict(native_val_acc_percent=100 * values['val_acc'], native_val_loss=values['val_loss'],
        exact_top1_accuracy_percent=100 * exact, correct=round(exact * len(rows)), images=len(rows),
        elapsed_s=time.monotonic() - started, predictions_sha256=sha256(out / 'predictions.jsonl'))
    save(out / 'status.json', {'state': 'complete' if len(rows) == 1000 else 'partial_complete',
                              'processed': len(rows), 'images': len(rows)})
    return result, predictions


def main():
    spec = json.loads((HERE / 'models_m92_native.json').read_text())
    base = json.loads((HERE / 'models_m92.json').read_text())
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--describe', action='store_true')
    p.add_argument('--preflight', action='store_true')
    p.add_argument('--max-images', type=int, default=0)
    p.add_argument('--output-root', type=Path, default=Path('__MODEL_ROOT__/resources/runtime/assets/workspace/results/m92_huggingpics_native_v1'))
    args = p.parse_args()
    if args.describe:
        print(json.dumps({'native_protocol': spec, 'frozen_model_and_data': base}, ensure_ascii=False, indent=2))
        return
    if not 0 <= args.max_images <= 1000:
        p.error('MAX_IMAGES须为0到1000')
    versions = {name: importlib.metadata.version(name) for name in spec['dependencies']}
    if versions != spec['dependencies']:
        raise ValueError(f'原生依赖版本不符：{versions}')
    source = args.output_root / 'assets' / 'HuggingPics.ipynb'
    source.parent.mkdir(parents=True, exist_ok=True)
    if not source.exists():
        part = source.with_suffix('.ipynb.part')
        subprocess.run(['curl', '--fail', '--location', '--retry', '3', '--connect-timeout', '30',
                        '--max-time', '180', spec['source_url'], '-o', str(part)], check=True)
        if sha256(part) != spec['source_sha256']:
            raise ValueError('下载源码哈希异常，保留文件停止')
        part.replace(source)
    classes = native_classes(source, spec)
    # The original pipeline used sklearn; native validation uses TorchMetrics instead.
    base_versions = {name: version for name, version in base['versions'].items() if name != 'scikit-learn'}
    rows, _ = audit(dict(base, versions=base_versions))
    baseline = Path(spec['baseline'])
    if sha256(baseline / 'predictions.jsonl') != spec['baseline_predictions_sha256']:
        raise ValueError('旧结果对照变化')
    if rows != json.loads((baseline / 'manifest.json').read_text()):
        raise ValueError('新旧测试图或标签不一致')
    import torch
    import pytorch_lightning as pl
    from transformers import ViTFeatureExtractor, ViTForImageClassification
    processor = ViTFeatureExtractor.from_pretrained(base['model_path'], local_files_only=True)
    if args.preflight:
        print('原生类、依赖、数据与对照预检通过；未加载模型或运行前向。', flush=True)
        return
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA不可用，不回退CPU推理')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    pl.seed_everything(spec['seed'], workers=True)
    selected = rows[:args.max_images] if args.max_images else rows
    full = len(selected) == 1000
    out = args.output_root / ('full' if full else 'smoke') / datetime.now().strftime('run_%Y%m%d_%H%M%S_%f')
    out.mkdir(parents=True)
    for name in ['run_m92_native.py', 'run_m92_native.sh', 'models_m92_native.json',
                 'run_m92.py', 'models_m92.json', 'README_m92.md']:
        shutil.copy2(HERE / name, out / name)
    shutil.copy2(source, out / source.name)
    protocol = dict(spec, model_and_data=base, versions={**base_versions, **versions},
        tf32=False, device=torch.cuda.get_device_name(0), torch_cuda=torch.version.cuda,
        preprocessing='HuggingPics collator + local ViTFeatureExtractor; ImageFolder-compatible RGB without EXIF transpose',
        snapshot_sha256={f.name: sha256(f) for f in out.iterdir() if f.is_file()})
    fingerprint = hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest()
    protocol['protocol_fingerprint'] = fingerprint
    save(out / 'protocol.json', protocol)
    save(out / 'manifest.json', selected)
    save(out / 'status.json', {'state': 'running', 'protocol_fingerprint': fingerprint})
    try:
        model, info = ViTForImageClassification.from_pretrained(base['model_path'], local_files_only=True,
            use_safetensors=True, torch_dtype=torch.float32, attn_implementation='eager', output_loading_info=True)
        if any(info.values()):
            raise ValueError(f'模型与权重不严格匹配：{info}')
        validate(model, selected[:10], processor, classes, base['zip_path'], out / 'smoke_check', spec)
        print('原生验证冒烟通过。', flush=True)
        if full or len(selected) > 10:
            result, predictions = validate(model, selected, processor, classes, base['zip_path'], out / 'evaluation', spec)
            if full:
                old = [json.loads(line) for line in (baseline / 'predictions.jsonl').read_text().splitlines()]
                old_acc = 100 * sum(r['prediction'] == r['target'] for r in old) / len(old)
                result.update(model_number=92, fidelity=spec['fidelity'], dataset_relation=spec['dataset_relation'],
                    officially_comparable=False, official_reference=base['official_reference'], official_difference_pp=None,
                    baseline_top1_percent=old_acc, baseline_difference_pp=result['exact_top1_accuracy_percent'] - old_acc,
                    changed_predictions=sum(a != b['prediction'] for a, b in zip(predictions, old)), protocol_fingerprint=fingerprint)
                save(out / 'metrics.json', result)
                print(f"原生val_acc={result['native_val_acc_percent']:.4f}%，旧结果={old_acc:.4f}%", flush=True)
        save(out / 'status.json', {'state': 'complete' if full else 'partial_complete', 'images': len(selected),
                                   'protocol_fingerprint': fingerprint, 'officially_comparable': False})
        print(f'结果：{out}', flush=True)
    except Exception as exc:
        save(out / 'status.json', {'state': 'failed', 'error': f'{type(exc).__name__}: {exc}', 'protocol_fingerprint': fingerprint})
        raise


if __name__ == '__main__':
    main()
