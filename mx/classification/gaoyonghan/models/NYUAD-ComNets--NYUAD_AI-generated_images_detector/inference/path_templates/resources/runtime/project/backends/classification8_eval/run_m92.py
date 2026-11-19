#!/usr/bin/env python3
"""M92: publisher model/processor + native Transformers batch pipeline on frozen test ZIP."""
import argparse
from collections import Counter
from datetime import datetime
import hashlib
import importlib.metadata
import inspect
import io
import json
import math
from pathlib import Path
import shutil
import time
import zipfile

HERE = Path(__file__).resolve().parent
SOURCES = ['run_m92.py', 'run_m92.sh', 'models_m92.json', 'README_m92.md']


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def save(path, value):
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temp.replace(path)


def audit(spec):
    from PIL import Image
    from transformers import ViTImageProcessor
    from safetensors import safe_open
    versions = {k: importlib.metadata.version(k) for k in spec['versions']}
    if versions != spec['versions']:
        raise ValueError(f'环境版本变化；不修改共享环境：{versions}')
    root = Path(spec['model_path'])
    for name, expected in spec['files_sha256'].items():
        if sha256(root / name) != expected:
            raise ValueError(f'模型文件变化：{name}')
    config = json.loads((root / 'config.json').read_text())
    if config['id2label'] != {str(i): label for i, label in enumerate(spec['labels'])}:
        raise ValueError('模型类别映射不符')
    with safe_open(root / 'model.safetensors', framework='pt') as f:
        if f.get_slice('classifier.weight').get_shape() != [5, 768]:
            raise ValueError('权重分类头不是5类')
    path = Path(spec['zip_path'])
    print('核验ZIP、1000张测试图及标签（不解压、不加载GPU模型）', flush=True)
    if sha256(path) != spec['zip_sha256']:
        raise ValueError('数据包SHA256不符')
    processor = ViTImageProcessor.from_pretrained(root, local_files_only=True)
    rows = []
    with zipfile.ZipFile(path) as z:
        if len(z.namelist()) != len(set(z.namelist())):
            raise ValueError('ZIP有重名路径')
        for name in sorted(z.namelist()):
            if not name.startswith(spec['split'] + '/') or not name.lower().endswith(('.jpg', '.jpeg', '.png')):
                continue
            parts = Path(name).parts
            if len(parts) != 4 or parts[-2] not in spec['labels']:
                raise ValueError(f'非预期标签/路径：{name}')
            payload = z.read(name)
            with Image.open(io.BytesIO(payload)) as im:
                rgb = im.convert('RGB')
                rgb.load()
                if not rows and processor(images=rgb, return_tensors='pt')['pixel_values'].shape != (1, 3, 224, 224):
                    raise ValueError('预处理尺寸不符')
            rows.append({'member': name, 'target': spec['labels'].index(parts[-2]),
                         'target_label': parts[-2], 'sha256': hashlib.sha256(payload).hexdigest(),
                         'source_dataset': spec['dataset'], 'source_split': spec['split']})
    if len(rows) != spec['images'] or dict(Counter(r['target_label'] for r in rows)) != spec['class_counts']:
        raise ValueError('测试集数量或类别覆盖不符')
    if len({r['sha256'] for r in rows}) != len(rows):
        raise ValueError('测试集出现字节相同的重复图')
    print(json.dumps({'status': 'preflight_ready', 'images': len(rows), 'class_counts': spec['class_counts'],
                      'dataset_relation': spec['dataset_relation']}, ensure_ascii=False), flush=True)
    return rows, processor


def main():
    spec = json.loads((HERE / 'models_m92.json').read_text())
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--describe', action='store_true')
    parser.add_argument('--preflight', action='store_true')
    parser.add_argument('--max-images', type=int, default=0)
    parser.add_argument('--output-root', type=Path, default=Path('__MODEL_ROOT__/resources/runtime/assets/workspace/results/m92_faceshape_v1'))
    args = parser.parse_args()
    if args.describe:
        print(json.dumps(spec, ensure_ascii=False, indent=2))
        return
    if not 0 <= args.max_images <= spec['images']:
        parser.error('MAX_IMAGES应为0到1000，0表示完整测试集')
    rows, processor = audit(spec)
    if args.preflight:
        return
    import torch
    from torch.utils.data import Dataset
    from PIL import Image
    from transformers import ViTForImageClassification, pipeline
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA不可用，不回退CPU推理')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    selected = rows[:args.max_images] if args.max_images else rows
    full = len(selected) == spec['images']
    out = args.output_root / ('full' if full else 'smoke') / datetime.now().strftime('run_%Y%m%d_%H%M%S_%f')
    out.mkdir(parents=True)
    for name in SOURCES:
        shutil.copy2(HERE / name, out / name)
    for name in ['README.md', 'config.json', 'preprocessor_config.json']:
        shutil.copy2(Path(spec['model_path']) / name, out / ('model_' + name))
    protocol = dict(spec, source_sha256={n: sha256(out / n) for n in SOURCES},
                    device_name=torch.cuda.get_device_name(0), torch_cuda=torch.version.cuda)
    fingerprint = hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest()
    protocol['protocol_fingerprint'] = fingerprint
    save(out / 'protocol.json', protocol)
    save(out / 'manifest.json', selected)
    targets, predictions = [], []

    def status(state, **extra):
        save(out / 'status.json', dict(state=state, processed=len(predictions), images=len(selected),
                                       protocol_fingerprint=fingerprint, **extra))

    status('running')
    try:
        model, loading = ViTForImageClassification.from_pretrained(spec['model_path'], local_files_only=True,
            use_safetensors=True, torch_dtype=torch.float32, attn_implementation='eager', output_loading_info=True)
        if any(loading.values()):
            raise ValueError(f'权重与模型结构不完整匹配：{loading}')
        classifier = pipeline('image-classification', model=model.eval(), image_processor=processor,
                              device=0, torch_dtype=torch.float32)
        native_sources = {}
        for obj in [type(classifier), type(classifier).__bases__[0], type(processor), type(model), accuracy_score]:
            path = Path(inspect.getfile(obj))
            native_sources[str(path)] = sha256(path)
        save(out / 'native_sources.json', native_sources)
        label_to_id = {label: i for i, label in enumerate(spec['labels'])}
        started = time.monotonic()
        with zipfile.ZipFile(spec['zip_path']) as z, (out / 'predictions.jsonl').open('w') as handle:
            class ZipImages(Dataset):
                def __init__(self, subset):
                    self.subset = subset

                def __len__(self):
                    return len(self.subset)

                def __getitem__(self, index):
                    row = self.subset[index]
                    payload = z.read(row['member'])
                    if hashlib.sha256(payload).hexdigest() != row['sha256']:
                        raise ValueError('推理期间图片发生变化')
                    with Image.open(io.BytesIO(payload)) as im:
                        im.load()
                        return im.copy()  # Leave RGB/EXIF handling to the native pipeline.

            # First 10 form the smoke gate; retain their predictions and continue without rerunning them.
            for subset in [selected[:10], selected[10:]]:
                if not subset:
                    continue
                before = len(predictions)
                for scores in classifier(ZipImages(subset), batch_size=8, num_workers=0, top_k=5):
                    if (len(scores) != 5 or {s['label'] for s in scores} != set(label_to_id)
                            or any(not math.isfinite(s['score']) or not 0 <= s['score'] <= 1 for s in scores)
                            or abs(sum(s['score'] for s in scores) - 1) > 1e-5):
                        raise ValueError('官方pipeline输出类别或概率无效')
                    row = selected[len(predictions)]
                    pred = label_to_id[max(scores, key=lambda s: s['score'])['label']]
                    handle.write(json.dumps(dict(row, prediction=pred, prediction_label=spec['labels'][pred], scores=scores)) + '\n')
                    targets.append(row['target'])
                    predictions.append(pred)
                    if len(predictions) % 50 == 0 or len(predictions) == len(selected):
                        handle.flush()
                        status('running')
                        print(f'[m92] {len(predictions)}/{len(selected)}', flush=True)
                if len(predictions) - before != len(subset):
                    raise ValueError('pipeline未返回全部样本')
                handle.flush()
                status('running')
                if before == 0:
                    print(f'[m92] 首{len(subset)}张冒烟通过，预测已保留。', flush=True)
        if len(predictions) != len(selected):
            raise ValueError('样本覆盖不完整')
        if full:
            result = {'state': 'complete', 'model_number': 92, 'images': len(selected),
                'top1_accuracy_percent': 100 * accuracy_score(targets, predictions),
                'classification_report': classification_report(targets, predictions, labels=list(range(5)),
                    target_names=spec['labels'], output_dict=True, zero_division=0),
                'confusion_matrix': confusion_matrix(targets, predictions, labels=list(range(5))).tolist(),
                'confusion_matrix_order': spec['labels'], 'elapsed_inference_s': time.monotonic() - started,
                'official_reference': spec['official_reference'], 'official_difference_pp': None,
                'fidelity': spec['fidelity'], 'dataset_relation': spec['dataset_relation'],
                'protocol_fingerprint': fingerprint, 'predictions_sha256': sha256(out / 'predictions.jsonl')}
            save(out / 'metrics.json', result)
            print(f"[m92] Top-1 accuracy = {result['top1_accuracy_percent']:.4f}%", flush=True)
        status('complete' if full else 'partial_complete', smoke_passed=True,
               predictions_sha256=sha256(out / 'predictions.jsonl'))
        save(out / 'audit_summary.json', {'model_number': 92, 'full_test_complete': full,
             'officially_comparable': False, 'result_directory': str(out), 'protocol_fingerprint': fingerprint})
        print(f'结果目录：{out}', flush=True)
    except Exception as exc:
        status('failed', error=f'{type(exc).__name__}: {exc}')
        raise


if __name__ == '__main__':
    main()
