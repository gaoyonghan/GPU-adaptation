#!/usr/bin/env python3
"""M99 author single-image inference on all published train rows; not a verified holdout."""
import argparse
import ast
from collections import Counter
from datetime import datetime
import hashlib
import importlib.metadata
import io
import json
from pathlib import Path
import re
import shutil
import time

HERE = Path(__file__).resolve().parent
FILES = ['run_m99.py', 'run_m99.sh', 'models_m99.json', 'README_m99.md']


def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def save(path, obj):
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def source_nodes(spec):
    card = Path(spec['model_path']) / 'README.md'
    if sha(card) != spec['model_files_sha256']['README.md']:
        raise ValueError('模型卡源码哈希变化')
    code = next(block for block in re.findall(r'```(?:python|py)\s*\n(.*?)```', card.read_text(), re.S)
                if 'def classify_image(' in block)
    nodes = [n for n in ast.parse(code).body if
             (isinstance(n, ast.FunctionDef) and n.name == 'classify_image') or
             (isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'id2label' for t in n.targets))]
    if len(nodes) != 2:
        raise ValueError('模型卡函数结构与预期不同')
    return ast.Module(body=nodes, type_ignores=[])


def rows(spec):
    import pyarrow.parquet as pq
    index = 0
    for batch in pq.ParquetFile(spec['dataset_path']).iter_batches(batch_size=64):
        for row in batch.to_pylist():
            yield index, row
            index += 1


def decode(row):
    import numpy as np
    from PIL import Image
    payload = row['image']['bytes']
    if not payload:
        raise ValueError('图片字节缺失，不联网补图或跳过')
    with Image.open(io.BytesIO(payload)) as image:
        image.load()
        # Gradio's default RGB numpy input, then the unmodified author function converts RGB again.
        array = np.array(image.convert('RGB'))
    return array, hashlib.sha256(payload).hexdigest()


def prepare(spec):
    import pyarrow.parquet as pq
    import torch
    from transformers import AutoImageProcessor, SiglipForImageClassification
    versions = {k: importlib.metadata.version(k) for k in spec['versions']}
    if versions != spec['versions']:
        raise ValueError(f'环境版本不同，不自动修改: {versions}')
    root = Path(spec['model_path'])
    for name, expected in spec['model_files_sha256'].items():
        if sha(root / name) != expected:
            raise ValueError(f'模型文件变化: {name}')
    if sha(spec['dataset_path']) != spec['dataset_sha256']:
        raise ValueError('作者数据文件变化')
    meta = json.loads(pq.ParquetFile(spec['dataset_path']).schema_arrow.metadata[b'huggingface'])
    if meta['info']['features']['label']['names'] != spec['class_names']:
        raise ValueError('数据标签名称或顺序变化')
    manifest, counts = [], Counter()
    for index, row in rows(spec):
        target = row['label']
        if target not in range(8):
            raise ValueError('未知类别')
        _, image_sha = decode(row)
        counts[target] += 1
        manifest.append(dict(row=index, target=target, source_path=row['image']['path'], image_sha256=image_sha))
        if (index + 1) % 2000 == 0:
            print(f'[m99预检] {index+1}/9165 图片解码通过', flush=True)
    if len(manifest) != spec['images'] or [counts[i] for i in range(8)] != spec['class_counts']:
        raise ValueError('样本数量或各类数量不符')
    processor = AutoImageProcessor.from_pretrained(root, local_files_only=True, use_fast=False)
    model, info = SiglipForImageClassification.from_pretrained(root, local_files_only=True, output_loading_info=True)
    if any(info[k] for k in ['missing_keys', 'unexpected_keys', 'mismatched_keys', 'error_msgs']):
        raise ValueError(f'权重严格加载失败: {info}')
    if model.config.id2label != dict(enumerate(spec['class_names'])):
        raise ValueError('模型类别映射不一致')
    seen = set()
    for _, row in rows(spec):
        if row['label'] in seen:
            continue
        array, _ = decode(row)
        values = processor(images=array, return_tensors='pt')['pixel_values']
        if values.shape != (1, 3, 224, 224) or not torch.isfinite(values).all():
            raise ValueError('预处理尺寸或像素无效')
        seen.add(row['label'])
        if len(seen) == 8:
            break
    source_nodes(spec)
    print('预检通过：9165张全部可解码，8类数量吻合，原权重严格加载；未执行模型前向。', flush=True)
    return model, processor, manifest, versions


def score(targets, predictions, names):
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
    if not targets or len(targets) != len(predictions) or any(x not in range(8) for x in targets + predictions):
        raise ValueError('计分输入无效')
    report = classification_report(targets, predictions, labels=list(range(8)), target_names=names,
                                   output_dict=True, zero_division=0)
    accuracy = float(accuracy_score(targets, predictions))
    correct = sum(a == b for a, b in zip(targets, predictions))
    if abs(accuracy - correct / len(targets)) > 1e-12:
        raise ValueError('准确率交叉检查失败')
    return dict(images=len(targets), correct=correct, accuracy=accuracy, accuracy_percent=accuracy * 100,
                macro_f1=report['macro avg']['f1-score'], weighted_f1=report['weighted avg']['f1-score'],
                classification_report=report, confusion_matrix=confusion_matrix(targets, predictions, labels=list(range(8))).tolist())


def evaluate(spec, model, processor, manifest, device, out, selected):
    import numpy as np
    import torch
    from PIL import Image
    out.mkdir()
    ns = dict(torch=torch, Image=Image, model=model,
              processor=lambda **kw: processor(**kw).to(device))
    exec(compile(source_nodes(spec), str(Path(spec['model_path']) / 'README.md'), 'exec'), ns)
    if ns['id2label'] != {str(i): n for i,n in enumerate(spec['class_names'])}:
        raise ValueError('模型卡函数的标签映射不符')
    captured = []
    def record(_module, _inputs, output):
        captured.append(output.logits.detach().float().cpu().numpy())
    hook = model.register_forward_hook(record)
    wanted = set(selected)
    targets, predictions, ties = [], [], 0
    started = time.monotonic()
    try:
        with (out / 'predictions.jsonl').open('w') as f:
            for index, row in rows(spec):
                if index not in wanted:
                    continue
                array, image_sha = decode(row)
                if image_sha != manifest[index]['image_sha256']:
                    raise ValueError('预检后图片变化')
                captured.clear()
                display = ns['classify_image'](array)  # Exact author function, not a rewritten forward.
                if len(captured) != 1 or captured[0].shape != (1, 8) or not np.isfinite(captured[0]).all():
                    raise ValueError('原生推理输出异常')
                logits = captured[0][0]
                pred = int(logits.argmax())
                shown = [display[name] for name in spec['class_names']]
                shown_pred = int(np.argmax(shown))
                ties += int(shown_pred != pred)
                targets.append(row['label']); predictions.append(pred)
                f.write(json.dumps(dict(manifest[index], prediction=pred, logits=logits.tolist(),
                                       author_display_probabilities=shown, author_display_top1=shown_pred)) + '\n')
                if len(targets) % 100 == 0 or len(targets) == len(selected):
                    f.flush()
                    save(out / 'status.json', dict(state='running', completed=len(targets), images=len(selected)))
                    print(f'[m99] {len(targets)}/{len(selected)}', flush=True)
                if len(targets) == len(selected):
                    break
    finally:
        hook.remove()
    if len(targets) != len(selected):
        raise ValueError('未完整覆盖指定样本')
    result = score(targets, predictions, spec['class_names'])
    result.update(author_rounding_top1_differences=ties, elapsed_s=time.monotonic()-started,
                  predictions_sha256=sha(out / 'predictions.jsonl'))
    save(out / 'status.json', dict(state='complete', images=len(targets)))
    return result


def main():
    spec = json.loads((HERE / 'models_m99.json').read_text())
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--describe', action='store_true')
    parser.add_argument('--preflight', action='store_true')
    parser.add_argument('--max-images', type=int, default=0)
    parser.add_argument('--output-root', type=Path, default=Path('__MODEL_ROOT__/resources/runtime/assets/workspace/results/m99_author_source_v1'))
    args = parser.parse_args()
    if args.describe:
        print(json.dumps(spec, ensure_ascii=False, indent=2)); return
    if args.max_images not in (0, 10):
        parser.error('MAX_IMAGES仅支持0（9165全量）或10（冒烟）')
    model, processor, manifest, versions = prepare(spec)
    if args.preflight:
        return
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA不可用，不自动改为CPU推理')
    device = torch.device('cuda:0')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    model.to(device).eval()
    out = args.output_root / ('smoke' if args.max_images else 'full') / datetime.now().strftime('run_%Y%m%d_%H%M%S_%f')
    out.mkdir(parents=True)
    for name in FILES:
        shutil.copy2(HERE / name, out / name)
    for name in ['README.md', 'config.json', 'preprocessor_config.json']:
        shutil.copy2(Path(spec['model_path']) / name, out / ('model_' + name))
    protocol = dict(spec, versions=versions, cuda=torch.version.cuda, gpu=torch.cuda.get_device_name(0),
                    source_sha256={n: sha(out/n) for n in FILES}, full_source=not bool(args.max_images),
                    tf32=False, amp=False, official_exact_reproduction=False)
    save(out / 'protocol.json', protocol)
    save(out / 'data_manifest.json', manifest)
    save(out / 'status.json', dict(state='running'))
    try:
        smoke = sorted({next(r['row'] for r in manifest if r['target'] == c) for c in range(8)} | {1, 2})
        if len(smoke) != 10:
            raise ValueError('冒烟未包含10个唯一样本')
        evaluate(spec, model, processor, manifest, device, out / 'smoke_check', smoke)
        print('10张原函数冒烟通过（覆盖8类）。', flush=True)
        if not args.max_images:
            result = evaluate(spec, model, processor, manifest, device, out / 'evaluation', list(range(spec['images'])))
            result.update(model_number=99, dataset=spec['dataset_id'], split='train', independent_test='unverified',
                          fidelity=spec['fidelity'], official_reference=spec['official_reference'],
                          official_exact_reproduction=False)
            save(out / 'metrics.json', result)
            print(f"accuracy={result['accuracy_percent']:.4f}%，macro-F1={result['macro_f1']:.6f}", flush=True)
        save(out / 'status.json', dict(state='partial_complete' if args.max_images else 'complete', images=10 if args.max_images else spec['images']))
        print(f'结果: {out}', flush=True)
    except Exception as exc:
        save(out / 'status.json', dict(state='failed', error=f'{type(exc).__name__}: {exc}'))
        raise


if __name__ == '__main__':
    main()
