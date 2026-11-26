#!/usr/bin/env python3
"""Original HF weights, official-val filename intersection; NOT full official reproduction."""
import argparse
import collections
import datetime
import hashlib
import importlib.metadata
import inspect
import io
import json
from pathlib import Path
import shutil
import time
import urllib.request
import zipfile

HERE = Path(__file__).resolve().parent
ZIP = Path('__DATASET_ROOT__/imagenet21k_kaggle/fqx-21k-comp-tars.zip')
MODEL_ROOT = Path('__MODEL_ROOT__/weights/microsoft')
VERSIONS = {'torch': '2.5.1+cu121', 'torchvision': '0.20.1+cu121',
            'transformers': '4.50.0', 'timm': '1.0.15', 'Pillow': '11.1.0',
            'numpy': '1.26.4', 'safetensors': '0.5.3'}
MODELS = {
    '85': ('swin-large-patch4-window12-384-in22k', 384, [
        'f2162687c1e8c50138c5c4cffe19ab0bdbe94497017a8c32945ce749fdd6831d',
        '32f157bd0d9d7908f35627660baa46cd4953aa60ee6504bfbf1e6699fdf7e11c',
        '87286674463d7ca8acdd0dd468a6bf1ddd2a202be1961f10cf16b6c6eb1856f0']),
    '86': ('swinv2-large-patch4-window12-192-22k', 192, [
        'd2376131d369f0ac869aaff212cd9e5b748657186ba789172cc86f0991cbaaa3',
        'bef92c370fd5d809b2771985d7c4710cf6e88b1ac6fb9d02d4b6b46d54d7bb5b',
        '42f37ac46a4e945d87ec52d032acca27ee4b9d1ed9e1d3687f57b02cf4efc153'])}
ASSETS = {
    'official_val.json': (
        'https://github.com/SwinTransformer/storage/releases/download/v2.0.1/ILSVRC2011fall_whole_map_val.txt',
        '5f6c9fe77da598dfb855d23ca3c5fa1169d73e421d6bfc12351eca935c87e424'),
    'id2label.json': (
        'https://huggingface.co/datasets/huggingface/label-files/resolve/main/imagenet-22k-id2label.json',
        '905e5b378a422aedfd51c8fa1696f47ebe9830f78fb6b7722c43024fa195dcd2'),
    'convert_swin.py': (
        'https://raw.githubusercontent.com/huggingface/transformers/v4.29.2/src/transformers/models/swin/convert_swin_timm_to_pytorch.py',
        '4ee725a5b82eb53ec1224411817fa1805cd890674b6c1ee2cdd3955a944fe215'),
    'convert_swinv2.py': (
        'https://raw.githubusercontent.com/huggingface/transformers/v4.29.2/src/transformers/models/swinv2/convert_swinv2_timm_to_pytorch.py',
        'cd3b40ea26cd880c641770100ad7970558f96cf9b8f5ba578b91ce221fae4334')}


def sha(data):
    return hashlib.sha256(data).hexdigest()


def file_sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def save(path, obj):
    tmp = path.with_suffix(path.suffix + '.part')
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def sources(root):
    root.mkdir(parents=True, exist_ok=True)
    for name, (url, expected) in ASSETS.items():
        path = root / name
        if path.exists():
            if file_sha(path) != expected:
                raise ValueError(f'源码/清单缓存哈希不符，保留原文件：{path}')
            continue
        print(f'获取公开清单/参考代码（非图片）：{name}', flush=True)
        for attempt in range(3):
            try:
                with urllib.request.urlopen(url, timeout=60) as response:
                    payload = response.read()
                if sha(payload) != expected:
                    raise ValueError(f'上游版本改变或下载不完整：{name}')
                part = path.with_suffix(path.suffix + '.part')
                part.write_bytes(payload)
                part.replace(path)
                break
            except Exception as exc:
                print(f'{name}: 第{attempt + 1}/3次失败：{type(exc).__name__}', flush=True)
                if attempt == 2:
                    raise
                time.sleep(2)


def select(z, official):
    names = z.namelist()
    if len(names) != len(set(names)):
        raise ValueError('ZIP中有重复文件名')
    names = set(names)
    rows = []
    seen = set()
    for original, label in official:
        if not original.startswith('fall11_whole/') or not 0 <= int(label) < 21841:
            raise ValueError('官方清单路径/标签格式异常')
        if original in seen:
            raise ValueError('官方清单有重复图片')
        seen.add(original)
        member = original.replace('fall11_whole/', 'winter21_whole/', 1)
        if member in names:
            rows.append({'official_path': original, 'zip_member': member,
                         'synset': member.split('/')[1], 'target': int(label)})
    return rows


def prepare(run, assets):
    from PIL import Image
    official = json.loads((assets / 'official_val.json').read_text())
    if len(official) != 197596:
        raise ValueError('官方清单样本数不符')
    with zipfile.ZipFile(ZIP) as z:
        images = [x for x in z.infolist() if x.filename.lower().endswith('.jpeg')]
        archive_classes = {x.filename.split('/')[1] for x in images}
        rows = select(z, official)
        if (len(images), len(archive_classes), len(rows), len({x['target'] for x in rows})) != (132486, 129, 1192, 124):
            raise ValueError('下载包或交集范围改变；停止，须重新核查')
        print('核验交集图片：1192张、124类；保持21841类输出。', flush=True)
        for row in rows:
            payload = z.read(row['zip_member'])  # ZIP CRC checked for every selected image.
            with Image.open(io.BytesIO(payload)) as im:
                im.convert('RGB').load()
            row['sha256'] = sha(payload)
        central_sha = sha(json.dumps([(x.filename, x.file_size, x.CRC) for x in z.infolist()]).encode())
    save(run / 'manifest.json', rows)
    coverage = {'dataset': 'Kaggle qiexifan/fqx-21k-comp-tars / official Swin val filename intersection',
                'zip_path': str(ZIP), 'zip_bytes': ZIP.stat().st_size, 'central_directory_sha256': central_sha,
                'archive_images': len(images), 'archive_classes': len(archive_classes),
                'evaluation_images': len(rows), 'evaluation_classes': len({x['target'] for x in rows}),
                'official_val_images': len(official), 'official_val_classes': len({y for _, y in official}),
                'missing_official_val_images': len(official) - len(rows), 'output_classes': 21841,
                'classes_without_val_intersection': sorted(archive_classes - {x['synset'] for x in rows}),
                'per_class_images': dict(collections.Counter(x['synset'] for x in rows)),
                'selected_images_crc_and_decode': 'passed', 'official_image_byte_identity_verified': False,
                'training_overlap': 'unknown', 'officially_comparable': False}
    save(run / 'coverage.json', coverage)
    return rows


def check_model(number, rows, labels):
    name, size, hashes = MODELS[number]
    root = MODEL_ROOT / name
    print(f'[{number}] 校验本地权重及配置', flush=True)
    for filename, expected in zip(['pytorch_model.bin', 'config.json', 'preprocessor_config.json'], hashes):
        if file_sha(root / filename) != expected:
            raise ValueError(f'本地模型哈希改变：{root / filename}')
    config = json.loads((root / 'config.json').read_text())
    if len(config['id2label']) != 21841 or config['image_size'] != size:
        raise ValueError('分类头或输入尺寸配置不符')
    if number == '85' and any(config['id2label'].get(str(r['target'])) != labels[str(r['target'])] for r in rows):
        raise ValueError('85号交集标签与发布标签文件不符')
    if number == '86' and any(v != f'LABEL_{k}' for k, v in config['id2label'].items()):
        raise ValueError('86号标签配置改变，需重新核对索引顺序')
    return root, size


def evaluate(number, rows, root, size, run, limit):
    import torch
    from PIL import Image
    from timm.utils import accuracy
    from transformers import AutoImageProcessor, AutoModelForImageClassification
    out = run / ('m' + number)
    out.mkdir()
    save(out / 'status.json', {'state': 'loading', 'processed': 0})
    model = None
    try:
        processor = AutoImageProcessor.from_pretrained(root, local_files_only=True, use_fast=False)
        model, info = AutoModelForImageClassification.from_pretrained(
            root, local_files_only=True, output_loading_info=True, torch_dtype=torch.float32)
        if any(info.get(k) for k in ['missing_keys', 'unexpected_keys', 'mismatched_keys', 'error_msgs']):
            raise ValueError(f'非完整权重加载：{info}')
        if model.classifier.out_features != 21841:
            raise ValueError('模型不是21841类分类头')
        model = model.cuda().eval()
        chosen = rows[:limit] if limit else rows
        protocol = {'model_number': int(number), 'model_id': 'microsoft/' + root.name,
                    'weight_sha256': MODELS[number][2][0], 'versions': VERSIONS,
                    'input_size': size, 'batch_size': 2, 'dtype': 'float32', 'tf32': False,
                    'processor': processor.to_dict(), 'output_classes': 21841,
                    'selection': 'official val filename intersection; no output-class restriction',
                    'model_interface': 'publisher HF AutoModelForImageClassification',
                    'preprocessing': 'publisher HF processor, NOT Microsoft build_transform',
                    'metric': 'timm.utils.accuracy(topk=(1,5))',
                    'metric_source_sha256': sha(inspect.getsource(accuracy).encode()),
                    'official_entrypoint_unmodified': False, 'officially_comparable': False,
                    'scope': 'smoke' if limit else 'full_partial_dataset',
                    'gpu': torch.cuda.get_device_name(0), 'manifest_sha256': file_sha(run / 'manifest.json'),
                    'reference': 'No directly comparable official score for this partial 22K evaluation'}
        save(out / 'protocol.json', protocol)
        counts = [0, 0]
        native = [0.0, 0.0]
        start = time.monotonic()
        with zipfile.ZipFile(ZIP) as z, (out / 'predictions.jsonl').open('w') as f, torch.inference_mode():
            for offset in range(0, len(chosen), 2):
                batch = chosen[offset:offset + 2]
                images = []
                for r in batch:
                    payload = z.read(r['zip_member'])
                    if sha(payload) != r['sha256']:
                        raise ValueError('预检后图片内容发生变化')
                    with Image.open(io.BytesIO(payload)) as im:
                        images.append(im.convert('RGB'))
                inputs = processor(images=images, return_tensors='pt')
                if tuple(inputs['pixel_values'].shape[1:]) != (3, size, size):
                    raise ValueError('预处理输出尺寸不符')
                logits = model(**{k: v.cuda() for k, v in inputs.items()}).logits
                if logits.shape != (len(batch), 21841) or not torch.isfinite(logits).all():
                    raise ValueError('输出形状错误或出现NaN/Inf')
                targets = torch.tensor([r['target'] for r in batch], device='cuda')
                scores = accuracy(logits, targets, topk=(1, 5))
                top5 = logits.topk(5, dim=1).indices.cpu().tolist()
                for i in range(2):
                    native[i] += scores[i].item() * len(batch)
                for r, predicted in zip(batch, top5):
                    counts[0] += int(r['target'] == predicted[0])
                    counts[1] += int(r['target'] in predicted)
                    f.write(json.dumps(dict(r, prediction=predicted[0], top5=predicted)) + '\n')
                done = offset + len(batch)
                if done == 10 or done % 50 == 0 or done == len(chosen):
                    f.flush()
                    save(out / 'status.json', {'state': 'running', 'processed': done, 'total': len(chosen)})
                    print(f'[{number}] {done}/{len(chosen)}，{done / (time.monotonic()-start):.2f} images/s', flush=True)
        percentages = [100 * c / len(chosen) for c in counts]
        if any(abs(native[i] / len(chosen) - percentages[i]) > 1e-4 for i in range(2)):
            raise ValueError('原生指标与逐图预测复算不一致')
        result = dict(model_number=int(number), images=len(chosen), classes=len({r['target'] for r in chosen}),
                      top1_percent=native[0] / len(chosen), top5_percent=native[1] / len(chosen),
                      correct_top1=counts[0], correct_top5=counts[1], independently_recomputed_percent=percentages,
                      scope=protocol['scope'], officially_comparable=False, training_overlap='unknown',
                      predictions_sha256=file_sha(out / 'predictions.jsonl'), elapsed_s=time.monotonic()-start)
        save(out / ('smoke_metrics.json' if limit else 'metrics.json'), result)
        save(out / 'status.json', {'state': 'complete', 'processed': len(chosen), 'scope': protocol['scope']})
        print(f'[{number}] 完成：Top1={percentages[0]:.4f}%，Top5={percentages[1]:.4f}%；{out}', flush=True)
    except Exception as exc:
        save(out / 'status.json', {'state': 'failed', 'error': f'{type(exc).__name__}: {exc}'})
        raise
    finally:
        del model
        torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--model', choices=['both', '85', '86'], default='both')
    parser.add_argument('--max-images', type=int, default=0)
    parser.add_argument('--preflight', action='store_true')
    args = parser.parse_args()
    if args.max_images < 0 or args.max_images > 1192:
        parser.error('--max-images必须在0到1192之间；0表示完整交集')
    versions = {k: importlib.metadata.version(k) for k in VERSIONS}
    if versions != VERSIONS:
        raise RuntimeError(f'现有环境版本改变，停止而不自动安装：{versions}')
    import torch
    if not args.preflight and not torch.cuda.is_available():
        raise RuntimeError('未检测到CUDA，不回退CPU推理')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    run = args.output_root / ('preflight' if args.preflight else 'smoke' if args.max_images else 'full') / datetime.datetime.now().strftime('run_%Y%m%d_%H%M%S_%f')
    run.mkdir(parents=True)
    assets = args.output_root / 'assets'
    sources(assets)
    source = run / 'source'
    source.mkdir()
    for name in ['run_m85_m86_partial.py', 'run_m85_m86_partial.sh', 'README_m85_m86_partial.md']:
        shutil.copy2(HERE / name, source / name)
    for name in ASSETS:
        shutil.copy2(assets / name, source / name)
    rows = prepare(run, assets)
    labels = json.loads((assets / 'id2label.json').read_text())
    numbers = ['85', '86'] if args.model == 'both' else [args.model]
    checked = {n: check_model(n, rows, labels) for n in numbers}
    print(f'预检通过：1192张/124类；记录：{run}', flush=True)
    if args.preflight:
        return
    failures = []
    for number in numbers:
        try:
            evaluate(number, rows, *checked[number], run, args.max_images)
        except Exception as exc:
            print(f'[{number}] 失败：{type(exc).__name__}: {exc}', flush=True)
            failures.append(number)
    if failures:
        raise SystemExit(f'失败模型：{failures}；请查看各模型status.json及启动日志')


if __name__ == '__main__':
    main()
