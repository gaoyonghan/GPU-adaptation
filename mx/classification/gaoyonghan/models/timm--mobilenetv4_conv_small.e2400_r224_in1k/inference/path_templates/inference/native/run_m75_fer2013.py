#!/usr/bin/env python3
"""M75 local weights on the original FER2013 PrivateTest (not FERPlus)."""
import argparse
from collections import Counter
import csv
from datetime import datetime
import hashlib
import json
from pathlib import Path
import shutil

FER_LABELS = ('angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral')
MODEL_LABELS = ('angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise')
MODEL = Path('__MODEL_ROOT__/weights/wasmdashai/vit-face-expression-v1')
DATA = Path('__DATASET_ROOT__/fer2013/fer2013.csv')
OUTPUT = Path('__MODEL_ROOT__/resources/runtime/assets/workspace/results/m75_fer2013_v1')
EXPECTED_SPLITS = {'Training': 28709, 'PublicTest': 3589, 'PrivateTest': 3589}


def sha256(path):
    with path.open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def save_json(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n')


def parse_pixels(value):
    import numpy as np
    tokens = value.split()
    if len(tokens) != 2304 or any(not t.isascii() or not t.isdecimal() for t in tokens):
        raise ValueError('pixels必须为2304个十进制整数')
    values = [int(t) for t in tokens]
    if min(values) < 0 or max(values) > 255:
        raise ValueError('pixels超出0—255范围')
    return np.asarray(values, dtype=np.uint8).reshape(48, 48)


def read_test(path):
    if not path.is_file():
        raise FileNotFoundError(f'缺少FER2013原始CSV: {path}；见README_m75.md，或设置FER_CSV。')
    splits, classes, samples = Counter(), Counter(), []
    with path.open(newline='', encoding='utf-8-sig') as handle:
        reader = csv.DictReader(handle)
        if set(reader.fieldnames or []) != {'emotion', 'pixels', 'Usage'}:
            raise ValueError('需要含emotion,pixels,Usage的原始fer2013.csv，不能使用FERPlus或无划分镜像')
        for index, row in enumerate(reader):
            try:
                if set(row) != {'emotion', 'pixels', 'Usage'}:
                    raise ValueError('CSV列数异常')
                split, label = row['Usage'], int(row['emotion'])
                if split not in EXPECTED_SPLITS or not 0 <= label < 7:
                    raise ValueError('划分或类别超出原始协议')
                pixels = parse_pixels(row['pixels'])
            except (TypeError, ValueError) as exc:
                raise ValueError(f'CSV第{index + 2}行无效: {exc}') from exc
            splits[split] += 1
            classes[label] += 1
            if split == 'PrivateTest':
                samples.append((index, label, pixels))
    if dict(splits) != EXPECTED_SPLITS:
        raise ValueError(f'原始划分数量不符: {dict(splits)}，预期{EXPECTED_SPLITS}')
    if [classes[i] for i in range(7)] != [4953, 547, 5121, 8989, 6077, 4002, 6198]:
        raise ValueError('全数据类别数量与FER2013发布论文不符；请核查原始文件')
    test_counts = Counter(label for _, label, _ in samples)
    if set(test_counts) != set(range(7)):
        raise ValueError('PrivateTest未覆盖全部七类')
    return samples, {FER_LABELS[i]: test_counts[i] for i in range(7)}


def score(targets, predictions):
    from sklearn.metrics import accuracy_score, confusion_matrix
    if not targets or len(targets) != len(predictions):
        raise ValueError('预测未完整覆盖评测样本')
    if any(x not in range(7) for x in targets + predictions):
        raise ValueError('标签超出FER2013七类')
    correct = sum(a == b for a, b in zip(targets, predictions))
    accuracy = float(accuracy_score(targets, predictions))
    if abs(accuracy - correct / len(targets)) > 1e-12:
        raise ValueError('准确率计分交叉检查失败')
    return {'accuracy': accuracy, 'accuracy_percent': accuracy * 100,
            'correct': correct, 'evaluation_images': len(targets),
            'confusion_matrix_label_order': FER_LABELS,
            'confusion_matrix': confusion_matrix(targets, predictions, labels=list(range(7))).tolist()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--csv', type=Path, default=DATA)
    parser.add_argument('--output-root', type=Path, default=OUTPUT)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--max-images', type=int, default=0)
    parser.add_argument('--preflight', action='store_true')
    args = parser.parse_args()
    if args.max_images < 0:
        parser.error('--max-images不能小于0')
    import importlib.metadata
    import torch
    from PIL import Image
    from transformers import AutoImageProcessor, AutoModelForImageClassification

    expected = {'torch': '2.5.1+cu121', 'transformers': '4.50.0', 'numpy': '1.26.4',
                'Pillow': '11.1.0', 'scikit-learn': '1.5.2', 'safetensors': '0.5.3'}
    versions = {name: importlib.metadata.version(name) for name in expected}
    if versions != expected:
        raise RuntimeError(f'请复用classification8_eval已验证环境，不修改共享依赖: {versions}')
    config = json.loads((MODEL / 'config.json').read_text())
    if tuple(config['id2label'][str(i)] for i in range(7)) != MODEL_LABELS:
        raise ValueError('75号输出类别与已核查配置不符')
    hashes = {name: sha256(MODEL / name) for name in
              ('model.safetensors', 'config.json', 'preprocessor_config.json', 'README.md')}
    csv_hash = sha256(args.csv) if args.csv.is_file() else None
    samples, counts = read_test(args.csv)
    if sha256(args.csv) != csv_hash:
        raise ValueError('读取期间CSV发生变化')
    processor = AutoImageProcessor.from_pretrained(MODEL, local_files_only=True, use_fast=False)
    batch_check = processor(images=[Image.fromarray(samples[0][2]).convert('RGB')], return_tensors='pt')
    if batch_check['pixel_values'].shape != (1, 3, 224, 224):
        raise ValueError('发布的预处理器输出尺寸异常')
    print(json.dumps({'status': 'ready', 'split': 'PrivateTest', 'images': len(samples),
                      'class_counts': counts, 'metric': 'top1_accuracy'}, ensure_ascii=False), flush=True)
    if args.preflight:
        return
    if args.device.startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError('CUDA不可用；未启动推理')
    if args.max_images:
        samples = samples[:args.max_images]
    full = len(samples) == 3589
    out = args.output_root / ('full' if full else 'smoke') / datetime.now().strftime('run_%Y%m%d_%H%M%S_%f')
    out.mkdir(parents=True, exist_ok=False)
    for name in ('run_m75_fer2013.py', 'run_m75_fer2013.sh', 'README_m75.md'):
        shutil.copy2(Path(__file__).with_name(name), out / name)
    protocol = {
        'model_number': 75, 'model_id': 'wasmdashai/vit-face-expression-v1',
        'model_path': str(MODEL), 'model_files_sha256': hashes,
        'dataset': 'FER2013 original labels', 'csv_path': str(args.csv.resolve()), 'csv_sha256': csv_hash,
        'dataset_origin': 'https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/data',
        'dataset_source': ('https://www.kaggle.com/datasets/deadskull7/fer2013' if csv_hash ==
                           '3b8d9617d1017f34733c8f2474d7784c563ce86c40a86ac12c2d37cc968f871b' else None),
        'download_provenance': 'Known third-party copy identified by local SHA256; original competition file byte equivalence unverified',
        'protocol_source': 'https://arxiv.org/html/1307.0414',
        'dataset_relation_to_model': 'public_replacement_author_test_unverified',
        'split': 'PrivateTest', 'original_split_counts': EXPECTED_SPLITS,
        'full_test_class_counts': counts, 'selected_images': len(samples),
        'fer_id_to_name': FER_LABELS, 'model_id_to_name': MODEL_LABELS,
        'inference': 'local AutoImageProcessor + AutoModelForImageClassification, RGB, FP32, batch16, argmax; no TTA/crop',
        'metric': 'Top-1 accuracy = correct / total; sklearn.metrics.accuracy_score',
        'code_boundary': 'Our CSV/batch/provenance wrapper; published Transformers inference; library scorer implementing dataset accuracy, not Kaggle server source',
        'official_model_reference': None,
        'upstream_hint_only': {'id': config.get('_name_or_path'), 'test_accuracy': 0.7116,
                               'weight_equivalence_verified': False, 'directly_comparable': False},
        'training_test_overlap': 'unknown for this checkpoint; no training images evaluated',
        'versions': versions, 'device': args.device,
        'source_sha256': {name: sha256(out / name) for name in
                          ('run_m75_fer2013.py', 'run_m75_fer2013.sh', 'README_m75.md')},
    }
    save_json(out / 'protocol.json', protocol)
    save_json(out / 'status.json', {'state': 'running', 'completed_images': 0})
    targets, predictions = [], []
    try:
        model = AutoModelForImageClassification.from_pretrained(
            MODEL, local_files_only=True, use_safetensors=True, torch_dtype=torch.float32,
            attn_implementation='eager').to(args.device).eval()
        with (out / 'predictions.csv').open('w', newline='') as handle:
            writer = csv.writer(handle)
            writer.writerow(['source_dataset', 'source_split', 'source_csv_row_0based', 'image_id',
                             'pixels_sha256', 'target_fer_id', 'prediction_fer_id', 'target_name', 'prediction_name'])
            with torch.inference_mode():
                for start in range(0, len(samples), 16):
                    batch = samples[start:start + 16]
                    inputs = processor(images=[Image.fromarray(p).convert('RGB') for _, _, p in batch], return_tensors='pt')
                    logits = model(**{k: v.to(args.device) for k, v in inputs.items()}).logits
                    if logits.shape != (len(batch), 7) or not torch.isfinite(logits).all():
                        raise ValueError('模型输出维度异常或含非有限值')
                    for (index, target, pixels), pred in zip(batch, logits.argmax(-1).cpu().tolist()):
                        predicted = FER_LABELS.index(MODEL_LABELS[pred])
                        targets.append(target)
                        predictions.append(predicted)
                        writer.writerow(['FER2013', 'PrivateTest', index, f'fer{index:07d}',
                                         hashlib.sha256(pixels.tobytes()).hexdigest(), target, predicted,
                                         FER_LABELS[target], FER_LABELS[predicted]])
                    handle.flush()
                    save_json(out / 'status.json', {'state': 'running', 'completed_images': len(targets)})
                    print(f'[m75] {len(targets)}/{len(samples)}', flush=True)
        if len(targets) != len(samples):
            raise ValueError('评测覆盖不完整')
        result = score(targets, predictions)
        result.update({'state': 'complete' if full else 'partial_complete',
                       'dataset_relation': protocol['dataset_relation_to_model'],
                       'official_model_reference': None, 'predictions_sha256': sha256(out / 'predictions.csv')})
        save_json(out / ('metrics.json' if full else 'smoke_metrics.json'), result)
        save_json(out / 'status.json', {'state': result['state'], 'completed_images': len(targets)})
        print(json.dumps(result, ensure_ascii=False), flush=True)
        print(f'输出: {out}', flush=True)
    except Exception as exc:
        save_json(out / 'status.json', {'state': 'failed', 'completed_images': len(targets), 'error': str(exc)})
        raise


if __name__ == '__main__':
    main()
