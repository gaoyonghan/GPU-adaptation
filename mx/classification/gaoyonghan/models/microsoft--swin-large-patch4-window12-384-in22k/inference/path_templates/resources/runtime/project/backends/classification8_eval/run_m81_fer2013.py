#!/usr/bin/env python3
"""M81 on FER2013 PrivateTest; provisional label semantics require explicit opt-in."""
import argparse
from datetime import datetime
import hashlib
import importlib.metadata
import json
from pathlib import Path
import shutil

from run_m75_fer2013 import FER_LABELS, read_test, save_json, score, sha256

HERE = Path(__file__).resolve().parent
SOURCES = ('run_m81_fer2013.py', 'run_m81_fer2013.sh', 'models_m81.json',
           'README_m81.md', 'run_m75_fer2013.py')


def map_predictions(indices, mapping):
    if sorted(mapping) != list(range(7)) or any(i not in range(7) for i in indices):
        raise ValueError('模型输出或候选映射不是七分类有效编号')
    return [mapping[i] for i in indices]


def main():
    spec = json.loads((HERE / 'models_m81.json').read_text())
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--csv', type=Path, default=Path(spec['csv_path']))
    parser.add_argument('--output-root', type=Path,
                        default=Path('__MODEL_ROOT__/resources/runtime/assets/workspace/results/m81_fer2013_v1'))
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--max-images', type=int, default=0)
    parser.add_argument('--preflight', action='store_true')
    parser.add_argument('--describe', action='store_true')
    parser.add_argument('--allow-unverified-labels', action='store_true')
    args = parser.parse_args()
    if args.describe:
        print(json.dumps(spec, ensure_ascii=False, indent=2))
        return
    if args.max_images < 0:
        parser.error('--max-images不能小于0')
    if not args.preflight and not args.allow_unverified_labels:
        parser.error('类别映射尚未确认；试跑须显式加--allow-unverified-labels，结果不作为正式指标')
    import torch
    from PIL import Image
    from transformers import BeitImageProcessor, BeitForImageClassification

    versions = {name: importlib.metadata.version(name) for name in spec['versions']}
    if versions != spec['versions']:
        raise RuntimeError(f'共享环境版本变化，请先核查；不自动改动环境: {versions}')
    model_path = Path(spec['model_path'])
    for name, expected in spec['files_sha256'].items():
        if sha256(model_path / name) != expected:
            raise ValueError(f'本地模型文件变化: {name}')
    if sha256(args.csv) != spec['csv_sha256']:
        raise ValueError('FER2013 CSV与已审核副本SHA256不符')
    samples, counts = read_test(args.csv)
    if sha256(args.csv) != spec['csv_sha256']:
        raise ValueError('读取期间CSV发生变化')
    config = json.loads((model_path / 'config.json').read_text())
    if config['id2label'] != {str(i): f'LABEL_{i}' for i in range(7)}:
        raise ValueError('配置类别发生变化，应重新核查映射')
    processor = BeitImageProcessor.from_pretrained(model_path, local_files_only=True)
    check = processor(images=Image.fromarray(samples[0][2]).convert('RGB'), return_tensors='pt')
    if check['pixel_values'].shape != (1, 3, 224, 224):
        raise ValueError('预处理尺寸异常')
    print(json.dumps({'status': 'data_ready_mapping_unverified', 'split': 'PrivateTest',
                      'images': len(samples), 'class_counts': counts,
                      'metric': 'provisional_top1_accuracy', 'accepted_for_report': False},
                     ensure_ascii=False), flush=True)
    if args.preflight:
        return
    if args.device.startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError('CUDA不可用；未启动推理')
    if args.max_images:
        samples = samples[:args.max_images]
    full = len(samples) == spec['images']
    out = args.output_root / ('full' if full else 'smoke') / datetime.now().strftime('run_%Y%m%d_%H%M%S_%f')
    out.mkdir(parents=True, exist_ok=False)
    for name in SOURCES:
        shutil.copy2(HERE / name, out / name)
    for name in spec['files_sha256']:
        if name != 'model.safetensors':
            shutil.copy2(model_path / name, out / name)
    protocol = dict(spec, csv_path=str(args.csv.resolve()), device=args.device,
                    source_sha256={name: sha256(out / name) for name in SOURCES},
                    unverified_labels_explicitly_allowed=True)
    fingerprint = hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest()
    protocol.update(protocol_fingerprint=fingerprint, selected_images=len(samples),
                    full_test_class_counts=counts)
    save_json(out / 'protocol.json', protocol)
    targets, predictions = [], []

    def status(state, **extra):
        save_json(out / 'status.json', dict(state=state, completed_images=len(targets),
                  selected_images=len(samples), protocol_fingerprint=fingerprint,
                  label_mapping_status=spec['label_mapping_status'], accepted_for_report=False, **extra))

    status('running')
    print('[m81] 警告：候选类别映射未确认，本次仅为临时评测。', flush=True)
    try:
        model, loading = BeitForImageClassification.from_pretrained(
            model_path, local_files_only=True, use_safetensors=True, torch_dtype=torch.float32,
            attn_implementation='eager', output_loading_info=True)
        if any(loading.values()):
            raise RuntimeError(f'模型权重未严格匹配: {loading}')
        model = model.to(args.device).eval()
        with (out / 'predictions.jsonl').open('w') as handle, torch.inference_mode():
            for start in range(0, len(samples), 8):
                batch = samples[start:start + 8]
                inputs = processor(images=[Image.fromarray(p).convert('RGB') for _, _, p in batch],
                                   return_tensors='pt')
                logits = model(**{k: v.to(args.device) for k, v in inputs.items()}).logits
                if logits.shape != (len(batch), 7) or not torch.isfinite(logits).all():
                    raise ValueError('模型输出形状异常或非有限')
                indices = logits.argmax(-1).cpu().tolist()
                mapped = map_predictions(indices, spec['candidate_model_to_fer'])
                for (index, target, pixels), raw, pred, values in zip(batch, indices, mapped, logits.cpu().tolist()):
                    record = dict(source_dataset='FER2013', source_split='PrivateTest',
                                  source_csv_row_0based=index, image_id=f'fer{index:07d}',
                                  pixels_sha256=hashlib.sha256(pixels.tobytes()).hexdigest(),
                                  target_fer_id=target, raw_prediction_model_id=raw, logits=values,
                                  provisional_prediction_fer_id=pred, target_name=FER_LABELS[target],
                                  provisional_prediction_name=FER_LABELS[pred])
                    handle.write(json.dumps(record, ensure_ascii=False) + '\n')
                    targets.append(target)
                    predictions.append(pred)
                handle.flush()
                status('running')
                print(f'[m81] {len(targets)}/{len(samples)}', flush=True)
        if len(targets) != len(samples):
            raise ValueError('未完整覆盖评测样本')
        state = 'provisional_complete' if full else 'smoke_complete'
        if full:
            result = dict(state=state, accepted_for_report=False,
                          label_mapping_status=spec['label_mapping_status'],
                          provisional_metrics=score(targets, predictions),
                          official_reference_comparable=False, protocol_fingerprint=fingerprint,
                          predictions_sha256=sha256(out / 'predictions.jsonl'))
            save_json(out / 'provisional_metrics.json', result)
            print(json.dumps(result, ensure_ascii=False), flush=True)
        status(state, predictions_sha256=sha256(out / 'predictions.jsonl'))
        print(f'输出: {out}', flush=True)
    except Exception as exc:
        status('failed', error=str(exc))
        raise


if __name__ == '__main__':
    main()
