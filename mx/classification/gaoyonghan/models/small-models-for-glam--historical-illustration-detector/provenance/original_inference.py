#!/usr/bin/env python3
"""M83: all published ImageIN rows; not a held-out evaluation."""
import argparse
from collections import Counter
from datetime import datetime
import hashlib
import importlib.metadata
import io
import json
from pathlib import Path
import shutil
import sys

HERE = Path(__file__).resolve().parent
SOURCES = ('run_m83_imagein.py', 'run_m83_imagein.sh', 'setup_m83_env.sh',
           'models_m83.json', 'README_m83.md')


def sha256(path):
    with path.open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def save_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def check_environment(spec):
    if sys.version_info[:2] != (3, 11):
        raise RuntimeError('请使用Python 3.11环境')
    actual = {name: importlib.metadata.version(name) for name in spec['versions']}
    if actual != spec['versions']:
        raise RuntimeError(f'环境版本与清单不符，请运行setup_m83_env.sh: {actual}')
    from optimum.onnxruntime import ORTModelForImageClassification
    return actual


def rows(spec, with_images=False):
    import pyarrow.parquet as pq
    columns = ['id', 'choice', 'image', 'manifest_url', 'annotation_id']
    if with_images:
        columns.append('loaded_image')
    for name in sorted(spec['dataset_files_sha256']):
        row_index = 0
        for batch in pq.ParquetFile(Path(spec['dataset_path']) / name).iter_batches(
                batch_size=1 if with_images else 128, columns=columns):
            for row in batch.to_pylist():
                row['parquet_file'], row['parquet_row_0based'] = name, row_index
                row_index += 1
                yield row


def validate_data(spec):
    for root, entries in ((spec['model_path'], spec['model_files_sha256']),
                          (spec['dataset_path'], spec['dataset_files_sha256'])):
        for name, expected in entries.items():
            if sha256(Path(root) / name) != expected:
                raise ValueError(f'文件哈希不符: {root}/{name}')
    config = json.loads((Path(spec['model_path']) / 'config.json').read_text())
    if config['id2label'] != {str(i): label for i, label in enumerate(spec['id_to_label'])}:
        raise ValueError('模型类别配置变化')
    ids, counts = set(), Counter()
    for row in rows(spec):
        if row['id'] in ids or row['choice'] not in spec['id_to_label']:
            raise ValueError(f'重复ID或未知choice: {row["id"]}')
        ids.add(row['id'])
        counts[row['choice']] += 1
    if len(ids) != spec['images'] or dict(counts) != spec['class_counts']:
        raise ValueError(f'数据数量不符: {len(ids)}, {counts}')
    return counts


def inputs_for_row(row, processor):
    from PIL import Image
    payload = row['loaded_image']['bytes']
    if not payload:
        raise ValueError(f'缺少内嵌图片: id={row["id"]}；不会联网抓取或跳过')
    with Image.open(io.BytesIO(payload)) as image:
        image.load()
        rgb = image.convert('RGB')
    try:
        inputs = processor(images=rgb, return_tensors='pt')
    finally:
        rgb.close()
    import torch
    values = inputs['pixel_values']
    if values.shape != (1, 3, 224, 224) or values.dtype != torch.float32 or not torch.isfinite(values).all():
        raise ValueError('预处理输出异常')
    return inputs, hashlib.sha256(payload).hexdigest()


def score(targets, predictions):
    from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
    if not targets or len(targets) != len(predictions) or any(x not in (0, 1) for x in targets + predictions):
        raise ValueError('计分输入无效')
    tn, fp, fn, tp = confusion_matrix(targets, predictions, labels=[0, 1]).ravel().tolist()
    accuracy = float(accuracy_score(targets, predictions))
    f1 = float(f1_score(targets, predictions, average='binary', pos_label=1, zero_division=0))
    denominator = 2 * tp + fp + fn
    if abs(accuracy - (tp + tn) / len(targets)) > 1e-12 or abs(f1 - (2 * tp / denominator if denominator else 0)) > 1e-12:
        raise ValueError('计分交叉检查失败')
    return dict(accuracy=accuracy, accuracy_percent=100 * accuracy, f1_illustrated=f1,
                f1_illustrated_percent=100 * f1, correct=tp + tn, images=len(targets),
                confusion_matrix=[[tn, fp], [fn, tp]],
                confusion_matrix_label_order=['not-illustrated', 'illustrated'])


def main():
    spec = json.loads((HERE / 'models_m83.json').read_text())
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-root', type=Path,
                        default=Path('/home/ubuntu/gaoyonghan/gpu/results/m83_imagein_source_v1'))
    parser.add_argument('--max-images', type=int, default=0)
    parser.add_argument('--describe', action='store_true')
    parser.add_argument('--check-environment', action='store_true')
    parser.add_argument('--preflight', action='store_true')
    args = parser.parse_args()
    if args.describe:
        print(json.dumps(spec, ensure_ascii=False, indent=2))
        return
    if args.max_images < 0:
        parser.error('--max-images不能小于0')
    check_environment(spec)
    if args.check_environment:
        print('M83 environment ready')
        return
    counts = validate_data(spec)
    import onnxruntime as ort
    import torch
    from optimum.onnxruntime import ORTModelForImageClassification
    from transformers import AutoConfig, AutoImageProcessor
    model_path = Path(spec['model_path'])
    processor = AutoImageProcessor.from_pretrained(model_path, local_files_only=True, use_fast=False)
    options = ort.SessionOptions()
    options.intra_op_num_threads, options.inter_op_num_threads = 2, 1
    model = ORTModelForImageClassification.from_pretrained(
        model_path, config=AutoConfig.from_pretrained(model_path, local_files_only=True),
        subfolder='onnx', file_name='model.onnx', local_files_only=True, export=False,
        provider='CPUExecutionProvider', session_options=options, use_io_binding=False)
    if model.session.get_providers() != ['CPUExecutionProvider']:
        raise RuntimeError('执行设备与冻结CPU协议不符')
    print(json.dumps(dict(status='ready', images=spec['images'], class_counts=counts,
                          scope='all_published_source_rows', independent_test=False), ensure_ascii=False), flush=True)
    if args.preflight:
        for index, row in enumerate(rows(spec, with_images=True), 1):
            inputs_for_row(row, processor)
            if index % 100 == 0:
                print(f'[m83 preflight] 图片解码/预处理 {index}/1896', flush=True)
        print('全部1896张图片预检通过；未进行推理。', flush=True)
        return
    limit = min(args.max_images or spec['images'], spec['images'])
    full = limit == spec['images']
    out = args.output_root / ('full' if full else 'smoke') / datetime.now().strftime('run_%Y%m%d_%H%M%S_%f')
    out.mkdir(parents=True, exist_ok=False)
    for name in SOURCES:
        shutil.copy2(HERE / name, out / name)
    for name in ('config.json', 'preprocessor_config.json', 'README.md'):
        shutil.copy2(model_path / name, out / ('model_' + name))
    protocol = dict(spec, source_sha256={name: sha256(out / name) for name in SOURCES})
    fingerprint = hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest()
    protocol.update(protocol_fingerprint=fingerprint, selected_images=limit)
    save_json(out / 'protocol.json', protocol)
    targets, predictions = [], []

    def status(state, **extra):
        save_json(out / 'status.json', dict(state=state, completed_images=len(targets),
                  selected_images=limit, protocol_fingerprint=fingerprint,
                  independent_test=False, official_reference_comparable=False, **extra))

    status('running')
    print('[m83] 作者数据来源全量评测，训练/测试重叠未知，不对比官方95.09%。', flush=True)
    try:
        with (out / 'predictions.jsonl').open('w') as handle, torch.inference_mode():
            for row in rows(spec, with_images=True):
                if len(targets) == limit:
                    break
                inputs, image_hash = inputs_for_row(row, processor)
                logits = model(**inputs).logits
                if logits.shape != (1, 2) or not torch.isfinite(logits).all():
                    raise ValueError('模型输出异常')
                pred = int(logits.argmax(-1).item())
                target = spec['id_to_label'].index(row['choice'])
                record = {key: row[key] for key in ('id', 'image', 'manifest_url', 'annotation_id',
                                                   'parquet_file', 'parquet_row_0based')}
                record.update(source_dataset=spec['dataset'], source_split='train', image_sha256=image_hash,
                              target_id=target, target_label=row['choice'], prediction_id=pred,
                              prediction_label=spec['id_to_label'][pred], logits=logits[0].tolist())
                handle.write(json.dumps(record, ensure_ascii=False) + '\n')
                handle.flush()
                targets.append(target)
                predictions.append(pred)
                if len(targets) == 10:
                    print('[m83] 10张冒烟测试通过，后续继续同一协议；不重复计算前10张。', flush=True)
                if len(targets) % 25 == 0 or len(targets) == limit:
                    status('running')
                    print(f'[m83] {len(targets)}/{limit}', flush=True)
        if len(targets) != limit:
            raise ValueError('推理样本覆盖不全')
        state = 'source_evaluation_complete' if full else 'smoke_complete'
        if full:
            result = dict(score(targets, predictions), state=state,
                          dataset_relation=spec['dataset_relation'], independent_test=False,
                          training_eval_overlap=spec['training_eval_overlap'],
                          official_reference_comparable=False, protocol_fingerprint=fingerprint,
                          predictions_sha256=sha256(out / 'predictions.jsonl'))
            save_json(out / 'metrics.json', result)
            print(json.dumps(result, ensure_ascii=False), flush=True)
        status(state, predictions_sha256=sha256(out / 'predictions.jsonl'))
        print(f'输出: {out}', flush=True)
    except Exception as exc:
        status('failed', error=str(exc))
        raise


if __name__ == '__main__':
    main()
