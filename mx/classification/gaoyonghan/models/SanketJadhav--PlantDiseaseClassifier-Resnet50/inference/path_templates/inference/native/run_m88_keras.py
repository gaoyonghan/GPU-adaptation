"""M88: native Keras weights/model/metrics + Google's unchanged eval preprocessing.

Project-owned code only coordinates assets, ImageNet labels, batching and reports.
This is not a claim that Keras's historical benchmark environment was reproduced.
"""
import argparse
import csv
import hashlib
import importlib.metadata
import importlib.util
import io
import json
import os
from pathlib import Path
import shutil
import time
import zipfile
import zlib

HERE = Path(__file__).resolve().parent
IMAGE_ROOT = Path('__MODEL_ROOT__/resources/runtime/assets/workspace/classification8_eval/.runtime/data/imagenet/val')
ARCHIVE = Path('__DATASET_ROOT__/ImageNet/imagenet-object-localization-challenge.zip')
OLD_RESULT = '__MODEL_ROOT__/resources/runtime/assets/workspace/results/classification8_v1/m88-efficientnet-b3'
VERSIONS = {'tensorflow': '2.15.1', 'keras': '2.15.0', 'numpy': '1.26.4', 'h5py': '3.16.0'}
CHECKSUMS = {
    'preprocessing.py': ('sha256', '8bd9f09079a59af79bb3533892aa14feae9539d7aee121945f5a33dcb3395f9b'),
    'imagenet_class_index.json': ('sha256', 'a1e7a966a1f601d39e4b43e119b3e7dd4a2ad3ea08cf69847cbaf021013767bc'),
    'models/efficientnetb3.h5': ('md5', 'ffd1fdc53d0ce67064dc6a9c7960ede0'),
}


def digest(path, algorithm='sha256'):
    h = hashlib.new(algorithm)
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def save(path, value):
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def environment():
    actual = {name: importlib.metadata.version(name) for name in VERSIONS}
    if actual != VERSIONS:
        raise RuntimeError(f'环境版本不符，不自动修改共享环境：{actual}; expected={VERSIONS}')
    return actual


def ground_truth(raw, classes):
    indices = {name: i for i, name in enumerate(classes)}
    annotations = {}
    for entry in csv.DictReader(io.StringIO(raw.decode())):
        tokens = entry['PredictionString'].split()
        if not tokens or len(tokens) % 5:
            raise ValueError('Invalid ImageNet annotation')
        synsets = set(tokens[::5])
        image_id = entry['ImageId']
        if len(synsets) != 1 or image_id in annotations:
            raise ValueError('Ambiguous/duplicate ImageNet annotation')
        annotations[image_id] = indices[synsets.pop()]
    return annotations


def dataset(assets):
    index = json.loads((assets / 'imagenet_class_index.json').read_text())
    classes = [index[str(i)][0] for i in range(1000)]
    assert len(set(classes)) == 1000 and classes == sorted(classes)
    with zipfile.ZipFile(ARCHIVE) as archive:
        raw = archive.read('LOC_val_solution.csv')
        annotations = ground_truth(raw, classes)
        members = sorted((x for x in archive.infolist()
                          if '/Data/CLS-LOC/val/' in x.filename and x.filename.endswith('.JPEG')),
                         key=lambda x: x.filename)
    assert len(annotations) == len(members) == 50000
    assert {Path(x.filename).stem for x in members} == set(annotations)
    counts = [0] * 1000
    rows = []
    fingerprint = hashlib.sha256(raw)
    for n, member in enumerate(members, 1):
        image_id = Path(member.filename).stem
        target = annotations[image_id]
        path = IMAGE_ROOT / classes[target] / Path(member.filename).name
        # Check existing extracted images against the original archive, without re-extracting it.
        data = path.read_bytes()
        if len(data) != member.file_size or zlib.crc32(data) != member.CRC:
            raise RuntimeError(f'Image differs from original archive: {path}')
        sha = hashlib.sha256(data).hexdigest()
        fingerprint.update(f'{image_id}\t{target}\t{sha}\n'.encode())
        rows.append({'id': image_id, 'path': str(path), 'target': target, 'sha256': sha})
        counts[target] += 1
        if n % 5000 == 0:
            print(f'[数据核查] {n}/50000', flush=True)
    assert counts == [50] * 1000
    return rows, {'images': len(rows), 'classes': 1000, 'split': 'ILSVRC2012 val',
                  'relation': 'original_exact', 'image_root': str(IMAGE_ROOT),
                  'annotation_sha256': hashlib.sha256(raw).hexdigest(),
                  'content_fingerprint': fingerprint.hexdigest()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--assets', type=Path)
    parser.add_argument('--output-root', type=Path)
    parser.add_argument('--max-images', type=int, default=0)
    parser.add_argument('--preflight', action='store_true')
    parser.add_argument('--check-environment', action='store_true')
    args = parser.parse_args()
    versions = environment()
    if args.check_environment:
        print(json.dumps(versions)); return
    if args.assets is None or args.output_root is None or not 0 <= args.max_images <= 50000:
        parser.error('assets/output-root are required; max-images must be 0..50000')
    if Path(os.environ.get('KERAS_HOME', '')).resolve() != args.assets.resolve():
        raise RuntimeError('请通过run_m88_keras.sh运行，以固定Keras权重缓存与GPU库路径')
    for name, (algorithm, expected) in CHECKSUMS.items():
        if digest(args.assets / name, algorithm) != expected:
            raise RuntimeError(f'Asset hash mismatch: {name}')
    rows, data_info = dataset(args.assets)

    import numpy as np
    import tensorflow as tf
    gpus = tf.config.list_physical_devices('GPU')
    if not gpus:
        raise RuntimeError('TensorFlow未识别GPU，停止；不静默退回CPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    tf.config.experimental.enable_tensor_float_32_execution(False)
    tf.keras.mixed_precision.set_global_policy('float32')
    tf.keras.utils.set_random_seed(88)
    source = importlib.util.spec_from_file_location('m88_google_preprocessing', args.assets / 'preprocessing.py')
    preprocessing = importlib.util.module_from_spec(source)
    source.loader.exec_module(preprocessing)
    # Keep weights='imagenet': Keras inserts an additional Rescaling layer for it.
    # weights=None + load_weights(path) would NOT construct the same preprocessing graph.
    model = tf.keras.applications.EfficientNetB3(weights='imagenet', include_top=True,
                                               classifier_activation=None)
    model.trainable = False
    assert model.input_shape == (None, 300, 300, 3) and model.output_shape == (None, 1000)
    keras_source = Path(importlib.util.find_spec('keras.src.applications.efficientnet').origin)
    protocol = {'sequence': 88, 'model_id': 'google/efficientnet-b3',
                'native_model': 'tf.keras.applications.EfficientNetB3(weights=imagenet)',
                'dataset': data_info, 'runtime_versions': versions,
                'keras_source_sha256': digest(keras_source),
                'weight_sha256': digest(args.assets / 'models/efficientnetb3.h5'),
                'asset_checksums': CHECKSUMS, 'batch_size': 16, 'dtype': 'float32',
                'preprocessing': 'Google TPU EfficientNet preprocess_for_eval, image_size=300, resize_method=None; no external normalization',
                'crop_fraction_constant': preprocessing.CROP_FRACTION,
                'gpu': tf.config.experimental.get_device_details(gpus[0]).get('device_name'),
                'old_hf_result_preserved': OLD_RESULT,
                'implementation': 'native Keras model/metrics + unchanged Google preprocessing; project data/batch/report wrapper',
                'reference': {'source': 'https://keras.io/api/applications/', 'top1_percent': 81.6, 'top5_percent': 95.7},
                'official_reproduction_verified': False,
                'caveats': ['Keras 2.15 runtime, not the conversion script cited Keras 2.11 runtime.',
                            'Keras benchmark historical preprocessing not completely documented; Google upstream eval preprocessing selected.',
                            'Native/HF weight tensor equality not yet independently verified; no attribution of score changes to a single cause.']}
    print(json.dumps({'status': 'ready', 'dataset': data_info, 'gpu': protocol['gpu']}, ensure_ascii=False))
    if args.preflight:
        print('预检查完成：未调用模型推理。'); return

    selected = rows[:args.max_images] if args.max_images else rows
    mode = 'smoke' if args.max_images else 'full'
    run = args.output_root / mode / f'run_{time.strftime("%Y%m%d_%H%M%S")}_{time.time_ns() % 1000000000}'
    run.mkdir(parents=True, exist_ok=False)
    for name in ['run_m88_keras.py', 'run_m88_keras.sh', 'README_m88.md']:
        shutil.copy2(HERE / name, run / name)
    save(run / 'protocol.json', protocol)
    status = {'state': 'running', 'requested_images': args.max_images, 'total_images': len(selected), 'completed_images': 0}
    save(run / 'status.json', status)
    start = time.monotonic()
    top1 = tf.keras.metrics.SparseCategoricalAccuracy(dtype='float32')
    top5 = tf.keras.metrics.SparseTopKCategoricalAccuracy(k=5, dtype='float32')
    correct1 = correct5 = 0

    @tf.function(input_signature=[tf.TensorSpec([None, 300, 300, 3], tf.float32)])
    def infer(images):
        return model(images, training=False)

    def load(path, target):
        with tf.device('/CPU:0'):
            image = preprocessing.preprocess_for_eval(tf.io.read_file(path), False, image_size=300)
        return image, target

    try:
        with (run / 'predictions.jsonl').open('x') as output:
            # First ten images are the smoke check, retained in the full result (not run twice).
            for group in (selected[:10], selected[10:]):
                if not group:
                    continue
                ds = tf.data.Dataset.from_tensor_slices(([r['path'] for r in group], [r['target'] for r in group]))
                options = tf.data.Options()
                options.threading.private_threadpool_size = 2
                options.experimental_deterministic = True
                ds = ds.with_options(options).map(load, num_parallel_calls=2).batch(16).prefetch(1)
                offset = 0
                for images, targets in ds:
                    scores = infer(images)
                    tf.debugging.assert_all_finite(scores, 'Non-finite native output')
                    top1.update_state(targets, scores); top5.update_state(targets, scores)
                    indices = tf.math.top_k(scores, k=5).indices.numpy()
                    # TensorFlow's in_top_k includes ties at the kth boundary; use its same definition.
                    hit5 = tf.math.in_top_k(targets, scores, 5).numpy()
                    pred = tf.argmax(scores, axis=1).numpy()
                    truth = targets.numpy()
                    correct1 += int(np.sum(pred == truth)); correct5 += int(np.sum(hit5))
                    for j in range(len(truth)):
                        row = group[offset + j]
                        output.write(json.dumps({'id': row['id'], 'target': int(truth[j]), 'prediction': int(pred[j]),
                                                 'top5': indices[j].tolist(), 'top5_correct': bool(hit5[j]),
                                                 'image_sha256': row['sha256']}) + '\n')
                    offset += len(truth)
                    status['completed_images'] += len(truth)
                    output.flush()
                    if status['completed_images'] == 10 or offset % 256 == 0 or offset == len(group):
                        elapsed = time.monotonic() - start
                        print(f"[88 Keras] {status['completed_images']}/{len(selected)}, {status['completed_images']/elapsed:.2f} images/s", flush=True)
                        save(run / 'status.json', status)
        n = status['completed_images']
        assert n == len(selected)
        assert abs(float(top1.result()) - correct1 / n) < 1e-7
        assert abs(float(top5.result()) - correct5 / n) < 1e-7
        metrics = {'evaluation_images': n, 'top1_percent': 100 * correct1 / n,
                   'top5_percent': 100 * correct5 / n, 'correct_top1': correct1, 'correct_top5': correct5,
                   'keras_metric_raw_percent': {'top1': 100 * float(top1.result()), 'top5': 100 * float(top5.result())},
                   'predictions_sha256': digest(run / 'predictions.jsonl'), 'official_reproduction_verified': False}
        if not args.max_images:
            metrics['difference_pp_vs_keras_reference'] = {'top1': metrics['top1_percent'] - 81.6, 'top5': metrics['top5_percent'] - 95.7}
            save(run / 'metrics.json', metrics)
        status.update(state='partial_complete' if args.max_images else 'complete', elapsed_seconds=time.monotonic() - start)
        save(run / 'status.json', status)
        print(f'完成，输出：{run}')
        if not args.max_images:
            print(json.dumps(metrics, ensure_ascii=False, indent=2))
    except BaseException as exc:
        status.update(state='interrupted' if isinstance(exc, KeyboardInterrupt) else 'failed', error=f'{type(exc).__name__}: {exc}')
        save(run / 'status.json', status)
        raise


if __name__ == '__main__':
    main()
