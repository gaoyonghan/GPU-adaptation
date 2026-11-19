#!/usr/bin/env python3
"""Run the pinned WildDet3D/Vis4D evaluator; never overwrite earlier results."""
import argparse
from datetime import datetime
from importlib.metadata import version
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys

from download_official_source import file_hash, verify_bundle

HERE = Path(__file__).resolve().parent
SOURCE_ROOT = Path('/data2/gaoyonghan/gpu_model_standalone_20261110/release/model/检测/iitolstykh--YOLO-Face-Person-Detector/resources/runtime/assets/workspace/official_sources/object_detection')
SOURCE = SOURCE_ROOT / 'WildDet3D'
DATA = Path('/home/ubuntu/文档/ChatGPT/idea 2/outputs/launch_audit_20261005/migration_inputs/datasets/wilddet3d_val')
WEIGHT = Path('/data2/gaoyonghan/gpu_model_standalone_20261110/release/model/检测/iitolstykh--YOLO-Face-Person-Detector/resources/model_index/allenai/WildDet3D/wilddet3d_alldata_all_prompt_v1.0.pt')
OUT = Path('/data2/gaoyonghan/gpu_model_standalone_20261110/release/model/检测/iitolstykh--YOLO-Face-Person-Detector/resources/runtime/assets/workspace/results/m27_native_text_v1')


def write(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False))


def preflight():
    source_records = verify_bundle(SOURCE_ROOT)
    annotation = DATA / 'annotations/InTheWild_v3_val.json'
    data = json.loads(annotation.read_text())
    ids = [image['id'] for image in data['images']]
    assert len(ids) == len(set(ids)) == 2470, 'Official validation must contain 2470 unique images'
    missing = [im['file_path'] for im in data['images'] if not (DATA / im['file_path']).is_file()]
    assert not missing, f'Missing images: {missing[:5]}'
    assert WEIGHT.is_file(), str(WEIGHT)
    annotation_sha = file_hash(annotation)
    checkpoint_sha = file_hash(WEIGHT)
    assert annotation_sha == '3480126cea309ea1b8724753fe574c5b6a077b8d0cf290b1b8e0f30872221370', 'Annotation changed'
    assert checkpoint_sha == 'f8b6a9e548f733ba62625a0d2adc4b0f4fdb6007ee11d9927f9c1027010fee57', 'Checkpoint changed'
    return data, {'source_bundle': source_records, 'annotation': str(annotation),
                  'annotation_sha256': annotation_sha,
                  'images': len(ids), 'raw_annotations': len(data['annotations']),
                  'categories': len(data['categories']), 'missing_images': len(missing),
                  'checkpoint': str(WEIGHT), 'checkpoint_sha256': checkpoint_sha,
                  'environment': {p: version(p) for p in ['torch', 'torchvision', 'vis4d', 'lightning', 'numpy']}}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--max-images', type=int, choices=[0, 10], default=10)
    parser.add_argument('--device', default='1')
    parser.add_argument('--preflight', action='store_true')
    args = parser.parse_args()
    data, audit = preflight()
    if args.preflight:
        print(json.dumps(audit, indent=2, ensure_ascii=False))
        return
    assert args.device.isdigit(), 'Use one physical GPU index, e.g. --device 1'
    run = OUT / ('smoke' if args.max_images else 'full') / datetime.now().strftime('run_%Y%m%d_%H%M%S_%f')
    work = run / 'workspace'
    annotations = work / 'data/in_the_wild/annotations'
    annotations.mkdir(parents=True)
    (annotations / 'InTheWild_v3_val.json').symlink_to(audit['annotation'])
    for name in ['images', 'camera', 'depth']:
        (annotations.parent / name).symlink_to(DATA / name, target_is_directory=True)
    # Official dataset/class-map caches are written here, not into the source dataset.
    for name in ['run_m27_native.py', 'run_m27_native.sh', 'm27_native_config.py', 'm27_lingbot_config.json']:
        shutil.copy2(HERE / name, run / name)
    ids = sorted(image['id'] for image in data['images'])  # Native COCO3DDataset order.
    write(run / 'expected_ids.json', ids[:args.max_images] if args.max_images else ids)
    audit.update({'mode': 'smoke' if args.max_images else 'full', 'official_config': 'configs/eval/in_the_wild/text.py',
                  'dataset': 'InTheWild v3 validation (official)', 'prompt': 'text', 'with_gt_depth': False,
                  'reference': {'AP_percent': 22.6, 'APr_percent': 28.3, 'APc_percent': 21.6, 'APf_percent': 18.7,
                                'url': 'https://arxiv.org/html/2604.08626v2', 'table': 3,
                                'row': 'WildDet3D (All Data), Text', 'mapping_status': 'candidate; verify native metric definition with result'},
                  'changes_from_native_entry': ['Construct components without auxiliary pretrained loads; strict-load complete author checkpoint',
                                                'Read exact LingBot model_config extracted from pinned public checkpoint metadata',
                                                'Single-GPU strategy auto instead of training DDP; numerical settings unchanged',
                                                'Record coverage/native metric dictionaries; skip formal scoring for smoke'],
                  'scripts_sha256': {name: file_hash(run / name) for name in
                                     ['run_m27_native.py', 'run_m27_native.sh', 'm27_native_config.py', 'm27_lingbot_config.json']}})
    identity = {key: audit[key] for key in ['source_bundle', 'annotation_sha256', 'checkpoint_sha256', 'environment', 'scripts_sha256']}
    fingerprint = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    audit['protocol_fingerprint'] = fingerprint  # Same for smoke/full with identical inputs and code.
    write(run / 'protocol.json', audit)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=args.device, HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
               PYTHONNOUSERSITE='1', PYTHONDONTWRITEBYTECODE='1', PYTHONUNBUFFERED='1',
               PYTHONPATH=os.pathsep.join([str(run), str(SOURCE)]), M27_RUN_DIR=str(run),
               M27_MAX_IMAGES=str(args.max_images), MIXED_PRECISION='32', SAM3_COMPILE='0',
               MPLCONFIGDIR=str(run / 'matplotlib'))
    command = [str(Path(sys.executable).parent / 'vis4d'), 'test', '--config', str(run / 'm27_native_config.py'), '--gpus', '1']
    write(run / 'status.json', {'state': 'running', 'command': command, 'gpu': args.device, 'protocol_fingerprint': fingerprint})
    print(f'M27 output: {run}', flush=True)
    try:
        with (run / 'native.log').open('w') as log:
            process = subprocess.Popen(command, cwd=work, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            try:
                for line in process.stdout:
                    print(line, end='', flush=True)
                    log.write(line)
                    log.flush()
                if process.wait():
                    raise RuntimeError(f'Official evaluation exit code: {process.returncode}; see native.log')
            except BaseException:
                process.terminate()
                process.wait()
                raise
        loaded = json.loads((run / 'weight_load_audit.json').read_text())
        assert loaded['strict_load'] and loaded['tensor_count'] == 2011
        expected = json.loads((run / 'expected_ids.json').read_text())
        for kind in ['detection_bbox', 'detection_dist']:
            actual = json.loads((run / f'{kind}_coverage.json').read_text())
            assert len(actual) == len(set(actual)) == len(expected) and set(actual) == set(expected)
        if not args.max_images:
            metrics = {key: json.loads((run / f'{key}_metrics.json').read_text())
                       for key in ['detection_bbox_2D', 'detection_bbox_3D', 'detection_dist_3D']}
            for key, values in metrics.items():
                ap = values['2D/AP' if key.endswith('2D') else '3D/AP']
                assert ap is not None and math.isfinite(ap) and 0 <= ap <= 1, (key, ap)
            write(run / 'metrics.json', {'native_metrics': metrics, 'AP_unit': 'fraction (multiply AP by 100 for percent)',
                                        'official_reference': audit['reference'], 'official_match_verified': False})
        write(run / 'status.json', {'state': 'partial_complete' if args.max_images else 'complete',
                                     'images': len(expected), 'formal_metrics': not bool(args.max_images),
                                     'official_match_verified': False, 'protocol_fingerprint': fingerprint})
        print(f'M27 completed: {run}', flush=True)
    except BaseException as exc:
        write(run / 'status.json', {'state': 'failed', 'error': f'{type(exc).__name__}: {exc}', 'protocol_fingerprint': fingerprint})
        raise


if __name__ == '__main__':
    main()
