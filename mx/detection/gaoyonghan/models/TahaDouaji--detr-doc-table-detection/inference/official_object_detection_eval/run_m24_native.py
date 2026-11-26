"""M24: official Ultralytics 8.0.21 DetectionValidator, no custom inference/AP."""
import argparse
from collections import Counter
from datetime import datetime
import hashlib
from importlib import import_module
from importlib.metadata import version
import json
from pathlib import Path
import shutil
import tempfile
from unittest.mock import patch
import zipfile

from run_m07_native import sha, write

OUT = Path('/data2/gaoyonghan/gpu_model_standalone_20261110/release/model/检测/TahaDouaji--detr-doc-table-detection/resources/runtime/assets/workspace/results/m24_native8021_v1')
MODEL = Path('/data2/gaoyonghan/gpu_model_standalone_20261110/release/model/检测/TahaDouaji--detr-doc-table-detection/resources/model_index/keremberke/yolov8m-table-extraction')
ARCHIVE = Path('/home/ubuntu/文档/ChatGPT/idea 2/outputs/launch_audit_20261005/migration_inputs/datasets/table_extraction_v2_valid/data/valid.zip')
WEIGHT_SHA = '93b9ab71462a376214872d2f6000775e10abee87441e5e8a3c3f5e4bbfff3489'
GT_SHA = '7488c82780ce2c060c51aef378ce32344196e0da4f9bc9d08afb216f7a80202c'
SPEC = {
    'sequence': 24, 'model_id': 'keremberke/yolov8m-table-extraction',
    'model_card': 'https://huggingface.co/keremberke/yolov8m-table-extraction',
    'dataset': 'keremberke/table-extraction', 'split': 'validation',
    'dataset_relation': 'original_exact', 'images': 70, 'boxes': 109,
    'classes': ['bordered', 'borderless'], 'official_reference_AP50': .95194,
    'fidelity': 'reference_baseline', 'case_type': 'special',
    'checkpoint_evidence': {'version': '8.0.21', 'train_batch': 2, 'v5loader': True,
                            'imgsz': 640, 'conf': .001, 'iou': .7, 'max_det': 300},
    'known_differences': [
        'Native 8.0.21 validation and AP replace predict plus COCOeval; no AP algorithm rewritten.',
        'Batch=4 inferred from checkpoint training batch=2 and native trainer validation batch*2.',
        'FP16 follows native CUDA training-time validation; standalone checkpoint has half=False.',
        'Author exact scoring command is unpublished; native validation is not certified historical reproduction.',
        'Reuse Python 3.9 / torch 2.2.2+cu121 / torchvision 0.17.2+cu121, not a historical dependency lock.',
        'ultralyticsplus 0.0.23 Hub loader is unnecessary: load local official best.pt directly.',
        'Official 8.4.0 convert_coco only prepares labels; images/geometry/classes remain unchanged.',
        'Old Ultralytics config is isolated during import; analytics disabled; evaluator source unmodified.',
        'Smoke predictions are not reused: native validator reruns all 70 images for consistent AP aggregation.'
    ]}


def prepare():
    from PIL import Image
    from ultralytics.data.converter import convert_coco
    assert version('ultralytics') == '8.4.0'
    OUT.mkdir(parents=True, exist_ok=True)
    root = OUT / 'dataset'
    if root.exists():
        saved = json.loads((root / 'prepared.json').read_text())
        assert saved['archive_sha256'] == sha(ARCHIVE), 'Source changed; do not reuse prepared data'
        assert saved['annotation_sha256'] == GT_SHA
        for relative, checksum in saved['files_sha256'].items():
            assert sha(root / relative) == checksum, f'Prepared file changed: {relative}'
        return
    with tempfile.TemporaryDirectory(prefix='m24-', dir=OUT) as tmp:
        source = Path(tmp) / 'annotations'
        source.mkdir()
        staging = Path(tmp) / 'dataset'
        with zipfile.ZipFile(ARCHIVE) as z:
            raw = z.read('_annotations.coco.json')
            a = json.loads(raw)
            assert len(a['images']) == 70 and len(a['annotations']) == 109
            assert {x['id']: x['name'] for x in a['categories']} == {0: 'bordered', 1: 'borderless'}
            assert len({x['id'] for x in a['images']}) == 70
            converted = json.loads(raw)
            for c in converted['categories']:
                c['id'] += 1  # Official converter expects 1-based COCO categories.
            for x in converted['images']:
                x['file_name'] = str(x['id']) + Path(x['file_name']).suffix
            for x in converted['annotations']:
                assert not x.get('iscrowd', 0)
                x['category_id'] += 1
            write(source / 'instances_val.json', converted)
            convert_coco(str(source), str(staging), cls91to80=False)
            images = staging / 'images/val'
            images.mkdir(parents=True, exist_ok=True)
            files, counts = {}, Counter()
            for x in a['images']:
                image = images / (str(x['id']) + Path(x['file_name']).suffix)
                image.write_bytes(z.read(x['file_name']))
                with Image.open(image) as im:
                    assert im.size == (x['width'], x['height'])
                    im.verify()
                label = staging / 'labels/val' / (str(x['id']) + '.txt')
                rows = [list(map(float, row.split())) for row in label.read_text().splitlines()]
                expected = [r for r in a['annotations'] if r['image_id'] == x['id']]
                assert len(rows) == len(expected), 'Converter dropped/deduplicated boxes'
                for row, ann in zip(rows, expected):
                    assert len(row) == 5 and row[0] == ann['category_id']
                    cx, cy, w, h = row[1:]
                    restored = [(cx-w/2)*x['width'], (cy-h/2)*x['height'], w*x['width'], h*x['height']]
                    assert max(abs(v-b) for v, b in zip(restored, ann['bbox'])) < .02
                    assert all(0 <= v <= 1 for v in row[1:]) and min(w, h) > 0
                    counts[int(row[0])] += 1
                for p in [image, label]:
                    files[str(p.relative_to(staging))] = sha(p)
            assert counts == {0: 23, 1: 86}
        (staging / 'annotation.json').write_bytes(raw)
        assert sha(staging / 'annotation.json') == GT_SHA
        write(staging / 'prepared.json', {'archive_sha256': sha(ARCHIVE),
              'annotation_sha256': GT_SHA, 'files_sha256': files,
              'converter_version': version('ultralytics'),
              'converter_sha256': sha(Path(import_module('ultralytics.data.converter').__file__)),
              'mapping': 'COCO ids 0,1 -> temporary ids 1,2 -> YOLO ids 0,1; numeric image filenames only',
              'counts': dict(counts)})
        staging.rename(root)
    print('Prepared official validation: 70 images, 109 boxes; image bytes and box geometry verified.', flush=True)


def native_imports():
    import yaml
    assert version('ultralytics') == '8.0.21'
    assert version('torch') == '2.2.2+cu121' and version('torchvision') == '0.17.2+cu121'
    # 8.0.21 ignores YOLO_CONFIG_DIR. Limit Path.home redirection to package import,
    # so the old settings schema cannot overwrite the user's newer YOLO settings.
    runtime = OUT / '.runtime'
    config = runtime / '.config/Ultralytics'
    config.mkdir(parents=True, exist_ok=True)
    settings = {'datasets_dir': str(OUT), 'weights_dir': str(MODEL), 'runs_dir': str(OUT),
                'sync': False, 'uuid': 0, 'settings_version': '0.0.1'}
    (config / 'settings.yaml').write_text(yaml.safe_dump(settings, sort_keys=False))
    font = Path('/data2/gaoyonghan/gpu_model_standalone_20261110/release/model/检测/TahaDouaji--detr-doc-table-detection/resources/runtime/assets/fonts/Arial.ttf')
    assert font.is_file(), 'Missing cached Arial.ttf; do not start an implicit network download'
    if not (config / font.name).exists():
        (config / font.name).symlink_to(font)
    with patch.object(Path, 'home', return_value=runtime):
        cfg = import_module('ultralytics.yolo.cfg')
        val = import_module('ultralytics.yolo.v8.detect.val')
    utils = import_module('ultralytics.yolo.utils')
    assert utils.USER_CONFIG_DIR == config and not utils.SETTINGS['sync']
    assert sha(MODEL / 'best.pt') == WEIGHT_SHA, 'Audited checkpoint changed'
    return cfg, val


def evaluate(args):
    cfg, val = native_imports()
    import torch
    import yaml
    if not torch.cuda.is_available():
        raise RuntimeError('This frozen run uses CUDA FP16; no silent CPU fallback')
    run = OUT / datetime.now().strftime('run_%Y%m%d_%H%M%S_%f')
    run.mkdir()
    root = OUT / 'dataset'
    a = json.loads((root / 'annotation.json').read_text())
    selected = sorted(a['images'], key=lambda x: x['id'])[:args.max_images or None]
    ids = {x['id'] for x in selected}
    counts = Counter(x['category_id'] for x in a['annotations'] if x['image_id'] in ids)
    listing = run / 'val.txt'
    listing.write_text(''.join(str(root / 'images/val' / (str(x['id']) + Path(x['file_name']).suffix)) + '\n' for x in selected))
    config = run / 'data.yaml'
    config.write_text(yaml.safe_dump({'path': str(root), 'train': str(listing), 'val': str(listing),
                                     'names': SPEC['classes'], 'nc': 2}, sort_keys=False))
    options = dict(data=str(config), task='detect', mode='val', imgsz=640, batch=4,
                   conf=.001, iou=.7, max_det=300, v5loader=True, half=True, rect=True,
                   device=args.device, workers=0, single_cls=False, agnostic_nms=False,
                   augment=False, plots=False, save_json=True, save_txt=False,
                   project=str(OUT), name=run.name, exist_ok=True)
    protocol = {**SPEC, 'weight_sha256': WEIGHT_SHA,
                'dataset_preparation': json.loads((root / 'prepared.json').read_text()),
                'software': {n: version(n) for n in ['ultralytics', 'torch', 'torchvision', 'numpy', 'Pillow', 'opencv-python']},
                'official_source_sha256': {n: sha(Path(import_module(n).__file__)) for n in [
                    'ultralytics.yolo.v8.detect.val', 'ultralytics.yolo.engine.validator',
                    'ultralytics.yolo.utils.metrics', 'ultralytics.yolo.data.dataloaders.v5loader']},
                'options': options, 'script_sha256': sha(Path(__file__)), 'sample_ids': sorted(ids)}
    frozen = {k: v for k, v in protocol.items() if k != 'sample_ids'}
    frozen['options'] = {k: v for k, v in options.items() if k not in ['data', 'project', 'name']}
    fingerprint = hashlib.sha256(json.dumps(frozen, sort_keys=True).encode()).hexdigest()
    state = {'state': 'running', 'protocol': protocol, 'protocol_fingerprint': fingerprint,
             'is_smoke': bool(args.max_images)}
    write(run / 'status.json', state)
    shutil.copy2(__file__, run / Path(__file__).name)
    shutil.copy2(Path(__file__).with_suffix('.sh'), run / Path(__file__).with_suffix('.sh').name)
    print(f'24号原生验证：{len(ids)} images；输出 {run}', flush=True)
    try:
        validator = val.DetectionValidator(args=cfg.get_cfg(overrides=options), save_dir=run)
        validator(model=str(MODEL / 'best.pt'))
        seen = [int(Path(p).stem) for p in validator.dataloader.dataset.im_files]
        assert len(seen) == validator.seen == len(ids) and set(seen) == ids
        assert validator.nt_per_class.tolist() == [counts[0], counts[1]]
        assert validator.names == {0: 'bordered', 1: 'borderless'} and not validator.is_coco
        assert validator.model.fp16 and validator.args.v5loader
        write(run / 'predictions.json', validator.jdict)
        write(run / 'actual_options.json', vars(validator.args))
        if args.max_images:
            state['state'] = 'partial_complete'
            print('冒烟通过；原生验证器输出的小样本指标不作为正式结果。', flush=True)
        else:
            scores = {k: float(v) for k, v in validator.metrics.results_dict.items()}
            per_class = {validator.names[int(c)]: dict(zip(['precision', 'recall', 'AP50', 'AP'],
                         map(float, validator.metrics.class_result(i))))
                         for i, c in enumerate(validator.metrics.ap_class_index)}
            write(run / 'metrics.json', {'unit': 'fraction_0_to_1', 'images': validator.seen,
                  'metrics': scores, 'per_class': per_class, 'official_reference_AP50': .95194,
                  'AP50_difference_percentage_points': 100 * (scores['metrics/mAP50(B)'] - .95194),
                  'strict_official_reproduction': False})
            state['state'] = 'complete'
            print(f'全量完成：AP50={100*scores["metrics/mAP50(B)"]:.4f}%；模型卡=95.1940%；{run}', flush=True)
        state.update(completed_images=validator.seen, predictions_sha256=sha(run / 'predictions.json'))
    except BaseException as exc:
        state.update(state='failed', error=repr(exc))
        raise
    finally:
        write(run / 'status.json', state)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--describe', action='store_true')
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--preflight', action='store_true')
    parser.add_argument('--device', default='0')
    parser.add_argument('--max-images', type=int, choices=[0, 10], default=0)
    args = parser.parse_args()
    if args.describe:
        print(json.dumps(SPEC, ensure_ascii=False, indent=2))
    elif args.prepare_only:
        prepare()
    elif args.preflight:
        native_imports()
        print('Native 8.0.21 imports and weight verified; no inference run.')
    else:
        evaluate(args)


if __name__ == '__main__':
    main()
