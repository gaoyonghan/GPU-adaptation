#!/usr/bin/env python3
"""Microsoft models + original build_transform/IN22KDATASET/validate, fixed partial dataset."""
import argparse
import ast
import datetime
import gc
import importlib.metadata
import importlib.util
import json
import logging
from pathlib import Path
import shutil
import subprocess
import time
from types import SimpleNamespace
import zipfile

import run_m85_m86_partial as common

REV = 'f82860bfb5225915aca09c3227159ee9e1df874d'
BASELINE = Path('__MODEL_ROOT__/resources/runtime/assets/workspace/results/m85_m86_imagenet21k_partial_v1/full/run_20260915_195624_743521')
MANIFEST_SHA = '9a7a2110e986cf81684885f00e5c23315cf1e6e5b5e3af33c2975118490524a5'
FILES = {
    'main.py': '42515ea96c1fd0a741543368c8f30fb427ec5c2eaa412e4c10b05e9a153bde46',
    'config.py': 'f44a2e92daedf28bc294c2d2328191df7793f12315a3d2c84c9390688aea57d8',
    'models/build.py': 'be6b1f6851cbd5372c1f0dff77df05cbc3ff29b85b887b53ed939f8809b93126',
    'models/swin_transformer.py': '779c9ad487e2c577eff12a60c7a2f92267e2bb55c68f7bb32c8edad5ad5bd1e1',
    'models/swin_transformer_v2.py': 'faccc1ff0ad4760716de985084eafa3998ceb5005f8496f8ac2956dbf96cfe07',
    'data/build.py': '0a7bbffd77989a65e761acd37e90b0b1bc6906d650eaaa4ab95931d6c41a7cc1',
    'data/imagenet22k_dataset.py': 'b529159f2b6814cb5b697272b78761572a68ef31267d37fb5d19729820e27356',
    'utils.py': 'b38939365135e87f9225bde4d32f9867c55aeb7108e8b7d741e8ceff914b2e4a',
    'configs/swin/swin_large_patch4_window12_384_22kto1k_finetune.yaml': 'a8331738549a971a87b9ada49fb614a76a4d0b7212a606735dfcdf736be0d1d5',
    'configs/swinv2/swinv2_large_patch4_window12_192_22k.yaml': '32466b3dd49a262c107d4bdc8dd659c159f90effbc28690ecf33382c911d31ca',
    'LICENSE': 'c2cfccb812fe482101a8f04597dfc5a9991a6b2748266c47ac91b6a5aae15383'}
WEIGHTS = {
    '85': ('v1.0.0', 'swin_large_patch4_window12_384_22k.pth', 928819451),
    '86': ('v2.0.0', 'swinv2_large_patch4_window12_192_22k.pth', 920684427)}
CONFIGS = {
    '85': 'configs/swin/swin_large_patch4_window12_384_22kto1k_finetune.yaml',
    '86': 'configs/swinv2/swinv2_large_patch4_window12_192_22k.yaml'}


def download(url, path, expected_sha=None, expected_size=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    receipt = path.with_suffix(path.suffix + '.receipt.json')
    if not path.exists():
        print(f'下载官方文件：{path.name}', flush=True)
        part = path.with_suffix(path.suffix + '.part')
        subprocess.run(['curl', '--fail', '--location', '--continue-at', '-', '--retry', '5',
                        '--connect-timeout', '30', '--max-time', '21600', url, '-o', str(part)], check=True)
        if expected_size is not None and part.stat().st_size != expected_size:
            raise ValueError(f'下载大小不符：{part}；文件保留')
        if expected_sha and common.file_sha(part) != expected_sha:
            raise ValueError(f'源码哈希不符：{part}；文件保留')
        part.replace(path)
    actual = common.file_sha(path)
    if expected_size is not None and path.stat().st_size != expected_size:
        raise ValueError(f'权重大小不符：{path}')
    if expected_sha and actual != expected_sha:
        raise ValueError(f'源码被修改：{path}')
    if receipt.exists():
        previous = json.loads(receipt.read_text())
        if previous['sha256'] != actual or previous['url'] != url:
            raise ValueError(f'缓存文件与此前记录不符：{path}')
    else:
        common.save(receipt, {'url': url, 'sha256': actual, 'bytes': path.stat().st_size,
                              'publisher_sha256_available': expected_sha is not None})
    return actual


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def original_function(path, name, namespace):
    """Extract an unchanged function AST; exclude training-only module imports/CLI side effects."""
    tree = ast.parse(path.read_text())
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name]
    if len(nodes) != 1:
        raise ValueError(f'原生函数缺失或重名：{path}:{name}')
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec'), namespace)
    return namespace[name]


def prepare_data(assets, run):
    import io
    from PIL import Image
    manifest = BASELINE / 'manifest.json'
    if common.file_sha(manifest) != MANIFEST_SHA:
        raise ValueError('既有HF对照清单改变；停止，不能更换测试样本')
    rows = json.loads(manifest.read_text())
    if len(rows) != 1192 or len({r['target'] for r in rows}) != 124:
        raise ValueError('对照范围异常')
    data = assets / 'data'
    data.mkdir(parents=True, exist_ok=True)
    print('校验并复用同一1192张图片；仅提取交集，不重复下载图片。', flush=True)
    with zipfile.ZipFile(common.ZIP) as z:
        for r in rows:
            path = data / r['official_path']
            if path.exists():
                payload = path.read_bytes()
            else:
                payload = z.read(r['zip_member'])
            if common.sha(payload) != r['sha256']:
                raise ValueError(f'图片与HF对照不一致：{path}')
            with Image.open(io.BytesIO(payload)) as im:
                im.convert('RGB').load()
            if not path.exists():
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(payload)
    shutil.copy2(manifest, run / 'manifest.json')
    shutil.copy2(BASELINE / 'coverage.json', run / 'coverage.json')
    return data, rows


def native_components(src):
    import torch
    import torch.distributed as dist
    from torchvision import transforms
    from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
    from timm.data.transforms import _pil_interp
    from timm.utils import accuracy, AverageMeter
    classes = {key: load_module(key, src / path).__dict__[key] for key, path in [
        ('SwinTransformer', 'models/swin_transformer.py'),
        ('SwinTransformerV2', 'models/swin_transformer_v2.py')]}
    builder = original_function(src / 'models/build.py', 'build_model', classes)
    transform = original_function(src / 'data/build.py', 'build_transform', {
        'transforms': transforms, '_pil_interp': _pil_interp,
        'IMAGENET_DEFAULT_MEAN': IMAGENET_DEFAULT_MEAN, 'IMAGENET_DEFAULT_STD': IMAGENET_DEFAULT_STD})
    reduce = original_function(src / 'utils.py', 'reduce_tensor', {'dist': dist})
    validator = original_function(src / 'main.py', 'validate', {
        'torch': torch, 'time': time, 'accuracy': accuracy, 'AverageMeter': AverageMeter,
        'reduce_tensor': reduce, 'logger': logging.getLogger('swin-native')})
    return load_module('native_config', src / 'config.py'), builder, transform, validator


def evaluate(number, args, src, assets, run, data, rows, components, versions):
    import torch
    import torch.distributed as dist
    from torch.utils.data import DataLoader
    out = run / ('m' + number)
    out.mkdir()
    common.save(out / 'status.json', {'state': 'preparing'})
    model = None
    hook = None
    try:
        tag, filename, size_bytes = WEIGHTS[number]
        weight = assets / 'weights' / filename
        url = f'https://github.com/SwinTransformer/storage/releases/download/{tag}/{filename}'
        weight_sha = download(url, weight, expected_size=size_bytes)
        config_module, builder, transform_builder, validator = components
        cfg = config_module.get_config(SimpleNamespace(cfg=str(src / CONFIGS[number]),
            opts=['MODEL.NUM_CLASSES', '21841', 'DATA.DATASET', 'imagenet22K',
                  'DATA.NUM_WORKERS', '0', 'PRINT_FREQ', '25', 'EVAL_MODE', 'True'],
            batch_size=2, disable_amp=True, output=str(out)))
        if (cfg.DATA.IMG_SIZE, cfg.TEST.CROP) != ((384, False) if number == '85' else (192, True)):
            raise ValueError('原生配置尺寸或测试裁剪改变')
        (out / 'resolved_config.yaml').write_text(cfg.dump())
        # Only official HTTPS release checkpoints are loaded here; the original may include a Yacs config.
        checkpoint = torch.load(weight, map_location='cpu', weights_only=False)
        state = checkpoint['model']
        if tuple(state['head.weight'].shape) != (21841, 1536):
            raise ValueError('下载的原生权重分类头不对应21841类')
        hf_root = common.MODEL_ROOT / common.MODELS[number][0]
        if common.file_sha(hf_root / 'pytorch_model.bin') != common.MODELS[number][2][0]:
            raise ValueError('HF对照权重改变')
        hf = torch.load(hf_root / 'pytorch_model.bin', map_location='cpu', weights_only=True)
        head_equal = all(torch.equal(state['head.' + k], hf['classifier.' + k]) for k in ['weight', 'bias'])
        del hf
        if not head_equal:
            raise ValueError('原生分类头与已下载HF权重不一致，停止以核查权重版本')
        model = builder(cfg)
        model.load_state_dict(state, strict=True)
        print(f'[{number}] 原生权重严格加载通过；分类头与HF逐元素一致。', flush=True)
        del checkpoint, state
        preprocessing = transform_builder(False, cfg)
        chosen = rows[:args.max_images] if args.max_images else rows
        ann = out / 'official_val_subset.json'
        common.save(ann, [[r['official_path'], r['target']] for r in chosen])
        dataset_type = load_module('native_imagenet22k', src / 'data/imagenet22k_dataset.py').IN22KDATASET
        dataset = dataset_type(str(data), ann_file=str(ann), transform=preprocessing)
        if len(dataset) != len(chosen):
            raise ValueError('原生读取器样本数量不符')
        for i in range(len(dataset)):
            image, target = dataset[i]
            if tuple(image.shape) != (3, cfg.DATA.IMG_SIZE, cfg.DATA.IMG_SIZE) or target != chosen[i]['target']:
                raise ValueError('原生数据读取器尺寸/标签不符')
        protocol = {'model_number': int(number), 'official_weight_url': url, 'weight_sha256': weight_sha,
            'source_commit': REV, 'source_hashes': FILES, 'versions': versions,
            'native_functions': ['models/build.py::build_model', 'data/build.py::build_transform',
                                 'data/imagenet22k_dataset.py::IN22KDATASET', 'main.py::validate', 'utils.py::reduce_tensor'],
            'native_model_implementation': True, 'native_validation_function_unmodified': True,
            'full_training_cli_used': False, 'hf_classifier_tensor_equal': head_equal,
            'backbone_tensor_equivalence_verified': False, 'preprocessing': str(preprocessing),
            'test_crop': cfg.TEST.CROP, 'input_size': cfg.DATA.IMG_SIZE, 'output_classes': 21841,
            'dtype': 'float32', 'amp': False, 'tf32': False, 'batch_size': 2,
            'manifest_sha256': MANIFEST_SHA, 'hf_baseline': str(BASELINE / ('m' + number)),
            'officially_comparable': False, 'scope': 'smoke' if args.max_images else 'full_partial_dataset',
            'special_case_85': 'Official 384 architecture/test YAML; explicit 21841-class head, no 1K finetuning',
            'training_overlap': 'unknown', 'original_image_byte_identity_verified': False}
        common.save(out / 'protocol.json', protocol)
        if args.preflight:
            common.save(out / 'status.json', {'state': 'preflight_passed', 'images': len(chosen)})
            print(f'[{number}] 原生CPU加载与数据预检通过（未运行推理）', flush=True)
            return
        loader = DataLoader(dataset, batch_size=2, shuffle=False, num_workers=0, pin_memory=True, drop_last=False)
        model = model.cuda().eval()
        dist.init_process_group('nccl', init_method=(out / 'dist_init').as_uri(), rank=0, world_size=1)
        processed = 0
        correct = [0, 0]
        with (out / 'predictions.jsonl').open('w') as f:
            def record(_module, _inputs, output):
                nonlocal processed
                if output.shape[1] != 21841 or not torch.isfinite(output).all():
                    raise ValueError('原生模型输出无效')
                for indices in output.topk(5, dim=1).indices.cpu().tolist():
                    row = chosen[processed]
                    f.write(json.dumps(dict(row, prediction=indices[0], top5=indices)) + '\n')
                    correct[0] += int(indices[0] == row['target'])
                    correct[1] += int(row['target'] in indices)
                    processed += 1
                if processed % 50 == 0 or processed == len(chosen):
                    f.flush()
                    common.save(out / 'status.json', {'state': 'running', 'processed': processed, 'total': len(chosen)})
            hook = model.register_forward_hook(record)
            started = time.monotonic()
            top1, top5, loss = validator(cfg, loader, model)
            elapsed = time.monotonic() - started
        if processed != len(chosen) or any(abs(v - 100 * c / processed) > 1e-4 for v, c in zip([top1, top5], correct)):
            raise ValueError('原生validate返回值与逐图复算不一致')
        baseline = json.loads((BASELINE / ('m' + number) / 'metrics.json').read_text())
        result = {'model_number': int(number), 'images': processed, 'classes': len({r['target'] for r in chosen}),
            'top1_percent': top1, 'top5_percent': top5, 'loss': loss, 'elapsed_s': elapsed,
            'correct_top1': correct[0], 'correct_top5': correct[1], 'scope': protocol['scope'],
            'predictions_sha256': common.file_sha(out / 'predictions.jsonl'), 'officially_comparable': False,
            'hf_baseline_top1': baseline['top1_percent'], 'hf_baseline_top5': baseline['top5_percent'],
            'delta_top1_pp': None if args.max_images else top1 - baseline['top1_percent'],
            'delta_top5_pp': None if args.max_images else top5 - baseline['top5_percent']}
        common.save(out / ('smoke_metrics.json' if args.max_images else 'metrics.json'), result)
        common.save(out / 'status.json', {'state': 'complete', 'processed': processed, 'scope': protocol['scope']})
        print(f'[{number}] 原生评测完成 Top1={top1:.4f}% Top5={top5:.4f}%：{out}', flush=True)
    except Exception as exc:
        common.save(out / 'status.json', {'state': 'failed', 'error': f'{type(exc).__name__}: {exc}'})
        raise
    finally:
        if hook is not None:
            hook.remove()
        if dist.is_initialized():
            dist.destroy_process_group()
        del model
        gc.collect()
        torch.cuda.empty_cache()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output-root', type=Path, required=True)
    p.add_argument('--model', choices=['both', '85', '86'], default='both')
    p.add_argument('--max-images', type=int, default=0)
    p.add_argument('--preflight', action='store_true')
    args = p.parse_args()
    if not 0 <= args.max_images <= 1192:
        p.error('MAX_IMAGES须为0到1192；0表示完整交集')
    expected = {**common.VERSIONS, 'timm': '0.4.12', 'yacs': '0.1.8', 'PyYAML': '6.0.2'}
    versions = {k: importlib.metadata.version(k) for k in expected}
    if versions != expected:
        raise RuntimeError(f'环境不符：{versions}')
    import torch
    if not args.preflight and not torch.cuda.is_available():
        raise RuntimeError('CUDA不可用，不回退CPU推理')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    run = args.output_root / ('preflight' if args.preflight else 'smoke' if args.max_images else 'full') / datetime.datetime.now().strftime('run_%Y%m%d_%H%M%S_%f')
    run.mkdir(parents=True)
    assets = args.output_root / 'assets'
    src = assets / ('Swin-Transformer-' + REV)
    for name, expected_sha in FILES.items():
        download(f'https://raw.githubusercontent.com/microsoft/Swin-Transformer/{REV}/{name}', src / name, expected_sha)
    snapshot = run / 'source'
    snapshot.mkdir()
    for name in ['run_m85_m86_native.py', 'run_m85_m86_native.sh', 'README_m85_m86_native.md', 'run_m85_m86_partial.py']:
        shutil.copy2(common.HERE / name, snapshot / name)
    shutil.copytree(src, snapshot / 'official')
    data, rows = prepare_data(assets, run)
    components = native_components(src)
    failures = []
    for number in (['85', '86'] if args.model == 'both' else [args.model]):
        try:
            evaluate(number, args, src, assets, run, data, rows, components, versions)
        except Exception as exc:
            print(f'[{number}] 失败：{type(exc).__name__}: {exc}', flush=True)
            failures.append(number)
    if failures:
        raise SystemExit(f'原生评测失败模型：{failures}；原始文件保留，未覆盖HF结果')


if __name__ == '__main__':
    main()
