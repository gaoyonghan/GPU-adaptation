#!/usr/bin/env python3
"""M76: published ViT inference + AnimeDL-2M test raw/t2i + native ImageAccuracy."""
import argparse
import ast
from collections import Counter
from datetime import datetime
import hashlib
import importlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import sys
import tarfile

DATA_REPO = 'FlyTweety/AnimeDL-2M'
DATA_REV = 'f1ef103a2ded7d3e1a39c09dd05cd5ecc357a33b'
REAL_REPO = 'deepghs/danbooru2024-sfw'
REAL_REV = '7ccab00fa0369e81c2a6e92074ed8ff7b43cf945'
CODE_REV = 'bd81880df8577aa3d7ae746e1b492bc837d3a77d'
SHARDS = ['0050', '0110', '0240', '0260', '0400', '0430', '0480', '0680', '0930', '0970']
COUNTS = [(6442,15282),(6378,15558),(6359,15228),(6326,14960),(6414,15392),
          (6425,15516),(6411,15546),(6403,0),(6343,0),(6367,0)]
MODEL = Path('__MODEL_ROOT__/weights/legekka/AI-Anime-Image-Detector-ViT')
ROOT = Path('__DATASET_ROOT__/m76_animedl_test_v1')
OUT = Path('__MODEL_ROOT__/resources/runtime/assets/workspace/results/m76_animedl_v1')


def digest(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def write_json(path, data):
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n')
    temp.replace(path)


def safe_path(value):
    p = PurePosixPath(value)
    if p.is_absolute() or '..' in p.parts or not p.parts or '\\' in value:
        raise ValueError(f'不安全的数据路径: {value}')
    return p


def selection(rows, shard):
    items = []
    for index, row in enumerate(rows):
        for label, paths in [(0, [row['raw']]), (1, row['t2i'])]:
            for name in paths:
                p = safe_path(name)
                if p.parts[0] != shard or (label == 1 and p.parts[1] != 'text2image'):
                    raise ValueError(f'清单路径与分片/任务不符: {name}')
                items.append({'source_dataset': DATA_REPO, 'split': 'test', 'shard': shard,
                              'source_index_row': index, 'source_path': name, 'target': label,
                              'image_id': ('real:' if label == 0 else 'ai:') + name})
    if len({x['image_id'] for x in items}) != len(items):
        raise ValueError('官方清单含重复图片引用；需核查后再确定计数')
    return items


def source_code(root):
    import requests
    base = f'https://raw.githubusercontent.com/FlyTweety/AnimeDL2M/{CODE_REV}/AniXplore/IMDLBenCo/IMDLBenCo/evaluation/'
    files = {f'm76_official/{name}': base + name for name in ('Accuracy.py', 'abstract_class.py')}
    files['resize.py'] = f'https://huggingface.co/datasets/{DATA_REPO}/resolve/{DATA_REV}/real_images/resize.py'
    metadata = {}
    for name, url in files.items():
        p = root / 'official_sources' / name
        p.parent.mkdir(parents=True, exist_ok=True)
        response = requests.get(url, timeout=60)
        response.raise_for_status()
        payload = response.content
        ast.parse(payload.decode('utf-8-sig'))
        if p.exists() and p.read_bytes() != payload:
            raise ValueError(f'固定版本官方源码与缓存不符: {p}')
        if not p.exists():
            p.write_bytes(payload)
        metadata[name] = {'url': url, 'sha256': digest(p)}
    return metadata


def resize_function(root):
    from PIL import Image
    # Keep the author's function unchanged; exclude its module-level CLI/pool launch.
    tree = ast.parse((root / 'official_sources/resize.py').read_text())
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'resize_images')
    namespace = {'Image': Image, 'os': os}
    exec(compile(ast.Module(body=[node], type_ignores=[]), 'official resize.py::resize_images', 'exec'), namespace)
    return namespace['resize_images']


def unpack_selected(archive, items, root, resize):
    from PIL import Image
    by_name = {PurePosixPath(x['source_path']).name: x for x in items}
    if len(by_name) != len(items):
        raise ValueError('选定图片文件名不唯一，不能安全匹配压缩包')
    seen = set()
    with tarfile.open(archive, mode='r|*') as tar:
        for member in tar:
            name = PurePosixPath(member.name).name
            if name not in by_name:
                continue
            safe_path(member.name)
            if not member.isfile() or name in seen:
                raise ValueError(f'压缩包成员不是独立普通文件: {member.name}')
            seen.add(name)
            item = by_name[name]
            # t2i filenames contain generator+ID; also verify the source subdirectory.
            if item['target'] == 1 and not member.name.lstrip('./').endswith(item['source_path']):
                raise ValueError(f'生成图片成员路径不匹配: {member.name}')
            dest = root / 'prepared' / ('real' if item['target'] == 0 else 'ai') / item['source_path']
            dest.parent.mkdir(parents=True, exist_ok=True)
            original = root / 'original_real' / item['source_path'] if item['target'] == 0 else dest
            original.parent.mkdir(parents=True, exist_ok=True)
            payload = tar.extractfile(member).read()
            if original.exists() and original.read_bytes() != payload:
                raise ValueError(f'已有图片与原压缩包不符，保留旧文件并停止: {original}')
            if not original.exists():
                original.write_bytes(payload)
            if item['target'] == 0 and not dest.exists():
                resize(original.name, str(original.parent), str(dest.parent), max_size=1024, align=16)
            # The author's resize catches errors; make missing/corrupt output fatal here.
            with Image.open(dest) as image:
                image.load()
            item.update({'prepared_path': str(dest.relative_to(root)), 'sha256': digest(dest),
                         'archive_member': member.name, 'original_sha256': hashlib.sha256(payload).hexdigest()})
    if seen != set(by_name):
        missing = sorted(set(by_name) - seen)
        raise ValueError(f'{archive.name}缺少{len(missing)}张选定图片，例如{missing[:5]}；不静默跳过')
    return items


def prepare(args):
    import requests
    from huggingface_hub import hf_hub_download
    root = args.data_root
    root.mkdir(parents=True, exist_ok=True)
    archives, all_items, indices = [], [], {}
    for shard, expected in zip(SHARDS, COUNTS):
        filename = f'data_list/test/{shard}.json'
        path = Path(hf_hub_download(DATA_REPO, filename, repo_type='dataset', revision=DATA_REV,
                                    local_dir=root / 'download/animedl'))
        indices[filename] = digest(path)
        items = selection(json.loads(path.read_text()), shard)
        counts = Counter(x['target'] for x in items)
        if (counts[0], counts[1]) != expected:
            raise ValueError(f'固定版本测试清单数量不符: {shard}, {counts}')
        for label, repo, rev, filename in [(0, REAL_REPO, REAL_REV, f'images/{shard}.tar'),
                                          (1, DATA_REPO, DATA_REV, f'fake_images/{shard}.tar.gz')]:
            if counts[label]:
                archives.append({'repo': repo, 'revision': rev, 'filename': filename, 'shard': shard,
                                 'label': label, 'selected_images': counts[label]})
        all_items.extend(items)
    endpoint = os.environ.get('HF_ENDPOINT', 'https://huggingface.co').rstrip('/')
    for repo, rev in [(DATA_REPO, DATA_REV), (REAL_REPO, REAL_REV)]:
        chosen = [a for a in archives if a['repo'] == repo]
        response = requests.post(f'{endpoint}/api/datasets/{repo}/paths-info/{rev}',
                                 json={'paths': [a['filename'] for a in chosen]}, timeout=60)
        response.raise_for_status()
        meta = {x['path']: x for x in response.json()}
        for a in chosen:
            a.update({'size': meta[a['filename']]['size'], 'sha256': meta[a['filename']]['lfs']['oid']})
    plan = {'dataset_revision': DATA_REV, 'real_revision': REAL_REV, 'code_revision': CODE_REV,
            'images': len(all_items), 'real_images': 63868, 'ai_images': 107482,
            'scope': 'all test raw + listed t2i; excludes inpaint, masks, civitai and train/val',
            'download_bytes': sum(a['size'] for a in archives), 'archives': archives,
            'index_sha256': indices, 'official_sources': source_code(root)}
    write_json(root / 'plan.json', plan)
    print(f"[m76] 测试范围: real=63868, ai=107482, 共171350张；压缩包{plan['download_bytes']/1e9:.2f} GB", flush=True)
    if args.plan:
        print(f'仅准备清单及官方源码，未下载图片: {root / "plan.json"}', flush=True)
        return
    resize = resize_function(root)
    manifest = []
    for a in archives:
        local = root / 'download' / ('real' if a['label'] == 0 else 'animedl')
        print(f"[m76] 获取/校验 {a['repo']}/{a['filename']}", flush=True)
        path = Path(hf_hub_download(a['repo'], a['filename'], repo_type='dataset', revision=a['revision'], local_dir=local))
        if path.stat().st_size != a['size'] or digest(path) != a['sha256']:
            raise ValueError(f'压缩包校验失败: {path}；原文件保留，请检查下载')
        stamp = root / f"prepared_{a['shard']}_{a['label']}.json"
        items = [x.copy() for x in all_items if x['shard'] == a['shard'] and x['target'] == a['label']]
        if stamp.exists():
            cached = json.loads(stamp.read_text())
            if [x['image_id'] for x in cached] != [x['image_id'] for x in items]:
                raise ValueError(f'已准备的图片清单不匹配: {stamp}')
            if any(not (root / x['prepared_path']).is_file() or digest(root / x['prepared_path']) != x['sha256'] for x in cached):
                raise ValueError(f'已准备图片缺失或损坏: {stamp}；保留现场，不覆盖')
        else:
            cached = unpack_selected(path, items, root, resize)
            write_json(stamp, cached)
        manifest.extend(cached)
    manifest.sort(key=lambda x: (x['shard'], x['source_index_row'], x['target'], x['source_path']))
    if len(manifest) != 171350 or len({x['image_id'] for x in manifest}) != 171350:
        raise ValueError('全量图片未完整覆盖或ID重复')
    write_json(root / 'manifest.json', manifest)
    write_json(root / 'prepared.json', {'state': 'ready', 'manifest_sha256': digest(root / 'manifest.json'),
                                       'plan_sha256': digest(root / 'plan.json')})
    print('[m76] 全量测试数据准备完成', flush=True)


def load_image(root, item):
    from PIL import Image
    path = root / safe_path(item['prepared_path'])
    payload = path.read_bytes()
    if hashlib.sha256(payload).hexdigest() != item['sha256']:
        raise ValueError(f'图片哈希不符: {path}')
    with Image.open(io.BytesIO(payload)) as image:
        return image.convert('RGB')


def evaluate(args):
    import importlib.metadata
    import torch
    import torch.distributed as dist
    from transformers import AutoModelForImageClassification, AutoFeatureExtractor
    root = args.data_root
    ready = json.loads((root / 'prepared.json').read_text())
    for name in ('manifest', 'plan'):
        if digest(root / f'{name}.json') != ready[f'{name}_sha256']:
            raise ValueError(f'{name}已改变，停止运行')
    plan = json.loads((root / 'plan.json').read_text())
    items = json.loads((root / 'manifest.json').read_text())
    if len(items) != 171350 or len({x['image_id'] for x in items}) != 171350:
        raise ValueError('正式清单不完整或ID重复')
    if Counter(x['target'] for x in items) != {0:63868, 1:107482}:
        raise ValueError('全量类别数量不符')
    for name, metadata in plan['official_sources'].items():
        if digest(root / 'official_sources' / name) != metadata['sha256']:
            raise ValueError(f'官方源码被修改: {name}')
    processor = AutoFeatureExtractor.from_pretrained(MODEL, local_files_only=True)
    config = json.loads((MODEL / 'config.json').read_text())
    if config['id2label'] != {'0':'real','1':'ai'}:
        raise ValueError('模型类别与既定二分类任务不符')
    if args.preflight:
        for i, item in enumerate(items):
            load_image(root, item)
            if i % 1000 == 0:
                print(f'[m76] 数据检查 {i}/{len(items)}', flush=True)
        print('ready: 171350张图片哈希、解码和两类覆盖检查通过；未运行推理', flush=True)
        return
    if args.max_images:
        # The small check must exercise both classes, unlike a raw-first slice.
        real = [x for x in items if x['target'] == 0]
        fake = [x for x in items if x['target'] == 1]
        mixed = [x for pair in zip(real, fake) for x in pair] + fake[len(real):]
        items = mixed[:args.max_images]
    full = len(items) == 171350
    out = args.output_root / ('full' if full else 'smoke') / datetime.now().strftime('run_%Y%m%d_%H%M%S_%f')
    out.mkdir(parents=True, exist_ok=False)
    for name in ('run_m76_animedl.py', 'run_m76_animedl.sh', 'README_m76.md'):
        shutil.copy2(Path(__file__).with_name(name), out / name)
    write_json(out / 'protocol.json', {'model_number':76, 'model_id':'legekka/AI-Anime-Image-Detector-ViT',
        'model_files_sha256':{p.name:digest(p) for p in MODEL.iterdir() if p.name in
                             ('config.json','preprocessor_config.json','model.safetensors','README.md')},
        'data_plan':plan, 'prepared_manifest_sha256':ready['manifest_sha256'],
        'dataset_relation':'public_replacement', 'author_accuracy_percent':94.68, 'officially_comparable':False,
        'selected_images':len(items), 'inference':'published AutoFeatureExtractor, 512x512, FP32, batch8, no TTA; softmax(ai)',
        'metric':'unchanged AnimeDL-2M ImageAccuracy, threshold > 0.5, single-rank NCCL; class rates are additional diagnostics',
        'adapter_boundary':'own raw/t2i test selection, download/extraction/batching; native resize_images function and ImageAccuracy; not AniXplore full pixel-evaluation pipeline',
        'source_sha256':{name:digest(out/name) for name in ('run_m76_animedl.py','run_m76_animedl.sh','README_m76.md')},
        'versions':{name:importlib.metadata.version(name) for name in ('torch','transformers','Pillow','numpy','huggingface-hub')}})
    write_json(out / 'status.json', {'state':'running','completed_images':0})
    completed, correct, counts, correct_class = 0, 0, Counter(), Counter()
    try:
        if not torch.cuda.is_available():
            raise RuntimeError('原生ImageAccuracy要求CUDA，本流程不回退到另写计分器')
        torch.cuda.set_device(0)
        dist.init_process_group('nccl', init_method=(out / 'distributed_init').resolve().as_uri(), rank=0, world_size=1)
        sys.path.insert(0, str(root / 'official_sources'))
        native = importlib.import_module('m76_official.Accuracy').ImageAccuracy(threshold=0.5)
        model = AutoModelForImageClassification.from_pretrained(MODEL, local_files_only=True,
            use_safetensors=True, torch_dtype=torch.float32, attn_implementation='eager').cuda().eval()
        with (out / 'predictions.jsonl').open('w') as stream, torch.inference_mode():
            for start in range(0, len(items), 8):
                batch = items[start:start+8]
                inputs = processor(images=[load_image(root, x) for x in batch], return_tensors='pt')
                logits = model(**{k:v.cuda() for k,v in inputs.items()}).logits
                if logits.shape != (len(batch),2) or not torch.isfinite(logits).all():
                    raise ValueError('输出维度或有限性检查失败')
                probs = logits.softmax(-1)[:,1]
                labels = torch.tensor([x['target'] for x in batch], device='cuda', dtype=torch.float32)
                native.batch_update(probs, labels)
                for item, prob in zip(batch, probs.cpu().tolist()):
                    pred = int(prob > 0.5)
                    record = dict(item, ai_probability=prob, prediction=pred)
                    stream.write(json.dumps(record, ensure_ascii=False)+'\n')
                    correct += int(pred == item['target'])
                    counts[item['target']] += 1
                    correct_class[item['target']] += int(pred == item['target'])
                completed += len(batch)
                stream.flush()
                if completed % 400 == 0 or completed == len(items):
                    write_json(out / 'status.json', {'state':'running','completed_images':completed})
                    print(f'[m76] {completed}/{len(items)}', flush=True)
        accuracy = native.epoch_update()
        if completed != len(items) or abs(accuracy - correct/completed) > 1e-12:
            raise ValueError('原生计分与逐图计数不一致或覆盖不全')
        result = {'state':'complete' if full else 'partial_complete', 'evaluation_images':completed,
                  'accuracy':accuracy,'accuracy_percent':accuracy*100,'correct':correct,
                  'class_counts':dict(counts), 'class_recognition_rates':{str(k):correct_class[k]/counts[k] for k in counts},
                  'dataset_relation':'public_replacement','author_accuracy_percent':94.68,'officially_comparable':False,
                  'predictions_sha256':digest(out/'predictions.jsonl')}
        write_json(out / ('metrics.json' if full else 'smoke_metrics.json'), result)
        write_json(out / 'status.json', {'state':result['state'],'completed_images':completed})
        print(json.dumps(result,ensure_ascii=False),flush=True)
        print(f'输出: {out}',flush=True)
    except Exception as exc:
        write_json(out/'status.json', {'state':'failed','completed_images':completed,'error':str(exc)})
        raise
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-root',type=Path,default=ROOT)
    p.add_argument('--output-root',type=Path,default=OUT)
    mode = p.add_mutually_exclusive_group()
    mode.add_argument('--plan',action='store_true')
    mode.add_argument('--prepare',action='store_true')
    mode.add_argument('--preflight',action='store_true')
    p.add_argument('--max-images',type=int,default=0)
    args=p.parse_args()
    if args.max_images < 0:
        p.error('--max-images必须为非负整数')
    if args.plan or args.prepare:
        prepare(args)
    else:
        evaluate(args)


if __name__ == '__main__':
    main()
