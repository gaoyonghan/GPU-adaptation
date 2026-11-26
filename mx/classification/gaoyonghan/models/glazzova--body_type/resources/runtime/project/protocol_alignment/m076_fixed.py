"""Replay the original 3000 samples using portable, content-verified identities."""
import argparse
from collections import Counter
import hashlib
import importlib.util
import json
from pathlib import Path, PurePosixPath
import sys

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def load_rows(root, manifest=HERE / 'm076_samples.jsonl', verify=True):
    root = Path(root)
    rows = []
    for line in manifest.read_text().splitlines():
        row = json.loads(line)
        rel = PurePosixPath(row['relative_path'])
        if rel.is_absolute() or '..' in rel.parts:
            raise ValueError('Sample path must stay relative to the prepared dataset')
        path = root / str(rel)
        if verify and sha(path) != row['image_sha256']:
            raise ValueError(f'Sample content changed: {rel}')
        rows.append({'path': str(path), 'label': row['label'], 'shard': row['shard'],
                     'generator': row['generator']})
    if len({r['path'] for r in rows}) != len(rows):
        raise ValueError('Duplicate samples in frozen manifest')
    return rows


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--dataset-root', type=Path, required=True)
    parser.add_argument('--model-root', type=Path, required=True)
    parser.add_argument('--device', choices=['cuda', 'cpu'], default='cuda')
    args, runner_args = parser.parse_known_args()
    meta = json.loads((HERE / 'm076_protocol.json').read_text())
    manifest = HERE / 'm076_samples.jsonl'
    if sha(manifest) != meta['portable_manifest_sha256']:
        raise ValueError('Frozen sample manifest changed')
    rows = load_rows(args.dataset_root / 'm76_animedl_test_v1/prepared')
    expected = Counter({(shard, label): 500 for shard in ['0050', '0110', '0240'] for label in [0, 1]})
    if Counter((r['shard'], r['label']) for r in rows) != expected:
        raise ValueError('Expected the original balanced 3000 samples')
    archived = PROJECT / 'code_archive/436e3789f7a10f9d_eval_m76_local.py'
    if sha(archived) != meta['archived_runner_sha256']:
        raise ValueError('Archived inference implementation changed')
    spec = importlib.util.spec_from_file_location('m076_archived', archived)
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    if args.device == 'cuda' and not runner.torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable; refusing silent CPU fallback')
    if args.device == 'cpu' and runner.torch.cuda.is_available():
        raise RuntimeError('For CPU evaluation set CUDA_VISIBLE_DEVICES empty')
    runner.MODEL = args.model_root / 'legekka/AI-Anime-Image-Detector-ViT'

    def frozen_manifest(per_shard):
        if per_shard != 500:
            raise ValueError('Frozen protocol requires --per-shard 500')
        return rows

    runner.manifest = frozen_manifest
    sys.argv = [str(archived), *runner_args]
    runner.main()
    output = Path(runner_args[runner_args.index('--out') + 1])
    metrics = json.loads((output / 'metrics.json').read_text())
    metrics['sampling_protocol'] = {
        'id': 'm076_original_3000_relative_content_v1',
        'portable_manifest_sha256': sha(manifest),
        'original_manifest_sha256': meta['original_manifest_sha256'],
        'wrapper_sha256': sha(Path(__file__)),
        'image_content_verified': True,
        'full_original_sample_set': metrics['images'] == 3000,
    }
    (output / 'metrics.json').write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + '\n')


if __name__ == '__main__':
    main()
