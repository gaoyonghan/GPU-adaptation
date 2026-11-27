#!/usr/bin/env python3
"""Automatic model-local conversion; no model imports and no scoring."""
import argparse
import json
from pathlib import Path
import sys
import time

MODEL = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MODEL / 'inference/output_format'))
from prepare_classification import finalize_run
from classification import write
from model_paths import configuration


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-dir', required=True, type=Path)
    p.add_argument('--phase', choices=['full', 'smoke'], default='full')
    a = p.parse_args()
    root = a.run_dir.resolve()
    if not root.is_relative_to(MODEL / 'result'):
        raise ValueError('Run directory must belong to this model')
    start = time.perf_counter()
    manifest_path = root / 'inference_output.json'
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    manifest.update(task='classification', sequence=configuration()['sequence'], phase=a.phase,
                    state='failed', scoring_ready=False, bundles=[])
    try:
        report = finalize_run(root, configuration()['sequence'], a.phase)
        bundles = [Path(row['output']).resolve() for row in report['items']]
        if not bundles or any(not p.is_relative_to(root) for p in bundles):
            raise ValueError('No valid model-local bundles')
        manifest.update(state='completed', scoring_ready=True, bundles=[str(p.relative_to(root)) for p in bundles],
                        conversion_seconds=time.perf_counter() - start)
        manifest.pop('reason', None)
        write(manifest_path, manifest)
    except BaseException as exc:
        manifest.update(reason=str(exc))
        write(manifest_path, manifest)
        raise


if __name__ == '__main__':
    main()
