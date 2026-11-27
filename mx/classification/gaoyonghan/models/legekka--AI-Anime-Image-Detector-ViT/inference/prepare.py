#!/usr/bin/env python3
"""Model-local output conversion, invoked automatically by inference.py."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

MODEL = Path(__file__).resolve().parents[1]
CONFIG = json.loads(Path(__file__).with_name('entrypoint.json').read_text())
sys.path.insert(0, str(MODEL / 'inference/output_format'))

def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2))
    temp.replace(path)

def verify_native_completion(root, phase):
    terminal = []
    names = ['metrics.json']
    if CONFIG['sequence'] == 41: names = ['summary.json']
    if CONFIG['sequence'] == 70: names = ['run_manifest.json']
    for path in sorted(p for name in names for p in root.rglob(name)):
        parts = path.relative_to(root).parts
        if any(p.startswith(('standardized', 'score_', '.')) or p in ('evaluation', 'scoring_inputs', 'project_source') for p in parts):
            continue
        if (phase == 'full' and 'smoke' in parts) or (phase == 'smoke' and 'full' in parts):
            continue
        value = json.loads(path.read_text())
        if path.name == 'metrics.json':
            if value.get('model_number') != CONFIG['sequence']: continue
            predictions = path.with_name('predictions.jsonl')
            data = predictions.read_bytes(); images = value.get('images')
            if not isinstance(images, int) or isinstance(images, bool) or images <= 0:
                raise RuntimeError('Native metrics have no completed image count')
            if hashlib.sha256(data).hexdigest() != value.get('predictions_sha256'):
                raise RuntimeError('Native predictions differ from completed metrics')
            if len([line for line in data.splitlines() if line.strip()]) != images:
                raise RuntimeError('Native prediction count differs from completed metrics')
            state = 'completed'
        else:
            state = value.get('state', value.get('status'))
        if state in ('failed', 'error', 'running', 'scoring'):
            raise RuntimeError('Native output is not completed: ' + str(path) + ' (' + str(state) + ')')
        terminal.append(state)
    if not any(s in ('complete', 'completed', 'partial_complete', 'prediction_only_complete', 'smoke_complete', 'provisional_complete', 'source_evaluation_complete') for s in terminal):
        raise RuntimeError('No completed native status.json found in the selected run and phase')

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--phase', choices=['full', 'smoke'], default='full')
    args = parser.parse_args()
    root = args.run_dir.resolve()
    bundles = []
    exchanges = []
    result = {'task': CONFIG['task'], 'sequence': CONFIG['sequence'], 'phase': args.phase, 'state': 'failed', 'bundles': []}
    try:
        verify_native_completion(root, args.phase)
        if CONFIG['task'] == 'classification':
            import prepare_classification as preparation
            report = preparation.finalize_run(root, CONFIG['sequence'], args.phase, successful=True)
            if report.get('state') != 'exported':
                raise RuntimeError(json.dumps(report, ensure_ascii=False))
            bundles = [Path(x['output']).resolve() for x in report['items']]
        else:
            import detection
            import prepare_detection as preparation
            report = detection.finalize_run(root, CONFIG['sequence'], args.phase, run_success=True)
            if not report or report.get('state') != 'exported':
                raise RuntimeError(json.dumps(report, ensure_ascii=False))
            exchanges = [(root / x['output']).resolve() for x in report['items']]
            if not CONFIG.get('scoring_protocol'):
                result.update(state='completed', scoring_ready=False, reason='No validated scoring protocol for this model', exchanges=[str(p.relative_to(root)) for p in exchanges])
                write(root / 'inference_output.json', result)
                return
            from model_paths import values
            mapping = values()
            config = {'dataset_root': mapping['__DATASET_ROOT__'],
                      'model_root': str(MODEL / 'resources/model_index'),
                      'official_source_root': str(MODEL / 'resources/runtime/assets/workspace/official_sources/object_detection'),
                      'phase': args.phase}
            prepared = preparation.score_model_run(root, CONFIG['sequence'], config, sys.executable, score=False)
            if prepared.get('state') != 'prepared':
                raise RuntimeError(json.dumps(prepared, ensure_ascii=False))
            bundles = [Path(x['bundle']).resolve() for x in prepared['items']]
        if not bundles:
            raise RuntimeError('No standardized scoring bundles produced')
        if any(not p.is_relative_to(root) for p in bundles):
            raise RuntimeError('Standardized bundle escaped the selected run directory')
        for bundle in bundles:
            for name in ('protocol.json', 'annotations.json', 'predictions.json'):
                if not (bundle / name).is_file():
                    raise RuntimeError('Incomplete standardized output: ' + str(bundle / name))
        result.update(state='completed', scoring_ready=True, bundles=[str(p.relative_to(root)) for p in bundles], exchanges=[str(p.relative_to(root)) for p in exchanges])
        write(root / 'inference_output.json', result)
    except BaseException as exc:
        result.update(state='failed', reason=str(exc), bundles=[])
        write(root / 'inference_output.json', result)
        raise

if __name__ == '__main__':
    main()
