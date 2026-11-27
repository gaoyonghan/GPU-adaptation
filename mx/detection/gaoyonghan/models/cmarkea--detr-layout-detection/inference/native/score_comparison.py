"""Recover workbook-specific metrics without changing the original predictions."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=int, choices=[42, 47], required=True)
    parser.add_argument('--result-root', type=Path, required=True)
    parser.add_argument('--dataset-root', type=Path, required=True)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    output = args.output or args.result_root / 'comparison_metrics.json'
    output.parent.mkdir(parents=True, exist_ok=True)
    if args.model == 42:
        script = HERE.parent / 'backends/remaining_detection_eval/review_20260905/model42_rescore.py'
        spec = importlib.util.spec_from_file_location('m042_frozen_rescore', script)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        module.RESULT = args.result_root
        module.ARCHIVE = args.dataset_root / 'doclaynet_test/DocLayNet_core.zip'
        module.OUTPUT = output
        module.main()
        data = json.loads(output.read_text())
        data['comparison_protocol'] = 'Existing 2026-09-05 workbook scorer; fixed score>0 and all-class micro F1. No new inference.'
    else:
        candidates = []
        for path in args.result_root.rglob('metrics.json'):
            if any(part.startswith('smoke') for part in path.parts):
                continue
            data = json.loads(path.read_text())
            if data.get('images') == 1144 and data.get('evaluated_classes') == 8 and 'native_metrics' in data:
                candidates.append(path)
        if not candidates:
            raise ValueError('Missing complete native 1144-image, eight-class result')
        source = max(candidates, key=lambda p: p.stat().st_mtime_ns)
        full = json.loads(source.read_text())
        names = ['person', 'bicycle', 'car']
        data = {
            'sequence': 47, 'images': full['images'], 'unit': 'fraction_0_to_1',
            'source_metrics': str(source), 'source_metrics_sha256': sha(source),
            'comparison_classes': names, 'comparison_class_count': 3,
            'comparison_AP50': sum(full['per_class'][name]['AP50'] for name in names) / len(names),
            'full_evaluation_class_count': full['evaluated_classes'],
            'full_AP50': full['native_metrics']['metrics/mAP50(B)'],
            'full_AP50_95': full['native_metrics']['metrics/mAP50-95(B)'],
            'per_class': full['per_class'], 'new_inference': False,
            'comparison_protocol': 'Explicit person/bicycle/car AP50 mean for the workbook 3/14-class entry; eight-class overall scores remain separate.',
            'official_comparable': False,
        }
    data['comparison_wrapper_sha256'] = sha(Path(__file__))
    output.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n')
    print(f'Comparison metrics: {output}', flush=True)


if __name__ == '__main__':
    main()
