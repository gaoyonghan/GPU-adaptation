"""M15: official 2-pixel dataset preparation + native TATR main.py evaluation.

No custom crop, forward pass, postprocessing, or GriTS formula. Runtime hooks
only skip redundant ImageNet initialization (all native weights must match)
and record progress. Use ground-truth table crops/words: this measures TSR,
not an end-to-end table detector or OCR system. Original data/results stay intact.
"""
import argparse
import contextlib
from datetime import datetime
import fcntl
import hashlib
import importlib
import json
import multiprocessing
from pathlib import Path
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET

REPO = Path('/data2/gaoyonghan/gpu_model_standalone_20261110/release/model/检测/nvidia--nemotron-table-structure-v1/resources/runtime/assets/workspace/official_sources/object_detection/table-transformer')
REVISION = '16d124f616109746b7785f03085100f1f6247575'
SOURCE = Path('/data2/gaoyonghan/gpu_model_standalone_20261110/release/model/检测/nvidia--nemotron-table-structure-v1/resources/runtime/assets/workspace/results/object_detection_official_v4_tables/_prepared_datasets/pubtables_structure_test')
OUT = Path('/data2/gaoyonghan/gpu_model_standalone_20261110/release/model/检测/nvidia--nemotron-table-structure-v1/resources/runtime/assets/workspace/results/m15_native_tight_v1')
WEIGHT = Path('/data2/gaoyonghan/gpu_model_standalone_20261110/release/model/检测/nvidia--nemotron-table-structure-v1/resources/runtime/assets/workspace/model_native/bsmock/TATR-v1.1-All/TATR-v1.1-All-msft.pth')
WEIGHT_REV = '30e6112ea20d72cc9fb8f443e34ca119d497cfe6'
WEIGHT_SHA = '9213f2dc873e5a54beeb4ddd4b5406fc9e492830abe0c44bb97f35b2229cf787'


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write(path, data):
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2, default=str, allow_nan=False))
    tmp.replace(path)


def download_weight():
    if not WEIGHT.exists():
        WEIGHT.parent.mkdir(parents=True, exist_ok=True)
        part = WEIGHT.with_suffix('.pth.part')
        for host in ['https://hf-mirror.com', 'https://huggingface.co']:
            url = f'{host}/bsmock/TATR-v1.1-All/resolve/{WEIGHT_REV}/{WEIGHT.name}'
            print(f'Download native checkpoint: {url}', flush=True)
            result = subprocess.run(['curl', '-fL', '--retry', '2', '--connect-timeout', '20',
                                     '--max-time', '600', '-C', '-', '-o', str(part), url])
            if result.returncode == 0:
                if sha(part) != WEIGHT_SHA:
                    raise RuntimeError(f'Native checkpoint checksum mismatch; retained for inspection: {part}')
                part.replace(WEIGHT)
                break
        else:
            raise RuntimeError(f'Download failed; rerun to resume {part}')
    if sha(WEIGHT) != WEIGHT_SHA:
        raise RuntimeError('Native checkpoint differs from the author-published SHA256')


def snapshot(run):
    revision = subprocess.check_output(['git', '-C', str(REPO), 'rev-parse', 'HEAD'], text=True).strip()
    assert revision == REVISION, 'Unexpected official source revision'
    subprocess.run(['git', '-C', str(REPO), 'diff', '--exit-code', 'HEAD', '--',
                    'src', 'detr', 'scripts/create_padded_dataset.py'], check=True)
    dest = run / 'official_source'
    for directory in ['src', 'detr']:
        shutil.copytree(REPO / directory, dest / directory,
                        ignore=shutil.ignore_patterns('__pycache__', '*.pyc', '.git'))
    (dest / 'scripts').mkdir()
    shutil.copy2(REPO / 'scripts/create_padded_dataset.py', dest / 'scripts/create_padded_dataset.py')
    for name in ['LICENSE', 'environment.yml']:
        shutil.copy2(REPO / name, dest / name)
    shutil.copy2(Path(__file__), run / Path(__file__).name)
    hashes = {str(p.relative_to(dest)): sha(p) for p in sorted(dest.rglob('*')) if p.is_file()}
    return dest, hashes


def prepare(case, official, stems):
    """Build a split-only input view, then execute the author's crop script verbatim."""
    raw = case / 'raw'
    tight = case / 'raw_PADDING_2'
    words = case / 'words_PADDING_2'
    marker = case / 'prepared.json'
    crop_script = official / 'scripts/create_padded_dataset.py'
    identity = {'source': str(SOURCE.resolve()), 'split': 'test', 'padding': 2,
                'crop_script_sha256': sha(crop_script), 'stems_sha256': hashlib.sha256('\n'.join(stems).encode()).hexdigest(),
                'source_filelist_sha256': sha(SOURCE / 'filelists/test_filelist.txt')}
    if marker.exists():
        saved = json.loads(marker.read_text())
        if saved['identity'] != identity:
            raise RuntimeError('Prepared dataset protocol changed; use a new output version')
        return tight, words, saved
    (raw / 'test').mkdir(parents=True, exist_ok=True)
    for link, target in [(raw / 'images', SOURCE / 'images'), (case / 'words', SOURCE / 'words')]:
        if not link.is_symlink():
            link.symlink_to(target.resolve(), target_is_directory=True)
        assert link.resolve() == target.resolve()
    for stem in stems:
        for folder, suffix in [('annotations', '.xml'), ('images', '.jpg'), ('words', '_words.json')]:
            if not (SOURCE / folder / (stem + suffix)).is_file():
                raise FileNotFoundError(SOURCE / folder / (stem + suffix))
        link = raw / 'test' / (stem + '.xml')
        if not link.is_symlink():
            link.symlink_to((SOURCE / 'annotations' / link.name).resolve())
        assert link.resolve() == (SOURCE / 'annotations' / link.name).resolve()
    print(f'Official crop preparation: {len(stems)} tables, padding=2', flush=True)
    subprocess.run([sys.executable, '-B', str(crop_script), '--pascal_data_dir', str(raw),
                    '--words_data_dir', str(case / 'words'), '--split', 'test', '--table_padding', '2'], check=True)
    (tight / 'test_filelist.txt').write_text(''.join(f'test/{s}.xml\n' for s in stems))
    # Check every transformed structure box and retained word against the source.
    from PIL import Image
    from fitz import Rect
    digest = hashlib.sha256()
    for stem in stems:
        source_xml = ET.parse(SOURCE / 'annotations' / (stem + '.xml')).getroot()
        new_xml = ET.parse(tight / 'test' / (stem + '.xml')).getroot()
        objects = source_xml.findall('object')
        tables = [o for o in objects if o.findtext('name') == 'table']
        assert len(tables) == 1, stem
        def box(obj):
            return [float(obj.findtext('bndbox/' + k)) for k in ['xmin', 'ymin', 'xmax', 'ymax']]
        crop = [round(v) + delta for v, delta in zip(box(tables[0]), [-2, -2, 2, 2])]
        shifted = new_xml.findall('object')
        assert len(objects) == len(shifted), stem
        for before, after in zip(objects, shifted):
            assert before.findtext('name') == after.findtext('name')
            expected = [v - crop[i % 2] for i, v in enumerate(box(before))]
            assert all(abs(a-b) < 1e-6 for a, b in zip(expected, box(after))), stem
        original_words = json.loads((SOURCE / 'words' / (stem + '_words.json')).read_text())
        expected_words = []
        for word in original_words:
            bbox = word['bbox']
            if Rect(bbox).intersect(crop).get_area() / Rect(bbox).get_area() >= .5:
                expected_words.append({**word, 'bbox': [v - crop[i % 2] for i, v in enumerate(bbox)]})
        assert json.loads((words / (stem + '_words.json')).read_text()) == expected_words, stem
        image_path = tight / 'images' / (stem + '.jpg')
        with Image.open(image_path) as image:
            assert image.size == (crop[2]-crop[0], crop[3]-crop[1]), stem
        for p in [SOURCE / 'images' / (stem + '.jpg'), SOURCE / 'annotations' / (stem + '.xml'),
                  SOURCE / 'words' / (stem + '_words.json'), image_path,
                  tight / 'test' / (stem + '.xml'), words / (stem + '_words.json')]:
            digest.update(str(p).encode() + sha(p).encode())
    saved = {'identity': identity, 'images': len(stems), 'source_and_derived_files_sha256': digest.hexdigest(),
             'geometry_audit': 'all image dimensions, structure box translations and retained word coordinates checked',
             'input': 'ground-truth table box crop and ground-truth words; not end-to-end detection/OCR'}
    write(marker, saved)
    return tight, words, saved


def native_eval(official, tight, words, run, device, count):
    # Native evaluation creates CPU pools after PyTorch/CUDA initialization.
    # Fresh interpreters must not inherit the parent's thread locks via fork.
    multiprocessing.set_start_method('spawn', force=True)
    import torch
    import torchvision
    sys.path[:0] = [str(official / 'src'), str(official / 'detr')]
    native = importlib.import_module('main')
    scoring = importlib.import_module('eval')
    original_get_model = native.get_model
    original_resnet18 = torchvision.models.resnet18

    def checked_model(args, target_device):
        # Initialization only: the following exact comparison requires ALL saved tensors to load.
        def no_imagenet(*a, **kw):
            kw.pop('pretrained', None)
            kw['weights'] = None
            return original_resnet18(*a, **kw)
        torchvision.models.resnet18 = no_imagenet
        try:
            result = original_get_model(args, target_device)
        finally:
            torchvision.models.resnet18 = original_resnet18
        state = torch.load(WEIGHT, map_location='cpu', weights_only=True)
        loaded = result[0].state_dict()
        assert set(state) == set(loaded), 'Incomplete native checkpoint key coverage'
        assert all(torch.equal(v.cpu(), state[k]) for k, v in loaded.items()), 'Native weight load differs'
        write(run / 'weight_load_check.json', {'all_tensors_exact': True, 'tensors': len(state)})
        return result

    native.get_model = checked_model
    sys.argv = [str(official / 'src/main.py'), '--mode', 'eval', '--data_type', 'structure',
                '--config_file', str(official / 'src/structure_config.json'), '--data_root_dir', str(tight),
                '--model_load_path', str(WEIGHT), '--table_words_dir', str(words), '--device', device,
                '--batch_size', '1', '--num_workers', '0', '--eval_pool_size', '4', '--eval_step', '10',
                '--metrics_save_filepath', str(run / 'per_table_metrics.json')]
    write(run / 'native_command.json', {'argv': sys.argv, 'effective_loader_batch': 2,
          'multiprocessing_start_method': multiprocessing.get_start_method(),
          'notes': ['Official loader uses 2 * batch_size.', 'Redundant ImageNet init disabled; all checkpoint tensors checked.',
                    'mode=eval computes core GriTS, not grits-all RawLoc/DAR diagnostics.']})
    native.main()
    samples = json.loads((run / 'per_table_metrics.json').read_text())
    expected_ids = {p.stem for p in (tight / 'test').glob('*.xml')}
    assert len(samples) == count and {s['id'] for s in samples} == expected_ids, 'Incomplete evaluation'
    summary = scoring.compute_metrics_summary(samples, 'eval')
    write(run / 'metrics.json', {'metrics': summary, 'images': count, 'full_test': count == 93834,
          'official_reference': None, 'reference_check': 'exact v1.1-All checkpoint reference still requires mapping',
          'protocol': 'native TATR-v1.1-All, official padding=2 preparation and core GriTS',
          'limitations': ['Runtime reuses PyTorch 2.5.1, not the full historical environment.',
                          'Native weights/runtime and crop changed together; not a crop-only causal ablation.']})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--max-images', type=int, default=10, help='0 = full test; positive = deterministic smoke subset')
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    if not 0 <= args.max_images <= 93834:
        parser.error('--max-images must be 0..93834')
    from importlib.metadata import version
    import fitz
    assert fitz.VersionBind == '1.21.1', 'Use run_m15_native_tight.sh with the existing overlay'
    stems = sorted(Path(line.strip()).stem for line in (SOURCE / 'filelists/test_filelist.txt').read_text().splitlines() if line.strip())
    assert len(stems) == len(set(stems)) == 93834
    if args.max_images:
        stems = stems[:args.max_images]
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / '.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        case = OUT / ('full' if not args.max_images else f'smoke{args.max_images}')
        case.mkdir(exist_ok=True)
        run = case / datetime.now().strftime('run_%Y%m%d_%H%M%S_%f')
        run.mkdir()
        print(f'Run directory: {run}', flush=True)
        state = {'state': 'preparing', 'sequence': 15, 'expected_images': len(stems),
                 'weight': str(WEIGHT), 'weight_sha256': WEIGHT_SHA, 'weight_revision': WEIGHT_REV,
                 'official_source_revision': REVISION,
                 'multiprocessing_start_method': 'spawn',
                 'software': {n: version(n) for n in ['torch', 'torchvision', 'numpy', 'Pillow', 'pycocotools']},
                 'PyMuPDF': fitz.VersionBind}
        write(run / 'status.json', state)
        try:
            official, hashes = snapshot(run)
            state['official_source_sha256'] = hashes
            download_weight()
            tight, words, prepared = prepare(case, official, stems)
            state.update(state='running', dataset=prepared)
            write(run / 'status.json', state)
            print(f'Native evaluation starts: {len(stems)} tables. Live log: {run / "native_eval.log"}', flush=True)
            with (run / 'native_eval.log').open('w', buffering=1) as log:
                with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
                    native_eval(official, tight, words, run, args.device, len(stems))
            state.update(state='complete', completed_images=len(stems))
            print((run / 'metrics.json').read_text(), flush=True)
        except BaseException as exc:
            state.update(state='failed', error=repr(exc))
            raise
        finally:
            write(run / 'status.json', state)


if __name__ == '__main__':
    main()
