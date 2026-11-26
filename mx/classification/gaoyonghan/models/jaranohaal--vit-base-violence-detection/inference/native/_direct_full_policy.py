"""Opt-in direct-full orchestration; archived source files are never modified."""
import ast
import hashlib
import json
from pathlib import Path

VERSION_MODELS = {71, 72, 74, 84, 87, 89, 91, 95}
SMOKE_MESSAGES = {
    'First run smoke', 'Run smoke first', 'First run 10-image smoke',
    'First run --max-images 10', 'First complete --max-images 10',
    'Run --max-images 10 successfully before full evaluation',
}

CATEGORY1_FILES = {
    15: {'run_m15_native_tight.py'},
    30: {'run_batch8.py'},
    33: {'run_m33x_native.py'},
    75: {'run_m75_fer2013.py'},
    81: {'run_m81_fer2013.py'},
    92: {'run_m92.py'},
    94: {'run_m94.py'},
    99: {'run_m99.py'},
}

def record_environment(expected, actual=None):
    """Record metadata differences; actual imports expose missing runtime dependencies."""
    import importlib.metadata
    import platform
    import sys
    if actual is None:
        actual = {}
        for name in expected:
            try:
                actual[name] = importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError:
                actual[name] = None
    differences = {name: {'expected': value, 'actual': actual.get(name)}
                   for name, value in expected.items() if actual.get(name) != value}
    print('DIRECT_FULL_ENVIRONMENT ' + json.dumps({
        'python': platform.python_version(), 'executable': sys.executable,
        'expected': expected, 'actual': actual, 'differences': differences,
        'policy': 'record_only_use_current_environment'}, ensure_ascii=False), flush=True)
    return actual

def patch_category1(path, sequence):
    """Patch only the eight approved entrypoints, including m92's imported helper."""
    original = path.read_text()
    if '# direct-full-category1-v1' in original:
        return None
    tree = ast.parse(original)
    lines = original.splitlines(keepends=True)
    edits = []
    def replace(node, text, kind):
        indent = ' ' * node.col_offset
        edits.append((node.lineno-1, node.end_lineno,
                      ''.join(indent + line + '\n' for line in text.splitlines()), kind))
    def gate(node, expected, actual=None):
        call = f'record_environment({expected}' + (f', actual={actual}' if actual else '') + ')'
        replace(node, 'from _direct_full_policy import record_environment\n' + call, 'record_environment_difference')
    for node in ast.walk(tree):
        if isinstance(node, ast.Assert):
            test = ast.unparse(node.test)
            if sequence == 15 and test == "fitz.VersionBind == '1.21.1'":
                gate(node, "{'PyMuPDF': '1.21.1'}", "{'PyMuPDF': fitz.VersionBind}")
            elif sequence == 33 and test == "version('ultralytics') == '8.3.123'":
                gate(node, "{'ultralytics': '8.3.123'}")
            elif sequence == 33 and test == "version('torch') == '2.5.1+cu121' and version('torchvision') == '0.20.1+cu121'":
                gate(node, "{'torch': '2.5.1+cu121', 'torchvision': '0.20.1+cu121'}")
            elif sequence == 30 and isinstance(node.msg, ast.Constant) and node.msg.value == 'Run --max-images 16 first':
                replace(node, 'print("Direct full evaluation: no prior 16-image smoke prerequisite; smoke equivalence not established.", flush=True)', 'omit_prior_smoke_prerequisite')
        if sequence in {75, 81, 92, 94, 99}:
            expected = 'expected' if sequence == 75 else "spec['versions']"
            if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name) and node.targets[0].id == 'versions':
                if isinstance(node.value, ast.DictComp) and 'importlib.metadata.version' in ast.unparse(node.value):
                    replace(node, f'from _direct_full_policy import record_environment\nversions = record_environment({expected})', 'record_environment_difference')
            if isinstance(node, ast.If) and ast.unparse(node.test) == f'versions != {expected}' and not node.orelse and len(node.body) == 1 and isinstance(node.body[0], ast.Raise):
                replace(node, 'pass  # Version differences are recorded above; continue with installed packages.', 'omit_exact_version_gate')
    required = 2 if sequence in {33, 75, 81, 92, 94, 99} else 1
    if len(edits) != required:
        raise ValueError(f'Unexpected category1 gate layout: model={sequence}, file={path}, edits={len(edits)}, expected={required}')
    for start, end, replacement, _ in sorted(edits, reverse=True):
        lines[start:end] = [replacement]
    updated = ''.join(lines) + '\n# direct-full-category1-v1\n'
    ast.parse(updated)
    path.write_text(updated)
    return dict(file=str(path), before_sha256=hashlib.sha256(original.encode()).hexdigest(),
                after_sha256=hashlib.sha256(updated.encode()).hexdigest(),
                changes=[edit[3] for edit in edits])

def patch_m15_source_revision(path):
    """Use the existing packaged source inventory for M15's official code."""
    original = path.read_text()
    old = """    revision = subprocess.check_output(['git', '-C', str(REPO), 'rev-parse', 'HEAD'], text=True).strip()
    assert revision == REVISION, 'Unexpected official source revision'
    subprocess.run(['git', '-C', str(REPO), 'diff', '--exit-code', 'HEAD', '--',
                    'src', 'detr', 'scripts/create_padded_dataset.py'], check=True)"""
    new = """    from portable_source_revision import verified_source_revision
    revision = verified_source_revision(REPO, expected=REVISION)"""
    if new in original and old not in original:
        return None
    if original.count(old) != 1:
        raise ValueError(f'Unexpected M15 source revision check: {path}')
    updated = original.replace(old, new)
    ast.parse(updated)
    path.write_text(updated)
    return dict(file=str(path), before_sha256=hashlib.sha256(original.encode()).hexdigest(),
                after_sha256=hashlib.sha256(updated.encode()).hexdigest(),
                changes=['verify_packaged_source_without_git'])

def patch_runtime_interfaces(path, sequence):
    """Adapt archived entrypoints after extraction, independent of GPU vendor."""
    replacements = {
        15: ('    return dest, hashes',
            r"""    # Adapt only the disposable scoring copy, including spawned score workers.
    # PyMuPDF accepts coordinate tuples across versions; the GriTS math is unchanged.
    grits = dest / 'src/grits.py'
    original = grits.read_text()
    anchor = '    intersection = Rect(bbox1).intersect(bbox2)'
    if original.count(anchor) != 1:
        raise RuntimeError('Unexpected official GriTS rectangle code')
    adapted = original.replace(anchor,
        '    bbox1, bbox2 = tuple(bbox1), tuple(bbox2)\n' + anchor)
    compile(adapted, str(grits), 'exec')
    grits.write_text(adapted)
    write(run / 'scoring_compatibility.json', {
        'adapter': 'coordinate sequences converted to tuples before PyMuPDF Rect',
        'scope': 'scoring only; official GriTS arithmetic unchanged',
        'original_sha256': hashes['src/grits.py'], 'runtime_sha256': sha(grits),
    })
    return dest, hashes"""),
        27: ("    command = [str(Path(sys.executable).parent / 'vis4d'), 'test', '--config', str(run / 'm27_native_config.py'), '--gpus', '1']",
            """    # Load the installed console entry point with this interpreter. A pip --target
    # installation can place its executable outside the virtualenv's bin directory.
    entry = ("from importlib.metadata import distribution; import sys; "
             "ep = next(e for e in distribution('vis4d').entry_points "
             "if e.group == 'console_scripts' and e.name == 'vis4d'); "
             "sys.exit(ep.load()())")
    command = [sys.executable, '-c', entry, 'test', '--config', str(run / 'm27_native_config.py'), '--gpus', '1']"""),
        83: ('HERE = Path(__file__).resolve().parent',
            """HERE = Path(__file__).resolve().parent

def restore_onnx_export_imports():
    # Optimum imports exporter helpers even when loading an existing ONNX file.
    # Reuse relocated PyTorch implementations only within this model's process.
    import importlib
    legacy = importlib.import_module('torch.onnx.symbolic_opset14')
    names = ('_attention_scale', '_causal_attention_mask', '_onnx_symbolic',
             '_type_utils', 'jit_utils', 'symbolic_helper')
    missing = [name for name in names if not hasattr(legacy, name)]
    if missing:
        relocated = importlib.import_module(
            'torch.onnx._internal.torchscript_exporter.symbolic_opset14')
        for name in missing:
            setattr(legacy, name, getattr(relocated, name))
    print('Optimum import compatibility: ' + json.dumps({'restored_names': missing,
          'existing_onnx_graph_unchanged': True, 'scope': 'current process only'}), flush=True)

restore_onnx_export_imports()"""),
    }
    if sequence not in replacements:
        return None
    original = path.read_text()
    marker = '# direct-full-runtime-interface-v1'
    if marker in original:
        return None
    anchor, replacement = replacements[sequence]
    if original.count(anchor) != 1:
        raise ValueError(f'Unexpected model {sequence} interface code: {path}')
    updated = original.replace(anchor, replacement) + '\n' + marker + '\n'
    ast.parse(updated)
    path.write_text(updated)
    return dict(file=str(path), before_sha256=hashlib.sha256(original.encode()).hexdigest(),
                after_sha256=hashlib.sha256(updated.encode()).hexdigest(),
                changes=[{15: 'adapt_scoring_rectangle_input',
                          27: 'resolve_current_python_console_entrypoint',
                          83: 'reuse_relocated_onnx_export_helpers_in_process'}[sequence]])

def patch_timing_consistency(path, sequence):
    """Keep full accuracy evaluation independent of timing reproducibility."""
    if sequence not in {179, 197, 198}:
        return None
    original = path.read_text()
    if '# direct-full-timing-record-v1' in original:
        return None
    runner = next(node for node in ast.parse(original).body
                  if isinstance(node, ast.FunctionDef) and node.name == 'hf_run')
    body = ast.get_source_segment(original, runner)
    old = "                            raise ValueError('Enabling timing changed predictions beyond the frozen tolerance')"
    new = """                            diagnostic = {
                                'sequence': spec['sequence'], 'segment': args.segment,
                                'policy': 'record_difference_continue_accuracy_evaluation',
                                'timing_eligible_for_comparison': False,
                                'checked_images': len(probs),
                                'max_absolute_probability_error': float(np.abs(baseline-probs).max()),
                                'top1_mismatches': int(np.count_nonzero(baseline.argmax(-1) != probs.argmax(-1))),
                                'rtol': 1e-5, 'atol': 1e-7,
                            }
                            (out / f'timing-consistency-{args.segment}.json').write_text(
                                json.dumps(diagnostic, ensure_ascii=False, indent=2))
                            print('Timing comparison requires review; continuing full accuracy evaluation: '
                                  + json.dumps(diagnostic), flush=True)"""
    anchor = "    with (out/'predictions.jsonl').open('w') as f:"
    metadata = """    timing['eligible_for_speed_comparison_this_segment'] = instrumentation_verified is True
    timing['consistency_policy'] = 'record_difference_continue_accuracy_evaluation'
"""
    if body.count(old) != 1 or body.count(anchor) != 1:
        raise ValueError(f'Unexpected timing consistency code: {path}')
    updated = original.replace(body, body.replace(old, new).replace(anchor, metadata + anchor), 1)
    updated += '\n# direct-full-timing-record-v1\n'
    ast.parse(updated)
    path.write_text(updated)
    return dict(file=str(path), before_sha256=hashlib.sha256(original.encode()).hexdigest(),
                after_sha256=hashlib.sha256(updated.encode()).hexdigest(),
                changes=['record_timing_difference_without_blocking_accuracy',
                         'mark_unverified_timing_ineligible_for_speed_comparison'])

def patch(path, sequence, record_versions=True):
    original = path.read_text()
    lines = original.splitlines(keepends=True)
    tree = ast.parse(original)
    edits = []
    for node in ast.walk(tree):
        if record_versions and sequence in VERSION_MODELS and isinstance(node, ast.FunctionDef) and node.name in {'environment', 'check_environment'}:
            body = ast.get_source_segment(original, node)
            expected = 'SPEC["environment"]' if sequence == 72 else 'MANIFEST["versions"]'
            if 'importlib.metadata.version' not in body or not any(isinstance(n, ast.Raise) for n in ast.walk(node)):
                raise ValueError(f'Unexpected environment function for {sequence}')
            suffix = '\n    actual["python"] = __import__("platform").python_version()' if sequence == 72 else ''
            new = f'def {node.name}():\n    from metax_runtime import check_archived_environment\n    actual = check_archived_environment({expected}){suffix}\n    return actual\n'
            edits.append((node.lineno-1, node.end_lineno, new, 'record_environment_difference'))
        if not isinstance(node, ast.If):
            continue
        segment = ast.get_source_segment(original, node)
        messages = {n.value for n in ast.walk(node) if isinstance(n, ast.Constant) and isinstance(n.value, str)}
        indent = ' ' * node.col_offset
        if messages & SMOKE_MESSAGES and not node.orelse and len(node.body) == 1 and isinstance(node.body[0], ast.Raise):
            edits.append((node.lineno-1, node.end_lineno, indent + 'print("Direct full evaluation: no prior smoke prerequisite; smoke equivalence not established.", flush=True)\n', 'omit_prior_smoke_prerequisite'))
        # This is the final elif branch; retain the surrounding preflight/smoke branches.
        if 'Run the matching 10-image smoke phase first' in messages and ast.unparse(node.test) == "args.phase == 'full'":
            first = lines[node.lineno-1]
            if first.lstrip().startswith("elif args.phase=='full':"):
                new = indent + "elif args.phase=='full':\n" + indent + '    print("Direct full evaluation: no prior smoke prerequisite; smoke equivalence not established.", flush=True)\n'
                edits.append((node.lineno-1,node.end_lineno,new,'omit_prior_smoke_prerequisite'))
        if sequence == 174 and not node.orelse and ast.unparse(node.test) == "args.phase == 'full'":
            if 'A completed smoke run is required before the full run' in messages or 'Full predictions differ from smoke run' in messages:
                edits.append((node.lineno-1,node.end_lineno,indent+'print("Direct full evaluation: smoke comparison not performed; predictions and metrics remain subject to review.", flush=True)\n','omit_cross_smoke_comparison'))
    # Nested candidate branches cannot be changed twice.
    edits.sort(key=lambda e:(e[0],-e[1]))
    accepted=[]
    for edit in edits:
        if accepted and edit[0] < accepted[-1][1]:
            continue
        accepted.append(edit)
    for start,end,replacement,_ in reversed(accepted):
        lines[start:end] = [replacement]
    updated=''.join(lines)
    ast.parse(updated)
    if updated != original:
        path.write_text(updated)
        return dict(file=str(path),before_sha256=hashlib.sha256(original.encode()).hexdigest(),after_sha256=hashlib.sha256(updated.encode()).hexdigest(),changes=[e[3] for e in accepted])
    return None

def apply(overlay, output, sequence, root, primary):
    import shutil
    receipts=[]
    receipt = patch_runtime_interfaces(Path(primary), sequence)
    if receipt:
        receipts.append(receipt)
    if sequence in {179, 197, 198}:
        receipt = patch_timing_consistency(Path(primary), sequence)
        if receipt:
            receipts.append(receipt)
    if sequence in CATEGORY1_FILES:
        for name in sorted(CATEGORY1_FILES[sequence]):
            path = Path(overlay) / name
            if not path.is_file():
                raise FileNotFoundError(f'Category1 evaluator is missing: {path}')
            receipt = patch_category1(path, sequence)
            if receipt:
                receipts.append(receipt)
        if sequence == 15:
            receipt = patch_m15_source_revision(Path(overlay) / 'run_m15_native_tight.py')
            if receipt:
                receipts.append(receipt)
            for name in ['portable_source_revision.py', 'source_revisions.json']:
                shutil.copy2(Path(root) / 'backends/source_revision' / name, Path(overlay) / name)
        shutil.copy2(Path(__file__), Path(overlay) / '_direct_full_policy.py')
    # Only evaluator entry files, not model implementations or dependencies.
    names={'run_model.py','run_proxy_174.py','run_widerface.py','run_qualcomm_coco.py','run_cmarkea.py','run_pp_layout.py','run_anzhc_text.py','run_m95_test.py'}
    for path in sorted(Path(overlay).glob('*.py')):
        if path.name in names or path == Path(primary):
            receipt=patch(path,sequence,record_versions=path == Path(primary))
            if receipt:receipts.append(receipt)
    if sequence in VERSION_MODELS:
        for name in ['metax_runtime.py','allowed_metax_environment.json']:
            shutil.copy2(Path(root)/'backends/environment_probe'/name,Path(overlay)/name)
    Path(output).mkdir(parents=True,exist_ok=True)
    (Path(output)/'direct_full_policy.json').write_text(json.dumps(dict(sequence=sequence,policy='record environment then direct full inference',prior_smoke_required=False,smoke_equivalence='not established',patches=receipts),ensure_ascii=False,indent=2)+'\n')
