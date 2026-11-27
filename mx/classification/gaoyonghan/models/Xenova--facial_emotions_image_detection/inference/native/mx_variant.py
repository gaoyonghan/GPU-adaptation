"""This model's previously executed MX PyTorch variant; no central model dispatcher."""
import argparse, hashlib, importlib.metadata, importlib.util, json, os, sys, time
from pathlib import Path

def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()

def save(path, value):
    path = Path(path)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(path)

def load_module(path):
    sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location('variant_backend', path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module

class DeviceEvidence:
    """Check real parameters and first forward tensors; never infer GPU use from CLI."""
    def __init__(self, out, device):
        self.out, self.device, self.evidence = out, device, None

    def attach(self, model):
        import torch
        def tensors(value):
            if torch.is_tensor(value):
                yield value
            elif isinstance(value, dict):
                for item in value.values():
                    yield from tensors(item)
            elif isinstance(value, (list, tuple)):
                for item in value:
                    yield from tensors(item)
        def before(module, args, kwargs):
            parameter = next(module.parameters())
            inputs = list(tensors((args, kwargs)))
            expected = torch.device(self.device)
            if parameter.device != expected or not inputs or any(t.device != expected for t in inputs):
                raise RuntimeError('Actual model/input device differs from requested GPU')
            self.evidence = dict(requested=self.device, parameter_device=str(parameter.device),
                                 parameter_dtype=str(parameter.dtype),
                                 input_devices=sorted({str(t.device) for t in inputs}))
        def after(module, args, kwargs, output):
            outputs = list(tensors(output))
            if not outputs or any(t.device != torch.device(self.device) for t in outputs):
                raise RuntimeError('Actual forward output is not on requested GPU')
            self.evidence.update(output_devices=sorted({str(t.device) for t in outputs}),
                                 gpu_name=torch.cuda.get_device_name(self.device), verified=True)
            save(self.out / 'device_evidence.json', self.evidence)
            for handle in handles:
                handle.remove()
        handles = [model.register_forward_pre_hook(before, with_kwargs=True),
                   model.register_forward_hook(after, with_kwargs=True)]

def rewritten(text, cfg):
    for old, new in cfg.get('path_rewrites', {}).items():
        text = text.replace(old, new)
    return text

def emotions(project, cfg, phase, device, checkpoint, receipt, out, evidence):
    backend = load_module(Path(__file__).parent / 'mx_backend/run_model.py')
    folder = Path(backend.__file__).parent
    manifest = json.loads(rewritten((folder / 'models.json').read_text(), cfg))
    spec = next(s for s in manifest['models'] if s['sequence'] == 173)
    original = json.loads(json.dumps(spec))
    # Rewrite only container paths in the frozen index; keep order, labels and membership.
    source_index = folder / 'indices/173.jsonl'
    unmodified = json.loads((folder / 'models.json').read_text())
    expected = next(s for s in unmodified['models'] if s['sequence'] == 173)['index_sha256']
    if sha(source_index) != expected:
        raise ValueError('Original frozen index changed')
    index = out / 'index.jsonl';index.write_text(rewritten(source_index.read_text(), cfg))
    spec.update(index=str(index), index_sha256=sha(index), model_path=str(checkpoint),
        files={n: {'sha256': x['sha256'], 'size': x['size']} for n, x in receipt['files'].items()},
        backend='shared_hf', dtype='float32', weight='model.safetensors',
        source_repository=receipt['repository'], revision=receipt['revision'],
        fidelity='explicit_checkpoint_variant',
        known_differences=['Upstream float PyTorch checkpoint replaces Xenova q8 ONNX; numeric equivalence not assumed'])
    rows = backend.check(spec, manifest)
    config = json.loads((checkpoint / 'config.json').read_text())
    if config['id2label'] != spec['label_mapping']:
        raise ValueError('Upstream checkpoint label order differs from frozen dataset mapping')
    if phase != 'full':
        rows = backend.smoke_rows(rows)
    from classification_capture import snapshot_gt
    snapshot_gt(out, rows)
    save(out / 'protocol.json', dict(original=original, selected=spec,
        original_index_sha256=expected, backend_sha256=sha(Path(backend.__file__)),
        smoke_equivalence='not established; direct full inference'))
    from transformers import AutoModelForImageClassification
    original_load = AutoModelForImageClassification.from_pretrained
    def observed(*args, **kwargs):
        result = original_load(*args, **kwargs)
        evidence.attach(result[0] if isinstance(result, tuple) else result)
        return result
    AutoModelForImageClassification.from_pretrained = observed
    args = argparse.Namespace(device=device, batch_size=1, warmup=10, workers=2,
                              phase=phase, segment=str(time.time_ns()))
    try:
        timing, metrics = backend.hf_run(spec, rows, args, out)
    finally:
        AutoModelForImageClassification.from_pretrained = original_load
    timing['performance_comparison_valid'] = 'requires matching NVIDIA variant and isolated timing run'
    save(out / 'timing.json', timing)
    if metrics is not None:
        save(out / 'metrics.json', metrics)

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--phase',choices=['full','smoke'],default='full')
    parser.add_argument('--device',default='cuda:0')
    parser.add_argument('--output-root',type=Path,required=True)
    args=parser.parse_args()
    model=Path(__file__).resolve().parents[2]
    sys.path.insert(0,str(model/'inference/output_format'))
    project=model/'resources/runtime/project'
    cfg=json.loads((Path(__file__).parent/'mx_variant_config.json').read_text())
    definition=json.loads((Path(__file__).parent/'mx_variant_definition.json').read_text())
    checkpoint=Path(cfg['model_root'])/definition['directory']
    out=args.output_root;out.mkdir(parents=True,exist_ok=True)
    receipt=(json.loads((checkpoint/'weight_source.json').read_text()) if (checkpoint/'weight_source.json').is_file() else definition)
    evidence=DeviceEvidence(out,args.device)
    state={'sequence':173,'variant':'pytorch_fp32','phase':args.phase,'state':'running','source':'existing MX validated variant','new_validation':False}
    save(out/'variant_status.json',state)
    try:
        import torch
        if not torch.cuda.is_available():raise RuntimeError('MX GPU is unavailable; no CPU fallback')
        torch.cuda.set_device(args.device)
        if 173<=70:
            import detection as capture
            capture.install_scoring_capture()
        emotions(project, cfg, args.phase, args.device, checkpoint, receipt, out, evidence)
        state['state']='completed'
    except BaseException as exc:
        state.update(state='failed',error=repr(exc));raise
    finally:
        save(out/'variant_status.json',state)

if __name__=='__main__':main()
