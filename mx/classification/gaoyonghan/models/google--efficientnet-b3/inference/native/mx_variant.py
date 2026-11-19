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

def efficientnet(project, cfg, phase, device, checkpoint, out, evidence):
    backend = load_module(Path(__file__).parent / 'mx_backend/run_model.py')
    backend.MANIFEST = json.loads(rewritten((Path(backend.__file__).parent / 'models.json').read_text(), cfg))
    spec = next(s for s in backend.MANIFEST['models'] if s['sequence'] == 88)
    spec['known_differences'].append('Existing HF PyTorch route; differs from selected Keras preprocessing')
    runtime = Path(cfg['classification8_runtime_root'])
    from imagenet_input import frozen_directory_index
    # Reuse the existing directory adapter and prove the same full sample/label/image bytes.
    with backend.locked(runtime / 'data/imagenet/.prepare.lock'):
        data = frozen_directory_index(backend.MANIFEST['imagenet_zip'], Path(cfg['dataset_root']) / 'imagenet1k_val',
            backend.MANIFEST['datasets']['imagenet'], backend.canonical, backend.save)
    mapping = backend.label_mapping(spec, data, checkpoint)
    from classification_capture import snapshot_gt
    snapshot_gt(out, backend.select_rows(data, 0 if phase == 'full' else 10), mapping)
    save(out / 'protocol.json', dict(selected=spec, dataset_fingerprint=data['fingerprint'],
        backend_sha256=sha(Path(backend.__file__)),
        native_baseline='Keras/Google preprocessing retained separately', nv_rerun_required=True))
    from transformers import AutoModelForImageClassification
    original_load = AutoModelForImageClassification.from_pretrained
    def observed(*args, **kwargs):
        result = original_load(*args, **kwargs)
        evidence.attach(result[0] if isinstance(result, tuple) else result)
        return result
    AutoModelForImageClassification.from_pretrained = observed
    try:
        records = backend.hf_inference(spec, data, checkpoint, out,
            0 if phase == 'full' else 10, device, mapping)
    finally:
        AutoModelForImageClassification.from_pretrained = original_load
    expected = 50000 if phase == 'full' else 10
    if len(records) != expected:
        raise RuntimeError('Incomplete EfficientNet evaluation coverage')
    if phase == 'full':
        save(out / 'metrics.json', dict(metrics=backend.classification_metrics(records, spec),
            evaluation_images=len(records), dataset_relation=spec['dataset_relation'],
            predictions_sha256=sha(out / 'predictions.jsonl')))

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
    state={'sequence':88,'variant':'pytorch_fp32','phase':args.phase,'state':'running','source':'existing MX validated variant','new_validation':False}
    save(out/'variant_status.json',state)
    try:
        import torch
        if not torch.cuda.is_available():raise RuntimeError('MX GPU is unavailable; no CPU fallback')
        torch.cuda.set_device(args.device)
        if 88<=70:
            import detection as capture
            capture.install_scoring_capture()
        efficientnet(project, cfg, args.phase, args.device, checkpoint, out, evidence)
        state['state']='completed'
    except BaseException as exc:
        state.update(state='failed',error=repr(exc));raise
    finally:
        save(out/'variant_status.json',state)

if __name__=='__main__':main()
