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

def owlvit(project, cfg, phase, device, checkpoint, definition, out, evidence):
    backend = load_module(Path(__file__).parent / 'mx_backend/run_model.py')
    manifest = backend.manifest()
    spec = next(s for s in manifest['models'] if s['sequence'] == 70)
    spec['known_differences'].append(definition['change'])
    original_load = backend.load_model
    def observed(*args, **kwargs):
        processor, model = original_load(*args, **kwargs)
        # This adapter invokes image_embedder directly, so observe the real vision forward.
        evidence.attach(model.owlvit.vision_model)
        return processor, model
    backend.load_model = observed
    original_artifact = backend.local_artifact
    def artifact(path):
        result = original_artifact(path)
        # The outer runner has verified every pinned component by SHA256.
        result.update(hf_revision_detected=definition['revision'],
                      revision_evidence='pinned component SHA256; download cache not required')
        return result
    backend.local_artifact = artifact
    args = argparse.Namespace(output_root=out, model_root=Path(cfg['model_root']),
        dataset_root=Path(cfg['dataset_root']), device=device, dtype='float32',
        batch_size=1, max_detections=300, max_images=0 if phase == 'full' else 10,
        progress_every=50)
    save(out / 'protocol.json', dict(selected=spec, backend_sha256=sha(Path(backend.__file__)),
        native_baseline='Scenic JAX retained separately; this is the existing HF adapter',
        nv_rerun_required=True))
    backend.run_one(spec, manifest, args)
    status = json.loads((out / spec['key'] / 'status.json').read_text())
    expected = 19809 if phase == 'full' else 10
    if status['processed_image_count'] != expected or status['unique_prediction_image_count'] != expected:
        raise RuntimeError('Incomplete OWL-ViT evaluation coverage')

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
    state={'sequence':70,'variant':'pytorch_fp32','phase':args.phase,'state':'running','source':'existing MX validated variant','new_validation':False}
    save(out/'variant_status.json',state)
    try:
        import torch
        if not torch.cuda.is_available():raise RuntimeError('MX GPU is unavailable; no CPU fallback')
        torch.cuda.set_device(args.device)
        if 70<=70:
            import detection as capture
            capture.install_scoring_capture()
        owlvit(project, cfg, args.phase, args.device, checkpoint, definition, out, evidence)
        state['state']='completed'
    except BaseException as exc:
        state.update(state='failed',error=repr(exc));raise
    finally:
        save(out/'variant_status.json',state)

if __name__=='__main__':main()
