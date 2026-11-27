"""Record imported top-level dependency versions once; no distribution-wide scan."""
import atexit,os,sys,json
from pathlib import Path
def capture():
    output=os.environ.get('ZERO_SHOT_RUN_DIR')
    if not output:return
    import importlib.metadata as metadata
    aliases={'PIL':'Pillow','sklearn':'scikit-learn','open_clip':'open_clip_torch','cv2':'opencv-python','yaml':'PyYAML','cn_clip':'cn-clip','huggingface_hub':'huggingface-hub','onnxruntime':'onnxruntime-gpu'}
    packages={};module_paths={}
    for name in sorted({key.split('.')[0] for key in list(sys.modules)}):
        module=sys.modules.get(name);path=str(getattr(module,'__file__','') or '')
        if not any(part in path for part in ('site-packages','dist-packages','dependency_overlays')):continue
        module_paths[name]=path
        version=getattr(module,'__version__',None)
        if version is None:
            try:version=metadata.version(aliases.get(name,name))
            except metadata.PackageNotFoundError:continue
        packages[aliases.get(name,name)]=str(version)
    torch=sys.modules.get('torch');device={}
    if torch is not None:
        device={'torch':str(torch.__version__),'cuda_runtime':torch.version.cuda,'cpu_threads':torch.get_num_threads(),'cuda_initialized':torch.cuda.is_initialized()}
        if torch.cuda.is_initialized():device['gpu']=torch.cuda.get_device_name(0)
    (Path(output)/('environment_actual_'+str(os.getpid())+'.json')).write_text(json.dumps({'python':sys.version,'executable':sys.executable,'packages':packages,'module_paths':module_paths,'device':device,'scope':'imported top-level packages; full observed pins are in environment manifests'},indent=2))
atexit.register(capture)
