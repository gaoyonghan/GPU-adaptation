"""Opt-in enforcement: validation may read external weights/data, never old code."""
import os,sys
from pathlib import Path
# Bound CPU helper threads only for the isolated delivery diagnostic.
# This does not change precision, prompts, preprocessing, or default full runs.
if int(os.environ.get('ZERO_SHOT_SMOKE_LIMIT','0')):
    os.environ.setdefault('OMP_NUM_THREADS','4')
    os.environ.setdefault('MKL_NUM_THREADS','4')
    os.environ.setdefault('TOKENIZERS_PARALLELISM','false')
roots=[Path(p).resolve() for p in os.environ.get('ZERO_SHOT_FORBIDDEN_SOURCE_ROOTS','').split(os.pathsep) if p]
if roots:
    data=Path(os.environ['ZERO_SHOT_DATASET_ROOT']).resolve()
    weights=Path(os.environ['ZERO_SHOT_WEIGHTS']).resolve()
    collection=Path(os.environ.get('ZERO_SHOT_WEIGHTS_ROOT',str(weights.parent.parent))).resolve()
    def audit(event,args):
        if event!='open' or not args or not isinstance(args[0],(str,bytes)):return
        path=Path(os.fsdecode(args[0])).resolve()
        if not any(path.is_relative_to(root) for root in roots):return
        if path.is_relative_to(data) or path.is_relative_to(weights):return
        if path.is_relative_to(collection):
            parts=path.relative_to(collection).parts
            if len(parts)>=2 and parts[1]=='weights':return
        raise PermissionError('Old engineering source access forbidden: '+str(path))
    sys.addaudithook(audit)
    output=Path(os.environ['ZERO_SHOT_RUN_DIR'])
    (output/('source_isolation_'+str(os.getpid())+'.json')).write_text(__import__('json').dumps({'enabled':True,'blocked_source_roots':[str(p) for p in roots],'external_weights':str(weights),'external_dataset':str(data),'pid':os.getpid()},indent=2))
