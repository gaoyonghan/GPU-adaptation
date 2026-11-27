"""Explicit resource relocation; never monkey-patches filesystem APIs."""
from pathlib import Path
import os,json,sys,hashlib
MODEL_DIR=Path(__file__).resolve().parents[1]
NATIVE=MODEL_DIR/'inference/native'
OUTPUT=Path(os.environ['ZERO_SHOT_RUN_DIR'])
DATA=Path(os.environ['ZERO_SHOT_DATASET_ROOT'])
WEIGHTS=Path(os.environ['ZERO_SHOT_WEIGHTS'])
WEIGHTS_ROOT=Path(os.environ.get('ZERO_SHOT_WEIGHTS_ROOT',MODEL_DIR.parent))
MODEL_ID=json.loads((MODEL_DIR/'model_config.json').read_text())['model_id']
MAPPING=json.loads((MODEL_DIR/'inference/resource_map.json').read_text())
def weight(mid):
    return WEIGHTS if mid==MODEL_ID else WEIGHTS_ROOT/mid.replace('/','__')/'weights'
def result(mid):
    if mid!=MODEL_ID:raise ValueError('Cross-model result path is forbidden: '+mid)
    return OUTPUT/'native_artifacts'
def legacy(value):
    value=str(value)
    if value.endswith('/bin/python') or value.endswith('/bin/python3'):return Path(sys.executable)
    if 'site-packages' in value or 'runtime_overlays/openclip' in value:return NATIVE
    for old,target in sorted(MAPPING.items(),key=lambda x:len(x[0]),reverse=True):
        if value==old or value.startswith(old.rstrip('/')+'/'):
            tail=value[len(old):].lstrip('/')
            if target.startswith('weight:'):return weight(target[7:])/tail
            if target.startswith('model:'):return (MODEL_DIR if target[6:]==MODEL_ID else WEIGHTS_ROOT/target[6:].replace('/','__'))/tail
            if target.startswith('data|'):return DATA/target[5:]/tail
            root={'data':DATA,'native':NATIVE,'metadata':OUTPUT/'metadata','output':OUTPUT/'native_artifacts','work':OUTPUT/'work','tmp':OUTPUT/'tmp','weights_collection':WEIGHTS_ROOT}[target]
            if target=='native' and tail.endswith('.json') and (NATIVE/'metadata'/Path(tail).name).exists():return OUTPUT/'metadata'/Path(tail).name
            return root/tail
    if value.startswith('/tmp/'):
        return MODEL_DIR/'weights/required_fixed_assets'/value[5:]
    raise ValueError('Unmapped original resource; configure explicitly: '+value)
def legacy_str(value):return str(legacy(value))
def relative(value,root):
    value=Path(value)
    try:return value.relative_to(root)
    except ValueError:return value
def revision(path,status=False):
    path=Path(path).resolve()
    manifest=json.loads((MODEL_DIR/'inference/source_snapshot.json').read_text())
    record=manifest['repositories'][path.name]
    for name,expected in record['portable_files'].items():
        actual=path/name
        if not actual.is_file() or hashlib.sha256(actual.read_bytes()).hexdigest()!=expected:
            raise ValueError('Native source snapshot mismatch: '+str(actual))
    return '' if status else record['revision']
def prepare_metadata():
    target=OUTPUT/'metadata';target.mkdir(parents=True,exist_ok=True)
    def convert(value):
        if isinstance(value,dict):return {k:convert(v) for k,v in value.items()}
        if isinstance(value,list):return [convert(v) for v in value]
        if isinstance(value,str) and value.startswith(('/data/zhoujianjie/','/home/dell/Projects/hengxiang/')):
            try:return str(legacy(value))
            except ValueError:return value
        return value
    for f in (NATIVE/'metadata').glob('*.json'):
        data=json.loads(f.read_text())
        # Reverse mappings retain original keys but resolve the actual external resources.
        data=convert(data)
        (target/f.name).write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n')
    for name in ['work','tmp','native_artifacts']: (OUTPUT/name).mkdir(exist_ok=True)
