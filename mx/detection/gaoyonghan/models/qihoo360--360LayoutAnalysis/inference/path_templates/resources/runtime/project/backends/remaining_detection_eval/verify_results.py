"""Independent audit of complete result artifacts, without rerunning inference."""
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path

HERE=Path(__file__).resolve().parent

def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()

def finite(value):
    if isinstance(value,float):assert math.isfinite(value),'Nonfinite JSON number'
    elif isinstance(value,dict):
        for v in value.values():finite(v)
    elif isinstance(value,list):
        for v in value:finite(v)

def main():
    progress=json.loads((HERE/'progress.json').read_text());models=progress['models']
    assert len(models)==25 and len({m['model_id'] for m in models})==25
    audit=[]
    for m in models:
        for row in m['results']:
            if not row['valid_complete']:continue
            root=Path(row['result_directory']);status=json.loads((root/'status.json').read_text());score=json.loads((root/'metrics.json').read_text());finite(score)
            assert not status.get('invalid_results')
            predictions=root/'predictions.jsonl'
            if not predictions.exists():predictions=Path(status['protocol']['predictions'])
            seen=set();count=0;hasher=hashlib.sha256()
            with predictions.open('rb') as stream:
                for line in stream:
                    hasher.update(line);record=json.loads(line);finite(record)
                    key=record.get('image_id')
                    if key is None:key=(record['event'],record['filename'])
                    assert key not in seen,(root,key)
                    seen.add(key);count+=1
            expected=status.get('total_images') or score.get('evaluation_images')
            assert count==expected==status['completed_images'],(root,count,expected,status['completed_images'])
            if m['sequence']==26:assert len(list((root/'wider_predictions').glob('*/*.txt')))==3226
            if m['sequence']==39:
                assert hasher.hexdigest()==status['protocol']['predictions_sha256']
                assert 'polygon_segm_AP' in score and 'reading_order' in score
            if m['sequence']==38:assert 'reading_order' in score
            if m['sequence']==42:assert 'semantic_F1' in score and 'bbox' in score
            audit.append({'sequence':m['sequence'],'model_id':m['model_id'],'task':row['task'],'result_directory':str(root),
              'unique_predictions':count,'predictions_file':str(predictions),'predictions_sha256':hasher.hexdigest(),
              'metrics_sha256':sha(root/'metrics.json'),'status_sha256':sha(root/'status.json'),'finite_metrics_and_predictions':True})
            print(m['sequence'],row['task'],count,'verified',flush=True)
    probe=[]
    for seq in [8,31]:
        path=Path('__MODEL_ROOT__/resources/runtime/assets/workspace/results/remaining25_private_probe')/f'{seq}.json';data=json.loads(path.read_text())
        assert len(data['weights'])==(21 if seq==8 else 3)
        assert all(w['state']=='smoke_passed' and w['completed_images']==10 for w in data['weights'])
        probe.append({'sequence':seq,'weights_smoke_passed':len(data['weights']),'record_sha256':sha(path),'accuracy_evaluation':False})
    counts=Counter(m['state'] for m in models);assert counts=={'complete':21,'partial_complete':3,'blocked':1},counts
    assert len(audit)==30,len(audit)
    output={'verified_at':datetime.now(timezone.utc).isoformat(),'scope':25,'parent_state_counts':dict(counts),
       'complete_benchmark_runs':len(audit),'records':audit,'private_weight_smokes':probe,
       'meaning':'Artifact integrity and complete frozen benchmark coverage verified. This does not remove documented data/protocol differences or turn partial task coverage into model completion.'}
    (HERE/'verification.json').write_text(json.dumps(output,ensure_ascii=False,indent=2)+'\n')
    print('Verified 30 benchmark runs; 21 complete models, 3 partial, 1 blocked.',flush=True)

if __name__=='__main__':main()
