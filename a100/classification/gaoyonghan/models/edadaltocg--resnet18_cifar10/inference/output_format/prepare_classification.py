"""Convert this model's captured native records to the shared scoring protocol."""
import json
from pathlib import Path
from common import model_spec,save
from output_format.classification import require,read,file_hash,publish,load_bundle

def finalize_run(root,sequence,phase,successful=None):
    root=Path(root).resolve();spec=model_spec()
    require(sequence==spec['sequence'],'Unexpected model sequence')
    require(successful is True,'A failed native launch cannot publish a completed run')
    native=root/phase/('m%03d'%sequence);status=read(native/'status.json')
    require(status.get('state')=='complete' and status.get('phase')==phase,'Native run is not complete for the requested phase')
    require(status.get('sequence')==sequence,'Native status belongs to another model')
    report=dict(sequence=sequence,phase=phase,state='exported',items=[],inference_executed=False)
    for name in spec['native_script'].get('branches',['default']):
        folder=native if name=='default' else native/name
        capture=read(folder/'classification_capture.json')
        require(capture.get('state')=='complete' and capture.get('phase')==phase,'Native capture is incomplete')
        source=folder/'predictions.jsonl';annotation=read(folder/'classification_annotations.json')
        rows=[json.loads(line) for line in source.read_text().splitlines() if line.strip()]
        valid_ids={r['sample_id'] for r in annotation['samples']}
        predictions=[dict(sample_id=r['id'],prediction=r['prediction'],topk=r['top5'],scores=r.get('scores')) for r in rows if r['id'] in valid_ids]
        require(len(predictions)==len(rows)==len(valid_ids),'Native prediction/annotation count mismatch')
        metadata=read(folder/'classification_capture_metadata.json')
        require(metadata['source']['predictions']['sha256']==file_hash(source),'Native predictions changed after capture')
        require(metadata['source']['ground_truth']['sha256']==file_hash(folder/'classification_annotations.json'),'Native annotations changed after capture')
        output=folder/'standardized_classification'
        if output.exists():
            old_annotation,old_predictions,old_protocol=load_bundle(output)
            require(old_annotation==annotation and old_predictions==predictions,'Existing bundle differs; use a new run directory')
            require(old_protocol.get('source')==metadata['source'],'Existing bundle has different source evidence')
        else:publish(output,annotation,predictions,metadata)
        load_bundle(output)
        report['items'].append(dict(source=str(source.relative_to(root)),output=str(output),state='exported'))
    save(root/('classification_export_'+phase+'.json'),report)
    return report
