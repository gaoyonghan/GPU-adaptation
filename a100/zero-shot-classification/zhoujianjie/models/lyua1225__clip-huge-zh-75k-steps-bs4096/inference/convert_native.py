"""Convert preserved native predictions, without reranking or inventing labels."""
import json,os
from pathlib import Path
from model_paths import OUTPUT,DATA,NATIVE,MODEL_ID
from export_bundle import export_bundle
def main():
    cfg=json.loads((Path(__file__).parent/'delivery_config.json').read_text())
    if cfg['native_task']!='single_label_classification':
        (OUTPUT/'inference_output.json').write_text(json.dumps({'state':'native_completed','task':cfg['native_task'],'bundles':[],'public_evaluator_status':cfg['public_evaluator'],'reason':'The shared evaluator only supports single-label classification; original metrics were preserved.'},indent=2))
        return
    bundles=[]
    for path in sorted(OUTPUT.rglob('predictions.jsonl')):
        rows=[json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        if not rows:continue
        categories=None;dataset=path.parent.name
        if cfg['launcher']=='run_chinese_native.py':
            metrics=json.loads((path.parent/'metrics.json').read_text());dataset=metrics['dataset']
            folder='voc2007' if dataset=='voc-2007-classification' else 'chinese_clip_'+dataset
            names=(DATA/'raw'/folder/'label_cn.txt').read_text().splitlines()
        elif cfg['launcher']=='run_pe_native.py':
            from pe_native_a100 import load_protocol
            names=load_protocol()[0];dataset='ImageNet-A'
        elif cfg['launcher']=='run_rsicd_native.py':
            from evaluate_clip_rsicd_a100 import MANIFEST
            names=json.loads(MANIFEST.read_text())['classes'];dataset='RSICD test'
        elif cfg['launcher']=='run_metaclip2_native.py':
            names=['option_0','option_1','option_2','option_3'];dataset='CVQA LOCAL'
        elif cfg['launcher']=='run_imagenet_remaining.py':
            from imagenet_native_a100 import protocol_text
            names=protocol_text('clip_80')[0];dataset='ImageNet validation'
        else:continue
        converted=[]
        for i,row in enumerate(rows):
            sid=row.get('sample_id')
            if sid is None:sid=(row['shard']+':'+str(row['batch']*16+row['index'])) if 'shard' in row else str(i)
            ranking=row.get('ranked_predictions',row.get('ranking'))
            prediction=row.get('prediction',ranking[0] if ranking else None)
            # ONNX argpartition top5 is an unordered set. Export Top1 only.
            if ranking is None:ranking=[prediction]
            converted.append({'sample_id':str(sid),'target':row['target'],'prediction':prediction,'topk':ranking})
        run=OUTPUT/('bundle_'+str(len(bundles)));run.mkdir()
        export_bundle(run,[{'id':i,'name':str(name)} for i,name in enumerate(names)],converted,MODEL_ID,dataset)
        bundles.append(str((run/'scoring').relative_to(OUTPUT)))
    def emit(names,rows,dataset):
        run=OUTPUT/('bundle_'+str(len(bundles)));run.mkdir()
        export_bundle(run,[{'id':i,'name':str(name)} for i,name in enumerate(names)],rows,MODEL_ID,dataset)
        bundles.append(str((run/'scoring').relative_to(OUTPUT)))
    if cfg['launcher']=='run_three_native.py' and MODEL_ID=='geolocal/StreetCLIP':
        for f in OUTPUT.rglob('country_predictions.json'):
            data=json.loads(f.read_text());names=json.loads((f.parent/'metrics.json').read_text())['result']['labels']
            rows=[{'sample_id':str(i),'target':t,'prediction':p,'topk':k} for i,(t,p,k) in enumerate(zip(data['targets'],data['predictions'],data['top5_predictions']))]
            emit(names,rows,'Country211')
    if cfg['launcher']=='run_medsiglip_native.py':
        import numpy as np
        for f in OUTPUT.rglob('predictions.npz'):
            data=np.load(f);rows=[{'sample_id':str(i),'target':int(t),'prediction':int(p)} for i,(t,p) in enumerate(zip(data['targets'],data['predictions']))]
            emit(['no metastatic tumor tissue','metastatic tumor tissue'],rows,'PCam frozen test / frozen 4090 threshold')
    if cfg['launcher']=='run_openclip_native.py':
        unsupported=[]
        for f in OUTPUT.rglob('*.native_ranked.jsonl'):
            payload=json.loads(f.with_name(f.name.replace('.native_ranked.jsonl','.json')).read_text())
            rows=[json.loads(line) for line in f.read_text().splitlines()]
            emit(payload['classnames'],rows,payload['dataset'])
        if not bundles:unsupported.append('No native single-label accuracy ranking was captured; multi-label AP is not convertible to single-label accuracy.')
        if unsupported:(OUTPUT/'public_protocol_issues.json').write_text(json.dumps(unsupported,indent=2))
    if cfg['launcher']=='run_bioclip_native.py':
        for f in OUTPUT.rglob('native_ranked.jsonl'):
            metadata=json.loads((f.parent/'native_class_map.json').read_text());names=[None]*len(metadata['class_to_idx'])
            for name,index in metadata['class_to_idx'].items():names[index]=name
            rows=[json.loads(line) for line in f.read_text().splitlines()]
            for i,row in enumerate(rows):row['sample_id']=str(metadata['samples'][i])
            emit(names,rows,f.parent.name+' / original zero-shot')
    if MODEL_ID=='qihoo360/fg-clip2-base':
        ann=json.loads((DATA/'raw/fgclip_box/annotations/valid_category_data_total_zh.json').read_text())
        names=[cat['name'] for cat in ann['categories']]
        for f in OUTPUT.rglob('rank0_predictions.jsonl'):
            if f.parent.name=='smoke' and not int(os.environ.get('ZERO_SHOT_SMOKE_LIMIT','0')):continue
            rows=[]
            for i,line in enumerate(f.read_text().splitlines()):
                row=json.loads(line);ranking=row['ranking']
                rows.append({'sample_id':str(i),'target':row['target'],'prediction':ranking[0],'topk':ranking})
            emit(names,rows,'FG-CLIP2 BoxClass CN / original annotation order')
    if MODEL_ID=='intfloat/mmE5-mllama-11b-instruct':
        import ast,pyarrow.parquet as pq
        frozen=pq.read_table(DATA/'raw/mmeb_country_1/test-00000-of-00001.parquet').to_pylist()
        names=frozen[0]['tgt_text'];index={name:i for i,name in enumerate(names)}
        for f in OUTPUT.rglob('Country211_pred.txt'):
            values=[ast.literal_eval(line) for line in f.read_text().splitlines()]
            rows=[{'sample_id':str(i),'target':index[frozen[i]['tgt_text'][0]],'prediction':index[value[0]]} for i,value in enumerate(values)]
            emit(names,rows,'MMEB Country211 original candidate strings')
    if not bundles:
        raise RuntimeError('Native prediction conversion is not implemented for this output yet; no standard success claimed.')
    (OUTPUT/'inference_output.json').write_text(json.dumps({'state':'completed','task':'classification','bundles':bundles,'smoke_only':bool(int(os.environ.get('ZERO_SHOT_SMOKE_LIMIT','0')))},indent=2))
if __name__=='__main__':main()
