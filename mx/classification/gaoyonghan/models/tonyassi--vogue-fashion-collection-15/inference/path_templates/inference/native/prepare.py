"""Freeze local checkpoints and data selections; never rewrite image bytes."""
import argparse
from collections import Counter
import gzip
import io
import json
from pathlib import Path
import re
import zipfile

import pyarrow.parquet as pq

from common import HERE, WORKSPACE, SOURCE, DataOnlyUnpickler, digest, fingerprint, save

DATA = WORKSPACE / 'datasets_171_200'
STATUS = Path('__MODEL_ROOT__/resources/runtime/assets/workspace/171-200/model_download_status.json')
IMAGENET = Path('__MODEL_ROOT__/resources/runtime/assets/workspace/classification8_eval/.runtime/data/imagenet/index.json')


def norm(s):
    return re.sub(r'[_\s-]+', ' ', s.strip().lower())


def parquet_rows(paths, labels, label_column='label', aliases=None, names=None, prediction_only=False):
    result = []
    mapping = {norm(v): int(k) for k, v in labels.items()}
    mapping.update(aliases or {})
    excluded = Counter()
    for path in paths:
        pf = pq.ParquetFile(path)
        features = json.loads((pf.schema_arrow.metadata or {}).get(b'huggingface', b'{"info":{"features":{}}}'))['info']['features']
        source_names = names or features[label_column]['names']
        image_column = 'img' if 'img' in pf.schema_arrow.names else 'image'
        for group in range(pf.num_row_groups):
            values = pf.read_row_group(group, columns=[label_column]).column(0).to_pylist()
            for offset, value in enumerate(values):
                name = source_names[value]
                target = mapping.get(norm(name))
                if target is None and not prediction_only:
                    excluded[name] += 1
                    continue
                result.append({'id': f'{path.name}:{group}:{offset}', 'parquet': str(path), 'group': group,
                               'offset': offset, 'image_column': image_column, 'target': target, 'source_label': name})
    return result, dict(excluded)


def zip_rows(path, labels, prefix='', aliases=None, members=None, inner=None, prediction_only=False):
    mapping = {norm(v): int(k) for k, v in labels.items()}
    mapping.update(aliases or {})
    result, excluded = [], Counter()
    with zipfile.ZipFile(path) as outer:
        archive = zipfile.ZipFile(io.BytesIO(outer.read(inner))) if inner else outer
        for info in sorted(archive.infolist(), key=lambda x: x.filename):
            name = info.filename
            if not name.startswith(prefix) or Path(name).suffix.lower() not in {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}:
                continue
            if members is not None and name not in members:
                continue
            label = Path(name).parent.name if inner is None else Path(name).name.split('.')[0]
            target = mapping.get(norm(label))
            if target is None and not prediction_only:
                excluded[label] += 1
                continue
            row = {'id': name, 'archive': str(path), 'member': name, 'crc32': info.CRC,
                   'bytes': info.file_size, 'target': target, 'source_label': label}
            if inner:
                row['inner_archive'] = inner
            result.append(row)
        if inner:
            archive.close()
    return result, dict(excluded)


def select_data(n, labels):
    notes, excluded = [], {}
    relation, split = 'public_replacement', 'published selection'
    rows = []
    if n in [179, 184, 188, 193, 196, 198]:
        data = json.loads(IMAGENET.read_text())
        assert len(data['rows']) == 50000 and len(data['classes']) == 1000
        if labels:
            actual = [norm(labels[str(i)]) for i in range(1000)]
            expected = [norm(s) for s in data['label_descriptions']]
            assert actual == expected, f'{n}: ImageNet output order differs; needs explicit mapping'
        rows = [{**r, 'target': r['label'], 'synset': data['classes'][r['label']]} for r in data['rows']]
        relation, split = 'original_exact', 'ImageNet-1K validation'
    elif n in [191, 199, 200, 192]:
        folder = DATA / ('llamastics' if n == 192 else f'imagenet_overlap_{n}')
        for r in map(json.loads, (folder / 'manifest.jsonl').read_text().splitlines()):
            rows.append({'id': r['path'], 'path': str(folder / r['path']), 'target': r['target_index'],
                         'sha256': r['sha256'], **({'synset': r['synset']} if 'synset' in r else {})})
        split = 'accepted external reference set' if n == 192 else 'ImageNet-1K val overlap 999 classes'
        notes.append('Keep full model output head. Not an independent original model test set.')
        if n == 192:
            notes.append('800 records, 793 unique hashes; original within-class duplicates retained; four conflicting files excluded.')
    elif n == 171:
        rows, excluded = zip_rows(DATA/'closed_eyes/raw/drivers-drowsiness-detection.zip', labels, 'Final/PreparedData/test/')
        split = 'publisher PreparedData/test'
    elif n == 172:
        rows, excluded = parquet_rows(sorted((DATA/'vogue_runway/raw/data').glob('*.parquet')), labels)
        relation, split = 'original_protocol_incomplete', 'published train (87547 images, not independent test)'
        notes.append('Model has 1677 collection classes, not 15 labels; same-source/train evaluation only.')
    elif n == 173:
        rows, excluded = zip_rows(DATA/'facial_emotions/raw/facial-emotion-expressions.zip', labels, 'images/validation/')
        split = 'validation, one archive tree only (exclude duplicate images/images tree)'
    elif n == 175:
        path = DATA/'fer2013_jeneral/raw/test.pt'
        with path.open('rb') as f:
            examples = DataOnlyUnpickler(f).load()
        names = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
        reverse = {v: int(k) for k, v in labels.items()}
        for i, r in enumerate(examples):
            name = names[r['labels']] if isinstance(r['labels'], int) else r['labels']
            if name in reverse:
                rows.append({'id': str(i), 'fer_pickle': str(path), 'offset': i, 'target': reverse[name]})
            else:
                excluded[name] = excluded.get(name, 0) + 1
        relation, split = 'original_protocol_incomplete', 'Jeneral test; four supported labels'
        notes.append('Four-class model evaluated only on its supported classes; do not claim full seven-class FER performance.')
    elif n == 176:
        rows, excluded = zip_rows(DATA/'cats_dogs_redux/raw/dogs-vs-cats-redux-kernels-edition.zip', labels, inner='train.zip')
        split = 'labelled competition train; test ground truth unavailable'
        notes.append('Train/reference scoring; not a Kaggle leaderboard reproduction. Record binary log loss as reference.')
    elif n in [177, 190]:
        path = Path('__DATASET_ROOT__/FairFace/0.25/validation-00000-of-00001-951dbd63c8724ee1.parquet')
        if n == 177:
            labels = {str(i): s for i, s in enumerate(['Black', 'East Asian', 'Indian', 'Latino_Hispanic', 'Middle Eastern', 'Southeast Asian', 'White'])}
        rows, excluded = parquet_rows([path], labels, 'race' if n == 177 else 'age', prediction_only=n == 190)
        relation, split = ('original_exact' if n == 177 else 'public_replacement'), 'FairFace 0.25 validation'
        if n == 190:
            for r in rows:
                r['target'] = None
            notes.append('Age bins differ (FairFace 70+ vs three model bins, other boundary definitions undocumented). Predictions/timing only; no fabricated accuracy.')
    elif n == 178:
        rows, excluded = zip_rows(DATA/'oahega/dataset.zip', labels)
        split = 'all published OAHEGA images; no independent split'
    elif n == 180:
        rows, excluded = parquet_rows(sorted((DATA/'portrait_gender/raw/dataset').glob('*.parquet')), labels)
        relation, split = 'original_protocol_incomplete', 'published Train shards; no independent test'
    elif n == 181:
        data = json.loads((DATA/'openfake/manifest.json').read_text())
        rows = [{'id': r['name'], 'path': str(DATA/r['name']), 'target': 0 if r['label']=='fake' else 1, 'sha256': r['sha256']} for r in data['images']]
        split = 'accepted core/test 5000 real + 5000 fake; seeded cluster sample'
    elif n == 183:
        aliases = {norm(s): i for i,s in enumerate(['0-12','13-20','21-44','45-64','65+'])}
        rows, excluded = parquet_rows(sorted((DATA/'age_classification/raw').glob('*.parquet')), labels, aliases=aliases)
        relation, split = 'original_protocol_incomplete', 'published train; no independent test'
    elif n == 185:
        rows, excluded = zip_rows(DATA/'fc45dhlpd/15 Crop 45 Disease and Healthy Leaf dataset.zip', labels)
        split = 'exact-name supported crop/disease classes only'
        notes.append('Full ViT state dict is published, no adapter_config or LoRA tensors; load original full checkpoint, no merging. Exclude nonmatching diseases rather than guess.')
    elif n == 186:
        path = DATA/'indoor67/raw/indoor-scenes-cvpr-2019.zip'
        with zipfile.ZipFile(path) as z:
            members = {'indoorCVPR_09/Images/'+s.strip() for s in z.read('TestImages.txt').decode().splitlines() if s.strip()}
        rows, excluded = zip_rows(path, labels, members=members, aliases={'livingroom': 4})
        split = 'MIT Indoor67 TestImages; five supported room classes'
    elif n == 187:
        data = json.loads((DATA/'streetscapes/accepted_manifest.json').read_text())
        reverse = {v: int(k) for k,v in labels.items()}
        rows = [{'id':r['uuid'],'path':str(DATA/r['name']),'target':reverse[r['city']],'sha256':r['sha256']} for r in data['records']]
        split = 'accepted KartaView 1376, London/New York/Tokyo'
    elif n == 189:
        path = DATA/'architectural_styles/raw/architectural-styles-dataset.zip'
        candidates = ['American craftsman style', 'Queen Anne architecture', 'Tudor Revival architecture']
        with zipfile.ZipFile(path) as z:
            members = {s for s in z.namelist() if s.startswith('architectural-styles-dataset/') and Path(s).parent.name in candidates}
        rows, excluded = zip_rows(path, labels, members=members, prediction_only=True)
        for row in rows:
            row['target'] = None
        notes.append('Broader dataset styles are not a verified bijection to craftsman bungalow/tudor cottage subclasses. Predictions/timing only until taxonomy is resolved.')
        split = 'three previously selected candidate styles, one archive tree'
    elif n == 194:
        info = json.loads((DATA/'eurosat/raw/dataset_infos.json').read_text())
        names = next(iter(info.values()))['features']['label']['names']
        rows, excluded = parquet_rows(sorted((DATA/'eurosat/raw/data').glob('*.parquet')), labels, names=names)
        relation, split = 'original_protocol_incomplete', 'EuroSAT demo entire train, 27000; author held-out split unknown'
    elif n == 195:
        rows, excluded = parquet_rows(sorted((DATA/'cifar100/raw/cifar100').glob('test*.parquet')), labels, 'fine_label')
        relation, split = 'original_exact', 'CIFAR100 test, 10000'
    elif n == 197:
        root = WORKSPACE/'model_197_assets/datasets/fashion_mnist/raw'
        data = gzip.open(root/'t10k-labels-idx1-ubyte.gz','rb').read()
        assert len(data)==10008
        rows = [{'id':str(i),'fashion_idx':str(root/'t10k-images-idx3-ubyte.gz'),'offset':i,'target':v} for i,v in enumerate(data[8:])]
        split = 'standard Fashion-MNIST test, not quality_drift variant'
    assert rows, f'{n}: no selected images'
    assert len(rows)==len({r['id'] for r in rows}), f'{n}: duplicate sample IDs'
    return rows, {'dataset_relation':relation,'split':split,'notes':notes,'excluded_source_labels':excluded}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,default=HERE)
    args=parser.parse_args()
    out=args.output.resolve()
    if (out/'models.json').exists():
        raise SystemExit('Frozen models.json already exists; preserve it or choose a new --output directory.')
    out.mkdir(parents=True,exist_ok=True)
    statuses=json.loads(STATUS.read_text())['results']
    model_list=[]
    code_status=json.loads(Path('__MODEL_ROOT__/resources/runtime/assets/workspace/171-200/code_download_status.json').read_text())['results']
    sources={r['code_key']:{'path':r['path'],'commit':r['commit']} for r in code_status}
    for item in statuses:
        n=item['sequence'];root=Path(item['path'])
        spec={'sequence':n,'model_id':item['model_id'],'model_path':str(root),'revision':item.get('revision'),
              'task_type':'image-classification','status':'not_started','official_model_card':f'https://huggingface.co/{item["model_id"]}',
              'official_reference':None,'reference_comparison':'manual_reference_required','known_differences':[]}
        if n in [174,182]:
            spec.update(status='blocked' if n==174 else 'skipped',reason='Dataset access approval pending' if n==174 else 'User requested skip')
            model_list.append(spec);continue
        config=json.loads((root/'config.json').read_text()) if (root/'config.json').exists() else {}
        spec['config']=config
        spec['preprocessing']=json.loads((root/'preprocessor_config.json').read_text()) if (root/'preprocessor_config.json').exists() else config.get('pretrained_cfg',config)
        spec['architecture']=config.get('architectures',config.get('architecture'))
        files=[]
        if n==177:
            files=sorted(root.glob('*.pt'))
            spec['variants']=[p.name for p in files]
        elif n==173:
            files=[root/'onnx/model_quantized.onnx']
        else:
            files=[root/('model.safetensors' if (root/'model.safetensors').exists() else 'pytorch_model.bin')]
        files += [root/name for name in ['config.json','preprocessor_config.json','README.md'] if (root/name).exists()]
        if n==179:
            files+=list(root.glob('*.py'))
        spec['files']={str(p.relative_to(root)):{'size':p.stat().st_size,'sha256':digest(p)} for p in files}
        spec['weight']=str(files[0].relative_to(root))
        if not spec['revision']:
            for name in [spec['weight'],'config.json']:
                p=root/'.cache/huggingface/download'/f'{name}.metadata'
                if p.exists():
                    spec['revision']=p.read_text().splitlines()[0];break
        spec['backend']='timm_native' if n in [184,193,199,200] else 'ultralytics_native' if n==177 else 'transformers_js' if n==173 else 'mambavision_published' if n==179 else 'hf_published'
        spec['environment']='new_hf' if n==183 else 'yolo' if n==177 else 'js' if n==173 else 'mamba' if n==179 else 'shared_hf'
        labels=config.get('id2label',{})
        if n==191:
            corrected=DATA/'imagenet_overlap_191/config.corrected_labels.json'
            spec['corrected_label_config']={'path':str(corrected),'sha256':digest(corrected)}
            labels=json.loads(corrected.read_text())['id2label']
        rows,info=select_data(n,labels)
        spec.update(info)
        spec['images']=len(rows);spec['covered_target_classes']=len({r['target'] for r in rows if r['target'] is not None})
        spec['num_output_classes']=7 if n==177 else len(labels) or config['num_classes']
        spec['label_mapping']=labels
        spec['scoring']='prediction_only' if n in [189,190] else 'native' if n in [177,184,193,199,200] else 'reference'
        spec['evaluator']='timm.validate' if spec['backend']=='timm_native' else 'Ultralytics ClassificationValidator' if n==177 else 'sklearn reference scoring' if spec['scoring']=='reference' else None
        spec['fidelity']='reference_baseline'
        spec['case_type']='special'
        spec['special_tags']=['framework_numeric_difference']+(['dataset_replacement'] if spec['dataset_relation']=='public_replacement' else [])+(['official_evaluator_missing'] if spec['scoring']!='native' else [])
        spec['known_differences']=spec.pop('notes')
        spec['inference_entry']='pipeline' if n in [172,181] else 'publisher_transformers_model_and_processor'
        spec['default_batch_size']=1
        spec['dtype']='float32' if n!=173 else 'published ONNX q8 on CPU'
        spec['warmup_batches']=10
        index=out/'indices'/f'{n}.jsonl';index.parent.mkdir(exist_ok=True)
        index.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows))
        spec.update(index=str(index),index_sha256=digest(index))
        source_paths={r[k] for r in rows for k in ['archive','parquet','fer_pickle','fashion_idx'] if k in r}
        spec['data_containers']={p:{'size':Path(p).stat().st_size,'mtime_ns':Path(p).stat().st_mtime_ns} for p in sorted(source_paths)}
        if n in [184,193]:
            spec['class_map']=str(SOURCE/'pytorch-image-models/timm/data/_info/imagenet_synsets.txt')
        elif n in [199,200]:
            spec['class_map']=str(DATA/f'imagenet_overlap_{n}/native_class_map.txt')
        if 'class_map' in spec:
            spec['class_map_sha256']=digest(spec['class_map'])
        references=HERE/'model_card_references.json'
        if references.exists():
            spec['official_reference']=json.loads(references.read_text()).get(str(n))
        spec['protocol_fingerprint']=fingerprint(spec)
        model_list.append(spec)
        print(n,spec['backend'],len(rows),spec['covered_target_classes'],spec['scoring'],flush=True)
    save(out/'models.json',{'schema_version':1,'sources':sources,'models':model_list,
         'runtime_defaults':{'batch_size':1,'device':'cuda:0','warmup_batches':10,'workers':2},
         'scope':'Each published model retained; 177 includes all five weights; no full GPU inference launched by preparation.'})


if __name__=='__main__':
    main()
