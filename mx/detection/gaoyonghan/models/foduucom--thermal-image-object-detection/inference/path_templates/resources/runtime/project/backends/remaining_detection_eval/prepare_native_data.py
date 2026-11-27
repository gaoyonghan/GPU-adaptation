"""Freeze public GT for Nemotron chart elements and table structure before inference."""
import hashlib
import importlib.util
import io
import json
import shutil
import tarfile
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path
from PIL import Image

HERE=Path(__file__).resolve().parent
ROOT=Path('__MODEL_ROOT__/resources/runtime/assets/workspace/results/remaining25_prepared')
DATA=Path('__DATASET_ROOT__')

def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()

def add(ann,image_id,category,box,crowd=0):
    x1,y1,x2,y2=box
    assert x2>x1 and y2>y1,(category,box)
    ann['annotations'].append({'id':len(ann['annotations'])+1,'image_id':image_id,'category_id':category,
                              'bbox':[x1,y1,x2-x1,y2-y1],'area':(x2-x1)*(y2-y1),'iscrowd':crowd})

def chart():
    archive=HERE/'assets/ICPR2022_PMC_TEST_v2.1.zip'
    part=archive.with_suffix('.zip.part')
    if not archive.exists():
        assert sha(part)=='c76ae4fcbf1a6e8d0db4be0fa94c21d9fdd043e5456501c0e6891a64f0c86b8b'
        part.rename(archive)
    out=ROOT/'chartinfo_text_roles';(out/'images').mkdir(parents=True,exist_ok=True)
    names=['chart_title','axis_title','tick_label','other','legend_label','legend_title','mark_label','value_label']
    ids={n:i+1 for i,n in enumerate(names)}
    ann={'images':[],'annotations':[],'categories':[{'id':ids[n],'name':n} for n in names]}
    source_members=[]
    with zipfile.ZipFile(archive) as z:
        index={Path(n).stem:n for n in z.namelist() if '/chart_images/' in n and n.endswith('.jpg')}
        for n in sorted(z.namelist()):
            if '/final_full_GT/' not in n or not n.endswith('.json'):continue
            data=json.loads(z.read(n))
            if not data.get('task3') or not data.get('task2'):continue
            roles=data['task3']['output']['text_roles'];blocks={b['id']:b for b in data['task2']['output']['text_blocks']}
            image_file=index[Path(n).stem];raw=z.read(image_file)
            with Image.open(io.BytesIO(raw)) as image:w,h=image.size
            image_id=len(ann['images'])+1; filename=Path(image_file).name
            (out/'images'/filename).write_bytes(raw)
            ann['images'].append({'id':image_id,'file_name':filename,'width':w,'height':h})
            source_members.append(n)
            for r in roles:
                p=blocks[r['id']]['polygon'];xs=[p[f'x{i}'] for i in range(4)];ys=[p[f'y{i}'] for i in range(4)]
                box=[min(xs),min(ys),max(xs),max(ys)]
                if r['role'] in ids:add(ann,image_id,ids[r['role']],box)
                else:
                    assert r['role']=='tick_grouping'
                    for category in ids.values():add(ann,image_id,category,box,1)
    assert len(ann['images'])==1873 and len({i['file_name'] for i in ann['images']})==1873
    (out/'annotations.json').write_text(json.dumps(ann))
    (out/'source.json').write_text(json.dumps({'url':'https://chartinfo.github.io/toolsanddata.html','archive_sha256':sha(archive),
      'selection':'All 1873 unique public test images with human task2 text polygons and task3 roles; no model-based selection.',
      'members':source_members,'label_merges':{'x_title+y_title':'axis_title','xlabel+ylabel':'tick_label'},
      'tick_grouping':'unmapped gold regions ignored for all evaluated classes','official_subset_match':False},indent=2))
    print('ChartInfo',len(ann['images']),len(ann['annotations']),flush=True)

def tables():
    base=DATA/'pubtables_structure_test';out=ROOT/'pubtables_structure_1000';(out/'images').mkdir(parents=True,exist_ok=True)
    source=Path('__MODEL_ROOT__/resources/runtime/assets/workspace/official_sources/object_detection/table-transformer/src/postprocess.py')
    spec=importlib.util.spec_from_file_location('official_table_postprocess',source);post=importlib.util.module_from_spec(spec);spec.loader.exec_module(post)
    ann={'images':[],'annotations':[],'categories':[{'id':1,'name':'cell'},{'id':2,'name':'row'},{'id':3,'name':'column'}]}
    with tarfile.open(base/'PubTables-1M-Structure_Annotations_Test.tar.gz') as t:
        members=[m for m in t.getmembers() if m.isfile() and m.name.endswith('.xml')]
        assert len(members)==93834
        # Freeze a model-independent sample; no sorting by ease, predictions or metrics.
        selected=sorted(members,key=lambda m:hashlib.sha256(m.name.encode()).hexdigest())[:1000]
        for m in selected:
            xml=ET.fromstring(t.extractfile(m).read());image_id=len(ann['images'])+1
            filename=xml.findtext('filename');w=int(xml.findtext('size/width'));h=int(xml.findtext('size/height'))
            objects=[]
            for obj in xml.findall('object'):
                b=obj.find('bndbox');objects.append({'name':obj.findtext('name'),'bbox':[float(b.findtext(k)) for k in ['xmin','ymin','xmax','ymax']]})
            rows=sorted([o for o in objects if o['name']=='table row'],key=lambda o:o['bbox'][1])
            columns=sorted([o for o in objects if o['name']=='table column'],key=lambda o:o['bbox'][0])
            spans=[dict(o,subheader=o['name']=='table projected row header') for o in objects if o['name'] in {'table spanning cell','table projected row header'}]
            table=next(o for o in objects if o['name']=='table')
            cells,_=post.table_structure_to_cells({'rows':rows,'columns':columns,'supercells':spans},[],table['bbox'])
            assert cells and rows and columns
            ann['images'].append({'id':image_id,'file_name':filename,'width':w,'height':h})
            for category,items in [(1,cells),(2,rows),(3,columns)]:
                for obj in items:add(ann,image_id,category,obj['bbox'])
    wanted={i['file_name'] for i in ann['images']};found=set()
    with tarfile.open(base/'PubTables-1M-Structure_Images_Test.tar.gz',mode='r|gz') as t:
        for member in t:
            if member.isfile() and Path(member.name).name in wanted:
                name=Path(member.name).name
                with (out/'images'/name).open('wb') as f:shutil.copyfileobj(t.extractfile(member),f)
                found.add(name)
    assert found==wanted
    (out/'annotations.json').write_text(json.dumps(ann))
    (out/'source.json').write_text(json.dumps({'source':'https://github.com/microsoft/table-transformer',
      'selection':'1000 of 93834 official test XML filenames with smallest SHA256(filename); frozen before inference. Full evaluation means this 1000-image replacement benchmark, not the full 93834-image official test.',
      'selected_xml':[m.name for m in selected],'cell_gold':'Official table_structure_to_cells on GT rows/columns/spanning/projected cells; no model predictions, no OCR tokens. Geometric grid-cell boxes.',
      'postprocess_source':str(source),'postprocess_sha256':sha(source),'annotations_archive_sha256':sha(base/'PubTables-1M-Structure_Annotations_Test.tar.gz')},indent=2))
    print('PubTables fixed benchmark',len(ann['images']),len(ann['annotations']),flush=True)

if __name__=='__main__':
    import sys
    {'chart':chart,'tables':tables}[sys.argv[1]]()
