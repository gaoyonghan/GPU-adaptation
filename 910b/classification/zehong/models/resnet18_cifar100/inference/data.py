"""独立GT与图像读取，不计算精度。"""
from pathlib import Path
import re
from PIL import Image

def norm(s):return re.sub(r'[^a-z0-9]','',str(s).lower())
class Images:
    def __init__(self,raw,samples):self.raw,self.samples=raw,samples
    def __len__(self):return len(self.samples)
    def __getitem__(self,i):
        x=self.raw[i]['image'] if not isinstance(self.raw,list) else Image.open(self.raw[i])
        return x.convert('RGB')

def load(root,model,kind,max_samples=None,mapping=None,vocab=None):
    root=Path(root)
    labels={int(k):str(v) for k,v in model.config.id2label.items()}
    categories=[{'id':i,'name':v} for i,v in sorted(labels.items())]
    if kind=='pet':
        # 官方test.txt独立GT，不用预测结果构造标签、不混训练图片。
        rows=[s.split() for s in (root/'annotations/test.txt').read_text().splitlines() if s.strip() and not s.startswith('#')]
        paths=[root/'images'/(r[0]+'.jpg') for r in rows]
        names=[r[0].rsplit('_',1)[0] for r in rows]
        lookup={norm(v):i for i,v in labels.items()}
        gt=[lookup[norm(n)] for n in names]
        raw=paths
    else:
        from datasets import Dataset,concatenate_datasets
        split='test' if kind=='cifar' else 'validation'
        files=sorted((root/split).rglob('*.arrow'))
        if not files:files=sorted(p for p in root.rglob('*.arrow') if (split+'-') in p.name)
        if not files:raise FileNotFoundError(f'No {split} Arrow shards under {root}; do not mix train/test')
        raw=concatenate_datasets([Dataset.from_file(str(p)) for p in files])
        if kind=='cifar':
            # Hugging Face CIFAR-100 uses img + fine_label (100 classes).
            # coarse_label is a different 20-class task and must not be scored here.
            if 'image' not in raw.column_names and 'img' in raw.column_names:
                raw=raw.rename_column('img','image')
            if 'label' not in raw.column_names and 'fine_label' in raw.column_names:
                raw=raw.rename_column('fine_label','label')
            class_names=getattr(raw.features.get('label'),'names',None)
            # HF CIFAR-100 metadata spells class 26 as cra; canonical name is crab.
            # This is a name alias only: fine_label integer IDs stay unchanged.
            if class_names and len(class_names)==100 and class_names[26]=="cra" and labels.get(26)=="crab":
                class_names=list(class_names);class_names[26]="crab"
            if class_names and [norm(n) for n in class_names]!=[norm(labels[i]) for i in range(len(labels))]:
                raise ValueError('CIFAR-100 fine label order differs from model labels')
        gt=[int(x) for x in raw['label']]
        names=getattr(raw.features.get('label'),'names',None)
        if mapping is not None:gt=[mapping[i] for i in gt]
        elif kind in ['food','vegetable'] and names:
            lookup={norm(v):i for i,v in labels.items()}
            if all(norm(n) in lookup for n in names):gt=[lookup[norm(names[i])] for i in gt]
            elif any(not re.fullmatch(r'(label_?)?\d+',v.lower()) for v in labels.values()):
                raise ValueError('Dataset/model class names differ; supply verified label mapping')
    count=len(gt) if max_samples is None else min(max_samples,len(gt))
    samples=[{'sample_id':f'image_{i:08d}','label':int(gt[i])} for i in range(count)]
    if not samples:raise ValueError('Empty test split')
    if any(x['label'] not in labels for x in samples):raise ValueError('GT outside classifier label space')
    return Images(raw,samples),categories,{'root':str(root.resolve()),'split':'test' if kind in ['pet','cifar'] else 'validation','scope':'debug_subset' if max_samples else 'full_split','gt_space':'22k_open_set' if mapping else 'model_class_ids'}
