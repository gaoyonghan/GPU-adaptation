"""Dataset adaptation preserves native GT mapping; optional manifest is for explicit prepared subsets."""
import json
from pathlib import Path
from PIL import Image
class ManifestImages:
    def __init__(self,root):
        self.root=Path(root)
        self.samples=[json.loads(line) for line in (self.root/'manifest.jsonl').read_text().splitlines() if line.strip()]
    def __len__(self):return len(self.samples)
    def __getitem__(self,i):
        with Image.open(self.root/self.samples[i]['relative_path']) as image:return image.convert('RGB')
class LimitedImages:
    def __init__(self,source,count):self.source=source;self.samples=source.samples[:count]
    def __len__(self):return len(self.samples)
    def __getitem__(self,i):return self.source[i]
def load(root,model):
    root=Path(root)
    if (root/'manifest.jsonl').is_file():
        dataset=ManifestImages(root)
        categories=[{'id':int(i),'name':str(n)} for i,n in sorted(model.config.id2label.items(),key=lambda x:int(x[0]))]
        return dataset,categories,{'root':str(root),'protocol':'explicit_manifest_model_label_ids','scope':'prepared_subset'}
    from .model_data import load_birds
    return load_birds(root,model.config.label2id)
