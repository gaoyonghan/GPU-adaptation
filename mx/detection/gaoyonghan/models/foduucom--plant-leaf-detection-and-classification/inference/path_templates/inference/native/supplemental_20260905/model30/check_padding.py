"""Compare author GT restoration for batch1 and batch8, without inference."""
import io
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / '_deps'))
import albumentations as A
import numpy as np
import pyarrow.parquet as pq
import torch
from PIL import Image
from transformers import AutoImageProcessor
from transformers.image_transforms import center_to_corners_format

processor = AutoImageProcessor.from_pretrained('__MODEL_ROOT__/weights/yainage90/fashion-object-detection', local_files_only=True, use_fast=False)
rows = pq.read_table('__DATASET_ROOT__/supplemental_20260905/model30/test-00000-of-00001.parquet').slice(0, 8).to_pylist()
validation = A.Compose([A.NoOp()], bbox_params=A.BboxParams(format='coco', label_fields=['category'], clip=True))
images, annotations = [], []
for i, row in enumerate(rows):
    image = np.array(Image.open(io.BytesIO(row['image']['bytes'])).convert('RGB'))
    obj = row['objects']
    item = validation(image=image, bboxes=obj['bbox'], category=obj['category'])
    images.append(item['image'])
    annotations.append({'image_id':i+1, 'annotations':[{'image_id':i+1,'category_id':int(c),'iscrowd':0,'area':a,'bbox':list(b)} for c,a,b in zip(item['category'],obj['area'],item['bboxes'])]})
batch = processor(images=images, annotations=annotations, return_tensors='pt')
checks = []
for i,(image,annotation) in enumerate(zip(images,annotations)):
    single = processor(images=image, annotations=annotation, return_tensors='pt')
    sizes = torch.tensor([[image.shape[1], image.shape[0], image.shape[1], image.shape[0]]])
    single_gt = center_to_corners_format(single['labels'][0]['boxes']) * sizes
    batch_gt = center_to_corners_format(batch['labels'][i]['boxes']) * sizes
    checks.append({'image_id':i+1,'single_tensor_size':list(single['pixel_values'].shape),'batch_tensor_size':list(batch['pixel_values'].shape),
                   'GT_restoration_max_abs_difference_pixels':float((single_gt-batch_gt).abs().max())})
result = {'scope':'CPU-only first8 fixed samples, current frozen Transformers4.50.0; no new inference or score tuning',
          'checks':checks, 'interpretation':'Author batch8 path can alter normalized GT restoration after padding compared with batch1. This is a demonstrated protocol difference under the current processor, not proof of the cause of the model-card score gap.'}
(HERE/'padding_diagnostic.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result,indent=2))
