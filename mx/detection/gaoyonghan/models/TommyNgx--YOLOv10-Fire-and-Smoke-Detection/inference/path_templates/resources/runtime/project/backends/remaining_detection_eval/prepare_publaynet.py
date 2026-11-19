"""Convert the locally available PubLayNet validation parquet without relabeling."""
import io
import json
from pathlib import Path

import pyarrow.parquet as pq
from PIL import Image

SOURCE = Path('__DATASET_ROOT__/publaynet_val/data')
OUT = Path('__MODEL_ROOT__/resources/runtime/assets/workspace/results/remaining25_prepared/publaynet_val')


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'images').mkdir(exist_ok=True)
    images, annotations = [], []
    for part in sorted(SOURCE.glob('*.parquet')):
        for batch in pq.ParquetFile(part).iter_batches(batch_size=32):
            for row in batch.to_pylist():
                payload = row['image']['bytes']
                image_id = row['id']
                name = f'{image_id}.jpg'
                with Image.open(io.BytesIO(payload)) as image:
                    width, height = image.size
                target = OUT / 'images' / name
                if not target.exists():
                    target.write_bytes(payload)
                else:
                    assert target.read_bytes() == payload
                images.append({'id': image_id, 'file_name': name, 'width': width, 'height': height})
                annotations.extend(row['annotations'])
        print(part.name, len(images), flush=True)
    assert len(images) == len({x['id'] for x in images}) == 11245
    assert len({x['id'] for x in annotations}) == len(annotations)
    assert {x['category_id'] for x in annotations} == {1, 2, 3, 4, 5}
    result = {'images': images, 'annotations': annotations, 'info': {}, 'licenses': [],
              'categories': [{'id': i+1, 'name': n} for i, n in enumerate(['text', 'title', 'list', 'table', 'figure'])]}
    (OUT / 'annotations.json').write_text(json.dumps(result))
    (OUT / 'source.json').write_text(json.dumps({'source': str(SOURCE), 'split': 'validation',
        'images': len(images), 'boxes': len(annotations), 'transformation': 'Original COCO boxes and segmentation retained unchanged.'}, indent=2))


if __name__ == '__main__':
    main()
