"""Extract only the published test split from the existing SKU110K archive."""
import csv
import fcntl
import json
import shutil
import tarfile
from pathlib import Path

SOURCE = Path('__DATASET_ROOT__/sku110k_test/SKU110K_fixed.tar.gz')
OUTPUT = Path('__MODEL_ROOT__/resources/runtime/assets/workspace/results/remaining25_prepared/sku110k_test')


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    with (OUTPUT / '.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        marker = OUTPUT / 'source.json'
        identity = {'source': str(SOURCE), 'bytes': SOURCE.stat().st_size, 'mtime_ns': SOURCE.stat().st_mtime_ns}
        if marker.exists():
            assert json.loads(marker.read_text())['source_identity'] == identity
            return
        (OUTPUT / 'images').mkdir(exist_ok=True)
        with tarfile.open(SOURCE, 'r|gz') as archive:
            count = 0
            for member in archive:
                name = Path(member.name).name
                if not member.isfile() or not (name == 'annotations_test.csv' or (name.startswith('test_') and name.endswith('.jpg'))):
                    continue
                target = OUTPUT / name if name.endswith('.csv') else OUTPUT / 'images' / name
                if not target.exists() or target.stat().st_size != member.size:
                    temporary = target.with_suffix(target.suffix + '.part')
                    with archive.extractfile(member) as reader, temporary.open('wb') as writer:
                        shutil.copyfileobj(reader, writer)
                    temporary.replace(target)
                count += 1
                if count % 500 == 0:
                    print('test files', count, flush=True)
        images, annotations = {}, []
        with (OUTPUT / 'annotations_test.csv').open(newline='') as stream:
            for row in csv.reader(stream):
                assert len(row) == 8, row
                name, x1, y1, x2, y2, label, width, height = row
                if name not in images:
                    images[name] = {'id': len(images)+1, 'file_name': name, 'width': int(width), 'height': int(height)}
                x1, y1, x2, y2 = map(float, (x1, y1, x2, y2))
                assert x2 > x1 and y2 > y1, row
                annotations.append({'id': len(annotations)+1, 'image_id': images[name]['id'],
                                    'category_id': 1, 'bbox': [x1, y1, x2-x1, y2-y1],
                                    'area': (x2-x1)*(y2-y1), 'iscrowd': 0})
        assert len(images) == 2936, len(images)
        assert set(images) == {p.name for p in (OUTPUT / 'images').glob('*.jpg')}
        annotation = {'images': list(images.values()), 'annotations': annotations,
                      'categories': [{'id': 1, 'name': 'product'}], 'info': {}, 'licenses': []}
        (OUTPUT / 'annotations.json').write_text(json.dumps(annotation))
        marker.write_text(json.dumps({'source_identity': identity, 'images': len(images),
                                     'boxes': len(annotations), 'split': 'official test',
                                     'rules': 'All positive-size CSV boxes retained; no artificial minimum-box filter.'}, indent=2))
        print(marker.read_text())


if __name__ == '__main__':
    main()
