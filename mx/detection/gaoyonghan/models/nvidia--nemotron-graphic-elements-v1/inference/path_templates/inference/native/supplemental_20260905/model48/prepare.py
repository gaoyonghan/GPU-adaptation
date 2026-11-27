"""Retrieve the published CACHED validation images with bounded HTTP ranges."""
import collections
import hashlib
import io
import json
import struct
import urllib.parse
import urllib.request
import zlib
from pathlib import Path
from PIL import Image

HERE = Path(__file__).resolve().parent
SOURCES = HERE.parents[1] / 'review_20260905/model48_sources'
OUT = Path('__DATASET_ROOT__/supplemental_20260905/model48')
LABELS = ['chart_title', 'x_title', 'y_title', 'xlabel', 'ylabel', 'other',
          'legend_label', 'legend_title', 'mark_label', 'value_label']
EXPECTED = [38, 437, 505, 4091, 3944, 464, 1077, 19, 219, 726]


def sha(data):
    return hashlib.sha256(data).hexdigest()


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'images').mkdir(exist_ok=True)
    source = (SOURCES / 'CACHED_val.json').read_bytes()
    assert sha(source) == '403e298f1bbe6f36f1aa3741a6cf9a65dd7ce85975193622c3e15d860f6b436b'
    original = json.loads(source)
    entries = json.loads((SOURCES / 'CACHED_val_image_entries.json').read_text())
    assert len(entries) == len(original['images']) == 560
    start = min(e['offset'] for e in entries)
    end = max(e['offset'] + e['compressed'] + 30 + len(e['name'].encode()) + 128 for e in entries)
    archive_part = OUT / 'cached_val_images.range.bin'
    url = 'https://drive.usercontent.google.com/download?' + urllib.parse.urlencode({
        'id': '1YZmRy9TcaOJ6tA8DLeG0sXhSjwqqAw_h', 'export': 'download', 'confirm': 't'})
    if not archive_part.exists():
        request = urllib.request.Request(url, headers={'Range': f'bytes={start}-{end-1}'})
        with urllib.request.urlopen(request, timeout=120) as response:
            assert response.status == 206, (response.status, response.headers)
            assert response.headers['Content-Range'] == f'bytes {start}-{end-1}/2125251674'
            data = response.read(end - start + 1)
        assert len(data) == end - start
        archive_part.write_bytes(data)
    block = archive_part.read_bytes()
    assert len(block) == end - start
    image_info = {i['file_name']: i for i in original['images']}
    checks = []
    for e in entries:
        pos = e['offset'] - start
        header = struct.unpack_from('<4s5H3I2H', block, pos)
        assert header[0] == b'PK\x03\x04' and header[3] == e['method'] == 8
        name = block[pos+30:pos+30+header[-2]].decode()
        assert name == e['name']
        offset = pos + 30 + header[-2] + header[-1]
        raw = zlib.decompress(block[offset:offset+e['compressed']], -15)
        assert len(raw) == e['size'] and zlib.crc32(raw) == e['crc']
        filename = Path(name).name
        info = image_info[filename]
        with Image.open(io.BytesIO(raw)) as image:
            assert image.size == (info['width'], info['height'])
            image.convert('RGB').load()
        (OUT / 'images' / filename).write_bytes(raw)
        checks.append({'file_name': filename, 'sha256': sha(raw), 'bytes': len(raw), 'crc32': e['crc']})
    old_categories = {c['id']: c['name'] for c in original['categories']}
    annotations = []
    for a in original['annotations']:
        label = old_categories[a['category_id']]
        if label in LABELS:
            annotations.append(dict(a, category_id=LABELS.index(label)+1))
    counts = collections.Counter(a['category_id'] for a in annotations)
    assert [counts[i+1] for i in range(10)] == EXPECTED
    assert len(annotations) == 11520 and len({a['id'] for a in annotations}) == 11520
    annotation = {'images': original['images'], 'annotations': annotations,
                  'categories': [{'id': i+1, 'name': n} for i, n in enumerate(LABELS)],
                  'info': {'description': 'Published CACHED val; unchanged boxes; NVIDIA native 10 categories'},
                  'licenses': original.get('licenses', [])}
    encoded = json.dumps(annotation, ensure_ascii=False).encode()
    (OUT / 'original_CACHED_val.json').write_bytes(source)
    (OUT / 'annotations_native10.json').write_bytes(encoded)
    info_by_id = {i['id']: i for i in original['images']}
    outside = []
    for a in annotations:
        x, y, w, h = a['bbox']; im = info_by_id[a['image_id']]
        assert w > 0 and h > 0
        if x < 0 or y < 0 or x+w > im['width'] or y+h > im['height']:
            outside.append({'annotation_id': a['id'], 'image_id': a['image_id'], 'bbox': a['bbox']})
    manifest = {'source_url': 'https://github.com/pengyu965/ChartDete', 'archive_url': url,
                'archive_bytes': 2125251674, 'archive_member': 'pmc_2022/pmc_coco/element_detection/val.json',
                'image_range': [start, end-1], 'image_range_sha256': sha(block),
                'source_annotation_sha256': sha(source), 'annotation_sha256': sha(encoded),
                'images': 560, 'boxes': 11520, 'native10_counts': dict(zip(LABELS, EXPECTED)),
                'conversion': 'Keep published 560 images and native 10-class boxes; remap category IDs only. No bbox changes.',
                'published_out_of_bounds_gt_retained': outside, 'image_checks': checks,
                'verified': 'All 560 images passed ZIP CRC32, dimensions, RGB decode and SHA256 recording.'}
    (OUT / 'source_manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    print(json.dumps({k: v for k, v in manifest.items() if k not in ['image_checks', 'published_out_of_bounds_gt_retained']}, indent=2))


if __name__ == '__main__':
    main()
