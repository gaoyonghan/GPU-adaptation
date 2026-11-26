"""Validate original competition ZIPs without changing the published labels."""
import collections
import hashlib
import io
import json
from pathlib import Path
import zipfile
from PIL import Image

DATA = Path('__DATASET_ROOT__/supplemental_20260905/model19')
BASE = 'https://ds4sd-icdar23-doclaynet-competition.s3.eu-de.cloud-object-storage.appdomain.cloud/'


def sha(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def save(name, value):
    path = DATA / name
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    return sha(path)


def main():
    files = ['competition-dataset-public.zip', 'competition_dataset_groundtruth.zip']
    archives = {name: zipfile.ZipFile(DATA / name) for name in files}
    for name, archive in archives.items():
        assert not (DATA / (name + '.aria2')).exists(), 'download incomplete'
        assert archive.testzip() is None, name
    gt_archive = archives[files[1]]
    original = gt_archive.read('coco.json')
    (DATA / 'coco.original.json').write_bytes(original)
    annotation = json.loads(original)
    public = json.loads(archives[files[0]].read('coco.json'))
    assert annotation['images'] == public['images'] and annotation['categories'] == public['categories']
    assert len(annotation['images']) == 498 and len(annotation['annotations']) == 9158 and len(annotation['categories']) == 11
    ids = {i['id'] for i in annotation['images']}
    assert len(ids) == 498 and len({a['id'] for a in annotation['annotations']}) == 9158
    assert all(a['image_id'] in ids and a['category_id'] in range(1, 12) and a['iscrowd'] == 0 for a in annotation['annotations'])
    image_members = {}
    image_hashes = {}
    for image in annotation['images']:
        member = 'PNG/' + image['file_name']
        image_members[image['file_name']] = member
        raw = archives[files[0]].read(member)
        assert raw == gt_archive.read(member), member
        with Image.open(io.BytesIO(raw)) as opened:
            opened.load()
            assert opened.size == (image['width'], image['height']) == (1025, 1025)
        image_hashes[image['file_name']] = hashlib.sha256(raw).hexdigest()
    metadata = {n: json.loads(gt_archive.read(n))['metadata'] for n in gt_archive.namelist() if n.startswith('JSON/') and n.endswith('.json')}
    category_counts = dict(collections.Counter(m['doc_category'] for m in metadata.values()))
    assert len(metadata) == 498
    # Published coco image records omit domains; accompanying JSON labels every page 'other'.
    assert not any('doc_category' in i for i in annotation['images']) and category_counts == {'other': 498}
    domain_mapping = {str(i['id']): None for i in annotation['images']}
    annotation['info'] = {'source': BASE + files[1], 'note': 'Only COCO info/licenses defaults added; original images/annotations/categories unchanged.'}
    annotation['licenses'] = []
    annotation_hash = save('annotation.json', annotation)
    save('domain_mapping.json', domain_mapping)
    save('published_json_metadata.json', metadata)
    save('image_sha256.json', image_hashes)
    report = {
        'sources': {name: {'url': BASE + name, 'size': (DATA/name).stat().st_size, 'sha256': sha(DATA/name)} for name in files},
        'images': 498, 'annotations': 9158, 'categories': 11, 'image_archive': files[0],
        'image_members': image_members, 'annotation_sha256': annotation_hash,
        'original_annotation_sha256': hashlib.sha256(original).hexdigest(),
        'public_and_GT_images_identical': True, 'public_and_GT_categories_identical': True,
        'all_PNG_payloads_identical_between_ZIPs': True, 'all_images_decode': True,
        'positive_GT_by_class': dict(collections.Counter(a['category_id'] for a in annotation['annotations'])),
        'domain_mapping_state': 'not_published_in_downloaded_official_archives',
        'published_metadata_doc_category_counts': category_counts,
        'expected_official_domain_counts': {'reports': 293, 'manuals': 114, 'patents': 48, 'others': 43},
        'domain_note': 'COCO image records omit doc_category and all 498 auxiliary JSON metadata rows say other. Do not invent four-domain membership.',
        'license_in_archive': 'Community Data License Agreement - Permissive - Version 1.0',
    }
    save('prepared.json', report)
    for archive in archives.values():
        archive.close()
    print(json.dumps({k:v for k,v in report.items() if k not in ['image_members','sources']}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
