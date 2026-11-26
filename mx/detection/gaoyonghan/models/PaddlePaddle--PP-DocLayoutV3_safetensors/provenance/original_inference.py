"""V3 local HF inference, preserving native IDs for official Omni detection scoring."""
import argparse
import fcntl
from importlib.metadata import version
import json
from pathlib import Path
import shutil
import time

import numpy as np
from PIL import Image
import yaml
from run_model import core, digest, file_hash

HERE = Path(__file__).resolve().parent
MODEL = Path('/data2/gaoyonghan/gpu/model/PaddlePaddle/PP-DocLayoutV3_safetensors')
DATA = Path('/data2/gaoyonghan/gpu/dataset/omnidocbench_eval')
OUT = Path('/home/ubuntu/gaoyonghan/gpu/results/m39_omni_native_v1')
THRESHOLD = 0.001


def preflight():
    from transformers.models.pp_doclayout_v3 import image_processing_pp_doclayout_v3_fast as processing
    from transformers.models.pp_doclayout_v3 import modeling_pp_doclayout_v3 as modeling
    old_path = Path('/home/ubuntu/gaoyonghan/gpu/results/remaining25_pp_v2/PP-DocLayoutV3_safetensors/status.json')
    old = json.loads(old_path.read_text())['protocol']
    weights = {n: file_hash(MODEL/n) for n in old['weights']}
    assert weights == old['weights'], 'V3 weights/config changed since previous run'
    assert file_hash(DATA/'OmniDocBench.json') == old['dataset_sha256']
    assert file_hash(Path(processing.__file__)) == old['postprocessor_sha256'], 'Reviewed postprocessor changed'
    config = json.loads((MODEL/'config.json').read_text())
    labels = yaml.safe_load((MODEL/'inference.yml').read_text())['label_list']
    assert len(labels) == 25 and labels[5] == 'display_formula' and labels[15] == 'inline_formula'
    assert config['id2label']['5'] == config['id2label']['15'] == 'formula'
    versions = {n: version(n) for n in ['transformers','torch','torchvision','numpy','Pillow','safetensors','PyYAML']}
    assert versions['transformers'] == '5.3.0' and versions['torch'] == '2.5.1+cu121'
    pages = json.loads((DATA/'OmniDocBench.json').read_text())
    assert len(pages) == 1651
    images = []
    print('39号CPU数据/文件预检开始，不执行模型前向。', flush=True)
    for i, page in enumerate(pages, 1):
        info = page['page_info']; path = DATA/'images'/Path(info['image_path']).name
        with Image.open(path) as image:
            assert image.size == (info['width'], info['height'])
        images.append(dict(id=i, path=str(path), sha256=file_hash(path)))
        if i % 250 == 0 or i == len(pages): print(f'图像校验 {i}/1651', flush=True)
    assert len({p['path'] for p in images}) == 1651
    sources = [Path(__file__), HERE/'run_m39_omni.sh', HERE/'run_model.py', HERE/'adapters.py',
               HERE.parent/'official_object_detection_eval/run_model.py',
               HERE.parent/'official_object_detection_eval/download_official_source.py',
               Path(processing.__file__), Path(modeling.__file__)]
    protocol = dict(sequence=39, model_id='PaddlePaddle/PP-DocLayoutV3_safetensors',
        dataset_sha256=file_hash(DATA/'OmniDocBench.json'), model_files=weights, images=images,
        id2label=config['id2label'], canonical_labels_by_id=labels, software=versions,
        source_sha256={str(p): file_hash(p) for p in sources},
        prediction_options=dict(threshold=THRESHOLD, batch_size=1, use_fast=True, device='cuda:0', dtype='float32', tf32=False),
        native_entry='AutoImageProcessor -> AutoModelForObjectDetection -> processor.post_process_object_detection',
        preprocessing='Local official preprocessor_config.json, RGB PIL, resize 800x800; no manual resizing or image changes.',
        label_resolution='Use native numeric class ID and official inference.yml label_list; preserve original config label as well. No model config edits.',
        postprocessing='Reuse existing reviewed Transformers 5.3.0 postprocessor; no extra project NMS or box merging. Threshold .001 is a project AP baseline choice, not claimed as author benchmark threshold.',
        existing_patch='Existing empty/degenerate polygon-mask rectangle fallback is retained and source-hashed. It does not change returned bbox/score/label tensors. This run scores bbox only, not polygon quality.',
        mask_threshold='This library version uses the same threshold for masks and detection scores; both .001. Saved polygons are diagnostics, not a new mask benchmark.',
        scope='Official Omni detection subset shared with accepted model 38; segmentation and reading-order results remain historical, not newly evaluated.',
        dataset_relation='author_used_public_benchmark_family_with_project_detection_subset', official_comparable=False,
        old_results_preserved=['remaining25_pp_v2','remaining25_pp_v3'],
        reason_for_new_inference='Old predictions discarded native IDs and some categories; cannot recover display/inline formula separation or formula-number predictions.')
    return config, labels, protocol


def load_model():
    from transformers import AutoImageProcessor, AutoModelForObjectDetection
    processor = AutoImageProcessor.from_pretrained(MODEL, local_files_only=True, use_fast=True)
    model, info = AutoModelForObjectDetection.from_pretrained(MODEL, local_files_only=True, output_loading_info=True)
    assert not info['missing_keys'] and not info['unexpected_keys'] and not info['mismatched_keys'], info
    assert processor.size == {'height':800, 'width':800}
    return processor, model.eval()


def serialize(result, config, labels):
    ids = result['labels'].detach().cpu().tolist()
    scores = result['scores'].detach().cpu().tolist()
    coords = result['boxes'].detach().cpu().tolist()
    orders = result['order_seq'].detach().cpu().tolist()
    polys = result['polygon_points']
    assert len(ids) == len(scores) == len(coords) == len(orders) == len(polys)
    boxes = []
    for cid, score, coordinate, order, poly in zip(ids, scores, coords, orders, polys):
        assert isinstance(cid, int) and 0 <= cid < 25
        assert np.isfinite([score, *coordinate, order]).all() and 0 < score <= 1
        assert coordinate[2] > coordinate[0] and coordinate[3] > coordinate[1]
        polygon = np.asarray(poly).reshape(-1, 2)
        assert np.isfinite(polygon).all() and len(polygon) >= 3
        boxes.append(dict(native_class_id=cid, config_label=config['id2label'][str(cid)], label=labels[cid],
                          score=score, coordinate=coordinate, order=order, polygon_points=polygon.tolist()))
    return boxes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--max-images', type=int, choices=[0,10], default=0)
    parser.add_argument('--preflight', action='store_true')
    args = parser.parse_args()
    config, labels, protocol = preflight()
    if args.preflight:
        processor, model = load_model()
        assert next(model.parameters()).device.type == 'cpu'
        print('ready: 1651 images, native ID mapping and strict CPU checkpoint load verified; NO forward executed.')
        return
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT/'.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        fingerprint = digest(protocol)
        status_path = OUT/'status.json'
        old = json.loads(status_path.read_text()) if status_path.exists() else None
        if old and old['protocol_fingerprint'] != fingerprint: raise RuntimeError('Changed protocol/data; do not mix with saved predictions.')
        if old and old['state'] == 'complete': print('39号1651页推理已完成，跳过。'); return
        core.atomic_json(OUT/'protocol.json', protocol)
        snapshot = OUT/'source_snapshot'; snapshot.mkdir(exist_ok=True)
        for i, name in enumerate(protocol['source_sha256']): shutil.copy2(name, snapshot/f'{i:03d}_{Path(name).name}')
        state = dict(sequence=39, state='running', protocol_fingerprint=fingerprint, started_at=time.time())
        core.atomic_json(status_path, state)
        try:
            records = core.read_jsonl(OUT/'native_predictions.jsonl')
            existing = {r['image_id'] for r in records}
            assert existing <= set(range(1,1652))
            for record in records:
                assert record['image_sha256'] == protocol['images'][record['image_id']-1]['sha256']
                assert all(b['label'] == labels[b['native_class_id']] for b in record['boxes'])
            limit = args.max_images or 1651
            if not set(range(1,limit+1)) <= existing:
                import torch
                assert torch.cuda.is_available(), 'No CPU inference fallback'
                torch.backends.cuda.matmul.allow_tf32 = False
                torch.backends.cudnn.allow_tf32 = False
                processor, model = load_model(); model = model.to('cuda:0')
                for info in protocol['images'][:limit]:
                    if info['id'] in existing: continue
                    print(f'39号 {info["id"]}/{limit}: {Path(info["path"]).name}', flush=True)
                    with Image.open(info['path']) as source: image = source.convert('RGB')
                    inputs = processor(images=image, return_tensors='pt').to('cuda:0')
                    with torch.inference_mode(): outputs = model(**inputs)
                    result = processor.post_process_object_detection(outputs, threshold=THRESHOLD,
                        target_sizes=[(image.height,image.width)])[0]
                    boxes = serialize(result, config, labels)
                    core.append_jsonl(OUT/'native_predictions.jsonl', dict(image_id=info['id'], image_path=info['path'],
                        image_sha256=info['sha256'], boxes=boxes))
                    existing.add(info['id'])
                    image.close()
            assert set(range(1,limit+1)) <= existing
            state.update(state='smoke_complete' if args.max_images else 'complete', completed_images=len(existing), total_images=1651)
            print(f'39号推理保存完成：{OUT}', flush=True)
        except BaseException as exc:
            state.update(state='failed', error=repr(exc)); raise
        finally:
            state['elapsed_seconds'] = time.time()-state['started_at']
            core.atomic_json(status_path, state)


if __name__ == '__main__': main()
