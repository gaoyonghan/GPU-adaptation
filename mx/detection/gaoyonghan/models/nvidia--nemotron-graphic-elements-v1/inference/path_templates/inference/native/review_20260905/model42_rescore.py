"""CPU diagnostics from frozen predictions; no model loading or parameter search."""
import hashlib
import json
import zipfile
from collections import defaultdict
from pathlib import Path

import numpy as np

RESULT = Path('__MODEL_ROOT__/resources/runtime/assets/workspace/results/remaining25_cmarkea_v1')
ARCHIVE = Path('__DATASET_ROOT__/doclaynet_test/DocLayNet_core.zip')
OUTPUT = Path(__file__).with_name('model42_supplemental_metrics.json')


def giou_xywh(a, b):
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    a[:, 2:] += a[:, :2]
    b[:, 2:] += b[:, :2]
    intersection = np.maximum(0, np.minimum(a[:, None, 2:], b[None, :, 2:])
                              - np.maximum(a[:, None, :2], b[None, :, :2])).prod(axis=2)
    area_a = (a[:, 2:] - a[:, :2]).prod(axis=1)
    area_b = (b[:, 2:] - b[:, :2]).prod(axis=1)
    union = area_a[:, None] + area_b[None, :] - intersection
    enclosing = (np.maximum(a[:, None, 2:], b[None, :, 2:])
                 - np.minimum(a[:, None, :2], b[None, :, :2])).prod(axis=2)
    return intersection / union - (enclosing - union) / enclosing


def main():
    with zipfile.ZipFile(ARCHIVE) as z:
        raw_gt = z.read('COCO/test.json')
    gt = json.loads(raw_gt)
    by_image = defaultdict(list)
    for ann in gt['annotations']:
        by_image[ann['image_id']].append(ann)
    # 0.0 = original saved detection protocol; 0.4 = card's usage example.
    # Neither is claimed to be the author's undisclosed evaluation threshold.
    totals = {t: {'gt_count': 0, 'pred_count': 0, 'giou_sum': 0.0,
                  'class_correct': 0, 'empty_prediction_pages': 0} for t in (0.0, 0.4)}
    pixel = np.zeros((12, 3), dtype=np.int64)
    seen = set()
    prediction_hash = hashlib.sha256()
    with (RESULT / 'predictions.jsonl').open('rb') as handle:
        for line in handle:
            prediction_hash.update(line)
            row = json.loads(line)
            assert row['image_id'] not in seen
            seen.add(row['image_id'])
            anns = by_image[row['image_id']]
            pixel += np.asarray(row['pixel_tp_fp_fn'], dtype=np.int64)
            for threshold, total in totals.items():
                preds = [p for p in row['detections'] if p['score'] > threshold]
                total['gt_count'] += len(anns)
                total['pred_count'] += len(preds)
                if not preds:
                    total['empty_prediction_pages'] += 1
                    # Diagnostic convention: unmatched GT receives GIoU 0, class wrong.
                    continue
                if not anns:
                    continue
                matrix = giou_xywh([a['bbox'] for a in anns], [p['bbox'] for p in preds])
                best = matrix.argmax(axis=1)
                total['giou_sum'] += float(matrix[np.arange(len(anns)), best].sum())
                total['class_correct'] += sum(
                    a['category_id'] == preds[j]['category_id'] for a, j in zip(anns, best))
    assert seen == {i['id'] for i in gt['images']}
    saved = json.loads((RESULT / 'metrics.json').read_text())
    assert pixel.tolist() == saved['pixel_counts']
    f1 = 2 * pixel[:, 0] / (2 * pixel[:, 0] + pixel[:, 1] + pixel[:, 2])
    assert abs(float(f1[1:].mean()) - saved['semantic_macro_foreground_F1']) < 1e-12
    for total in totals.values():
        total['mean_best_giou'] = total['giou_sum'] / total['gt_count']
        total['class_accuracy_at_best_giou'] = total['class_correct'] / total['gt_count']
    semantic = {}
    for name, counts in [('all_classes_including_background', pixel), ('foreground', pixel[1:])]:
        tp, fp, fn = counts.sum(axis=0)
        semantic['micro_F1_' + name] = float(2 * tp / (2 * tp + fp + fn))
    semantic['macro_F1_foreground'] = float(f1[1:].mean())
    semantic['support_weighted_F1_all_classes_including_background'] = float(
        (f1 * (pixel[:, 0] + pixel[:, 2])).sum() / (pixel[:, 0] + pixel[:, 2]).sum())
    result = {
        'sequence': 42, 'unit': 'fraction_0_to_1', 'evaluation_images': len(seen),
        'new_inference': False, 'device': 'CPU',
        'source_result_directory': str(RESULT), 'prediction_sha256': prediction_hash.hexdigest(),
        'original_metrics_sha256': hashlib.sha256((RESULT / 'metrics.json').read_bytes()).hexdigest(),
        'annotation_member_sha256': hashlib.sha256(raw_gt).hexdigest(),
        'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'source_protocol_fingerprint': saved['protocol_fingerprint'],
        'author_matching_source': 'https://huggingface.co/cmarkea/detr-layout-detection/discussions/1',
        'bbox_protocol': 'For each GT independently choose prediction with maximum GIoU, regardless of class; report GT-weighted mean GIoU and accuracy of the chosen prediction. No one-to-one constraint, NMS, clipping, or extra-FP penalty. Empty predictions: GIoU=0, class incorrect.',
        'limitations': ['Author 500-page list unknown; ours is official ordinary test4999.',
                       'Author exact score threshold and aggregation not confirmed. Both fixed diagnostic thresholds reported; no selection by closeness.',
                       'Pixel micro-F1 is computed from existing independent per-class masks; author overlap and aggregation conventions remain unknown.'],
        'semantic': semantic, 'bbox_diagnostics_by_score_threshold': totals,
    }
    OUTPUT.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'semantic': semantic, 'bbox': totals}, indent=2))


if __name__ == '__main__':
    main()
