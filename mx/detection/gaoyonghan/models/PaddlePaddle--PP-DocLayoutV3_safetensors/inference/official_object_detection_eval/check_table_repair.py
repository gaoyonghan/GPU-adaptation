#!/usr/bin/env python3
"""Reproduce the original GriTS failure and verify the standard official metrics."""
import argparse
import copy
import json
import math
from pathlib import Path
import xml.etree.ElementTree as ET

import run_model as runner


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, help="Require >=400 real-image records for both models")
    args = parser.parse_args()
    old_root = Path("/data2/gaoyonghan/gpu_model_standalone_20261110/release/model/检测/PaddlePaddle--PP-DocLayoutV3_safetensors/resources/runtime/assets/workspace/results/object_detection_official_v3_repairs")
    dataset = old_root / "_prepared_datasets/pubtables_structure_test"
    stem = "PMC2845586_table_1"
    official = runner.load_table_official_eval(Path("/data2/gaoyonghan/gpu_model_standalone_20261110/release/model/检测/PaddlePaddle--PP-DocLayoutV3_safetensors/resources/runtime/assets/workspace/official_sources/object_detection"))
    objects = ET.parse(dataset / "annotations" / f"{stem}.xml").getroot().findall("object")
    raw_boxes = [[float(obj.findtext("bndbox/" + key)) for key in ("xmin", "ymin", "xmax", "ymax")] for obj in objects]
    assert all(box[2] > box[0] and box[3] > box[1] for box in raw_boxes)
    labels = [official.structure_class_map[obj.findtext("name")] for obj in objects]
    scores = [1.0] * len(labels)
    tokens = json.loads((dataset / "words" / f"{stem}_words.json").read_text())
    mutated_boxes = copy.deepcopy(raw_boxes)
    _, cells, _ = official.objects_to_cells(mutated_boxes, labels, scores, tokens,
                                           official.structure_class_names, official.structure_class_thresholds,
                                           official.structure_class_map)
    try:
        official.compute_metrics("grits-all", mutated_boxes, labels, scores, cells,
                                 mutated_boxes, labels, scores, cells)
    except ZeroDivisionError:
        print("PASS: sample 351 reproduces the original RawLoc division by zero")
    else:
        raise AssertionError("Expected original RawLoc failure was not reproduced")
    standard = official.compute_metrics("grits", raw_boxes, labels, scores, cells,
                                        raw_boxes, labels, scores, cells)
    extended = official.compute_metrics("grits-all", copy.deepcopy(raw_boxes), labels, scores, cells,
                                        copy.deepcopy(raw_boxes), labels, scores, cells)
    for name, value in standard.items():
        assert math.isfinite(value) and value == 1.0, (name, value)
        assert value == extended[name], name
    print("PASS: official standard metrics are finite, self-comparison=1, identical to grits-all core metrics")
    if args.results:
        for key in ("m06-table-transformer-structure-recognition", "m15-table-transformer-structure-recognition-v1-1-all"):
            records = runner.read_jsonl(args.results / key / "predictions.jsonl")
            by_id = {record["image_id"]: record for record in records}
            assert len(by_id) == len(records) and set(range(1, 401)) <= by_id.keys(), key
            for record in records:
                for name in standard:
                    value = record["grits"][name]
                    assert math.isfinite(value) and 0 <= value <= 1, (key, record["image_id"], name, value)
            previous = runner.read_jsonl(old_root / key / "predictions.jsonl")
            delta = max(abs(record["grits"][name] - by_id[record["image_id"]]["grits"][name])
                        for record in previous for name in standard)
            print(json.dumps({"model": key, "verified_images": len(records),
                              "old_350_max_core_metric_delta": delta}, ensure_ascii=False))


if __name__ == "__main__":
    main()
