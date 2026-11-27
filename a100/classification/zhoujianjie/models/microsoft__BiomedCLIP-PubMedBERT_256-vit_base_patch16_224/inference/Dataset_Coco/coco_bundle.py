#!/usr/bin/env python3
"""Validate and score the shared COCO-style classification package.

The package uses explicit category and sample IDs in annotations.json, plus
predictions.json and protocol.json. It does not convert the original images.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

BUNDLE = Path(__file__).resolve().parents[1]
if str(BUNDLE) not in sys.path:
    sys.path.insert(0, str(BUNDLE))

from shared.classification_contract import load_bundle
from shared.standard_scoring import evaluate, export_jsonl, records

__all__ = ["evaluate", "export_jsonl", "load_bundle", "records"]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["validate", "evaluate"])
    parser.add_argument("bundle", type=Path)
    args = parser.parse_args()
    if args.action == "validate":
        annotations, predictions, protocol = load_bundle(args.bundle)
        result = {"status": "passed", "samples": len(annotations["samples"]),
                  "predictions": len(predictions), "schema": protocol["schema"]}
    else:
        result = evaluate(args.bundle)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
