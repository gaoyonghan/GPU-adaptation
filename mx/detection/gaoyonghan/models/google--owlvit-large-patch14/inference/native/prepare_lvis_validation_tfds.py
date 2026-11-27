#!/usr/bin/env python3
"""Materialize only LVIS validation with TFDS's unmodified LVIS generator."""

import argparse
import json
import pathlib
import types

import tensorflow_datasets as tfds


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--download-dir", required=True)
    parser.add_argument("--train-images-zip", required=True)
    parser.add_argument("--validation-images-zip", required=True)
    parser.add_argument("--validation-annotation-zip", required=True)
    parser.add_argument("--report", required=True)
    args = parser.parse_args()

    sources = {
        "train_images": pathlib.Path(args.train_images_zip).resolve(),
        "validation_images": pathlib.Path(args.validation_images_zip).resolve(),
        "validation_annotation": pathlib.Path(
            args.validation_annotation_zip).resolve(),
    }
    for name, path in sources.items():
        if not path.is_file():
            raise SystemExit(f"Missing {name}: {path}")

    builder = tfds.builder("lvis", data_dir=args.data_dir)

    def validation_only_split_generators(self, dl_manager):
        paths = dl_manager.extract({name: str(path) for name, path in sources.items()})
        image_dirs = [
            paths["train_images"] / "train2017",
            paths["validation_images"] / "val2017",
        ]
        return {
            tfds.Split.VALIDATION: self._generate_examples(
                image_dirs,
                paths["validation_annotation"] / "lvis_v1_val.json",
            )
        }

    # Only split selection is narrowed. Example generation, decoding, features,
    # class order, invalid-annotation handling and serialization stay upstream.
    builder._split_generators = types.MethodType(  # pylint: disable=protected-access
        validation_only_split_generators, builder)
    builder.download_and_prepare(download_dir=args.download_dir)

    count = builder.info.splits["validation"].num_examples
    classes = builder.info.features["objects"]["label"].num_classes
    if (count, classes) != (19809, 1203):
        raise ValueError(
            f"Prepared LVIS validation mismatch: images={count}, classes={classes}")

    report = {
        "status": "verified",
        "tfds_name": "lvis",
        "tfds_version": str(builder.version),
        "data_dir": str(builder.data_dir),
        "prepared_splits": {"validation": count},
        "categories": classes,
        "generation": "upstream TFDS LVIS Builder._generate_examples",
        "scope_adjustment": (
            "Materialized validation only; evaluator input records are unchanged."
        ),
        "source_archives": {name: str(path) for name, path in sources.items()},
    }
    report_path = pathlib.Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
