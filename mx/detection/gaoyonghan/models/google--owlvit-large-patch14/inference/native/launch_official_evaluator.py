#!/usr/bin/env python3
"""Launch the upstream evaluator with a verified local CLIP vocabulary."""

import hashlib
import os
import pathlib
import runpy

from scenic.projects.owl_vit.clip import tokenizer


EXPECTED_SHA256 = "924691ac288e54409236115652ad4aa250f48203de50a9e4722a6ecd48d6804a"


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


bpe_path = pathlib.Path(os.environ["OWL_VIT_BPE_PATH"]).resolve()
if not bpe_path.is_file():
    raise SystemExit(f"CLIP vocabulary not found: {bpe_path}")
actual_sha256 = sha256(bpe_path)
if actual_sha256 != EXPECTED_SHA256:
    raise SystemExit(
        f"CLIP vocabulary SHA256 mismatch: {actual_sha256} != {EXPECTED_SHA256}")

# The 2022 helper re-downloads the vocabulary even when an identical file is
# already present. Change only the bound default path, then execute the
# upstream evaluator module unchanged as __main__.
original_build_tokenizer = tokenizer.build_tokenizer.__wrapped__
original_build_tokenizer.__defaults__ = (
    str(bpe_path), *original_build_tokenizer.__defaults__[1:])
runpy.run_module("scenic.projects.owl_vit.evaluator", run_name="__main__")
