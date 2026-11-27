#!/usr/bin/env python3
"""Download the two auxiliary checkpoints required by the unmodified official eval config."""
import hashlib
import os
from pathlib import Path

os.environ["HF_HUB_OFFLINE"] = "0"
from huggingface_hub import get_hf_file_metadata, get_token, hf_hub_download, hf_hub_url

SOURCE = Path(os.environ.get("OFFICIAL_SOURCE_ROOT", "/data2/gaoyonghan/gpu_model_standalone_20261110/release/model/检测/nickmuchi--yolos-small-finetuned-license-plate-detection/resources/runtime/assets/workspace/official_sources/object_detection")) / "WildDet3D"
MODEL = Path(os.environ.get("MODEL_ROOT", "/data2/gaoyonghan/gpu_model_standalone_20261110/release/model/检测/nickmuchi--yolos-small-finetuned-license-plate-detection/resources/model_index"))
WEIGHTS = [
    ("facebook/sam3", "sam3.pt", "3c879f39826c281e95690f02c7821c4de09afae7", "9999e2341ceef5e136daa386eecb55cb414446a00ac2b55eb2dfd2f7c3cf8c9e", "pretrained/sam3/sam3_detector.pt"),
    ("robbyant/lingbot-depth-postrain-dc-vitl14", "model.pt", "d1b06f7538d4b56387412799c528542c795df169", "3a4512e05445857d1404fc00227285bd8fa4abdd97250dc65f9636aa9cc71325", "pretrained/lingbot-depth/postrain-dc-vitl14/model.pt"),
]

def main():
    # Check access before downloading gigabytes; never bypass gated model access.
    for repo, filename, revision, _, _ in WEIGHTS:
        if not (MODEL / repo / filename).is_file():
            get_hf_file_metadata(hf_hub_url(repo, filename, revision=revision), token=get_token())
    for repo, filename, revision, expected, relative in WEIGHTS:
        path = MODEL / repo / filename
        if not path.is_file():
            path = Path(hf_hub_download(repo, filename, revision=revision, local_dir=MODEL / repo))
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                digest.update(block)
        if digest.hexdigest() != expected:
            raise RuntimeError(f"Refusing mismatched official checkpoint: {path}")
        link = SOURCE / relative
        link.parent.mkdir(parents=True, exist_ok=True)
        if link.exists() or link.is_symlink():
            if link.resolve() != path.resolve():
                raise RuntimeError(f"Refusing to overwrite existing path: {link}")
        else:
            link.symlink_to(path)
        print(f"Verified {repo}@{revision}: {link}")

if __name__ == "__main__":
    main()
