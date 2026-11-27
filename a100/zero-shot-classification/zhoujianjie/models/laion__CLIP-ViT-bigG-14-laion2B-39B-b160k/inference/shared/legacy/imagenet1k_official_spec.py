#!/usr/bin/env python3
"""Frozen, preflight-enforced ImageNet-1k claims for the B01 scope.

Only ``reference_status == verified`` entries may enter the zero-shot GPU
runner.  Values and protocols are checkpoint-specific official claims, not
scores copied from third-party benchmark tables.
"""

from __future__ import annotations


DATASET = {
    "repo_id": "ILSVRC/imagenet-1k",
    "revision": "49e2ee26f3810fb5a7536bbf732a7b07389a47b5",
    "split": "validation",
    "rows": 50_000,
    "parquet_shards": 14,
}

# Protocols:
# - clip_80: official OpenAI CLIP ImageNet class names and 80-template ensemble;
#   checkpoint-native official preprocessing.
# - bigvision_81: Google Big Vision canonicalized class names and 81-template
#   ensemble; resize(int(resolution/.875), square), center crop, value range
#   [-1, 1].
# - bigvision_naflex_256: the same text protocol with the official NaFlex
#   aspect-ratio-preserving processor and a maximum sequence length of 256
#   image patches.
# - align_knn: official Kakao ALIGN k-NN, ImageNet train reference bank and
#   ImageNet validation queries. It is deliberately not executable here.
MODEL_SPECS = {
    "laion/CLIP-ViT-H-14-laion2B-s32B-b79K": {
        "protocol": "clip_80", "resolution": 224, "official_top1_percent": 77.96,
        "reference_status": "verified",
        "evidence": "official LAION checkpoint card and OpenCLIP PRETRAINED.md, ImageNet zero-shot 78.0",
    },
    "laion/CLIP-ViT-bigG-14-laion2B-39B-b160k": {
        "protocol": "clip_80", "resolution": 224, "official_top1_percent": 80.09,
        "reference_status": "verified",
        "evidence": "official LAION checkpoint card and OpenCLIP README, ImageNet zero-shot 80.1",
    },
    "laion/CLIP-ViT-B-32-laion2B-s34B-b79K": {
        "protocol": "clip_80", "resolution": 224, "official_top1_percent": 66.56,
        "reference_status": "verified",
        "evidence": "official LAION checkpoint card and OpenCLIP PRETRAINED.md, ImageNet zero-shot 66.6",
    },
    "laion/CLIP-ViT-L-14-DataComp.XL-s13B-b90K": {
        "protocol": "clip_80", "resolution": 224, "official_top1_percent": 79.21,
        "reference_status": "verified",
        "evidence": "official LAION checkpoint card and OpenCLIP DataComp notes, ImageNet zero-shot 79.2",
    },
    "laion/CLIP-ViT-L-14-laion2B-s32B-b82K": {
        "protocol": "clip_80", "resolution": 224, "official_top1_percent": 75.25,
        "reference_status": "verified",
        "evidence": "official LAION checkpoint card and OpenCLIP PRETRAINED.md, ImageNet zero-shot 75.3",
    },
    "laion/CLIP-ViT-B-16-laion2B-s34B-b88K": {
        "protocol": "clip_80", "resolution": 224, "official_top1_percent": 70.23,
        "reference_status": "verified",
        "evidence": "official LAION checkpoint card, ImageNet zero-shot 70.2",
    },
    "laion/CLIP-ViT-g-14-laion2B-s34B-b88K": {
        "protocol": "clip_80", "resolution": 224, "official_top1_percent": 78.47,
        "reference_status": "verified",
        "evidence": "official LAION checkpoint card and OpenCLIP results, ImageNet zero-shot 78.4",
    },
    "laion/CLIP-convnext_xxlarge-laion2B-s34B-b82K-augreg-soup": {
        "protocol": "clip_80", "resolution": 256, "official_top1_percent": 79.47,
        "reference_status": "verified",
        "evidence": "official LAION checkpoint card, exact augreg-soup row, ImageNet zero-shot 79.4",
    },
    "apple/MobileCLIP-S2-OpenCLIP": {
        "protocol": "clip_80", "resolution": 256, "official_top1_percent": 74.4,
        "reference_status": "verified",
        "evidence": "official Apple checkpoint card and MobileCLIP README, S2 ImageNet zero-shot 74.4",
    },
    "facebook/metaclip-h14-fullcc2.5b": {
        "protocol": "clip_80", "resolution": 224, "official_top1_percent": 80.5,
        "reference_status": "verified",
        "evidence": "official MetaCLIP README, ViT-H-14-quickgelu metaclip_2_5b row, ImageNet 80.5",
    },
    "UCSC-VLAA/ViT-L-16-HTxt-Recap-CLIP": {
        "protocol": "clip_80", "resolution": 224, "official_top1_percent": 74.2,
        "reference_status": "verified",
        "evidence": (
            "official Recap-DataComp-1B paper Table 4, L/16 vision encoder + huge text "
            "encoder + recaption row; official model card maps this HF checkpoint to that variant"
        ),
    },
    "facebook/metaclip-2-worldwide-huge-quickgelu": {
        "protocol": "clip_80", "resolution": 224, "official_top1_percent": None,
        "reference_status": "missing_same_checkpoint_imagenet_reference",
        "evidence": (
            "official MetaCLIP README reports 57.4 for CVQA-LOCAL zero-shot accuracy, "
            "not ImageNet; no same-checkpoint ImageNet reference is published there"
        ),
    },
    "facebook/metaclip-b32-400m": {
        "protocol": "clip_80", "resolution": 224, "official_top1_percent": 65.5,
        "reference_status": "verified",
        "evidence": "official MetaCLIP repository README, ViT-B/32 400M ImageNet zero-shot",
    },
    "google/siglip-base-patch16-224": {
        "protocol": "bigvision_81", "resolution": 224, "official_top1_percent": 76.2,
        "reference_status": "verified", "evidence": "official SigLIP paper, Table 3 (B/16, 196 patches)",
    },
    "google/siglip-base-patch16-512": {
        "protocol": "bigvision_81", "resolution": 512, "official_top1_percent": 79.2,
        "reference_status": "verified", "evidence": "official SigLIP paper, Table 3 (B/16, 1024 patches)",
    },
    "google/siglip-so400m-patch14-224": {
        "protocol": "bigvision_81", "resolution": 224, "official_top1_percent": None,
        "reference_status": "missing_same_checkpoint_reference",
        "evidence": "no same-checkpoint official Google ImageNet value found; third-party tables are not accepted",
    },
    "google/siglip-so400m-patch14-384": {
        "protocol": "bigvision_81", "resolution": 384, "official_top1_percent": 83.2,
        "reference_status": "verified", "evidence": "official SigLIP paper, Table 3 (SO400M/14, 729 patches)",
    },
    "google/siglip2-base-patch16-224": {
        "protocol": "bigvision_81", "resolution": 224, "official_top1_percent": 78.2,
        "reference_status": "verified", "evidence": "official Big Vision SigLIP 2 README (B16 224)",
    },
    "google/siglip2-base-patch16-512": {
        "protocol": "bigvision_81", "resolution": 512, "official_top1_percent": 81.2,
        "reference_status": "verified", "evidence": "official Big Vision SigLIP 2 README (B16 512)",
    },
    "google/siglip2-base-patch32-256": {
        "protocol": "bigvision_81", "resolution": 256, "official_top1_percent": 74.0,
        "reference_status": "verified", "evidence": "official Big Vision SigLIP 2 README (B32 256)",
    },
    "google/siglip2-large-patch16-512": {
        "protocol": "bigvision_81", "resolution": 512, "official_top1_percent": 83.5,
        "reference_status": "verified", "evidence": "official Big Vision SigLIP 2 README (L16 512)",
    },
    "google/siglip2-so400m-patch14-384": {
        "protocol": "bigvision_81", "resolution": 384, "official_top1_percent": 84.1,
        "reference_status": "verified", "evidence": "official Big Vision SigLIP 2 README (SO400M/14 384)",
    },
    "google/siglip2-so400m-patch16-512": {
        "protocol": "bigvision_81", "resolution": 512, "official_top1_percent": 84.3,
        "reference_status": "verified", "evidence": "official Big Vision SigLIP 2 README (SO400M/16 512)",
    },
    "google/siglip2-base-patch16-naflex": {
        "protocol": "bigvision_naflex_256", "resolution": "variable",
        "max_num_patches": 256, "official_top1_percent": 78.5,
        "reference_status": "verified",
        "evidence": "official Big Vision SigLIP 2 README (NaFlex B/16, sequence length 256)",
    },
    "google/siglip2-so400m-patch16-naflex": {
        "protocol": "bigvision_naflex_256", "resolution": "variable",
        "max_num_patches": 256, "official_top1_percent": 83.5,
        "reference_status": "verified",
        "evidence": "official Big Vision SigLIP 2 README (NaFlex SO400M/16, sequence length 256)",
    },
    "google/siglip2-giant-opt-patch16-384": {
        "protocol": "bigvision_81", "resolution": 384, "official_top1_percent": 85.0,
        "reference_status": "verified",
        "evidence": "official Big Vision SigLIP 2 README (g-opt/16 384)",
    },
    "kakaobrain/align-base": {
        "protocol": "align_knn", "resolution": 289, "official_top1_percent": 68.6,
        "reference_status": "requires_imagenet_train_reference_bank",
        "evidence": (
            "official Kakao Brain COYO-ALIGN repository at commit "
            "8cea82e5381b7c4806967746473c6053b6817905: normalized image-embedding "
            "1-NN on ImageNet train/validation, Top-1 68.6"
        ),
    },
    "laion/CLIP-convnext_large_d_320.laion2B-s29B-b131K-ft-soup": {
        "protocol": "clip_80", "resolution": 320, "official_top1_percent": 76.9,
        "reference_status": "verified", "evidence": "official LAION checkpoint model card, ImageNet zero-shot 76.9",
    },
    "openai/clip-vit-base-patch16": {
        "protocol": "clip_80", "resolution": 224, "official_top1_percent": 68.6,
        "reference_status": "verified", "evidence": "official OpenAI CLIP paper/model results (ViT-B/16)",
    },
    "openai/clip-vit-base-patch32": {
        "protocol": "clip_80", "resolution": 224, "official_top1_percent": 63.2,
        "reference_status": "verified", "evidence": "official OpenAI CLIP paper/model results (ViT-B/32)",
    },
    "openai/clip-vit-large-patch14-336": {
        "protocol": "clip_80", "resolution": 336, "official_top1_percent": 76.2,
        "official_top5_percent": 95.0,
        "reference_status": "verified", "evidence": "official OpenAI CLIP paper (ViT-L/14@336px)",
        "top5_evidence": (
            "official OpenAI CLIP paper, Section 3.1.4: ViT-L/14@336px "
            "ImageNet zero-shot top-5 accuracy 95%"
        ),
    },
}


def official_metric_claims(model_id: str) -> dict[str, dict[str, object]]:
    """Return published same-checkpoint ImageNet claims and explicit gaps."""
    spec = MODEL_SPECS[model_id]
    top5 = spec.get("official_top5_percent")
    return {
        "top1_percent": {
            "reference_status": spec["reference_status"],
            "official_value": spec["official_top1_percent"],
            "evidence": spec["evidence"],
        },
        "top5_percent": {
            "reference_status": (
                "verified" if top5 is not None else "missing_same_checkpoint_official_reference"
            ),
            "official_value": top5,
            "evidence": spec.get(
                "top5_evidence",
                "official source reports no same-checkpoint ImageNet top-5 reference value",
            ),
        },
    }


def verified_zero_shot_models() -> tuple[str, ...]:
    return tuple(
        model_id for model_id, spec in MODEL_SPECS.items()
        if spec["reference_status"] == "verified"
        and spec["protocol"] in {"clip_80", "bigvision_81", "bigvision_naflex_256"}
    )


if len(MODEL_SPECS) != 31 or len(verified_zero_shot_models()) != 28:
    raise RuntimeError("ImageNet official scope must be 28 runnable + 2 missing reference + 1 k-NN")
