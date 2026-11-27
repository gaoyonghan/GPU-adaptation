#!/usr/bin/env python3
"""Canonical scope for the 22 B01-smoked non-specialty dual encoders."""

from __future__ import annotations


IMAGENET_ZERO_SHOT_MODELS = (
    "facebook/metaclip-b32-400m",
    "google/siglip-base-patch16-224",
    "google/siglip-base-patch16-512",
    "google/siglip-so400m-patch14-224",
    "google/siglip-so400m-patch14-384",
    "google/siglip2-base-patch16-224",
    "google/siglip2-base-patch16-512",
    "google/siglip2-base-patch32-256",
    "google/siglip2-large-patch16-512",
    "google/siglip2-so400m-patch14-384",
    "google/siglip2-so400m-patch16-512",
    "laion/CLIP-convnext_large_d_320.laion2B-s29B-b131K-ft-soup",
    "openai/clip-vit-base-patch16",
    "openai/clip-vit-base-patch32",
    "openai/clip-vit-large-patch14-336",
)

# The official kakaobrain chart reports ImageNet k-NN (68.6), not zero-shot.
# Reproducing it requires the ImageNet train split as the reference bank and
# validation as the query split, so it must never enter the zero-shot runner.
IMAGENET_KNN_MODELS = ("kakaobrain/align-base",)
IMAGENET_MODELS = IMAGENET_ZERO_SHOT_MODELS + IMAGENET_KNN_MODELS

CHINESE_MODELS = (
    "OFA-Sys/chinese-clip-vit-base-patch16",
    "OFA-Sys/chinese-clip-vit-large-patch14",
    "OFA-Sys/chinese-clip-vit-large-patch14-336px",
)

MULTILINGUAL_MODELS = (
    "BAAI/AltCLIP",
    "google/siglip-base-patch16-256-multilingual",
    "google/siglip-so400m-patch16-256-i18n",
)

ALL_MODELS = IMAGENET_MODELS + CHINESE_MODELS + MULTILINGUAL_MODELS


if len(ALL_MODELS) != 22 or len(set(ALL_MODELS)) != 22:
    raise RuntimeError("the official B01 scope must contain 22 unique models")
