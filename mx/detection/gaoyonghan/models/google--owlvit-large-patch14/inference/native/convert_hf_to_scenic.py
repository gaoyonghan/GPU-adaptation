#!/usr/bin/env python3
"""Reconstruct a Scenic OWL-ViT checkpoint from official HF weights.

The mapping is the exact inverse of Hugging Face's official
convert_owlvit_original_flax_to_hf.py. Every converted tensor is mapped back
to the HF layout and compared bit-for-bit before the checkpoint is accepted.
"""

import argparse
import hashlib
import json
import os
import pathlib
import re

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/owlvit-matplotlib")
os.environ.setdefault("FLAX_USE_ORBAX_CHECKPOINTING", "0")

import jax
import jax.numpy as jnp
import numpy as np
import torch
from flax import traverse_util
from flax.training import checkpoints
from scenic.projects.owl_vit import models
from scenic.projects.owl_vit.configs import clip_b32, clip_l14


IGNORED_HF_KEYS = {
    "owlvit.text_model.embeddings.position_ids",
    "owlvit.vision_model.embeddings.position_ids",
    # Scenic detection does not instantiate this unused CLIP projection.
    "owlvit.visual_projection.weight",
}


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        for chunk in iter(lambda: file.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def mapping_for(scenic_key):
    """Return (HF key, transform kind) for one flattened Scenic parameter."""
    if scenic_key == "backbone/clip/logit_scale":
        return "owlvit.logit_scale", "direct"

    match = re.fullmatch(r"backbone/merged_class_token/(scale|bias)", scenic_key)
    if match:
        suffix = "weight" if match.group(1) == "scale" else "bias"
        return f"layer_norm.{suffix}", "direct"

    if scenic_key.startswith("class_head/"):
        name = scenic_key.removeprefix("class_head/").replace("/", ".")
        name = name.replace(".kernel", ".weight").replace("Dense_0", "dense0")
        kind = "dense_kernel" if scenic_key.endswith("/kernel") else "direct"
        return f"class_head.{name}", kind

    if scenic_key.startswith("obj_box_head/"):
        name = scenic_key.removeprefix("obj_box_head/").replace("/", ".")
        name = name.replace(".kernel", ".weight").replace("_", "").lower()
        kind = "dense_kernel" if scenic_key.endswith("/kernel") else "direct"
        return f"box_head.{name}", kind

    simple = {
        "backbone/clip/text/token_embedding/embedding": (
            "owlvit.text_model.embeddings.token_embedding.weight", "direct"),
        "backbone/clip/text/positional_embedding": (
            "owlvit.text_model.embeddings.position_embedding.weight", "direct"),
        "backbone/clip/text/ln_final/scale": (
            "owlvit.text_model.final_layer_norm.weight", "direct"),
        "backbone/clip/text/ln_final/bias": (
            "owlvit.text_model.final_layer_norm.bias", "direct"),
        "backbone/clip/text/text_projection/kernel": (
            "owlvit.text_projection.weight", "projection_kernel"),
        "backbone/clip/visual/class_embedding": (
            "owlvit.vision_model.embeddings.class_embedding", "direct"),
        "backbone/clip/visual/conv1/kernel": (
            "owlvit.vision_model.embeddings.patch_embedding.weight", "conv_kernel"),
        "backbone/clip/visual/positional_embedding": (
            "owlvit.vision_model.embeddings.position_embedding.weight", "direct"),
        "backbone/clip/visual/ln_pre/scale": (
            "owlvit.vision_model.pre_layernorm.weight", "direct"),
        "backbone/clip/visual/ln_pre/bias": (
            "owlvit.vision_model.pre_layernorm.bias", "direct"),
        "backbone/clip/visual/ln_post/scale": (
            "owlvit.vision_model.post_layernorm.weight", "direct"),
        "backbone/clip/visual/ln_post/bias": (
            "owlvit.vision_model.post_layernorm.bias", "direct"),
    }
    if scenic_key in simple:
        return simple[scenic_key]

    match = re.fullmatch(
        r"backbone/clip/(text|visual)/transformer/resblocks\.(\d+)/"
        r"(attn/(query|key|value|out)/(kernel|bias)|"
        r"ln_([12])/(scale|bias)|mlp/(c_fc|c_proj)/(kernel|bias))",
        scenic_key,
    )
    if not match:
        raise KeyError(f"No official inverse mapping for Scenic parameter: {scenic_key}")

    tower, layer, tail = match.group(1), match.group(2), match.group(3)
    prefix = (
        "owlvit.text_model.encoder.layers"
        if tower == "text"
        else "owlvit.vision_model.encoder.layers"
    )

    attn = re.fullmatch(r"attn/(query|key|value|out)/(kernel|bias)", tail)
    if attn:
        name, param = attn.groups()
        projection = {"query": "q_proj", "key": "k_proj",
                      "value": "v_proj", "out": "out_proj"}[name]
        suffix = "weight" if param == "kernel" else "bias"
        if param == "bias":
            kind = "attn_bias"
        elif name == "out":
            kind = "attn_out_kernel"
        else:
            kind = "attn_in_kernel"
        return f"{prefix}.{layer}.self_attn.{projection}.{suffix}", kind

    layer_norm = re.fullmatch(r"ln_([12])/(scale|bias)", tail)
    if layer_norm:
        number, param = layer_norm.groups()
        suffix = "weight" if param == "scale" else "bias"
        return f"{prefix}.{layer}.layer_norm{number}.{suffix}", "direct"

    mlp = re.fullmatch(r"mlp/(c_fc|c_proj)/(kernel|bias)", tail)
    name, param = mlp.groups()
    dense = "fc1" if name == "c_fc" else "fc2"
    suffix = "weight" if param == "kernel" else "bias"
    kind = "dense_kernel" if param == "kernel" else "direct"
    return f"{prefix}.{layer}.mlp.{dense}.{suffix}", kind


def from_hf(tensor, shape, kind):
    value = tensor.detach().cpu().numpy()
    if kind in ("dense_kernel", "projection_kernel"):
        value = value.T
    elif kind == "conv_kernel":
        value = value.transpose(2, 3, 1, 0)
    elif kind in ("attn_in_kernel", "attn_out_kernel", "attn_bias"):
        value = value.T.reshape(shape) if kind != "attn_bias" else value.reshape(shape)
    elif kind != "direct":
        raise ValueError(f"Unknown transform: {kind}")
    value = np.array(value, copy=True, order="C")
    if tuple(value.shape) != tuple(shape):
        raise ValueError(f"Converted shape {value.shape} does not match Scenic {shape}")
    return value


def to_hf(value, kind):
    if kind in ("dense_kernel", "projection_kernel"):
        return value.T
    if kind == "conv_kernel":
        return value.transpose(3, 2, 0, 1)
    if kind == "attn_in_kernel":
        return value.reshape(value.shape[0], -1).T
    if kind == "attn_out_kernel":
        return value.reshape(-1, value.shape[2]).T
    if kind == "attn_bias":
        return value.reshape(-1)
    if kind == "direct":
        return value
    raise ValueError(f"Unknown transform: {kind}")


def parameter_shapes(config_name):
    config_module = {"clip_b32": clip_b32, "clip_l14": clip_l14}[config_name]
    config = config_module.get_config("canonical_checkpoint")
    module = models.TextZeroShotDetectionModule(
        body_configs=config.model.body,
        normalize=config.model.normalize,
        box_bias=config.model.box_bias,
    )
    variables = jax.eval_shape(
        lambda: module.init(
            jax.random.PRNGKey(0),
            jnp.zeros(
                (1, config.dataset_configs.input_size,
                 config.dataset_configs.input_size, 3),
                jnp.float32,
            ),
            jnp.ones((1, 1, 16), jnp.int32),
            False,
        )
    )
    return traverse_util.flatten_dict(variables["params"], sep="/")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hf-weights", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--scenic-revision", required=True)
    parser.add_argument("--config", choices=("clip_b32", "clip_l14"),
                        default="clip_l14")
    parser.add_argument("--repo-id", default="google/owlvit-large-patch14")
    args = parser.parse_args()

    source_path = pathlib.Path(args.hf_weights).resolve()
    output_dir = pathlib.Path(args.output_dir).resolve()
    report_path = pathlib.Path(args.report).resolve()
    if not source_path.is_file():
        raise SystemExit(f"HF weight file does not exist: {source_path}")

    print(f"Loading official HF weights: {source_path}", flush=True)
    hf_state = torch.load(source_path, map_location="cpu")
    shapes = parameter_shapes(args.config)
    print(f"Scenic parameter tensors: {len(shapes)}", flush=True)

    scenic_flat = {}
    used_hf_keys = set()
    max_abs_error = 0.0
    for scenic_key, spec in shapes.items():
        hf_key, kind = mapping_for(scenic_key)
        if hf_key not in hf_state:
            raise KeyError(f"HF tensor missing for {scenic_key}: {hf_key}")
        value = from_hf(hf_state[hf_key], spec.shape, kind)
        reconstructed = to_hf(value, kind)
        source = hf_state[hf_key].detach().cpu().numpy()
        if not np.array_equal(reconstructed, source):
            error = float(np.max(np.abs(reconstructed - source)))
            max_abs_error = max(max_abs_error, error)
            raise ValueError(
                f"Round-trip mismatch: {scenic_key} -> {hf_key}; max error={error}")
        scenic_flat[tuple(scenic_key.split("/"))] = value
        used_hf_keys.add(hf_key)

    unused = sorted(set(hf_state) - used_hf_keys)
    if set(unused) != IGNORED_HF_KEYS:
        raise ValueError(f"Unexpected unused HF tensors: {unused}")

    params = traverse_util.unflatten_dict(scenic_flat)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoints.save_checkpoint(
        str(output_dir), {"optimizer": {"target": params}},
        step=0, overwrite=True)

    report = {
        "status": "verified",
        "source": args.repo_id,
        "source_weight_file": str(source_path),
        "source_weight_sha256": sha256(source_path),
        "source_format": "Hugging Face PyTorch",
        "target_format": "Scenic 2022 pre-Optax Flax checkpoint (optimizer.target)",
        "mapping_basis": (
            "exact inverse of Hugging Face official "
            "convert_owlvit_original_flax_to_hf.py"
        ),
        "scenic_revision": args.scenic_revision,
        "scenic_config": args.config,
        "converted_tensor_count": len(scenic_flat),
        "roundtrip_exact": True,
        "roundtrip_max_abs_error": max_abs_error,
        "ignored_hf_tensors": unused,
        "checkpoint_path": checkpoint_path,
        "notes": [
            "HF position_ids are derived buffers, not learned Scenic parameters.",
            "HF visual_projection is unused by the Scenic detection graph.",
        ] + ([
            "Native Scenic ViT-L/14 uses 12 text attention heads; the current HF config says 16, but head count is not encoded in projection tensor shapes.",
        ] if args.config == "clip_l14" else []),
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
