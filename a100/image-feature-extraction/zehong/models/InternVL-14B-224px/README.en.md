[中文](README.md) | [English](README.en.md)

# InternVL-14B-224px

## Assets and environment

Model source information follows. https://huggingface.co/OpenGVLab/InternVL-14B-224px

Prepare the checkpoint and configuration files in this model directory. Required files include `weights/__init__.py`, `weights/config.json`, `weights/configuration_intern_vit.py`, `weights/configuration_internvl.py`, `weights/flash_attention.py`, `weights/model-00001-of-00003.safetensors`, `weights/model-00002-of-00003.safetensors`, `weights/model-00003-of-00003.safetensors`, `weights/model.safetensors.index.json`, `weights/modeling_intern_vit.py`, `weights/modeling_internvl.py`, `weights/modeling_qllama.py`, `weights/preprocessor_config.json`, `weights/pytorch_model-00003-of-00003.bin`, `weights/pytorch_model.bin.index.json`, `weights/special_tokens_map.json` See assets.json for the remaining files.
Weights, datasets, and run outputs are prepared separately. See [assets.json](assets.json) for the complete asset description.

| Input option | Default location relative to this model |
| --- | --- |
| `--weights` | `weights` |
| `--bundle` | `../../../../dataset/coco_val_bundle_1000` |

Select the model environment using [environment/README.en.md](../../environment/README.en.md). The mapping records its Python version. Install dependencies using this model's requirements.txt.

## Inference

Activate the corresponding environment and run from this model directory.

```bash
python inference.py
```

The default output is result/run_001. Existing run directories are rejected. Use --output-root result/run_002 for another run. See python inference.py --help for model-specific options.

Omit --max-samples to process the declared complete split. Use --max-samples 1 for a functional check.

## Cross-device output comparison

Run the task-level evaluator from this model directory.

```bash
python ../../../evaluate.py /path/to/a100/run /path/to/candidate/run --output-root "$PWD/result/comparison_001"
```

The first path is the A100 baseline run and the second is the candidate-device run. Both must be accessible on the evaluation machine. Use matching checkpoints, original samples and feature definitions. See the [task README](../../../README.en.md) for directory contents and path organization. The report includes cosine similarity and numerical errors.
