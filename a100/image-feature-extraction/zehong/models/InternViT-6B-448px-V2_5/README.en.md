[中文](README.md) | [English](README.en.md)

# InternViT-6B-448px-V2_5

## Assets and environment

Model source information follows. https://huggingface.co/OpenGVLab/InternViT-6B-448px-V2_5

Prepare the checkpoint and configuration files in this model directory. Required files include `weights/config.json`, `weights/configuration_intern_vit.py`, `weights/flash_attention.py`, `weights/model-00001-of-00003.safetensors`, `weights/model-00002-of-00003.safetensors`, `weights/model-00003-of-00003.safetensors`, `weights/model.safetensors.index.json`, `weights/modeling_intern_vit.py`, `weights/preprocessor_config.json`
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
