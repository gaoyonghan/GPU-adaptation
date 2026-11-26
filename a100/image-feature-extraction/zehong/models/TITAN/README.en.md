[中文](README.md) | [English](README.en.md)

# TITAN

## Assets and environment

Model source information follows. https://huggingface.co/MahmoodLab/TITAN

Prepare the checkpoint and configuration files in this model directory. Required files include `weights/conch_tokenizer.py`, `weights/conch_v1_5.py`, `weights/conch_v1_5_pytorch_model.bin`, `weights/config.json`, `weights/configuration_titan.py`, `weights/model.safetensors`, `weights/modeling_titan.py`, `weights/special_tokens_map.json`, `weights/text_transformer.py`, `weights/tokenizer.json`, `weights/tokenizer_config.json`, `weights/vision_transformer.py`
Weights, datasets, and run outputs are prepared separately. See [assets.json](assets.json) for the complete asset description.

| Input option | Default location relative to this model |
| --- | --- |
| `--weights` | `weights` |
| `--bundle` | `../../../../dataset/aimv2_features/imagenet_validation_1000` |

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
