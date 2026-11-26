[中文](README.md) | [English](README.en.md)

# vit_tiny_patch16_224.augreg_in21k

## Assets and environment

Model source information follows. https://huggingface.co/timm/vit_tiny_patch16_224.augreg_in21k

Prepare the checkpoint and configuration files in this model directory. Required files include `weights/config.json`, `weights/model.safetensors`, `weights/source.json`
Weights, datasets, and run outputs are prepared separately. See [assets.json](assets.json) for the complete asset description.

| Input option | Default location relative to this model |
| --- | --- |
| `--weights` | `weights` |
| `--dataset-root` | `../../../../dataset/ILSVRC___imagenet-1k` |

Select the model environment using [environment/README.en.md](../../environment/README.en.md). The mapping records its Python version. Install dependencies using this model's requirements.txt.

## Inference

Activate the corresponding environment and run from this model directory.

```bash
python inference.py
```

The default output is result/run_001. Existing run directories are rejected. Use --output-root result/run_002 for another run. See python inference.py --help for model-specific options.

Omit --max-samples to process the declared complete split. Use --max-samples 1 for a functional check.

## Evaluation

Run the task-level evaluator from this model directory.

```bash
python ../../../evaluate.py --run-dir "$PWD/result/run_001"
```

Inference produces inference_output.json and scoring_bundle. The evaluator writes evaluation_summary.json and metrics.json inside the bundle. Dataset splits and label mappings follow the model implementation. See assets.json and inference/ for the model-specific protocol.

This model retains all 21,843 output classes. The current alternative protocol maps ImageNet-1K validation labels to the 21K WordNet synsets and ranks the complete logits. Samples of the missing synset n04399382 remain in evaluation and count as errors. This protocol does not establish native ImageNet-21K test accuracy or independence from pretraining data.
