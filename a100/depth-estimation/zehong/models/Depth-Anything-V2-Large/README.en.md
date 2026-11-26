[中文](README.md) | [English](README.en.md)

# Depth-Anything-V2-Large

## Assets and environment

Model source information follows. See assets.json for recorded source information and revision status.

Prepare the checkpoint and configuration files in this model directory. Required files include `weights/depth_anything_v2_vitl.pth`
Weights, datasets, and run outputs are prepared separately. See [assets.json](assets.json) for the complete asset description.

| Input option | Default location relative to this model |
| --- | --- |
| `--weights` | `weights` |
| `--dataset-root` | `../../../../dataset` |

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

The evaluator reads saved depth predictions and ground truth and reports AbsRel, RMSE, Log10 and delta metrics. Scale alignment, cropping and depth limits follow the declared prediction type and dataset protocol. See the task README for details.
