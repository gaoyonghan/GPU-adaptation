[中文](README.md) | [English](README.en.md)

# Sumsub-ffs-synthetic-1.0_mj_5

## Assets and environment

Model source information follows. See assets.json for recorded source information and revision status.

Prepare the checkpoint and configuration files in this model directory. Required files include `weights/__init__.py`, `weights/config.json`, `weights/download_model.py`, `weights/midjourney5M.pt`, `weights/midjourney5M.py`, `weights/pipeline.py`
Weights, datasets, and run outputs are prepared separately. See [assets.json](assets.json) for the complete asset description.

| Input option | Default location relative to this model |
| --- | --- |
| `--weights` | `weights` |
| `--hf-cache` | `weights/.runtime_cache` |
| `--dataset-root` | `../../../../dataset/ILSVRC___imagenet-1k` |
| `--dataset-root` | `../../../../dataset/coco/arrow_val` |

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
