[中文](README.md) | [English](README.en.md)

# granite-geospatial-uki

## Assets and environment

Model source information follows. https://huggingface.co/ibm-granite/granite-geospatial-uki

Prepare the checkpoint and configuration files in this model directory. Required files include `weights/config.yaml`, `weights/granite_geospatial_uki.pt`, `weights/granite_geospatial_uki_flood_detection_v2.ckpt`
Weights, datasets, and run outputs are prepared separately. See [assets.json](assets.json) for the complete asset description.

| Input option | Default location relative to this model |
| --- | --- |
| `--weights` | `weights` |
| `--bundle` | `../../../../dataset/granite_biomass_bundle` |

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

This entry compares outputs for multispectral remote-sensing inputs. Biomass, canopy height and land-surface temperature use regression outputs. UKI uses segmentation outputs. The common feature evaluator measures cross-device numerical differences. Task accuracy requires the corresponding ground truth and evaluation protocol.
