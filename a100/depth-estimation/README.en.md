[中文](README.md) | [English](README.en.md)

# Depth estimation

## Directory layout

Each model starts through zehong/models/<model>/inference.py. The task-level evaluate.py evaluates saved outputs. Environment mappings are in zehong/environment/. Shared datasets belong in a100/dataset and checkpoints belong in each model's weights/ directory.

```text
<task>/
  evaluate.py
  README.md
  README.en.md
  zehong/
    environment/
    models/<model>/
      inference.py
      inference/
      weights/
      result/
      requirements.txt
      README.md
      README.en.md
```

## Inference and evaluation

Activate the model environment and prepare assets before running these commands from the task directory.

```bash
python zehong/models/DepthPro/inference.py --output-root result/run_001
python evaluate.py --run-dir "$PWD/zehong/models/DepthPro/result/run_001"
```

The entry point saves per-image depth predictions and an index. evaluate.py independently computes AbsRel, RMSE, Log10 and delta1/2/3 using the saved predictions and ground truth, recording successful, skipped and failed samples. Dataset protocols include NYUv2, KITTI, DIODE and Sintel. Each model's supported inputs are defined by its --help and dataset loader.

evaluation_config.json records dataset paths and prediction type. Metric depth, relative depth and relative inverse depth follow their respective protocols. Cropping, valid depth ranges and scale-and-shift fitting are implemented by the evaluator. Runs contain inference_output.json, inference_summary.json, evaluation_config.json, dataset-specific inference_records.csv and prediction files. Evaluation produces summary.json and per_image_metrics.csv.

forward_ms records per-image network-forward latency. Diffusion models may sum multiple network components. stage1_dataset_seconds covers formal dataset iteration and prediction saving. Warmup and loading occur outside formal timing. Cross-device comparisons require matching datasets, precision, resolution, diffusion steps, ensemble counts and timing boundaries.

## Paths and completion

Default asset paths are derived from the script location. Model-relative input and output arguments resolve against the model directory. The default output is result/run_001. Use a fresh directory for each run. Evaluation examples use absolute result paths.

Omit --max-samples for the complete declared split and use it for functional checks. Confirm sample counts and error records as well as completion status. Reorganization does not establish a new inference pass on another environment or device.
