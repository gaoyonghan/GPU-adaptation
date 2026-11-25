[中文](README.md) | [English](README.en.md)

# Image classification

## Structure and entry points

Models live under zehong/models/<model>/ with inference.py, inference/, weights/, result/, requirements.txt and bilingual README files. The shared evaluate.py lives in the classification task directory. Dependency manifests and model mappings are in zehong/environment/. Datasets are shared under 910b/dataset/.

## Usage

Prepare checkpoints and datasets, activate the model environment and initialize CANN. Run these commands from the classification task directory.

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
python zehong/models/bit-50/inference.py --output-root result/run_001
python evaluate.py --run-dir "$PWD/zehong/models/bit-50/result/run_001"
```

Default asset paths are derived from the entry point location. Model-relative input and output arguments resolve against the model directory. The default result is result/run_001 and existing runs are rejected. Relative --run-dir and --bundle arguments to the public evaluator resolve against the classification task directory. Absolute paths are supported.

Omit --max-samples to process the complete declared dataset split. Use --max-samples 1 for a functional check. Batch size, workers and other options are described by the model's --help. Prepare checkpoints, datasets and the actual runtime environment separately.

## Outputs and scoring

Inference produces inference_output.json with a completed state and relative scoring-bundle paths. Bundles contain predictions.json, annotations.json and protocol.json. The public evaluator writes evaluation_summary.json and metrics.json with Top-1, Top-5 when applicable and other classification metrics. Dataset splits, label mappings and limitations follow each model's protocol.

Check completion status, sample counts and metric status. Forward and end-to-end timing come from inference records; offline scoring does not measure inference latency again.
