[中文](README.md) | [English](README.en.md)

# Image feature extraction

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

## Inference and comparison

Activate the model environment and prepare assets before running these commands from the task directory.

```bash
python zehong/models/GenView-vit-small/inference.py --output-root result/run_001
python evaluate.py /path/to/a100/run /path/to/candidate/run --output-root /path/to/comparison_001
```

evaluate.py compares saved features for the same samples and reports cosine similarity, absolute errors and relative L2. Use matching checkpoints, original samples and feature definitions. The metrics describe cross-device output consistency.

### The two input directories

The first path is the A100 baseline run. The second is the candidate-device run, for example a 910B run. Each path points to a single extraction run, such as a model's result/run_001. Preserve the complete directory structure.

```text
run_001/
  evaluation_config.json     # Checkpoint identity and input/output definitions
  summary.json               # Extraction status, counts and performance
  input_manifest.jsonl       # Sample IDs and original-image hashes
  feature_index.jsonl        # Feature file paths and hashes
  features/                  # Indexed feature tensors
  model_environment.json     # Optional runtime information
```

Pass the run root containing these files. features/ is a subdirectory of that root. Both runs must use the same supported result schema.

Both directories must be readable on the machine running the evaluator. Copy or mount the other device's complete run before comparison. For example, after collecting both runs on a 910B machine, run the following command from the feature-extraction task directory.

```bash
python evaluate.py /workspace/feature_compare/a100_run /workspace/feature_compare/910b_run --output-root /workspace/feature_compare/comparison_001
```

Order determines comparison direction. Relative L2 uses the norm of the first, baseline feature as its denominator. --output-root selects a new comparison directory containing comparison_summary.json and image_similarity.csv. The output directory must not already exist. Scoring reads saved features and needs no inference connection to the other device.

--strict-inputs optionally requires matching recorded preprocessing configurations. Sample sets and original-image contents are always checked. Processor-file and input-tensor hash differences are reported without blocking comparison. Output differences may include numerical differences introduced by preprocessing. See python evaluate.py --help for thresholds and optional arguments. Model READMEs and assets.json describe their inputs, feature semantics and output scope.

## Paths and completion

Default asset paths are derived from the script location. Model-relative input and output arguments resolve against the model directory. The default output is result/run_001. Use a fresh directory for each run. Evaluation examples use absolute result paths.

Omit --max-samples for the complete declared split and use it for functional checks. Confirm sample counts and error records as well as completion status. Reorganization does not establish a new inference pass on another environment or device.
