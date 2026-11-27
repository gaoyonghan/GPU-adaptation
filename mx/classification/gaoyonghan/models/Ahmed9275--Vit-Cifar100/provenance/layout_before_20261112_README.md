# Ahmed9275/Vit-Cifar100

This folder contains the verified single-model ViT-Cifar100 workflow. `weights/` contains the publisher's actual files. `inference/run_model.py` contains only this model's Hugging Face inference branch, with the previously verified weight loading, preprocessing, forward pass, probabilities, prediction records, timing, and smoke-before-full checks. `evaluation/` contains the original format conversion and scoring code; `provenance/runner_diff.patch` records the inference extraction.

The test-set parquet and original sample index are under the project-level `dataset/` and this folder's `inference/indices/`, respectively. Runtime paths are relative to this model folder. There are no symbolic links or dependencies on another model folder or a central shared code directory. `requirements.txt` records the previously used environment; installation on a new machine has not been independently validated.

Run from this model folder:

```bash
python inference/run_model.py --phase smoke --device cuda:0 --batch-size 1 --warmup 10 --workers 2
python inference/run_model.py --phase full --device cuda:0 --batch-size 1 --warmup 10 --workers 2
python evaluation/prepare_classification.py --model 195 --phase full \
  --source results/run_001/full/m195/predictions.jsonl \
  --output results/run_001/full/m195/evaluation
python evaluation/score_classification.py \
  --bundle results/run_001/full/m195/evaluation
```

The inference script is fixed to this model and needs no `--model` argument. The evaluation scripts retain their historical `--model 195` interface. Repeated runs should use the same new `--output-root` for both inference phases.

Local smoke predictions and evaluation were checked against the original script field by field. The full 10,000-image run on the server A100 completed with Top-1 **0.8985** and Top-5 **0.9866**; the folder's independent conversion and scoring code produced the same values. Its archived outputs and hashes are in `results/server_a100_20261110/`, and the environment and verification details are in `results/server_a100_20261110/server_run_report.json`. The server dataset file has a different object-storage timestamp, which is recorded in this folder's `inference/models.json`; its content hash was checked against the local file.

The other reorganized models have not undergone a full inference or evaluation pass in the new layout.
