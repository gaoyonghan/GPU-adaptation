[中文](README.md) | [English](README.en.md)

# A100 dependency environments

model_environment.csv maps each model to its installation manifest and Python version. manifests/ preserves dependency groups under their original owner names to avoid filename collisions. Create actual Conda environments separately.

Look up the model's Python version before creating the environment. From the model directory, install its requirements.

```bash
conda create -n model-runtime python=<version-from-mapping>
conda activate model-runtime
python -m pip install -r requirements.txt
python inference.py --help
```

The manifests were captured from existing server environments. Choose PyTorch/CUDA builds compatible with the driver. A new machine still requires installation and inference validation. Model-specific source and additional dependencies are described in inference/ and assets.json.
