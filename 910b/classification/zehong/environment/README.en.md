[中文](README.md) | [English](README.en.md)

# Ascend 910B dependency environments

model_environment.csv maps each model to its installation manifest and Python version. manifests/ preserves dependency groups under their original owner names to avoid filename collisions. Create actual Conda environments separately.

Look up the model's Python version before creating the environment. From the model directory, install its requirements.

```bash
conda create -n model-runtime python=<version-from-mapping>
conda activate model-runtime
python -m pip install -r requirements.txt
python inference.py --help
```

The manifests were captured from existing server environments. Match PyTorch, torch_npu and CANN versions to the driver. Source /usr/local/Ascend/ascend-toolkit/set_env.sh before inference. A new machine still requires installation and inference validation. Model-specific source and additional dependencies are described in inference/ and assets.json.
