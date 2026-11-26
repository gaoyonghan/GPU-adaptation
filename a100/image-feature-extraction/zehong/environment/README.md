[中文](README.md) | [English](README.en.md)

# A100 依赖环境

model_environment.csv 将每个模型映射到其安装依赖清单和 Python 版本。manifests/ 内按原负责人保留依赖分组，防止同名清单覆盖。实际 Conda 环境单独创建。

先查询映射表，再用表中的 Python 版本创建环境。在对应模型目录安装依赖，例如执行以下命令。

```bash
conda create -n model-runtime python=<映射表中的版本>
conda activate model-runtime
python -m pip install -r requirements.txt
python inference.py --help
```

依赖清单来自服务器现有环境记录。PyTorch/CUDA 构建须匹配驱动。新机器上仍需完成环境安装和实际推理验证。自定义源码或额外安装要求见各模型的 inference/ 与 assets.json。
