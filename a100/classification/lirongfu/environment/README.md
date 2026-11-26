# 分类模型运行环境

全部 58 个模型共用 classification-eval；对应关系见 model_environment.csv。
classification-eval.txt 是 2026-11-19 从目标服务器现有环境导出的实际 pip 版本清单，pip check 已通过。没有复制环境目录。

当前服务器激活已有环境：

```bash
source /opt/venvs/classification-eval/bin/activate
export PYTHONNOUSERSITE=1
unset PYTHONPATH
```

重建环境时使用 Python 3.10，并选择允许执行文件的本地磁盘目录（本服务器的 /data 挂载曾出现执行权限问题）：

```bash
python3 -m venv --copies /opt/venvs/classification-eval
source /opt/venvs/classification-eval/bin/activate
python -m pip install -U pip setuptools wheel
python -m pip install torch==2.8.0+cu126 torchvision==0.23.0+cu126 --index-url https://download.pytorch.org/whl/cu126
python -m pip install --no-build-isolation -r classification/lirongfu_recovered/environment/classification-eval.txt
python -m pip check
```

以上安装命令从项目根目录执行。系统需提供 OpenCV 所需 libGL/libglib；若 mamba_ssm 无可用预编译包，还需匹配的 CUDA 编译工具链。驱动须能支持所安装 CUDA 运行库及模型使用的编译内核。

LSNet 使用 Triton 的 ptxas，复制环境时必须保留它的执行权限。此服务器已验证 ptxas 可执行。
不要把别的环境目录直接混入当前环境，也不要把虚拟环境文件放进模型 weights/。

从项目根目录运行单个模型：

```bash
python classification/lirongfu_recovered/models/Sumsub__Sumsub-ffs-synthetic-1.0_sd_5/inference.py
```

不需要 main.py。模型入口会独立推理、计时、转换输出、调用公共评测，并把日志和结果放在自己的 result/ 中。

## 平台和路径

本目录内容属于 A100 平台。按当前部署约定，`image-classification` 对应规范中的 `GPU-adaptation/a100` 平台根目录。
整体移入 `GPU-adaptation/a100/` 时不需要改代码；从仓库根目录先 `cd a100` 再执行模型命令。
CSV 中路径均相对平台根目录，列出模型目录、环境文件、Python 版本、GPU/运行时要求和已知限制。
数据、权重和结果的环境变量配置见各模型 README。
