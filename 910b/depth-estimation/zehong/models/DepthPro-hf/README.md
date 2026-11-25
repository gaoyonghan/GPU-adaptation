[中文](README.md) | [English](README.en.md)

# DepthPro-hf

## 资产与环境

模型来源为 来源与必要文件见 assets.json，未固定的版本需在准备资产时确认。

在本模型目录准备权重与配置。必要文件包括 `weights/config.json`, `weights/model.safetensors`, `weights/preprocessor_config.json`
权重、数据与运行结果单独准备，完整资产信息见 [assets.json](assets.json)。

| 输入参数 | 模型目录下的默认位置 |
| --- | --- |
| `--weights` | `weights` |
| `--dataset-root` | `../../../../dataset` |

依赖安装与环境选择见 [environment/README.md](../../environment/README.md)。Python 版本为 3.11.16，本模型依赖清单为 requirements.txt。

## 推理

激活对应环境后，从本模型目录运行。

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
python inference.py --dataset NYUv2 --dataset-root ../../../../dataset --weights weights --output-root result/run_001
```

默认结果为 result/run_001，已有目录不会覆盖。可用 --output-root result/run_002 选择新目录。完整参数见 python inference.py --help。

完整数据推理省略 --max-samples，功能检查可加 --max-samples 1。

运行前按环境映射表选择 m03-depth 对应的 Python 3.11.16 环境。每次运行在结果目录写入 run_metadata.json，记录完整 Python 命令、参数、工作目录、Python 可执行文件与环境路径、Conda 环境名（若已激活）、PyTorch 与 torch_npu 版本、NPU 设备和解析后的资产路径。功能检查时，在推理命令末尾加入 --max-samples 1，并使用一个新的输出目录。

## 评分

在本模型目录执行。

```bash
python ../../../evaluate.py --run-dir "$PWD/result/run_001"
```

评分读取保存的深度预测、数据协议与真实深度，输出 AbsRel、RMSE、Log10 和 delta 指标。尺度对齐、裁剪和深度范围按 prediction_type 与数据协议处理，详见任务 README。

NYUv2 使用 654 图测试列表，预测为米制深度，焦距由模型估计。前向计时采用设备事件包围 model.forward，默认预热 3 次，每批 1 图。
