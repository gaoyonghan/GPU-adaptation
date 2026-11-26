[中文](README.md) | [English](README.en.md)

# dinov2-base

## 资产与环境

模型来源为 https://huggingface.co/facebook/dinov2-base

在本模型目录准备权重与配置。必要文件包括 `weights/config.json`, `weights/model.safetensors`, `weights/preprocessor_config.json`, `weights/pytorch_model.bin`
权重、数据与运行结果单独准备，完整资产信息见 [assets.json](assets.json)。

| 输入参数 | 模型目录下的默认位置 |
| --- | --- |
| `--weights` | `weights` |
| `--bundle` | `../../../../dataset/aimv2_features/imagenet_validation_1000` |

依赖安装与环境选择见 [environment/README.md](../../environment/README.md)。Python 版本为 3.10.22，本模型依赖清单为 requirements.txt。

## 推理

激活对应环境后，从本模型目录运行。

```bash
python inference.py
```

默认结果为 result/run_001，已有目录不会覆盖。可用 --output-root result/run_002 选择新目录。完整参数见 python inference.py --help。

完整数据推理省略 --max-samples，功能检查可加 --max-samples 1。

## 跨设备输出比较

在本模型目录执行。两份结果须来自相同权重、输入清单和特征定义。 两个路径分别为 A100 基准 run 和候选设备 run，均须在评分机器上可读。目录内容和组织示例见[任务说明](../../../README.md#两个输入结果目录)。

```bash
python ../../../evaluate.py /path/to/a100/run /path/to/candidate/run --output-root "$PWD/result/comparison_001"
```

公共评分器比较同一样本的特征，指标含余弦相似度和数值误差。准备的数据包与模型特征语义见 assets.json 和 inference/ 中的实现。
