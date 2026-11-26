[中文](README.md) | [English](README.en.md)

# tf_efficientnetv2_m.in21k_ft_in1k

## 资产与环境

模型来源为 https://huggingface.co/timm/tf_efficientnetv2_m.in21k_ft_in1k

在本模型目录准备权重与配置。必要文件包括 `weights/model.safetensors`
权重、数据与运行结果单独准备，完整资产信息见 [assets.json](assets.json)。

| 输入参数 | 模型目录下的默认位置 |
| --- | --- |
| `--weights` | `weights` |
| `--dataset-root` | `../../../../dataset/imagenet-1k` |

依赖安装与环境选择见 [environment/README.md](../../environment/README.md)。Python 版本为 3.10.21，本模型依赖清单为 requirements.txt。

## 推理

激活对应环境后，从本模型目录运行。

```bash
python inference.py
```

默认结果为 result/run_001，已有目录不会覆盖。可用 --output-root result/run_002 选择新目录。完整参数见 python inference.py --help。

完整数据推理省略 --max-samples，功能检查可加 --max-samples 1。

## 评分

在本模型目录执行，使用分类任务已有的公共评分器。

```bash
python ../../../evaluate.py --run-dir "$PWD/result/run_001"
```

推理输出包含 inference_output.json 和 scoring_bundle。评分结果见 evaluation_summary.json 与评分包内的 metrics.json。
数据划分和标签映射保留模型实现中的协议，具体限制见 assets.json 与 inference/ 内的实现。
