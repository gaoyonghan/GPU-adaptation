[中文](README.md) | [English](README.en.md)

# marigold-depth-lcm-v1-0

## 资产与环境

模型来源为 https://huggingface.co/prs-eth/marigold-depth-lcm-v1-0

在本模型目录准备权重与配置。必要文件包括 `weights/model_index.json`, `weights/scheduler`, `weights/text_encoder`, `weights/tokenizer`, `weights/unet`, `weights/vae`
权重、数据与运行结果单独准备，完整资产信息见 [assets.json](assets.json)。

| 输入参数 | 模型目录下的默认位置 |
| --- | --- |
| `--weights` | `weights` |
| `--dataset-root` | `../../../../dataset` |

依赖安装与环境选择见 [environment/README.md](../../environment/README.md)。Python 版本为 3.11.16，本模型依赖清单为 requirements.txt。

## 推理

激活对应环境后，从本模型目录运行。

```bash
python inference.py
```

默认结果为 result/run_001，已有目录不会覆盖。可用 --output-root result/run_002 选择新目录。完整参数见 python inference.py --help。

完整数据推理省略 --max-samples，功能检查可加 --max-samples 1。

## 评分

在本模型目录执行。

```bash
python ../../../evaluate.py --run-dir "$PWD/result/run_001"
```

评分读取保存的深度预测、数据协议与真实深度，输出 AbsRel、RMSE、Log10 和 delta 指标。尺度对齐、裁剪和深度范围按 prediction_type 与数据协议处理，详见任务 README。
