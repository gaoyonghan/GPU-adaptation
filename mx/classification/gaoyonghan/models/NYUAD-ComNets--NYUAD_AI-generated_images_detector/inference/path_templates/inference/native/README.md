# 72 号 NYUAD AI 生成图像三分类

```bash
cd __MODEL_ROOT__/resources/runtime/assets/workspace/nyuad72_eval
GPU_ID=1 MAX_IMAGES=0 bash run_all.sh
```

自动核对环境、权重与数据 → 10 图验证 → 固定 900 图评测 → 指标与审计。默认不在 Codex 内启动全量。
仅检查小样本：`MAX_IMAGES=10 bash run_all.sh`。同一指令可断点续跑；换协议必须指定新 `OUTPUT_ROOT`，保留旧结果。
环境只读复用已配置的 `__MODEL_ROOT__/resources/runtime/assets/workspace/classification8_eval/.venv`，不升级、不修改其他任务。

模型卡的官方 `image-classification` pipeline 直接加载本地原生 safetensors 和处理器。
保留三类顺序 **dalle / real / sd**，不套用 95 号的二分类流程。采用 FP32、eager attention、禁用 TF32、作者图像处理配置，无额外增强。

这是明确标记的公开替代评测：Synthbuster 的 DALL-E 3、SDXL 各 300 张，加 RAISE-TIFF 的 300 张真实图，共 900 张。每组按路径 SHA256 固定选取，三类平衡。归档实际含 300 张真实图，不是完整 RAISE-1k，也不是总清单旧备注的 303 张。
原模型卡 Accuracy=97.36%、Loss=0.0987 来自另一验证集，不能计算本次与官方的差值。
级别为 `reference_baseline` / `public_replacement` / `not_officially_comparable`；原因和固定哈希见 `models.json`。
尚未定位原始 evaluator；指标使用 scikit-learn 的官方 API，报告三分类 Accuracy（%）、Macro-F1（0..1）、Macro OvR AUROC（0..1）、混淆矩阵（行=真实，列=预测）。

结果目录 `__MODEL_ROOT__/resources/runtime/assets/workspace/results/nyuad72_v1`：

- `dataset_index.json`：完整样本、来源、标签、CRC 和大小；不复制原图。
- `predictions.json`：概率、预测类别和输入图像 SHA256，每 25 图原子保存；中断最多重做 24 图。
- `status.json`、`audit_summary.json`：覆盖、协议指纹、清单、环境、运行状态和资源。
- `metrics.json`：仅完整覆盖固定 900 图后生成；冒烟不报告指标。
- `run.log`、`_source_snapshot`：日志和本次实际运行脚本。

查看清单：`__MODEL_ROOT__/resources/runtime/assets/workspace/classification8_eval/.venv/bin/python run_model.py --describe 72`。

来源：[模型卡](https://huggingface.co/NYUAD-ComNets/NYUAD_AI-generated_images_detector)、[论文](https://ssrn.com/abstract=5029893)、[Synthbuster](https://doi.org/10.5281/zenodo.10066460)。模型卡提供推理示例，但未提供可唯一复用的原验证集划分；论文摘要说明使用 DALL-E 3、SDXL 与网上收集的真实图。本轮只对齐生成器家族，不宣称原数据集一致。
