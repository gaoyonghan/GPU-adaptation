# jacoballessio/ai-image-detect-distilled
模型编号：221；任务：classification；MX 环境：e07。

## 运行

从模型目录执行（简版）：

```bash
python inference.py
```

从项目根执行（wrapper 版，等价于上面的命令）：

```bash
sudo bash classification/gaoyonghan/environment/run-e07.sh /opt/model-envs/e07/bin/python \
  /project/classification/gaoyonghan/models/jacoballessio--ai-image-detect-distilled/inference.py --phase full
```

补充用法：

- `--phase smoke` 只跑 10 张，验链路，结果不作数。
- `--output-root result/run_002` 换运行目录，默认 `result/run_001`。
- `--show-command` 只打印命令不推理。

评测用顶层共用脚本：

```bash
sudo bash classification/gaoyonghan/environment/run-e07.sh /opt/model-envs/e07/bin/python \
  /project/classification/evaluate.py \
  --run-dir /project/classification/gaoyonghan/models/jacoballessio--ai-image-detect-distilled/result/run_001
```

## 权重

随本目录交付在 `weights/`，共 4 个文件、约 **56 MB**，离线加载：

| 文件 | 大小 | 说明 |
|---|---|---|
| `model.safetensors` | 58,289,792 B | 权重本体，136 个张量，全 F32 |
| `config.json` | 612 B | 模型结构；`id2label` 为 `{0: fake, 1: real}` |
| `preprocessor_config.json` | 326 B | 预处理参数 |
| `README.md` | 2,857 B | 上游 model card |

逐文件 sha256 见 `weights/weight_verification.json`。

## 数据集

共享 `dataset/defactify/data/test-*.parquet`，**5625 张**：真图 937 张（MS COCO）+ AI 生成图 4688 张（SD2.1 / SDXL / SD3 / DALL·E 3 / Midjourney 各约 940）。

数据集字段 `Label_A`：`0 = real`、`1 = AI 生成`。注意方向与模型相反——模型 `config.id2label` 是 `{0: fake, 1: real}`。脚本在启动时从模型 config 自动建立类名→索引映射再翻转，不硬编码方向；缺失 real/fake 会直接报错退出。

## 输出

`result/run_001/` 下：

| 文件 / 目录 | 是什么 |
|---|---|
| `inference_output.json` | 运行状态机 running → completed，登记评分包相对路径 |
| `native/predictions.jsonl` | 逐样本预测，含原生排序 top-k 与分数 |
| `native/classification_annotations.json` | 逐样本真值（已映射到模型类别空间） |
| `native/classification_capture.json` | 原生捕获状态 |
| `native/status.json` | 完成态、张数（应为 5625）、设备、权重路径、real/fake 映射 |
| `native/standardized_classification/` | 标准化评分包（protocol / annotations / predictions） |

## provenance

| 文件 | 记录什么 |
|---|---|
| `mx_layout.json` | MX 侧目录布局、入口命令、设备与精度 |
| `split_audit.json` | 数据分片、样本编号规则（`<parquet 名>#<行号>`）、2 类、标签方向说明 |
| `validation.json` | 已通过的校验项、参考结果、本次运行后的预期 |

## 实测结果

| 项 | 值 |
|---|---|
| 评测集 | `dataset/defactify` · 5625 张（真 937 / AI 4688） |
| **Top-1 accuracy** | **0.16658** |
| Balanced accuracy | 0.49957 |
| Macro-F1 | 0.14290 |
| Top-5 accuracy | 不适用（类别数 2，少于 5） |
| 墙钟与单图耗时 | **28.2 s** / **0.2877 ms**（单图为纯前向口径，warmup 5 批已剔除；吞吐 3476.35 张/s） |
| 峰值显存 | 533.9 MiB（batch 64 · 224px · fp32，运行期分配峰值） |
| 设备与精度 | MetaX C500 / fp32 |

以上为 **2026-11-25 在 e07 实测**的结果（`result/run_001`，`--phase full`，5625/5625 张），与交付前统一口径复跑值**逐位相同**（Accuracy 0.16657777777777777、Balanced 0.49957303736054）。

精度口径为 **fp32**：参照实现记录的精度字段是 `amp_bf16`，本模型未启用 autocast，目的是与交付前终值逐项对账。如需 bf16 数据，须另跑一轮并重新对账。

## 已知限制

**这个模型本身已退化，16.66% 不是评测失误。** 它在 5625 张上几乎恒定输出 `real`：真实图召回 99.89%，AI 图召回仅 **0.02%**，Balanced accuracy 49.96%（≈ 抛硬币），ROC AUC 0.6176。数据集里 83.3% 是 AI 图，所以「全判 real」这个平凡策略正好得到 16.66%——两个数字完全相等，说明模型没有提供任何判别信息。

这个结论有八条独立证据支撑，不是猜的：标签映射从 config 自动对齐、7 种预处理组合全试过、权重加载 missing/unexpected keys 均为 0、跨版本权重逐位一致（Δ=0）、用模型原生 transformers 4.41.2 复现结果相同、CLS 与全局平均池化都救不活、在模型训练同分布的 GenImage SD1.4 上 fake 召回也只有 47%、以及决定性对照——同一数据同一管线换成 `dima806/ai_vs_real_image_detection`，五类假图召回 92–97%。

其余限制：

- 只有 2 类，Top-5 不适用，ROC AUC 不在本评分器范围内。
- 数据集无公开官方评测集，属于自定义口径，因此 `qualification` 记为 `existing_protocol_not_officially_verified`。
- 精度不具备可用性，**但速度指标仍然有效**（单图 0.311 ms、吞吐 3211.20/s）——如果是要用它的速度特性，请分开看这两件事。
