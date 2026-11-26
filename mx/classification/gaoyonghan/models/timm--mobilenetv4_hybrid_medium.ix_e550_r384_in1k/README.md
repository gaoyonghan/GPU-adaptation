# timm/mobilenetv4_hybrid_medium.ix_e550_r384_in1k
模型编号：220；任务：classification；MX 环境：e07。

## 运行

从模型目录执行（简版）：

```bash
python inference.py
```

从项目根执行（wrapper 版，等价于上面的命令）：

```bash
sudo bash classification/gaoyonghan/environment/run-e07.sh /opt/model-envs/e07/bin/python \
  /project/classification/gaoyonghan/models/timm--mobilenetv4_hybrid_medium.ix_e550_r384_in1k/inference.py --phase full
```

补充用法：

- `--phase smoke` 只跑 10 张，验链路，结果不作数。
- `--output-root result/run_002` 换运行目录，默认 `result/run_001`。
- `--show-command` 只打印命令不推理。

评测用顶层共用脚本：

```bash
sudo bash classification/gaoyonghan/environment/run-e07.sh /opt/model-envs/e07/bin/python \
  /project/classification/evaluate.py \
  --run-dir /project/classification/gaoyonghan/models/timm--mobilenetv4_hybrid_medium.ix_e550_r384_in1k/result/run_001
```

## 权重

随本目录交付在 `weights/`，共 3 个文件、约 **43 MB**，离线加载：

| 文件 | 大小 | 说明 |
|---|---|---|
| `model.safetensors` | 44,664,368 B | 权重本体，632 个张量（536 个 F32 + 96 个 I64 的 BN 计数） |
| `config.json` | 686 B | 模型结构；文件很小，因为这是 timm 原生格式，超参在 `pretrained_cfg` 里 |
| `README.md` | 13,093 B | 上游 model card |

逐文件 sha256 见 `weights/weight_verification.json`。

未随目录交付：`pytorch_model.bin`（与 safetensors 内容重复的 44 MB，transformers 优先读 safetensors）。

**这份权重里没有 `preprocessor_config.json`，上游仓库本来就没有。** 图像预处理参数需要由 transformers + timm 从 `config.json` 的 `pretrained_cfg` 合成，因此运行环境有版本门槛，见「已知限制」。

## 数据集

共享 `dataset/imagenet-1k`，`validation` 划分，**50000 张、1000 类**（每类 50 张），输入 384px。

官方 ILSVRC2012 CLS-LOC 验证集。脚本按 `{数据集根}/data/validation-*.parquet` 找分片，找不到则回退按目录名加载。

## 输出

`result/run_001/` 下：

| 文件 / 目录 | 是什么 |
|---|---|
| `inference_output.json` | 运行状态机 running → completed，登记评分包相对路径 |
| `native/predictions.jsonl` | 逐样本预测，含原生排序的 top-5 与分数 |
| `native/classification_annotations.json` | 逐样本真值 |
| `native/classification_capture.json` | 原生捕获状态 |
| `native/status.json` | 完成态、张数（应为 50000）、设备、权重路径 |
| `native/standardized_classification/` | 标准化评分包（protocol / annotations / predictions） |

## provenance

| 文件 | 记录什么 |
|---|---|
| `mx_layout.json` | MX 侧目录布局、入口命令、设备与精度 |
| `split_audit.json` | 官方 val 划分、样本编号规则（`validation#<行号>`）、1000 类 |
| `validation.json` | 已通过的校验项、参考结果、本次运行后的预期 |

## 实测结果

| 项 | 值 |
|---|---|
| 评测集 | `dataset/ImageNet-1K` · val · 50000 张 |
| **Top-1 accuracy** | **0.8296** |
| Balanced accuracy | 0.8296 |
| Macro-F1 | 0.82716 |
| Top-5 accuracy | 0.96474 |
| 墙钟与单图耗时 | **152.5 s** / **0.7117 ms**（单图为纯前向口径，warmup 5 批已剔除；吞吐 1405.03 张/s） |
| 峰值显存 | 1647.1 MiB（batch 64 · 384px · fp32，运行期分配峰值） |
| 设备与精度 | MetaX C500 / fp32 |

以上为 **2026-11-25 在 e07 实测**的结果（`result/run_001`，`--phase full`，50000/50000 张）。与交付前统一口径复跑值 0.82962 相比**相差 1 张图**（41480 对 41481），Top-5 完全一致——差异属图像解码/插值边界翻转，与 A100↔C500 之间已记录的 2 张图差异同性质。

精度口径为 **fp32**：参照实现记录的精度字段是 `amp_bf16`，本模型未启用 autocast，目的是与交付前终值逐项对账。如需 bf16 数据，须另跑一轮并重新对账。

## 已知限制

**这套权重没有 `preprocessor_config.json`**（上游仓库本来就没有），处理器由 transformers 从 `config.json` 的 `pretrained_cfg` 合成。这一点曾在 A100 的 transformers 4.44.2 上失败过，因此当时另建了 5.x 隔离环境。**MX 侧实测无需 5.x**：e07 的 transformers 4.50.0 可正常合成，返回 `TimmWrapperImageProcessor`，模型走 `TimmWrapperForImageClassification` 加载。

但仍需注意：加载路径依赖 transformers 的 timm 集成（约 4.48 以后才稳定），更老的环境仍会失败；timm 侧则由其 `MobileNetV3` 类实现 mobilenetv4 这一族，属正常。

此外：

- `config.json` 只含 `architecture` / `num_classes` / `num_features` / `pretrained_cfg`，**不含类别名**，所以评分包里的类名会退化为 `output_class_N`，数字类 ID 才是权威值。
- Top-5 必须原生捕获，无法由 Top-1 反推。
- 跨平台一致性：A100 与 C500 的 Top-1 相差 2 张图（0.8297 对 0.8296），属两岸图像解码/插值边界翻转，不是评测口径差异。
