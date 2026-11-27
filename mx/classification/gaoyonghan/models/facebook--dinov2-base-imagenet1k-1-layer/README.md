# facebook/dinov2-base-imagenet1k-1-layer

模型编号：169；任务：classification；MX 环境：e07。

## 运行

从模型目录执行：

```bash
python inference.py --phase full
python ../../../evaluate.py --run-dir "$(pwd)/result/run_001"
```

从项目根目录执行：

```bash
bash classification/gaoyonghan/environment/run-e07.sh /opt/model-envs/e07/bin/python \
  /project/classification/gaoyonghan/models/facebook--dinov2-base-imagenet1k-1-layer/inference.py --phase full
bash classification/gaoyonghan/environment/run-e07.sh /opt/model-envs/e07/bin/python \
  /project/classification/evaluate.py --run-dir /project/classification/gaoyonghan/models/facebook--dinov2-base-imagenet1k-1-layer/result/run_001
```

冒烟（只跑 10 张，只验证链路，结果不作数）：

```bash
python inference.py --phase smoke
```

`--show-command` 只打印将要执行的命令，不跑模型。`--output-root result/run_002` 可另存一次运行。

## 权重

随本目录交付，位于 `weights/`（5 个文件，336.2 MB：`config.json`、`preprocessor_config.json`、`model.safetensors` 与上游仓库 model card）。推理时直接从该目录离线加载，不依赖外部路径；逐文件 sha256 与上游仓库信息见 `weights/weight_verification.json`。若改用外部权重目录，把 `WEIGHTS_ROOT` 指向包含 `<dirname>/` 的目录即可覆盖。

## provenance

`provenance/` 记录来源与验证：

| 文件 | 内容 |
|---|---|
| `mx_layout.json` | 本目录布局、权重文件与上游仓库的对应关系 |
| `split_audit.json` | 评测集文件哈希、张数、类别分布，并与 `expected_images` 比对 |
| `validation.json` | 本次实测指标、作者公布值，以及该数值能否作为精度引用的判定 |

## 数据集

共享 `dataset/imagenet1k_val`，50000 张。

mrm8488/ImageNet1K-val (50000 张, 非 gated 镜像；与官方 gated split 同批图，50000 张原始字节 md5 全等)

## 输出

`result/run_001/`：

| 文件 | 内容 |
|---|---|
| `predictions.jsonl` | 逐样本 `id / target / prediction / topk / topk_scores` |
| `status.json` | 完成度、口径、设备与环境；`prepare.py` 靠它判断原生运行是否完成 |
| `timing.json` | 加载、预热、前向计时与峰值显存 |
| `standardized_classification/` | `prepare.py` 生成的评分包（protocol / annotations / predictions） |

`classification/evaluate.py` 读的是 `inference_output.json` 里登记的评分包，评分结果写回包内 `metrics.json`，并在运行目录生成 `evaluation_summary.json`。

## 实测结果（run_001）

| 项 | 值 |
|---|---|
| 评测集 | `dataset/imagenet1k_val` · 50000 张 |
| Top-1 accuracy | 0.84298 |
| Balanced accuracy | 0.84298 |
| Macro-F1（观测到的 GT 类） | 0.84150 |
| Top-5 accuracy | 0.96844 |
| 全流程墙钟 | 73.7 s（加载 16.6 s + 预热 4.7 s + 前向 35.54 s） |
| 前向吞吐 | 0.711 ms/张（1407 张/s） |
| 峰值显存 | 1144 MiB（本进程本段） |
| 设备 / 精度 / batch | MetaX C500 · amp_bf16 · 64 |

指标由 `classification/evaluate.py` 计算（sklearn），逐类明细见 `result/run_001/standardized_classification/metrics.json`，运行记录见 `status.json` / `timing.json`。

## 已知限制

- 评测集为 mrm8488/ImageNet1K-val 镜像，50000 张原始字节 md5 与官方 gated validation 全等。
- 参考值 0.845 取 DINOv2 论文 ViT-B/14 linear probe，非作者声明数值。
- `predictions.jsonl` 保存原生 Top-5 与对应分数（`topk` / `topk_scores`），Top-5 accuracy 由 evaluate.py 直接统计，未重建 logits。
