# WT-MM/vit-base-blur

模型编号：168；任务：classification；MX 环境：e07。

## 运行

从模型目录执行：

```bash
python inference.py --phase full
python ../../../evaluate.py --run-dir "$(pwd)/result/run_001"
```

从项目根目录执行：

```bash
bash classification/gaoyonghan/environment/run-e07.sh /opt/model-envs/e07/bin/python \
  /project/classification/gaoyonghan/models/WT-MM--vit-base-blur/inference.py --phase full
bash classification/gaoyonghan/environment/run-e07.sh /opt/model-envs/e07/bin/python \
  /project/classification/evaluate.py --run-dir /project/classification/gaoyonghan/models/WT-MM--vit-base-blur/result/run_001
```

冒烟（只跑 10 张，只验证链路，结果不作数）：

```bash
python inference.py --phase smoke
```

`--show-command` 只打印将要执行的命令，不跑模型。`--output-root result/run_002` 可另存一次运行。

## 权重

随本目录交付，位于 `weights/`（7 个文件，327.3 MB：`config.json`、`preprocessor_config.json`、`model.safetensors` 与上游仓库 model card）。推理时直接从该目录离线加载，不依赖外部路径；逐文件 sha256 与上游仓库信息见 `weights/weight_verification.json`。若改用外部权重目录，把 `WEIGHTS_ROOT` 指向包含 `<dirname>/` 的目录即可覆盖。

## provenance

`provenance/` 记录来源与验证：

| 文件 | 内容 |
|---|---|
| `mx_layout.json` | 本目录布局、权重文件与上游仓库的对应关系 |
| `split_audit.json` | 评测集文件哈希、张数、类别分布，并与 `expected_images` 比对 |
| `validation.json` | 本次实测指标、作者公布值，以及该数值能否作为精度引用的判定 |

## 数据集

共享 `dataset/blur_sd21`，900 张。

按作者公开方法自建：SD2.1 30 步跑完(clean) / 30 步调度只跑前 10 步(blurry)，512×512

## 评测集准备

评测集不在共享 `dataset/` 时，`inference.py` 会自动调用 `inference/native/prepare_dataset.py`，参数：`--row 168 --n-prompts 75 --seeds-per-prompt 6`。
该步骤需要原始素材（见脚本头部说明），不满足时会直接报出缺什么。

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
| 评测集 | `dataset/blur_sd21` · 900 张 |
| Top-1 accuracy | 0.92667 |
| Balanced accuracy | 0.92667 |
| Macro-F1（观测到的 GT 类） | 0.92627 |
| Top-5 accuracy | —（`not_applicable_fewer_than_five_classes`，类别数 < 5 不适用） |
| 全流程墙钟 | 27.7 s（加载 15.7 s + 预热 4.8 s + 前向 0.48 s） |
| 前向吞吐 | 0.538 ms/张（1859 张/s） |
| 峰值显存 | 987 MiB（本进程本段） |
| 设备 / 精度 / batch | MetaX C500 · amp_bf16 · 64 |

指标由 `classification/evaluate.py` 计算（sklearn），逐类明细见 `result/run_001/standardized_classification/metrics.json`，运行记录见 `status.json` / `timing.json`。

## 已知限制

- 评测集按作者公开方法重建：SD2.1 同 prompt 同 seed，30 步调度只跑前 10 步解码得到 blurry，跑满 30 步得 clean，共 450 对 900 张。
- 作者 100% 是 173 张 held-out 上的结果，与本次 900 张口径不同，差值仅参照。
- 类别数少于 5，Top-5 不适用（`metric_status.top5_accuracy` = `not_applicable_fewer_than_five_classes`）。
