# playrobin/furniture-styles

模型编号：167；任务：classification；MX 环境：e07。

## 运行

从模型目录执行：

```bash
python inference.py --phase full
python ../../../evaluate.py --run-dir "$(pwd)/result/run_001"
```

从项目根目录执行：

```bash
bash classification/gaoyonghan/environment/run-e07.sh /opt/model-envs/e07/bin/python \
  /project/classification/gaoyonghan/models/playrobin--furniture-styles/inference.py --phase full
bash classification/gaoyonghan/environment/run-e07.sh /opt/model-envs/e07/bin/python \
  /project/classification/evaluate.py --run-dir /project/classification/gaoyonghan/models/playrobin--furniture-styles/result/run_001
```

冒烟（只跑 10 张，只验证链路，结果不作数）：

```bash
python inference.py --phase smoke
```

`--show-command` 只打印将要执行的命令，不跑模型。`--output-root result/run_002` 可另存一次运行。

## 权重

随本目录交付，位于 `weights/`（5 个文件，327.3 MB：`config.json`、`preprocessor_config.json`、`model.safetensors` 与上游仓库 model card）。推理时直接从该目录离线加载，不依赖外部路径；逐文件 sha256 与上游仓库信息见 `weights/weight_verification.json`。若改用外部权重目录，把 `WEIGHTS_ROOT` 指向包含 `<dirname>/` 的目录即可覆盖。

## provenance

`provenance/` 记录来源与验证：

| 文件 | 内容 |
|---|---|
| `mx_layout.json` | 本目录布局、权重文件与上游仓库的对应关系 |
| `split_audit.json` | 评测集文件哈希、张数、类别分布，并与 `expected_images` 比对 |
| `validation.json` | 本次实测指标、作者公布值，以及该数值能否作为精度引用的判定 |

## 数据集

共享 `dataset/interior_3cls`，1047 张。

tonijhanel/my_interior_design_dataset 官方 test split（4182 张）中的 3 个可对应子类

## 评测集准备

评测集不在共享 `dataset/` 时，`inference.py` 会自动调用 `inference/native/prepare_dataset.py`，参数：`--row 167 --apply`。
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
| 评测集 | `dataset/interior_3cls` · 1047 张 |
| Top-1 accuracy | 0.36772 |
| Balanced accuracy | 0.33333 |
| Macro-F1（观测到的 GT 类） | 0.17924 |
| Top-5 accuracy | —（`not_applicable_fewer_than_five_classes`，类别数 < 5 不适用） |
| 全流程墙钟 | 24.6 s（加载 6.0 s + 预热 4.4 s + 前向 0.50 s） |
| 前向吞吐 | 0.481 ms/张（2078 张/s） |
| 峰值显存 | 987 MiB（本进程本段） |
| 设备 / 精度 / batch | MetaX C500 · amp_bf16 · 64 |

指标由 `classification/evaluate.py` 计算（sklearn），逐类明细见 `result/run_001/standardized_classification/metrics.json`，运行记录见 `status.json` / `timing.json`。

## 已知限制

- 模型为 5 类输出，评测集只覆盖可对应的 3 类（模型 class id 1/2/4）；prediction 是在这 3 类内 argmax 的结果，不是原生 5 类 Top-1。
- 模型 5 类输出中实际只激活 2 类：原生 argmax 为 modern(4) 960 张、Scandinavian(0) 87 张，bohemian(1)/coastal(2)/glamor(3) 从未胜出；被判为 Scandinavian 的 87 张在 3 个 GT 类上为 31/27/29，与按先验随机的期望 26.3/28.7/32.0 一致（±1.2 倍内），该分支无判别信息。
- 因评测集只覆盖模型 id 1/2/4，被判为 id 0 的 87 张在 3 类筛选后全部落入 id 4，故 prediction 全为 4；accuracy 36.77% 等于 id 4 的样本占比，balanced accuracy 33.33% 恰等于三类随机水平。该行不能作为模型有效性证据，只能作为「评测集与模型类别空间不匹配」的记录。
- 作者 52.4% 是 21 张验证集准确率（float32(11/21)），无统计意义，不作复现判定。
- 类别数少于 5，Top-5 不适用（`metric_status.top5_accuracy` = `not_applicable_fewer_than_five_classes`）。
