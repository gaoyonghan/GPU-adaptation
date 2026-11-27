# dima806/medicinal_plants_image_detection

模型编号：170；任务：classification；MX 环境：e07。

## 运行

从模型目录执行：

```bash
python inference.py --phase full
python ../../../evaluate.py --run-dir "$(pwd)/result/run_001"
```

从项目根目录执行：

```bash
bash classification/gaoyonghan/environment/run-e07.sh /opt/model-envs/e07/bin/python \
  /project/classification/gaoyonghan/models/dima806--medicinal_plants_image_detection/inference.py --phase full
bash classification/gaoyonghan/environment/run-e07.sh /opt/model-envs/e07/bin/python \
  /project/classification/evaluate.py --run-dir /project/classification/gaoyonghan/models/dima806--medicinal_plants_image_detection/result/run_001
```

冒烟（只跑 10 张，只验证链路，结果不作数）：

```bash
python inference.py --phase smoke
```

`--show-command` 只打印将要执行的命令，不跑模型。`--output-root result/run_002` 可另存一次运行。

## 权重

随本目录交付，位于 `weights/`（5 个文件，327.5 MB：`config.json`、`preprocessor_config.json`、`model.safetensors` 与上游仓库 model card）。推理时直接从该目录离线加载，不依赖外部路径；逐文件 sha256 与上游仓库信息见 `weights/weight_verification.json`。若改用外部权重目录，把 `WEIGHTS_ROOT` 指向包含 `<dirname>/` 的目录即可覆盖。

## provenance

`provenance/` 记录来源与验证：

| 文件 | 内容 |
|---|---|
| `mx_layout.json` | 本目录布局、权重文件与上游仓库的对应关系 |
| `split_audit.json` | 评测集文件哈希、张数、类别分布，并与 `expected_images` 比对 |
| `validation.json` | 本次实测指标、作者公布值，以及该数值能否作为精度引用的判定 |

## 数据集

共享 `dataset/indian_170`，3680 张。

sydney31/Indian_Medicinal_Plants_and_Leaf_Dataset_Large 的 med.zip（13.28GB）流式读取

## 评测集准备

评测集不在共享 `dataset/` 时，`inference.py` 会自动调用 `inference/native/prepare_dataset.py`，参数：`--row 170 --apply --classmap indian_170_classmap.json`。
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
| 评测集 | `dataset/indian_170` · 3680 张 |
| Top-1 accuracy | 0.86196 |
| Balanced accuracy | 0.86196 |
| Macro-F1（观测到的 GT 类） | 0.86188 |
| Top-5 accuracy | 0.89375 |
| 全流程墙钟 | 65.5 s（加载 4.8 s + 预热 4.3 s + 前向 2.21 s） |
| 前向吞吐 | 0.600 ms/张（1666 张/s） |
| 峰值显存 | 988 MiB（本进程本段） |
| 设备 / 精度 / batch | MetaX C500 · amp_bf16 · 64 |

指标由 `classification/evaluate.py` 计算（sklearn），逐类明细见 `result/run_001/standardized_classification/metrics.json`，运行记录见 `status.json` / `timing.json`。

## 已知限制

- 模型为 52 类输出，其中 6 对同物异名（Curry/Curry_Leaf、Pappaya/Papaya、Doddpathre/Doddapatre、Tulsi/Tulasi、Gauva/Guava、Bhrami/Brahmi）合并为 46 簇；prediction 是簇内 logsumexp 后的簇号，不是原生 52 类索引。
- 评测集自建：从 med.zip 的 169 个类目录中取模型能对上的 46 个，每簇抽样不超过 80 张，与训练集同源、可能重叠，高分属合理范围。
- 作者 99.62% 为 Kaggle 私有切分（6012 张），公开 zip 无法还原，不作复现判定。
- `predictions.jsonl` 保存原生 Top-5 与对应分数（`topk` / `topk_scores`），Top-5 accuracy 由 evaluate.py 直接统计，未重建 logits。
