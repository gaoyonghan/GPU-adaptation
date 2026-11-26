# dima806/traffic_sign_detection
模型编号：217；任务：classification；MX 环境：e07。

## 运行

从模型目录执行（简版）：

```bash
python inference.py
```

从项目根执行（wrapper 版，等价于上面的命令）：

```bash
sudo bash classification/gaoyonghan/environment/run-e07.sh /opt/model-envs/e07/bin/python \
  /project/classification/gaoyonghan/models/dima806--traffic_sign_detection/inference.py --phase full
```

补充用法：

- `--phase smoke` 只跑 10 张，用来验证链路是否通，结果不作数。
- `--output-root result/run_002` 换一个运行目录，默认 `result/run_001`。
- `--show-command` 只打印将要执行的命令，不推理；排查路径问题用这个。

`inference.py` 内部依次做四件事：把 `inference_output.json` 写成 running → 调 `inference/native/run_model.py` 做全量前向 → 自动调 `inference/prepare.py` 生成标准化评分包 → 写成 completed 并登记评分包相对路径。

**本模型自带一个不在 e07 里的依赖。** `inference/vendor/` 随目录携带 `imbalanced-learn 0.14.2`（纯 Python wheel，解包约 1.1 MB），由 `entrypoint.json` 的 `PYTHONPATH` 注入。原因：e07 镜像没有这个包，而容器以 `--network none` 启动、又是 `--rm` 一次性容器，无法现场安装。它的依赖（scipy 1.12.0 / scikit-learn 1.5.2 / numpy 1.26.4）全部由 e07 提供，只有 imblearn 自身这层代码需要随包走。

版本必须是 **0.14.2**：这是原始评测环境（`sunyuqiyi-c500` 容器）里安装的版本。作者的测试集是运行时用 `RandomOverSampler(random_state=83)` 现算出来的，换版本可能切出不同的 9252 张，分数就对不上了。

评测用顶层共用脚本，不要在模型目录里另建评测入口：

```bash
sudo bash classification/gaoyonghan/environment/run-e07.sh /opt/model-envs/e07/bin/python \
  /project/classification/evaluate.py \
  --run-dir /project/classification/gaoyonghan/models/dima806--traffic_sign_detection/result/run_001
```

## 权重

随本目录交付在 `weights/`，共 4 个文件、约 **328 MB**，全部离线加载（`local_files_only=True`），不访问网络：

| 文件 | 大小 | 说明 |
|---|---|---|
| `model.safetensors` | 343,356,252 B | 权重本体，200 个张量，全 F32 |
| `config.json` | 3,166 B | 模型结构与 45 类 `id2label` |
| `preprocessor_config.json` | 578 B | 预处理参数（224、ImageNet 均值方差） |
| `README.md` | 3,832 B | 上游仓库自带的 model card |

逐文件 sha256 与仓库来源见 `weights/weight_verification.json`。

未随目录交付：三个训练检查点（`checkpoint-1302` / `checkpoint-2895` / `checkpoint-3255`，约 2.9 GB）与 `training_args.bin`，均为训练期产物，推理不使用。

## 数据集

共享 `dataset/traffic_signs_tuanai/archive.zip`，**9252 张、45 类**。

来源是作者 Kaggle 数据集（`tuanai/traffic-signs-dataset`）。需要注意：磁盘上没有现成的测试集，作者的 test 划分是运行时动态构造的——先按 `Unknown*` 合并成 45 类，用 `RandomOverSampler(random_state=83)` 过采样到每类 514 张，再用 `train_test_split(test_size=0.4, stratify, random_state=42)` 切出 9252 张。`inference/native/run_model.py` 内联了这套流程。

## 输出

`result/run_001/` 下：

| 文件 / 目录 | 是什么 |
|---|---|
| `inference_output.json` | 本次运行状态机：running → completed，含评分包的相对路径登记 |
| `native/predictions.jsonl` | 逐样本预测，每行 `sample_id` / `target` / `prediction` / `topk` / `scores` |
| `native/classification_annotations.json` | 逐样本真值（`samples[].label`） |
| `native/classification_capture.json` | 原生捕获状态，`state=complete` |
| `native/status.json` | 原生推理完成态、张数、设备、权重路径 |
| `native/standardized_classification/` | 标准化评分包，含 `protocol.json` / `annotations.json` / `predictions.json` |

## provenance

`provenance/` 三个记录文件，不参与运行，删掉也能跑；作用是别人质疑结果时能查证：

| 文件 | 记录什么 |
|---|---|
| `mx_layout.json` | MX 侧的目录布局、运行入口、设备与精度 |
| `split_audit.json` | 数据集切分协议、样本编号规则、类别数、**泄漏情况** |
| `validation.json` | 交付前已通过的校验项、参考结果与本次运行完成后的预期 |

## 实测结果

| 项 | 值 |
|---|---|
| 评测集 | `dataset/traffic_signs_tuanai` · 9252 张 |
| **Top-1 accuracy** | **0.94012** |
| Balanced accuracy | 0.94013 |
| Macro-F1 | 0.93841 |
| Top-5 accuracy | 0.99546 |
| 墙钟与单图耗时 | **36.9 s** / **0.9856 ms**（单图为纯前向口径，warmup 5 批已剔除；吞吐 1014.56 张/s） |
| 峰值显存 | 1118.0 MiB（batch 64 · 224px · fp32，运行期分配峰值） |
| 设备与精度 | MetaX C500 / fp32 |

以上为 **2026-11-25 在 e07 实测**的结果（`result/run_001`，`--phase full`，9252/9252 张），与交付前统一口径复跑值**逐位相同**（Top-1 0.9401210549070471、Top-5 0.9954604409857328）。

精度口径为 **fp32**：参照实现记录的精度字段是 `amp_bf16`，本模型未启用 autocast，目的是与交付前终值逐项对账。如需 bf16 数据，须另跑一轮并重新对账。

## 已知限制

**这个数字不能当泛化精度看。** 作者构造测试集时先过采样再切分，导致 **84.1%（7779/9252）的测试图在训练集中出现过**，属于数据泄漏，精度因此虚高。复现作者自报的 93.88% 可以证明权重与标签映射正确，但不代表模型对新图片的判别力。

此外：

- 只有 tuanai 口径登记在册。另一套跨拍摄独立测试集（ahemateja，官方 TEST 1992 张，Top-1 **85.94%**，更能代表泛化水平）未登记——因为它涉及 58→45 的类别合并，属于协议变更，不是格式转换。
- 45 类中有 8 个 `Unknown` 子类被合并为一类，类名来自数据集的 `labels.csv`，与模型 `config.id2label` 顺序逐位对齐（脚本启动时会断言，不一致直接拒绝运行）。
