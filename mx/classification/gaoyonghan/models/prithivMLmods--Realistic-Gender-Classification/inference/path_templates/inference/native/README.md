# 171–200：官方推理外层循环与计时

交付目录：`__MODEL_ROOT__/resources/runtime/assets/workspace/classification171_200_eval`。
模型、数据均使用本机已有文件。脚本不下载模型、不训练、不重新编码图片、不替换指定权重。

## 直接运行

```bash
cd __MODEL_ROOT__/resources/runtime/assets/workspace/classification171_200_eval
bash run_all.sh full --device cuda:0
```

`full` 对每个模型先运行固定 10 张的冒烟验证，通过后才运行冻结清单中的全部数据；串行执行，单个失败会记录原因并继续其余模型。177 的五个权重逐一运行。
本次交付只进行了冒烟验证，没有启动全量推理。需要换到第二张卡时用 `--device cuda:1`。

可单独检查或运行指定模型：

```bash
bash run_all.sh preflight
bash run_all.sh smoke --device cuda:0
bash run_all.sh full --models 191 192 --device cuda:0
```

默认结果目录：`__MODEL_ROOT__/resources/runtime/assets/workspace/results/classification171_200_v1`。
同一命令可再次执行：已完成项跳过，HF 和 Transformers.js 从完整写入的 batch 日志继续；timm、Ultralytics 保留官方评估循环，失败后该模型重新评估，失败记录和日志保留。
代码、环境、GPU 编号、批量大小或预热参数变化时必须换结果目录，例如：

```bash
bash run_all.sh full --device cuda:1 --output-root __MODEL_ROOT__/resources/runtime/assets/workspace/results/classification171_200_gpu1_v1
```

默认 batch=1、PyTorch FP32。173 是原始量化 ONNX，通过官方 Transformers.js 在 CPU 上逐图运行，不能放进 GPU FP32 的速度排名。修改全局 batch 不会改变 173 的逐图流程，实际批量会记录。
每个 GPU 对本任务包加进程锁，每个结果目录另有锁；不会停止其他任务的 GPU 作业。正式效率比较前应确保设备没有其他计算任务。

返回码：0 表示请求项成功或按计划跳过；1 表示有失败；2 表示有阻塞。全量默认包含仍被阻塞的 174，因此完成其余项目后退出码 2 是预期情况，应看汇总表的具体状态。

## 官方流程如何保留

| 模型 | 执行入口 |
|---|---|
| 184、193、199、200 | 固定 commit 的 timm `validate.py`，原始 checkpoint，官方预处理和 top-1/top-5 评估 |
| 177 五个权重 | 固定版本 Ultralytics `YOLO(...).val(...)`，FairFace 0.25 validation |
| 179 | NVIDIA 模型卡的 `trust_remote_code=True`、原始模型文件和 `timm.create_transform` 参数 |
| 173 | 官方 Transformers.js `pipeline('image-classification')`，原始 q8 ONNX |
| 172、181 | 作者模型卡的 Transformers pipeline |
| 其余可运行模型 | 发布的 Transformers 模型类与 image processor；外层读图、循环、计时和保存 |

没有重写网络 forward，也没有统一替换 resize/crop/normalize。198 使用发布配置指定的 `ViTHybridForImageClassification` 与 `ViTHybridImageProcessor`，兼容历史 `vit-hybrid` 配置名称。
HF/JS 的首次计时预测会与关闭计时时的预测比较；必须保持类别一致，概率在冻结容差内。
177 为满足原生 validator 读取类别目录的要求，创建 `train -> val` 路径别名；只调用 `split='val'`，不训练、不重新划分数据。
部分 Gradio 演示把概率四舍五入到三位小数用于展示；指标保留舍入前概率，这一差异已记录。

只有单图示例而没有完整数据集 evaluator 的模型，依照用户授权在外面套循环，再用 sklearn 计算参考指标；结果明确标记 `reference_baseline`。
即使调用原生 evaluator，在作者历史运行环境或报告协议未完全核实时，也不声称 `exact_official`。
模型卡参考值和来源见 `model_card_references.json`，原始模型卡哈希、数据清单、标签映射和协议指纹见 `models.json`。
替代集、训练集参考评估不能与模型卡数字直接计算“掉点”。没有清楚定义的参考值保留为空。

## 输出和计时口径

- `results_full.csv`：每个模型/权重一行，包含状态、数据关系、准确率/F1、时间、吞吐量和结果路径。CSV 的 accuracy/top5 均统一为 0–1；原生 JSON 保留其原始单位。
- `audit_summary_full.json`：按唯一模型计数的进度；177 只有五份都完成才计为模型完成。
- `full/m序号[/对应权重]/status.json`：实际参数、版本、模型与数据清单、GPU 信息、失败原因及协议指纹。177 实际目录名为 `m177-权重文件名`。
- 每个结果目录的 `metrics.json`、`timing.json`、`console.log`：完整指标、计时及官方输出。HF/JS 另外保存逐样本预测与断点日志；原生 evaluator 保留自己的评估产物。
- `smoke/` 只用于流程验证，不生成正式 `metrics.json`，不能作为准确率或效率结论。

PyTorch forward 用 CUDA events 测量，并在读取计时前同步；CPU 用单调时钟。模型加载、预热单列，不计入 forward。
HF/JS 另记每批图像读入/解码、官方预处理、传输、forward、后处理到 CPU 预测可用的端到端时间；指标计算和写结果不计入这段时间。JS 读入的是原始编码图片的本地缓存，缓存准备时间属于整段运行时间。
timm 保留官方一次随机输入预热；它的评估循环混有 loss/accuracy，因此记录 forward 和评估循环耗时，`inference_e2e_seconds` 留空。
Ultralytics 使用原生 validator 的 `speed`，并单列验证总时间；不把总时间伪装成纯推理时间。
毫秒/张是实际批量下的摊销值，不是单请求延迟分位数。记录峰值显存及运行前后的 GPU 状态；续跑时加载、预热与显存字段明确标为当前分段。

参考指标包括 accuracy、balanced accuracy、逐类 precision/recall/F1、macro/micro/weighted F1，以及适用的 top-5、二分类 AUROC/log loss。
参考 cross entropy 使用已保存的真实类别概率（下限 `1e-300`），不冒充作者 Trainer 的原始 logits loss；197 的模型卡未明确 F1 平均方式，分别报告各种平均值。

## 当前数据边界

| 模型 | 冻结范围及限制 |
|---|---|
| 174 | ChangeLing18K 数据访问仍缺少可用的本地数据，记录阻塞 |
| 182 | 按用户决定跳过 |
| 189 | 三类候选建筑风格，共 1,539 张；源标签比模型子类型更宽，只输出预测和耗时 |
| 190 | FairFace 年龄段与模型年龄段不能严格一一对应，10,954 张只输出预测和耗时 |
| 191、199、200 | 各 49,950 张、999 个重合类；保留完整 21K/22K 输出头，不改成 999 类 |
| 192 | 四类 800 条，793 个唯一图片哈希；已排除四条跨标签冲突，同类重复保留并披露 |
| 187 | 已接受的三城市 1,376 张；不再等待 Mapillary |
| 186 | 本地档案与官方 TestImages 清单相交的五类 98 张 |
| 172、180、183、194 | 已有公开 Train 分片或完整集参考评估，不宣称独立测试 |
| 175、185 | 只取标签含义可严格匹配的类别，保留模型完整输出空间 |
| 176 | 有标签的 25,000 张竞赛 train，不能当作 Kaggle 私有测试排名 |
| 197 | 标准 Fashion-MNIST test，区别于作者的 quality_drift 版本 |

因此准备就绪的是 **28 个模型的推理**，其中 **26 个可计算当前协议指标，2 个只做预测/计时**；174 阻塞、182 跳过。数据可用不等于原论文完整协议已经复现。

## 环境及验证

兼容的模型共用已有 `classification8_eval/.venv`，没有修改其依赖。183、177、179 各使用独立环境，173 使用独立 npm 目录及 lockfile。
交付目录中的这些环境和数据索引链接到本机已有目录，属于本机任务包，不是跨机器便携包。
环境已经配置，正常运行无需安装。需要核对/恢复时运行 `bash setup_env.sh all`；安装阶段需要网络，推理阶段离线。
Mamba 使用作者要求的 PyTorch 2.6/CUDA 12.4 和官方 Mamba 2.2.4 预编译包。Transformers.js 使用官方 CPU 安装选项跳过不需要的 CUDA 扩展下载。

验证记录见 `VALIDATION.md`。修改运行代码后请使用新输出目录；不要覆盖旧的模型清单或结果。`prepare.py` 仅用于生成另一份新协议清单，不是运行前必需步骤。
