# 首批八个分类模型

## 一键运行

```bash
cd __MODEL_ROOT__/resources/runtime/assets/workspace/classification8_eval
bash run_all.sh
```

默认在 GPU 0、1 各启一个工作进程；自动配置独立 `.venv`、校验资产、每模型先测 10 图，通过后全量。失败模型单独记录，其他模型继续。不会启动、终止或修改其他检测任务；同时运行会共享 GPU 资源。

只验收流程：`MAX_IMAGES=10 bash run_all.sh`。仅用一张卡：`GPU_IDS=1 bash run_all.sh`。自选新结果根目录：`OUTPUT_ROOT=/绝对路径/新目录 bash run_all.sh`。

## 范围与协议

| 编号 | 模型 | 数据 / 数量 | 实现及说明 |
|---|---|---|---|
| 71 | Diginsa/Plant-Disease-Detection-Project | PlantVillage color test / 10,709 | 发布者 HF 处理器与模型；公开替代基线 |
| 74 | timm/mobilenetv4_conv_small.e2400_r224_in1k | ImageNet val / 50,000 | 官方 timm validate；测试 256px、crop 0.95；模型卡 Top-1 74.616% |
| 84 | facebook/convnext-base-224 | ImageNet val / 50,000 | 发布者 HF 转换模型与处理器；224px |
| 87 | timm/efficientnet_b0.ra_in1k | ImageNet val / 50,000 | 官方 timm validate；224px、crop 0.875 |
| 88 | google/efficientnet-b3 | ImageNet val / 50,000 | 发布者 HF 模型与处理器；300px，不擅改 resize/std |
| 89 | SanketJadhav/PlantDiseaseClassifier-Resnet50 | PlantVillage color test / 10,709 | 发布者 HF 模型与处理器；公开替代基线 |
| 91 | ENOT-AutoDL/imagenet-benchmark | ImageNet val / 50,000 | ResNet50-ENOT-x2.pth；作者 load_data/evaluate；模型卡 Top-1 75.48% |
| 95 | haywoodsloan/ai-image-detector-dev-deploy | Synthbuster + RAISE / 3,000 | 发布者 HF 模型与处理器；公开替代基线 |

统一的是调度、记录和输出，不是擅自统一预处理。HF 五款用配套处理器/分类头，项目适配器负责读图、类别对应、批量推理、断点及 scikit-learn 指标；不是声称作者提供了完整 benchmark 脚本。timm 和 ENOT 调用未修改的官方评估函数；冒烟仅在数据入口截取子集。

71、89 的真实分类头均为 38 类植物病害，模型卡有模板文字或未说明原 split，不能拿替代测试集分数直接对照模型卡。95 原评估集未确认：按文件名 SHA256 固定每个生成器 300 张，加 300 张真实图；主指标是九组“生成器 vs 真实图”平衡准确率的均值，并输出各组 AUROC、整体 9:1 准确率及混淆矩阵。此三款一律 `not_officially_comparable`。

ImageNet 使用原 val 50,000 图和 LOC_val_solution.csv 单类别标签，校验无重复、类别与输出顺序一致。HF 两款 ImageNet 的 1,000 类描述与 timm 官方顺序逐项精确验证。植物类只规范化空格/下划线，必须有 38 类双射。

## 复现与输出

必要交付仅五个文件：本说明、models.json、setup_env.sh、run_all.sh、run_model.py。清单冻结本地 HF revision、文件 SHA256、权重变体、处理器配置、官方链接、数据关系、特殊情况和依赖版本。

环境复用现有 object-detection-hf-v2 的匹配依赖（只读），缺失依赖装到本包 `.venv`；不会更改既有环境。ENOT 原依赖为 torch 1.13.1 / torchvision 0.14.1，本包为 2.5.1 / 0.20.1；官方代码保持原样，但框架差异明确保留，不宣称已对齐数值。ENOT PTH 是作者提供的完整模型 pickle，仅加载清单冻结哈希的资产。

初次会联网装依赖并获取固定版本、哈希校验的 timm validate.py。后续推理离线，不自动下载/替换模型或数据。现有 ImageNet 大压缩包只解出 val 到 `.runtime/data/imagenet/val`（约 6 GB，需足够空间）；其他数据直接读取 ZIP。数据指纹含路径、大小、mtime、所选成员 CRC 和类别映射；不是对整个巨大 ZIP 做密码学全量哈希。

输出默认 `__MODEL_ROOT__/resources/runtime/assets/workspace/results/classification8_v1`：

- `smoke/`：10 图状态与预测；不生成正式指标。
- `m编号-模型/`：status.json；全量成功才有 metrics.json；HF 另有逐图 predictions.jsonl，原生 evaluator 保留官方日志。
- `编号.log`：该模型运行日志；`audit_summary.json`：批次状态与协议标签。
- `_source_snapshot/`：首次执行的五文件快照，后续重跑沿用，不静默更新协议。

中断后重跑同一命令：完成项跳过；HF 从完整已保存的预测继续，损坏/重复记录会报错；原生 timm/ENOT 未完成项从该模型开头重跑。锁防止同一输出目录重复启动。修改协议/源码后必须使用新 OUTPUT_ROOT，保留旧结果。旧 zero-shot / 检测结果不覆盖。

`complete` 只代表完整评估成功，不等于与官方论文数字一致；模型卡参考行与最终差值需要在结果产生后单独核查。替代基线不能列为官方复现。

## 交付验收记录（2026-09-08）

八款均已在实际 GPU 上完成 10 图测试，状态为 `partial_complete`，尚未启动全量；证据在默认结果目录的 `smoke/audit_summary.json`。语法、冻结环境、类别对应、权重加载、Top-1/Top-5 与二分类指标函数也已验证（指标函数单元测试为合成数据，不作为实验结果）。

首轮两款 timm 因 `persistent_workers` 不支持 `workers=0` 失败，现采用 `workers=2` 后均通过；未修改官方计分算法。首次测试证据另保留在 `results/classification8_validation_v1`，不得把其中两条旧失败当作当前状态。框架参数按实际运行记录在 status.json 的 backend_flags 中。
