# 图像分类：模型推理与统一评测

本目录是图像分类项目的任务目录，包含 `lirongfu` 下 58 个模型的推理入口、环境配置、运行日志和已保存的评测结果。各模型沿用各自的数据集、预处理、类别映射和原生推理流程，将预测转换为统一评测包，使用公共分类评测脚本计算指标。

## 目录结构

本 README 位于 `classification/`。完整部署中，`classification/` 与共享数据目录 `dataset/` 同级；放入 GitHub 仓库时，这一层对应 `GPU-adaptation/a100/` 平台根目录。

```text
classification/                         # 本 README 所在目录
├── README.md
├── evaluate.py                         # 公共分类评测脚本，当前副本未包含
└── lirongfu/
    ├── environment/
    │   ├── README.md                   # 环境说明与部署要求
    │   ├── classification-eval.txt     # Python 依赖版本清单
    │   └── model_environment.csv       # 模型与环境的对应关系
    └── models/<模型名称>/              # 共 58 个模型目录
        ├── README.md                   # 本模型的数据集、运行方法和限制
        ├── inference.py                # 单模型入口、日志与状态管理
        ├── inference/
        │   ├── native.py               # 权重加载、预处理、原生推理与计时
        │   ├── bundle.py               # 标准评测包转换与公共评测调用
        │   └── paths.py                # 数据、权重和结果路径配置
        ├── weights/                    # 当前副本为空，运行前需准备权重
        │   └── best_model.pth          # 部署时提供的模型权重包
        └── result/
            ├── results.json            # 保存的完整评测指标
            ├── inference_time.json     # 前向推理时延
            ├── latest.json             # 最近发布的完整评测包引用
            ├── smoke_results.json      # 受限样本运行时按需生成
            └── runs/<RUN_ID>/
                ├── stdout.log          # 此次运行的终端输出与报错
                ├── status.json         # 命令、环境、输入范围及运行状态
                ├── inference_time.json # 此次运行的前向计时
                └── evaluation_bundle.json
                                        # 通过相对路径指向此次评测包
```

58 个模型均采用上述入口和辅助脚本布局。目录名通常以 `__` 分隔模型发布者与模型名称，例如 `timm__inception_v4.tf_in1k` 对应 `timm/inception_v4.tf_in1k`。

当前本地副本包含模型代码和已有结果，58 个 `weights/` 均为空，任务目录内也未包含 `evaluate.py`。实际推理还需准备对应的 `best_model.pth`、共享数据和配套公共评测脚本；仅凭当前副本不能直接完成完整评测。

## 依赖与环境

环境清单记录全部模型共用 `classification-eval`，使用 Python 3.10、PyTorch 2.8.0+cu126、torchvision 0.23.0+cu126、NumPy 1.26.4。完整版本见 [classification-eval.txt](lirongfu/environment/classification-eval.txt)，模型对应关系见 [model_environment.csv](lirongfu/environment/model_environment.csv)。

在已有服务器上激活环境：

```bash
source /opt/venvs/classification-eval/bin/activate
export PYTHONNOUSERSITE=1
unset PYTHONPATH
```

新机器需先创建并激活 Python 3.10 环境。以下依赖安装命令从本 README 所在目录执行：

```bash
python -m pip install torch==2.8.0+cu126 torchvision==0.23.0+cu126 \
  --index-url https://download.pytorch.org/whl/cu126
python -m pip install --no-build-isolation -r lirongfu/environment/classification-eval.txt
python -m pip check
```

GPU 运行需要匹配的 NVIDIA 驱动；OpenCV 所需的 libGL/libglib 由系统提供。Mamba 相关依赖在没有可用预编译包时需要 CUDA 编译工具链，LSNet 使用的 Triton `ptxas` 需保留执行权限。具体说明见 [环境 README](lirongfu/environment/README.md)。

环境 README 和 CSV 中部分路径仍记录为 `classification/lirongfu_recovered/...`，这是历史部署目录名；当前目录为 `classification/lirongfu/...`，使用这些记录时需按实际目录对应。本 README 的命令均使用当前名称。

## 推理与评测流程

单模型入口执行：准备权重和数据 → 预处理 → 原生推理并记录前向时延 → 转换标准评测包 → 调用公共 `classification/evaluate.py` → 保存指标、日志与运行状态。每个模型可以独立启动，无需中央 `main.py` 调度。

以下运行与评测命令用于已准备好数据、权重及配套评测脚本的完整部署，默认从平台根目录执行。在本 README 所在目录执行 `cd ..` 即可到达该层；若位于 `GPU-adaptation` 仓库根目录，则先执行 `cd a100`。

### 运行一个模型

以 `timm__inception_v4.tf_in1k` 为例：

```bash
python -u classification/lirongfu/models/timm__inception_v4.tf_in1k/inference.py --device cuda
```

也可以进入模型目录后运行：

```bash
cd classification/lirongfu/models/timm__inception_v4.tf_in1k
python -u inference.py --device cuda
```

入口使用当前激活的 Python 环境，默认按模型原有配置处理测试数据。常用参数如下：

| 参数 | 含义 |
| --- | --- |
| `--device cuda` / `--device cpu` | 指定设备，默认 CUDA；具体后端的设备支持以模型实现为准 |
| `--batch-size N` | 覆盖模型原配置的批量大小 |
| `--max-images N` | 只处理前 N 张图片，0 为全量；也兼容 `--max-samples` |
| `--checkpoint 路径` | 指定权重包，相对路径按平台根目录解析 |
| `--output 路径` | 指定指标文件，须位于当前模型的结果目录内 |
| `--check` | 检查权重包和数据路径，不执行模型推理 |

从平台根目录进行路径检查或少量样本验证：

```bash
python classification/lirongfu/models/timm__inception_v4.tf_in1k/inference.py --check
python classification/lirongfu/models/timm__inception_v4.tf_in1k/inference.py --max-images 8
```

受限样本运行默认写入 `result/smoke_results.json`，不覆盖完整评测的 `result/results.json`。具体数据集、样本范围及模型特有条件见各模型 README。

### 评测包与共享数据

原始数据共享一份，评测包与原始数据集同级，按 `<数据集名>_<模型目录名>_imagenet` 命名。例如：

```text
<平台根目录>/
├── classification/
│   ├── evaluate.py
│   └── lirongfu/models/...
└── dataset/
    ├── imagenet_1k_val/                 # 原始数据集
    ├── _decoded_images/                # 按需解码的共享图片
    └── imagenet_1k_val_timm__inception_v4.tf_in1k_imagenet/
        ├── inference_output.json       # 最近发布的成功全量运行清单
        └── runs/<RUN_ID>/
            ├── annotations.json        # 真实标签
            ├── predictions.json        # 模型预测及可用的原生排序
            ├── protocol.json           # 数据范围、类别规则和评测协议
            ├── inference_output.json   # 此次运行清单
            └── metrics.json            # 公共评测输出
```

图片通过相对路径引用，不为每个评测包重复复制原始图片，也不创建软链接。需要从原始数据中解码的图片按需保存在共享的 `dataset/_decoded_images/`。目录后缀 `_imagenet` 表示统一的图片与标签清单组织方式，不表示所有模型都使用 ImageNet 数据集。

### 只重新评测

准备好公共脚本及完整评测包后，可读取已有预测重新计算指标，无需重新加载模型推理：

```bash
python classification/evaluate.py \
  --run-dir dataset/imagenet_1k_val_timm__inception_v4.tf_in1k_imagenet
```

评测指定历史运行时，将 `--run-dir` 改为该评测包的 `runs/<RUN_ID>/` 路径。公共命令生成包内 `metrics.json` 和评测汇总；正常推理入口还会将指标保存到模型的 `result/results.json`。

## 指标与推理时延

`result/results.json` 按 `model-eval/score-classification/v1` 协议保存分类指标。当前结果包含 `accuracy`、`top1_accuracy`、`top5_accuracy`、`balanced_accuracy`、宏平均精确率/召回率/F1、加权 F1，以及逐类别统计和评测范围。

指标单位为比例，例如 `0.85` 表示 85%。Top-5 依赖原生预测的类别排序；缺少排序时标记为不可用，少于五类时标记为不适用，应结合 `metric_status` 读取。不同数据集、类别映射和样本筛选规则下的指标，应结合 `protocol.json` 与结果中的 `qualification` 等字段理解。

`result/inference_time.json` 以模型目录名为键，包含：

- `total_latency_s`：累计前向推理时间，单位秒。
- `sample_count`：参与前向计时的样本数。
- `avg_latency_ms`：累计前向时间除以样本数，换算为毫秒。

这些字段记录前向推理时延，不等同于包含数据读取、预处理、结果转换和指标计算的整次任务耗时。不同设备、批量大小和输入配置下的时延需分别比较。

## 日志、结果与运行状态

每次正常启动会创建独立的 `result/runs/<RUN_ID>/`，将终端输出保存到 `stdout.log`，并在 `status.json` 中记录命令、环境、输入范围、输出目录和退出码。入口使用 `running`、`passed`、`failed`、`interrupted` 等状态；实际推理错误返回非零退出码。

`evaluation_bundle.json` 指向该次运行的评测包；成功发布完整评测后，`result/latest.json` 记录对应的包位置。失败或中断的运行可能只有日志和状态文件，不一定有计时与评测包引用。

当前副本保留了历史指标和运行记录，部分结果中的 `inference_executed` 为 `false`，表示不能将该次指标计算直接当作重新执行模型推理的证明。查看结果时应结合具体运行的日志、状态、输入范围和评测包；下载或整理目录不会自动触发新一轮推理。

## 数据与路径配置

各模型的 `inference/paths.py` 基于脚本位置确定平台根目录，默认使用以下位置：

| 内容 | 默认位置 | 可配置环境变量 |
| --- | --- | --- |
| 共享数据 | `<平台根目录>/dataset/` | `CLASSIFICATION_DATASET_ROOT` |
| 模型权重 | `<模型目录>/weights/best_model.pth` | `CLASSIFICATION_WEIGHTS_ROOT` |
| 模型结果 | `<模型目录>/result/` | `CLASSIFICATION_RESULT_ROOT` |

设置权重根目录后，程序按 `<权重根目录>/<模型目录名>/best_model.pth` 查找权重；设置结果根目录后，按 `<结果根目录>/<模型目录名>/` 保存结果。环境变量中的相对路径均按平台根目录解析，也可显式指定外部存储的绝对路径。

迁移部署时应一并准备匹配的公共评测脚本、原始数据、权重及所需评测包，并保持路径关系一致。仅复制本任务目录中的 `result/`，不会自动包含位于共享数据目录中的预测、标签和评测协议文件。
