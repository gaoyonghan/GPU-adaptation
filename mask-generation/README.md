# 掩码生成：模型推理与统一评测

本目录是掩码生成项目的任务目录，包含公共评测脚本和 16 个模型的推理入口、环境配置、运行日志及结果。MAC 专用模型不在本项目中。模型加载各自的开源权重，沿用已配置的数据集、预处理和推理流程，统一输出 mIoU。

## 目录结构

本文件位于下图内层 `mask-generation/`。外层同名目录是项目根目录，包含串行入口 `main.py` 和共享数据 `dataset/`。

```text

mask-generation/                     # 本 README 所在目录
    ├── README.md
    ├── evaluate.py                      # 所有模型共用的评测脚本
    └── lirongfu/
        ├── README.md
        ├── environment/
        │   ├── README.md
        │   ├── mask-eval.txt             # Python 依赖版本记录
        │   ├── model_environment.csv     # 模型与环境的对应关系
        │   └── deployment_ready.json     # 部署检查状态
        └── models/<模型名称>/
            ├── README.md
            ├── inference.py             # 单模型入口、日志与状态管理
            ├── inference/
            │   ├── native.py            # 权重加载、预处理与原生推理
            │   ├── bundle.py            # COCO 风格评测包导出
            │   ├── paths.py             # 数据、权重、评测及结果路径
            │   └── config.json          # 数据集、后端及批次设置
            ├── weights/best_model.pth    # 模型权重包（无权重下载版本不含此文件）
            └── result/
                ├── results.json         # 最近保存的 mIoU
                ├── inference_time.json  # 纯前向计时
                ├── latest.json          # 最近一次成功完整评测包的信息
                └── runs/<RUN_ID>/
                    ├── stdout.log
                    ├── status.json
                    ├── results.json
                    ├── inference_time.json
                    └── evaluation_bundle.json
```

结果文件按运行阶段生成；尚未结束或失败的运行可能只有日志、状态或部分产物。保留整个 `result/`，不要仅凭目录存在或旧指标文件判断本次运行成功。

## 依赖与环境

已配置环境使用 Python 3.10、PyTorch 2.12.0+cu126、torchvision 0.27.0+cu126 和 NumPy 1.26.4。完整 Python 包版本见 `lirongfu/environment/mask-eval.txt`，模型对应关系见 `model_environment.csv`。该文件记录项目环境，不表示各模型作者都使用同一套依赖。

在已有服务器上激活掩码环境：

```bash
source /opt/venvs/mask-eval/bin/activate
export PYTHONNOUSERSITE=1
unset PYTHONPATH
```

新机器需先准备 Python 3.10 环境，再安装依赖。以下命令从本 README 所在目录执行：

```bash
python -m pip install -r lirongfu/environment/mask-eval.txt \
  --extra-index-url https://download.pytorch.org/whl/cu126
python -m pip check
```

两个 ONNX 模型还需要 Node.js 20 或更新版本；Node.js 不由上述 pip 文件安装。使用 nvm 时执行 `nvm use 20`，然后用 `node --version` 确认。GPU 推理需要可用的 NVIDIA 驱动及 CUDA 运行环境；OpenCV 所需系统共享库也需由主机提供。

代码不会自动安装依赖或下载数据。仅下载本任务目录且排除权重时，得到的是代码与结果副本；实际推理还需提供权重和外层 `dataset/`。公共评测需要完整的评测包。

## 推理与评测流程

单模型入口依次执行：加载权重及配置 → 读取数据与预处理 → 推理并记录前向时延 → 导出 COCO 风格评测包 → 调用公共 `evaluate.py` → 保存指标、日志和运行状态。

所有示例均使用相对项目路径。除特别说明外，下面命令从外层项目根目录执行；从本 README 所在目录执行一次 `cd ..` 即可到达该目录。

### 运行一个模型

以 `facebook__sam-vit-base` 为例：

```bash
# 1、在已激活的掩码环境中，从项目根目录进入模型目录
cd mask-generation/lirongfu/models/facebook__sam-vit-base

# 2、推理、计时、导出评测包并自动评测
python -u inference.py --device cuda
```

也可以留在项目根目录执行：

```bash
python -u mask-generation/lirongfu/models/facebook__sam-vit-base/inference.py --device cuda
```

直接使用当前激活的 Python 环境，无需添加 `--environment current`。默认保留各模型现有配置；视频模型按原流程逐帧传播，SAM/ONNX 中的批次可能指提示点批次，并不等于一次处理同样数量的图片。具体设置见各模型 `inference/config.json`。




### 只重新评测

下面的命令不加载模型、不重新推理，读取已保存的最近一次成功完整 COCO 评测包：

```bash
python mask-generation/evaluate.py \
  --run-dir dataset/COCO-2017_facebook__sam-vit-base_coco
```

评测指定历史运行时，将路径改为该评测包的 `runs/<RUN_ID>/`。`--bundle` 是 `--run-dir` 的别名。评测会保存包内 `metrics.json` 和对应运行的指标；默认也写回模型的 `result/results.json`。

## 指标和推理时延

`result/results.json` 保存 mIoU，例如 `{"miou": 0.85}` 表示 85%。IoU 为预测掩码与真实掩码的交集除以并集。公共评测保留各任务的评测单元和聚合规则：单掩码计算 IoU；实例集合按一对一匹配并对未匹配项记零分；视频按对象与帧分组；医学体积按原协议处理。COCO 风格是数据组织方式，这里计算的不是 COCO AP，也不是 J&F。

不同数据集、提示方式及聚合规则下的 mIoU 不能仅凭指标名称相同就直接横向比较。评测包内的 `units`、`mask_pairs` 等为统计计数，不是额外的精度指标。

`result/inference_time.json` 以模型名称为键，保存：

- `total_latency_s`：总前向推理时间，单位秒。
- `sample_count`：累计参与前向计时的样本数。
- `avg_latency_ms`：总前向时间除以样本数，换算为毫秒。

计时不包含数据读取、预处理、评测包封装和指标计算。样本单位随模型流程而定，可能是图片、视频帧、提示实例或切片查询，不能将所有模型的计数一律视为独立图片数。

## 日志、结果与未完成运行

- 模型 `result/runs/<RUN_ID>/stdout.log` 保存该模型的终端输出与报错。
- `status.json` 记录运行状态、命令、环境、起止时间及退出码；应结合该文件判断是否完成。
- `evaluation_bundle.json` 通过相对路径指向此次评测包。
- 项目根目录的 `inference_logs/<RUN_ID>/` 保存串行入口日志、各模型日志和 `summary.json`；`inference_logs/completed/` 保存独立完成记录。
- 底层可续跑缓存可能位于服务器 `/tmp/mask-unified-*`，不在本任务目录内；下载 `result/` 不等同于下载全部断点缓存。

运行期间复制下来的日志和状态是下载当时的副本，可能仍显示 `running`，也可能尚未生成最终指标。下载不改变服务器上的运行状态；后续结果需要再次同步。

## 数据与路径配置

原始数据共享一份，评测包与原始数据集同级，按 `<原始数据集名>_<模型名>_coco` 命名。图片通常通过相对路径引用；需要解码的内嵌图片或切片存放在项目 `dataset/_decoded_images/`。

默认路径基于脚本位置解析；可以通过 `MASK_DATASET_ROOT`、`MASK_WEIGHTS_ROOT`、`MASK_RESULT_ROOT` 指定数据、权重、结果根目录，相对值按外层项目根目录解析。`MASK_NODE_BINARY` 可指定 Node.js 程序。

`environment/deployment_ready.json` 是部署检查记录，不是模型测试完成记录。其状态与当前部署资产对应；更换机器或只取得无权重代码副本后，仍需检查实际文件和环境。不要用改动该标记来替代数据及权重检查。
