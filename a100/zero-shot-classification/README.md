# 零样本图像分类项目说明

`zero-shot-classification/zhoujianjie/` 存放首批 62 个模型代码、环境清单及评测入口。目录层级与同分支的 `segmentation/` 一致：任务目录维护项目说明，每位贡献者在自己的子文件夹中维护实现。

```text
a100/
├── classification/
├── mask-generation/
├── segmentation/
└── zero-shot-classification/
    ├── README.md                       # 本任务的项目说明
    └── zhoujianjie/                     # 个人子文件夹
        ├── README.md                   # 个人代码和运行说明
        ├── evaluate.py                 # 公共分类评测入口
        ├── EVALUATE.md                  # 评测参数与输出说明
        ├── environments/               # 依赖清单与模型环境映射
        │   ├── README.md
        │   ├── model_environment.csv
        │   ├── zeroshot-*.txt
        │   └── delivery62_status.csv
        └── models/
            └── <组织>__<模型>/
                ├── inference.py        # 单模型原生推理和输出转换
                ├── inference/          # 模型专用源码、配置与辅助文件
                ├── weights/README.md   # 外部权重使用说明
                ├── result/.gitkeep     # 新预测、日志与评测结果保存位置
                ├── model_config.json
                └── README.md
```

## 模型与评测约定

1. 每个模型通过自己的 `inference.py` 启动原生流程，不依赖原始工程的中央启动器。模型专用 Prompt、类别映射、预处理和原有评分逻辑保留在个人模型目录中。
2. 权重、数据集、虚拟环境、缓存和大型历史预测不随 GitHub 代码上传。运行时通过 `--weights`、`--dataset-root` 指定真实资源。需要额外模型资源时，使用 `ZERO_SHOT_WEIGHTS_ROOT` 指定包含 `<组织>__<模型>/weights/` 的共享模型目录。
3. 推理结果保存在本模型的 `result/` 内；默认完整流程使用 `result/run_001/`，已有结果不覆盖，重新运行时应选择新的输出目录。
4. 分类推理生成 `inference_output.json`，其中的 bundle 包含 `annotations.json`、`predictions.json` 和 `protocol.json`。`zhoujianjie/evaluate.py` 沿用仓库公共分类评分规则，输出 Top-1，并在原生有序排名可用时输出 Top-5；缺少排名时不伪造 Top-5。
5. 首批 62 个模型包含 51 个分类模型和 11 个保留原始检索协议的模型。检索模型不能用单标签分类评测器计算指标，不将 Recall@K 冒充 Top-1；具体限制见个人目录的协议说明。
6. 这批模型的原工程运行状态、独立代码 GPU 小样本验证状态和预测一致性分别记录。小样本验证不代表全量精度或官方指标验收，目录整理也不代表重新完成推理。

## 环境与依赖

使用 Python 3.10，按 `zhoujianjie/environments/model_environment.csv` 为模型选择环境清单。main、MetaCLIP2、OneAlign、mmE5、FG-CLIP 等保留各自兼容版本，不强制合并依赖。

依赖文件记录原 A100 已观测版本，尚未保证全新安装一次成功。只在自己的可执行存储目录创建环境，不修改共享服务器原有运行环境。

```bash
# 从仓库根目录进入任务目录
cd a100/zero-shot-classification

# main 环境示例，其他模型按映射表选择清单
python3.10 -m venv /自己的可执行目录/venvs/zeroshot-main
source /自己的可执行目录/venvs/zeroshot-main/bin/activate
python -m pip install -r zhoujianjie/environments/zeroshot-main.txt
```

## 运行单个分类模型

以下命令从 `a100/zero-shot-classification/` 执行，以 RSICD 模型为例；资源路径需要替换为实际挂载位置。

```bash
python zhoujianjie/models/flax-community__clip-rsicd-v2/inference.py \
  --weights /实际共享模型目录/flax-community__clip-rsicd-v2/weights \
  --dataset-root /实际dataset目录 \
  --output-root result/run_001

python zhoujianjie/evaluate.py \
  --run-dir models/flax-community__clip-rsicd-v2/result/run_001
```

推理的相对 `--output-root` 以该模型目录为基准；评测的相对 `--run-dir` 以 `zhoujianjie/` 为基准，均可从其他工作目录启动。数据目录保留原 `raw/`、`standardized/` 层级，默认共享数据位置为仓库平台目录 `a100/dataset/`，也可以显式传入外部路径。

少量真实样本诊断可在推理命令中追加 `--smoke --max-images 8`；不同模型的最低样本数和诊断范围见各自说明。省略 `--smoke` 时执行原完整流程。本次目录规范调整没有运行任何模型。

## 相关文档

- [个人目录与代码说明](zhoujianjie/README.md)
- [环境使用说明](zhoujianjie/environments/README.md)
- [模型环境映射](zhoujianjie/environments/model_environment.csv)
- [62 个模型交付状态](zhoujianjie/environments/delivery62_status.csv)
- [分类评测说明](zhoujianjie/EVALUATE.md)
- [原生任务与公共分类协议兼容性](zhoujianjie/environments/PUBLIC_PROTOCOL_ISSUES.md)
