# 图像分割项目说明

`segmentation/qiuchuangxin/` 提供 70 个分割模型的推理、COCO 评测包导出和统一评测入口。原始评测数据由项目级 `dataset/segmentation/` 集中管理，目前包括 31 个数据集。

```text
model inference for GPU adaptation/
├── dataset/
│   └── segmentation/               # 共享的原图、标注和冻结样本清单
└── segmentation/
    ├── README.md
    └── qiuchuangxin/
        ├── evaluate.py             # 公共分割评测入口
        ├── run_all_inference.py    # 串行批量入口
        ├── environment/            # 依赖清单和模型环境映射
        ├── venv/                   # 主环境
        ├── venv_tf/                # TensorFlow 推理环境
        └── Model/
            └── <模型名>/
                ├── inference.py    # 单模型推理与导出入口
                ├── evaluate.py     # 单模型评测入口
                ├── weights/        # 权重、必要资源和 model.json
                ├── result/         # 预测缓存、环境、计时和指标
                │   └── coco_bundle/  # COCO 评测包
                └── README.md       # 数据集、环境与运行说明
```

## 目录与评测约定

1. 每个模型的独立入口、权重和运行结果都位于 `qiuchuangxin/Model/<模型名>/`。模型使用的数据集和划分见该模型的 `README.md` 与 `weights/model.json`。
2. `inference.py --stage inference` 生成预测缓存，`--stage export` 从缓存导出评测包，`--stage full` 执行完整流程。在完整流程中加入 `--skip-evaluation` 可在导出后暂停，由 `evaluate.py` 单独评测。
3. COCO 评测包保存到模型的 `result/coco_bundle/`，包含 `manifest.json`、`ground_truth.json` 和 `predictions.json`。统一指标为二值 IoU 或多类 mIoU，以及区域 Mask AP@[.50:.95]、AP50 和 AP75。区域 AP 按每图每类合并区域，预测区域分数统一为 1。指标写入 `result/results.json`，环境和计时记录也保存在 `result/`。
4. 命令行传入的相对路径以 `segmentation/qiuchuangxin/` 为基准，`weights/model.json` 中的相对路径以该模型目录为基准。Shell 中的含空格路径须加引号。

## 运行单个模型

以 `CIDAS__clipseg-rd64-refined` 为例，以下命令从 `segmentation/` 开始执行。

```bash
cd '/data/model inference for GPU adaptation/segmentation'
source ./qiuchuangxin/venv/bin/activate
cd qiuchuangxin/Model/CIDAS__clipseg-rd64-refined

python inference.py --device cuda --environment current --stage full --skip-evaluation
python evaluate.py --environment current --bundle "$PWD/result/coco_bundle"
```

## 批量运行

进入 `segmentation/qiuchuangxin/`，用 `python --version` 确认当前解释器是 Python 3.10，且与项目虚拟环境版本匹配。批量脚本按模型目录名排序串行运行。

如需重测，可以分步执行以下命令。推理成功后再导出，导出成功后再评测。导出需要对应模型有完整且兼容的预测缓存。

```bash
cd '/data/model inference for GPU adaptation/segmentation/qiuchuangxin'

# 重新推理，替换当前预测缓存
python -I -S -B run_all_inference.py --venv auto --mode inference --force

# 从新缓存导出，替换当前 COCO 评测包
python -I -S -B run_all_inference.py --venv auto --mode export --force

# 使用公共入口评测已有评测包
python -I -S -B evaluate.py --venv venv
```

也可以在同一目录用一条命令完成这三个阶段。

```bash
python -I -S -B run_all_inference.py --venv auto --mode full --force
```

新缓存或评测包通过校验后会成为当前结果，被替换的文件保存在各模型的 `result/.history-*/` 下。省略 `--force` 时，脚本会尝试复用通过校验的已有结果。

批量脚本也支持 `--mode evaluation`。用 `--model <模型名>` 选定模型，或用 `--start-from <模型名>` 从指定模型起继续运行，二者择一。`--dry-run` 可查看运行计划。

## 环境与依赖

项目使用 Python 3.10。大部分模型的推理及所有模型的导出、评测使用 `venv/`。TensorFlow 模型 `Narsil__pet-segmentation`、`SerdarHelli__Segmentation-of-Teeth-in-Panoramic-X-ray-Image-Using-U-Net` 和 `keras-io__semantic-segmentation` 的推理使用 `venv_tf/`。批量入口的 `--venv auto` 会按阶段和模型选择环境。

逐模型环境对应关系和两套依赖清单集中在 `qiuchuangxin/environment/`。 [环境使用说明](qiuchuangxin/environment/README.md) · [模型环境映射](qiuchuangxin/environment/model_environment.csv) · [主环境依赖](qiuchuangxin/environment/requirements-main.txt) · [TensorFlow 环境依赖](qiuchuangxin/environment/requirements-tensorflow.txt)
