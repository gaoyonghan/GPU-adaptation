# GPU-adaptation

按 GPU 平台选择模型和运行环境。

| 平台 | 任务 | 模型环境对照 |
| --- | --- | --- |
| A100 | [分类](a100/classification/) | [分类模型与环境](a100/classification/gaoyonghan/environment/README.md) |
| A100 | [检测](a100/detection/) | [检测模型与环境](a100/detection/gaoyonghan/environment/README.md) |

使用 A100 时，从仓库根目录进入 `a100/`。根据环境对照表激活环境，在模型目录运行 `python inference.py`，再按模型 README 的命令运行任务目录的 `evaluate.py`。

运行时将共享数据集挂载到 `a100/dataset/`，模型权重放在对应模型的 `weights/` 下。仓库仅提供代码和说明。

贡献者请参阅[项目结构规范](PROJECT_STRUCTURE.md)。
