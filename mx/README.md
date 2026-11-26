# MX 模型交付

| 任务 | 模型数 | 模型与运行环境 |
| --- | ---: | --- |
| 分类 | 115 | [分类环境清单](classification/gaoyonghan/environment/README.md) |
| 检测 | 61 | [检测环境清单](detection/gaoyonghan/environment/README.md) |

环境镜像为 `mx-model-envs:20261124`。将本目录挂载至 `/project`，共享数据放在 `dataset/`，权重放在各模型的 `weights/`。按环境清单激活模型对应的环境，在模型目录执行 `python inference.py`，再执行 `python ../../../evaluate.py --run-dir result/run_001`。将 `run_001` 替换为本次推理生成的目录名。
