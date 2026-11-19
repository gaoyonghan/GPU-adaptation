# MX 模型交付预览

适用于 MetaX C500。117 个模型已在 MX 上完成 GPU 小样本推理和评测。

| 任务 | 模型数 | 运行环境 |
| --- | ---: | --- |
| [分类](classification/) | 56 | [按模型查环境](classification/gaoyonghan/environment/README.md) |
| [检测](detection/) | 61 | [按模型查环境](detection/gaoyonghan/environment/README.md) |

环境镜像 `mx-model-envs:20261118` 保存在 MX 验收机器上。运行时提供模型对应的权重和数据，具体位置见模型目录的 README。使用镜像时，将本目录挂载到 `/project`，共享数据放在 `dataset/`，权重放在模型目录的 `weights/`。

例如，在镜像内进入 `classification/gaoyonghan/models/Modotte--AIRealNet/`，执行以下三步。

```bash
source /opt/model-envs/e33/bin/activate
python inference.py
python ../../../evaluate.py --run-dir result/run_001
```

部分分类模型还需要样本索引 `.jsonl`，随数据提供。具体相对路径见对应模型 README 和 `inference/local_paths.json`。
