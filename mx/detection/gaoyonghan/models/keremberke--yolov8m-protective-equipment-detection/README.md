# keremberke/yolov8m-protective-equipment-detection

模型编号：61；任务：detection；MX 环境：e26。

从模型目录执行：

```bash
python inference.py --phase full
python ../../../evaluate.py --run-dir "$(pwd)/result/run_001"
```

从项目根目录执行推理时使用 `detection/gaoyonghan/models/keremberke--yolov8m-protective-equipment-detection/inference.py`。默认结果在本模型 `result/run_001`；可以用 `--output-root result/run_002` 保留不同运行。`--show-command` 只展示命令，不执行模型。

权重在 `weights/`，共享数据在项目根 `dataset/`。依赖、容器和逐模型状态见 `../../environment/README.md` 与 `../../environment/model_environment.csv`。本次仅完成目录整理，未重新执行推理。排除或历史阻塞状态不因整理而改变。原 MX 推理来源和本次迁移记录保存在 `provenance/`；原项目目录保留。
