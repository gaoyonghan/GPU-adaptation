# bwittmann__vesselFM

图像分割模型。推理环境为 `venv`，导出和评测环境为 `venv`。

## 数据集

- `OCTA`；默认划分：`public_manual_test`。

## 文件与数据

- 模型配置和权重：`weights/`，入口配置为 `weights/model.json`。
- 数据集根目录：`/data/model inference for GPU adaptation/dataset/segmentation`。
- 预测缓存、运行环境和推理时间：`result/`；精度指标保存于 `result/results.json`。
- COCO 评测包：`/data/model inference for GPU adaptation/segmentation/qiuchuangxin/Model/bwittmann__vesselFM/result/coco_bundle`。

## 使用

以下命令从项目根目录运行，由系统 Python 入口选择已有环境：

```bash
cd '/data/model inference for GPU adaptation/segmentation/qiuchuangxin'
/usr/bin/python3 -I -S -B Model/bwittmann__vesselFM/inference.py --venv venv --stage inference
/usr/bin/python3 -I -S -B Model/bwittmann__vesselFM/inference.py --venv venv --stage export
/usr/bin/python3 -I -S -B Model/bwittmann__vesselFM/inference.py --venv venv --stage full
/usr/bin/python3 -I -S -B Model/bwittmann__vesselFM/evaluate.py --venv venv
```

`inference` 生成预测缓存；`export` 使用缓存导出标准化结果；`full` 执行完整流程。也可通过独立的 `evaluate.py` 评测已有结果。

命令行传入的相对路径以项目根目录为基准；`weights/model.json` 中的相对路径以本模型目录为基准。查看参数可在对应命令末尾添加 `--help`。
