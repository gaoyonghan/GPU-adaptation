# tiiuae__Falcon-Perception

使用 environment/ 中记录的掩码环境。原始数据集：`PBench`。

在项目根目录运行：

```bash
python mask-generation/lirongfu/models/tiiuae__Falcon-Perception/inference.py
```

也可以进入当前模型目录执行 `python inference.py`。ONNX 模型需要 Node.js 20 或更新版本。
入口加载 `weights/best_model.pth`，使用模型原有预处理和原生推理流程，记录纯前向时延，然后封装 COCO 评测包并自动调用公共 `mask-generation/evaluate.py`。
批量设置：保留原生入口的逐图、逐帧或引擎批量设置，不由总入口覆盖。

结果与日志：

- `result/results.json`：mIoU，取值 0–1。
- `result/inference_time.json`：总前向秒数、样本数、平均每样本毫秒数；不包括数据预处理、封装和评分时间。
- `result/runs/<运行编号>/`：本次日志、状态、指标、计时，以及指向评测包的相对路径。
- `dataset/PBench_tiiuae__Falcon-Perception_coco/runs/<运行编号>/`：COCO 风格的 GT、预测掩码、协议和评分信息。

只重新评测最近一次成功的完整评测包：

```bash
python mask-generation/evaluate.py --run-dir dataset/PBench_tiiuae__Falcon-Perception_coco
```

默认数据、权重、结果路径均由脚本位置解析。可使用 `MASK_DATASET_ROOT`、`MASK_WEIGHTS_ROOT`、`MASK_RESULT_ROOT` 调整共享根路径；相对值相对于项目根目录。
公共评测按原协议处理单掩码、实例集合、视频对象分组或医学体积，不将 mIoU 改成 COCO AP。
