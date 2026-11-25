[中文](README.md) | [English](README.en.md)

# 图像分类

## 目录与入口

```text
910b/
  dataset/
  classification/
    evaluate.py
    README.md
    zehong/
      environment/
      models/<model>/
        inference.py
        inference/
        weights/
        result/
        requirements.txt
        README.md
```

每个模型通过 inference.py 完成推理及评分包生成。公共 evaluate.py 读取已保存的预测、真实标签和评测协议，离线计算指标。模型目录中的 README 说明数据划分和标签映射，environment/model_environment.csv 指定依赖清单及 Python 版本。

## 使用方法

先准备模型 weights 和共享 dataset，在对应环境中加载 CANN，再从 classification 目录运行。

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
python zehong/models/bit-50/inference.py --output-root result/run_001
python evaluate.py --run-dir "$PWD/zehong/models/bit-50/result/run_001"
```

默认权重和数据路径按入口位置解析。模型相对输入输出路径以模型目录为准，默认结果为 result/run_001，重复运行须选择新目录。公共评分器的相对 --run-dir 与 --bundle 以 classification 任务目录为准，绝对路径也可使用。

完整推理省略 --max-samples，功能检查可加 --max-samples 1。batch、workers 等参数见模型 --help。模型权重、数据集和实际环境需单独准备。

## 结果与评分

inference_output.json 记录 completed 状态和 scoring_bundle 相对路径。评分包包含 predictions.json、annotations.json、protocol.json。evaluate.py 写入 evaluation_summary.json 和评分包内的 metrics.json，提供 Top-1、可用时的 Top-5 及其他分类指标。数据划分与指标限制按各模型协议处理。

检查完成状态、样本数与指标状态后再验收。前向计时及端到端耗时来自推理记录，离线评分不重新测量推理性能。
