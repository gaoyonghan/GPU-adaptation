[中文](README.md) | [English](README.en.md)

# depth-estimation

## 目录与入口

```text
a100/
  dataset/
  depth-estimation/
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

每个模型通过 inference.py 启动，任务目录 evaluate.py 执行公共评分。依赖映射见 zehong/environment/。数据集共用 a100/dataset，权重放入模型 weights。

## 使用方法与评分范围

下面命令从本任务目录运行，先激活模型对应的环境并准备资产。

推理入口读取共享数据并保存逐图预测与索引。evaluate.py 使用这些预测和真实深度独立计算 AbsRel、RMSE、Log10、delta1/2/3，并记录成功、跳过和失败样本。支持的任务数据集协议包括 NYUv2、KITTI、DIODE 和 Sintel，每个模型实际支持范围以其 --help 和数据加载实现为准。

evaluation_config.json 描述数据路径和预测类型。米制深度、相对深度和相对逆深度采用各自协议，不能互换。具体裁剪、有效范围和尺度平移处理由公共评分器实现。结果目录包含 inference_output.json、inference_summary.json、evaluation_config.json，以及数据集子目录中的 inference_records.csv 和原始预测。评分后生成 summary.json 与 per_image_metrics.csv。

```bash
python zehong/models/DepthPro/inference.py --output-root result/run_001
python evaluate.py --run-dir "$PWD/zehong/models/DepthPro/result/run_001"
```

forward_ms 记录每图网络前向耗时，扩散模型可累计多个网络组件调用。stage1_dataset_seconds 覆盖正式数据迭代和预测保存。预热和加载位于正式计时之外。跨设备比较须保持数据、精度、分辨率、去噪步数、集成次数和计时边界一致。

## 路径与结果

默认路径依据脚本位置解析，模型参数可指定权重、数据和输出。模型相对输入输出以模型目录为准。默认输出为 result/run_001，重复运行请选择新目录。评分示例使用绝对结果路径。

模型 README 给出其输入和评测限制。省略 --max-samples 使用完整声明划分，指定该参数用于功能检查。判断完成状态时同时核对样本数及错误记录。代码整理和迁移不构成新环境或新硬件上的推理验证。
