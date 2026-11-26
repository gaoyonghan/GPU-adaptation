[中文](README.md) | [English](README.en.md)

# image-feature-extraction

## 目录与入口

```text
a100/
  dataset/
  image-feature-extraction/
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

推理入口按模型原生预处理生成特征。evaluate.py 比较相同样本在基准端与候选端的特征，报告余弦相似度、绝对误差和相对 L2 等指标。比较需要相同 checkpoint、输入内容和特征语义。该评分表示跨设备输出一致性。

```bash
python zehong/models/GenView-vit-small/inference.py --output-root result/run_001
python evaluate.py /path/to/a100/run /path/to/candidate/run --output-root /path/to/comparison_001
```

### 两个输入结果目录

第一个路径 /path/to/a100/run 是 A100 生成的基准结果目录（baseline），第二个路径 /path/to/candidate/run 是待比较设备生成的结果目录（candidate），例如 910B 的结果。两者都指向某一次推理的 run 目录，例如模型下的 result/run_001。

每个 run 目录需整体保留以下评分文件与相对目录结构。

```text
run_001/
├── evaluation_config.json    # 权重标识、预处理、输入输出定义
├── summary.json              # 提取状态、样本数及性能信息
├── input_manifest.jsonl      # 样本标识与原始图片校验值
├── feature_index.jsonl       # 每个样本的特征文件路径和校验值
├── features/                # 索引指向的特征张量文件
└── model_environment.json    # 可选的运行环境信息
```

应传入包含上述文件的 run 根目录；features/ 是其中的子目录。两端须使用相同模型权重、同一组原始样本和相同特征定义，并使用评分器支持的相同结果格式。

评分脚本需要在运行它的机器上同时读取这两个目录。先将另一端的完整 run 目录复制或挂载到本机，再组织路径。例如在 910B 上将 A100 结果保存到 /workspace/feature_compare/a100_run，将本端结果保存到 /workspace/feature_compare/910b_run，可执行下面的命令。

```bash
python evaluate.py /workspace/feature_compare/a100_run /workspace/feature_compare/910b_run   --output-root /workspace/feature_compare/comparison_001
```

前后顺序决定比较方向，相对 L2 的分母取第一个目录的基准特征范数。--output-root 是新建的比较报告目录，其中保存 comparison_summary.json 和 image_similarity.csv。该目录需尚未存在。评分读取已保存特征，运行时无需连接另一端执行推理。

结果格式、输入清单和特征布局保留各模型已有输出契约。--strict-inputs 可选，仅要求记录的预处理配置一致。原图内容与样本集合始终检查；processor 文件和输入张量的哈希差异只记录，不阻止评分。输出差异可能包含预处理数值差异。阈值和可选参数见 python evaluate.py --help。各模型的输入要求、特征定义和输出范围见模型 README 与 assets.json。

## 路径与结果

默认路径依据脚本位置解析，模型参数可指定权重、数据和输出。模型相对输入输出以模型目录为准。默认输出为 result/run_001，重复运行请选择新目录。评分示例使用绝对结果路径。

模型 README 给出其输入和评测限制。省略 --max-samples 使用完整声明划分，指定该参数用于功能检查。判断完成状态时同时核对样本数及错误记录。代码整理和迁移不构成新环境或新硬件上的推理验证。
