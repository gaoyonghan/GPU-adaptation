# 零样本模型代码交付

这里的 a100/ 是 GitHub 协作仓库的平台目录名称。本次只整理仓库副本，不移动服务器原工程、权重、数据或历史结果。

```text
a100/zero-shot-classification/zhoujianjie/
├── README.md
├── evaluate.py
├── EVALUATE.md
├── environments/
│   ├── README.md
│   ├── model_environment.csv
│   ├── zeroshot-*.txt
│   └── delivery62_status.csv
└── models/
    └── 组织__模型/
        ├── inference.py
        ├── inference/
        ├── weights/README.md
        ├── result/.gitkeep
        ├── model_config.json
        └── README.md
```

首批仅上传 62 个通过独立 GPU 小样本验证的模型代码，不携带权重、数据集、虚拟环境、缓存、密钥或大型预测。其余 15 个目录在本地保留并由个人 .gitignore 排除。

51 个分类模型使用公共分类输出；11 个检索模型保留原始流程，公共分类评测器不支持其指标，不能冒充 Top-1。逐模型记录见 environments/delivery62_status.csv。

## 运行示例

从仓库根目录进入本目录，环境按 environments/model_environment.csv 选择，模型资源使用外部真实路径：

```bash
cd a100/zero-shot-classification/zhoujianjie
source /自己的环境目录/bin/activate
python models/flax-community__clip-rsicd-v2/inference.py --weights /实际权重目录 --dataset-root /实际dataset目录 --output-root result/run_001
python evaluate.py --run-dir models/flax-community__clip-rsicd-v2/result/run_001
```

第一条推理命令是默认全量流程示例，本次目录整理没有执行全量评测。如需小样本诊断，在推理命令追加 --smoke --max-images 8；不同原生协议的最少样本限制见模型说明。已有结果目录不能覆盖，请改用新的运行目录。

GPU 验证记录来自迁移目录前的独立副本。本次层级调整只验证路径、语法和公共评测接口，不声称重新完成 GPU 验证。默认共享数据根仍为仓库 a100/dataset/，也可显式指定。
