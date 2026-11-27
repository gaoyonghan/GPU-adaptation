# 周建杰：独立零样本模型代码交付副本

首批上传 62 个已通过独立 A100 GPU 小样本验证的模型；未通过的 OneAlign、FG-CLIP2、PickScore 不进入本批。原工程已跑通 65 个模型。原有 31 个入口保留，另 34 个已迁入模型自己的原生源码、配置与路径适配。代码整理完成不代表 GPU 验证全部通过；逐模型真实状态见 [交付表](environment/delivery62_status.csv)。其余 12 个不阻塞首批整理。

```text
zhoujianjie/
├── environment/  # 依赖、逐模型映射、交付验证与协议问题
└── models/组织__模型/
    ├── inference.py
    ├── inference/  # 原生实现、资源映射、配置、许可证与来源摘要
    ├── weights/README.md
    ├── result/.gitkeep
    ├── model_config.json
    └── README.md
```

默认入口沿用原生全量流程；--smoke 才使用固定前缀真实样本诊断，不替代正式精度评测。外部权重和数据由命令行指定，不依赖旧工程源码。本批有 11 个原生检索模型无法用公共单标签分类协议表达，详见 environment/PUBLIC_PROTOCOL_ISSUES.md。

未上传权重、数据、虚拟环境、缓存、密钥和大型预测。公共 classification/README.md、evaluate.py 及其他贡献者文件不作修改。本轮没有 Git push，也没有重跑已完成的全量数据集。
