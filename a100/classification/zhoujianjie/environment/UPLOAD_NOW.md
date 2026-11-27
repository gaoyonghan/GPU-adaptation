# 首批上传说明

本批仅含 62 个独立 GPU 小样本验证通过的模型代码；不是 62 个全量重新评测结果，也不代表全部完成公共分类协议验收。

51 个分类模型可生成公共分类评测输出；11 个保留原始检索流程，公共分类评测器不支持其指标，不能冒充 Top-1。详细状态见 delivery62_status.csv。

排除：q-future/one-align、qihoo360/fg-clip2-base、yuvalkirstain/PickScore_v1，以及原工程剩余 12 个未跑通模型。目录保留在本地但已由个人 .gitignore 排除。

只提交 a100/classification/zhoujianjie/。不上传权重、数据集、虚拟环境、缓存、历史大型预测或密钥；不修改公共 evaluate.py、公共 README 或其他贡献者代码。
