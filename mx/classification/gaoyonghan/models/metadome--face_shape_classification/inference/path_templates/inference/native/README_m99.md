# 99号：作者公开数据与原生单图推理

```bash
cd __MODEL_ROOT__/resources/runtime/assets/workspace/classification8_eval
DEVICE=0 bash run_m99.sh
```

仅使用已有权重及Face-Age-10K全部9165张，不下载、不训练、不安装依赖。复用已存在的`.venv-m83-optimum`，不使用ONNX后端，实际模型仍是原生Transformers SiglipForImageClassification。环境版本不符立即停止，不修改共享环境。`--preflight`仅CPU严格加载和图片/预处理核验，无模型前向；`MAX_IMAGES=10`只做覆盖8类的冒烟；默认冒烟后完整跑9165张，正式全量重新包含这10张，不重复计分。

结果独立存到`__MODEL_ROOT__/resources/runtime/assets/workspace/results/m99_author_source_v1/full/run_时间/`，日志在根目录logs，每次另建目录，保留历史结果。完整输出含逐图原始logits、原函数展示概率、混淆矩阵、各类别precision/recall/F1、Accuracy、macro-F1、weighted-F1。失败保留日志，不跳过断图，不自动从部分结果续算。

## 原生与补充部分

- 原生部分：从已冻结SHA256的作者模型卡直接提取并执行未修改的`classify_image`和`id2label`。复用本地原权重及AutoImageProcessor，RGB转换、单图前向、softmax、三位小数展示均保持原函数。跳过Gradio界面；仅通过外层包装把processor输出移动到GPU。FP32、batch1，不用AMP/TF32。
- 补充部分：Parquet读图循环、校验和记录、GPU调度、scikit-learn计分。作者仅公布分类报告与单图演示，未找到独立批量评测/计分脚本，因此不声称整个评测是未经修改的官方脚本。
- 指标决策：用未舍入logits argmax计分；原函数展示概率舍入至3位，可能产生并列，另记舍入前后top1差异。不根据结果调参或修改标签。
- 以发布权重config及processor的224×224为准。模型卡文字声称的512不用于覆盖实际发布配置。

## 数据与比较边界

数据为作者公开的train split，不另造test。8类计数为2474/1181/1523/1010/1181/799/653/344，与模型卡逐类一致；但没有独立测试清单或训练划分证据。输出必须标注`author_public_source_not_verified_holdout`，不能解释为独立未见测试成绩。

模型卡参考Accuracy82.25%、macro-F1=0.7971、weighted-F1=0.8223；本地官方checkpoint-1722日志第6轮accuracy=0.8224768139661757，与模型卡四位小数吻合。二者是参考线索，不证明评估图像逐一相同或无训练泄漏。

交付检查为CPU预检及合成输入接口测试；实际GPU全量由用户运行。
