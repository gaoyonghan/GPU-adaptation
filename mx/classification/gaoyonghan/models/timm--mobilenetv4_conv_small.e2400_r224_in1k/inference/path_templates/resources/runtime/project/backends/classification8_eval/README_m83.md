# 83号：ImageIN作者数据来源全量评测

用户已确认使用全部1,896页。不是替代集，也不是已确认的独立测试集。训练/测试重叠未知；不直接对比作者95.09%或F1 0.92。

## 一键运行

```bash
cd __MODEL_ROOT__/resources/runtime/assets/workspace/classification8_eval
bash run_m83_imagein.sh
```

首次自动补环境，之后离线读取本地权重和图片，先检查前10张能否推理，再继续余下1,886张，前10张不重复计算。
默认CPU、单图推理，沿用作者量化ONNX版本；不是GPU性能/国产GPU适配测试。
完整指标为Accuracy和插图类二分类F1，不是检测AP。

可选：`MAX_IMAGES=10 bash run_m83_imagein.sh`仅冒烟测试，不汇报小样本指标。
`bash run_m83_imagein.sh --preflight`校验全部文件哈希、1,896张图片解码/预处理及模型加载，不执行模型推理。
`bash run_m83_imagein.sh --describe`打印协议，不安装环境。

输出：`__MODEL_ROOT__/resources/runtime/assets/workspace/results/m83_imagein_source_v1`。
每次新建时间戳目录，保留protocol.json、status.json、逐图predictions.jsonl、全量metrics.json及脚本配置快照。
只有 `source_evaluation_complete` 表示完整来源集评测完成；不表示官方独立测试成绩复现。
新运行不覆盖旧结果，失败不跳过图片，不自动改变模型或精度；重启会新建一次运行，不复用残缺结果。

## 权重与环境

直接调用[模型卡](https://huggingface.co/small-models-for-glam/historical-illustration-detector/blob/3aab2a6336291b732a1552c562fd15b1bd606213/README.md)中的
`ORTModelForImageClassification`和`AutoImageProcessor`，显式选择本地 `onnx/model.onnx`，禁止重新导出。
该文件与 `model_quantized.onnx` 哈希完全相同；不能根据文件名把它当成FP32原始权重。
ONNX Runtime执行模型已经导出的量化图，输入/输出为float32。

现有分类环境缺少Optimum/ONNX，因此建立小型隔离venv `.venv-m83-optimum`，只读复用基础环境中的PyTorch等依赖。
不会升级75/76/81号共享环境。Optimum 1.27.0的onnxruntime额外依赖要求Transformers<4.54，使用已验证预处理的4.50.0。
模型配置保存版本4.55.4；本次不调用其PyTorch MobileNet实现，而运行固定ONNX图。作者没给完整环境锁文件，差异如实记录。

## 数据和标签

[ImageIN/ImageIn_annotations](https://huggingface.co/datasets/ImageIN/ImageIn_annotations/tree/102d18f8eaf27f9c9cf8c66492cbafea6a5f0c40)四个Parquet已在本地，嵌入原图，无需再次下载或联网访问Internet Archive。
本地目录名含 `holdout_unpublished` 只是历史命名，不能作为测试集独立性的证据。
读取全部train划分：1,430张not-illustrated、466张illustrated；校验原始ID唯一，不重分割、不删除记录、不重新标注。
真实类别列是 `choice`；名为 `label` 的列是页码等文字，绝不能拿它作为分类标签。
预测编号0=not-illustrated，1=illustrated，与作者配置一致，不存在81号的编号歧义。
每张结果记录来源数据集、原split、原id、书籍manifest地址、图像URL、Parquet文件和行号、图片字节哈希。
数据许可字段未提供清晰统一许可；使用和再分发仍需遵守原书籍/Internet Archive来源条件，本脚本不重新发布数据。

按作者预处理：整页RGB，短边双线性缩放256，中心裁剪224，1/255及mean/std 0.5；不额外分块查找角落插图。
标签遵循原始choice：照片/地图/图解等算插图，装饰边框/首字母/表格等不算。没有基于模型预测修改标签。

## 代码边界和验收

作者提供的是推理示例，未找到与测试成绩完整对应的评估划分及evaluator。本包只补充数据循环、标准指标计算、校验和归档。
调用sklearn accuracy_score和f1_score(pos_label=1, average=binary)，另用混淆矩阵正确数及2TP/(2TP+FP+FN)交叉核查；不宣称拿到了作者的计分源码。
可信等级reference_baseline，数据关系original_protocol_incomplete；固定权重/数据版本、SHA256、软件版本及已知差异见models_m83.json。
全量完成后仍须核查实际预测数为1,896、类别覆盖、结果文件哈希和异常分布。

交付时已通过静态/计分测试、四个数据文件哈希、全部1,896张图片解码/预处理检查（图片字节哈希也有1,896个唯一值），以及ONNX图CPU加载检查。
没有运行模型推理，也未执行新环境安装。
首次Optimum环境安装与完整推理由用户终端验收。
