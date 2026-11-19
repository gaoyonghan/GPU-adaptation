# 92号：五类脸型分类

2026-09-16用户确认暂按本轮原生结果保留：`m92_huggingpics_native_v1/full/run_20260916_115335_886950`，1000张正确523张，accuracy 52.30%；旧pipeline 52.40%保留对照，只有1张预测不同。不将0.10个百分点变化归因于单一因素，也不与模型卡85.31%直接比较。

## 最新入口：HuggingPics原生验证

用户批准改用模型关联的原生验证流程，运行：

```bash
cd __MODEL_ROOT__/resources/runtime/assets/workspace/classification8_eval
DEVICE=0 bash run_m92_native.sh
```

新结果位于`__MODEL_ROOT__/resources/runtime/assets/workspace/results/m92_huggingpics_native_v1/full/run_时间/`，不覆盖旧52.40%结果。首次仅下载小型依赖及约22KB笔记本，不下载模型或图片。`.venv-m92-huggingpics`继承现有torch/CUDA，隔离安装Lightning 2.1.2、TorchMetrics 1.2.0和lightning-utilities 0.10.0；不修改共享环境。这是已固定的兼容组合，不声称确认了作者历史完整环境。

原生入口不依赖scikit-learn，计分由TorchMetrics执行；旧pipeline清单中的scikit-learn版本仅属于历史对照。2026-09-16修复了复用旧预检时误要求原生环境安装scikit-learn的问题，未改变推理或计分逻辑。

固定HuggingPics提交`8b39429eaf39750d913bcc5368be395bb4d41b12`及笔记本SHA256，原样提取`ImageClassificationCollator`与`Classifier`两个类，包括`validation_step`。主结果直接由Lightning `Trainer.validate`执行和汇总原生TorchMetrics `Accuracy`产生；逐图结果只用于独立复核。绝不执行安装、训练、随机划分、删除文件、账户或上传等笔记本单元。

采用原笔记本batch8和FP16混合精度（`16-mixed`），seed42；用本地发布权重/处理器替换预训练初始化，保留相同1000张测试图。ZIP读取模拟原ImageFolder的RGB转换，不使用旧pipeline的EXIF转正。num_workers改为0以避免ZIP跨进程问题；其余差异见`models_m92_native.json`。因此新旧对照不仅是计分库差异，还包括精度和图像读取行为，不将变化单独归因于计分。

默认先10张原生冒烟，通过后在全1000张上重新验证（首10张会重复计算，以保持原生batch8布局；不将两次样本混合计分）。`MAX_IMAGES=10`只冒烟，不发布完整指标；`--preflight`检查源码、依赖、权重文件、数据及旧清单，不做模型前向；`--describe`打印协议。

该流程属于**模型关联工具原生验证＋公开替代集**。原作者当年的代码/环境版本和85.31%对应验证图仍未确认，继续标记`reference_baseline`、`officially_comparable=false`。旧流程和下述核查记录保留，不能把本次补查到的工具验证类误写成数据集官方evaluator。

交付检查：已在/tmp隔离依赖下用10张合成图片、虚构模型跑通原生collator、Classifier、Lightning验证与TorchMetrics，确认含尾批的3/10结果为30%，预测hook和独立复算一致；未加载实际92号权重或进行GPU推理。

## 2026-09-16评估来源复核

按用户“使用哪个数据集，就用该数据集评估和计算流程”的要求，核查了[发布者数据页](https://www.kaggle.com/datasets/niten19/face-shape-dataset/data)及本地下载包。数据页说明五类、每类800张训练/200张测试；未找到发布者指定的完整评估脚本或指标计算规范。ZIP含5000张图片及3个desktop.ini，不含评估代码。检索到的第三方Notebook/仓库不能冒充数据集官方实现；这里记录“未找到”，不推断作者绝对未曾发布。

已完成运行`__MODEL_ROOT__/resources/runtime/assets/workspace/results/m92_faceshape_v1/full/run_20260916_113236_197929`：1000张中正确524张，accuracy 52.40%，macro F1 52.9330%。预测哈希、清单、计分复算已通过。现有流程的测试划分来自数据集；accuracy/F1由项目调用scikit-learn计算，并非发布者提供的evaluator。旧结果及冻结快照不修改，保留为参考基线；是否接受这种计分作为最终交付口径，须按新要求向用户确认，不能标记为数据集官方流程已对齐。

## 一键运行

```bash
cd __MODEL_ROOT__/resources/runtime/assets/workspace/classification8_eval
DEVICE=0 bash run_m92.sh
```

复用既有classification8环境，不安装依赖、不联网、不解压或复制数据集。默认完整testing_set共1000张，前10张先验证加载、输出与流水线，再复用预测继续剩余990张。不训练、不随机重新划分、不把training_set混入指标。

`bash run_m92.sh --preflight`只做CPU数据/处理器/权重文件预检，不运行模型前向；`--describe`显示协议。`MAX_IMAGES=10 DEVICE=0 bash run_m92.sh`只做冒烟，不产生正式accuracy。

输出：`__MODEL_ROOT__/resources/runtime/assets/workspace/results/m92_faceshape_v1/full/run_时间/`。包含冻结协议和指纹、脚本与模型配置快照、原生组件源码哈希、逐图来源/标签/SHA256/预测、状态、指标与核查摘要。日志在根目录logs。每次新建目录，不覆盖旧结果；失败保留预测，但不自动跨运行续写。

## 数据与类别

发布者：`metadome/face_shape_classification`，revision `f4d28b03f39ff77f1feb5bb8094e32a23c730cff`。直接使用发布的ViT safetensors，不转换、不换权重。

数据：[Kaggle Face Shape Dataset](https://www.kaggle.com/datasets/niten19/face-shape-dataset)，本地ZIP 670249355字节；文件SHA256和其他细节见models_m92.json。仅用`FaceShape Dataset/testing_set`，Heart/Oblong/Oval/Round/Square各200张，映射到模型ID 0/1/2/3/4。

2026-09-16交付核验：5000张图片均通过ZIP读取与CRC；1000张测试图全部可解码。测试集内部、训练/测试之间没有字节完全相同的图片；训练集内部有4组完全重复图。训练集Heart/heart (633).jpg和Square/square (84).jpg解码截断，不参与评测，因此不影响这1000张测试图。未做人物身份或近似图去重；作者模型训练图是否与该测试集重叠未知。

五类标签完全覆盖，但均为女性名人照片，不代表其他人群。原模型测试集及划分未公布/未确认，因此标记`public_replacement`，不是已证实的原测评集。

## 原生范围与指标

- 原样使用发布者ViT模型与ViTImageProcessor：224x224双线性Resize、1/255、mean/std均为0.5。原生pipeline负责EXIF转正与RGB，不额外裁脸、检测、对齐或增强。
- 推理批处理直接调用Transformers `ImageClassificationPipeline`；项目代码仅提供ZIP Dataset、校验、消费返回结果及归档，没有重写模型前向/预处理/softmax。
- FP32、batch8、eager attention、禁用TF32，无AMP。共享Transformers 4.50.0；模型配置记录4.35.2，明确存在软件版本差异。
- 主指标用scikit-learn `accuracy_score`（百分比）；补充每类precision/recall/F1、macro F1和混淆矩阵。五分类不报告无区分能力的Top-5。标准计分不是作者专属原生evaluator，标记`reference_baseline`。
- [模型卡](https://huggingface.co/metadome/face_shape_classification)给出accuracy=0.8530884981155396（85.30885%），未说明对应样本与精确评测协议。因此仅展示参考值，不计算掉点、差值或宣称官方复现。

交付时仅运行静态检查与CPU预检，GPU推理由用户终端执行。
