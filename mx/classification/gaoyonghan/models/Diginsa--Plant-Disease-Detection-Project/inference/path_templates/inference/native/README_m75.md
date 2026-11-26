# 75号：原权重 + FER2013原始测试协议

## 协议与边界

- 模型：`wasmdashai/vit-face-expression-v1`本地原始safetensors，不更换上游权重。
- 数据：原始`fer2013.csv`，含`emotion,pixels,Usage`。Training 28709、PublicTest 3589、PrivateTest 3589；仅PrivateTest参与正式评测。不重新划分，不使用训练集，不将PublicTest与PrivateTest合并，不使用FERPlus标签。
- 任务：48×48灰度人脸的七分类。FER原编号0 angry、1 disgust、2 fear、3 happy、4 sad、5 surprise、6 neutral。模型编号后3类不同，按名字映射，既不合并也不删除类别。灰度复制为RGB后直接调用发布的预处理器：224×224双线性resize、1/255、mean/std=0.5。无额外人脸检测、裁剪或测试增强。
- 主指标：PrivateTest Top-1 Accuracy，正确数÷全部测试图片数；额外保存混淆矩阵便于核查。单模型FP32、batch16、argmax；并非FERPlus八类/投票/多标签评测。
- [原始比赛报告第3节、表1](https://arxiv.org/html/1307.0414)明确报告Private test accuracy与原始七类数量。[FERPlus作者说明](https://github.com/microsoft/FERPlus)也明确原CSV的Training/PublicTest/PrivateTest用途，但本实验不采用FERPlus重标注。
- 没有取得Kaggle服务器原始计分源码；本脚本调用`sklearn.metrics.accuracy_score`实现同一定义，并以正确数/总数交叉验证。CSV读取、批量调度、标签编号映射和记录为项目自写；模型及预处理直接使用发布者的Transformers组件。不能标注为整套官方脚本原封不动复现。
- [75号模型卡](https://huggingface.co/wasmdashai/vit-face-expression-v1)未披露测试集和指标。本地config的`_name_or_path`指向[trpakov/vit-face-expression](https://huggingface.co/trpakov/vit-face-expression)，后者报告FER2013测试Accuracy 71.16%。这仅是来源线索，未核对权重等价性，也未确认75号训练数据与测试图像是否重合；71.16%不能填作75号已确认的官方参考值。
- 数据集原始协议不等于该权重作者原评测协议：本次标记`public_replacement_author_test_unverified`。

## 数据准备

2026-09-15更新：原比赛下载403、网页未显示可完成的加入入口；用户已下载并解压[deadskull7/fer2013公开重发布副本](https://www.kaggle.com/datasets/deadskull7/fer2013)，文件位于`__DATASET_ROOT__/fer2013/deadskull7/fer2013.csv`。SHA256为`3b8d9617d1017f34733c8f2474d7784c563ce86c40a86ac12c2d37cc968f871b`，三列、全部划分数量、像素及七类检查已通过。PrivateTest各类样本数为491、55、528、879、594、416、626（按FER原编号）。尚未与原比赛压缩包逐字节比对，不标为原比赛直接下载；重发布页面的许可标签也不视为原数据授权的独立证明。

本次运行需设置`FER_CSV=__DATASET_ROOT__/fer2013/deadskull7/fer2013.csv`。下方保留原比赛下载方法供追溯；若平台要求登录或接受条款，请本人处理，脚本不读取或保存账号密钥。

若已安装并配置Kaggle CLI，可在终端执行：

```bash
mkdir -p __DATASET_ROOT__/fer2013
kaggle competitions download -c challenges-in-representation-learning-facial-expression-recognition-challenge -p __DATASET_ROOT__/fer2013
```

平台当前下载可用性及最终压缩包结构尚未验收。解压后使用其中的`fer2013.csv`，默认放在`__DATASET_ROOT__/fer2013/fer2013.csv`，或通过`FER_CSV`指定其实际位置。只含train/test且没有原始Usage的重打包数据、FERPlus CSV、没有标签的比赛test.csv均不可代替。不要为了通过检查手工伪造Usage或标签。

## 运行

只读复用已验证的`__MODEL_ROOT__/resources/runtime/assets/workspace/classification8_eval/.venv`，不安装、升级共享环境。torch2.5.1+cu121、transformers4.50.0、numpy1.26.4、Pillow11.1.0、scikit-learn1.5.2、safetensors0.5.3；模型config记载4.41.2，框架差异明确保留。

```bash
cd __MODEL_ROOT__/resources/runtime/assets/workspace/classification8_eval
# CPU数据与预处理检查，不运行模型推理
bash run_m75_fer2013.sh --preflight
# 10图检查通过后自动评测PrivateTest全量3589张
DEVICE=0 bash run_m75_fer2013.sh
```

文件在其他位置：`FER_CSV=/实际路径/fer2013.csv DEVICE=0 bash run_m75_fer2013.sh`。仅冒烟：`MAX_IMAGES=10 bash run_m75_fer2013.sh`。

输出在`__MODEL_ROOT__/resources/runtime/assets/workspace/results/m75_fer2013_v1`，smoke/full分开，每次时间戳新目录，保留旧结果，不自动覆盖已确认基准。只有3589张完整完成才生成正式`metrics.json`。无断点续跑；失败后重新运行产生新目录。

每轮保留protocol.json（数据/权重/代码SHA256、依赖版本及协议边界）、status.json、必要脚本快照、逐图predictions.csv和指标。每条预测保留原数据集、原划分、CSV数据行编号、像素SHA256和类别映射，可追溯来源，不生成多份解码图片。哈希用于追溯，并非声称已经比对数据发布者官方校验值。

交付检查：语法、类别映射、像素校验、计分、预处理及CPU权重加载检查通过。用户已完成真实CSV的preflight，3589张PrivateTest覆盖七类；尚无GPU推理结果。
