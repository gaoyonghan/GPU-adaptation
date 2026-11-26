# 81号：FER2013 临时评测

## 当前采用结果（2026-09-15）

用户确认采用本次结果，不再重跑：FER2013 PrivateTest，3,589张，正确2,354张，Top-1 Accuracy **65.59%**。
采用运行：`__MODEL_ROOT__/resources/runtime/assets/workspace/results/m81_fer2013_v1/full/run_20260915_160723_995381`。
汇报标记为“用户采用，附带未核实项”：替代评测协议、类别映射未确认、训练测试重叠未知、不可直接对比作者混合集指标。
原始输出编号0在全量中从未成为最高分，原因仍未确定；不得将其改写为已修复或已确认模型缺陷。
原运行目录的 `provisional_complete`、`accepted_for_report=false` 等字段保持运行时原样；
后续采用决定见 `models_m81.json` 的 `selected_result`，不将用户采用等同于技术核实通过。

数据已具备；类别编号还没有作者明确确认。本包默认不启动推理，必须显式接受未核实映射才试跑。

## 使用

复用75号数据和classification8_eval环境，不下载、不改共享依赖。预检只读文件、校验哈希及预处理，不做模型推理：

```bash
cd __MODEL_ROOT__/resources/runtime/assets/workspace/classification8_eval
bash run_m81_fer2013.sh --preflight
bash run_m81_fer2013.sh --describe
```

接受候选类别顺序、仅供后续核查时，可执行：

```bash
cd __MODEL_ROOT__/resources/runtime/assets/workspace/classification8_eval
ALLOW_UNVERIFIED_LABELS=1 DEVICE=0 bash run_m81_fer2013.sh
```

先10张加载测试，成功后全量3,589张。只跑10张加 `MAX_IMAGES=10`。小样本不报告准确率。
每次新建时间戳结果目录，不覆盖之前结果，不修改75号或正在下载的76号。
输出根目录：`__MODEL_ROOT__/resources/runtime/assets/workspace/results/m81_fer2013_v1`。
全量成功状态为 `provisional_complete`，不是正式验收通过。
指标只写入 `provisional_metrics.json`，`accepted_for_report=false`；不得计入已确认模型数量。

## 数据与计分

使用已有deadskull7公开副本的原始FER2013 CSV，固定SHA256，读取Usage=PrivateTest的3,589张，覆盖全部七类。
不重划分、不混入Training/PublicTest、不拼接其他数据。输入48x48灰度像素转RGB，调用模型自带预处理。
作者使用FER2013、RAF-DB、AffectNet的混合集，未明确每张图的归属；本次属于替代评测协议，不能声称未见测试集或复现作者指标。

主指标沿用[FER2013挑战赛](https://arxiv.org/html/1307.0414)的分类准确率：正确张数/总张数。
调用sklearn accuracy_score，并用正确数交叉核查。没有取得Kaggle服务器计分源码，不将本脚本称为原生服务器评测脚本。
复用 `run_m75_fer2013.py` 的CSV校验和计分函数，不改动75号脚本。

## 类别映射缺口

[作者模型卡](https://huggingface.co/Tanneru/Facial-Emotion-Detection-FER-RAFDB-AffectNet-BEIT-Large/blob/68cbe4de6a93da0ec70267ab792b539311ad3d3f/README.md)
的类列表、分类报告和混淆矩阵都按anger/disgust/fear/happy/neutral/sad/surprise排列。
但config仅有LABEL_0至LABEL_6，公开仓库未找到训练标签编码器，因此不能证明这就是输出索引顺序。
候选模型编号到FER编号固定为 `[0,1,2,3,6,4,5]`，其中anger与angry仅为同义名称。
不得根据PrivateTest准确率寻找最佳标签排列，否则会产生测试集泄漏。
保存逐图原始logits和模型输出编号，日后取得明确映射证据可重新计分，无需重复GPU推理。

## 原生组件与差异记录

直接使用作者示例的BeitImageProcessor和BeitForImageClassification，自写部分仅为CSV读取、批量循环、计分及归档。
RGB；224x224双线性缩放；1/255；mean/std均0.5；无中心裁剪、外加人脸检测或增强。
FP32、eval模式、eager attention、batch8、argmax。作者训练参数提到eval batch32及fp16训练，但没有明确推理dtype；本次选择FP32。
共享Transformers为4.50.0，保存配置为4.54.1；CPU加载无缺失、多余或形状不匹配参数，数值等价未验证。不升级共享环境以免影响其他任务。

作者参考数字存在冲突：model-index 73.57%、正文约76.2%、所附22,494张混淆矩阵算得76.6382%。不选定单一参考值，不计算本次相对掉点。
`models_m81.json`保存固定revision、模型和数据哈希、全部已知差异；每次结果同时归档脚本、配置、作者报告及协议指纹。原权重不复制，只记录原路径和SHA256。

交付验证：共享环境CPU严格加载、预处理、数据和映射/计分测试；没有在Codex内进行GPU推理，完整运行由用户终端验收。
