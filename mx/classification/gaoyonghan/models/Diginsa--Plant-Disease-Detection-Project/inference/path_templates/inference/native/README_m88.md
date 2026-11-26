# 88号 Keras原生对照

## 当前采用结果（用户确认，2026-09-15）

用户确认以本次Keras原生对照作为88号后续汇总和汇报的主要结果，旧HF结果保留作为对照，不覆盖旧指标、模型、脚本或运行快照。

| 项目 | 采用值 |
|---|---|
| 数据 | 官方ImageNet-1K验证集，50,000张，非替代集 |
| Top-1 | 81.472%（40,736/50,000） |
| Top-5 | 95.646%（47,823/50,000） |
| Keras参考 | Top-1 81.6%、Top-5 95.7% |
| 差值 | 分别−0.128、−0.054个百分点 |
| 采用状态 | 用户确认的主要结果；不是已验证的官方精确复现 |

结果目录：`__MODEL_ROOT__/resources/runtime/assets/workspace/results/m88_keras_native_v1/full/run_20260915_191146_394309360`。
预测文件SHA256：`808341c55352a2c094bba0580f6095196903ced15b36a48419badcd4af0dae09`。
已核查50,000条预测ID唯一且与原验证集一致、标签一致、每类50张、Top-1/Top-5正确计数及保存指标一致。
旧HF结果仍为Top-1 80.436%、Top-5 95.346%，路径见下文。当前采纳决定单独记录，不回写历史结果中的 `official_reproduction_verified=false`，也不改变旧批量清单指纹。没有因用户采纳而重新运行推理。

## 运行

```bash
cd __MODEL_ROOT__/resources/runtime/assets/workspace/classification8_eval
DEVICE=0 bash run_m88_keras.sh
```

复用现有 `/home/ubuntu/gaoyonghan/miniconda3/envs/object-detection-tensorflow-v2`：TensorFlow 2.15.1、Keras 2.15.0、NumPy 1.26.4、h5py 3.16.0。不安装或修改共享环境。脚本配置其现有CUDA动态库目录；GPU不可用则停止。

首次下载约50MB的Keras ImageNet B3 H5权重、35KB类别索引和10KB Google原评测预处理脚本；保存到生产项目 `classification8_eval/.runtime/m88_keras`，均校验哈希。Google源码URL随master更新时哈希不匹配将停止，不能静默用新版。Google/GitHub不可达时可能需用户现有合规代理，不自动改系统网络。

- 只检查资产、完整图像校验、GPU及模型构建（不推理）：`bash run_m88_keras.sh --preflight`。
- 仅10张：`MAX_IMAGES=10 DEVICE=0 bash run_m88_keras.sh`，不生成正式metrics.json。
- 默认先推理10张，成功后继续剩余49,990张，前10张保留计入全量，不重复推理。
- 输出：`__MODEL_ROOT__/resources/runtime/assets/workspace/results/m88_keras_native_v1/full/run_*`；包含协议、脚本快照、预测、状态和全量指标。
- 每次运行新建目录，失败后重新运行会从头开始，不覆盖、删除或修改旧HF结果。没有后台通知；终端打印进度与结果。

## 哪些是原生，哪些是项目代码

| 环节 | 采用内容 |
|---|---|
| 权重/模型 | 未改写Keras EfficientNetB3，`weights='imagenet'`，保留1000类头；H5 MD5 `ffd1fdc53d0ce67064dc6a9c7960ede0` |
| 预处理 | 未修改Google TPU EfficientNet `preprocess_for_eval`，300像素，默认双三次；当前冻结源码CROP_FRACTION=0.875，300输入计算crop_padding=43 |
| 归一化 | 由Keras模型内部Rescaling/Normalization完成；输入原像素尺度，不外部再次除255或套HF处理器 |
| 计分 | Keras SparseCategoricalAccuracy、SparseTopKCategoricalAccuracy(k=5)，FP32推理与指标累加，保留TensorFlow Top-K边界并列语义；交叉核对正确计数并以正确数/总数输出百分比，同时保存原生指标浮点值 |
| 数据 | 原ImageNet-1K val 50,000张；按官方Keras synset索引对齐本地LOC_val_solution.csv，原ZIP逐图大小/CRC验证，额外记录图像SHA256；每类50张 |
| 项目代码 | 资产下载校验、原数据CSV/文件读取、批量调度、进度/结果归档及计数交叉检查 |

这是“原生组件组成的评测适配流程”，不是声称直接运行一份Keras作者发布的完整ImageNet benchmark脚本。没有借用别的timm权重替代本模型，也没有训练。

## 与HF版关系及已知边界

HF v4.29.2官方转换脚本明确从Keras EfficientNetB3(weights=imagenet)转换；该脚本的单图等价测试使用PIL最近邻整图缩放。现有HF发布的处理器也是不中心裁剪，resample=0。原HF有效结果保持：Top-1 80.436%、Top-5 95.346%，位于 `results/classification8_v1/m88-efficientnet-b3`。

本次采用Google原评测的中心裁剪/双三次路径，不沿用HF转换用的单图缩放。Keras源码在weights=imagenet时另加Rescaling层，因此不得改成weights=None后自行load_weights，避免遗漏原生归一化层。

参考Keras官方表：Top-1 81.6%、Top-5 95.7%。仍存在：转换文档引用Keras 2.11而本地为2.15；未逐张量独立核实HF与本次H5相同；Keras历史表的全部评测环境/裁剪细节未完整公开核实。因此运行完成不自动设为“论文精确复现”，也不能把所有差异单独归因于裁剪。协议保留上述限制，结果产生后再审计。

## 来源

- HF模型卡：https://huggingface.co/google/efficientnet-b3
- 官方转换代码：https://github.com/huggingface/transformers/blob/v4.29.2/src/transformers/models/efficientnet/convert_efficientnet_to_pytorch.py
- Keras模型/权重定义：https://github.com/keras-team/keras/blob/v2.11.0/keras/applications/efficientnet.py
- Keras参考成绩：https://keras.io/api/applications/
- Google原评测入口：https://github.com/tensorflow/tpu/blob/master/models/official/efficientnet/eval_ckpt_main.py
- Google预处理：https://github.com/tensorflow/tpu/blob/master/models/official/efficientnet/preprocessing.py

## 交付验证

仅做不运行模型推理的检查：Bash/Python语法、共享环境版本、标签CSV异常拒绝及原生指标合成样例。自检发现TensorFlow的in_top_k不接受float64预测，已采用原生默认FP32指标输入，未修改计分定义。未在Codex内下载原生权重、安装环境或启动全量。实际GPU推理与最终指标以用户终端运行为准。
