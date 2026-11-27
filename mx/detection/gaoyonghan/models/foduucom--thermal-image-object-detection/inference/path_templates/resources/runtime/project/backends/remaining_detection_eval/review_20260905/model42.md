# 42号 cmarkea/detr-layout-detection 复核

核查日期：2026-09-05。结论：此前沿用了官方DocLayNet数据源和模型推理接口，但没有沿用作者的500页清单及完整评估程序。我们选择全test4999页、COCO AP及前景macro F1作为参考指标；用这些数值直接对照作者GIoU和未注明汇总方式的F1，口径不成立。作者公开讨论实际上给出了框匹配规则，此前漏查了该说明。

## 作者到底如何评分

[模型卡](https://huggingface.co/cmarkea/detr-layout-detection)报告500页的像素F1 91.27%、框GIoU 80.66%、类别accuracy 90.46%。模型仓库当前固定revision `b30837562d210bde4a3a0bb246133f4fdeb29fbe`，仅README、config、preprocessor、权重和.gitattributes，未包含500页名单或评分脚本。

作者在[官方讨论 #1](https://huggingface.co/cmarkea/detr-layout-detection/discussions/1)于2024-10-02至03日解释：COCO mAP约0.5可以与其结果相容；其方法对每个真实框寻找GIoU最大的预测框，再度量框位置及类别。这不使用COCO AP的置信度排序和一对一匹配，也不会同样惩罚额外重叠框。讨论没有补充完整代码、确切阈值、500页名单、跨图/跨类汇总及分割冲突处理。

原报告未沿用官方脚本的原因不是数据集没有公开：官方DocLayNet公开，但完整作者评估程序未取得。已披露的匹配规则可以复算，不能继续只写“指标未知”。模型卡推理示例的0.4阈值不能自动当成发布分数时的阈值。

## 使用已有预测的CPU补充评分

原结果目录：`__MODEL_ROOT__/resources/runtime/assets/workspace/results/remaining25_cmarkea_v1`。原始GT为DocLayNet `COCO/test.json`，4999页、66531框、11个前景类。预测逐页保留100个框、RLE掩码及12类像素TP/FP/FN。本次不加载模型，不重跑GPU，不覆盖原metrics/status。

分割聚合结果：

| 指标 | 本轮4999页 | 模型卡 |
|---|---:|---:|
| 前景macro F1（原报告） | 78.0394% | 未确认采用此口径 |
| 前景micro F1 | 86.9069% | 未确认采用此口径 |
| 含背景micro F1 | 91.2744% | F1 91.27%，汇总口径未披露 |
| 含背景、按GT像素支持度加权F1 | 91.3865% | 未确认采用此口径 |

micro F1 = 2×ΣTP/(2×ΣTP+ΣFP+ΣFN)；macro为逐类F1算术平均。含背景micro结果四舍五入恰为91.27，说明此前78.04与91.27的差异可能主要来自汇总口径；**不能仅因相等就确认作者也采用此口径。** 当前掩码允许不同前景类独立重叠，背景取所有前景的补集，作者是否同样处理仍未知。

检测按作者披露的规则补算：每个GT独立选择所有候选预测中GIoU最大者，不按类别筛选候选；用GT数量加权平均GIoU，并计算所选预测类别正确率。两个阈值均有既定来源且完整报告，没有尝试搜索最接近模型卡的阈值。

| 评分输入 | 候选预测数 | 平均最佳GIoU | 最佳GIoU框类别准确率 |
|---|---:|---:|---:|
| 原已冻结框协议：score > 0 | 499900 | 82.4693% | 88.4340% |
| 仅诊断：模型卡使用示例score > 0.4 | 93646 | 78.2356% | 89.6755% |
| 作者报告，500页完整协议未知 | — | 80.66% | 90.46% |

阈值0.4下有6页无预测；本次约定这些GT的GIoU为0、类别记错，作者处理方式未知。GT及预测均用原像素坐标，不额外裁边或NMS。原COCO AP为48.4086%，保留为另一项参考指标，不能拿它与80.66%直接作差。

可复算脚本：[model42_rescore.py](__MODEL_ROOT__/resources/runtime/assets/workspace/remaining_detection_eval/review_20260905/model42_rescore.py)。机器记录及输入、脚本哈希：[model42_supplemental_metrics.json](__MODEL_ROOT__/resources/runtime/assets/workspace/remaining_detection_eval/review_20260905/model42_supplemental_metrics.json)。

已验证：预测ID集合恰为全4999页且无重复；逐页累计像素计数等于原metrics；原macro F1复算误差小于1e-12；GIoU实现对相同、包含、分离及负坐标框与torchvision官方实现一致。结果不构成作者500页实验的完整复现。

## 记录及下一步

42号保持待用户审阅；任务覆盖已有框检测、分类和分割。后续国产GPU比较可以冻结4999页及本次明确的评分规则，同时保留COCO AP。优先追溯作者500页名单及汇总代码；未取得时明确使用本轮自有固定协议，而不是用接近的数字认定已恢复作者协议。
