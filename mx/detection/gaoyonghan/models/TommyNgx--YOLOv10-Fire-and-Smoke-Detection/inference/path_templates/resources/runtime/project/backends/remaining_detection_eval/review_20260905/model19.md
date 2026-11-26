# 19号 Aryn/deformable-detr-DocLayNet 复核

**后续执行更新**：本报告为补跑前复核；现已完成新数据全量推理，最新结果见[model19/结果报告.md](../supplemental_20260905/model19/结果报告.md)。下文保留当时核查过程。

复核日期：2026-09-05。**本次只查文档、数据及代码，并用已有预测做CPU重评分；没有启动新的GPU推理，没有修改已有结果或共享环境。**

## 结论

本轮普通 DocLayNet val/test 没有发现划分混用或评分虚高：原始 test JSON 配合既有预测独立重算，AP仍为 **74.420891%**。但这不能直接与模型卡57.1作差。找到了先前漏查的两项具体差异：

1. **作者论文使用另一套比赛基准。** Aryn自己的论文《The Design of an LLM-powered Unstructured Analytics System》§4明确链接这个HF仓库，给出 DocLayNet **competition** benchmark mAP **60.2%**、mAR **74.3%**。普通DocLayNet验证/测试集并非该比赛集。[作者论文](https://arxiv.org/html/2409.00847v2#S4)
2. **置信度过滤能显著影响本模型AP。** 本轮使用阈值0；把完全相同的test预测仅按模型卡示例阈值0.7过滤，AP变为 **58.102575%**。这是诊断，不能据此推定作者57.1用了0.7，也不能选择接近的数字宣称对齐。[模型卡使用示例及57.1声明](https://huggingface.co/Aryn/deformable-detr-DocLayNet)

因此，原结果可保留为“普通DocLayNet固定基线”，19号的“作者结果对应关系”应标为**待补比赛集核验，57.1来源未确定**，不能再笼统认为只是不知道val还是test。

## 数据划分实证

核对对象是本轮实际读取的 `__DATASET_ROOT__/doclaynet_test/DocLayNet_core.zip`。目录名叫 `doclaynet_test`，但ZIP内含官方三个split，加载器按指定JSON选图，不是扫描目录随机分割。

| split | JSON成员 | 图像数 | 框数 | 独立文档数 | 既有逐图预测覆盖 |
|---|---|---:|---:|---:|---|
| train | COCO/train.json | 69375 | 941123 | 2587 | 本次未跑 |
| val | COCO/val.json | 6489 | 99816 | 160 | 6489条，图像ID集合完全相同 |
| test | COCO/test.json | 4999 | 66531 | 196 | 4999条，图像ID集合完全相同 |

以上图像规模与[官方DocLayNet仓库](https://github.com/DS4SD/DocLayNet)完全一致。实际三组两两比较：**文件名交集0、文档名交集0、文档名+页码交集0**。ZIP中PNG无重复basename，每个JSON图像都有对应PNG，全部1025×1025。每个split内部图像ID和文件名均唯一，全部标注 `precedence=0`、`iscrowd=0`。没有因为去除难例、重复标注或crowd忽略造成的分数抬高。

注意：不同COCO split各自编号，数值image_id存在交集是正常的；它们不是同一图片。本轮两个split独立输出、独立评分，没有合并ID。

原始JSON SHA256：

- val：`eacd2ba2a32e6c2ebe2dc5ea1545f405d8461c46bba53e0c35d9c54fba0bcef6`
- test：`37ccce8c7951dcfd9c7c832bbff5aee71cb59c032ed7021e125e6188e6ac4fe1`

实际预测与导出的 `predictions.json` 完全一致。所有11个DocLayNet类别都参与评分；模型额外的 `N/A=0` 是未定义占位标签，没有从11类中删掉难类。本次审计只能排除**我们的读取/划分混用**，不能反推出作者训练该权重时实际使用了哪些样本，作者训练清单仍未公开核实。

## 权重和实际推理、评分协议

本地下载元数据记录HF revision：`d5503a90ae08dd43565de6984a5dd7924cad2400`。本地权重重新计算SHA256与下载记录、两次全量结果status中的值一致：`e3861d34685d3b36e5f38370597daf98fb2f98a852cfa5c125035057ded06809`。config、preprocessor两文件SHA也与运行记录一致。

实际链路为作者的DeformableDetrForObjectDetection及其AutoImageProcessor；RGB图、800短边/1333最长边、0均值/1标准差（按作者preprocessor配置，默认rescale到0–1），FP32，eval模式，原图尺寸恢复bbox。没有训练或依据GT后修正预测。读取和HF推理源码见冻结的 [core第339行](__MODEL_ROOT__/resources/runtime/assets/workspace/results/remaining25_v3/official_object_detection_eval/run_model.py:339)、[第704行](__MODEL_ROOT__/resources/runtime/assets/workspace/results/remaining25_v3/official_object_detection_eval/run_model.py:704)。

**发现一处协议元数据记录不准，分数计算本身未受影响：** 通用status写了 `conf=0.001`、`nms_iou=0.7`、`prediction_max_det=300`，但HF分支真实行为是：

- [冻结core第733行](__MODEL_ROOT__/resources/runtime/assets/workspace/results/remaining25_v3/official_object_detection_eval/run_model.py:733)：`post_process_object_detection(..., threshold=0.0)`。
- 同一HF分支没有NMS。
- [实际Transformers处理器第1571行](/home/ubuntu/gaoyonghan/miniconda3/envs/object-detection-hf-v2/lib/python3.11/site-packages/transformers/models/deformable_detr/image_processing_deformable_detr.py:1571) 默认 `top_k=100`，先对sigmoid分类概率按所有query×类别取top100；core随后截到300没有增加输出。因此每图最多100条，少数N/A结果被排除。
- 评分使用pycocotools 2.0.8原生bbox COCOeval，IoU .50:.05:.95，11类，maxDets=[1,10,100]，没有类别合并。[评分代码](__MODEL_ROOT__/resources/runtime/assets/workspace/remaining_detection_eval/run_model.py:35)

修复建议：保留原status及指纹；通过附加勘误记录真实 `score_threshold=0.0`、`nms=None`、`processor_top_k=100`、`postprocessor_output_cap=300`。未来新版本生成协议时应从HF实际执行参数填写，防止国产GPU端误用通用YOLO参数。

冻结core SHA为 `5c35f210f04d5ba2173edc0528b5869c0ab9137eecf08847d3c7a414af10ce3f`。当前共享core SHA已变化，但与冻结版仅差一个ONNX CUDA provider检查，与19号HF链路无关；本次没有改该文件。

## 独立CPU重评分

使用ZIP中未经适配过滤的原始 `COCO/test.json`、完整4999张和66531框，对既有499896条预测直接调用COCOeval；没有导入原评分辅助函数。

| 协议 | 留存预测数 | AP | AP50 | AR100 |
|---|---:|---:|---:|---:|
| 本轮真实阈值 >0 | 499896 | 74.420891% | 90.742149% | 84.004436% |
| 仅作诊断：模型卡使用示例 >0.7 | 50085 | 58.102575% | 69.056060% | 61.964351% |

第一行与原metrics的AP数值误差小于1e-12。第二行比第一行低16.318315个百分点，说明在评估数据相同的情况下，单是阈值不同就可能产生这个量级的差距。但模型卡示例是展示检测框的用法，并未声明57.1以该阈值评分。不能用这个诊断替代真实协议证据。

可复跑脚本：[model19_cpu_audit.py](__MODEL_ROOT__/resources/runtime/assets/workspace/remaining_detection_eval/review_20260905/model19_cpu_audit.py)。审计全量统计：[model19_cpu_audit.json](__MODEL_ROOT__/resources/runtime/assets/workspace/remaining_detection_eval/review_20260905/model19_cpu_audit.json)。本地文件版本：[model19_local_provenance.json](__MODEL_ROOT__/resources/runtime/assets/workspace/remaining_detection_eval/review_20260905/model19_local_provenance.json)。

## 论文与比赛集的区别

模型卡直接引用的DocLayNet数据论文是2022年的数据集论文，原始Deformable DETR论文则发表于2020年；两者都不是Aryn于2024年发布的这个微调checkpoint的实验报告。不能从这两篇基础论文寻找一个同名架构的数字便当作本权重指标。

新找到的Aryn作者论文给出competition benchmark 60.2/74.3，但没有绑定具体checkpoint hash、推理阈值或详细评估脚本。作者[2024年9月博客](https://www.aryn.ai/post/benchmarking-pdf-segmentation-and-parsing-models)也明确比赛集与原DocLayNet是不同文档、不同分布；该博客为跨产品比较还删去部分页面并合并标签，最终333页/5组标签。**不能把博客修改版333页、比赛原版498页、本轮普通test4999页当成同一协议，也不能自动认为论文60.2使用了博客修改版。**

[比赛官网](https://ds4sd.github.io/icdar23-doclaynet/task/)及[比赛论文§3](https://arxiv.org/pdf/2305.14962)说明：比赛原版为498页，包括reports293、manuals114、patents48、others43；其中有普通DocLayNet之外的布局。评分先对4种文档分别计算COCO AP，再对4个AP等权平均。该汇总方式也不同于本轮将全部图片混合计算一次COCO AP。

比赛数据与金标现有公开入口，不能再记录为“只缺无法定位的划分”。最小下一步是补498页比赛基准，原val/test结果保留：

- [比赛图像ZIP](https://ds4sd-icdar23-doclaynet-competition.s3.eu-de.cloud-object-storage.appdomain.cloud/competition-dataset-public.zip)
- [比赛GT ZIP](https://ds4sd-icdar23-doclaynet-competition.s3.eu-de.cloud-object-storage.appdomain.cloud/competition_dataset_groundtruth.zip)

本次仅确认了官网公开链接。GT请求已开始传输，但为避免在本轮研究核查中继续大下载，主动终止；未落盘ZIP，未核验包内内容。没有据此声称比赛数据已准备完毕。补跑时应核实498页、四种文档及11类，先固定阈值/processor版本、保留全部预测，按比赛的四种文档等权方式和整体COCO AP分别输出；不依据哪个结果更接近57.1选择参数。
