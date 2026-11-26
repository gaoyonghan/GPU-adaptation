# 17号相关第三方论文：DETR权重身份补核

2026-09-05，只核查论文与公开代码线索，没有GPU运行、没有改既有结果。

**不能回答“论文也没有指标”。找到的WACV2026第三方论文确实有DETR零样本及再微调两行结果；但没有足够证据把零样本行绑定到17号现有checkpoint。**

来源：[论文正式入口](https://openaccess.thecvf.com/content/WACV2026/html/Thomas_Zero-Shot_Table_Extraction_in_Business_Documents_A_Unified_Benchmark_with_WACV_2026_paper.html)、[论文PDF](https://openaccess.thecvf.com/content/WACV2026/papers/Thomas_Zero-Shot_Table_Extraction_in_Business_Documents_A_Unified_Benchmark_with_WACV_2026_paper.pdf)。论文是 Eliott Thomas 等的《Zero-Shot Table Extraction in Business Documents: A Unified Benchmark with Error Taxonomy and Ecological Analysis》，并非Taha Douaji发布17号权重时的原始实验论文。

## Table 2：两行均需保留

以下GIoU为均值±95%置信区间半宽，不是AP。

| 论文行 | DocILE-QUEST（141页）GIoU | STM154（154页）GIoU | 与17号关系 |
|---|---:|---:|---|
| DETR (41M)，零样本 | 0.20 ± 0.05 | 0.21 ± 0.06 | 未明确绑定HF ID或checkpoint |
| DETR† (41M)，有监督再微调 | 0.56 ± 0.06 | 0.87 ± 0.03 | 起点明确指向Taha的ICDAR2019表格模型，但结果属于新增训练后的模型 |

论文§2.4说明再微调从ICDAR2019表格检测版本起步，并引用[7]。参考文献[7]明确是Taha Douaji的 `detr-doc-table-detection` 模型卡，父模型为facebook/detr-resnet-50。§5.4也把有监督基线写为ICDAR2019预训练的DETR-19。因此，再微调行与17号的**初始化关系明确**，它的成绩不能作为17号未改权重的原始成绩。

对于零样本行，Table 1/2只写DETR (41M)；§6.5使用“vanilla DETR”一词，没有提供模型ID、revision或权重hash。§1将零样本定义为没有使用目标测试数据集的训练划分，因而“零样本”本身不排除已经在ICDAR上微调过。**既不能凭同名认定它就是Taha原权重，也不能凭vanilla一词认定它一定是facebook通用COCO原权重。**

论文§3.2/§5.1采用置信度0.5、每页只取最高置信度的一张表框，报告平均GIoU，漏检记0。当前17号结果是cTDaR Track A 439图的COCO AP，因此即使将来核实了零样本权重身份，也不能直接与这里的0.20/0.21作数值比较。

## 公开代码核查范围

正式CVF条目只列PDF，没有附录或评估代码链接。已检查全文出现的代码/仓库引用，主要是第三方组件及[27]作者的[Data_QUEST数据仓库](https://github.com/eliottthomas99/Data_QUEST)。该仓库公开DocILE-QUEST的图像/OCR/金标与670/143/141训练/验证/测试划分；其可见README和顶层目录没有给出这篇WACV基准各行加载checkpoint的评估脚本。

因此本轮**没有找到能固定零样本DETR权重身份的公开评估代码**。作者GitHub完整仓库清单API读取超时，不能进一步声称作者所有仓库都没有代码。STM154在论文中明确为私有，未公开。

建议面向用户的表述：

> 17号模型卡没有指标，但论文检索有相关结果：第三方基准的DETR零样本GIoU为0.20/0.21，再微调后为0.56/0.87。再微调起点确实用了17号；零样本行没有明确到同一权重版本，所以目前还不能把这些数字当成17号原始权重在cTDaR上的官方成绩。

如需继续对齐，最小工作是找到该论文零样本DETR的实际模型ID/加载代码，再按公开DocILE-QUEST 141页和论文GIoU协议建立一项补充对照。不要根据分数接近程度猜权重，不需要为了对照而重新训练17号。

本次PDF已直接从CVF下载到 `/tmp/model17_thomas_wacv2026.pdf`（883323字节，SHA256 `162369e478239294ae2a9123cecc85be49d7b53c65b481c70436bbd8f63e15d9`），全文经pdftotext提取至 `/tmp/model17_thomas_wacv2026.txt`；文字提取器提示PDF字典字符串语法警告，但相关章节、表格及参考文献均成功提取，数值与CVF搜索索引表格一致。
