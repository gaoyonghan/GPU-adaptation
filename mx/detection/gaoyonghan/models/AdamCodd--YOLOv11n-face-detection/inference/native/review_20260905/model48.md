# 48号 Nemotron Graphic Elements v1 重新核查

**后续执行更新**：本报告为补跑前复核；现已完成新数据全量推理，最新结果见[model48/结果报告.md](../supplemental_20260905/model48/结果报告.md)。下文保留当时核查过程。

核查日期：2026-09-05。范围：只读检查现有结果、原始标注和官方来源；本次没有修改原结果、环境，没有启动GPU推理。

## 结论

**原模型卡对应的560图、11520框、10类评估标注已经准确找回，而且是公开数据。此前只用1873图、合并8类，是我们漏查了模型卡引用的CACHED公开数据包，不能解释为原评估集无法获取。48号需要换回该560图10类协议重新评估，旧63.13 AP只能保留为替代集参考。**

从CACHED作者公开的 `pmc_2022.zip` 中，仅用HTTP Range读取ZIP目录及标注，取得 `pmc_2022/pmc_coco/element_detection/val.json`。它包含560图、18类共22684框；筛选NVIDIA原生输出的10类后恰为11520框。**10类的框数和包含该类的图片数，全部逐项与NVIDIA模型卡一致。** 这是比只匹配数据集名称强得多的来源证据。

官方入口：[NVIDIA模型卡固定版本](https://huggingface.co/nvidia/nemotron-graphic-elements-v1/blob/4a76546bb1bb4cbab3401361c91cf01706321805/README.md)；[CACHED作者仓库](https://github.com/pengyu965/ChartDete)；[作者公开PMC数据包](https://drive.google.com/file/d/1YZmRy9TcaOJ6tA8DLeG0sXhSjwqqAw_h/view)。

| 原生类别 | 模型卡图数／框数 | 找回val图数／框数 | 模型卡AP% |
|---|---:|---:|---:|
| chart_title | 38／38 | 38／38 | 82.38 |
| x_title | 404／437 | 404／437 | 88.77 |
| y_title | 502／505 | 502／505 | 89.48 |
| xlabel | 553／4091 | 553／4091 | 85.04 |
| ylabel | 534／3944 | 534／3944 | 86.22 |
| other | 113／464 | 113／464 | 55.14 |
| legend_label | 318／1077 | 318／1077 | 84.09 |
| legend_title | 17／19 | 17／19 | 60.61 |
| mark_label | 42／219 | 42／219 | 49.31 |
| value_label | 52／726 | 52／726 | 62.66 |
| 合计（图片去重） | 560／11520 | 560／11520 | — |

原卡逐类AP的算术均值为74.37%；这是本次计算值，原卡没有单独列出此总值。不能直接拿它与旧8组AP63.13作同口径差值。

## 已取得的证据和图片获取量

- [CACHED_val.json](model48_sources/CACHED_val.json)：原始COCO标注，无转换；SHA256 `403e298f1bbe6f36f1aa3741a6cf9a65dd7ce85975193622c3e15d860f6b436b`。
- [数据统计](model48_sources/CACHED_dataset_stats.json)：同时检查了381图 `split3_test.json`，后者原生10类共8014框，不是模型卡那560图。
- [560图ZIP条目](model48_sources/CACHED_val_image_entries.json)：560张全部在公开ZIP内，无缺图；压缩总量22012973B，解压27111453B。图片集中于约22.09MB连续区间，可按Range只取此段，无需下载2.13GB整包。**本次尚未读取这些图片内容。**
- [获取摘要](model48_sources/CACHED_val_retrieval_summary.json)：560图文件名与当前ChartInfo 2022测试包全部13259图的交集为0。
- 源ZIP大小2125251674B；已按ZIP CRC与解压字节数验证两份标注提取正确。

## 旧1873图从哪里来，为什么不能代替560图

现有 `prepare_native_data.py:30` 的 `chart()` 使用竞赛官方测试包 `ICPR2022_PMC_TEST_v2.1.zip`，挑选具有 `task2` 文字多边形和 `task3` 文字角色的全部1873图。它们由 split_1的2图、split_2的603图、split_3的381图、split_4的444图和split_5的443图组成。这些split原本对应不同竞赛任务，并非NVIDIA模型卡的validation。

ZIP自带readme说明：全包13259张互不重复测试图片，split_1到5分别服务图像分类、OCR、角色／坐标轴／图例、图表数据提取、端到端提取。1873这个选择本身有确定规则，并非按模型分数挑选，但其用途应明确叫“全测试包中有文字角色GT的替代检测集”。[竞赛数据入口](https://chartinfo.github.io/toolsanddata.html)

旧8组的真实正GT数量为：chart_title226、axis_title3022、tick_label28032、other2741、legend_label3888、legend_title116、mark_label764、value_label1575，共40364框。另290个 `tick_grouping` 区域复制为每个评分组的crowd忽略框，不能计入真实类别覆盖；原status的每类GT数量因包含这些忽略框而偏大290。

旧结果逐组AP为：chart_title76.72、axis_title88.72、tick_label82.02、other30.27、legend_label82.82、legend_title65.81、mark_label27.55、value_label51.17。低分主要出现在other、mark_label、value_label等组，不能把总体差异单一归因于硬件或权重错误。

## 合并x/y并非不可避免

旧转换器只读取JSON的 `task2`、`task3`，没有读取其余人工标注。原始JSON的 `task4.output.axes.x-axis/y-axis` 有tick文本ID关联。原始XML更完整：`Panels/ChartInfo/Axes/AxisX1/AxisValues/Title`、`Labels/TextId`、`AxisY1`及第二坐标轴字段，直接指明轴标题和刻度标签属于哪根轴。

只读扫描1873图的XML人工关联后，可直接分出xlabel10517、ylabel10820、x_title966、y_title1239；1373图的全部轴相关文本都能明确分配。另500图仍有缺失或冲突（6695个tick_label、817个axis_title无法唯一归属，含9个双轴ID冲突），不应硬猜。详见[XML核查统计](model48_sources/xml_axis_audit.json)。

此外，NVIDIA引用的[CACHED论文](https://arxiv.org/abs/2305.04151)专门讨论把原PMC混合的轴类别拆成x/y，其[公开转换器](https://github.com/pengyu965/ChartDete/blob/main/pmc2coco_converter/dete_ann_coco_gen.py)用task4的ID拆刻度，并以标注几何距离派生轴标题类别。该转换器的主例程并未给出可重建NVIDIA560图的确定随机种子，因此**应使用已经发布的val.json，不重新随机切分，也不重新用我们自己的规则猜类别**。

## 推理实现检查

本地README、`model.py`、`graphic_element_v1.py`、`utils.py`与HF固定版本 `4a76546bb1bb4cbab3401361c91cf01706321805`逐字节SHA一致，见[源码比对](model48_sources/native_source_comparison.json)。现有适配器使用官方 `Exp`、`YoloXWrapper`、`preprocess`、`postprocess_preds_graphic_element`，从权重 `state['model']`严格加载；权重SHA为 `580b566b6746a5f1f045e1b935a5d7e3eed18d2e8a5d731246c4011243d2caa6`。

1024等比缩放、右下114填充、RGB 0–255、conf0.01、NMS IoU0.25及class-agnostic NMS与官方源码一致。官方wrapper先恢复原图坐标再归一化，本地适配器乘原图宽高恢复像素，未发现重复缩放或x/y颠倒。后处理0.1阈值也是官方示例默认值；模型卡未说明发布AP是否也使用这个额外阈值，仍应记录该限制。

确认的实现/记录问题有两处：

1. 数据处理漏查公开CACHED val及XML轴关联，导致换集、类别合并不必要。
2. 旧status仍继承 `dataset_relation: original_exact`，与实际1873图替代集矛盾；此前台账已有“不官方可比”备注，但机器字段仍需以新增勘误记录修正，不能静默修改旧冻结协议。

现有JSONL仅保存合并后的 `category_id`，没有保存合并前的10类ID。因此新10类基线需要重新推理，不能仅通过旧JSONL恢复原生类别。

## 最小正确修正路径

1. 取已定位的560张val图片，按ZIP CRC、图片尺寸及COCO文件名逐一核验；冻结原始val.json及筛出原生10类的COCO标注哈希。
2. 新建 `cached_pmc_val560_native10` 协议和新结果目录，取消4个x/y别名合并；用官方现有权重、预处理和后处理，以原生10类评分，保留原生类别ID。
3. 先10图检查坐标和分类映射，再全560图；报告10类逐类AP、AR和总AP，与模型卡逐项并排。若仍有差异，优先核对额外score0.1过滤以及COCO maxDets/IoU设置，避免用调参追分掩盖差异。
4. 保留旧1873/8组输出作为历史替代参考，并明确它被560/10类方案替代为优先基线。原模型卡全部10类任务可由新方案覆盖。
