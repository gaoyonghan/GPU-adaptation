# 30号模型重新核查（2026-09-05）

**后续执行更新**：本报告为补跑前复核；现已完成新数据全量推理，最新结果见[model30/补评结果.md](../supplemental_20260905/model30/补评结果.md)。下文保留当时核查过程。

结论：**ModaNet没有因为“未公开”而无法使用。本轮漏查了作者已经公开、带图片和七类标注的合并测试集，应补用作者5544图测试集重新评估。** 原1158图Fashionpedia结果可以保留为替代基线，但不应继续作为“已按作者数据方案完成”的依据。

## 直接证据

- 模型卡写明用ModaNet与Fashionpedia混合训练，第96/100个epoch取得mAP 0.7542，并链接作者仓库：https://huggingface.co/yainage90/fashion-object-detection 。
- 作者仓库README直接链接公开合并数据：https://github.com/yainage90/fashion-visual-search ，数据地址 https://huggingface.co/datasets/yainage90/fashion-object-detection 。
- 2026-09-05实际HF API返回 `private=false`、`gated=false`；固定revision `959476ce2fde6bd870c6578be22183114b0da131`；数据2024-08-22创建及最后更新。train为84431图，test为5544图，七类名称和bbox格式已经保存。元信息见 `sources30/hf_dataset.json`。
- 公开测试集只有一个Parquet，包含图片字节和GT，不需要再去PaperDoll下载原始全集。对下面固定URL进行了16字节范围探测，实际返回HTTP206、`Content-Range: bytes 0-15/260915347`、文件头`PAR1`：https://huggingface.co/datasets/yainage90/fashion-object-detection/resolve/959476ce2fde6bd870c6578be22183114b0da131/data/test-00000-of-00001.parquet 。本次核查未下载完整260.9 MB数据，未启动推理。

## 对应七类发布时的代码

当前GitHub HEAD `7ef84081f528e3b6cc12de95e453e6fff0eb552f`已经混有12类v2转换代码，不能直接套用。已回溯并固定七类版本 `76c1dcd166ebae50bd1cad5ed1bbf3417e2fcc7c`（2024-12-02），相关原始Notebook和可读代码已保存到 `sources30/author_7class/`，SHA256见 `sources30/source_manifest.json`。

该版本可直接核对以下链路：

| 数据部分 | 作者处理后的train | 作者处理后的test | 证据 |
|---|---:|---:|---|
| Fashionpedia | 44932 | 1155 | 转换Notebook的已保存输出 |
| ModaNet | 39499 | 4389 | 转换Notebook的已保存输出 |
| 合并 | 84431 | 5544 | merge_dataset.ipynb的已保存输出，与公开HF数据完全吻合 |

作者ModaNet处理调用 `maskrcnn-modanet` 的整理版数据，先过滤无有效目标、非法框及鞋框数量大于2的图片，再 `train_test_split(test_size=0.1)`，没有显式记录seed。**无需猜这个随机划分：直接使用作者已发布Parquet即可冻结5544图。** Fashionpedia从1158图过滤到1155图，两个来源分别生成划分后拼接。原始来源和处理后划分不能混为一谈。

固定代码链接：

- 数据合并：https://github.com/yainage90/fashion-visual-search/blob/76c1dcd166ebae50bd1cad5ed1bbf3417e2fcc7c/object_detection/merge_dataset.ipynb
- ModaNet转换：https://github.com/yainage90/fashion-visual-search/blob/76c1dcd166ebae50bd1cad5ed1bbf3417e2fcc7c/object_detection/convert_modanet_to_huggingface_dataset.ipynb
- Fashionpedia转换：https://github.com/yainage90/fashion-visual-search/blob/76c1dcd166ebae50bd1cad5ed1bbf3417e2fcc7c/object_detection/convert_fashionpedia_to_huggingface_dataset.ipynb
- 训练和评估：https://github.com/yainage90/fashion-visual-search/blob/76c1dcd166ebae50bd1cad5ed1bbf3417e2fcc7c/object_detection/train_model.ipynb

## 本轮为什么只用了Fashionpedia

本地 `models.json` 的原始依据只指向eBay/modanet，将作者七类映射及合并划分记为未知；随后按“替代数据覆盖全部七类”的放宽要求，使用已在本地的Fashionpedia 1158图，自行定义合并映射。因此省略的真实原因是**前期来源追溯不完整、作者合并集没有接入**，不能解释为ModaNet未开源或下载被封。

已冻结结果：`__MODEL_ROOT__/resources/runtime/assets/workspace/results/remaining25_v3/m30-fashion-object-detection/fashionpedia7/status.json`，AP 0.7159294163，AP50 0.8731342119；GT哈希 `ff46c9b57d032daac932efb51716540e97292e66851a7fb519f76565cca467b1`；权重SHA256 `01f78edc0c0e3abfcf8b09555d04ab5a43216a501bf24559ce5619673c4ce824`。不应覆盖或修改这些历史结果。

另发现本轮自拟映射与作者七类转换有四处差异：

| Fashionpedia原类别 | 本轮1158图映射 | 作者七类代码 |
|---|---|---|
| vest | top | outer |
| cape | 未映射 | outer |
| tights, stockings | 未映射 | bottom |
| sweater | top | 未映射 |

这进一步说明只比较71.59与75.42没有充分依据。七个大类“都有正样本”不等于遵循了作者类别定义。

## 原指标可以进一步确定

作者训练脚本明确 `eval_dataset=dataset['test']`，每epoch评估，使用 `eval_map` 挑选最佳；`compute_metrics`用 `MeanAveragePrecision(box_format="xyxy", class_metrics=True)`，置信度阈值0.0。原requirements固定Torchmetrics 1.4.1、Transformers4.44.0、PyTorch2.3.0、pycocotools2.0.8。

已读取Torchmetrics1.4.1源码：默认IoU为0.50到0.95步长0.05，101点召回，最大检测数[1,10,100]。因此原mAP不是完全未知口径，应更新为COCO风格AP@0.50:0.95。官方源码：https://github.com/Lightning-AI/torchmetrics/blob/v1.4.1/src/torchmetrics/detection/mean_ap.py 。作者脚本评分时只向GT传boxes和labels，故目标面积由框计算；不要直接用Parquet中多边形面积替代这一行为。

模型卡展示用的threshold0.4、推理Notebook中的NMS是展示流程，不能套在训练评估的threshold0.0流程上。

## ModaNet公开性与可取得性

原ModaNet是公开标注集，eBay仓库给出了训练/验证标注、哈希和下载说明，图片源于PaperDoll；标注许可证CC BY-NC4.0。来源：https://github.com/eBay/modanet 。

作者实际使用的 `maskrcnn-modanet==1.0.3` 对原鞋类框做了整理，并提供图片和标注下载命令：https://github.com/cad0p/maskrcnn-modanet 。老PaperDoll入口本次Web访问404，但这不代表原数据未公开，更不能否定作者合并Parquet当前可直接下载的实证。

## 最小补做方案

1. 仅下载固定revision的作者test Parquet（260,915,347字节），核实SHA256、5544图、七类schema、图片解码和bbox合法性。不要重建随机split，也不用下载约5GB训练集。
2. 直接使用作者已编码的七类category和COCO xywh框；以唯一行ID避免两个来源image_id碰撞。记录作者原image_id用于追踪，检查各类GT覆盖。
3. 采用作者评估函数的threshold0.0、Torchmetrics1.4.1/pycocotools行为，保留原图尺度，GT area按作者评分路径由框计算。权重沿用本轮同一个哈希。
4. 新输出根目录先10图验证再5544图完整评估，产出总体AP/AP50/AP75和七类AP；旧1158图结果继续标为替代参考，不覆盖。结果再与0.7542比较，不因接近而调划分或参数。

本次核查只读取公开资料和已有结果，保存了少量来源代码及元信息，没有安装依赖、下载完整数据、运行GPU任务或修改既有模型台账。
