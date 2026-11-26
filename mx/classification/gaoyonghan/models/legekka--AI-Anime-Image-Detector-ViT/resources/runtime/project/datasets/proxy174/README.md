# 174 号模型三类替代评测集（v1）

## 用途与映射

目标模型：[glazzova/body_type](https://huggingface.co/glazzova/body_type)，固定模型版本 `49c5c2cc437566416b56364095cdc9e6b357e7cf`。模型原始类别 ID 为 `0=skinny`、`1=ordinary`、`2=overweight`、`3=hulk`。

本集从 [Celeb-FBI](https://huggingface.co/datasets/alecccdd/celeb-fbi) 的 `train` 划分抽取 36 张单人、可见大部分身体的照片，经**人工视觉筛选**，每类 12 张：

原有本地候选经核查不够支撑三类可靠评测：SomaSet 的 50 个基础人物缺少体型标签且图像高度同质；`smolVLM_bodytype` 主要是脸部裁剪；本地人体测量照片只有 6 位不同受试者。因此本版使用另外一个公开数据源，**不是**从本地旧集直接抽取。

| 替代集标签 | 模型标签 | 数量 | 选择依据 |
| --- | --- | ---: | --- |
| 偏瘦 | `skinny` (0) | 12 | 全身轮廓明显纤瘦 |
| 普通 | `ordinary` (1) | 12 | 可见体型处于中间档 |
| 偏胖 | `overweight` (2) | 12 | 腰腹/整体轮廓明显偏宽厚 |

`hulk` (3) 缺席。身高体重/BMI 无法区分肌肉与脂肪，而且原数据的性别、身高体重记录存在错配，所以**没有把 BMI 自动当作标签**。三类标签仅为单人视觉复核的工作标签，未做独立复标。

## 文件

- `images/<label>/*.jpg`：原始图片字节，未裁剪或改色。
- `index.jsonl`：每行一个样本，含相对路径 `member`、标签 `target`、模型类 ID `target_id`、来源划分与行号、SHA-256、尺寸。
- `checksums.sha256`：图片校验值。
- `review_*.jpg`：三张 12 图预览，供人工复核。

## 建议评测口径

对 36 张图保持模型原始四类输出，直接取四类 argmax；任何预测为 `hulk` 的结果在本集上计为错误。报告 36 张整体准确率、三类各自召回率和包含 `hulk` 预测列的混淆矩阵。**不要**重新归一化前三类概率。本集只能作为 174 号模型的三类代理/流程验证，不能替代四类官方测试，也不能与模型卡数字直接比较。

## 限制与来源

这 36 张是为了核验流程而人工挑选的小样本，统计误差、名人照片偏差、衣着遮挡和人工标签主观性都较大。原数据集标为 Apache-2.0；第三方照片的原始版权状态仍需按实际使用场景核查。数据来源：`alecccdd/celeb-fbi`，`train` 划分，仓库版本 `df25535c4416a001d05fd8169b1bdc993c3a05d2`。原数据说明：[Celeb-FBI 数据卡](https://huggingface.co/datasets/alecccdd/celeb-fbi)。
