# 85号：数据与官方协议核查

最新入口（2026-09-15）：按用户“能原生就原生”的要求，85/86改用 `bash run_m85_m86_native.sh`；详见 [原生评测说明](README_m85_m86_native.md)。旧HF结果保留作对照，原生结果等待终端运行验收。

更新（2026-09-15）：用户已批准Kaggle下载包与官方验证清单交集的**部分覆盖评测**，两模型统一入口为 `bash run_m85_m86_partial.sh`，详见 [85/86部分覆盖说明](README_m85_m86_partial.md)。已确认交集1192张/124类，保留21841类输出；这不意味着下述官方全量数据缺口已解决。下文保留此前官方原版获取与核查记录。

核查日期：2026-09-15。状态：协议要求已核对，数据未就绪；未运行推理，不能标记为官方复现完成。

## 权重与目标

- 保留清单中的 `microsoft/swin-large-patch4-window12-384-in22k`，不替换为1K微调版。
- 本地目录：`__MODEL_ROOT__/weights/microsoft/swin-large-patch4-window12-384-in22k`。
- 已读本地配置：Swin-L，输入384，window12，depths=[2,2,18,2]，embed_dim=192，id2label共21,841项。尚未执行本轮权重张量、原生/HF数值等价核查。
- 本地模型卡正文为21,841类；单图示例中“1000类”的文字与配置不一致，不采用该注释确定任务。
- 官方模型库的Swin-L/384行87.3% Top-1、98.2% Top-5对应1K微调评测，不能作为现有22K权重在22K上的参考数值。本轮未找到该权重可直接对应的22K参考分数。

## 数据对齐要求

1. 原流程指定 `fall11_whole` 图片与官方 `ILSVRC2011fall_whole_map_val.txt`；不得自己随机划分或把整个预训练集作为独立测试集。
2. 官网授权下载页当前列出 `winter21_whole.tar.gz`，未列出Fall 2011。官网更新公告明确删除了person子树2,702个synset；因此2021版不等于2011版，不能通过改目录名完成对齐。具体验证样本缺口数量尚未核算。
3. 不下载官网 `imagenet21k_resized.tar.gz` 作为原版替身：该文件是另一个处理版本。
4. 发现Academic Torrents候选存档，页面标注Fall 2011、`fall11_whole.tar`、1.31 TB；它是第三方渠道，尚未验证可下载性、实际文件及原版一致性。未开始P2P下载，也未代替用户接受条款。
5. 原版获取后应核对清单路径、样本数、类别ID、重复项和解码。官方当前IN22KDATASET读取JSON格式（即使文件后缀为txt），不得按普通空格分隔文件臆测解析。

已在线读取官方验证清单：10,766,810字节，197,596条记录，21,076个实际出现的类别ID，标签范围0至21,839；SHA256为 `5f6c9fe77da598dfb855d23ca3c5fa1169d73e421d6bfc12351eca935c87e424`。来源：https://github.com/SwinTransformer/storage/releases/download/v2.0.1/ILSVRC2011fall_whole_map_val.txt 。该清单自身并非21,841类都有正样本；缺少765类正样本，不应归因于2021版删类，也不得自行补图后仍声称沿用同一划分。清单仅在内存中核查，本轮未保存本地副本或获取图片。

## 运行协议核查

- 官方 `data/build.py` 的 `imagenet22K` 分支使用上述train/val清单和21,841类别；分数由 `main.py::validate` 中timm的Top-1/Top-5 accuracy计算。
- 官方当前 `IN22KDATASET._load_image` 在读图异常后使用随机图继续。项目须在运行前拦截缺失/损坏图片，不能把此行为产生的分数算作有效全量结果。
- 本地HF处理器保存384尺寸、双三次插值、ImageNet mean/std；官方是否启用 `TEST.CROP` 需结合最终冻结的384配置继续核对，不得宣称所有预处理参数已完全对齐。
- 官方 `utils.py::load_pretrained` 也提供22K到1K分类头行选择，映射文件有1,000个互异索引。该分支用于加载/微调，不等于现有22K权重已经取得87.3%；若选择1K子任务评测，必须单独标注协议并获得用户确认，不能默默取代完整22K任务。
- 即使采用仓库现有22K划分，也须继续核实其与2021年发布的该权重的训练样本关系，不能仅因清单来自同一仓库就认定是该权重未见过的独立测试集。

## 下一步与边界

下载脚本：`bash __MODEL_ROOT__/resources/runtime/assets/workspace/classification8_eval/download_m85_fall11.sh`。用户于本轮要求提供旧版下载指令；脚本未在Codex内执行下载。保存到 `__DATASET_ROOT__/imagenet22k_fall11`，使用Academic Torrents种子，固定info hash `564a77c1e1119da199ff32622a1609431b9f1c47`，期望归档大小1,309,848,811,520字节，支持续传和分块校验。P2P期间可能上传分块（限制1 MiB/s），完成后不继续做种；分块校验只能证明与该种子一致，不等于已经与官方原版或验证清单逐图对齐。

优先取得可核验的Fall 2011原版（向ImageNet官方咨询旧版研究访问方式，或核验第三方候选）。取得数据后冻结源码、清单和权重版本，补齐标签顺序与处理器核查，再交付终端运行脚本。当前没有启动大文件下载、安装环境或运行推理，也没有改变现有模型清单及历史结果。

## 证据链接

- 模型卡：https://huggingface.co/microsoft/swin-large-patch4-window12-384-in22k
- 官方数据准备：https://github.com/microsoft/Swin-Transformer/blob/main/get_started.md#data-preparation
- 官方模型库：https://github.com/microsoft/Swin-Transformer/blob/main/MODELHUB.md
- 官方读取器：https://github.com/microsoft/Swin-Transformer/blob/main/data/imagenet22k_dataset.py
- 官方预处理：https://github.com/microsoft/Swin-Transformer/blob/main/data/build.py
- 官方评测：https://github.com/microsoft/Swin-Transformer/blob/main/main.py
- 官方分类头映射：https://github.com/microsoft/Swin-Transformer/blob/main/utils.py
- 数据集更新：https://www.image-net.org/update-mar-11-2021.php
- 授权下载页：https://www.image-net.org/download-images.php
- 处理版本区别：https://github.com/Alibaba-MIIL/ImageNet21K/blob/main/dataset_preprocessing/processing_instructions.md
- 第三方旧版候选：https://academictorrents.com/details/564a77c1e1119da199ff32622a1609431b9f1c47
