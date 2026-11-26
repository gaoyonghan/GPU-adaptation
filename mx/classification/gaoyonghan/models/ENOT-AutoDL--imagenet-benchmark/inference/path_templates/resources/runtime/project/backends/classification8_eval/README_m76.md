# 76号：原ViT权重 + AnimeDL-2M图像级替代评测

2026-09-15用户最终确认：采用全量测试范围，接受约259GB下载。分阶段执行：先用`--prepare`下载和准备完整测试数据，再用无参数入口评测。为避免误操作，评测入口不自动触发大包下载；它在数据未准备时会提示先执行`--prepare`。

## 运行与下载规模

```bash
cd __MODEL_ROOT__/resources/runtime/assets/workspace/classification8_eval
# 建议先执行：只下载测试清单及少量官方源码，打印下载规模，不下载图片、不跑推理
bash run_m76_animedl.sh --plan
# 仅适用于已具备全量数据的情况；数据未准备时会停止，不自动下载大包
DEVICE=0 bash run_m76_animedl.sh
```

首次需要约259GB压缩包、另需空间保存原始人工图和准备后的图片。只选10个test分片，非全库；真实图约82.86GB、生成图约175.79GB（后三个分片无t2i，跳过其fake包）。已检查数据盘剩余约14TB。时间主要取决于网络和解压；不是小型几千图实验。

数据目录`__DATASET_ROOT__/m76_animedl_test_v1`，结果目录`__MODEL_ROOT__/resources/runtime/assets/workspace/results/m76_animedl_v1`。重复运行保留原包，下载由Hugging Face缓存续传/复用，按上游SHA256校验。每个已提取分片重新检查图片哈希后复用。不自动删除坏文件或覆盖不一致缓存。

仅准备数据：`bash run_m76_animedl.sh --prepare`。全部数据就绪后只检查哈希与解码：`bash run_m76_animedl.sh --preflight`。`MAX_IMAGES=10`只限制推理数量，**首次仍需准备全套测试数据**；不把它描述为10张图片下载。

复用`__MODEL_ROOT__/resources/runtime/assets/workspace/classification8_eval/.venv`，不安装/更新共享依赖。已有torch2.5.1+cu121、transformers4.50.0、Pillow11.1.0、numpy1.26.4、huggingface-hub0.29.3；作者模型config记录4.43.3，差异保留。需要单张CUDA GPU；模型FP32 batch8。纯CPU不能直接运行原生ImageAccuracy中的CUDA计分代码。

## 范围及原生程度

- 原模型：[legekka/AI-Anime-Image-Detector-ViT](https://huggingface.co/legekka/AI-Anime-Image-Detector-ViT)，本地原始safetensors。直接用模型卡示例的AutoFeatureExtractor/AutoModelForImageClassification，本地配套预处理，512×512、RGB、FP32，不裁额外ROI、无TTA。
- 替代数据：[FlyTweety/AnimeDL-2M](https://huggingface.co/datasets/FlyTweety/AnimeDL-2M)，固定revision `f1ef103a2ded7d3e1a39c09dd05cd5ecc357a33b`。使用作者`data_list/test`的0050、0110、0240、0260、0400、0430、0480、0680、0930、0970共10份清单。
- 全部test的`raw`为real=0（63868张），清单实际列出的`t2i`为ai=1（107482张），合计171350张。空t2i列表不是报错，不虚构文件；排除inpaint局部编辑、像素mask和额外civitai子集。不抽取train/val，不为凑1:1比例删除真实图片；总体类别比例不均衡，额外提供两类识别率。
- 原始人工图按作者说明来自[deepghs/danbooru2024-sfw](https://huggingface.co/datasets/deepghs/danbooru2024-sfw)，revision `7ccab00fa0369e81c2a6e92074ed8ff7b43cf945`。当前API标记gated=false；仍须遵守来源条款。只下载所需10个分片，保留原包和被选中的原图。
- 人工图准备直接提取并执行作者`real_images/resize.py`中的`resize_images`函数，不修改函数体；仅用AST排除模块级CLI/16进程启动。按作者入口参数最长边1024、对齐16，Lanczos缩放及JPEG quality95重编码。这是作者的数据准备步骤，不是人为拼接或合并标签。生成图保留包内字节。
- [官方评测代码](https://github.com/FlyTweety/AnimeDL2M/tree/bd81880df8577aa3d7ae746e1b492bc837d3a77d/AniXplore/IMDLBenCo/IMDLBenCo/evaluation)固定commit `bd81880df8577aa3d7ae746e1b492bc837d3a77d`。下载未修改的`Accuracy.py`和`abstract_class.py`，直接调用`ImageAccuracy(threshold=0.5).batch_update/epoch_update`，单rank NCCL满足原生分布式归并要求。输入是模型softmax的ai概率和0/1图像标签，形状均[B]，规则严格为概率>0.5；与此二分类argmax一致（并列归real）。额外逐图正确数/总数交叉检查，不用其值覆盖原生指标。
- 作者完整`test-AniXplore.py`使用AniXplore、AnimeDatasetNoReal及PixelF1等像素级接口，不能直接拿来加载76号图像分类权重。这里复用适用的原生计分组件、原生数据准备函数和发布模型推理组件；选择raw/t2i、文件下载/提取、批量调度、溯源及两类诊断统计是项目包装代码。**不冒称整套AniXplore评测入口原样运行，也不生成伪分割mask。**

## 结果及审计

模型卡94.68%对应作者约5000张图片，未公开对应列表。本替代集不同，`officially_comparable=false`，不得把94.68%作为本集必达分数。两类均覆盖不代表生成器、画风、压缩方式或训练重叠关系一致。

运行结果每次新时间戳目录，smoke/full分开；只有完整171350张才生成metrics.json。无推理断点续跑；失败后保留记录并在新目录重跑。逐图记录包含source_dataset、split、shard、原清单行、原始路径、压缩包成员、原始/准备后图片SHA256及预测。protocol.json记录模型/源码哈希、依赖版本、数据revision和计分边界。

数据仅用于符合[作者条款](https://github.com/FlyTweety/AnimeDL2M)的非商业学术研究，原图权利仍归原作者；国产GPU适配若涉及商业使用，需先核实授权。本项目不重新分发图片。

交付检查通过：真实10份测试索引的171350个唯一ID和两类计数、计划阶段网络读取及固定版本官方源码获取、合成压缩包定向提取、原生resize函数、原生ImageAccuracy的CPU测试替身、语法及本地模型CPU加载。测试替身仅用于单元检查，没有修改运行时官方源码。大包头部抽查遇到HF CDN读取超时，未完成真实包内路径验证；交付验证不包含GPU推理或259GB真实包端到端验收。正式下载/提取过程中若包内图片与官方清单不一致，脚本停止并报告缺失，不跳过、不假定成功。
