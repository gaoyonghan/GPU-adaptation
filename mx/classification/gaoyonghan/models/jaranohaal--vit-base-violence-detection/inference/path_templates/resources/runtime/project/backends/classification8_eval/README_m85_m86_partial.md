# 85、86号：用户批准的ImageNet-21K部分覆盖评测

后续原生入口：`bash run_m85_m86_native.sh`，详见 [原生评测说明](README_m85_m86_native.md)。本文件及下述HF入口保留作旧结果复现与对照，不再作为本次原生重跑入口。

2026-09-15用户批准：使用已下载Kaggle子集，保留两个原模型，不冒充官方全量结果。入口不安装环境、不下载图片、不修改旧结果，不执行ZIP附带的`deletetar.py`。

## 一键运行

```bash
cd __MODEL_ROOT__/resources/runtime/assets/workspace/classification8_eval
DEVICE=0 bash run_m85_m86_partial.sh
```

顺序运行85和86；FP32、batch=2，避免同时占用GPU。复用`__MODEL_ROOT__/resources/runtime/assets/workspace/classification8_eval/.venv`。首次下载约11MB公开验证清单、标签名与参考转换源码（冻结SHA256）；失败可重试相同命令，已校验资产复用。完成模型不会自动跨运行跳过，失败后可用`MODEL_NUMBER=85`或`MODEL_NUMBER=86`只重跑指定模型。每次新建时间戳结果，不覆盖此前结果。

可选：`bash run_m85_m86_partial.sh --preflight`仅校验数据解码、映射、权重哈希；不证明模型前向已跑通。`MAX_IMAGES=10 DEVICE=0 bash run_m85_m86_partial.sh`仅冒烟，不能作为完整结果。默认1192张，首10张后及每50张打印进度；不另跑重复的冒烟进程。

输出：`__MODEL_ROOT__/resources/runtime/assets/workspace/results/m85_m86_imagenet21k_partial_v1/{full,smoke,preflight}/run_时间/`。共享`coverage.json`、逐图SHA256与标签的`manifest.json`；各模型目录`m85/`、`m86/`包含指标、预测、协议和状态。完整交集为`metrics.json`，冒烟为`smoke_metrics.json`。日志在`logs/`。

## 数据与覆盖

- 输入：`__DATASET_ROOT__/imagenet21k_kaggle/fqx-21k-comp-tars.zip`，12644314186字节，129个synset目录、132486张JPEG。全ZIP CRC于交付核查时通过；入口每次校验全部选中图片CRC、解码和SHA256。不全量解压、不复制图片。
- 来源：https://www.kaggle.com/datasets/qiexifan/fqx-21k-comp-tars 。这不是完整Winter 2021，也不是Fall 2011官方全量；文件名中的21K不代表有21K类。
- 验证清单：https://github.com/SwinTransformer/storage/releases/download/v2.0.1/ILSVRC2011fall_whole_map_val.txt ，原197596张/21076个有正样本的类别。固定SHA256见脚本。
- 仅对路径前缀`fall11_whole/`与`winter21_whole/`进行对应，synset和文件名必须相同，不随机划分、不凑类别。交集1192张/124类；缺196404张官方验证图片。ZIP其余131294张不参与评测。包中5类没有验证交集，名称写入覆盖报告。
- 只验证了来源文件名对应及本地字节一致性，尚未与官方原图字节逐一比较；该验证清单与这两份已发布权重训练数据的重叠关系未知。因此是部分覆盖参考结果，不是独立泛化性或官方精度复现证明。

## 模型、适配依据和差异

| 编号 | 保留的原模型 | 输入 | 输出 |
|---|---|---|---|
| 85 | microsoft/swin-large-patch4-window12-384-in22k | 384×384 | 21841类 |
| 86 | microsoft/swinv2-large-patch4-window12-192-22k | 192×192 | 21841类 |

1. 模型与预处理直接使用各模型卡的HF `AutoImageProcessor`、`AutoModelForImageClassification`接口、本地发布权重及预处理文件。固定哈希，缺失、多余、尺寸不符的权重全部报错；不忽略分类头、不选取124类子头、不调参、不微调。
2. 标签编号直接来自官方Swin验证清单，不按本地目录重排。HF v4.29.2两份官方转换脚本的`head`→`classifier`转换不重排分类头行，源码只留档、不执行转换或下载另一份权重。本轮未重新证明HF和原生权重逐张数值等价。
3. 85号已选中124类的标签名还与HF发布的ImageNet-22K标签表逐索引交叉核对。标签表共有21843个条目，不能用其长度改变模型输出。85号配置的非连续标签键、86号通用`LABEL_n`都不能用来按文字排序；86号沿用原分类头编号。官方读取器记载额外同义类别21841→9205、21842→15027；本交集不含这四个编号。
4. 主指标调用原Swin验证流程所使用的`timm.utils.accuracy(topk=(1,5))`（当前复用timm1.0.15），并用逐图预测独立复算校验。输出始终在完整21841类上Top-k；可选类别外的预测照常算错。不是124类闭集准确率。
5. 项目代码负责ZIP读取、验证交集、批处理、进度、保存和复算；不是原Microsoft `main.py`未经修改运行。预处理沿用HF发布配置，没有切换到原Swin `build_transform`；后者有`TEST.CROP`参数，不能宣称本入口已复现该参数。当前依赖比原发布版本新，版本及处理器完整记录在`protocol.json`。
6. `officially_comparable=false`。85号常见87.3%/98.2%对应1K微调版，不作为本22K子集基准；86号也不引用不同分辨率的1K微调分数。本次报告Top-1、Top-5及样本/类别覆盖，不报告“相较官方掉点”。

参考：https://github.com/microsoft/Swin-Transformer/blob/main/MODELHUB.md 、https://github.com/microsoft/Swin-Transformer/blob/main/data/imagenet22k_dataset.py 。原模型卡分别为 https://huggingface.co/microsoft/swin-large-patch4-window12-384-in22k 和 https://huggingface.co/microsoft/swinv2-large-patch4-window12-192-22k 。原图许可和使用限制仍遵从ImageNet及图片权利方；本项目只交付脚本，不重新分发图像。

交付验证已通过：Bash/Python语法；正常、异常路径/标签及重复官方记录的筛选测试；21841类合成logits的原生Top-k计分；两套发布处理器的384/192输出尺寸；实际1192张交集的CRC、RGB解码及85号标签名交叉核对；两个完整本地权重的CPU加载（missing/unexpected/mismatched/error均为空，分类头均21841）。没有执行模型前向或GPU推理，GPU端到端由用户终端运行验收。
