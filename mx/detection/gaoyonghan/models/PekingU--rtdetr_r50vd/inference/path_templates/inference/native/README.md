# 62 个检测模型：绿色数据集批量评测

## 60号模型卡版本原生验证（2026-09-15，待终端验收）

```bash
cd __MODEL_ROOT__/resources/runtime/assets/workspace/official_object_detection_eval
MAX_IMAGES=0 DEVICE=0 bash run_m60_native.sh
```

先10张验证，再自动全量57张。仅10张：`MAX_IMAGES=10 DEVICE=0 bash run_m60_native.sh`。CPU全量预检：`bash run_m60_native.sh --preflight`。原生CPU数据加载及原权重加载已通过：57图、221框、1张无目标图、无坏图和丢框；不在Codex执行模型前向。

- 原权重：`keremberke/yolov5n-construction-safety/best.pt`，SHA256 `3ce9a9eaf22c56759786ff7a52952d0d09c38877a53ce3eb016db2b2baceb989`。原数据为作者指定 `construction_safety_v1_valid/data/valid.zip`，不是替代集。57图、221框、定义17类但仅11类有GT；其余6类AP记null，不误用原生返回数组对缺席类填入的整体AP。
- 模型卡原始元数据明确 `library_version: 7.0.6`。随项目提供PyPI原版wheel和所缺IPython8.12.3及纯Python依赖ZIP，在运行目录隔离解包并校验SHA256。复用 `object-detection-yolov5-v2` 的Python3.10/Torch2.5.1等，不降级或安装进共享环境、不联网；未修改7.0.6源码。两个离线包及Python/Shell脚本均为必要运行文件。依赖的LICENSE和元数据保留于包内。
- 直接调用 `yolov5.val.run`，原生loader、letterbox、NMS、匹配和`ap_per_class`计分；只通过观察回调记录处理图片ID、调用原生JSON导出器，不改预测或评分。`save_json=False`避免该包将自定义数据误送到硬编码COCO路径；导出JSON不参与主指标二次计算。正确处理7.0.6的四项返回值：总体指标、逐类AP、逐类AP50、耗时。
- 采用原生验证conf=.001、NMS IoU=.6、multi_label=True、max_det300、img640、rect=True/pad=.5、无增强。batch16依据checkpoint训练batch8及原生train.py验证batch翻倍；FP16依据原生best模型验证流程。模型卡conf=.25/IoU=.45等为展示示例，不冒充评测参数。其他依赖沿用现有版本，不能宣称完整重建作者历史环境或保证36.5356%精确复现。
- 原zip仅提供COCO标注，必要转换为YOLO归一化xywh；类别编号保持0–16，名称只核对大小写，坐标逐框回算校验；图片字节不变，不裁框、不合框、不删除标注。原生CPU加载后的图片ID及每类框数与原标注严格核对，包括无目标图片。
- 输出：`__MODEL_ROOT__/resources/runtime/assets/workspace/results/m60_native706_v1/full/run_时间戳/`，含metrics、逐类AP/AP50、原生预测、protocol、源码快照及status。10张分开存smoke_metrics；旧 `object_detection_official_v2/m60-yolov5n-construction-safety` 的AP50=34.3914%保留，不自动更改基准。官方参考为模型卡validation AP50=36.535576104287554%。
- 协议仅保存checkpoint的epochs/batch_size/imgsz/seed四个必要训练字段，不复制其中可能存在的上传凭据。失败状态和已处理图片ID单独记录。

## 33号x版原生补跑（2026-09-14）

用户要求补跑x版以对照模型卡的x版指标。入口：`DEVICE=0 BATCH=16 bash run_m33x_native.sh`；先10图冒烟，通过后全量1020图test。`MAX_IMAGES=10`仅冒烟，`bash run_m33x_native.sh --describe`查看协议。结果独立保存到 `__MODEL_ROOT__/resources/runtime/assets/workspace/results/m33x_native83123_v1/{smoke,full}/run_*`，n版旧权重、脚本和 `object_detection_official_v2/m33-yolov11-license-plate-detection` 结果均保留。33号仍只占100模型清单中的一个位置，不把n/x两次实验计为两个完成模型。

本地已包含作者 `license-plate-finetune-v1x.pt`，SHA256 `bf87430a3b2d0d0511a153a58eb73e97da6212861d258016cbad65a105068a14` 与HF发布文件一致，不需再次下载。checkpoint版本为8.3.123、scale=x、单类License_Plate。脚本用 `.venv-m33x-ultralytics83123` 隔离安装Ultralytics8.3.123（只增加轻量包），共享现有Torch2.5.1+cu121/torchvision0.20.1，不改现有Conda或n版环境；镜像可用 `PIP_INDEX_URL` 指定。

采用作者数据源Roboflow License Plate Recognition v11 test，1020图/1085框。原图只建链接不复制，使用该版本官方 `convert_coco` 把原category1转换为YOLO class0，原category0是无框的父类；每框做坐标往返和数量核查，不合并、不删改有效标注。原始COCO JSON保留不变。随后直接 `YOLO(x.pt).val(split='test')`，原生读取/letterbox/NMS/AP均未重写；640、conf=.001、iou=.7、max_det300、FP32、rect=True、batch16。完整作者test命令未公开，batch16显式记录，并非完全历史环境复刻。

主结果使用原生 `native_metrics.AP50` 和 `native_metrics.AP`（0–1），对照模型卡x版98.13%和72.60%，不再用n版对比x版。checkpoint内的train_metrics（训练验证阶段AP73.118%）不是模型卡test参考值，不混用。每次保存配置、源码/数据/权重哈希、完整图像覆盖、原生预测和脚本快照。作者明确警告训练/test有重复，当前保留原划分用于协议对照；所有结果必须标注数据泄漏风险，不能视为干净的泛化基准。全量完成后再人工确认是否替换33号主结果。

## 29号原生验证补跑（2026-09-14）

入口：`DEVICE=0 BATCH=16 bash run_m29_native.sh`，先10张冒烟，通过后全量COCO val2017 5000张。`MAX_IMAGES=10`仅冒烟；`bash run_m29_native.sh --describe`核查并打印协议，不推理。复用 `object-detection-yolov10-v2` 和此前4/7/20/21号已准备的COCO数据，不重新下载、安装或复制全量数据。新结果：`__MODEL_ROOT__/resources/runtime/assets/workspace/results/m29_native_v1/{smoke,full}/run_*`；旧 `object_detection_official_v2/m29-yolov10x` 的AP=53.7819161%保留，不自动替换基准。

使用作者 `THU-MIG/yolov10` commit `453c6e38a51e9d1d5a2aa5fb7f1014a711913397`、原生Ultralytics 8.1.34分支。现有 `jameslahm/yolov10x` safetensors由作者支持的 `YOLOv10.from_pretrained(..., strict=True)` 加载，再直接 `model.val()`，不再逐图predict。CPU严格权重加载、官方原始COCO标注一致性已验证；本次GPU冒烟和全量尚待用户启动。

官方验证入口：[作者说明](https://github.com/THU-MIG/yolov10#validation)。采用640、conf=.001、max_det300、FP32、rect=True、原生NMS-free后处理；官方示例batch256，本地默认16，显存充足时可显式设 `BATCH=256`。保留Torch 2.2.2+cu121、torchvision .17.2+cu121、pycocotools 2.0.8环境并记录版本；不是完整历史环境复刻，不把数值接近自动标记为严格复现。

复用现有COCO格式准备器：YOLO标签用于原生加载器，原始36781条COCO标注（包括crowd）用于最终COCOeval；官方导出的预测未经我们自定义NMS或坐标算法处理。原生加载器内部AP和官方COCO AP分开记录，主指标为 `metrics.json → coco_metrics.AP`（0–1，汇报乘100），对照官方YOLOv10-X 54.4%。每次保留脚本快照、真实参数、权重/代码/输入校验、图像覆盖、预测和结果；冒烟不产出可汇报AP。`native_internal_metrics` 是COCO JSON重评分前的原生内部统计，不能与最终COCO AP混写。

## 27号官方评测重新接入（2026-09-14）

入口：`DEVICE=1 bash run_m27_native.sh`，复用 `object-detection-wilddet3d-v2` 环境，先10图冒烟、通过后全量2470图。`MAX_IMAGES=10`仅冒烟；`bash run_m27_native.sh --preflight`只核查。输出到 `__MODEL_ROOT__/resources/runtime/assets/workspace/results/m27_native_text_v1/{smoke,full}/run_*`，不覆盖历史结果。不沿用旧demo冒烟入口。

本次查明：现有WildDet3D完整权重已包含SAM3、LingBot和3D预测头，CPU严格加载2011个张量通过，无缺失/多余参数。此前“必须先下载两份辅助权重”的判断仅适用于旧初始化入口，并非完整权重本身缺参数。当前采用官方模型结构、用发布者完整权重严格加载；LingBot结构配置逐字提取自固定revision的公开辅助checkpoint元数据，来源、SHA和提取范围记在 `m27_lingbot_config.json`，不下载其大张量，也不访问受限SAM3权重。不是随机组件参与推理。

沿用锁定官方 `configs/eval/in_the_wild/text.py` 的数据加载、预处理、纯文本collator、模型前向、后处理和Detect3DEvaluator评分；输入1008、实际推理batch=1、score_threshold=0、分类内NMS IoU=0.6、无GT深度、canonical_rotation=True。只调整组件初始化、单卡运行strategy（auto替代训练DDP）和覆盖/指标记录。两种3D评价（bbox IoU和中心距离）分开保存，另保留2D AP。所有AP原始单位为0–1；不把ATE、AOE等误乘100。冒烟不计算正式指标。

本地官方InTheWild v3 validation为2470图、785类、43867条原始标注，图片缺失0；正式有效目标由官方过滤逻辑决定，不能把原始标注总数当有效3D框数。官方当前代码频次划分与论文类别统计还需结合结果核对。论文表3 All Data/Text的AP=22.6作为待核对参考，不提前宣称严格复现。运行会保存源码commit/校验、权重与标注SHA、脚本快照、官方完整配置、环境版本、预测和逐图覆盖证明。

当前状态：准备和CPU严格权重加载已验证；尚未进行本次GPU冒烟/全量，不计为新增完成结果。下文9月6日“辅助权重授权阻塞”等条目保留为历史记录，以本节新入口为准。

## 25号检测部分结果确认（2026-09-14）

用户确认暂按当前检测部分结果记录：`racineai/UI-DETR-1`，作者评测使用的WebClick test全量1639条，非替代集。主记录为**目标中心覆盖率89.3837706%**，附加记录为**IoU≥0.5覆盖率60.0976205%**。前者表示任一预测框包含目标中心，后者表示任一预测框与目标框IoU≥0.5；均为检测覆盖诊断，不是AP或模型自主选择目标后的点击准确率。

固定目录：`__MODEL_ROOT__/resources/runtime/assets/workspace/results/object_detection_official_v2/m25-ui-detr-1/`；原始结果和预测保持不变，`models.json`的`accepted_baseline`记录采用范围。RF-DETR 1.5.2原生predict、RFDETRMedium、输入1600、阈值0.35、FP32；后续检测器适配需固定原status.json协议。模型卡70.8%是检测器加Gemini 2.5 Pro的端到端点击准确率，未单独提供可对应的检测精度指标；不得与89.38%直接比较、计算掉点或标记官方复现通过。记录状态为“检测部分全量完成；端到端点击评测未完成”，保留该限制，不启动额外大模型调用。

## 24号原生验证补跑（2026-09-14）

**基准确认（2026-09-14）**：用户确认采用 `m24_native8021_v1/run_20260914_194044_047863` 的validation全量70图/109框结果作为后续汇报及国产GPU适配对照基准。主指标为原生 **AP50=94.6291542%**，模型卡95.194%，差−0.5648458个百分点；原生AP50–95=90.4839709%。标注“接近模型卡”，不标注完全一致或严格历史复现。旧AP50=92.0131132%及其权重、脚本、结果保留为历史，不再作为24号主结果。

固定结果目录：`__MODEL_ROOT__/resources/runtime/assets/workspace/results/m24_native8021_v1/run_20260914_194044_047863/`。原始`status.json`、`metrics.json`、`predictions.json`及源码快照保持不变；`models.json`中的`accepted_baseline`记录固定路径及指标。后续对照采用本次原生8.0.21、v5loader=True、batch=4、FP16、640、conf=.001、iou=.7、max_det=300的协议；清单中旧predict+COCOeval协议仅用于保留历史流程。后续新运行不自动替换此基准。

运行 `DEVICE=0 bash run_m24_native.sh`：自动准备隔离环境、核查原validation 70图/109框，先10图冒烟，再全量原生验证。`MAX_IMAGES=10`只冒烟；`bash run_m24_native.sh --describe`查看协议，不安装或推理。依赖下载默认官方PyPI，国内镜像可设 `PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple`。

模型为 `keremberke/yolov8m-table-extraction`，使用本地官方 `best.pt`，固定Ultralytics 8.0.21原生DetectionValidator（含数据加载、NMS、坐标还原、AP）；不沿用旧predict+COCOeval评分。作者数据集 `keremberke/table-extraction` 的validation保持原图及109框，COCO格式只通过官方convert_coco转成YOLO标签；类别0/1不变，逐框反算误差需小于0.02像素。图像仅改文件名为原image_id，字节不变。

权重记录8.0.21、640、conf=.001、iou=.7、max_det=300、**v5loader=True**、训练batch=2。使用原生v5loader及rect/pad=.5；验证batch=4由该版本trainer的训练batch×2推导，CUDA FP16遵循训练中验证的强制half逻辑。作者完整评分命令未公开，checkpoint中的standalone half=False与训练验证FP16需区分，不能宣称完全历史复现。底座复用Python 3.9、torch2.2.2+cu121、torchvision0.17.2+cu121，另建 `.venv-m24-ultralytics8021`，不修改共享环境；不需要ultralyticsplus的远程模型加载器。旧版配置在导入阶段隔离到结果目录，禁用遥测，不修改原生评估源码。

新结果位于 `__MODEL_ROOT__/resources/runtime/assets/workspace/results/m24_native8021_v1/run_时间戳/`；保留模型/数据/源码哈希、完整参数、原生预测和逐类指标。冒烟标记partial_complete，不生成正式metrics。全量AP50参考模型卡0.95194（95.194%）。旧 `object_detection_official_v2/m24-yolov8m-table-extraction` 的AP50=92.0131132%继续保留为历史。

## 2026-09-14：18、20、21号原生验证对照

**基准确认（2026-09-14）**：用户确认以下三次全量运行作为后续汇报和国产GPU适配的主结果。

| 编号 | 固定运行目录（均在 `__MODEL_ROOT__/resources/runtime/assets/workspace/results/native_18_20_21_v1/` 下） | 主指标 | 官方参考 | 差值（百分点） |
|---|---|---|---|---|
| 18 | `m18/run_20260914_184422_569626` | 原生AP50 **99.0120992%** | 98.8298275% | +0.1822717 |
| 20 | `m20/run_20260914_184546_498917` | COCO AP50–95 **38.5319503%** | 38.5% | +0.0319503 |
| 21 | `m21/run_20260914_185133_806340` | COCO AP50–95 **44.9714143%** | 44.9% | +0.0714143 |

18号为validation全量1765图；20、21号均为COCO val2017全量5000图。样本ID数量和预测SHA256核查通过。
后续对照固定各次`status.json`的权重、数据、运行参数和版本；`metrics.json`与`predictions.json`保留原样。
清单`native_18_20_21.json`的`accepted_baseline`提供可机器读取的路径与指标口径。
旧18号COCO AP50 98.4137%、20号COCO AP38.2178%、21号COCO AP44.1905%仅保留历史，不再作为主结果。
18号新COCO指标和20、21号框架内部原生指标仍保留作附加对照，不与上表主指标混用。
数值标注为“接近官方”，保留以下版本/协议限制；不改写为严格官方复现，也不自动采用后续运行替换本基准。

入口 `run_native_18_20_21.sh`，清单 `native_18_20_21.json`。只封装框架原生验证器，
不自行实现模型前向、NMS、匹配或AP。复用已有环境，不安装/升级依赖。

```bash
cd __MODEL_ROOT__/resources/runtime/assets/workspace/official_object_detection_eval
# 先检查三个模型原生入口（小样本指标仅诊断，不作正式结果）
MAX_IMAGES=10 DEVICE=1 bash run_native_18_20_21.sh
# 全量入口会自动先做每个模型10张测试，通过后再跑该模型全量
MAX_IMAGES=0 DEVICE=1 bash run_native_18_20_21.sh
```

`DEVICE` 默认0（未设置CUDA_VISIBLE_DEVICES时就是物理GPU编号）；15号若仍在GPU0运行，可选择空闲GPU1。
`BATCH` 默认16，可因显存限制减小，但必须保留实际批大小记录。
输出位于 `__MODEL_ROOT__/resources/runtime/assets/workspace/results/native_18_20_21_v1/m18`、`m20`、`m21`；
每次新建时间戳目录，不覆盖旧结果。日志在同级 `logs`。失败立即停止；修复后可使用对应环境
单独执行 `run_native_18_20_21.py --model-number 18|20|21 --device 1`（先完成批量入口的数据准备）。
原生验证中断后需要重跑该模型，不宣称支持逐图推理续跑。

| 模型 | 原生入口和数据 | 对照方式及限制 |
|---|---|---|
| 18 | `yolov5.val.run`，原validation 1765张/1840框 | 主指标原生AP50，对照98.8298%；附COCO AP。采用原生NMS IoU=0.6、conf=0.001、max_det=300。保留FP32及已装7.0.14；模型卡写7.0.6，不能称历史环境完全复现。 |
| 20 | 固定作者源码的`YOLOv10.val`，COCO val2017 5000张 | 主指标COCO AP，对照上游YOLOv10-N 38.5%；附框架原生AP。无额外NMS。官方示例batch=256，本地默认16，明确记录差异。 |
| 21 | `YOLO.val`，COCO val2017 5000张 | 主指标COCO AP，对照44.9%；附框架原生AP。延续4、7号原生对照的8.4.0环境、FP32；模型卡写8.0.4，未复现历史环境。 |

COCO准备复用 `run_m07_native.py` 的官方转换器及已有缓存，不复制COCO图像。
18号原COCO标签从0开始，为适配官方转换器，在转换副本中临时改为1，生成YOLO标签0；
原始标注字节不变，图像只改为数字ID文件名、不重新编码，回到COCO评分时类别仍为0。
运行时检查图片和标签、记录权重/标注/脚本/原生代码哈希、软件版本、实际参数及样本ID。
18号使用官方逐图导出函数保存原生预测；关闭该版本`val.run`硬编码COCO路径的自动评分，
避免其覆盖自定义车牌集的原生指标，再用原标注单独调用COCO API。
这些结果标为原生流程参考对照（`reference_baseline`），指标接近不自动认证完整官方复现。

## 2026-09-14：原生流程优先及7号验证入口

用户确认：有可运行的官方原生评测流程时优先沿用；适配器仅负责必要的数据布局、调度和记录，不重写已有的推理、后处理或评分算法。只有原生入口无法适用时，才记录原因及具体偏离项。

7号 YOLO11n：执行 `bash run_m07_native.sh`，默认 GPU0、batch=16、全量 COCO val2017 5000图。脚本复用 `object-detection-yolo-v2` 的 Ultralytics 8.4.0，仅补原生验证所需的 faster-coco-eval；调用官方 `convert_coco` 准备标签、`YOLO.val()` 推理及评分，并对同一份原生预测再调用标准 COCOeval 核查。原始 COCO JSON 保留 crowd 标注，YOLO标签采用官方转换规则；原生AP和COCO AP分开记录。

结果独立保存于 `__MODEL_ROOT__/resources/runtime/assets/workspace/results/m07_native840_v1/run_时间戳/`，不覆盖旧38.6615%结果。输入640、conf=0.001、iou=0.7、max_det=300、FP32、rect=True、无增强。权重记录版本8.2.100，当前保留8.4.0用于先比较流程，不能声称完整历史环境复现。

**7号基准确认（2026-09-14）**：用户确认采用本次实验作为后续汇报及国产GPU适配对照基准。COCO val2017 全量5000图、80类已完成，原生 `YOLO.val()` 预测加标准 COCOeval 的 **AP@50–95为39.3747071%，AP@50为55.2054684%**；官方参考AP为39.5%，差 **−0.1252929个百分点**，标注“接近官方”，不标注完全一致。原生内部AP为39.1984250%，单独保留作对照，不与COCO AP混用。旧38.6615%仅作历史，不再作为7号主结果。

固定基准目录：`__MODEL_ROOT__/resources/runtime/assets/workspace/results/m07_native840_v1/run_20260914_170708_896109/`。其中 `metrics.json` 保存两种评分，`status.json` 保存权重/数据/代码哈希、软件版本及完整参数，`predictions.json` 保存原生预测。后续复跑应遵循该次记录（包括batch=16），不自动将新运行替换为基准。

### 4号 YOLOv8n 原生补跑

执行 `bash run_m07_native.sh --model-number 4`。复用同一原生流程、环境和已准备的COCO数据，仅切换为本地 `Ultralytics/YOLOv8/yolov8n.pt`、参考AP 37.3%及独立结果目录 `__MODEL_ROOT__/resources/runtime/assets/workspace/results/m04_native840_v1/run_时间戳/`。默认仍为GPU0、batch=16、输入640、FP32；原生AP与标准COCO AP分开记录。4号权重元数据为8.0.0.dev0，保留当前8.4.0先核查流程差异，不声称完整历史环境复现。旧36.6908%结果及7号已确认基准均保留。入口不传 `--model-number` 时仍运行7号，原默认参数不变。

**4号基准确认（2026-09-14）**：用户确认本次原生实验为后续汇报及国产GPU适配对照基准。COCO val2017 全量5000图、80类已完成；原生 `YOLO.val()` 预测加标准 COCOeval 的 **AP@50–95为37.3624458%，AP@50为52.5777785%**。官方参考AP为37.3%，差 **+0.0624458个百分点**，标注“接近官方”，不宣称性能超越官方或完整历史条件复现。原生内部AP为37.1457155%，仅作另一评分口径对照；旧36.6908%仅保留历史，不再作为4号主结果。

固定基准目录：`__MODEL_ROOT__/resources/runtime/assets/workspace/results/m04_native840_v1/run_20260914_171537_827272/`。`metrics.json`、`status.json`、`predictions.json`分别保存指标、完整协议和预测；后续适配对照固定该次配置（Ultralytics 8.4.0、batch=16、640、FP32、rect=True、conf=0.001、iou=0.7、max_det=300、无增强），不自动用后续新运行替换基准。

### 15号 v1.1-All：补齐紧裁剪并使用原生评测

2026-09-14小样本运行修复：首次10图已完成下载、裁剪和295个权重张量校验，但评测子进程长时间处于锁等待，用户已中止。包装入口现显式采用 `multiprocessing` 的 `spawn` 启动方式，避免原生评分池在PyTorch/CUDA初始化后通过默认 `fork` 继承线程锁；官方源码、裁剪、前向和GriTS算法不变。启动方式写入 `status.json` 和 `native_command.json`；已准备数据和权重可复用，失败运行保留，新运行使用独立时间戳目录。CPU回归检查使用两张真实裁剪样本及构造的预测张量：4个spawn进程在3.59秒内完成官方GriTS，各项指标与串行结果一致；未运行模型推理，实际GPU10图仍待用户重跑。

旧v4结果使用带留白的PubTables图像、HF前向及官方GriTS组件；论文要求2像素紧裁剪，此前未落实。新入口：`MAX_IMAGES=10 bash run_m15_native_tight.sh` 先做固定10图检查；通过后 `MAX_IMAGES=0 bash run_m15_native_tight.sh` 跑完整93834图。仅运行15号，原6号、15号v4结果不覆盖。

- 直接调用官方 `scripts/create_padded_dataset.py --split test --table_padding 2`，在独立目录生成裁剪图、平移后的结构框和按官方规则筛选/平移的文字框；逐图检查坐标与图像尺寸。裁剪使用真值表格框及真值文字，因此是表格结构识别协议，不是端到端检测/OCR。原始数据保持不变。
- 使用作者原生 `bsmock/TATR-v1.1-All/TATR-v1.1-All-msft.pth`，固定revision `30e6112ea20d72cc9fb8f443e34ca119d497cfe6`，SHA256 `9213f2dc873e5a54beeb4ddd4b5406fc9e492830abe0c44bb97f35b2229cf787`。首次运行从镜像/官方源下载，保存在 `__MODEL_ROOT__/resources/runtime/assets/workspace/model_native/bsmock/TATR-v1.1-All/`；不覆盖HF权重。
- 官方源码固定 `16d124f616109746b7785f03085100f1f6247575`，每次运行保留源码及包装脚本快照。直接调用 `main.main()`，参数使用 `--mode eval --data_type structure`，沿用官方 `structure_config.json`、前向、后处理及核心GriTS。`--batch_size 1` 在官方加载器中实际为2图；最大边1000、FP32、seed=42，核心GriTS不包含附加RawLoc/DAR。
- 包装层仅负责准备、下载校验、记录与完整权重加载检查；禁用随后会被完整权重覆盖的冗余ImageNet骨干初始化，不改变前向或评分，所有加载张量必须与原生权重逐一相同。复用现有HF环境及PyMuPDF 1.21.1覆盖层，不声称已恢复作者全部历史软件版本。
- 结果位于 `__MODEL_ROOT__/resources/runtime/assets/workspace/results/m15_native_tight_v1/{smoke10,full}/run_时间戳/`；`native_eval.log`为原生实时日志，`per_table_metrics.json`为原生逐表指标，`metrics.json`为调用官方汇总函数得到的结果，`status.json`为完成状态和协议记录。小样本不能作为完整基准；全量原生流程不支持中途推理断点续跑，失败后重跑会产生独立运行目录，已准备的裁剪数据可复用。
- 交付前仅完成源码导入、3图官方裁剪/全部结构框及文字坐标检查、官方数据加载及变换检查（含旧失败样本 `PMC2845586_table_1`），未下载/加载原生权重或运行推理。裁剪与HF→原生实现一同改变，结果差异不能单独归因于裁剪；v1.1-All对应的官方参考数字仍需核对，不能使用v1.0或Pub-only的数字替代。

本目录是检测任务的一键交付包。清单仍保留 62 个模型，但严格按照更新版工作簿的绿色填充（H 列 `E2F0D9`）选择本轮对象：37 个 `ready`，其余 25 个记为 `not_selected`，默认不运行。

## 2026-09-07：两款表格结构模型补跑（当前入口）

2026-09-08续跑记录：从m06的15607张、m15的15052张断点恢复，未改v4冻结代码或已有预测。现由临时用户级服务 `codex-table-v4-resume-20260908.service` 运行，不依赖当前聊天终端；没有设置失败自动重启或开机自启。检查：`systemctl --user status codex-table-v4-resume-20260908.service`。日志仍在v4目录的`run_full.log`和每模型日志中。服务终止后可用下文`REPAIR_SCOPE=table MAX_IMAGES=0 bash run_repairs.sh`继续，运行锁会防止重复启动。

37项中已有34项全量完成：原31项，加v3的RetinaNet、Xenova DETR、YOLOE。它们不重跑、不覆盖。WildDet3D仍需SAM3访问授权（本日复查403）；授权步骤见下文。

两款TATR在v3均于第351张 `PMC2845586_table_1` 失败。原XML框面积正常，但官方 `objects_to_cells` 原地修改框，导致一行的y1大于y2；随后 `grits-all` 的附加RawLoc计算除零。v4采用官方现成的 `grits` 模式，保留Top、Con、Loc、Acc_Con，统一不计算附加RawLoc/DAR；**没有删除样本、修改指标公式或修改官方源码**。传入后处理前复制框，避免导出的原始预测框也被改写。HF权重和已核对的官方预处理不变，仍属于官方协议适配器，不能标作完整原生复现。

```bash
cd __MODEL_ROOT__/resources/runtime/assets/workspace/official_object_detection_eval
REPAIR_SCOPE=table MAX_IMAGES=400 bash run_repairs.sh  # 覆盖原失败的第351张
PYTHONPATH="$PWD/.runtime/table" /home/ubuntu/gaoyonghan/miniconda3/envs/object-detection-hf-v2/bin/python check_table_repair.py --results __MODEL_ROOT__/resources/runtime/assets/workspace/results/object_detection_official_v4_tables
REPAIR_SCOPE=table MAX_IMAGES=0 bash run_repairs.sh    # 两GPU各跑一款，93834张/款
```

结果写入独立的 `__MODEL_ROOT__/resources/runtime/assets/workspace/results/object_detection_official_v4_tables`。入口自动冻结本轮代码并复用已有解压缓存；400图验证通过后，同一目录继续全量，不重复已完成图像。v3结果及源码快照保持原样，不把其前350张记录移入v4。失败或中断后重跑最后一条命令可续跑。`partial_complete`不是全量完成。

修复验证脚本会重现旧错误，核查标准GriTS自比较为1且与扩展模式的同名核心指标一致；指定 `--results` 后还核查两款各至少400张的唯一图像ID和指标有效性，并报告与旧350张核心指标的最大差值。

本日实测：两款均通过400/400张验证，原失败的351号样本正常；与v3前350张的同名核心指标最大差值均为0。验收日志：`object_detection_official_v4_tables/validation.log`。该检查不是全量完成证明，全量状态以各模型 `status.json` 为准。

依据：[微软官方评估代码](https://github.com/microsoft/table-transformer/blob/16d124f/src/eval.py)的 `compute_metrics` / `compute_metrics_summary` 和[官方结果表](https://github.com/microsoft/table-transformer#evaluation-metrics)。逐模型记录在 `repair_protocol.json`；实际生效版本以每个结果目录的 `_source_snapshot` 为准。

## 2026-09-06：31份结果核查及6项补跑（历史v3入口）

31份旧结果不覆盖。核查报告：[逐模型对照表](__MODEL_ROOT__/resources/runtime/assets/workspace/results/object_detection_official_v2/official_reference_audit_20260906.md)，[含分项差值、来源及结果哈希的JSON](__MODEL_ROOT__/resources/runtime/assets/workspace/results/object_detection_official_v2/official_reference_audit_20260906.json)。运行 `python audit_completed.py` 可重新生成；它不修改原始指标。

核查不是统一放宽误差：11项主指标差≤0.1个百分点，16项仍有差异，4项无直接可比基准。AP50与AP50–95分开；发布仓库与论文分开；DETA的800/1333配置与作者1200/2000不一致；UI-DETR的框覆盖率不是完整点击准确率。**全量完成不等于严格复现官方。**

只补跑原先失败的m06、m15、m23、m27、m51、m53：

```bash
cd __MODEL_ROOT__/resources/runtime/assets/workspace/official_object_detection_eval
bash setup_repairs.sh
MAX_IMAGES=10 bash run_repairs.sh   # 首次检查
MAX_IMAGES=0 bash run_repairs.sh    # 正式全量；锁防止重复运行
```

新结果：`__MODEL_ROOT__/resources/runtime/assets/workspace/results/object_detection_official_v3_repairs`。每个模型有同名顶层日志，目录内有status/metrics/predictions；`partial_complete`仅表示冒烟通过。2026-09-06本次smoke1已验证5个模型各10张；WildDet3D缺辅助权重，不能计入通过。YOLOE额外验证了官方类别映射和Fixed-AP子程序，其10图对全GT的冒烟分数不作为正式结果。

修复依据与边界：

逐模型机器可读记录见 [repair_protocol.json](__MODEL_ROOT__/resources/runtime/assets/workspace/official_object_detection_eval/repair_protocol.json)。它明确补充本次实际预处理/计分方式；历史models.json保留原貌，不能将其旧的通用processor描述当作本次修复的最终协议。续跑入口优先采用结果目录已有的源码快照，并追加日志。

- 两个TATR结构模型：直接调用固定微软源码的 `table_datasets.get_structure_transform("val")`（最大边1000），继续调用原版GriTS；PyMuPDF按作者environment.yml固定1.21.1，仅放入 `.runtime/table` 隔离覆盖层，其他环境不降级。权重仍为已下载HF转换版；不是换成原生权重。
- YOLOE：仍运行作者prompt-free validator、MobileCLIP类别映射和Fixed-AP。显式设置子进程的官方LVIS包路径及环境Python路径，防止tools/lvis目录遮蔽包、子程序调用错Python。按官方copypaste列名读取AP/AP50/AP75，缺最终AP时拒绝标为完成。
- RetinaNet：`retinanet_official.py`原样保留模型发布者Space的五个推理定义；固定来源revision及原文件SHA写在文件头，AST一致性已核对。SavedModel原始84维输出由官方AnchorBox/DecodePredictions解码；先resize/pad后ResNet预处理；坐标按作者XYXY返回。阈值采用作者DecodePredictions默认0.05（网页演示0.5只用于显示），计分仍为本项目COCOeval，不冒称论文原生checkpoint的AP。
- Xenova DETR：按发布者Transformers.js 2.17.2的DetrFeatureExtractor，使用int64的64×64全1 pixel_mask；图像仍按模型配置800/1333预处理，**不是缩小图像为64×64**。Python与JS图像实现并未声称逐位一致。
- WildDet3D：修复Vis4D子进程PYTHONPATH，保留作者 `configs/eval/in_the_wild/text.py`，不改数据、类别或几何协议。原配置还需SAM3与LingBot-Depth辅助权重；本机SAM3下载实测403，需要账号获得作者授权。目前完整原生流程尚未验证，不能宣称已跑通。

WildDet3D授权完成后，在终端（不要在聊天里发送Token）运行：

```bash
PYTHONNOUSERSITE=1 /home/ubuntu/gaoyonghan/miniconda3/envs/object-detection-wilddet3d-v2/bin/huggingface-cli login
PYTHONNOUSERSITE=1 /home/ubuntu/gaoyonghan/miniconda3/envs/object-detection-wilddet3d-v2/bin/python prepare_wilddet_weights.py
MAX_IMAGES=0 bash run_repairs.sh
```

下载脚本固定两份辅助权重的revision及LFS SHA256，验证后仅创建官方配置所需的路径链接，不覆盖不同的已有文件。SAM3约3.45GB，LingBot约1.28GB。来源：[WildDet3D官方数据与预训练说明](https://github.com/allenai/WildDet3D/blob/main/docs/TRAINING_DATA.md)、[模型发布者RetinaNet推理源码](https://huggingface.co/spaces/keras-io/Object-Detection-Using-RetinaNet/blob/0798aa064c15a1f7ea87de4ededa529de28890ff/app.py)、[Transformers.js处理器](https://github.com/huggingface/transformers.js/blob/2.17.2/src/processors.js)。

旧31份结果对应代码已保存到旧结果根目录的 `_source_snapshot`；新补跑代码也在新根目录 `_source_snapshot` 冻结。**不要用修改后的run_all.sh向旧v2目录重新跑37个**：指纹会拒绝混用。下文run_all用于新建整批实验时，必须指定全新的 `OUTPUT_ROOT`；复查旧结果使用其代码快照。

默认结果写入新目录：

```text
__MODEL_ROOT__/resources/runtime/assets/workspace/results/object_detection_official_v2
```

旧的 `object_detection_official` 不会被覆盖或混用。

## 一键运行

先跑 10 张冒烟：

```bash
cd __MODEL_ROOT__/resources/runtime/assets/workspace/official_object_detection_eval
MAX_IMAGES=10 bash run_all.sh
```

确认冒烟结果后，全量断点续跑：

```bash
cd __MODEL_ROOT__/resources/runtime/assets/workspace/official_object_detection_eval
MAX_IMAGES=0 bash run_all.sh
```

`run_all.sh` 默认使用 GPU 0、1，并在首次执行时自动创建隔离环境、下载官方源码。常用覆盖：

```bash
GPU_IDS=0 MAX_IMAGES=10 bash run_all.sh
GPU_IDS=2,3 MAX_IMAGES=0 AUTO_SETUP=0 bash run_all.sh
```

首次环境准备也可单独执行：

```bash
bash setup_env.sh
```

这个入口只安装、检查环境，不启动推理。已有环境也会检查必要模块、官方源码包的实际来源和 GPU 可见性；只有检查通过才写完成标记。`check_environment.py` 不加载权重、不读取数据集。

## 为什么不是一个 Python 环境

这 37 个模型跨越 Transformers、当前 Ultralytics、旧 YOLOv5、官方 YOLOv10、YOLOE、YOLOv13、RF-DETR、TensorFlow、ONNX Runtime 和 WildDet3D。旧 YOLOv5 依赖的 `huggingface-hub` 与 Transformers 4.50 的最低版本要求直接冲突；YOLOv13 又需要自己修改过的 Ultralytics 模块。因此调度统一，但运行环境隔离。

2026-09-05 排查记录：YOLOv5 环境曾读入 `/home/ubuntu/.local/lib/python3.10/site-packages` 中的 Transformers 5.16.1，与本环境的 `huggingface-hub==0.24.7` 冲突，导致安装末尾的 `pip check` 失败，尚未进入推理。`setup_env.sh` 和 `run_all.sh` 现均设置 `PYTHONNOUSERSITE=1`，阻止用户级包混入；不删除或降级公共目录中的包（[Python 官方说明](https://docs.python.org/3.10/using/cmdline.html#envvar-PYTHONNOUSERSITE)）。已有完成标记的环境也要重新通过隔离后的 `pip check` 才能跳过安装；曾借用公共目录的依赖会由安装流程补到各自环境内。直接运行 Python 脚本时也需加 `PYTHONNOUSERSITE=1`，或使用这里的一键入口。

YOLO 和 ONNX 环境的 `onnxruntime-gpu` 固定为 `1.20.2`，替代当前包源缺少可用安装包的 `1.20.1`。已核对其 Python 3.11/Linux x86_64 wheel；1.20.x 对应 CUDA 12.x/cuDNN 9.x。包版本会随运行状态记录，用于后续结果核查（[PyPI 发布文件](https://pypi.org/project/onnxruntime-gpu/1.20.2/)、[官方 CUDA 兼容表](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html#cuda-12x)）。

2026-09-05 专用环境补充检查和修正：

- YOLOv10：已有完成标记的环境实际装入 NumPy 2.0.2，与 PyTorch 2.2.2 的 NumPy 接口不兼容。将本地源码安装和 NumPy 1.26.4 等约束合并为同一次解析，避免后一步重新升级 NumPy；完成检查增加版本核对和 PyTorch / NumPy 数据转换验证。
- YOLOE：官方 CLIP 的 `setup.py` 导入 `pkg_resources`，已在安装预演中复现新版 setuptools 下的失败。构建工具固定为 setuptools 79.0.1 / wheel 0.45.1，本地源码使用 `--no-build-isolation`；主仓库、LVIS Fixed、MobileCLIP、CLIP 仍按官方 `requirements.txt` 一起安装。OpenCLIP 2.26.1 与 MobileCLIP 要求的 timm 0.9.5 一并解析，防止后续安装再次改动核心依赖。额外 MobileCLIP 权重先写 `.part`，下载成功才改为正式文件名。
- YOLOv13：不再用 `--no-deps` 漏掉本地 Ultralytics 声明的依赖；将官方 Python 包与既定版本约束放在同一次解析中，并补上 opencv-python 4.9.0.80。保留官方代码已有的 PyTorch SDPA 回退，不安装其 CUDA 11 FlashAttention wheel，不宣称与 FlashAttention 路径逐位相同。
- RF-DETR：固定 Transformers 4.50.0、PEFT 0.15.1、Hub 0.29.3，避免无限制的 PEFT 更新引入新的 Transformers 接口要求。RF-DETR 仍为 1.5.2。
- TensorFlow：使用官方 `tensorflow[and-cuda]==2.15.1` 补齐该版本声明的 CUDA 12.2 / cuDNN 8.9 依赖，不改系统驱动（[TensorFlow 安装说明](https://www.tensorflow.org/install/pip)）。
- ONNX：补齐独立环境自己的 CUDA 12.1 / cuDNN 9.1 运行库。检查会加载 CUDA provider 共享库，而不只查看 provider 名称；正式运行若请求 GPU 却退回 CPU，将明确失败。
- WildDet3D：沿用官方安装顺序和 Vis4D 1.0.0；NumPy、Transformers、Hub、timm 与 Torch 增加兼容版本约束，在前后安装步骤中保持一致。CUDA 扩展编译前检查 GPU / Toolkit，默认最多 4 个编译进程。源码 CUDA 扩展仍须在实际安装时编译并验证。

以上是本项目的兼容约束，并非作者发布的完整环境锁文件。安装和一键推理均优先使用对应环境内的 NVIDIA 动态库目录，避免不同 cuDNN 主版本串用。安装预演通过仅表示依赖可解析；全部环境是否可用，以实际 `setup_env.sh` 完成 GPU 和导入检查为准。

脚本创建 10 个环境：

```text
object-detection-hf-v2
object-detection-yolo-v2
object-detection-yolov5-v2
object-detection-yolov10-v2
object-detection-yoloe-v2
object-detection-yolov13-v2
object-detection-rfdetr-v2
object-detection-tensorflow-v2
object-detection-onnx-v2
object-detection-wilddet3d-v2
```

官方源码首次克隆到：

```text
__MODEL_ROOT__/resources/runtime/assets/workspace/official_sources/object_detection
```

五个官方仓库都固定到 `models.json` 记录的 commit，不跟随 `main` 漂移。如果已有仓库含未提交修改，脚本会停止并保留现场；否则切到固定 commit。实际 commit 和每个环境的包版本写入 `status.json`，也进入运行指纹；版本变化后脚本会拒绝续写旧结果。

源码克隆最多自动重试 3 次。如果上次使用旧脚本的 partial clone 在 checkout 中断，且目录内只有 Git 元数据和空目录、远程地址又与本项目一致，新脚本会安全清理这个残留并完整重下；真实源码修改不会被删除。

WildDet3D 改为直接从 GitHub 官方 `codeload.github.com` 下载固定 commit 的源码归档，避开当前频繁超时的 Git 传输。`download_official_source.py` 同时准备主仓库锁定的 SAM3、LingBot-Depth 子模块，以及安装时原本要通过 Git 下载的 `vis4d_cuda_ops` 和 `utils3d`。原始 requirements 只在传给 pip 时把 utils3d 地址替换为对应本地源码，不修改官方文件。

归档保存在源码目录的 `.archives` 缓存；每个源码目录的 `.official-source.json` 记录完整 commit、官方 URL、归档 SHA-256 和逐文件校验值。重跑会核查并复用这些文件。GitHub 归档不包含子模块，因此不能只手动解压一个主仓库 ZIP 就当作准备完成。WildDet3D 的五个源码包版本也会写入推理状态记录。

可先仅准备源码（不安装环境、不运行推理）：

```bash
bash setup_env.sh --sources-only
```

此下载方式仍需系统能访问 `codeload.github.com:443`。连接或下载失败会在有限重试后退出；已完成的源码包可复用。

## 协议分层

- 普通 HF/YOLO/ONNX/Keras 模型：沿用模型自己的预处理和后处理，再按冻结的数据集类别映射调用 COCO API。
- Table Transformer Detection：PubTables-1M 官方 test，COCO bbox AP。
- Table Transformer Structure：调用微软官方 `src/eval.py` 计算 GriTS。
- YOLOE：`yoloe-11s-seg-pf.pt` + `yoloe-11s-seg.pt`，作者 minival 4809 图，官方 prompt-free 和 LVIS Fixed AP。
- WildDet3D：冒烟调用作者公开推理 API；全量调用官方 Vis4D `in_the_wild/text.py`。
- YOLOv13：模型和推理来自官方 iMoonLab 代码；本批统一用 COCO API 计分，作者表中 41.6 来自仓库内 Ultralytics validator，因此保留参考值但标记为不严格可比。
- UI-DETR-1：按模型卡使用 RF-DETR-M、1600 分辨率、阈值 0.35。模型卡的 70.8% 还包含 Gemini 2.5 Pro 选择编号框；本地只报告检测器的目标覆盖率，明确标记为不可直接比较。

`models.json` 对每个模型记录权重文件、数据集、split、输入尺寸、阈值、最大检测数、evaluator、官方参考值、特殊情况和环境。

## 数据准备说明

PubTables 的图像、XML 和 table words 是压缩 TAR。第一次遇到对应模型时会一次性解压到结果根目录下的 `_prepared_datasets`；即使只跑 10 张，首次准备仍需要较长时间和额外磁盘空间。之后自动复用并校验源文件大小。

YOLOE 同样会把 LVIS 标签和 COCO val2017 图像准备到 `_prepared_datasets`。这些是运行缓存，不是项目源码文件。

## 查看和核查

```bash
PY=/home/ubuntu/gaoyonghan/miniconda3/envs/object-detection-hf-v2/bin/python
$PY run_model.py --list-models
$PY run_model.py --describe m23-yoloe
$PY run_model.py --preflight --paths-only
$PY run_model.py --summarize
```

路径预检不会加载 GPU 权重。单个模型要使用 `models.json` 中指定的环境；批量脚本会自动选择，不建议手工用其他环境运行。

## 输出和断点

每个模型目录包含：

```text
status.json             状态、完整协议快照、环境版本、源码 commit、异常和指纹
predictions.jsonl       普通适配器逐图断点（适用时）
predictions.json        evaluator 输入（适用时）
metrics.json            全量指标、官方参考值与可比性判定
official_evaluator.log  原生 evaluator 完整输出（适用时）
native/                 官方脚本自己的输出（适用时）
```

冒烟状态是 `partial_complete`，不把 10 张结果当正式 AP；`MAX_IMAGES=0` 才计算正式指标。失败后直接重跑相同命令即可。任务根目录的 `audit_summary.json` 同时列出 37 个运行对象和 25 个未选择对象。

工作簿来源：

```text
__MODEL_ROOT__/resources/runtime/assets/workspace/outputs/second_batch_status_update_20260902/100个模型及数据集下载状态_第二批完成更新版.xlsx
```
