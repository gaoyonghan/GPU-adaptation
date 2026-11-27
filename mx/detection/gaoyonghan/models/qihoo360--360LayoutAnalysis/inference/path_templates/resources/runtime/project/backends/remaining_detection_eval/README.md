# 原未标绿25个检测模型的补充评估

## 54号原权重＋SKU-110K官方挑战评分（2026-09-15，待终端验收）

```bash
cd __MODEL_ROOT__/resources/runtime/assets/workspace/remaining_detection_eval
DEVICE=0 bash run_m54_sku_official.sh
```

先10张推理及原生评分，通过后自动全量2936张。仅10张：`MAX_IMAGES=10 DEVICE=0 bash run_m54_sku_official.sh`。CPU全量图片/权重/源码预检：`bash run_m54_sku_official.sh --preflight`。复用 `.venv-m12-ultralytics8043`（Ultralytics8.0.43、ultralyticsplus0.0.28），不安装依赖，不重新下载模型或数据；原版3.8KB官方评分ZIP已随脚本保存，启动时校验SHA256并复制到运行目录，无需联网。此项修复首次启动下载超时，未更改官方算法。

- 保留54号 `foduucom/product-detection-in-shelf-yolov8/best.pt`，SHA256 `aac44618866c15502f8158778c1b2ae0e8f66b57a9a7bce318b269c7b283b148`。不替换为SKU110K作者的RetinaNet，不使用其EM-Merger。
- 现有官方 `SKU110K_fixed` 公开test：2936图、431546商品框，原 `annotations_test.csv` 的坐标及全部标注保留，不去重、不合框、不按最小尺寸删除。仅编码预测CSV为 `image_name,x1,y1,x2,y2,confidence`（无表头）。模型class1 product对应唯一评分目标；class0 empty预测记录但不评分，不能当作商品或假装覆盖了空位。
- **评分来源**：SKU110K作者仓库链接的Retail-Vision CVPR2020挑战官网源码，固定提交 `9303b450077244fb8c7729717b67c2da6c807819` 的 `_data/densely_packed_eval_2020-02-06.zip`。包SHA256 `2db06f173154650e8a182ad9a905a7e3b7b8e21fd9c829f2367d946fa77a603f`。直接调用未修改的 `demo.eval_results`、`eval.eval_single_threshold` 和CSV读取器；源码快照及哈希随结果保存。
- 该官方包自带demo指向公开 `annotations_test.csv`；本次是“官方挑战评分算法用于公开test”，**不是隐藏比赛测试集成绩**，也不是模型卡91%的复现。作者另有 `eg4000/SKU110K-benchmark` 的COCO分支，这次选用官方挑战发布包，不混用两种AP。
- 评分遵循源码：IoU面积使用包含端点的 `+1`，匹配为严格 `IoU > threshold`，VOC式精确率包络积分；主AP取 `.50:.05:.95` 平均，另报告AP50/AP75；没有COCO每图100框上限。不把此AP误标为标准COCO AP。
- 推理使用原生YOLO8.0.43，640、FP32、batch1、无增强、类别独立NMS，conf=.001、iou=.7；最多1000框（采用模型卡展示代码的容量上限，不声称这些参数就是91%的评测配置）。旧流程为8.4.0、最多300框、COCO maxDets100，因此新旧结果不能解释成仅更换评分器的影响。
- 原图不改写。PIL RGB解码后转BGR数组交给原生YOLO；只有明确的JPEG truncated异常启用容错解码，逐图记录，不跳图。旧记录中的 `test_274.jpg` 有截断异常，需作为数据质量限制保留。
- 预测框边界检查：8.0.43原生predict会裁剪并取整坐标，可能产生x1=x2或y1=y2；官方挑战评分使用包含端点的+1面积，故原样保留并统计这些框，不过滤、不扩大、不修改评分源码。2026-09-15首轮在第6张触发过项目严格正宽高断言，未保存触发框，具体原因待新日志确认；现已按评分器语义修正该检查，反向坐标、非有限值、非法类别/分数仍报出图片名及完整框值。
- 输出：`__MODEL_ROOT__/resources/runtime/assets/workspace/results/m54_sku_official_v1/full/run_时间戳/`，含原始两类预测、送评商品CSV、GT CSV、metrics、status、protocol和官方源码快照；10图结果单独标为smoke。日志在根目录logs；推理结束后CPU官方逐阈值评分仍可能耗时数分钟，status显示scoring。旧 `remaining25_v5/m54-product-detection-in-shelf-yolov8/sku110k` 完整保留，不自动更改基准。

## 48号NVIDIA原生入口复跑（2026-09-15，待终端验收）

```bash
cd __MODEL_ROOT__/resources/runtime/assets/workspace/remaining_detection_eval
DEVICE=0 bash run_m48_native.sh
```

先10张验证并计分，通过后自动全量560张。仅10张：`MAX_IMAGES=10 DEVICE=0 bash run_m48_native.sh`；CPU预检：`bash run_m48_native.sh --preflight`。复用已有 `object-detection-eval` 环境，不安装、不下载。CPU预检已验证全部图片、标注、权重和源码哈希，并通过官方入口严格加载权重；未在Codex执行模型前向。

- 原权重及源码：`nvidia/nemotron-graphic-elements-v1`，固定版本 `4a76546bb1bb4cbab3401361c91cf01706321805`。直接调用未修改的 `define_model("graphic_element_v1")`；以输出目录内的离线HF缓存软链接复用原权重，不复制权重、不修改官方加载函数。
- 数据：已恢复的公开CACHED PMC `element_detection/val.json` 对应560张、11520框、10类；各类图数及框数与模型卡一致。复用已审计的10类COCO标注，不裁图、不合框、不改变原始坐标（包括12个越界GT框）。
- 原生推理：官方RGB预处理、1024尺寸、FP32、batch1；官方NMS默认conf=.01、IoU=.25、class-agnostic，以及官方额外后处理阈值.1。仅将输出坐标/类别编号编码成COCO格式，不另加NMS或裁框。
- **评分来源与限制**：已核查的NVIDIA仓库提供最小推理代码，未找到生成模型卡成绩表的完整评估程序。因此仍调用标准 `pycocotools.COCOeval`，报告bbox AP@.50:.95、AP50、AR100及10类明细；不能将其写作“完整官方评估复现”，也不保证与模型卡数字相同。协议文件明确标注 `official_comparable=false`。
- 旧流程已经手动构造相同的Exp/model、严格加载相同权重并使用相同官方wrapper和后处理；本次改用原生工厂入口是独立核验，并非已确认旧结果差距由加载入口造成。
- 新输出：`__MODEL_ROOT__/resources/runtime/assets/workspace/results/m48_native_factory_v1/full/run_时间戳/`，含metrics、逐类指标、coverage、protocol、源码快照、逐图预测及status。10图输出在smoke目录，日志在根目录logs。每次新建目录，不覆盖旧 `remaining25_supplemental_20260905/m48-cached-val560-native10` 及raw-wrapper诊断结果；不自动更改已记录的基准。

## 47号FLIR八类原生复跑（2026-09-15，待终端验收）

```bash
cd __MODEL_ROOT__/resources/runtime/assets/workspace/remaining_detection_eval
DEVICE=0 bash run_m47_native.sh
```

先10张验证，再自动全量1144张。仅10张：`MAX_IMAGES=10 DEVICE=0 bash run_m47_native.sh`；CPU全量预检：`bash run_m47_native.sh --preflight`。复用已有 `.venv-m12-ultralytics8043`，版本与47号权重训练记录一致：Ultralytics 8.0.43（ultralyticsplus 0.0.28）。不安装、不下载，不在Codex执行模型前向。

- 原权重：`foduucom/thermal-image-object-detection/best.pt`；实际是YOLOv8n、14输出类别，保持权重及全部类别编号不变。`empty`含义未确认，不当背景目标评分。
- 数据：现有FLIR `images_thermal_val/coco.json` 全部1144张热成像验证图，未混入RGB。选择person/bicycle/car/motorcycle/bus/truck/traffic light/fire hydrant八类。FLIR bike/motor/light/hydrant分别对应bicycle/motorcycle/traffic light/fire hydrant；相应编号保持1/2/3/4/6/8/10/11。一般sign不冒充stop sign；其他类别从本次GT中排除并记录。
- 只做必要的数据编码：原图字节复制至独立运行缓存；COCO xywh转YOLO归一化坐标，并验证还原坐标误差小于1e-5像素，不裁框、合框或更改原数据。全部14个模型输出保持，原生AP仅对有GT的8类求平均；未覆盖类别不记0分，原生保存的预测仍可能包含这些类别。
- 调用原版 `DetectionValidator`；透明子类只记录实际处理图片ID，随后原样调用父类，不修改预处理、NMS、匹配或AP。使用 `get_cfg` 构造旧版所需参数对象。固定640、batch1、FP32、conf=.001、iou=.7、max_det=300、rect=True、无增强；阈值与权重记录一致，不把模型卡单图展示阈值当评测参数。
- **原生去重记录**：选定GT原有14152框，图5808存在3条、图5825存在5条完全重复标注。保留这些原始行，由Ultralytics自带加载器去重，不自行删除原文件。实际评分14144框：person4470、bicycle170、car7128、motorcycle55、bus179、truck46、traffic light2002、fire hydrant94。日志中的两条“duplicate labels removed”是该已审计行为，不是运行失败。
- 1144图均保留，其中43图没有本次八类GT，仍参与评测，不把它们剔除。此处“背景图”仅指八类子集无标注，不保证整图没有其他对象。
- 新输出：`__MODEL_ROOT__/resources/runtime/assets/workspace/results/m47_native8043_flir8_v1/full/run_时间戳/metrics.json`；另有protocol、源图/标签/权重哈希、源码快照、逐类指标、覆盖记录及原生predictions.json。日志在根目录logs。10图仅保存smoke_metrics.json；重复运行新建时间戳目录，不覆盖旧 `remaining25_v2/m47-thermal-image-object-detection/bbox`。

本次为“原权重＋FLIR八类替代集＋训练版本框架原生评估”，不声称FLIR作者完整评估协议或模型卡74.3% precision的复现。类别从6增至8，且框架版本/评分/去重等变化，不能将新旧整体AP直接作差解释成精度变化。全部1144图原生CPU加载、14144框计数及CPU权重检查已通过，正式GPU结果待终端运行。

## 46号原生Ultralytics替代集复跑（2026-09-15，待终端验收）

```bash
cd __MODEL_ROOT__/resources/runtime/assets/workspace/remaining_detection_eval
DEVICE=0 bash run_m46_native.sh
```

先10张冒烟，再自动全量703张；仅冒烟用 `MAX_IMAGES=10 DEVICE=0 bash run_m46_native.sh`。使用已有 `object-detection-yolo-v2` 环境，不下载/安装依赖，不修改原权重或原始数据。只做CPU预检：`bash run_m46_native.sh --preflight`；在临时目录展开测试数据，严格校验权重/压缩包SHA、全量图片和标签，并通过原生验证数据加载器读取全部703张，临时数据自动清理，不执行前向。

- 权重：`TommyNgx/YOLOv10-Fire-and-Smoke-Detection/best.pt`，CPU加载已确认是两类YOLOv10端到端模型（v10Detect），不是根据仓库名称猜测架构。
- 数据：ISFire-4原test，703图，Fire=543框、Smoke=535框。蓝火/黄橙火→Fire，黑烟/白灰烟→Smoke；只替换YOLO标签的类别ID，坐标字符串和图像字节保持不变，不合框、不重新划分。原始ZIP保留。该测试划分没有纯背景图，不能据此全面衡量无火场景误报率。
- 入口：直接 `YOLO(best.pt).val()`，不替换validator，不重写AP，不经项目COCO转换；官方回调只读取最终验证器供核对图数、类别、框数。原生读取/预处理/YOLOv10后处理/匹配/AP均使用Ultralytics。设置640、FP32、batch1、conf=.001、iou=.7、max_det=300、rect=True、无增强，保留端到端路径，不加项目NMS。IoU参数不意味着端到端模型会执行普通NMS。
- 版本限制：复用Ultralytics **8.4.0**。权重记录训练版本 **8.2.59**，模型目录演示requirements写 **8.2.56**，三者不混称。本次属于框架原生评估，不是作者历史环境完整复现；更换评分实现、原生预处理和预测数量上限等可能共同影响新旧分数，不能直接认定单一原因。
- 输出：`__MODEL_ROOT__/resources/runtime/assets/workspace/results/m46_native_v1/full/run_时间戳/metrics.json`；对应目录保存protocol、图像/标签来源哈希、源码快照、原生predictions.json、逐类指标和覆盖统计。10张仅保存smoke_metrics.json，不作为正式成绩。日志在根目录logs；同一指令再次运行会新建实验目录，不覆盖旧结果，不承诺图级断点续跑。
- 旧 `remaining25_v2/m46-yolov10-fire-and-smoke-detection/bbox` 结果完整保留，本次新成绩待用户执行及确认，不预先更改已汇报基准。

**来源勘误**：[作者模型附带app链接的GitHub](https://github.com/X-Men01/YOLOv10-Fire-and-Smoke-Detection) 实际提供清洗后原数据的Google Drive下载链接；之前仅核查模型卡而写“没有原数据”不完整。链接是否仍可下载、是否与本地best.pt及卡中85%对应的切分完全一致尚未核实，本次不扩展下载或切换数据。[ISFire-4发布页](https://huggingface.co/datasets/shahriar-5/ISFire-4) 和本地包提供YOLO数据及data.yaml，未取得其独立评估程序。因此只能称“ISFire-4两类归并＋Ultralytics原生评估”，不能称“ISFire-4作者原生协议”或把结果直接对比模型卡85%。

交付检查：CPU权重加载、703图原生加载、1078框计数、无损标签映射和shell语法均通过；未在Codex执行GPU推理，10张模型前向验证由上述指令在终端首先执行。

## 39号沿用Omni官方检测流程（2026-09-14，待运行验收）

```bash
cd __MODEL_ROOT__/resources/runtime/assets/workspace/remaining_detection_eval
DEVICE=0 bash run_m39_omni.sh
```

先10页推理和官方评分，通过后自动继续全1651页；只试10页用 `MAX_IMAGES=10 DEVICE=0 bash run_m39_omni.sh`。复用已有 `object-detection-hf-v2`＋本包 `_ppdeps` 推理环境和 `omnidocbench` 评分环境，不新增下载或安装。CPU预检：`bash run_m39_omni.sh --preflight`，检查所有图像/模型/源码哈希、原生类别并严格加载CPU权重，不执行前向。

- 模型：本地 `PaddlePaddle/PP-DocLayoutV3_safetensors`，沿用官方HF processor/model/postprocessor，800×800、FP32、batch=1。阈值0.001是记录在案的项目AP配置，不宣称论文原参数。
- 评分：复用38号已接受结果的固定Omni提交、7类配置、GT映射和原官方检测指标实现。每次核验源码SHA、软件版本、配置；全量GT须与38号相同（1651页、30133框）。上文/下文记录的官方映射拼写、ignore行为等限制同样适用，不暗中修正。
- 必须重新推理：旧39号文件只保留合并后的类别，原生ID及公式编号等已丢失。新脚本按原始数值ID与模型自带 `inference.yml` 对应，保留全部25类预测以及 `config.json` 别名；ID5独立公式与ID15行内公式不再因同名formula混在一起。评分仅选上述7类，排除类别单独统计，不合并检测框。
- 沿用原已审阅的空/退化mask转矩形兜底补丁，源码哈希冻结；此补丁不改返回的框/分数/类别。本轮只给bbox AP，不重评掩码或阅读顺序。该后处理器的mask与候选分数共用0.001阈值，保存的多边形仅供诊断。
- 原生预测：`__MODEL_ROOT__/resources/runtime/assets/workspace/results/m39_omni_native_v1`；正式评分：`__MODEL_ROOT__/resources/runtime/assets/workspace/results/m39_omni_official_v1/full/run_时间戳/metrics.json`；启动日志在后者根目录的 `logs`。同协议可断点续跑，旧 `remaining25_pp_v2/v3` 保留，不把旧结果当新流程结果。

来源澄清：[V3模型卡](https://huggingface.co/PaddlePaddle/PP-DocLayoutV3_safetensors) 链接的 [RT-DocLayout论文](https://arxiv.org/html/2606.23344v1) 确实使用OmniDocBench与Real5-OmniDocBench公开基准，不能记录成“作者没有公开评测集”。本次属于同一公开基准系列上的检测子集协议，尚非论文结合下游识别器的文档解析综合成绩复现；本地数据版本及7类配置单独固定，不以论文综合分数与bbox AP直接比较。

交付前已通过CPU文件/权重预检、数值ID与公式类别序列化检查，以及合成预测的10页官方评分链路测试（测试文件仅在 `/tmp`，不计实验结果）。未在Codex运行GPU推理，39号新正式成绩及是否采用仍待用户运行确认。

## 38号改用Omni官方检测流程（2026-09-14，最新要求）

用户明确：以Omni支持的检测类别/流程评测，不再要求本模型25类全部覆盖。保留旧原生7类与合并11组结果作历史对照，不把缺失25类作为本次流程验收阻塞。

**2026-09-14用户确认采用本次替代集结果作为38号检测基准。** 固定结果目录：`__MODEL_ROOT__/resources/runtime/assets/workspace/results/m38_omni_official_v1/full/run_20260914_232048_399395`；1651页、30133个标注框，Omni配置7类，mAP@0.5 **50.74184752451487%**、mAP@0.5:0.95 **40.93418445776538%**。后续汇报优先引用这份结果，不再以旧46.30%作为38号主结果。验收范围仅为本次版面检测子集，不代表25类完整覆盖或阅读顺序指标一并验收；官方自建集81.4%仍只作不可直接比较的参考。旧推理、结果、脚本全部保留，已知配置限制见下文。

```bash
cd __MODEL_ROOT__/resources/runtime/assets/workspace/remaining_detection_eval
bash score_m38_omni_official.sh
```

只使用CPU评分，复用 `m38_omni_native_v1/native_predictions.jsonl` 中完整1651页的原生预测，**不重新推理，不调整阈值或NMS，不下载或重建环境**。复用已有 `omnidocbench` 环境（mmeval 0.2.1、pycocotools 2.0.7、numpy 1.24.4、PyYAML 6.0.2）。先10页验证官方评分，再对1651页全量评分。新结果独立写入 `__MODEL_ROOT__/resources/runtime/assets/workspace/results/m38_omni_official_v1/full/run_时间戳/metrics.json`，旧结果不动。`bash score_m38_omni_official.sh --preflight` 只检查全量输入和官方数据读取，不计算指标。

### 哪些是官方代码，哪些是项目配置

- 从本地官方Git仓库中提取固定提交 `85f760aa60e5f9f087e3f582109f55277937a565`（2025-11-12）的检测必需源码，不使用本地未提交改动，也不升级到最新重构版本。此提交的数据结构经本地1651页实测兼容，不宣称数据本身等于某一最新发行版。
- 原样调用 `DetectionDataset → DetectionEval → MMEval COCODetection`。官方任务仅打印返回值，项目透明包装原指标对象保存该返回值，不改变输入或计算逻辑。未调用 `pdf_validation.py` 的全任务导入入口，避免引入OCR/公式识别等无关依赖；实际检测读取、匹配、AP计算均来自原官方代码。
- 项目仅负责把已有原生xyxy预测转换成Omni接受的矩形poly格式、配置路径/类别映射、保存来源和结果。不合并框、不改GT、不重写评分算法。
- 使用官方 `detection_dataset`（Omni原生JSON格式），而不是官方示例的 `detection_dataset_simple_format`；后者会给图片名自动追加`.jpg`，不适合本地`.png`命名。这是选择官方已有读取器，不是修改源码。
- 在官方10类示例配置中选择7类：`title / text / abandon / figure / table / isolate_formula / formula_caption`。**这不是前一轮的模型原生7类**；Omni允许通过 `eval_cat` 配置参与类别。`abandon`包含页眉、页脚、页码、脚注等，应理解为一个可评分类别，不是全局忽略开关。
- 暂不选 `figure_caption / table_caption / table_footnote` 三类：模型图表标题/图注输出无法直接可靠区分它们，不以关键词、位置或GT辅助猜预测类别。行内公式与阅读顺序不属于这次layout-only评分配置。并非Omni端到端文档解析综合成绩。
- 对应预测：两种标题→title；正文/摘要/目录/侧栏/竖排文本→text；页眉页脚及其图像/页码/脚注→abandon；image/chart→figure；table→table；display_formula→isolate_formula；formula_number→formula_caption。完整映射保存在每次 `layout_detection.yaml` 与 `protocol.json` 中。

### 官方版本中已发现、未暗中修补的行为

1. 官方示例GT映射存在 `refernece` 拼写，而实际GT为 `reference`，所以参考文献在该原样配置中不参与评分。本轮不改这个映射，同时不把模型reference/reference_content计入正文；排除情况单独记录。
2. 官方读取器不传递原始ignore标记，且不会像旧项目评分那样把各类mask变成COCO crowd忽略区；此次保留官方行为。不要把新旧分数差简单解释成模型精度变化。
3. 官方GT `code_txt → figure`，但模型没有精确的code_txt输出，因此该类代码块可能漏检；不擅自把algorithm/seal都当figure。这个残余映射限制单独记录。
4. 官方poly2bbox取特定顶点坐标，而不是旧项目的全点min/max；不修改该函数。1651页GT及预测经过官方读取后未发现退化框。
5. MMEval会提示面积按bbox面积计算，可能影响small/medium/large细分指标；这是该官方调用路径自身行为，不是推理失败。主要汇报`bbox_mAP`（0.5:0.95）和`bbox_mAP_50`；类名以`_precision`结尾的官方字段实际为该类别AP，不是普通precision。

每次保存官方源码/LICENSE快照、固定提交号、输入预测和GT哈希、推理原协议哈希、评分环境/配置、类别覆盖以及排除统计。10页测试的无GT类别为null，不能作为正式全量结果。交付前已通过官方全量读取预检及10页CPU评分；正式全量评分待用户执行上述指令。新结果属于“Omni官方检测实现＋明确记录的类别子集/预测映射”，不是Paddle自建1000页25类的81.4%复现。

## 38号OmniDocBench覆盖整理（2026-09-14，非新推理结果）

逐类结论、样本数、标注层级及来源哈希见 [m38_omni_coverage.md](m38_omni_coverage.md)。复查：`python3 audit_m38_omni.py`，只做CPU标注统计和图像文件头尺寸检查，依赖已有Pillow，不加载模型、不改数据或旧结果。

25个原生类别中，8类有明确对应标注（行内公式仅部分页有span标注，完整评测范围待核）；8类只有上位标签或语义边界待核；9类没有独立正样本标注。不能把旧11组合并评分当成完整25类评测。已确认旧公式预测映射遗漏、行内公式/公式编号未纳入，本轮仅记录，尚未修改推理流程或重跑。暂不补下载，先按覆盖审计决定后续处理。

### Omni对照试跑入口（后续用户已授权，待终端运行）

```bash
cd __MODEL_ROOT__/resources/runtime/assets/workspace/remaining_detection_eval
DEVICE=0 bash run_m38_omni.sh
```

复用 `paddleocr-vl16` + 本包已有 `_paddledeps`，不安装新环境、不下载数据或权重。先10张并检查评分，再接着跑完本地1651页；仅冒烟用 `MAX_IMAGES=10 DEVICE=0 bash run_m38_omni.sh`。CPU预检用 `bash run_m38_omni.sh --preflight`（逐图文件头尺寸和SHA、标注/模型文件、版本、GT构造；不构造模型、不执行前向）。CPU预检已通过：7类6987个正样本框，旧11组30710个正样本框；GPU推理由用户启动验收。

新输出：`__MODEL_ROOT__/resources/runtime/assets/workspace/results/m38_omni_native_v1`。不覆盖旧 `remaining25_pp_v1`。每图保存全部原生类别的官方后处理框、分数、顺序及图像SHA；同一指令可续跑，评分失败后保留预测重试，不重复推理。协议/数据/依赖源码指纹变化时拒绝混用旧断点。只有全1651页推理和评分成功才生成正式 `metrics.json`；10张仅生成 `smoke_metrics.json`。

- 官方入口：`paddleocr.LayoutDetection`，指定本地模型目录、`paddle_static`、FP32、单卡batch1；输入尺寸/归一化遵循本地官方配置。官方示例的 `layout_nms=True` 显式启用；原生overlap过滤与skip_order_labels保持默认，不另写推理/合框算法。
- AP候选分数阈值固定为0.001，避免0.3较早截断低分候选；这是本次项目评测选择，官方81.4%对应的完整参数未公开核实。阈值和NMS设置相对旧实验也有变化，不能把所有分数变化归因于公式映射修复。
- 主结果 `primary_7_explicit_block_classes`：独立公式、页脚、脚注、公式编号、页眉、页码、表格。输出AP50、AP50:95及各类AP50/正样本数；不代表完整25类。
- 辅助结果 `diagnostic_11_merged_groups`：沿用旧GT/预测合并口径，仅修正原生 `display_formula` 的名称映射。标题/正文/参考文献等仍为合并类别诊断，不作为原生细类验收；旧未纳入的table_footnote仍不加入这组，保证类别定义不暗变。
- `reading_order_diagnostic`：沿用旧项目匹配与顺序计数，报告匹配后正确率、元素召回、全GT对正确比例。官方默认跳过部分类别的顺序；不编造缺失order，不把条件正确率当成全部元素正确率。
- 行内公式预测保留，但暂不评分，因为完整span标注页及负样本范围未核实。其余未覆盖原生类别不按0分处理。
- 指标由项目包装 `pycocotools.COCOeval` 计算，不冒充Paddle自建1000张25类官方评测。GT仅用顶层原始框，不合并框、不改类别语义、不自动裁剪；忽略区域按项目COCO crowd约定处理。报告单位为0—1，周报百分数乘100。原生源码、项目源码、权重和每图SHA记录在 `protocol.json`，相关源码保存快照。

## 37号官方入口复核（2026-09-14，待用户运行）

```bash
cd __MODEL_ROOT__/resources/runtime/assets/workspace/remaining_detection_eval
DEVICE=1 bash run_m37_native.sh
```

复用 `object-detection-eval`，不安装新环境、不下载新权重。先10张冒烟，成功后全量运行固定的1000张PubTables替代集（不是完整93834张test）。只跑冒烟：`MAX_IMAGES=10 DEVICE=1 bash run_m37_native.sh`。新输出在 `__MODEL_ROOT__/resources/runtime/assets/workspace/results/m37_native_entry_v1`，旧v2结果原样保留。

直接采用NVIDIA `define_model("table_structure_v1") → model.preprocess → model(...) → postprocess_preds_table_structure`。仅将官方函数中的HF下载请求解析到SHA校验通过的本地权重；原生严格加载、RGB预处理、1024输入、conf=.01、NMS IoU=.25、按类别NMS、后处理阈值=.05保持不变。单卡、batch1、FP32。旧v2已调用相同核心代码，因此结果可能接近或相同，不能预先宣称精度提升。

仍由项目脚本做批量循环、归一化xyxy转COCO像素xywh及pycocotools评分（IoU .50:.05:.95、maxDets=[1,10,100]）。模型卡未提供完整的NVIDIA评估脚本，不能标成“全部使用官方评测协议”。同一份冻结标注覆盖cell/row/column三类、82129个框；cell是由PubTables真实行列及合并结构派生的几何框，不等同于Azure标注。每次记录原生源码、权重、图像、标注哈希及来源，保存官方后处理输出、COCO预测、覆盖清单和全量指标。旧记录的 `original_exact` 不沿用，新记录明确为 `public_replacement`、不可直接对比模型卡。

CPU预检：`bash run_m37_native.sh --preflight`，只检查完整数据/权重及官方CPU构造加载，不执行模型前向。交付前该预检已通过，GPU推理待用户运行。

## 34号原生复核（2026-09-14，待用户运行）

```bash
cd __MODEL_ROOT__/resources/runtime/assets/workspace/remaining_detection_eval
DEVICE=0 bash run_m34_native.sh
```

复用12号 `.venv-m12-ultralytics8043`，不自动安装或改动共享环境；先10张冒烟，再全量237张。仅检查10张用 `MAX_IMAGES=10 DEVICE=0 bash run_m34_native.sh`。输出独立保存在 `__MODEL_ROOT__/resources/runtime/assets/workspace/results/m34_native8043_v1`，历史v4结果不覆盖。

采用模型卡和权重都指定的Ultralytics 8.0.43原生 `DetectionValidator`，由官方实现负责预处理、NMS及AP计算。输入640、conf=.001、IoU=.7、max_det=300与权重train_args一致；采用batch=1、FP32独立验证，不声称复现训练时验证或作者94.6%。模型卡的conf=.25、IoU=.45、max_det=1000属于单图预测示例，不冒充评测参数。

沿用v4冻结XML读取器、PlantDoc TEST划分、VOC坐标解释及植物类别映射，并验证旧标注指纹完全一致。仅转换为YOLO标签，保留权重原始46类ID；237张、12个共有类别，仍缺34类，不能作为完整模型评测。为避免EXIF自动旋转改变图像与框的对应，缓存图像为旧PIL读取器所解码RGB像素的无损PNG，逐图验证像素一致；不旋转、不缩放、不合并框。来源、转换及每张图像哈希保存在dataset/provenance.json。

每次运行保存独立的protocol/status/coverage/metrics及脚本快照；仅全量成功后生成正式metrics.json。本轮是原生流程对照，版本、原生预处理/NMS和AP实现同时变化，不能用单次差值判断具体哪个因素造成影响。模型卡AP50=.946与本地config.json的score_map50=.58305还存在来源不一致，两者均不是PlantDoc的可比目标。

依据本轮用户指令处理原62个模型清单中剩余25个模型。原37个绿色模型的清单、环境和结果目录不作变更。本轮放宽的是替代数据来源与类别范围；仍须覆盖模型卡任务。缺少类别不记0分，单个子任务完成不代表整个多任务模型完成。

- `阻塞统计.md` / `blocker_audit.json`：根据之前文档恢复的初始原因统计，非最新状态。
- `处理进展.md` / `progress.json` / `audit_summary.json`：按唯一model_id汇总的最新结果与状态。
- `尚未闭环任务.md`：尚缺任务、已尝试替代来源、证据及后续动作。
- `models.json`：25个模型、任务、替代集、类别映射清单。继承的旧审计字段用于追溯；实际运行协议结合各结果目录status.json的冻结protocol、对应源码和后续protocol_errata.json勘误核对。
- `supplemental_20260905/补跑结果汇总.md` / `summary.json`：19、30、48原数据补跑后的最新全量结果（含batch及过滤诊断），代码、来源和独立核验均在同目录。
- `review_20260905/复核结论.md`：第二组七模型的最新复核、公开原数据找回及CPU补评分；其中protocol_errata.json修正旧冻结记录的已知字段误差，旧输出不覆盖。
- `verification.json`：独立核对图像数量、唯一性、指标有限值和结果文件哈希。

所有结果为reference_baseline，不能宣称精确复现官方数字。WIDER使用同一验证集和难度评估器，但作者推理设置未完整披露；数值接近也不足以证明完整协议一致。其他公开替代、切分不明、类别合并等情况不能与原模型卡排名混排。

2026-09-05用户明确验收目标：建立后续国产GPU适配的推理与指标对照基线，不要求完全复现模型卡数字。模型卡已明确的设置优先遵循；未披露的设置由本轮固定并记录，不单独作为基线阻塞。后续两端使用同一权重版本、评估数据与划分、标签映射、预处理、后处理和指标定义，精度及硬件差异另行记录。官方复现程度与基线是否可用分开统计；原数据不可得时仍按用户要求选覆盖模型卡全部任务的替代数据。逐项审阅中，12、26、58号已接受为对照基线，其余22个继续逐项确认，记录见official_comparison.json的baseline_review。

## 本轮执行入口

执行包交付到 `__MODEL_ROOT__/resources/runtime/assets/workspace/remaining_detection_eval`。工作区保留同一份可审查代码。基础运行环境在 `/home/ubuntu/gaoyonghan/miniconda3/envs`，新增依赖仅放本包 `_deps`、`_ppdeps`、`_paddledeps`。environment_versions.json是收尾时的只读环境快照；各次实际运行版本以status.json为准，其他队列可能更新基础环境。setup_env.sh仅补本包依赖。

通用入口支持只读描述：

```bash
/home/ubuntu/gaoyonghan/miniconda3/envs/object-detection-hf-v2/bin/python __MODEL_ROOT__/resources/runtime/assets/workspace/remaining_detection_eval/run_model.py --model 37 --describe
```

通用模型先 `--max-images 10`，通过后相同设备、环境和输出目录再 `--max-images 0`。不支持的专用任务使用下表入口。模型/协议/代码变化会拒绝混用旧断点；需要新结果目录或原有源码快照，不修改旧指纹绕过检查。

| 范围 | 入口 | 环境 / PYTHONPATH | 本轮有效结果根目录（均在results下） |
|---|---|---|---|
| 3/9/12/28/46/47/58 | run_model.py；复跑旧协议用v2源码快照 | YOLO类object-detection-eval；28为hf-v2 | remaining25_v2 |
| 17/19/30 | run_model.py | object-detection-hf-v2 | remaining25_v3 |
| 22 | v3/_source_snapshot/run_model.py | object-detection-hf-v2 | remaining25_v3 |
| 34/41/55 | run_model.py | object-detection-eval + _deps | remaining25_v4 |
| 37/48 | run_model.py | object-detection-eval + _deps | remaining25_nemotron_v2 |
| 54 | run_sku110k.py | object-detection-eval + _deps | remaining25_v5 |
| 26 | run_widerface.py | object-detection-eval + _deps | remaining25_wider_v2 |
| 42 | run_cmarkea.py | object-detection-hf-v2 | remaining25_cmarkea_v1 |
| 38 | run_pp_layout.py --model 38；旧源码见pp_v1 | paddleocr-vl16 + _paddledeps | remaining25_pp_v1 |
| 39 | run_pp_layout.py --model 39 --output-root 新目录；已有预测用score_pp_v3.py补评分 | object-detection-hf-v2 + _ppdeps | 推理pp_v2；修正评分pp_v3 |
| 32 | run_qualcomm_coco.py；历史源码见qualcomm_v1/v2 | object-detection-eval + _deps，CPU ONNX | remaining25_qualcomm_v3/hand |
| 35 | run_qualcomm_coco.py --model 35 | object-detection-eval + _deps，CPU ONNX | remaining25_qualcomm_v3/face |
| 8文本子任务 | run_anzhc_text.py | object-detection-eval + _deps | remaining25_anzhc_v3 |
| 8/31全部权重加载验证 | probe_private_weights.py | object-detection-eval + _deps | remaining25_private_probe；无准确率指标 |

设置 `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 YOLO_AUTOINSTALL=false OMP_NUM_THREADS=4`，YOLO_CONFIG_DIR设为/tmp下独立目录。已完成结果不要再用修改后的入口原地启动；交付的快照按文件哈希追踪旧协议。

## 替代集与重要限制

- 表结构：从93834张官方test XML中，推理前按文件名SHA256排序冻结1000图。单元格用微软原代码由真实行、列、合并单元格几何构造，不用模型预测造金标。1000图评估完成不等于跑完原93834图。
- 图表元素：PMC ChartInfo公开测试中有完整文字多边形/角色标注的1873图；x/y轴标题合并为axis_title，x/y刻度合并为tick_label，共8个评分组。10个原生标签有映射不等于10类独立分辨能力已验证。
- 页面元素：DocLayNet与InfoDet按来源分别评分。InfoDet整图infographic框属于明确的几何派生标注，chart框来自原始标注，不宣称NVIDIA原评测集。
- 股票未来预测：类别映射正确，但抽查显示标注趋势区间和模型K线局部框的粒度可能不一致；很低AP须连同该限制解释。
- 密集货架：保留431546个标注框；COCO maxDets=100、预测上限300，与原密集检测协议不同。原始test_274.jpg截断，解码重试逐条记录，不删图提高分数。
- PP V2/V3：检测、阅读顺序分别评分；排序同时给匹配后正确率、匹配元素召回和端到端正确配对比例。V3另算多边形AP。
- cmarkea：必须用DetrForSegmentation，按卡中掩码阈值计算像素F1。旧的检测自动加载会随机初始化，相关尝试不计有效结果。
- Qualcomm：二维任务结果不能代替缺失的三维或完整关键点指标。人脸旧ROI方向错误结果已排除。

数据转换入口：prepare_sku110k.py、prepare_publaynet.py、prepare_native_data.py。转换数据放 `results/remaining25_prepared`，原文件保留。build_audit.py是初始审计生成器，已有丰富清单后禁止覆盖生成。

补充数值审计：旧Qualcomm手部42个、面部3个低姿态分数预测含NaN，未进入原PCK评分。原因是原生像素量化令旋转关键点重合。v3保留检测框并记录landmark_failure，禁止生成非有限坐标；两项均重新执行完整5000图，最终以v3为准。

## 19、30、48补充数据重跑

新数据位于 `__DATASET_ROOT__/supplemental_20260905/`，新结果位于 `__MODEL_ROOT__/resources/runtime/assets/workspace/results/remaining25_supplemental_20260905/`。每个model目录提供实际运行代码、命令、结果报告和校验；30号的作者评分依赖安装在该子目录 `_deps`，未更新共享环境。优先协议分别为19的498页pooled评分、30的作者batch8完整路径、48的560图10类官方示例默认。补充诊断和旧结果同时保留，不能跨协议拼接指标。30批量1协议字段的勘误见 `supplemental_20260905/model30/protocol_clarification_batch1.json`。
