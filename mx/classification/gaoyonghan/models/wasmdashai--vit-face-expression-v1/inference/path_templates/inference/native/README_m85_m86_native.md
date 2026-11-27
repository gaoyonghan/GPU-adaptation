# 85/86号Microsoft原生部分覆盖评测

用户于2026-09-15要求能原生就原生。本入口改用原生权重、原生模型实现、原生验证预处理、原生数据读取器以及原生验证循环，保留已完成的HF结果，不替换模型编号或测试样本。

## 运行

```bash
cd __MODEL_ROOT__/resources/runtime/assets/workspace/classification8_eval
DEVICE=0 bash run_m85_m86_native.sh
```

首次需从Microsoft官方GitHub releases下载85/86两份原生`.pth`，共1849503878字节（约1.85GB）；原HF权重保持不动。只提取已有ZIP中1192张交集图片，不下载新图片。首次建立`.venv-m85-m86-swin-native`，继承既有torch/CUDA，用`--no-deps`隔离安装官方要求的`timm==0.4.12`和`yacs==0.1.8`，不修改HF共享环境。PyTorch仍为2.5.1+cu121，torchvision0.20.1，Pillow11.1，非完整历史软件栈。

依次运行85、86；batch2，FP32，通过官方disable_amp选项关闭AMP，TF32关闭。`MAX_IMAGES=10`仅冒烟；无MAX_IMAGES为完整1192张。`MODEL_NUMBER=85`或`86`仅跑指定模型。`--preflight`会下载缺少的源码和原生权重并检查CPU严格加载、数据处理，不做任何模型前向。

结果：`__MODEL_ROOT__/resources/runtime/assets/workspace/results/m85_m86_swin_native_v1/full/run_时间/{m85,m86}/`。启动日志在`logs/`。每次独立时间戳，不覆盖结果；失败后指定模型重试，不自动跨运行续写预测。下载支持续传；网络或哈希异常保留文件并停止，绝不静默切回HF。

## 原生范围

- 官方代码固定提交`f82860bfb5225915aca09c3227159ee9e1df874d`，下载每个必要文件并验证预先核查的SHA256，MIT LICENSE一起保留。
- 直接使用未改动的`models/swin_transformer.py` / `swin_transformer_v2.py`，以及`models/build.py::build_model`构建原生模型。
- 配置使用原仓库`config.py`的默认值和对应YAML。86是原生192/22K配置，`TEST.CROP=True`，短边缩放219后中心裁剪192；85使用官方384结构与测试设置，`TEST.CROP=False`，直接双三次缩放384。
- **85特殊情况**：官方模型库提供384/22K权重，但没有链接单独的384/22K YAML，只有384/22K→1K YAML。本入口借用其同一Swin-L结构和测试设置，明确覆盖`MODEL.NUM_CLASSES=21841`及数据集类型，绝不加载1K权重或进行微调。其他训练参数在eval模式不参与；最终配置完整落盘。这是明确记录的配置组合，不宣称找到了原作者完整历史22K/384评测命令。
- 直接调用原`data/build.py::build_transform(False, config)`与`IN22KDATASET`，不用HF处理器。原生读取器可能对坏图返回随机图，因此入口先核验所有交集图片SHA256、CRC（提取时）、解码和尺寸；不跳过坏图。
- 直接调用原`main.py::validate`的整个函数和`utils.py::reduce_tensor`，用真实单rank NCCL归并；不重新编写推理循环或主指标计算。原`timm==0.4.12`计分，21,841类输出不裁剪。前向hook只记录返回logits，不修改输入或输出，逐图复算验证原生返回指标。
- 通过AST选择原文件中未修改的函数体和装饰器，排除模块级训练专属依赖及CLI启动；并非直接运行整个`main.py`训练入口（该入口还构建训练集、优化器等）。这是调用官方验证函数的项目启动器，不声称完全没有项目包装。模型实现、预处理、数据读取、验证循环和归并均原样使用。

## 权重和对照约束

原生权重来自官方模型表指向的：

- https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_large_patch4_window12_384_22k.pth ，928819451字节。
- https://github.com/SwinTransformer/storage/releases/download/v2.0.0/swinv2_large_patch4_window12_192_22k.pth ，920684427字节。

未取得上游发布的权重SHA256；首次下载验证官方URL与已查询大小，并记录SHA256，后续必须与该记录一致。这是首次信任记录，不冒充上游签名。仅反序列化这些官方来源权重。`load_state_dict(strict=True)`必须完整通过，分类头必须为21841×1536；分类头weight/bias还与已冻结HF权重逐元素比较，不一致则停止核查。不自动删除、重排分类头或插值权重；整骨干逐张数值等价尚未验证。

固定HF对照：`__MODEL_ROOT__/resources/runtime/assets/workspace/results/m85_m86_imagenet21k_partial_v1/full/run_20260915_195624_743521`。manifest SHA256 `9a7a2110e986cf81684885f00e5c23315cf1e6e5b5e3af33c2975118490524a5`，同1192图/124类；HF Top1/Top5为85号34.47986577/65.10067114%，86号33.72483221/65.52013423%。全交集原生结果自动保存相对HF的百分点差异，冒烟不计算此差异。

本次依旧是Kaggle子集与官方验证清单的文件名交集：不是全量21K，未证明镜像图片与官方原图逐字节一致，也未证明这份验证清单与原权重训练集无重叠。原生流程不消除这些限制，`officially_comparable=false`；不能拿1K微调版87%左右指标作本次基准。

原生GPU推理和大权重下载由用户终端执行验收，交付检查不能代替真实端到端结果。
