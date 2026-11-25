# HQQ 0.1.6.post2 局部移植

来源：本地 NVIDIA 复现环境 clip-hqq-repro 已安装的 hqq==0.1.6.post2，
上游 https://github.com/mobiusml/hqq 。仅保留 HQQtimm/ViTCLIPHQQ 所需的 Python 模块，
版权说明与 Apache-2.0 许可证保留在源码和 licenses/。
start.py 优先加载此目录；不修改任何系统环境的 HQQ。

修改的文件：
- core/quantize.py：禁用 CUDA hqq_aten；只允许未编译 PYTORCH backend；
  不创建未使用的 CUDA streams；bias 采用通用 to(device)；
  已加载权重元数据统一随设备迁移，禁用自动 CPU pin_memory/offload 推断；
  活跃加载路径的 empty_cache 使用 NPU，CPU 校验时跳过。
- models/base.py：torch.load(map_location="cpu", weights_only=True)，
  避免序列化 CUDA device 影响服务器；清理与同步使用 NPU。
- models/timm/vit_clip.py：cls_token 和 pos_embed 按模型已有参数设备/dtype 加载。

这些改动不修改 BitPack 解包、Quantizer 反量化数学公式或 checkpoint。
保留的上游非推理/其他 backend 方法不是此移植支持范围；入口强制限制后端。
不支持在这个裁剪包中重新训练、重新量化或切换 CUDA ATEN/compile 后端。
