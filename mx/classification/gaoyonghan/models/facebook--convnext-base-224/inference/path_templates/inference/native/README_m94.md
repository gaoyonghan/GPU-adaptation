# 94号：作者图片数据与原生timm评测

```bash
cd __MODEL_ROOT__/resources/runtime/assets/workspace/classification8_eval
DEVICE=0 bash run_m94.sh
```

第一次仅安装作者同版timm 0.9.16到独立`.venv-m94-timm0916`。约1MB作者原始笔记本已随脚本保存在`official_m94.ipynb`，按固定SHA256校验后使用，无需连接GitHub；仅本地源码缺失时才尝试下载。继承既有torch/CUDA，不改共享环境，不再下载数据或权重，不训练。先10张以内冒烟，通过后完整跑1660张；全量会重新包含首10张，以保持原batch64布局，不将两轮样本混合计分。

`MAX_IMAGES=10`仅冒烟、不输出正式accuracy；`--preflight`执行源码/数据/环境核验和CPU权重严格加载，不做模型前向；`--describe`查看冻结协议。结果每次新建：`__MODEL_ROOT__/resources/runtime/assets/workspace/results/m94_native_v1/full/run_时间/`，日志位于根目录logs；不覆盖旧结果，失败不自动跨运行续写。

## 对齐依据

- [作者仓库](https://github.com/jaranohaal/vit-base-violence-detection)，固定提交`0961a54da65dffffd95ac3abda869e4cbc4a674a`，笔记本SHA256见models_m94.json。
- 作者cell 7直接下载[abdulmananraja图片版](https://www.kaggle.com/datasets/abdulmananraja/real-life-violence-situations)，不是另外一个2000段视频数据集。11073张图片均可解码；非暴力5231、暴力5842。
- 原生ImageFolder排序，执行未修改cell 14的random_split与种子42；train7753、valid1660、test1660。本次只用第三段test，其中non_violence=771、violence=889。0/1标签来自作者ImageFolder顺序，不由测试结果反推。
- 执行作者cell 8的原始尺寸和transform赋值：224x224双线性Resize、ToTensor、mean/std均0.5，ImageFolder负责RGB读取。没有额外抽帧、裁剪、增强或EXIF转正。
- 发布者HF配置与演示指向Transformers，但实际两个权重文件都包含timm格式的152个参数，全部张量相同。本入口直接构建原生`vit_base_patch16_224`并替换2类head后严格加载发布的safetensors。构建时pretrained=False仅阻止下载初始化权重，所有参数均由发布权重严格覆盖，不更换模型。
- 用作者cell 19的整个`eval_violence`函数执行原生推理和accuracy计算，函数体不改。其分母使用全局val_len，本任务valid/test恰好都是1660，因此全量分母正确；冒烟也保留此分母，但不发布该子集分数。
- 不执行笔记本账户、训练、保存完整模型、其他网络、上传等单元。也不执行原测试loss循环中的loss.backward，因为accuracy函数无需它。作者打印的Loss仅为最后一个batch的label-smoothed CE，不作为全测试平均损失报告。

## 必须保留的特殊情况

1. 作者笔记本ViT Base输出**98.49%**，模型卡写**98.80%**。未获得作者训练产物`vit1.pth`与公开权重的对应证明，两个参考值并列，不任选一个来判断掉点。
2. 保留作者随机按图片划分的协议，未按来源视频重新划分。全数据有108组完全重复图片，44组跨划分，23组同时出现在train/test，涉及26张test图；全数据另有7组相同字节但标签冲突。全部风险记录，不偷偷删图、去重或改标签。结果不代表严格无泄漏的泛化水平。
3. 图片来源、总数和公开划分代码已对应，但缺作者历史逐文件哈希/测试清单，不能保证上游当前数据与历史数据逐文件完全相同。冻结本地全清单、图片SHA256、划分索引及标签，运行时逐一核对。
4. timm 0.9.16与笔记本日志一致；当前torch2.5.1/torchvision0.20.1/Python3.11不同于作者日志中的2.2.1/0.17.1/Python3.10。FP32、batch64，不启用AMP，TF32关闭；未人为更换timm注意力实现。

因此是**作者数据来源/划分代码＋原生模型/评估函数**，但不标记exact_official：采用reference_baseline、original_protocol_incomplete，并标记重复泄漏、标签冲突、参考值映射和框架差异。不是另选的公开替代集。

交付验证仅CPU预检、严格加载及合成数据接口测试；真实GPU推理由用户执行。
