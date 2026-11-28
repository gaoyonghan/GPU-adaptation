# 环境说明与模型对应关系

本目录整理当前 **70 个模型**的运行环境与依赖依据。逐模型对应关系见 [model_environment.csv](model_environment.csv)。`venv/` 和 `venv_tf/` 位于项目根目录，保留各自已安装的依赖包。

复制环境的完整已安装分发包版本记录见 [requirements-main.txt](requirements-main.txt)（235 项）和 [requirements-tensorflow.txt](requirements-tensorflow.txt)（66 项）。这些清单按副本里的包元数据生成，包含传递依赖与额外已装包；它们是存档，不是经过重新安装测试的最小依赖锁，不用于覆盖现有环境。

历史结果来自原服务器路径。`Model/<模型>/result/environment.json` 记录原预测/评测运行环境；有 `forward_timing.json` 的模型，其 `benchmark_environment` 记录后续计时补测环境。两种记录均保留原样，不把历史路径替换成新路径，也不将旧结果解释为新目录已验证通过。

## 环境分组

| 用途 | 模型数 | 入口环境参数 | 关键说明 |
| --- | ---: | --- | --- |
| 常规 PyTorch、ONNX 与 IBM flood 推理 | 66 | `--venv venv` | 主环境；其中 4 个为 ONNX 模型 |
| TensorFlow 推理 | 3 | `--venv venv_tf` | Narsil、keras-io、SerdarHelli Teeth |
| DiNAT 推理 | 1 | `--venv venv` | 主环境的 PyTorch，加 `venv_tf` 的 Transformers 4.49.0 与 NATTEN |
| 全部模型的统一评测 | 70 | `--venv venv` | COCO JSON / 像素计数评测由 CPU 完成，不加载模型权重 |

三个 TensorFlow 模型的完整名称：

- `Narsil__pet-segmentation`
- `keras-io__semantic-segmentation`
- `SerdarHelli__Segmentation-of-Teeth-in-Panoramic-X-ray-Image-Using-U-Net`

DiNAT 为 `shi-labs__oneformer_ade20k_dinat_large`；IBM flood 为 `ibm-granite__granite-geospatial-uki-flooddetection`。CSV 中的“已安装版本”仅表示历史环境元数据中存在该包，不能据此认定该模型使用这个后端。例如主环境记录包含 TensorFlow，但多数模型实际使用 PyTorch。

## 环境激活与启动

两个环境的 `bin/python`、`python3`、`python3.10` 使用指向服务器 `/usr/bin/python3.10` 的标准符号链接；Python 的环境前缀与依赖目录仍分别属于本项目的 `venv/`、`venv_tf/`。激活脚本指向当前项目，并支持目录名中的空格。

```bash
cd '/data/model inference for GPU adaptation/segmentation/qiuchuangxin'
source ./venv/bin/activate
python --version
python -m pip --version
deactivate

# TensorFlow 环境
source ./venv_tf/bin/activate
python --version
python -m pip --version
deactivate
```

当前 `/data` 为 GeeseFS 挂载，普通文件的执行权限不能通过 `chmod` 持久设置。安装或查询依赖请使用 `python -m pip`，不要直接执行 `./venv/bin/pip`。已激活环境中的 `python` 和完整路径 `./venv/bin/python` 均可启动；命令脚本也可显式交给解释器，例如 `sh ./venv/bin/pip --version`。跨服务器复制环境后，需要目标机器提供兼容的 Python 3.10。

项目推理与评测仍可使用以下方式，由入口选择相应依赖和动态库。批量运行建议保留 `--venv auto` 的环境选择方式；不要把所有 TensorFlow 模型直接放到主环境中运行。

```bash
cd '/data/model inference for GPU adaptation/segmentation/qiuchuangxin'

# 常规模型推理；示例明确只生成推理缓存，不重新导出/评测已有结果
/usr/bin/python3 -I -S -B Model/CIDAS__clipseg-rd64-refined/inference.py \
  --venv venv --stage inference

# TensorFlow 模型使用同一系统 Python，选择另一套项目包
/usr/bin/python3 -I -S -B Model/Narsil__pet-segmentation/inference.py \
  --venv venv_tf --stage inference

# DiNAT 的入口负责混合栈初始化；参数仍是 venv
/usr/bin/python3 -I -S -B Model/shi-labs__oneformer_ade20k_dinat_large/inference.py \
  --venv venv --stage inference
```

`-I -S -B` 隔离外部 Python 搜索路径、跳过自动 `site` 初始化及 `.pth` 执行，并禁止生成 `.pyc`。入口需明确定位本项目 `lib/python3.10/site-packages`，不能回退到原项目或用户全局包。系统 Python 的 major/minor 必须与环境一致（3.10）。这些项目启动命令不会改写环境文件。

新项目路径包含空格，Shell 中包含项目绝对路径的参数必须加引号。动态库目录应由入口构造 `LD_LIBRARY_PATH`；不要把含空格的完整库路径放入 `LD_PRELOAD`。

## 历史运行记录确认的核心版本

下表取自模型结果文件，不是完整的 `pip freeze`。模型记录只采集少数包；未记录的包不能由本表推断其版本。

| 项目 | 主环境历史记录 | TF / 特殊环境历史记录 |
| --- | --- | --- |
| Python | 3.10.12 | 3.10.12 |
| NumPy | 1.26.4 | 1.26.4 |
| Pillow | 12.3.0 | 12.3.0 |
| pycocotools | 2.0.11 | 2.0.11 |
| PyTorch | 2.6.0+cu124 | DiNAT 仍使用主环境 2.6.0+cu124 |
| torchvision | 0.21.0+cu124 | DiNAT 仍使用主环境 0.21.0+cu124 |
| PyTorch CUDA | 12.4 | 不代表 TensorFlow 使用同一套 cuDNN |
| Transformers | 5.15.1 | DiNAT 为 4.49.0；TF 环境计时记录也列出 4.49.0 |
| TensorFlow | 主环境有安装记录，非主推理后端 | 3 个 TF 模型为 2.21.0 |
| ONNX Runtime GPU | 1.20.2 | 4 个 ONNX 模型使用主环境 |
| ultralytics | 8.4.132 | YOLO 模型 |
| MONAI | 1.4.0 | vesselFM、NV-Segment-CT 等医学模型 |
| GPU | NVIDIA A100-SXM4-40GB | 多份历史记录；不是当前 GPU 可用性验证 |

统一评测所需主要包为 NumPy、Pillow、pycocotools；根评测脚本输出 Excel 时还需 openpyxl。COCO 区域 AP 的连通区域构造及部分数据读取路径还会用到 SciPy，具体应保留代码所需依赖。评测不需要导入 TensorFlow、NATTEN 或实例化模型。

## 当前复制环境包元数据

此表来自本次对复制环境 `*.dist-info/METADATA` 的只读检查，与上表的历史运行记录分开保存。读取版本不执行这些包，也不能证明二进制扩展在新路径可加载。

| 依赖 | venv | venv_tf |
| --- | --- | --- |
| torch / torchvision | 2.6.0+cu124 / 0.21.0+cu124 | 未发现这两个分发包的元数据 |
| tensorflow / tf_keras | 2.21.0 / 2.21.0 | 2.21.0 / 2.21.0 |
| transformers | 5.15.1 | 4.49.0 |
| tokenizers | 0.22.2 | 0.21.4 |
| huggingface_hub | 1.28.0 | 0.36.2 |
| natten | 未发现分发包元数据 | 0.17.5+torch260cu124 |
| nvidia-cudnn-cu12 | 9.1.0.70 | 9.28.1.4 |
| terratorch | 1.0 | 未发现分发包元数据 |
| lightning / pytorch-lightning | 2.5.1.post0 / 2.5.1.post0 | 未发现分发包元数据 |
| NumPy / SciPy | 1.26.4 / 1.15.3 | 1.26.4 / 1.15.3 |
| pycocotools / openpyxl | 2.0.11 / 3.1.5 | 2.0.11 / 3.1.5 |

因此 DiNAT 不能单独换成 `--venv venv_tf`：其 NATTEN wheel 标记对应 torch2.6/CUDA12.4，而该环境没有 torch 的分发包记录。需要主环境 torch 与另一环境的 NATTEN/Transformers 组合。读取 `terratorch==1.0` 元数据不能确认安装来源提交；若将来重建，仍需核对源码来源或保留现有包。

`venv/00_segmentation_runtime.pth` 中原有的 Pydantic 插件禁用设置，不会在 `-S` 下自动运行；如果要保持该性能设置，应由项目入口显式设置 `PYDANTIC_DISABLE_PLUGINS=__all__`，不能为此恢复执行所有未知 `.pth`。

## 各类模型的补充依赖

| 模型 / 代码路径 | 必需或按相关路径使用的依赖 |
| --- | --- |
| Transformers 语义/全景、CLIPSeg、Breast SAM | torch、torchvision、transformers、tokenizers、huggingface-hub、safetensors；原生 COCO 全景代码还引用 panopticapi |
| BEN / BiRefNet / RMBG / ToonOut | timm、einops、kornia、safetensors、opencv；具体组合见模型代码与 CSV |
| Medical-SAM3 / MedSAM2 | hydra-core、omegaconf、iopath、einops/ftfy、scipy、opencv 等；代码包含训练/视频可选分支，其全部 import 不等于本次静态图像任务都要启用 |
| YOLO | ultralytics、ultralytics-thop、torch、torchvision、opencv |
| ONNX | onnxruntime-gpu、NumPy；CUDA/cuDNN 动态库；当前计时逻辑还用 PyTorch 做 CUDA 预检 |
| Narsil / keras-io Pets | tensorflow、**tf-keras（模块名 tf_keras）**；仅有 Keras 3 不能替代该 import |
| Teeth | tensorflow；当前计时采用 TF2 SavedModel serving，原预测运行曾用 TF1 Session；差异记在 forward_timing.json |
| DiNAT | NATTEN CUDA 扩展与其 torch/CUDA ABI、Transformers 4.49.0；先加载主环境 torch，再选择兼容栈 |
| IBM flood | TerraTorch、Lightning、torchgeo、timm、rasterio，以及内嵌 granite 模块要求的 TerraTorch API |
| 医学 / 体数据读取 | SimpleITK、nibabel、MONAI、SciPy、tifffile，按各模型数据格式使用 |
| 数据集 / 图像后处理 / 结果汇总 | pyarrow、scikit-image、scikit-learn、pandas、PyYAML、openpyxl，按实际数据与输出路径使用 |

旧 `requirements.txt` 中许多核心包行被注释，不能视为“执行一次即可完整重建两个环境”的锁定清单。它对 TerraTorch 声明了官方提交 `c897e86cec8eb5334e205d6e31e50cf972bddac7`，但仅凭这条注释不能证明当前副本实际安装的提交。NATTEN、tf-keras 及 TF cuDNN 的当前版本已在上表和完整清单中记录。现有 requirements 中 h5py==3.16.0、fsspec==2025.12.0 与当前主环境元数据 h5py==3.14.0、fsspec==2026.9.0 不同，因此不能把旧声明当作当前实装记录。此处不提供会改变现有环境的批量升级命令。

## 特殊运行约束

- **DiNAT**：不能在 Python 启动前把整个 `venv_tf/site-packages` 放到主环境前面。先固定主环境 NumPy/PyTorch，再加载 TF 环境里的 Transformers/NATTEN；当前模型已有 NATTEN API 适配。历史成功组合需要保持这种导入顺序。
- **IBM flood / DiNAT**：当前模型已禁用三个可选 FP8 模块（transformer_engine、transformer_engine_extensions、msamp）；历史启动还隔离了不兼容的 flash_attn 与 flash_attn_2_cuda。新入口应避免从系统包误加载这些扩展。
- **TensorFlow**：使用 `venv_tf` 自己的 CUDA/cuDNN 库，不能直接继承主环境的 cuDNN 9.1 搜索路径。旧 TF sitecustomize 启动层与本次 `-S` 入口是不同启动路线；新入口必须自行完成包路径与动态库初始化。
- **ONNX**：历史记录同时注册 CUDA 和 CPU provider；不能把 provider 列表当成所有算子均在 GPU 的证据。补测使用 CUDA IOBinding，并将外部数据上传/下载排除在 forward 计时外。

## 记录口径

CSV 每行对应一个实际存在的模型目录，包含关键直接依赖、选用环境、历史解释器及依据文件。版本以原始 `environment.json` 或计时补测 `benchmark_environment` 为依据；原预测环境与补测环境分别保留。推理时间、预测缓存、评测 JSON、模型权重均不因环境文档生成而修改。

