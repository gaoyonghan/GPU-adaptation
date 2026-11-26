# prithivMLmods/Deepfake-Detect-Siglip2

当前测试数据：

- [deepfake_vs_real](../../../../dataset/deepfake_vs_real/)

默认使用已配置测试划分，共 8,000 个样本；保持现有筛选、类别映射和样本顺序。

权重：`weights/best_model.pth`，包含开源权重、配置和原生模型代码。
输入图像处理与测试划分沿用原配置。运行依赖见 `../../environment/README.md`。

从项目根目录运行：

```bash
python classification/lirongfu/models/prithivMLmods__Deepfake-Detect-Siglip2/inference.py
```

也可进入本模型目录执行 `python inference.py`。准备、预处理、原生推理、前向计时、标准输出转换、公共评测与结果保存全部由入口完成，不依赖中央调度器。

共享数据从 `../../../../dataset/` 读取，不建立软链接。
评测包与原始数据集同级，目录名为 `数据集名_模型名_imagenet`：

```text
dataset/
├── deepfake_vs_real/
└── deepfake_vs_real_prithivMLmods__Deepfake-Detect-Siglip2_imagenet/
    ├── inference_output.json         # 最近一次成功全量运行的清单
    └── runs/<RUN_ID>/
        ├── annotations.json
        ├── predictions.json
        ├── protocol.json
        ├── inference_output.json
        └── metrics.json
```

原始图片通过相对路径引用，不复制到每个评测包；内嵌图片按需解码到共享 `dataset/_decoded_images/`。
推理后自动调用公共 `classification/evaluate.py`，完整指标覆盖模型 `result/results.json`。
时延、日志和状态仍保存在本模型 `result/`；每次运行的 `evaluation_bundle.json` 用相对路径指向对应评测包。
Top-5 缺少原生排序时标为不可用，少于五类时标为不适用。

只评测最近一次成功全量运行（从平台根目录）：

```bash
python classification/evaluate.py --run-dir dataset/deepfake_vs_real_prithivMLmods__Deepfake-Detect-Siglip2_imagenet
```

评测指定历史或受限运行：

```bash
python classification/evaluate.py --run-dir dataset/deepfake_vs_real_prithivMLmods__Deepfake-Detect-Siglip2_imagenet/runs/RUN_ID
```

公共命令生成包内 `metrics.json` 和评测汇总；正常推理入口会同步模型 `result/results.json`。

## 参数与可配置路径

- `--device cuda`（默认）或 `cpu`：推理设备。
- `--batch-size N`：覆盖该模型原配置的批量大小。
- `--max-images N`：仅处理前 N 张图片，0 为全量；也兼容 `--max-samples`。
- `--checkpoint 路径`：指定权重包；相对路径以平台根目录为基准。
- `--output 路径`：指定指标文件，必须在当前模型的结果目录内。
- `--check`：只检查权重包和数据路径，不进行模型推理。

受限运行默认写 `result/smoke_results.json`，不覆盖全量 `result/results.json`；
预测保存在上述评测包中；每次运行的日志、状态和时延在 `result/runs/<RUN_ID>/` 中。
每次运行记录命令、Python/环境标识、输入范围和输出目录，实际推理错误返回非零退出码。

可通过以下环境变量调整共享存储位置，无需修改源码：

- `CLASSIFICATION_DATASET_ROOT`：默认平台根目录的 `dataset`，各原始数据集直接位于该目录下。
- `CLASSIFICATION_WEIGHTS_ROOT`：设置后按 `<根目录>/<模型目录名>/best_model.pth` 加载。
- `CLASSIFICATION_RESULT_ROOT`：设置后写 `<根目录>/<模型目录名>/`，防止模型结果混用。

以上变量的相对路径均以平台根目录解析，也接受使用者显式配置的外部挂载路径。
目前完成了新目录的路径和隔离流程检查，未在新布局下执行该模型全量推理。

受限运行示例（从平台根目录）：

```bash
python classification/lirongfu/models/prithivMLmods__Deepfake-Detect-Siglip2/inference.py --max-images 8
```
