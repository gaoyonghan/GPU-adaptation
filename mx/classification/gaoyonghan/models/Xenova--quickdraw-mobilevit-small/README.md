# Xenova/quickdraw-mobilevit-small

当前目录已按 gaoyonghan/mx 格式迁移，环境名称统一为 e07。下述 48 模型全量记录来自迁移前；迁移后的真实验证范围见 [ALIGNMENT.md](../../ALIGNMENT.md)。

MX GPU 分类模型 m113，贡献者 caichengrui。整理状态：已从交付入口完成 full 推理和公共评分，覆盖 250,000 个有效原生样本；结果在 [result/manual_smoke_20261116_235332/](result/manual_smoke_20261116_235332/)。新机器环境重建尚未验证。e07 历史运行与历史评分见 result/historical_e07_20261107/，不能替代 Upload 的运行验证。

数据：mx 平台根 dataset/；原生登记范围 quickdraw-small，250000 个有效样本。划分、预处理、类别映射、精度、原生模型后端保留。详细原生限定见 inference/native/models.json 及原生 metrics.json。m104 保留跳过 1 个损坏图像的逻辑；m149 分别输出 age/gender/race 三个单标签分支；m152 的一个样本为原图的一个旋转角度。

权重：本目录 weights/，原生所需文件和 SHA-256 固定在 inference/native/models.json。离线加载；无中央调度器或原工作区代码依赖。目录对齐 gaoyonghan/mx：原生文件位于 inference/native/；run_model.py、common.py、timing.py、prepare.py、entrypoint.json、models.json、output_format/。

从 Upload/mx 平台根目录（已激活 environment/README.md 中的 MX 环境）执行：

```bash
python classification/caichengrui/models/Xenova--quickdraw-mobilevit-small/inference.py --phase preflight
python classification/caichengrui/models/Xenova--quickdraw-mobilevit-small/inference.py --show-command
python classification/caichengrui/models/Xenova--quickdraw-mobilevit-small/inference.py --device cuda:0
python classification/evaluate.py --run-dir classification/caichengrui/models/Xenova--quickdraw-mobilevit-small/result/run_001
```

从本模型目录也可执行 `python inference.py`；路径均基于 __file__。`--output-root result/run_002` 指定本模型内的新运行目录；`--prepare-only` 转换已完成的同阶段原生输出；`--phase smoke --max-images 10` 仅对原生已有 MAX_SAMPLES 的 m112、m146 开放，先做原生完整合法性检查再按其原有限制逻辑选样；其余 46 个模型明确拒绝 smoke，避免擅自截断导致原生指标分母或数据范围不一致。full 直接全量运行，与 MX 分支不强制前置 smoke 的格式一致。

参数默认 batch-size=512、workers=4、warmup=3，保留 e07（SPIDER 为已验证 batch=8）。`--device cuda:N` 使用 MX 逻辑编号，要求 MetaX torch 和实际 C500，不允许 CPU 回退。数据/权重可通过 MODEL_DATASET_ROOT / MODEL_WEIGHTS_ROOT 选择项目 dataset/ 内、本模型 weights/ 内的路径；MODEL_OUTPUT_ROOT 仍须属于本模型 result/。

预测/标注/协议：Example v1 分类合同；稳定相对路径、归档成员名或数据集标识。原生 GT 与指标输入逐样本核对，标准标注有 schema 且仅包含 categories/samples 的规定字段。保留原生排序；原生仅导出 Top-3 时公共 Top-5 记为 unavailable。 分支：default。公共评分完全在 classification/evaluate.py，模型内没有独立评分器。

计时：timing.json 独立记录原生预热和当前流 MX events 纯模型调用（秒），actual_batch_sizes 为批大小到批次数，吞吐量为 timed_images / forward_seconds。模型加载是被测原生加载块之和，与 Example 连续加载块边界不同。inference_e2e_seconds 实测原生 DataLoader 取批开始至原生预测/排序到达 CPU，扣除后续指标更新和记录写入；多 worker 预取可能提前解码，SPIDER 拼图先于此边界，部分原生 loss 已在预测前计算，故与 Example 边界不同，不宣称完全一致。具体差异列于 inference/native/models.json 的 timing_alignment；另记更宽的 native_pipeline_seconds。公共评分由任务评测器独立执行；本原生 timing.json 的 evaluation_seconds 为 null，尚未单独测量。没有恢复续跑，重复阶段保留到 _history。instrumentation_predictions_unchanged 为 null；历史逐样本预测比对另行记录，不能替代 Example 无计时 baseline 的数值等价证明。详见贡献者 TIMING_ALIGNMENT.md。

环境精确映射：../../environment/e07.txt，Python 3.12.11 / MetaX torch 2.10.0+metax3.8.1.0。已有 e07 全量通过；Upload 本次 full 推理与公共评分已通过，新机器重建尚未验证。不能用 CUDA/NVIDIA 依赖替换 MX 运行库。

实际后端：transformers；原生精度：See native metrics and source。原生数据范围（保留选择逻辑，不将代理/重叠集称作独立测试）：

```json
{
  "name": "quickdraw-small",
  "source": "Xenova/quickdraw-small",
  "split": "test",
  "image_count": 250000,
  "num_classes": 345,
  "image_mode": "grayscale (L)",
  "split_note": "Uses the published 250,000-image test parquet; no local random split is created."
}
```

历史数值比对存在差异，不能标为逐样本完全一致；同卡同参复跑与实际差异见 [VALIDATION.md](../../VALIDATION.md)。清单状态明确标为 full_scored_historical_prediction_differences。

路径配置见 [paths.json](paths.json)；[inference/local_paths.json](inference/local_paths.json) 登记路径模板。相对 MODEL_DATASET_ROOT 和 MODEL_WEIGHTS_ROOT 均以本模型目录为基准。
