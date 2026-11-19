# 零样本检测：官方协议评估包

本任务包遵循 `__MODEL_ROOT__/resources/runtime/assets/workspace/项目总要求.md`。

这个目录解决的是“结果能否和官方数字比较”，而不只是“模型能否输出框”。每次运行都会把
模型权重版本、数据集、协议、环境、预测和指标一起记录。原来的
`__MODEL_ROOT__/resources/runtime/assets/workspace/zero_shot_eval` 是单模板 Transformers 基线，保留但不再作为
OWL/OWLv2 的官方可比结果。

## 判定规则

- `exact_official`：直接运行官方代码、原生权重和官方环境。
- `official_protocol_adapter`：本地权重是官方转换格式；适配器严格实现官方协议，并明确记录
  与原生框架的差异。
- `reference_baseline`：官方没有发布与报告值完整对应的 evaluator，只能做参考基线。

八个模型的逐项记录都在 `models.json`。例如：

```bash
cd __MODEL_ROOT__/resources/runtime/assets/workspace/official_zero_shot_eval
/home/ubuntu/gaoyonghan/miniconda3/envs/zero-shot-eval/bin/python run_model.py \
  --describe google/owlv2-base-patch16-ensemble
```

Grounding DINO base/tiny 的已有结果分别是 56.90 和 48.49 AP，已经对齐官方 56.7 和
48.4/48.5，因此默认队列不重复运行；它们仍会出现在总审计表中。

## 运行

检查或补齐统一环境：

```bash
cd __MODEL_ROOT__/resources/runtime/assets/workspace/official_zero_shot_eval
bash setup_env.sh
```

先用少量图片检查 6 个待处理模型的加载和数据链路。少量图片不会计算 AP：

```bash
MAX_IMAGES=10 bash run_all.sh
```

确认后从断点继续全量：

```bash
bash run_all.sh
```

默认使用 GPU 0、1、float32 和 batch size 1；可用 `GPU_IDS=1` 改成单卡。正式可比结果
建议保持默认 float32。新结果单独写到：

```text
__MODEL_ROOT__/resources/runtime/assets/workspace/results/zero_shot_official/
```

每个模型目录包含：

- `status.json`：完整模型清单快照、协议指纹、HF revision、权重文件标识、数据标注哈希、
  软件版本、运行状态和显存统计；
- `predictions.jsonl`：逐图断点；
- `predictions.json`：提交给官方 COCO/LVIS API 的预测；
- `metrics.json`：指标、官方参考值比较和可比性结论。

根目录 `audit_summary.json` 汇总八个模型的协议等级、状态、特殊情况和指标。可随时重新生成：

```bash
/home/ubuntu/gaoyonghan/miniconda3/envs/zero-shot-eval/bin/python run_model.py \
  --summarize
```

## 本批次的特殊情况

- OWL-ViT/OWLv2：使用完整 LVIS v1 val、Scenic 七模板、模板 logits 求平均后 sigmoid、
  全局 top-300、补边坐标还原和官方 `LVISEval`。本地是 PyTorch 转换权重，不是 Scenic
  原生 JAX checkpoint，所以标为 `official_protocol_adapter`。
- OWLv2 名称里的 `ensemble` 是 checkpoint 权重集成，不是七模板提示词集成；两者都会记录。
- OmDet-Turbo：官方仓库没有给出能定位到报告 42.5 AP 的完整 COCO 评估入口和所有参数，
  因此标为 `reference_baseline`，不能把差值简单解释为模型复现失败。
- 旧 OWL/OWLv2 结果使用单提示词，并且没有完整执行 Scenic 的补边坐标还原；只作为调试
  历史，不纳入官方对比。
