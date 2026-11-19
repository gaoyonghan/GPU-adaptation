# 交付验证记录

日期：2026-09-22。

最终脚本从交付目录 `__MODEL_ROOT__/resources/runtime/assets/workspace/classification171_200_eval` 启动验证，未启动全量评估。

- 30 个唯一模型均在清单中：28 个通过固定 10 张冒烟，174 阻塞，182 按要求跳过。
- 177 的五个权重分别通过；总计 32 个模型/权重运行、320 次计时样本。
- 全部成功项的 `timed_images=10`，forward 耗时为正；没有生成正式 `metrics.json`。
- 核对全部成功项的运行代码哈希与交付代码一致。
- HF、Mamba 和 JS 共 23 个运行项的计时开关前后预测检查全部通过；timm 和 YOLO 使用原生 evaluator，未重写其循环。
- GPU 模型使用本机 RTX 4090 的 `cuda:1`，batch=1，PyTorch FP32；173 使用官方 Transformers.js、CPU、原始 q8 ONNX。
- 8 项单元测试通过：计时不改变 forward 输出/排除预热；断点尾部恢复；拒绝错序/错标签断点；完整输出头的域外预测计错；冒烟类别覆盖；原始图像字节保留；pickle 数据读取禁止执行全局对象；冒烟失败阻止全量执行并正确汇总。
- Python 编译、Shell 语法、Node 语法检查通过；共享环境未修改，三个独立 Python 环境与冻结依赖清单一致。

最终验证结果：

- `validation_final/audit_summary_smoke.json`
- `validation_final/results_smoke.csv`
- `validation_final/smoke/m*/status.json`、`timing.json`、`console.log`

汇总状态为 `partial_complete: 28, blocked: 1, skipped: 1`；进程退出码 2 对应 174 的已知数据阻塞，失败列表为空。

这些记录证明加载、标签索引、原生入口、计时和写结果的小样本流程可运行；不证明全量数据已完成推理，也不作为正式准确率或性能比较结果。
