# 评测使用说明

evaluate.py 复制自仓库现有公共 classification/evaluate.py，仅将相对输入路径基准改为当前 zhoujianjie/ 目录；指标定义、Top-1/Top-5 策略、bundle 校验和评分代码保持不变，原公共脚本未修改。

```bash
# 仓库根目录执行
python a100/zero-shot-classification/zhoujianjie/evaluate.py --run-dir models/组织__模型/result/run_001
# 也可传入运行目录的绝对路径
python a100/zero-shot-classification/zhoujianjie/evaluate.py --run-dir /自己的运行结果目录
```

每个运行目录由模型自己的 inference.py 生成 inference_output.json，里面记录相对 bundle 路径；bundle 包含 predictions.json、annotations.json、protocol.json。评测结果保存在相应 bundle 的 metrics.json 和运行目录内。

仅支持单标签分类；原始检索任务不能套用此脚本，也不能把 Recall@K 冒充 Top-1。原生缺少有序排名时不伪造 Top-5。
