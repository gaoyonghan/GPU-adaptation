# laion/CLIP-ViT-bigG-14-laion2B-39B-b160k

原 A100 全量评测：completed；本提交副本验证状态见 ../../environment/delivery65_status.csv。
原生任务：classification。独立入口和所需源码位于本模型目录，不调用旧工程中央启动器。

依赖：../../environment/zeroshot-main.txt；权重和固定测试数据不随代码提交。

从仓库 a100/ 运行（实际资源路径由使用者指定）：

```bash
python classification/zhoujianjie/models/laion__CLIP-ViT-bigG-14-laion2B-39B-b160k/inference.py --help
ZERO_SHOT_WEIGHTS_ROOT=/实际共享模型目录 python classification/zhoujianjie/models/laion__CLIP-ViT-bigG-14-laion2B-39B-b160k/inference.py --smoke --max-images 8 --weights /实际权重目录 --dataset-root /实际dataset目录 --output-root result/delivery_smoke_001
```

少量样本只验证交付链路，不能当作官方精度验证。默认不传 --smoke 时保留原全量流程；本轮没有重新执行全量评测。
多模型共用的辅助权重可通过 ZERO_SHOT_WEIGHTS_ROOT 指定包含组织__模型/weights 的目录。
分类输出在运行目录的 inference_output.json 与其 bundles 下；公共评测命令：

```bash
python classification/evaluate.py --run-dir classification/zhoujianjie/models/laion__CLIP-ViT-bigG-14-laion2B-39B-b160k/result/delivery_smoke_001
```

Top-5 只在原生代码输出有序排名时提供，不从 Top-1 伪造。
配置、Prompt、类别映射与预处理沿用原生代码；环境重建尚未验证全新安装。
