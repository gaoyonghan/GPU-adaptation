# 环境与验证

逐模型环境版本见 model_environment.csv 与 zeroshot-*.txt。main、metaclip2、onealign、mme5、fgclip 各自保留兼容版本，不合并升级。依赖文本来自原 A100 已观测环境；未声称全新 pip 安装成功。

原 A100 历史：65 completed、1 failed（LLM2CLIP Apex ABI）、11 blocked；具体记录见 delivery65_status.json。独立交付状态与历史状态分开。

A100 独立验证目录：/data/zhoujianjie/delivery_validation_20261126。使用 /usr/bin/python 与复制到该目录的独立依赖 overlay；共享挂载不能执行复制的解释器二进制，故没有调用 venv/bin/python。原运行环境不修改。

验证工具 delivery_tools/run_delivery_validation.py 串行运行少量真实样本，带文件锁、状态持久化与失败继续；已通过项跳过。模型入口不依赖此工具。运行时禁止读取原工程源码，只允许明确指定的外部 weights 与 dataset。每次输出保留 source_isolation_*.json。

```bash
# A100 本人目录，运行或恢复少量样本队列
/usr/bin/python /data/zhoujianjie/delivery_validation_20261126/run_delivery_validation.py --root /data/zhoujianjie/delivery_validation_20261126
# 单模型重试（已成功项仍跳过）
/usr/bin/python /data/zhoujianjie/delivery_validation_20261126/run_delivery_validation.py --root /data/zhoujianjie/delivery_validation_20261126 --only 组织/模型
```

同样本比较只读取历史预测，不调用原推理代码。找不到一一对应历史预测时标记 not_compared，不宣称一致。CPU/静态检查不能替代 GPU 验证。小样本结果不得用于官方指标验收。

创建与激活示例（尚未执行全新安装）：

```bash
python3.10 -m venv /自己的可执行目录/venvs/zeroshot-main
source /自己的可执行目录/venvs/zeroshot-main/bin/activate
python -m pip install -r classification/zhoujianjie/environment/zeroshot-main.txt
```

不同模型按 CSV 选择专用依赖，不强行合并。国产硬件需要匹配其运行时，不保证直接沿用 CUDA 构建。
