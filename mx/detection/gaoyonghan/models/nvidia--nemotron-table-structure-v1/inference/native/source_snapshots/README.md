# 补存的历史源码

- `plantdoc/adapters.py`：PlantDoc限定TEST修复后、加入NVIDIA表格/图表分派前的适配器；内容SHA256与模型34的v4冻结协议一致。
- `pp_v2/run_pp_layout.py`：PP V3完整1651图推理时使用的入口；按已知评分修复逆向恢复，内容SHA256与pp_v2/status.json完全一致。旧评分错误使用score_pp_v3.py对同一份预测补算；不覆盖旧预测。

其他运行快照在对应results目录，最新可运行代码在本包根目录。
