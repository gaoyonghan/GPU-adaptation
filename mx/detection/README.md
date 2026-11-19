# detection

模型在 gaoyonghan/models，个人环境说明和逐模型映射在 gaoyonghan/environment。

```bash
python gaoyonghan/models/<model-name>/inference.py --phase full
python evaluate.py --run-dir gaoyonghan/models/<model-name>/result/run_001
```

运行前选择对应的 MX 环境。推理入口包含格式转换；评测入口只评分，不重新推理。检测统一使用 COCO bbox AP。
