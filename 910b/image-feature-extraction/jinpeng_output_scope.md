# JinPeng跨设备输出比较范围

使用公共evaluate.py，要求两端完整raw-tensor-features-v4结果根目录；先A100 baseline、后910B candidate。

```bash
python evaluate.py /完整/a100_run /完整/910b_run --output-dir /新的比较目录
```

普通模型沿用原select_output字段，不任意改为池化向量。Granite四项保留多波段TIFF预处理，比较biomass/canopyheight/LST回归输出及UKI分割输出，只说明跨设备数值差异，不说明对应任务精度。当前不具备其独立GT评测协议，不制造回归或分割准确率。ControlNet仅保留原固定零latent/零文本/timestep0的mid_block_res_sample比较，不表示生成图像质量；合成条件构建在forward事件外。

逐图完整张量展平计算cosine、L2和相对L2（分母A100），输出comparison_summary.json与image_similarity.csv；状态measured不是自动验收通过。数据/输出字段不同不能强行配对，缺失样本不取交集冒充完整比较。GenView三份checkpoint分别为GenView-resnet50、GenView-vit-small、GenView-vit-base，按同图特征输出比较。
