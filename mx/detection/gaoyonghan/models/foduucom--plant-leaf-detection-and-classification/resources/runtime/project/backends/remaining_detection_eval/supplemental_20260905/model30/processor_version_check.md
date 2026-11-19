# 30号 Conditional DETR处理器版本核查

2026-09-05。仅比较官方Transformers v4.44.0源码与本次实际环境的4.50.0源码；没有安装或更换依赖、没有运行GPU、没有修改runner。

**结论：在本次实际输入和参数路径上，没有找到能改变batch padding、GT归一化或预测后处理语义的明确版本变化。不能把剩余AP差距直接归因于“4.44升级到4.50改变了这些处理”。两版均有相同的padding后GT框更新；batch1与batch8的区别不是4.50才引入。**

## 直接处理器代码

比较[官方v4.44.0 ConditionalDetrImageProcessor](https://github.com/huggingface/transformers/blob/v4.44.0/src/transformers/models/conditional_detr/image_processing_conditional_detr.py)与本地实际文件：

`/home/ubuntu/gaoyonghan/miniconda3/envs/object-detection-hf-v2/lib/python3.11/site-packages/transformers/models/conditional_detr/image_processing_conditional_detr.py`

两版以下函数的完整AST相同（不仅是名称相同）：

- `__init__`、`prepare_annotation`、`prepare_coco_detection_annotation`。
- `resize`、`resize_annotation`、`normalize_annotation`。
- `_pad_image`、`pad`、`get_max_height_width`、`make_pixel_mask`。
- `get_size_with_aspect_ratio`、`get_resize_output_image_size`。
- `post_process_object_detection`：sigmoid后top100、类别／query索引、cxcywh转xyxy、按传入target_sizes缩放、阈值过滤均相同。

`preprocess`的实际差异只有旧`max_size`兼容参数处理，以及`do_rescale`告警条件的短路顺序；本轮直接使用权重配置的 `size={shortest_edge:800,longest_edge:1333}`，没有传deprecated `max_size`，因此没有触发该不同分支。文件末尾新增`__all__`也不改变推理。

## 公共图像辅助函数

还比较了官方v4.44.0的[image_transforms.py](https://github.com/huggingface/transformers/blob/v4.44.0/src/transformers/image_transforms.py)、[image_processing_utils.py](https://github.com/huggingface/transformers/blob/v4.44.0/src/transformers/image_processing_utils.py)、[image_utils.py](https://github.com/huggingface/transformers/blob/v4.44.0/src/transformers/image_utils.py)与本地4.50.0。

| 变化 | 是否影响本轮实际路径 |
|---|---|
| `rescale`先显式astype(float64)，再乘scale，最后转float32 | 本轮RGB解码→NoOp→PIL resize输出uint8。在实际NumPy1.26.4下，旧uint8×浮点scale本就先产生float64；对全部256种uint8值比较1/255缩放，结果逐位相同，最大差0 |
| `normalize`将Iterable检查改为Collection | 本模型mean/std为各3元素的普通list，两版走同一分支，计算方式相同 |
| `to_pil_image`新增可选image_mode=None | 本轮未指定mode，两版均由RGB uint8数组推断模式；resize调用使用命名参数，不存在新增参数造成的位置错传 |
| `resize`、`pad`、get_size_dict等类型标注更新；坐标转换注释纠错 | 相关计算函数体不变 |
| make_list_of_images的PIL识别辅助函数变化 | 本轮传入已有NumPy数组列表，两版直接返回相同列表 |

因此，已检查路径中的像素缩放、归一化、batch padding、归一化GT更新、cxcywh／xyxy换算及top100预测后处理没有发现本次可触发的语义偏差。

## 范围及证据

这次是处理器与上述直接辅助路径的源码核对，以及一个256值的CPU数值小检查；**没有验证整个4.44环境与4.50环境完全等价**，没有比较模型前向、PyTorch/CUDA内核、Pillow/Albumentations其他版本的实现，也没有用不同库版本重新推理。因此可以排除所述处理器路径的已知明确语义差异，不能笼统断言所有软件版本都不会影响结果。

官方源码已保存为本目录下 `processor_version_check_v444*.py`，保留原Apache许可证头。主处理器SHA256：

- 官方4.44.0：`3f7c68f0681423e01a3ec2192238ba23ad75c88de504f8520580652d05d91e78`
- 本地4.50.0：`288d1b83bb19c655295ed794a607817e54246387df1225ec5ec4e9b754ea1272`

本轮实际代码在 `run_batch8.py`中明确按8张连续行调用processor、取出labels、移除pixel_mask；其后两端框均按作者公式恢复。padding对该公式的影响应作为作者协议本身记录，不应错误归咎于处理器版本更新。
