# MX detection 环境

当前主目录共 61 个模型。镜像 `mx-model-envs:20261124`；各环境的依赖版本见同目录下对应的 `.txt`。

| 环境名称 | 可运行的模型名称 |
|---|---|
| `e24` | `jameslahm/yolov10n`, `jameslahm/yolov10x`, `kadirnar/Yolov10` |
| `e26` | `AdamCodd/YOLOv11n-face-detection`, `Anzhc/Anzhcs_YOLOs`, `Aryn/deformable-detr-DocLayNet`, `cmarkea/detr-layout-detection`, `facebook/detr-resnet-101`, `facebook/detr-resnet-101-dc5`, `facebook/detr-resnet-50`, `foduucom/stockmarket-future-prediction`, `foduucom/stockmarket-pattern-detection-yolov8`, `foduucom/table-detection-and-extraction`, `google/owlv2-base-patch16`, `google/owlv2-base-patch16-ensemble`, `google/owlv2-large-patch14-ensemble`, `google/owlvit-base-patch32`, `google/owlvit-large-patch14`, `hustvl/yolos-base`, `hustvl/yolos-small`, `hustvl/yolos-tiny`, `IDEA-Research/grounding-dino-base`, `IDEA-Research/grounding-dino-tiny`, `iitolstykh/YOLO-Face-Person-Detector`, `jozhang97/deta-swin-large`, `keremberke/yolov5m-license-plate`, `keremberke/yolov5n-license-plate`, `keremberke/yolov8m-protective-equipment-detection`, `keremberke/yolov8s-table-extraction`, `kittendev/YOLOv8m-smoke-detection`, `microsoft/table-transformer-detection`, `morsetechlab/yolov11-license-plate-detection`, `nickmuchi/yolos-small-finetuned-license-plate-detection`, `nvidia/nemotron-graphic-elements-v1`, `nvidia/nemotron-page-elements-v3`, `nvidia/nemotron-table-structure-v1`, `omlab/omdet-turbo-swin-tiny-hf`, `PekingU/rtdetr_r101vd_coco_o365`, `PekingU/rtdetr_r50vd`, `PekingU/rtdetr_v2_r50vd`, `qihoo360/360LayoutAnalysis`, `SenseTime/deformable-detr`, `TahaDouaji/detr-doc-table-detection`, `tech4humans/yolov8s-signature-detector`, `TommyNgx/YOLOv10-Fire-and-Smoke-Detection`, `Ultralytics/YOLO11`, `Ultralytics/YOLO26`, `Ultralytics/YOLOv8`, `ultralyticsplus/yolov8s`, `unity/inference-engine-yolo`, `valentinafevu/yolos-fashionpedia`, `yainage90/fashion-object-detection` |
| `e27` | `foduucom/plant-leaf-detection-and-classification`, `foduucom/product-detection-in-shelf-yolov8`, `foduucom/thermal-image-object-detection`, `keremberke/yolov5n-construction-safety` |
| `e28` | `atalaydenknalbant/Yolov13` |
| `e29` | `jameslahm/yoloe` |
| `e30` | `allenai/WildDet3D` |
| `e31` | `Xenova/detr-resnet-50` |
| `e32` | `PaddlePaddle/PP-DocLayoutV3_safetensors` |

在镜像内进入对应模型目录后执行（复跑时换一个新的输出目录名）：

```bash
source /opt/model-envs/e24/bin/activate
python inference.py --phase full --output-root result/full_001
python /project/detection/evaluate.py --run-dir "$PWD/result/full_001"
```

示例环境对应 `kadirnar/Yolov10`；其他模型请使用表中对应环境。历史小样本验收记录见 `../result/delivery_20261118/REPORT.md`。
<!-- MX_CURRENT_ENV_END -->
