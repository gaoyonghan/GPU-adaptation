# gianlab/swin-tiny-patch4-window7-224-finetuned-skin-cancer

A100 使用 `/opt/model-envs/e07`。`requirements.txt` 和 `inference/environment.json` 记录此环境中的依赖版本。路径配置见 `paths.json`。

编号 m135；贡献者 caichengrui；单标签分类。框架：torch。

## 统一入口

从本模型目录执行：

~~~bash
python inference.py
python inference.py --show-command --device cuda:0 --warmup 10
python inference.py --phase full --device cuda:0 --warmup 10 --output-root result/run_002
~~~

也可从项目根执行 python classification/gaoyonghan/models/gianlab--swin-tiny-patch4-window7-224-finetuned-skin-cancer/inference.py；默认路径不依赖当前工作目录。
默认输出 result/run_001/；入口自动执行原生推理和 prepare.py，无须手动运行准备脚本。
已有 inference_output.json 的运行目录需换成新的 --output-root，以免覆盖结果。

--phase preflight 检查本地必需输入及路径，不加载框架、不运行推理、不写 result/。

## 内部脚本与配置

- inference.py：统一对外入口、日志、运行命令和自动格式转换。
- inference/native/run_model.py：保留模型加载、预处理、前向、标签、划分、原生指标和计时流程。
- inference/common.py：本模型路径、公共参数、原生预测采集及原生指标辅助函数。
- resources/runtime/project/timing_runtime/timing.py：预热参数及原生计时记录；不增加模型前向，不改变计时口径。
- inference/prepare.py：核对原生完成状态，自动发布统一输出清单。
- inference/output_format/prepare_classification.py：转换已保存的预测与标注，不加载模型、不评分。
- inference/output_format/classification.py：标准结果协议及校验。
- inference/entrypoint.json：按 Example 组织的本模型命令及能力配置。
- inference/models.json：按 Example 的 models 数组组织，只含当前模型一条记录。

## 参数与原生默认值

统一提供 --phase（full/smoke/preflight）、--output-root、--show-command、--prepare-only、--device、--batch-size、--workers、--warmup、--max-images。
不传覆盖参数时，保持原脚本默认 batch_size=64、warmup=3、workers=0。
支持 --workers N，默认 0
原脚本没有已核对的受限样本流程，只支持 --phase full；smoke/--max-images 会明确报错。
--device auto 保留原生设备选择；显式 cuda 请求不可访问时会报错，不会静默回退。
cuda:N 的 N 为继承的 CUDA_VISIBLE_DEVICES 可见列表索引，入口只向子进程暴露选中的卡。
--prepare-only 只转换已经完成的同模型、同 phase 原生记录，不运行推理。

## 数据、权重与环境

Parquet 读取生成的 Hugging Face datasets 缓存默认写入服务器本地临时目录。可用 `HF_DATASETS_CACHE` 覆盖缓存路径。

权重为本模型 `weights/gianlab/swin-tiny-patch4-window7-224-finetuned-skin-cancer/` 中的真实文件。数据默认从 ../../../../dataset/ 读取。
支持 MODEL_WEIGHTS_ROOT、MODEL_DATASET_ROOT；相对覆盖路径以模型目录为基准。
结果使用 --output-root 或 MODEL_OUTPUT_ROOT，并限定在本模型 result/ 下。
依赖和逐模型环境映射见 ../../environment/README.md。

原生数据范围：HAM10000
train 分层抽取 20%，seed=42
2,003 张，非官方 test

## 输出与公共评分

原生记录位于 result/run_001/full/m135/（smoke 时为 smoke/m135/）。
保留 predictions.jsonl、status.json、timing.json、原生 metrics.json 和采集标注。
转换后每个 standardized_classification/ 包含 annotations.json、predictions.json、protocol.json，
由 result/run_001/inference_output.json 列出相对 bundle 路径。
多头模型按分支生成结果；不补造原模型缺失的 Top-5。

从 classification/ 目录执行：

~~~bash
python evaluate.py --run-dir gaoyonghan/models/gianlab--swin-tiny-patch4-window7-224-finetuned-skin-cancer/result/run_001
~~~

从项目根执行：

~~~bash
python classification/evaluate.py --run-dir classification/gaoyonghan/models/gianlab--swin-tiny-patch4-window7-224-finetuned-skin-cancer/result/run_001
~~~

评测器兼容这两种路径和绝对路径，评分规则与 Example 一致。
原生指标只用于保留原流程，统一最终指标由公共 evaluate.py 计算。

## 当前状态

2026-11-15 RTX4090全量测试通过：2003个样本，1个评分分支。统一入口和Example公共评测器均完成，结果位于 result/run_001/。
审计、历史参考及文件哈希保存在工作区 outputs/，不参与交付包运行。
