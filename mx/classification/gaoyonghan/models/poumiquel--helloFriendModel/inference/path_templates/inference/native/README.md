# 第二批：七款猫狗分类替代基线

## 一键全量

```bash
cd __MODEL_ROOT__/resources/runtime/assets/workspace/classification7_pet_eval
bash run_all.sh
```

默认 GPU 0、1 各一进程，每款先检查 10 图，通过后续跑完整 3,669 图。只验证流程：`MAX_IMAGES=10 bash run_all.sh`。单卡：`GPU_IDS=1 bash run_all.sh`。不会启动、终止或修改表格检测任务，但同时运行会共享显存和算力。

复用上一批 classification8_eval/.venv（只读，不改依赖）。若该环境不存在，setup_env.sh 在本包创建 .venv；关键版本固定，Python 3.11。自选已有兼容环境可设 EVAL_PYTHON；不匹配时报错，不向已有环境强制安装。

## 七个独立编号

| 编号 | 模型 ID |
|---|---|
| 77 | mperalsapa/helloworld_model |
| 78 | Marcblanes/iabdmarcblanes_model |
| 79 | poumiquel/helloFriendModel |
| 80 | atarensi/test_helloworldModel |
| 96 | fbarragan/helloworld_model |
| 97 | Hemant201/hmansukhaniModel |
| 98 | apineda88/helloworld_model |

各自的本地权重、revision、SHA256、配套配置和固定 revision 模型卡 URL 均在 models.json。77 显式加载 safetensors，其余显式加载各自 pytorch_model.bin，不下载或换成共同上游的权重。六个 bin 文件的 SHA256 完全一致，77 的 safetensors 与它们的 200 个 state_dict 张量也逐项 torch.equal 一致：七个 ID 是相同权重和配置。仍按项目清单逐个运行并保留七份独立模型记录，不据此声称七种独立模型；相同分数是预期现象。

## 数据与官方要求的边界

采用 [Oxford-IIIT Pet 官方资源](https://www.robots.ox.ac.uk/~vgg/data/pets/) 中 annotations/test.txt 的 3,669 张图（猫 1,183、狗 2,486），校验与本地 trainval 无 ID 交集。按第三列 SPECIES：1=猫、2=狗，映射输出 0=cat、1=dog；不使用 37 类品种编号。每类按 image_id 的 SHA256 排序后交错，前 10 图为 5 猫、5 狗，全量成员不变。

压缩包及标注内容 SHA256 已冻结。首次只解 test 图像到 .runtime/pet_test，共享一次缓存，逐图验证可解码并记录 SHA256；推理前逐图校验内容，不跳过坏图。没有下载或随机生成新的数据划分。缓存和原包留在本机。

七款实际配置一致：ViT base patch16、全图 RGB、224×224 双线性 resize、1/255 rescale、mean/std=0.5；直接调用发布者配套 AutoImageProcessor/AutoModelForImageClassification，不裁动物头部 ROI，不使用 trimap，不做测试时增强。float32、batch=8、argmax 决策，dog 为概率/AUROC正类。

七款本地模型卡都仅有简短元信息、共同上游来源，未给出与原独立测试指标完整对应的 evaluator；config 记录 Transformers 4.22.1，本包运行 4.50.0，框架差异保留。我们的脚本负责数据循环、断点、全量覆盖验证和 scikit-learn 1.5.2 指标。**全部标为 public_replacement / reference_baseline / not_officially_comparable。**

不能用 [共同上游模型卡](https://huggingface.co/omarques/autotrain-dogs-and-cats-1527055142) 的 100% 验证准确率冒充七款在 Oxford test 上的参考值；作者训练/验证图像与本替代 test 是否重合未知，不宣称严格未见数据泛化。数据页面与压缩包 README 的使用条款描述存在历史差异，后续分发图像需核对官方当前条款；本包不分发数据。

## 输出、断点与验收

默认结果：`__MODEL_ROOT__/resources/runtime/assets/workspace/results/classification7_pet_v1`。每款 m编号-名称 下保存 status.json、predictions.jsonl；只在完整覆盖 3,669 个唯一 ID 后生成 metrics.json。指标包括准确率、平衡准确率、macro-F1、dog precision/recall、dog-positive AUROC、猫/狗混淆矩阵，百分比与 0–1 单位按字段名区分。

根目录编号.log 是逐模型日志，audit_summary.json 记录状态、指标和特殊情况。_source_snapshot 保存必要五文件快照；本批统计不代替全项目 100 模型总表。冒烟仅 partial_complete，不生成正式指标；七份结果不等于七种不同权重。

重复同一命令：完整完成项校验后跳过；未完成项保留完整逐图记录并续跑，冒烟的 10 图直接用于全量，无需重复推理。损坏/重复的断点行会停止该模型，不静默丢弃。单模型失败后其他模型继续，批处理最终非零退出。相同结果目录有锁；代码或协议变化时必须通过 OUTPUT_ROOT 指定新目录，旧结果不覆盖。

必要交付仅 README.md、models.json、setup_env.sh、run_model.py、run_all.sh。查看单模型协议：`__MODEL_ROOT__/resources/runtime/assets/workspace/classification8_eval/.venv/bin/python run_model.py --model 77 --describe`。

## 本次验收（2026-09-08）

七款均通过真实 GPU 10 图测试（5 猫、5 狗），相同输入的预测逐项一致。77 号追加到 12 图，验证断点续跑；重复执行七款均正确跳过。重复 ID、损坏 JSON、错误 target 的临时测试断点均被拒绝，二分类计分函数通过合成数据单元测试。上述均非全量结果，未生成正式 metrics.json。

当前证据为 results/classification7_pet_v1/audit_summary.json 和各模型 status.json；77 为 12 图，其余为 10 图，全部 partial_complete。用户执行上方默认命令即可从这些记录继续到每款 3,669 图。elapsed_seconds 记录当前执行轮次耗时，不自动累加历史重试耗时。旧的八款分类全量结果未改动。
