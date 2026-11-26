# 95号：作者test与AutoTrain原生评测

```bash
cd __MODEL_ROOT__/resources/runtime/assets/workspace/classification8_eval
DEVICE=0 bash run_m95_test.sh
```

仅下载固定版本haywoodsloan/ai-images的data/test，包含set-000至set-003；不下载train、不重新训练、不另换权重。下载前先用已保存HF凭据检查实际文件权限，401/403立即停止。浏览器能列出文件不代表已获下载权限。若未获批，应在数据集页面申请；获批后用下列命令交互登录同一账号，不要在命令行或聊天中粘贴Token：

```bash
__MODEL_ROOT__/resources/runtime/assets/workspace/classification8_eval/.venv/bin/python -c 'from huggingface_hub import login; login(add_to_git_credential=False)'
```

`bash run_m95_test.sh --download`只下载；`--preflight`检查已下载test、环境、原生预处理及CPU加载，不做前向；`--eval`只评测已下载数据。`MAX_IMAGES=10`只冒烟；默认两类各5张冒烟后完整test。下载可重试并复用已验证文件；推理失败不自动续算，每次另建结果。断图不静默跳过。日志/结果在`__MODEL_ROOT__/resources/runtime/assets/workspace/results/m95_author_test_v1/`；保留之前classification8_v1替代集结果。

## 官方依据及边界

- 模型同一固定revision的`training_params.json`明确指定`data_path=haywoodsloan/ai-images`、`valid_split=test`、batch_size=32、mixed_precision=null。比作者当前GitHub配置更直接；本入口以权重随附参数为准，不用当前模板的BF16/batch12。
- AutoTrain使用`per_device_eval_batch_size=2*batch_size`，因此评测batch64、FP32。沿用原始`ImageClassificationDataset`、`process_data`、`_binary_classification_metrics`，由Transformers `Trainer.evaluate`计算loss和指标。未执行原始main的训练、上传或保存权重分支。
- 原生验证预处理是Albumentations Resize(256,256，默认OpenCV线性插值)+Normalize，不是旧流程ViTImageProcessor的PIL三次插值。HF datasets ImageFolder加载本地test，类名与模型严格对齐：artificial=0、real=1。
- 原生F1/Precision/Recall正类是real=1；AUC使用原始logits[:,1]，不是softmax概率。不得把该AUC与旧实验的artificial概率AUC直接混用。
- 源码固定在AutoTrain d137c8c3477a52f4a5d9082b50cdf6951470a3e4（模型日期前image_classification最后一次变更），原文件随脚本保存并校验SHA256。不是作者明确提供的AutoTrain环境锁文件。Transformers4.48.0与权重config一致；相关AutoTrain依赖固定，torch/CUDA复用现有环境，隔离安装不改共享环境。
- 模型卡参考：accuracy=98.1488090989126%、F1=0.9875742669136907、precision=0.9817413946399085、recall=0.9934768637532133、AUC=0.9953790023764837、loss=0.08581268042325974。
- 已确认作者数据来源和test作为valid的用途，但当前数据revision是否与历史训练时逐文件相同尚未确认。因此报告为official_protocol_adapter，不冒充exact_official。未下载train，不能据此声明不存在训练/测试重叠。

我们补写的部分仅是：test专用下载及清单校验、离线数据路径绑定、选取原生函数的加载入口、冒烟调度与逐图审计。原函数体不改。真实数据权限尚未通过时，不能宣称已下载或GPU评测通过。
