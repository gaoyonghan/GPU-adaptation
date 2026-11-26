# 掩码运行环境

`mask-eval.txt` 是本次从服务器已有掩码环境导出的依赖版本记录，不是宣称所有作者发布了同一个环境。
Python 版本为 3.10，当前 PyTorch 为 2.12.0+cu126，torchvision 为 0.27.0+cu126，NumPy 为 1.26.4。

先创建或激活自己的掩码环境，再从项目根目录安装记录的依赖：

```bash
python -m pip install -r mask-generation/lirongfu/environment/mask-eval.txt --extra-index-url https://download.pytorch.org/whl/cu126
python -m pip check
```

当前服务器已有对应环境，无需重复安装。Node.js 不由 pip 管理，两个 ONNX 模型需要 Node.js 20+；使用 nvm 时先执行 `nvm use 20`。
OpenCV 还依赖主机可用的系统共享库；系统依赖不包含在 pip freeze 中。

model_environment.csv 列出 16 个模型与这份依赖文件的对应关系。模型权重、官方原生推理依赖代码和配置已经保存在各模型的权重包与 inference/ 中，不依赖旧 Model/ 目录。
代码不会自动安装依赖，也不会自动下载新数据集。
