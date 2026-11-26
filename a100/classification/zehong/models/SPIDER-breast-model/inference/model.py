from inference.options import ARGS
from inference.runtime import local_weight
"""
test.py —— 复现 histai/SPIDER-breast-model 在官方 test 集上的评估结果
====================================================================
流程：
  1) 环境/镜像配置（与模型卡一致，可下载模型权重）
  2) 加载 histai/SPIDER-breast-model（AutoModel + trust_remote_code）
  3) print_model_info: 输出模型结构与环境信息（model_structure.txt / model_environment.json）
  4) utils.evaluate_model 在 SPIDER-breast 官方 test 集（12,034 条，1120x1120 context）
     上推理，输出与原版评测一致的文件结构：
        results/<exp>_<时间戳>/
          model_structure.txt / model_environment.json
          logits.npy / predictions.csv / summary.json
  复现目标（模型卡 Results / 论文 Table 2, breast, test set, context=1120x1120）:
        Accuracy = 0.902    Precision(macro) = 0.896    F1(macro) = 0.897

可调环境变量（可选）：
  EVAL_SPLIT=test           默认评估官方 test（即模型卡所用评测集）
  EVAL_MAX_SAMPLES=100      调试时限制样本数
  EVAL_BATCH_SIZE=8         1120x1120 每批张数（每张含 25 个 224x224 patch）
  EVAL_NUM_WORKERS=4        DataLoader 进程数

运行：
  python test.py
"""
import os
import torch
from datetime import datetime
from pathlib import Path
from transformers import AutoModel
from inference.runner import evaluate_model, build_processor
from inference.runtime import print_model_and_environment
MODEL_ID = 'histai/SPIDER-breast-model'
DATASET_NAME = 'spider-breast'
EVAL_SPLIT = os.environ.get('EVAL_SPLIT', 'test')
MAX_SAMPLES = int(os.environ.get('EVAL_MAX_SAMPLES', '0') or 0) or None
BATCH_SIZE = int(os.environ.get('EVAL_BATCH_SIZE', '64'))
NUM_WORKERS = int(os.environ.get('EVAL_NUM_WORKERS', '8'))

def main():
    device = torch.device(ARGS.device)
    print(f'计算设备: {device}')
    print(f'加载模型 {MODEL_ID} ...')
    model = AutoModel.from_pretrained(str(ARGS.weights), trust_remote_code=True, local_files_only=True).to(device)
    processor = build_processor(MODEL_ID)
    exp_name = MODEL_ID.split('/')[-1]
    output_dir = ARGS.output_dir
    None
    print(f'输出目录: {output_dir.resolve()}')
    print_model_and_environment(model, output_dir)
    summary = evaluate_model(model, dataset_name=DATASET_NAME, split=EVAL_SPLIT, context_size=5, output_dir=output_dir, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, max_samples=MAX_SAMPLES, warmup_steps=3, processor=processor)

