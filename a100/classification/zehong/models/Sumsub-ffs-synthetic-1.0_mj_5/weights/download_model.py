from pathlib import Path
import os
from huggingface_hub import hf_hub_download

# ================= 1. 环境与镜像配置 =================
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HUB_CACHE"] = str(Path(__file__).resolve().parents[1] / 'weights/.runtime_cache')
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# ================= 2. 目标路径配置 =================
# 注意：更新为你最新的模型存放路径
target_dir = str(Path(__file__).resolve().parents[1] / 'weights')

print("开始重新下载真实的 midjourney5M.pt 权重文件 ...")

# 使用 force_download=True 强制覆盖之前下载破损的文本文件
file_path = hf_hub_download(
    repo_id="Sumsub/Sumsub-ffs-synthetic-1.0_mj_5",
    filename="midjourney5M.pt",
    local_dir=target_dir,
    force_download=True
)

print(f"权重文件下载完成，保存至: {file_path}")