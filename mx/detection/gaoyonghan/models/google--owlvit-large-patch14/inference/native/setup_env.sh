#!/usr/bin/env bash
set -euo pipefail

export PYTHONNOUSERSITE=1

CONDA_BIN="${CONDA_BIN:-$(command -v conda || true)}"
RUNTIME_ROOT="${BENCH_RUNTIME_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)/.runtime}"
ENV_PREFIX="${ENV_PREFIX:-${RUNTIME_ROOT}/envs/e27-owlvit-scenic-native}"
SOURCE_ROOT="${SOURCE_ROOT:-${RUNTIME_ROOT}/assets/workspace/official_sources}"
SCENIC_ROOT="${SCENIC_ROOT:-${SOURCE_ROOT}/scenic-08ee9dc4}"
SCENIC_REPO="https://github.com/google-research/scenic.git"
SCENIC_REVISION="08ee9dc4288a1a103df053915fd92e755a6b05da"
CLIP_REVISION="dcba3cb2e2827b402d2701e7e1c7d9fed8a20ef1"
PYPI_INDEX_URL="${PYPI_INDEX_URL:-https://pypi.org/simple}"
JAX_WHEEL_INDEX="${JAX_WHEEL_INDEX:-https://storage.googleapis.com/jax-releases/jax_cuda_releases.html}"
PYTORCH_CPU_INDEX="${PYTORCH_CPU_INDEX:-https://download.pytorch.org/whl/cpu}"

if [[ ! -x "${CONDA_BIN}" ]]; then
  echo "找不到 conda: ${CONDA_BIN}" >&2
  exit 2
fi

if [[ ! -x "${ENV_PREFIX}/bin/python" ]]; then
  "${CONDA_BIN}" create --prefix "${ENV_PREFIX}" python=3.10 pip -y
fi
PYTHON_BIN="${ENV_PREFIX}/bin/python"

mkdir -p "${SOURCE_ROOT}"
if [[ ! -e "${SCENIC_ROOT}" ]]; then
  git clone --no-checkout "${SCENIC_REPO}" "${SCENIC_ROOT}"
  git -C "${SCENIC_ROOT}" checkout --detach "${SCENIC_REVISION}"
elif [[ ! -d "${SCENIC_ROOT}/.git" ]]; then
  echo "目标已存在但不是 Git 仓库: ${SCENIC_ROOT}" >&2
  exit 2
fi

CURRENT_REVISION="$(git -C "${SCENIC_ROOT}" rev-parse HEAD)"
if [[ "${CURRENT_REVISION}" != "${SCENIC_REVISION}" ]]; then
  echo "现有 Scenic 源码不是固定 revision: ${CURRENT_REVISION}" >&2
  echo "请改用空的 SCENIC_ROOT，或先人工核查该目录；脚本不会覆盖现有源码。" >&2
  exit 2
fi

export PIP_INDEX_URL="${PYPI_INDEX_URL}"
PIP=("${PYTHON_BIN}" -m pip)
"${PIP[@]}" install --upgrade "setuptools==69.5.1" "wheel==0.43.0"

# RTX 4090 使用 CUDA 12 JAX wheel；TensorFlow 只负责 TFDS，安装 CPU 版以免占用显存。
"${PIP[@]}" install --find-links "${JAX_WHEEL_INDEX}" \
  "jax[cuda12_pip]==0.4.23" \
  "nvidia-cublas-cu12==12.2.5.6" \
  "nvidia-cuda-cupti-cu12==12.2.142" \
  "nvidia-cuda-nvcc-cu12==12.2.140" \
  "nvidia-cuda-nvrtc-cu12==12.2.140" \
  "nvidia-cuda-runtime-cu12==12.2.140" \
  "nvidia-cudnn-cu12==8.9.7.29" \
  "nvidia-cufft-cu12==11.0.8.103" \
  "nvidia-cusolver-cu12==11.5.2.141" \
  "nvidia-cusparse-cu12==12.1.2.141" \
  "nvidia-nccl-cu12==2.18.3" \
  "nvidia-nvjitlink-cu12==12.2.140"
"${PIP[@]}" install \
  "numpy==1.24.4" \
  "ml-dtypes==0.3.2" \
  "jax==0.4.23" \
  "flax==0.8.2" \
  "orbax-checkpoint==0.5.9" \
  "optax==0.2.2" \
  "chex==0.1.86" \
  "clu==0.0.12" \
  "tensorflow-cpu==2.15.1" \
  "tensorflow-datasets==4.9.4" \
  "tensorflow-metadata==1.15.0" \
  "apache-beam==2.54.0" \
  "ml-collections==0.1.1" \
  "immutabledict==4.2.0" \
  "ott-jax==0.3.1" \
  "lvis==0.5.3" \
  "pycocotools==2.0.7" \
  "matplotlib==3.7.5" \
  "opencv-python-headless==4.8.1.78" \
  "scipy==1.10.1" \
  "pillow==10.3.0" \
  "tqdm==4.66.4" \
  "ftfy==6.2.0" \
  "regex==2024.5.15"

# lvis 0.5.3 still uses the removed np.float alias in two dtype conversions.
# Replacing it with builtin float is numerically equivalent and is required for NumPy >=1.24.
LVIS_EVAL="${ENV_PREFIX}/lib/python3.10/site-packages/lvis/eval.py"
sed -i 's/dtype=np\.float)/dtype=float)/g' "${LVIS_EVAL}"

# CLIP 在这里仅用于 Scenic 的官方 tokenizer/model implementation，固定到明确 revision。
"${PYTHON_BIN}" -m pip install \
  --index-url "${PYTORCH_CPU_INDEX}" \
  --extra-index-url "${PYPI_INDEX_URL}" \
  "torch==2.1.2+cpu" "torchvision==0.16.2+cpu"
"${PIP[@]}" install --no-deps --no-build-isolation \
  "git+https://github.com/openai/CLIP.git@${CLIP_REVISION}"

# 避免 Scenic setup.py 拉入与 OWL-ViT evaluator 无关的 Lingvo 等全项目依赖。
"${PIP[@]}" install --no-deps --editable "${SCENIC_ROOT}"

"${PYTHON_BIN}" - <<'PY'
import importlib.metadata as md
import jax
import tensorflow as tf
import tensorflow_datasets as tfds

print('环境安装完成')
for name in ('jax', 'jaxlib', 'flax', 'tensorflow-cpu', 'tensorflow-datasets', 'lvis'):
    print(f'{name}={md.version(name)}')
print('jax_devices=', jax.devices())
print('tensorflow=', tf.__version__, 'tfds=', tfds.__version__)
if not any(device.platform == 'gpu' for device in jax.devices()):
    raise SystemExit('JAX 没有识别到 GPU，请先解决 CUDA/JAX 环境再运行全量评估。')
PY

echo "Scenic revision: ${CURRENT_REVISION}"
echo "环境路径: ${ENV_PREFIX}"
