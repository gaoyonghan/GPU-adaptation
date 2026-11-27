#!/usr/bin/env bash
set -euo pipefail

# 不让用户级 site-packages 混入各模型的独立环境。
export PYTHONNOUSERSITE=1

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
CONDA_BIN="${CONDA_BIN:-/home/ubuntu/gaoyonghan/miniconda3/bin/conda}"
ENV_ROOT="${OBJECT_DETECTION_ENV_ROOT:-/home/ubuntu/gaoyonghan/miniconda3/envs}"
SOURCE_ROOT="${OFFICIAL_SOURCE_ROOT:-/data2/gaoyonghan/gpu_model_standalone_20261110/release/model/检测/foduucom--stockmarket-future-prediction/resources/runtime/assets/workspace/official_sources/object_detection}"
PYPI_INDEX="${PYPI_INDEX:-https://pypi.org/simple}"
PYTORCH_INDEX="${PYTORCH_INDEX:-https://download.pytorch.org/whl/cu121}"
TABLE_TRANSFORMER_REVISION="${TABLE_TRANSFORMER_REVISION:-16d124f}"
YOLOV10_REVISION="${YOLOV10_REVISION:-453c6e38a51e9d1d5a2aa5fb7f1014a711913397}"
YOLOE_REVISION="${YOLOE_REVISION:-40cd606cabdbe2b566d6f14a6b162c89206e9a1b}"
YOLOV13_REVISION="${YOLOV13_REVISION:-70f23ed}"
WILDDET3D_REVISION="${WILDDET3D_REVISION:-1b8aa52}"

if [[ ! -x "$CONDA_BIN" ]]; then
  echo "找不到 conda: $CONDA_BIN" >&2
  exit 1
fi
mkdir -p "$SOURCE_ROOT"

same_repository() {
  local left="${1%.git}"
  local right="${2%.git}"
  [[ "$left" == "$right" ]]
}

remove_incomplete_partial_clone() {
  local url="$1"
  local directory="$2"
  local remote
  [[ "$directory" == "$SOURCE_ROOT/"* ]] || return 1
  [[ "$(git -C "$directory" config --get remote.origin.promisor 2>/dev/null || true)" == "true" ]] || return 1
  remote="$(git -C "$directory" remote get-url origin 2>/dev/null || true)"
  same_repository "$remote" "$url" || return 1
  [[ -z "$(find "$directory" -path "$directory/.git" -prune -o \( -type f -o -type l \) -print -quit)" ]] || return 1
  echo "发现上次网络中断留下的空工作树，将重新克隆: $directory"
  rm -rf -- "$directory"
}

remove_headless_clone() {
  local url="$1"
  local directory="$2"
  local remote
  [[ "$directory" == "$SOURCE_ROOT/"* ]] || return 1
  git -C "$directory" rev-parse --verify HEAD >/dev/null 2>&1 && return 1
  remote="$(git -C "$directory" remote get-url origin 2>/dev/null || true)"
  same_repository "$remote" "$url" || return 1
  [[ -z "$(find "$directory" -path "$directory/.git" -prune -o \( -type f -o -type l \) -print -quit)" ]] || return 1
  echo "发现上次克隆中断留下的无 HEAD 仓库，将重新克隆: $directory"
  rm -rf -- "$directory"
}

retry() {
  local attempt
  for attempt in 1 2 3; do
    if "$@"; then
      return 0
    fi
    echo "网络操作失败（$attempt/3）: $*" >&2
    [[ "$attempt" -lt 3 ]] && sleep $((attempt * 3))
  done
  return 1
}

clone_once() {
  local url="$1"
  local directory="$2"
  local revision="$3"
  local clone_ref="$4"
  local fetch_ref="$5"
  if [[ -d "$directory/.git" ]]; then
    remove_headless_clone "$url" "$directory" || true
  fi
  if [[ -d "$directory/.git" ]]; then
    remove_incomplete_partial_clone "$url" "$directory" || true
  fi
  if [[ -d "$directory/.git" ]]; then
    if ! git -C "$directory" diff --quiet || ! git -C "$directory" diff --cached --quiet; then
      echo "官方源码有未提交修改，拒绝切换版本: $directory" >&2
      exit 1
    fi
  elif [[ -e "$directory" ]]; then
    echo "目标存在但不是 Git 仓库，拒绝覆盖: $directory" >&2
    exit 1
  else
    local attempt
    for attempt in 1 2 3; do
      if git -c http.version=HTTP/1.1 clone --depth 1 --branch "$clone_ref" --single-branch "$url" "$directory"; then
        break
      fi
      echo "克隆失败（$attempt/3）: $url" >&2
      if [[ -d "$directory/.git" ]]; then
        local remote
        remote="$(git -C "$directory" remote get-url origin 2>/dev/null || true)"
        if [[ "$directory" == "$SOURCE_ROOT/"* ]] && same_repository "$remote" "$url"; then
          rm -rf -- "$directory"
        else
          echo "不安全的残留目录，拒绝自动删除: $directory" >&2
          exit 1
        fi
      elif [[ -e "$directory" && "$directory" == "$SOURCE_ROOT/"* ]]; then
        rm -rf -- "$directory"
      fi
      [[ "$attempt" -lt 3 ]] && sleep $((attempt * 3))
    done
    if [[ ! -d "$directory/.git" ]]; then
      echo "三次克隆均失败: $url" >&2
      exit 1
    fi
  fi
  if ! git -C "$directory" cat-file -e "${revision}^{commit}" 2>/dev/null; then
    retry git -c http.version=HTTP/1.1 -C "$directory" fetch --depth 256 origin "$fetch_ref"
  fi
  if ! git -C "$directory" cat-file -e "${revision}^{commit}" 2>/dev/null; then
    echo "官方引用 $fetch_ref 未包含固定 commit $revision: $directory" >&2
    exit 1
  fi
  git -C "$directory" checkout --detach "$revision"
  if [[ -f "$directory/.gitmodules" ]]; then
    retry git -c http.version=HTTP/1.1 -C "$directory" submodule update --init --recursive --depth 1
  fi
  local actual
  actual="$(git -C "$directory" rev-parse HEAD)"
  if [[ "$actual" != "$revision" && "$actual" != "$revision"* ]]; then
    echo "官方源码版本校验失败: $directory expected=$revision actual=$actual" >&2
    exit 1
  fi
  echo "官方源码已固定: $directory @ $actual"
}

clone_once https://github.com/microsoft/table-transformer.git "$SOURCE_ROOT/table-transformer" "$TABLE_TRANSFORMER_REVISION" main refs/heads/main
clone_once https://github.com/THU-MIG/yolov10.git "$SOURCE_ROOT/yolov10" "$YOLOV10_REVISION" main refs/heads/main
clone_once https://github.com/THU-MIG/yoloe.git "$SOURCE_ROOT/yoloe" "$YOLOE_REVISION" main refs/heads/main
clone_once https://github.com/iMoonLab/yolov13.git "$SOURCE_ROOT/yolov13" "$YOLOV13_REVISION" yolov13 refs/tags/yolov13
if [[ "$WILDDET3D_REVISION" != "1b8aa52" && "$WILDDET3D_REVISION" != "1b8aa52b6ff3f00d0ebfa07175efc0c0c440964a" ]]; then
  echo "WildDet3D 归档和子模块已冻结版本，请勿单独覆盖 WILDDET3D_REVISION。" >&2
  exit 1
fi
if [[ -d "$SOURCE_ROOT/WildDet3D/.git" ]]; then
  remove_headless_clone https://github.com/allenai/WildDet3D.git "$SOURCE_ROOT/WildDet3D" || true
fi
if [[ -d "$SOURCE_ROOT/WildDet3D/.git" ]]; then
  remove_incomplete_partial_clone https://github.com/allenai/WildDet3D.git "$SOURCE_ROOT/WildDet3D" || true
fi
"$(dirname "$CONDA_BIN")/python" "$SCRIPT_DIR/download_official_source.py" --source-root "$SOURCE_ROOT"
if [[ "${1:-}" == "--sources-only" ]]; then
  exit 0
fi

create_env() {
  local directory="$1"
  local version="$2"
  if [[ ! -x "$directory/bin/python" ]]; then
    "$CONDA_BIN" create -y -p "$directory" "python=$version" pip
  fi
  "$directory/bin/python" -m pip install --index-url "$PYPI_INDEX" --upgrade \
    pip setuptools==79.0.1 wheel==0.45.1
}

check_environment() {
  local directory="$1"
  local library_path="${LD_LIBRARY_PATH:-}"
  local library_dir
  for library_dir in "$directory"/lib/python*/site-packages/nvidia/*/lib; do
    [[ -d "$library_dir" ]] && library_path="$library_dir:$library_path"
  done
  LD_LIBRARY_PATH="$library_path" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 YOLO_AUTOINSTALL=false \
    YOLO_CONFIG_DIR="$directory/.config/Ultralytics" \
    "$directory/bin/python" "$SCRIPT_DIR/check_environment.py" --source-root "$SOURCE_ROOT"
}

mark_ready() {
  local directory="$1"
  "$directory/bin/python" -m pip check
  check_environment "$directory"
  touch "$directory/.object_detection_eval_v2_ready"
}

environment_ready() {
  local directory="$1"
  [[ -f "$directory/.object_detection_eval_v2_ready" ]] && \
    "$directory/bin/python" -m pip check && check_environment "$directory"
}

setup_hf() {
  local directory="$ENV_ROOT/object-detection-hf-v2"
  environment_ready "$directory" && return
  create_env "$directory" 3.11
  "$directory/bin/python" -m pip install --index-url "$PYTORCH_INDEX" torch==2.5.1 torchvision==0.20.1
  "$directory/bin/python" -m pip install --index-url "$PYPI_INDEX" \
    transformers==4.50.0 huggingface-hub==0.29.3 timm==1.0.15 numpy==1.26.4 scipy==1.12.0 \
    Pillow==11.1.0 pycocotools==2.0.8 PyYAML==6.0.2 safetensors==0.5.3 pyarrow==19.0.1 \
    PyMuPDF==1.25.3 matplotlib==3.10.0
  mark_ready "$directory"
}

setup_yolo() {
  local directory="$ENV_ROOT/object-detection-yolo-v2"
  environment_ready "$directory" && return
  create_env "$directory" 3.11
  "$directory/bin/python" -m pip install --index-url "$PYTORCH_INDEX" torch==2.5.1 torchvision==0.20.1
  "$directory/bin/python" -m pip install --index-url "$PYPI_INDEX" \
    ultralytics==8.4.0 numpy==1.26.4 scipy==1.12.0 Pillow==11.1.0 pycocotools==2.0.8 \
    PyYAML==6.0.2 pyarrow==19.0.1 onnxruntime-gpu==1.20.2
  mark_ready "$directory"
}

setup_yolov5() {
  local directory="$ENV_ROOT/object-detection-yolov5-v2"
  environment_ready "$directory" && return
  create_env "$directory" 3.10
  "$directory/bin/python" -m pip install --index-url "$PYTORCH_INDEX" torch==2.5.1 torchvision==0.20.1
  "$directory/bin/python" -m pip install --index-url "$PYPI_INDEX" \
    setuptools==79.0.1 yolov5==7.0.14 huggingface-hub==0.24.7 numpy==1.26.4 scipy==1.12.0 \
    Pillow==11.1.0 pycocotools==2.0.8 PyYAML==6.0.2
  mark_ready "$directory"
}

setup_yolov10() {
  local directory="$ENV_ROOT/object-detection-yolov10-v2"
  environment_ready "$directory" && return
  create_env "$directory" 3.9
  "$directory/bin/python" -m pip install --index-url "$PYTORCH_INDEX" torch==2.2.2 torchvision==0.17.2
  "$directory/bin/python" -m pip install --index-url "$PYPI_INDEX" --no-build-isolation \
    -e "$SOURCE_ROOT/yolov10" torch==2.2.2 torchvision==0.17.2 \
    numpy==1.26.4 scipy==1.12.0 Pillow==10.4.0 pycocotools==2.0.8 PyYAML==6.0.2 \
    huggingface-hub==0.24.7 safetensors==0.4.5 opencv-python==4.11.0.86
  mark_ready "$directory"
}

setup_yoloe() {
  local directory="$ENV_ROOT/object-detection-yoloe-v2"
  environment_ready "$directory" && return
  create_env "$directory" 3.10
  "$directory/bin/python" -m pip install --index-url "$PYTORCH_INDEX" torch==2.5.1 torchvision==0.20.1
  (
    cd "$SOURCE_ROOT/yoloe"
    "$directory/bin/python" -m pip install --index-url "$PYPI_INDEX" --no-build-isolation \
      -r requirements.txt torch==2.5.1 torchvision==0.20.1 \
      huggingface-hub==0.26.3 transformers==4.50.0 numpy==1.26.4 pycocotools==2.0.8 \
      open-clip-torch==2.26.1 supervision==0.25.1 opencv-python==4.11.0.86
  )
  if [[ ! -f "$SOURCE_ROOT/yoloe/mobileclip_blt.pt" ]]; then
    curl -L --fail --retry 3 --connect-timeout 20 \
      https://docs-assets.developer.apple.com/ml-research/datasets/mobileclip/mobileclip_blt.pt \
      -o "$SOURCE_ROOT/yoloe/mobileclip_blt.pt.part"
    mv "$SOURCE_ROOT/yoloe/mobileclip_blt.pt.part" "$SOURCE_ROOT/yoloe/mobileclip_blt.pt"
  fi
  mark_ready "$directory"
}

setup_yolov13() {
  local directory="$ENV_ROOT/object-detection-yolov13-v2"
  environment_ready "$directory" && return
  create_env "$directory" 3.11
  "$directory/bin/python" -m pip install --index-url "$PYTORCH_INDEX" torch==2.2.2 torchvision==0.17.2
  "$directory/bin/python" -m pip install --index-url "$PYPI_INDEX" --no-build-isolation \
    -e "$SOURCE_ROOT/yolov13" torch==2.2.2 torchvision==0.17.2 \
    timm==1.0.14 albumentations==2.0.4 pycocotools==2.0.7 PyYAML==6.0.1 scipy==1.13.0 \
    opencv-python==4.9.0.80 opencv-python-headless==4.9.0.80 psutil==5.9.8 py-cpuinfo==9.0.0 \
    huggingface-hub==0.23.2 safetensors==0.4.3 numpy==1.26.4 supervision==0.22.0
  mark_ready "$directory"
}

setup_rfdetr() {
  local directory="$ENV_ROOT/object-detection-rfdetr-v2"
  environment_ready "$directory" && return
  create_env "$directory" 3.11
  "$directory/bin/python" -m pip install --index-url "$PYTORCH_INDEX" torch==2.5.1 torchvision==0.20.1
  "$directory/bin/python" -m pip install --index-url "$PYPI_INDEX" \
    rfdetr==1.5.2 torch==2.5.1 torchvision==0.20.1 \
    transformers==4.50.0 peft==0.15.1 huggingface-hub==0.29.3 \
    numpy==1.26.4 Pillow==11.1.0 pyarrow==19.0.1
  mark_ready "$directory"
}

setup_tensorflow() {
  local directory="$ENV_ROOT/object-detection-tensorflow-v2"
  environment_ready "$directory" && return
  create_env "$directory" 3.10
  "$directory/bin/python" -m pip install --index-url "$PYPI_INDEX" \
    "tensorflow[and-cuda]==2.15.1" numpy==1.26.4 Pillow==10.4.0 pycocotools==2.0.8
  mark_ready "$directory"
}

setup_onnx() {
  local directory="$ENV_ROOT/object-detection-onnx-v2"
  environment_ready "$directory" && return
  create_env "$directory" 3.11
  "$directory/bin/python" -m pip install --index-url "$PYPI_INDEX" \
    transformers==4.50.0 huggingface-hub==0.29.3 onnxruntime-gpu==1.20.2 \
    numpy==1.26.4 scipy==1.12.0 Pillow==11.1.0 pycocotools==2.0.8 \
    nvidia-cuda-runtime-cu12==12.1.105 nvidia-cuda-nvrtc-cu12==12.1.105 \
    nvidia-cublas-cu12==12.1.3.1 nvidia-curand-cu12==10.3.2.106 \
    nvidia-cufft-cu12==11.0.2.54 nvidia-cudnn-cu12==9.1.0.70
  mark_ready "$directory"
}

setup_wilddet3d() {
  local directory="$ENV_ROOT/object-detection-wilddet3d-v2"
  environment_ready "$directory" && return
  create_env "$directory" 3.11
  "$directory/bin/python" -m pip install --index-url "$PYTORCH_INDEX" torch==2.5.1 torchvision==0.20.1
  local constraints=(torch==2.5.1 torchvision==0.20.1 numpy==1.26.4 setuptools==79.0.1
    transformers==4.50.0 huggingface-hub==0.29.3 timm==1.0.15)
  "$directory/bin/python" -m pip install --index-url "$PYPI_INDEX" "${constraints[@]}" vis4d==1.0.0
  "$directory/bin/python" -c 'import torch; from torch.utils.cpp_extension import CUDA_HOME; assert torch.cuda.is_available() and CUDA_HOME, "WildDet3D 编译需要可见 GPU 和 CUDA Toolkit；当前条件不满足"'
  MAX_JOBS="${MAX_JOBS:-4}" "$directory/bin/python" -m pip install --index-url "$PYPI_INDEX" --no-build-isolation --no-cache-dir \
    "$SOURCE_ROOT/_deps/vis4d_cuda_ops"
  "$directory/bin/python" -m pip install --index-url "$PYPI_INDEX" "${constraints[@]}" -r <(
    sed "s|^utils3d @ git+https://github.com/EasternJournalist/utils3d.git$|utils3d @ file://$SOURCE_ROOT/_deps/utils3d|" \
      "$SOURCE_ROOT/WildDet3D/requirements.txt"
  )
  mark_ready "$directory"
}

setup_hf
setup_yolo
setup_yolov5
setup_yolov10
setup_yoloe
setup_yolov13
setup_rfdetr
setup_tensorflow
setup_onnx
setup_wilddet3d

echo "全部 10 个隔离环境和 5 个官方源码仓库已准备完成。"
