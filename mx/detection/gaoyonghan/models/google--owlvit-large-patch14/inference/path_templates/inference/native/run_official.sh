#!/usr/bin/env bash
set -euo pipefail

export PYTHONNOUSERSITE=1
export FLAX_USE_ORBAX_CHECKPOINTING=0

ENV_PREFIX="${ENV_PREFIX:-/home/ubuntu/gaoyonghan/miniconda3/envs/owlvit-scenic-native}"
PYTHON_BIN="${ENV_PREFIX}/bin/python"
SCENIC_ROOT="${SCENIC_ROOT:-__MODEL_ROOT__/resources/runtime/assets/workspace/official_sources/scenic-08ee9dc4}"
SCENIC_REVISION="08ee9dc4288a1a103df053915fd92e755a6b05da"
GPU_ID="${GPU_ID:-0}"

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OWL_CONFIG="${OWL_CONFIG:-clip_l14}"
case "${OWL_CONFIG}" in
  clip_l14)
    MODEL_DISPLAY_NAME="OWL-ViT L/14"
    MODEL_KEY="owlvit-large-patch14"
    REPO_ID="google/owlvit-large-patch14"
    OFFICIAL_CHECKPOINT="gs://scenic-bucket/owl_vit/checkpoints/clip_vit_l14_d83d374"
    DEFAULT_HF_WEIGHTS="__MODEL_ROOT__/weights/google/owlvit-large-patch14/pytorch_model.bin"
    DEFAULT_OUTPUT_ROOT="__MODEL_ROOT__/resources/runtime/assets/workspace/results/zero_shot_native/owlvit-large-patch14"
    ;;
  clip_b32)
    MODEL_DISPLAY_NAME="OWL-ViT B/32"
    MODEL_KEY="owlvit-base-patch32"
    REPO_ID="google/owlvit-base-patch32"
    OFFICIAL_CHECKPOINT="gs://scenic-bucket/owl_vit/checkpoints/clip_vit_b32_b0203fc"
    DEFAULT_HF_WEIGHTS="__MODEL_ROOT__/weights/google/owlvit-base-patch32/pytorch_model.bin"
    DEFAULT_OUTPUT_ROOT="__MODEL_ROOT__/resources/runtime/assets/workspace/results/zero_shot_native/owlvit-base-patch32"
    ;;
  *)
    echo "不支持的 OWL_CONFIG: ${OWL_CONFIG}" >&2
    exit 2
    ;;
esac
RESULT_SUBDIR="${OWL_CONFIG}/lvis"
BPE_PATH="${BPE_PATH:-${ENV_PREFIX}/lib/python3.10/site-packages/clip/bpe_simple_vocab_16e6.txt.gz}"
BPE_SHA256="924691ac288e54409236115652ad4aa250f48203de50a9e4722a6ecd48d6804a"
DATASET_ROOT="${DATASET_ROOT:-__DATASET_ROOT__}"
ANNOTATION_ZIP="${ANNOTATION_ZIP:-${DATASET_ROOT}/lvis_v1_val/lvis_v1_val.json.zip}"
ANNOTATION_SHA256="5cae9a3c79aadb667550c2b5dcf7f4d86e059a41ec91ef690225b667e28e9ba5"
TRAIN_IMAGES_ZIP="${TRAIN_IMAGES_ZIP:-${DATASET_ROOT}/COCO2017/train2017.zip}"
VALIDATION_IMAGES_ZIP="${VALIDATION_IMAGES_ZIP:-${DATASET_ROOT}/COCO2017/val2017.zip}"
NATIVE_DATA_ROOT="${NATIVE_DATA_ROOT:-__MODEL_ROOT__/resources/runtime/assets/workspace/native_owlvit_l14/data}"
ANNOTATION_DIR="${ANNOTATION_DIR:-${NATIVE_DATA_ROOT}/annotations}"
ANNOTATION_JSON="${ANNOTATION_JSON:-${ANNOTATION_DIR}/lvis_v1_val.json}"
TFDS_DATA_DIR="${TFDS_DATA_DIR:-${NATIVE_DATA_ROOT}/tfds}"
TFDS_DOWNLOAD_DIR="${TFDS_DOWNLOAD_DIR:-${NATIVE_DATA_ROOT}/downloads}"
TFDS_REPORT="${TFDS_REPORT:-${NATIVE_DATA_ROOT}/tfds_validation_report.json}"
HF_WEIGHTS="${HF_WEIGHTS:-${DEFAULT_HF_WEIGHTS}}"
RECONSTRUCTED_CHECKPOINT_DIR="${NATIVE_DATA_ROOT}/checkpoint_hf_roundtrip"
CONVERSION_REPORT="${NATIVE_DATA_ROOT}/checkpoint_conversion_report.json"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-${RECONSTRUCTED_CHECKPOINT_DIR}}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${DEFAULT_OUTPUT_ROOT}}"

ALLOW_TFDS_DOWNLOAD="${ALLOW_TFDS_DOWNLOAD:-0}"
PREPARE_DATA_ONLY="${PREPARE_DATA_ONLY:-0}"
PREPARE_WEIGHTS_ONLY="${PREPARE_WEIGHTS_ONLY:-0}"
PREFLIGHT_ONLY="${PREFLIGHT_ONLY:-0}"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "环境不存在: ${PYTHON_BIN}" >&2
  echo "请先运行 __MODEL_ROOT__/resources/runtime/assets/workspace/native_owlvit_l14/setup_env.sh" >&2
  exit 2
fi
if [[ ! -d "${SCENIC_ROOT}/.git" ]]; then
  echo "Scenic 源码不存在: ${SCENIC_ROOT}" >&2
  exit 2
fi
CURRENT_REVISION="$(git -C "${SCENIC_ROOT}" rev-parse HEAD)"
if [[ "${CURRENT_REVISION}" != "${SCENIC_REVISION}" ]]; then
  echo "Scenic revision 不匹配: ${CURRENT_REVISION}" >&2
  exit 2
fi
if [[ ! -f "${ANNOTATION_ZIP}" ]]; then
  echo "找不到 LVIS 标注包: ${ANNOTATION_ZIP}" >&2
  exit 2
fi
if [[ ! -f "${TRAIN_IMAGES_ZIP}" ]]; then
  echo "找不到 COCO train2017 图像包: ${TRAIN_IMAGES_ZIP}" >&2
  exit 2
fi
if [[ ! -f "${VALIDATION_IMAGES_ZIP}" ]]; then
  echo "找不到 COCO val2017 图像包: ${VALIDATION_IMAGES_ZIP}" >&2
  exit 2
fi
if [[ ! -f "${BPE_PATH}" ]]; then
  echo "找不到 OpenAI CLIP 词表: ${BPE_PATH}" >&2
  exit 2
fi
ACTUAL_BPE_SHA256="$(sha256sum "${BPE_PATH}" | awk '{print $1}')"
if [[ "${ACTUAL_BPE_SHA256}" != "${BPE_SHA256}" ]]; then
  echo "OpenAI CLIP 词表 SHA256 不匹配: ${ACTUAL_BPE_SHA256}" >&2
  exit 2
fi
ACTUAL_SHA256="$(sha256sum "${ANNOTATION_ZIP}" | awk '{print $1}')"
if [[ "${ACTUAL_SHA256}" != "${ANNOTATION_SHA256}" ]]; then
  echo "LVIS 标注包 SHA256 不匹配: ${ACTUAL_SHA256}" >&2
  exit 2
fi

mkdir -p "${ANNOTATION_DIR}" "${TFDS_DATA_DIR}" "${TFDS_DOWNLOAD_DIR}"
if [[ ! -f "${ANNOTATION_JSON}" ]]; then
  TEMP_ANNOTATION="$(mktemp "${ANNOTATION_DIR}/.lvis_v1_val.json.XXXXXX")"
  trap 'rm -f "${TEMP_ANNOTATION}"' EXIT
  unzip -p "${ANNOTATION_ZIP}" lvis_v1_val.json > "${TEMP_ANNOTATION}"
  mv "${TEMP_ANNOTATION}" "${ANNOTATION_JSON}"
  trap - EXIT
fi

"${PYTHON_BIN}" - "${ANNOTATION_JSON}" <<'PY'
import json
import sys

with open(sys.argv[1], 'rb') as file:
    data = json.load(file)
images = len(data['images'])
categories = len(data['categories'])
if (images, categories) != (19809, 1203):
    raise SystemExit(f'LVIS 数据不匹配: images={images}, categories={categories}')
print(f'LVIS v1 val 已核验: images={images}, categories={categories}')
PY

prepare_tfds() {
  "${PYTHON_BIN}" "${PROJECT_ROOT}/prepare_lvis_validation_tfds.py" \
    --data-dir "${TFDS_DATA_DIR}" \
    --download-dir "${TFDS_DOWNLOAD_DIR}" \
    --train-images-zip "${TRAIN_IMAGES_ZIP}" \
    --validation-images-zip "${VALIDATION_IMAGES_ZIP}" \
    --validation-annotation-zip "${ANNOTATION_ZIP}" \
    --report "${TFDS_REPORT}"
}

prepare_weights() {
  PYTHONPATH="${SCENIC_ROOT}${PYTHONPATH:+:${PYTHONPATH}}" \
  JAX_PLATFORMS=cpu \
    "${PYTHON_BIN}" "${PROJECT_ROOT}/convert_hf_to_scenic.py" \
      --hf-weights "${HF_WEIGHTS}" \
      --output-dir "${RECONSTRUCTED_CHECKPOINT_DIR}" \
      --report "${CONVERSION_REPORT}" \
      --scenic-revision "${SCENIC_REVISION}" \
      --config "${OWL_CONFIG}" \
      --repo-id "${REPO_ID}"
}

if [[ "${CHECKPOINT_PATH}" == "${RECONSTRUCTED_CHECKPOINT_DIR}" ]] && \
   [[ ! -f "${RECONSTRUCTED_CHECKPOINT_DIR}/checkpoint_0" ]]; then
  prepare_weights
fi

if [[ "${PREPARE_WEIGHTS_ONLY}" == "1" ]]; then
  exit 0
fi

if [[ "${PREPARE_DATA_ONLY}" == "1" ]]; then
  if [[ "${ALLOW_TFDS_DOWNLOAD}" != "1" ]]; then
    echo "数据准备被保护开关阻止；确认后设置 ALLOW_TFDS_DOWNLOAD=1。" >&2
    exit 3
  fi
  prepare_tfds
  exit 0
fi

CUDA_VISIBLE_DEVICES="${GPU_ID}" "${PYTHON_BIN}" - "${CHECKPOINT_PATH}" <<'PY'
import pathlib
import sys

import jax
import tensorflow as tf

checkpoint = sys.argv[1]
devices = jax.devices()
print('JAX devices:', devices)
if not any(device.platform == 'gpu' for device in devices):
    raise SystemExit('JAX 没有识别到 GPU。')

try:
    exists = tf.io.gfile.exists(checkpoint)
    matches = tf.io.gfile.glob(checkpoint + '*') if not exists else [checkpoint]
except Exception as error:
    raise SystemExit(f'无法访问原生 checkpoint {checkpoint}: {type(error).__name__}: {error}')
if not exists and not matches:
    if checkpoint.startswith('gs://'):
        raise SystemExit(
            '官方 GCS checkpoint 当前不可访问。请配置 Google 凭据，或用 '
            'CHECKPOINT_PATH 指向已核验的同一原生 checkpoint；不会自动改用 HF 权重。'
        )
    raise SystemExit(f'本地 checkpoint 不存在: {checkpoint}')
if not checkpoint.startswith('gs://') and not pathlib.Path(checkpoint).exists():
    raise SystemExit(f'本地 checkpoint 不存在: {checkpoint}')
print('原生 checkpoint 可访问:', checkpoint)
PY

TFDS_READY="$(${PYTHON_BIN} - "${TFDS_DATA_DIR}" <<'PY'
import sys
import tensorflow_datasets as tfds

builder = tfds.builder('lvis', data_dir=sys.argv[1])
ready = (
    'validation' in builder.info.splits
    and builder.info.splits['validation'].num_examples == 19809
    and builder.info.features['objects']['label'].num_classes == 1203
)
print('1' if ready else '0')
PY
)"
if [[ "${TFDS_READY}" != "1" ]]; then
  if [[ "${ALLOW_TFDS_DOWNLOAD}" == "1" ]]; then
    prepare_tfds
  else
    echo "TFDS lvis 尚未准备。先运行：" >&2
    echo "ALLOW_TFDS_DOWNLOAD=1 PREPARE_DATA_ONLY=1 bash $0" >&2
    exit 3
  fi
fi

if [[ "${PREFLIGHT_ONLY}" == "1" ]]; then
  echo "预检通过：原生 checkpoint、JAX GPU、Scenic revision 和 TFDS LVIS 均已就绪。"
  exit 0
fi

EXISTING_MANIFEST="${OUTPUT_ROOT}/run_manifest.json"
EXISTING_PREDICTIONS="${OUTPUT_ROOT}/${RESULT_SUBDIR}/predictions_validation.json"
if [[ -f "${EXISTING_MANIFEST}" ]]; then
  EXISTING_STATUS="$("${PYTHON_BIN}" - "${EXISTING_MANIFEST}" <<'PY'
import json
import sys

with open(sys.argv[1], encoding='utf-8') as file:
    print(json.load(file).get('status', 'unknown'))
PY
)"
  if [[ "${EXISTING_STATUS}" == "completed" ]]; then
    echo "已有完整结果，按保留规则跳过: ${OUTPUT_ROOT}"
    echo "如需再次运行，请用 OUTPUT_ROOT 指向新的结果目录。"
    exit 0
  fi
  echo "输出目录已有状态为 ${EXISTING_STATUS} 的记录；为防止覆盖，请人工核查或改用新的 OUTPUT_ROOT。" >&2
  exit 4
fi
if [[ -f "${EXISTING_PREDICTIONS}" ]]; then
  echo "输出目录已有预测文件但没有清单；为防止覆盖，请人工核查或改用新的 OUTPUT_ROOT。" >&2
  exit 4
fi

mkdir -p "${OUTPUT_ROOT}"
if [[ "${CHECKPOINT_PATH}" == "${OFFICIAL_CHECKPOINT}" ]]; then
  WEIGHT_PROVENANCE="author_native"
elif [[ "${CHECKPOINT_PATH}" == "${RECONSTRUCTED_CHECKPOINT_DIR}" ]]; then
  WEIGHT_PROVENANCE="official_hf_reverse_converted_roundtrip_exact"
else
  WEIGHT_PROVENANCE="user_supplied_requires_verification"
fi

COMMAND=(
  "${PYTHON_BIN}" "${PROJECT_ROOT}/launch_official_evaluator.py"
  --alsologtostderr=true
  "--config=${SCENIC_ROOT}/scenic/projects/owl_vit/configs/${OWL_CONFIG}.py"
  "--checkpoint_path=${CHECKPOINT_PATH}"
  "--output_dir=${OUTPUT_ROOT}"
  --tfds_name=lvis
  --split=validation
  "--annotations_path=${ANNOTATION_JSON}"
  --data_format=lvis
  --platform=gpu
  "--tfds_data_dir=${TFDS_DATA_DIR}"
  "--tfds_download_dir=${TFDS_DOWNLOAD_DIR}"
  --num_example_images_to_save=0
  --label_shift=1
)

"${PYTHON_BIN}" - "${OUTPUT_ROOT}/run_manifest.json" "${CURRENT_REVISION}" \
  "${CHECKPOINT_PATH}" "${WEIGHT_PROVENANCE}" "${TFDS_DATA_DIR}" "${ACTUAL_SHA256}" \
  "${CONVERSION_REPORT}" "${TFDS_REPORT}" "${HF_WEIGHTS}" "${BPE_PATH}" \
  "${ACTUAL_BPE_SHA256}" "${OWL_CONFIG}" "${MODEL_DISPLAY_NAME}" "${REPO_ID}" \
  "${COMMAND[@]}" <<'PY'
import datetime
import importlib.metadata as md
import json
import pathlib
import sys

manifest_path, revision, checkpoint, provenance, tfds_dir, annotation_sha, conversion_report, tfds_report, hf_weights, bpe_path, bpe_sha, owl_config, model_name, repo_id, *command = sys.argv[1:]

def load_json(path):
    candidate = pathlib.Path(path)
    return json.loads(candidate.read_text(encoding='utf-8')) if candidate.is_file() else None

references = {
    'clip_l14': [
        {'id': 'scenic_readme_checkpoint_table',
         'metrics': {'AP': 34.6, 'APr': 31.2}},
    ],
    'clip_b32': [
        {'id': 'clip_b32_config_docstring',
         'metrics': {'AP': 20.9, 'APr': 16.9}},
        {'id': 'scenic_readme_checkpoint_table',
         'metrics': {'AP': 19.3, 'APr': 16.9}},
    ],
}[owl_config]

record = {
    'status': 'running',
    'started_at': datetime.datetime.now(datetime.timezone.utc).isoformat(),
    'model': model_name,
    'repo_id': repo_id,
    'weight_provenance': provenance,
    'execution_path': 'upstream_evaluator',
    'environment_fidelity': 'reconstructed_compatible_pins',
    'compatibility_adjustments': [
        'lvis 0.5.3: replace removed np.float dtype alias with builtin float; numerical behavior unchanged',
        'TFDS materialization narrowed to validation only; upstream LVIS example generator and evaluator input are unchanged',
        'tensorflow-cpu provides import tensorflow; Scenic package metadata still reports tensorflow distribution as absent',
        'verified local OpenAI CLIP vocabulary injected to bypass the 2022 helper re-downloading an existing identical file'
    ],
    'result_validation': 'pending',
    'scenic_revision': revision,
    'checkpoint_path': checkpoint,
    'checkpoint_conversion': load_json(conversion_report),
    'source_hf_weights': hf_weights,
    'tokenizer_vocabulary': {'path': bpe_path, 'sha256': bpe_sha},
    'dataset': {'tfds_name': 'lvis', 'split': 'validation', 'tfds_data_dir': tfds_dir,
                'images': 19809, 'categories': 1203,
                'annotation_zip_sha256': annotation_sha,
                'preparation': load_json(tfds_report)},
    'protocol': {'config': owl_config, 'data_format': 'lvis', 'label_shift': 1,
                 'top_k': 300, 'metric': 'official lvis-api bbox'},
    'official_reference_percent': references[0]['metrics'],
    'official_references_percent': references,
    'versions': {name: md.version(name) for name in
                 ('jax', 'jaxlib', 'flax', 'tensorflow-cpu', 'tensorflow-datasets', 'lvis')},
    'command': command,
}
with open(manifest_path, 'w', encoding='utf-8') as file:
    json.dump(record, file, ensure_ascii=False, indent=2)
PY

set +e
CUDA_VISIBLE_DEVICES="${GPU_ID}" \
XLA_PYTHON_CLIENT_PREALLOCATE=false \
OWL_VIT_BPE_PATH="${BPE_PATH}" \
PYTHONPATH="${SCENIC_ROOT}${PYTHONPATH:+:${PYTHONPATH}}" \
  "${COMMAND[@]}"
RUN_CODE=$?
set -e

"${PYTHON_BIN}" - "${OUTPUT_ROOT}" "${RUN_CODE}" "${OWL_CONFIG}" <<'PY'
import collections
import datetime
import hashlib
import json
import math
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
run_code = int(sys.argv[2])
owl_config = sys.argv[3]
result_dir = root / owl_config / 'lvis'
manifest_path = root / 'run_manifest.json'
record = json.loads(manifest_path.read_text(encoding='utf-8'))
record['finished_at'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
record['exit_code'] = run_code
record['status'] = 'failed' if run_code else 'completed'

def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as file:
        for chunk in iter(lambda: file.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()

if run_code == 0:
    results = sorted(result_dir.glob('results_validation.json'),
                     key=lambda path: path.stat().st_mtime)
    if not results:
        record['status'] = 'failed'
        record['result_validation'] = 'reference_unresolved'
        record['error'] = '官方 evaluator 返回 0，但未找到 results_validation.json'
    else:
        result_path = results[-1]
        metrics = json.loads(result_path.read_text(encoding='utf-8'))
        record['official_results_file'] = str(result_path)
        record['official_results_sha256'] = sha256(result_path)
        record['metrics_raw'] = metrics
        measured = {}
        for key in ('AP', 'APr'):
            if key in metrics:
                value = float(metrics[key])
                measured[key] = value * 100.0 if abs(value) <= 1.0 else value
        record['metrics_percent'] = measured
        comparisons = []
        for reference in record['official_references_percent']:
            expected = reference['metrics']
            differences = {
                key: measured[key] - expected[key]
                for key in expected if key in measured
            }
            complete = all(key in measured for key in expected)
            comparisons.append({
                'reference_id': reference['id'],
                'metrics_percent': expected,
                'difference_percent_points': differences,
                'within_tolerance': (
                    complete and
                    all(abs(value) <= 1.0 for value in differences.values())
                ),
            })
        record['reference_comparisons'] = comparisons
        matches = [item for item in comparisons if item['within_tolerance']]
        if matches:
            record['result_validation'] = 'within_declared_tolerance'
            record['matched_official_reference_ids'] = [
                item['reference_id'] for item in matches
            ]
            record['difference_percent_points'] = matches[0][
                'difference_percent_points'
            ]
        elif comparisons and all(
                len(item['difference_percent_points']) ==
                len(item['metrics_percent']) for item in comparisons):
            record['result_validation'] = 'outside_declared_tolerance'
            record['difference_percent_points'] = comparisons[0][
                'difference_percent_points'
            ]
        else:
            record['result_validation'] = 'reference_unresolved'

    prediction_path = result_dir / 'predictions_validation.json'
    if not prediction_path.is_file():
        record['status'] = 'failed'
        record['prediction_validation'] = {
            'status': 'failed',
            'error': '未找到 predictions_validation.json',
        }
    else:
        predictions = json.loads(prediction_path.read_text(encoding='utf-8'))
        per_image = collections.Counter()
        category_ids = set()
        invalid_numeric_records = 0
        for prediction in predictions:
            per_image[int(prediction['image_id'])] += 1
            category_ids.add(int(prediction['category_id']))
            bbox = prediction['bbox']
            score = prediction['score']
            if (len(bbox) != 4 or
                    any(not math.isfinite(float(value)) for value in bbox) or
                    not math.isfinite(float(score))):
                invalid_numeric_records += 1
        prediction_ok = (
            len(per_image) == 19809 and
            per_image and
            max(per_image.values()) <= 300 and
            invalid_numeric_records == 0
        )
        record['prediction_validation'] = {
            'status': 'verified' if prediction_ok else 'failed',
            'file': str(prediction_path),
            'file_size_bytes': prediction_path.stat().st_size,
            'sha256': sha256(prediction_path),
            'predictions': len(predictions),
            'unique_images': len(per_image),
            'min_per_image': min(per_image.values()) if per_image else 0,
            'max_per_image': max(per_image.values()) if per_image else 0,
            'category_id_min': min(category_ids) if category_ids else None,
            'category_id_max': max(category_ids) if category_ids else None,
            'unique_categories_with_predictions': len(category_ids),
            'invalid_numeric_records': invalid_numeric_records,
            'upstream_class_zero_filter': (
                'Scenic 08ee9dc format_predictions uses `if not label: continue`; '
                'with label_shift=1 this intentionally omits category_id=1.'
            ),
        }
        if not prediction_ok:
            record['status'] = 'failed'

manifest_path.write_text(json.dumps(record, ensure_ascii=False, indent=2) + '\n',
                         encoding='utf-8')
PY

if [[ "${RUN_CODE}" -ne 0 ]]; then
  echo "官方 Scenic evaluator 失败，退出码: ${RUN_CODE}" >&2
  exit "${RUN_CODE}"
fi
echo "${MODEL_DISPLAY_NAME} Scenic 上游评估完成: ${OUTPUT_ROOT}"
