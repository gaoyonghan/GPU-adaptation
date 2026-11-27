#!/usr/bin/env python3
"""Stage-2 per-model RTX 4090 load and one-sample inference validation.

Each model is validated in an isolated subprocess so an OOM or model-specific
failure cannot poison the remaining batch. Results are written after every
model to a dedicated workbook and to one JSON file per model.
"""

from __future__ import annotations
from model_paths import legacy as _legacy, legacy_str as _legacy_str, relative as _relative, revision as _revision, weight as _weight, result as _result

import argparse
import collections
import gc
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any

if "--worker-model-id" not in sys.argv:
    from openpyxl import Workbook, load_workbook
    from openpyxl.styles import Alignment, Font, PatternFill


ROOT = Path(__file__).resolve().parent
LEDGER = ROOT / "负责的模型.xlsx"
REPORT = ROOT / "第二阶段_4090模型验证.xlsx"
MODEL_ROOT = ROOT / "models"
RESULT_ROOT = ROOT / "results" / "stage2_4090"
WORKER_PYTHON_OVERRIDES = {
    # One-Align's bundled LLaMA implementation requires the pre-4.45 API.
    "q-future/one-align": _legacy('/home/dell/anaconda3/envs/gwit/bin/python'),
}

MODEL_BACKEND_OVERRIDES = {
    "Marqo/marqo-fashionCLIP": "OpenCLIP",
    "Marqo/marqo-fashionSigLIP": "OpenCLIP",
    "zer0int/LongCLIP-SAE-ViT-L-14": "Transformers AutoModel/AutoProcessor",
    "philschmid/clip-zero-shot-image-classification": "Transformers AutoModel/AutoProcessor",
    "google/tipsv2-b14": "Transformers audited local custom code",
    "google/tipsv2-l14": "Transformers audited local custom code",
    "google/tipsv2-g14": "Transformers audited local custom code",
    "microsoft/LLM2CLIP-EVA02-L-14-336": "Transformers audited local custom code",
    "HuggingFaceM4/siglip-so400m-14-980-flash-attn2-navit": "Transformers audited local custom code",
    "qihoo360/fg-clip2-base": "Transformers audited local custom code",
    "q-future/one-align": "Transformers audited local custom code",
    "facebook/PE-Core-L14-336": "Perception Encoder official local code",
    "facebook/PE-Core-G14-448": "Perception Encoder official local code",
    "microsoft/colipri": "COLIPRI official Python package",
    "Xenova/clip-vit-base-patch32": "ONNX Runtime CUDA",
}

DETAIL_COLUMNS = [
    "序号", "模型名称", "架构家族", "建议加载后端", "测试批次", "执行优先级",
    "本地权重大小(GB)", "文件检查", "验证状态", "实际加载后端", "CPU加载状态",
    "GPU加载状态", "单样本推理状态", "精度模式", "参数量(百万)",
    "峰值GPU显存(MB)", "CPU加载耗时(s)", "GPU迁移耗时(s)", "推理耗时(ms)",
    "输入键", "输出摘要", "开始时间", "完成时间", "错误类型", "错误摘要", "结果JSON",
]

STATUS_COLORS = {
    "通过": "C6EFCE",
    "验证中": "FFF2CC",
    "待验证": "D9EAF7",
    "待专项适配": "FCE4D6",
    "文件异常": "F4CCCC",
    "加载失败": "F4CCCC",
    "推理失败": "F4CCCC",
    "显存不足": "F4CCCC",
    "超时": "F4CCCC",
    "进程异常": "F4CCCC",
}


def now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", action="append")
    parser.add_argument("--batch", action="append")
    parser.add_argument("--retry-failed", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--initialize-only", action="store_true")
    parser.add_argument("--timeout-seconds", type=int, default=1800)
    parser.add_argument("--worker-model-id", help=argparse.SUPPRESS)
    parser.add_argument("--worker-backend", help=argparse.SUPPRESS)
    parser.add_argument("--worker-result", type=Path, help=argparse.SUPPRESS)
    return parser.parse_args()


def ledger_rows() -> list[dict[str, Any]]:
    workbook = load_workbook(LEDGER, read_only=True, data_only=True)
    sheet = workbook["模型台账"]
    values = sheet.iter_rows(values_only=True)
    headers = [str(value or "") for value in next(values)]
    return [dict(zip(headers, row)) for row in values]


def local_dir(model_id: str) -> Path:
    return MODEL_ROOT.joinpath(*model_id.split("/"))


def result_path(model_id: str) -> Path:
    return RESULT_ROOT.joinpath(*model_id.split("/")).with_suffix(".json")


def worker_python(model_id: str) -> Path:
    preferred = WORKER_PYTHON_OVERRIDES.get(model_id)
    return preferred if preferred and preferred.is_file() else Path(sys.executable)


def receipt_info(model_id: str) -> tuple[str, float]:
    directory = local_dir(model_id)
    receipt_path = directory / "download_receipt.json"
    if not receipt_path.is_file():
        return "缺少下载凭据", round(directory_size(directory) / 1024**3, 3)
    try:
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        files = receipt.get("files", [])
        missing = [name for name in files if not (directory / name).is_file()]
        if missing:
            return f"缺少{len(missing)}个文件", round(directory_size(directory) / 1024**3, 3)
        byte_count = int(receipt.get("total_bytes") or directory_size(directory))
        return "通过", round(byte_count / 1024**3, 3)
    except Exception as error:
        return f"凭据异常:{type(error).__name__}", round(directory_size(directory) / 1024**3, 3)


def directory_size(directory: Path) -> int:
    if not directory.exists():
        return 0
    return sum(path.stat().st_size for path in directory.rglob("*") if path.is_file() and ".cache" not in path.parts)


def initial_detail_rows() -> list[list[Any]]:
    output = []
    for row in ledger_rows():
        model_id = str(row["模型名称"])
        file_status, size_gb = receipt_info(model_id)
        output.append([
            row["序号"], model_id, row["架构家族"], row["建议加载后端"],
            row["测试批次"], row["执行优先级"], size_gb, file_status,
            "待验证" if file_status == "通过" else "文件异常",
            "", "未执行", "未执行", "未执行", "FP16", "", "", "", "", "",
            "", "", "", "", "", "", str(_relative(result_path(model_id),ROOT)),
        ])
    return output


def style_sheet(sheet, freeze: str = "A2") -> None:
    sheet.freeze_panes = freeze
    sheet.auto_filter.ref = sheet.dimensions
    sheet.sheet_view.zoomScale = 85
    for cell in sheet[1]:
        cell.font = Font(bold=True, color="FFFFFF")
        cell.fill = PatternFill("solid", fgColor="1F4E78")
        cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
    widths = {
        "A": 8, "B": 44, "C": 18, "D": 34, "E": 15, "F": 12, "G": 18,
        "H": 18, "I": 16, "J": 24, "K": 14, "L": 14, "M": 18, "N": 12,
        "O": 16, "P": 20, "Q": 18, "R": 18, "S": 18, "T": 30, "U": 46,
        "V": 24, "W": 24, "X": 18, "Y": 80, "Z": 56,
    }
    for letter, width in widths.items():
        sheet.column_dimensions[letter].width = width
    for row in sheet.iter_rows(min_row=2):
        for cell in row:
            cell.alignment = Alignment(vertical="top", wrap_text=True)


def save_atomic(workbook, path: Path) -> None:
    temporary = path.with_name(f".{path.stem}.tmp.xlsx")
    workbook.save(temporary)
    os.replace(temporary, path)


def initialize_report() -> None:
    existing: dict[str, dict[str, Any]] = {}
    if REPORT.exists():
        old = load_workbook(REPORT, read_only=True, data_only=True)
        if "逐模型验证" in old.sheetnames:
            sheet = old["逐模型验证"]
            rows = sheet.iter_rows(values_only=True)
            headers = list(next(rows))
            for values in rows:
                record = dict(zip(headers, values))
                existing[str(record["模型名称"])] = record

    workbook = Workbook()
    workbook.remove(workbook.active)
    detail = workbook.create_sheet("逐模型验证")
    detail.append(DETAIL_COLUMNS)
    for values in initial_detail_rows():
        model_id = str(values[1])
        if model_id in existing:
            old = existing[model_id]
            values = [old.get(column, values[index]) for index, column in enumerate(DETAIL_COLUMNS)]
        detail.append(values)
    style_sheet(detail)
    rebuild_support_sheets(workbook)
    save_atomic(workbook, REPORT)


def rebuild_support_sheets(workbook) -> None:
    detail = workbook["逐模型验证"]
    values = list(detail.iter_rows(values_only=True))
    headers = list(values[0])
    records = [dict(zip(headers, row)) for row in values[1:]]
    counts = collections.Counter(str(record["验证状态"]) for record in records)

    for name in ("阶段总览", "失败与风险", "字段说明"):
        if name in workbook.sheetnames:
            del workbook[name]

    overview = workbook.create_sheet("阶段总览", 0)
    overview.append(["指标", "当前值", "说明"])
    overview_rows = [
        ["更新时间", now(), "每完成一个模型自动刷新"],
        ["模型总数", len(records), "本阶段覆盖全部已下载模型"],
        ["通过", counts["通过"], "CPU加载、GPU加载及单样本推理均成功"],
        ["验证中", counts["验证中"], "当前独立子进程正在执行"],
        ["待验证", counts["待验证"], "尚未运行"],
        ["待专项适配", counts["待专项适配"], "自定义代码或未知后端，需先审查适配"],
        ["失败合计", sum(counts[key] for key in ("文件异常", "加载失败", "推理失败", "显存不足", "超时", "进程异常")), "可按错误类型继续处理"],
        ["验证设备", "NVIDIA GeForce RTX 4090", "实际运行时再次校验 CUDA"],
        ["精度策略", "FP16为主；mmE5使用BF16", "按模型官方推荐精度执行"],
        ["统一输入", "1张合成RGB图片 + 2条英文文本", "仅验证加载/前向链路，不代表最终精度验收"],
    ]
    for row in overview_rows:
        overview.append(row)
    style_sheet(overview)
    overview.column_dimensions["A"].width = 24
    overview.column_dimensions["B"].width = 28
    overview.column_dimensions["C"].width = 72

    risk = workbook.create_sheet("失败与风险", 2)
    risk_headers = ["序号", "模型名称", "验证状态", "建议加载后端", "测试批次", "错误类型", "错误摘要", "结果JSON"]
    risk.append(risk_headers)
    for record in records:
        if record["验证状态"] != "通过" and record["验证状态"] not in ("待验证", "验证中"):
            risk.append([record[column] for column in risk_headers])
    style_sheet(risk)

    fields = workbook.create_sheet("字段说明", 3)
    fields.append(["字段", "说明"])
    descriptions = {
        "验证状态": "通过/加载失败/推理失败/显存不足/待专项适配等最终结论。",
        "文件检查": "依据 download_receipt.json 检查所选模型文件是否存在。",
        "CPU加载状态": "模型结构和本地权重是否能在 CPU 侧完成实例化。",
        "GPU加载状态": "模型是否能迁移到 RTX 4090，并记录峰值显存。",
        "单样本推理状态": "使用一张合成图片和两条文本完成一次前向。",
        "推理耗时(ms)": "单次工程冒烟前向时间，包含一次输入，不作为性能验收。",
        "待专项适配": "不自动执行未经审查的远程代码或未知后端；后续单独适配。",
        "结果JSON": "每个模型的完整机器可读结果，便于后续昆仑芯对照。",
    }
    for field, description in descriptions.items():
        fields.append([field, description])
    style_sheet(fields)
    fields.column_dimensions["A"].width = 28
    fields.column_dimensions["B"].width = 100

    status_index = DETAIL_COLUMNS.index("验证状态") + 1
    for row_number in range(2, detail.max_row + 1):
        cell = detail.cell(row_number, status_index)
        color = STATUS_COLORS.get(str(cell.value))
        if color:
            cell.fill = PatternFill("solid", fgColor=color)


def update_report(model_id: str, result: dict[str, Any]) -> None:
    workbook = load_workbook(REPORT)
    sheet = workbook["逐模型验证"]
    headers = [cell.value for cell in sheet[1]]
    index = {str(value): position + 1 for position, value in enumerate(headers)}
    target = None
    for row_number in range(2, sheet.max_row + 1):
        if str(sheet.cell(row_number, index["模型名称"]).value) == model_id:
            target = row_number
            break
    if target is None:
        raise RuntimeError(f"报告缺少模型: {model_id}")
    mapping = {
        "文件检查": "file_check", "验证状态": "status", "实际加载后端": "actual_backend",
        "CPU加载状态": "cpu_load_status", "GPU加载状态": "gpu_load_status",
        "单样本推理状态": "inference_status", "精度模式": "precision",
        "参数量(百万)": "parameter_million", "峰值GPU显存(MB)": "peak_gpu_mb",
        "CPU加载耗时(s)": "cpu_load_seconds", "GPU迁移耗时(s)": "gpu_transfer_seconds",
        "推理耗时(ms)": "inference_ms", "输入键": "input_keys", "输出摘要": "output_summary",
        "开始时间": "started_at", "完成时间": "finished_at", "错误类型": "error_type",
        "错误摘要": "error_summary", "结果JSON": "result_json",
    }
    for column, key in mapping.items():
        if key in result:
            sheet.cell(target, index[column], result[key])
    rebuild_support_sheets(workbook)
    save_atomic(workbook, REPORT)


def update_ledger(model_id: str, result: dict[str, Any]) -> None:
    workbook = load_workbook(LEDGER)
    sheet = workbook["模型台账"]
    headers = [cell.value for cell in sheet[1]]
    index = {str(value): position + 1 for position, value in enumerate(headers) if value}
    for row_number in range(2, sheet.max_row + 1):
        if str(sheet.cell(row_number, index["模型名称"]).value) != model_id:
            continue
        status = str(result["status"])
        ledger_status = "已通过" if status == "通过" else ("待适配" if status == "待专项适配" else "未通过")
        sheet.cell(row_number, index["4090运行状态"], ledger_status)
        marker = "[第二阶段4090]"
        old = str(sheet.cell(row_number, index["备注"]).value or "")
        kept = [line for line in old.split("\n") if marker not in line]
        summary = result.get("error_summary") or result.get("output_summary") or status
        kept.append(f"{marker} {status}; {summary}; result={result.get('result_json', '')}")
        sheet.cell(row_number, index["备注"], "\n".join(line for line in kept if line))
        break
    temporary = LEDGER.with_name(f".{LEDGER.stem}.stage2-update.xlsx")
    workbook.save(temporary)
    os.replace(temporary, LEDGER)


def output_summary(outputs: Any) -> str:
    import torch

    items: list[str] = []
    if hasattr(outputs, "items"):
        iterable = outputs.items()
    elif isinstance(outputs, (tuple, list)):
        iterable = enumerate(outputs)
    else:
        iterable = [(type(outputs).__name__, outputs)]
    for key, value in iterable:
        if isinstance(value, torch.Tensor):
            items.append(f"{key}:{list(value.shape)}:{str(value.dtype).replace('torch.', '')}")
        elif isinstance(value, (tuple, list)) and value and isinstance(value[0], torch.Tensor):
            items.append(f"{key}[0]:{list(value[0].shape)}")
        if len(items) >= 8:
            break
    return "; ".join(items) or type(outputs).__name__


def patch_openclip_config(source: Path, model_id: str, destination: Path) -> tuple[str, dict[str, Any]]:
    config = json.loads(source.read_text(encoding="utf-8"))
    model_cfg = config.get("model_cfg", {})
    text_cfg = model_cfg.get("text_cfg", {})
    directory = str(local_dir(model_id))
    # Tokenizer assets are part of the downloaded model directory.  The text
    # encoder config is a different repository for BiomedCLIP, so pointing it
    # at the CLIP config makes Transformers interpret it as a ViT (KeyError
    # 'vit').  Reconstruct the standard BERT-base structure locally; all of its
    # trained parameters are then loaded from the BiomedCLIP checkpoint.
    if "hf_tokenizer_name" in text_cfg:
        text_cfg["hf_tokenizer_name"] = directory
    if model_id == "microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224":
        bert_config_dir = destination / "biomedbert_config"
        bert_config_dir.mkdir()
        bert_config = {
            "architectures": ["BertModel"], "model_type": "bert",
            "attention_probs_dropout_prob": 0.1, "hidden_act": "gelu",
            "hidden_dropout_prob": 0.1, "hidden_size": 768,
            "initializer_range": 0.02, "intermediate_size": 3072,
            "layer_norm_eps": 1e-12, "max_position_embeddings": 512,
            "num_attention_heads": 12, "num_hidden_layers": 12,
            "pad_token_id": 0, "position_embedding_type": "absolute",
            "type_vocab_size": 2, "use_cache": True, "vocab_size": 30522,
        }
        (bert_config_dir / "config.json").write_text(json.dumps(bert_config), encoding="utf-8")
        text_cfg["hf_model_name"] = str(bert_config_dir)
    safe_name = model_id.replace("/", "--").replace(".", "_")
    path = destination / f"{safe_name}.json"
    # OpenCLIP's local registry expects the inner model_cfg rather than the
    # Hugging Face open_clip_config.json wrapper.
    path.write_text(json.dumps(model_cfg), encoding="utf-8")
    return safe_name, dict(config.get("preprocess_cfg", {}))


def choose_openclip_checkpoint(directory: Path) -> Path:
    preferred = [
        "open_clip_model.safetensors", "open_clip_pytorch_model.safetensors",
        "open_clip_pytorch_model.bin", "pytorch_model.bin", "model.safetensors",
    ]
    for name in preferred:
        path = directory / name
        if path.is_file():
            return path
    candidates = sorted(list(directory.glob("*.safetensors")) + list(directory.glob("*.pt")) + list(directory.glob("*.pth")) + list(directory.glob("*.bin")))
    if not candidates:
        raise RuntimeError("未找到可加载的 OpenCLIP 权重")
    return candidates[0]


def move_inputs(inputs: dict[str, Any], device: str) -> dict[str, Any]:
    import torch

    return {key: value.to(device, non_blocking=True) if isinstance(value, torch.Tensor) else value for key, value in inputs.items()}


def nvidia_process_memory_mb() -> float | None:
    """Return this process's current NVIDIA memory when CUDA is outside PyTorch."""
    try:
        completed = subprocess.run(
            [
                "nvidia-smi", "--query-compute-apps=pid,used_gpu_memory",
                "--format=csv,noheader,nounits",
            ],
            text=True, capture_output=True, check=True,
        )
        values = []
        for line in completed.stdout.splitlines():
            fields = [field.strip() for field in line.split(",")]
            if len(fields) == 2 and int(fields[0]) == os.getpid():
                values.append(float(fields[1]))
        return sum(values) if values else None
    except Exception:
        return None


def worker_validate(model_id: str, backend: str, output: Path) -> int:
    result: dict[str, Any] = {
        "model_id": model_id, "status": "加载失败", "actual_backend": backend,
        "file_check": "通过", "cpu_load_status": "未执行", "gpu_load_status": "未执行",
        "inference_status": "未执行", "precision": "FP16", "started_at": now(),
        "finished_at": "", "error_type": "", "error_summary": "", "input_keys": "",
        "output_summary": "", "result_json": str(_relative(result_path(model_id),ROOT)),
    }
    model = None
    try:
        import psutil
        import torch
        from PIL import Image

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA不可用")
        device = "cuda:0"
        directory = local_dir(model_id)
        image = Image.new("RGB", (448, 448), color=(72, 132, 204))
        prompts = ["a photo of a blue square", "a photo of a red circle"]
        cpu_start = time.perf_counter()
        rss_before = psutil.Process().memory_info().rss

        if backend.startswith("COLIPRI"):
            from colipri import get_model, get_processor

            dependency = MODEL_ROOT / "_dependencies" / "microsoft" / "BiomedVLP-CXR-BERT-specialized"
            model = get_model(
                pretrained=False,
                **{
                    "model.text_encoder.backbone.pretrained_model_name_or_path": str(dependency),
                    "model.text_encoder.backbone.local_files_only": True,
                },
            )
            # colipri 0.1.3's accelerate loader leaves meta tensors with
            # accelerate 1.14.  Instantiate normally, then load the same local
            # safetensors checkpoint strictly to guarantee every key matches.
            from safetensors.torch import load_file
            incompatible = model.load_state_dict(load_file(directory / "model.safetensors"), strict=True)
            if incompatible.missing_keys or incompatible.unexpected_keys:
                raise RuntimeError(f"COLIPRI权重键不匹配: {incompatible}")
            model = model.eval().half()
            processor = get_processor(**{
                "processor.tokenizer.pretrained_model_name_or_path": str(dependency),
                "processor.tokenizer.local_files_only": True,
            })
            result["cpu_load_seconds"] = round(time.perf_counter() - cpu_start, 3)
            result["cpu_load_status"] = "通过"
            result["actual_backend"] = "COLIPRI 3D CT official package/local checkpoint"
            result["parameter_million"] = round(sum(p.numel() for p in model.parameters()) / 1e6, 3)
            gpu_start = time.perf_counter()
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            model.to(device)
            torch.cuda.synchronize()
            result["gpu_transfer_seconds"] = round(time.perf_counter() - gpu_start, 3)
            result["gpu_load_status"] = "通过"
            # COLIPRI is a 3D CT encoder.  A bounded synthetic CT volume tests
            # the real 192^3 image path without requiring patient data.
            ct_volume = torch.zeros((1, 1, 192, 192, 192), dtype=torch.float16, device=device)
            token_ids, attention_mask = processor.process_text(prompts)
            token_ids = token_ids.to(device)
            attention_mask = attention_mask.to(device)
            result["input_keys"] = "3D CT [1,1,192,192,192], input_ids, attention_mask"
            infer_start = time.perf_counter()
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
                outputs = {
                    "image_embeds": model.encode_image(ct_volume, pool=True, project=True, normalize=True),
                    "text_embeds": model.encode_text(token_ids, attention_mask, pool=True, normalize=True),
                }
            torch.cuda.synchronize()
        elif backend.startswith("Perception Encoder"):
            sys.path.insert(0, str(ROOT / "adapters" / "perception_models"))
            import core.vision_encoder.pe as pe
            import core.vision_encoder.transforms as pe_transforms

            config_name = model_id.split("/", 1)[1]
            checkpoint = directory / f"{config_name}.pt"
            model = pe.CLIP.from_config(
                config_name, pretrained=True, checkpoint_path=str(checkpoint),
            ).eval().half()
            preprocess = pe_transforms.get_image_transform(model.image_size)
            tokenizer = pe_transforms.get_text_tokenizer(model.context_length)
            result["cpu_load_seconds"] = round(time.perf_counter() - cpu_start, 3)
            result["cpu_load_status"] = "通过"
            result["actual_backend"] = "Meta perception_models local checkpoint"
            result["parameter_million"] = round(sum(p.numel() for p in model.parameters()) / 1e6, 3)
            gpu_start = time.perf_counter()
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            model.to(device)
            torch.cuda.synchronize()
            result["gpu_transfer_seconds"] = round(time.perf_counter() - gpu_start, 3)
            result["gpu_load_status"] = "通过"
            image_tensor = preprocess(image).unsqueeze(0).to(device)
            text_tensor = tokenizer(prompts).to(device)
            result["input_keys"] = "image, text"
            infer_start = time.perf_counter()
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
                outputs = model(image_tensor, text_tensor)
            torch.cuda.synchronize()
        elif backend == "ONNX Runtime CUDA":
            import numpy as np
            import onnxruntime as ort
            from transformers import AutoTokenizer, CLIPImageProcessor

            checkpoint = directory / "onnx" / "model_fp16.onnx"
            if not checkpoint.is_file():
                raise RuntimeError("缺少 ONNX FP16 权重")
            tokenizer = AutoTokenizer.from_pretrained(directory, local_files_only=True)
            processor = CLIPImageProcessor.from_pretrained(directory, local_files_only=True)
            # Same architecture and parameterization as openai/clip-vit-base-patch32;
            # this repository supplies its exported ONNX weights only.
            result["parameter_million"] = 151.278
            options = ort.SessionOptions()
            # ORT 1.23 can mis-optimize this older Transformers.js FP16 graph.
            options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
            providers = ort.get_available_providers()
            if "CUDAExecutionProvider" not in providers:
                raise RuntimeError(f"ONNX Runtime CUDA Provider 不可用: {providers}")
            session = ort.InferenceSession(
                str(checkpoint), sess_options=options,
                providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
            )
            if session.get_providers()[0] != "CUDAExecutionProvider":
                raise RuntimeError(f"ONNX 模型未加载到 CUDA Provider: {session.get_providers()}")
            result["cpu_load_seconds"] = round(time.perf_counter() - cpu_start, 3)
            result["cpu_load_status"] = "通过"
            result["actual_backend"] = "ONNX Runtime CUDAExecutionProvider (FP16)"
            result["gpu_transfer_seconds"] = 0.0
            result["gpu_load_status"] = "通过"
            text_inputs = tokenizer(prompts, padding=True, return_tensors="np")
            image_inputs = processor(images=image, return_tensors="np")
            inputs = {
                "input_ids": text_inputs["input_ids"].astype(np.int64),
                "attention_mask": text_inputs["attention_mask"].astype(np.int64),
                "pixel_values": image_inputs["pixel_values"].astype(np.float32),
            }
            result["input_keys"] = ", ".join(inputs.keys())
            infer_start = time.perf_counter()
            output_names = [item.name for item in session.get_outputs()]
            values = session.run(output_names, inputs)
            outputs = {
                name: torch.from_numpy(value) for name, value in zip(output_names, values)
            }
        elif backend.startswith("Transformers"):
            from transformers import AutoModel, AutoProcessor

            actual_backend = "Transformers AutoModel/AutoProcessor"
            is_tips = model_id.startswith("google/tipsv2-")
            if is_tips:
                model = AutoModel.from_pretrained(
                    directory, local_files_only=True, trust_remote_code=True,
                    torch_dtype=torch.float16, low_cpu_mem_usage=True,
                ).eval()
                processor = None
                actual_backend = "Transformers AutoModel + audited local TIPSv2 code"
            elif model_id == "microsoft/LLM2CLIP-EVA02-L-14-336":
                model = AutoModel.from_pretrained(
                    directory, local_files_only=True, trust_remote_code=True,
                    torch_dtype=torch.float16, low_cpu_mem_usage=True,
                ).eval()
                processor = None
                actual_backend = "Transformers AutoModel + audited local EVA02 vision code"
            elif model_id == "HuggingFaceM4/siglip-so400m-14-980-flash-attn2-navit":
                model = AutoModel.from_pretrained(
                    directory, local_files_only=True, trust_remote_code=True,
                    torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
                ).eval()
                processor = None
                result["precision"] = "BF16"
                actual_backend = "Transformers AutoModel + audited local NaViT SigLIP code"
            elif model_id == "qihoo360/fg-clip2-base":
                from transformers import AutoImageProcessor, AutoModelForCausalLM, AutoTokenizer

                model = AutoModelForCausalLM.from_pretrained(
                    directory, local_files_only=True, trust_remote_code=True,
                    torch_dtype=torch.float16, low_cpu_mem_usage=True,
                ).eval()
                tokenizer = AutoTokenizer.from_pretrained(directory, local_files_only=True)
                processor = AutoImageProcessor.from_pretrained(directory, local_files_only=True)
                actual_backend = "Transformers FG-CLIP2 + audited local model code"
            elif model_id == "q-future/one-align":
                from transformers import AutoConfig, AutoModelForCausalLM
                import transformers.models.llama.modeling_llama as llama_modeling

                class LegacyLlamaRotaryEmbedding(torch.nn.Module):
                    """RoPE API expected by One-Align's bundled LLaMA code."""

                    def __init__(self, dim=None, max_position_embeddings=2048, base=10000, device=None, config=None, **_: Any):
                        super().__init__()
                        if config is not None:
                            dim = config.hidden_size // config.num_attention_heads
                            base = config.rope_theta
                        inv_freq = 1.0 / (
                            base ** (torch.arange(0, dim, 2, dtype=torch.float32, device=device) / dim)
                        )
                        self.register_buffer("inv_freq", inv_freq, persistent=False)

                    def forward(self, x, position_ids=None, seq_len=None):
                        if position_ids is None:
                            position_ids = torch.arange(
                                seq_len, device=x.device, dtype=self.inv_freq.dtype,
                            ).unsqueeze(0)
                        frequencies = position_ids.to(self.inv_freq.dtype).unsqueeze(-1) * self.inv_freq
                        embedding = torch.cat((frequencies, frequencies), dim=-1)
                        return embedding.cos().to(x.dtype), embedding.sin().to(x.dtype)

                llama_modeling.LlamaRotaryEmbedding = LegacyLlamaRotaryEmbedding

                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
                config = AutoConfig.from_pretrained(
                    directory, local_files_only=True, trust_remote_code=True,
                )
                # One-Align predates the explicit bias fields added to newer
                # Transformers LLaMA configs.  Its checkpoint uses no MLP bias.
                if not hasattr(config, "mlp_bias"):
                    config.mlp_bias = False
                model = AutoModelForCausalLM.from_pretrained(
                    directory, local_files_only=True, trust_remote_code=True,
                    config=config,
                    torch_dtype=torch.float16, low_cpu_mem_usage=True,
                    attn_implementation="eager", device_map="auto",
                    max_memory={0: "18GiB", "cpu": "64GiB"}, offload_buffers=True,
                ).eval()
                # One-Align's copied LLaMA forward still reads internal flags
                # removed by newer Transformers releases.  This validation
                # deliberately uses its manual/eager attention path.
                model.model._use_flash_attention_2 = False
                model.model._use_sdpa = False
                model.config.use_cache = False
                processor = None
                actual_backend = "Transformers One-Align scorer + audited local code"
            elif model_id == "lyua1225/clip-huge-zh-75k-steps-bs4096":
                from transformers import AutoTokenizer, CLIPImageProcessor, CLIPModel

                # clip_tokenizer_roberta.py was reviewed before enabling local
                # custom code; it only adapts token IDs around the BERT tokenizer.
                tokenizer = AutoTokenizer.from_pretrained(
                    directory, local_files_only=True, trust_remote_code=True,
                )
                processor = CLIPImageProcessor.from_pretrained(directory, local_files_only=True)
                model = CLIPModel.from_pretrained(
                    directory, local_files_only=True, torch_dtype=torch.float16,
                    low_cpu_mem_usage=True,
                ).eval()
                actual_backend = "Transformers CLIPModel + audited local tokenizer"
            elif model_id == "intfloat/mmE5-mllama-11b-instruct":
                from transformers import MllamaForConditionalGeneration

                processor = AutoProcessor.from_pretrained(
                    directory, local_files_only=True, trust_remote_code=False,
                )
                # This 11B checkpoint nearly fills a 24 GB card.  Keep enough
                # room for vision activations (and coexist with the user's
                # existing GPU process) by dispatching overflow layers to CPU.
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
                model = MllamaForConditionalGeneration.from_pretrained(
                    directory, local_files_only=True, trust_remote_code=False,
                    torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
                    device_map="auto", max_memory={0: "18GiB", "cpu": "64GiB"},
                    offload_buffers=True,
                ).eval()
                result["precision"] = "BF16"
                actual_backend = "Transformers MllamaForConditionalGeneration (GPU/CPU dispatch)"
            else:
                processor = AutoProcessor.from_pretrained(directory, local_files_only=True, trust_remote_code=False)
                model = AutoModel.from_pretrained(
                    directory, local_files_only=True, trust_remote_code=False,
                    torch_dtype=torch.float16, low_cpu_mem_usage=True,
                ).eval()
            result["cpu_load_seconds"] = round(time.perf_counter() - cpu_start, 3)
            result["cpu_load_status"] = "通过"
            result["actual_backend"] = actual_backend
            result["parameter_million"] = round(sum(p.numel() for p in model.parameters()) / 1e6, 3)
            gpu_start = time.perf_counter()
            torch.cuda.empty_cache()
            if model_id not in {"intfloat/mmE5-mllama-11b-instruct", "q-future/one-align"}:
                torch.cuda.reset_peak_memory_stats()
                model.to(device)
            torch.cuda.synchronize()
            result["gpu_transfer_seconds"] = round(time.perf_counter() - gpu_start, 3)
            result["gpu_load_status"] = "通过"
            if is_tips:
                from torchvision.transforms.functional import pil_to_tensor

                size = int(model.config.img_size)
                image_tensor = pil_to_tensor(image.resize((size, size))).float().div_(255).unsqueeze(0).to(device)
                result["input_keys"] = "pixel_values, raw text strings"
                infer_start = time.perf_counter()
                with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
                    image_output = model.encode_image(image_tensor)
                    text_output = model.encode_text(prompts)
                    outputs = {"image_cls": image_output.cls_token, "text_embeds": text_output}
                torch.cuda.synchronize()
            elif model_id == "microsoft/LLM2CLIP-EVA02-L-14-336":
                from torchvision.transforms.functional import normalize, pil_to_tensor

                size = int(model.config.vision_config.image_size)
                pixel_values = pil_to_tensor(image.resize((size, size))).float().div_(255)
                pixel_values = normalize(
                    pixel_values,
                    mean=[0.48145466, 0.4578275, 0.40821073],
                    std=[0.26862954, 0.26130258, 0.27577711],
                ).unsqueeze(0).to(device)
                result["input_keys"] = "pixel_values (vision encoder checkpoint)"
                infer_start = time.perf_counter()
                with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
                    outputs = {"image_embeds": model.get_image_features(pixel_values=pixel_values)}
                torch.cuda.synchronize()
            elif model_id == "HuggingFaceM4/siglip-so400m-14-980-flash-attn2-navit":
                from transformers import AutoTokenizer

                tokenizer_dir = MODEL_ROOT / "google" / "siglip-so400m-patch14-384"
                tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir, local_files_only=True)
                text_inputs = dict(tokenizer(prompts, padding="max_length", return_tensors="pt"))
                inputs = move_inputs(text_inputs, device)
                # The model card explicitly demonstrates arbitrary NaViT grids;
                # 28x42 exercises its patch-mask path without a 4,900-token
                # quadratic attention allocation at the nominal 980px size.
                inputs["pixel_values"] = torch.rand((1, 3, 28, 42), dtype=torch.bfloat16, device=device)
                result["input_keys"] = ", ".join(inputs.keys())
                infer_start = time.perf_counter()
                with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                    outputs = model(**inputs)
                torch.cuda.synchronize()
            elif model_id == "qihoo360/fg-clip2-base":
                inputs = dict(tokenizer(prompts, padding="max_length", max_length=64, truncation=True, return_tensors="pt"))
                inputs.update(dict(processor(images=image, max_num_patches=256, return_tensors="pt")))
                inputs.pop("token_type_ids", None)
                inputs = move_inputs(inputs, device)
                result["input_keys"] = ", ".join(inputs.keys())
                infer_start = time.perf_counter()
                with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
                    outputs = model(**inputs, walk_type="short")
                torch.cuda.synchronize()
            elif model_id == "q-future/one-align":
                result["input_keys"] = "PIL image, task=quality, input=image"
                infer_start = time.perf_counter()
                with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
                    outputs = {"quality_score": model.score([image], task_="quality", input_="image")}
                torch.cuda.synchronize()
            elif model_id == "lyua1225/clip-huge-zh-75k-steps-bs4096":
                inputs = dict(tokenizer(text=prompts, padding="max_length", max_length=77, return_tensors="pt"))
                # The BERT-derived tokenizer emits segment IDs, while
                # Transformers CLIPModel accepts only IDs and attention mask.
                inputs.pop("token_type_ids", None)
                inputs.update(dict(processor(images=image, return_tensors="pt")))
            elif model_id == "intfloat/mmE5-mllama-11b-instruct":
                inputs = processor(
                    text="<|image|><|begin_of_text|>Represent the given image with the following question: What is in the image\n",
                    images=[image], return_tensors="pt",
                )
            else:
                inputs = processor(text=prompts, images=[image], return_tensors="pt", padding=True)
            if not (is_tips or model_id in {
                "microsoft/LLM2CLIP-EVA02-L-14-336",
                "HuggingFaceM4/siglip-so400m-14-980-flash-attn2-navit",
                "qihoo360/fg-clip2-base",
                "q-future/one-align",
            }):
                inputs = move_inputs(dict(inputs), device)
                result["input_keys"] = ", ".join(inputs.keys())
                infer_start = time.perf_counter()
                autocast_dtype = torch.bfloat16 if result["precision"] == "BF16" else torch.float16
                with torch.inference_mode(), torch.autocast("cuda", dtype=autocast_dtype):
                    if model_id == "intfloat/mmE5-mllama-11b-instruct":
                        # One-token logits prove the full image/text forward path
                        # while avoiding a large [sequence, vocabulary] allocation.
                        outputs = model(**inputs, use_cache=False, logits_to_keep=1)
                    else:
                        outputs = model(**inputs)
                torch.cuda.synchronize()
        elif backend == "OpenCLIP":
            import open_clip

            config_path = directory / "open_clip_config.json"
            if not config_path.is_file():
                raise RuntimeError("缺少 open_clip_config.json")
            with tempfile.TemporaryDirectory(prefix="stage2_openclip_") as temp:
                model_name, preprocess_cfg = patch_openclip_config(config_path, model_id, Path(temp))
                open_clip.add_model_config(Path(temp))
                checkpoint = choose_openclip_checkpoint(directory)
                model, _, preprocess = open_clip.create_model_and_transforms(
                    model_name, pretrained=str(checkpoint), precision="fp16", device="cpu",
                    pretrained_hf=False,
                    image_mean=tuple(preprocess_cfg["mean"]) if preprocess_cfg.get("mean") else None,
                    image_std=tuple(preprocess_cfg["std"]) if preprocess_cfg.get("std") else None,
                    image_interpolation=preprocess_cfg.get("interpolation"),
                    image_resize_mode=preprocess_cfg.get("resize_mode"),
                )
                tokenizer = open_clip.get_tokenizer(model_name)
                result["cpu_load_seconds"] = round(time.perf_counter() - cpu_start, 3)
                result["cpu_load_status"] = "通过"
                result["actual_backend"] = "OpenCLIP local config/checkpoint"
                result["parameter_million"] = round(sum(p.numel() for p in model.parameters()) / 1e6, 3)
                gpu_start = time.perf_counter()
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
                model.eval().to(device)
                torch.cuda.synchronize()
                result["gpu_transfer_seconds"] = round(time.perf_counter() - gpu_start, 3)
                result["gpu_load_status"] = "通过"
                image_tensor = preprocess(image).unsqueeze(0).to(device)
                text_tensor = tokenizer(prompts).to(device)
                result["input_keys"] = "image, text"
                infer_start = time.perf_counter()
                with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
                    outputs = model(image_tensor, text_tensor)
                torch.cuda.synchronize()
        else:
            raise RuntimeError(f"不支持的自动后端: {backend}")

        result["inference_ms"] = round((time.perf_counter() - infer_start) * 1000, 3)
        if backend == "ONNX Runtime CUDA":
            result["peak_gpu_mb"] = nvidia_process_memory_mb()
        else:
            result["peak_gpu_mb"] = round(torch.cuda.max_memory_allocated() / 1024**2, 2)
        result["inference_status"] = "通过"
        result["output_summary"] = output_summary(outputs)
        result["status"] = "通过"
        result["cpu_rss_delta_mb"] = round((psutil.Process().memory_info().rss - rss_before) / 1024**2, 2)
    except Exception as error:
        error_name = type(error).__name__
        message = str(error).replace("\n", " ")[:2000]
        result["error_type"] = error_name
        result["error_summary"] = message
        if error_name == "OutOfMemoryError" or "out of memory" in message.lower():
            result["status"] = "显存不足"
        elif result["gpu_load_status"] == "通过":
            result["status"] = "推理失败"
        else:
            result["status"] = "加载失败"
        result["traceback"] = traceback.format_exc(limit=12)
    finally:
        result["finished_at"] = now()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        try:
            del model
            gc.collect()
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
    print(json.dumps(result, ensure_ascii=False), flush=True)
    return 0 if result["status"] == "通过" else 1


def current_report_statuses() -> dict[str, str]:
    workbook = load_workbook(REPORT, read_only=True, data_only=True)
    sheet = workbook["逐模型验证"]
    rows = sheet.iter_rows(values_only=True)
    headers = list(next(rows))
    ni = headers.index("模型名称")
    si = headers.index("验证状态")
    return {str(row[ni]): str(row[si]) for row in rows}


def risk_result(model_id: str, backend: str, file_check: str) -> dict[str, Any]:
    return {
        "model_id": model_id, "file_check": file_check, "status": "待专项适配",
        "actual_backend": "未执行（安全策略）", "cpu_load_status": "未执行",
        "gpu_load_status": "未执行", "inference_status": "未执行", "precision": "FP16",
        "started_at": now(), "finished_at": now(), "error_type": "BackendNeedsReview",
        "error_summary": f"{backend}；自动批次不执行未经审查的自定义代码或未知后端",
        "result_json": str(_relative(result_path(model_id),ROOT)),
    }


def run_batch(args: argparse.Namespace) -> int:
    initialize_report()
    if args.initialize_only:
        print(f"report={REPORT}")
        return 0
    requested_ids = set(args.model_id or [])
    requested_batches = set(args.batch or [])
    rows = [row for row in ledger_rows() if (not requested_ids or str(row["模型名称"]) in requested_ids) and (not requested_batches or str(row["测试批次"]) in requested_batches)]
    rank = {"P0": 0, "P1": 1, "P2": 2, "P3": 3}
    rows.sort(key=lambda row: (rank.get(str(row["测试批次"])[:2], 9), int(row["序号"])))
    statuses = current_report_statuses()
    failures = 0
    for position, row in enumerate(rows, 1):
        model_id = str(row["模型名称"])
        backend = MODEL_BACKEND_OVERRIDES.get(model_id, str(row["建议加载后端"]))
        old_status = statuses.get(model_id, "待验证")
        if not args.force:
            if old_status == "通过":
                print(f"[{position}/{len(rows)}] SKIP passed {model_id}", flush=True)
                continue
            if old_status not in ("待验证", "验证中") and not args.retry_failed:
                print(f"[{position}/{len(rows)}] SKIP terminal={old_status} {model_id}", flush=True)
                continue
        file_check, _ = receipt_info(model_id)
        if file_check != "通过":
            result = risk_result(model_id, backend, file_check)
            result.update(status="文件异常", error_type="FileCheckError", error_summary=file_check)
        elif backend not in ("Transformers AutoModel/AutoProcessor", "OpenCLIP") and model_id not in MODEL_BACKEND_OVERRIDES:
            result = risk_result(model_id, backend, file_check)
        else:
            running = {
                "file_check": file_check, "status": "验证中", "actual_backend": backend,
                "cpu_load_status": "执行中", "gpu_load_status": "未执行", "inference_status": "未执行",
                "precision": "FP16", "started_at": now(), "finished_at": "", "error_type": "",
                "error_summary": "", "result_json": str(_relative(result_path(model_id),ROOT)),
            }
            update_report(model_id, running)
            path = result_path(model_id)
            path.parent.mkdir(parents=True, exist_ok=True)
            command = [
                str(worker_python(model_id)), str(Path(__file__).resolve()), "--worker-model-id", model_id,
                "--worker-backend", backend, "--worker-result", str(path),
            ]
            print(f"[{position}/{len(rows)}] RUN {model_id} backend={backend}", flush=True)
            try:
                completed = subprocess.run(command, text=True, capture_output=True, timeout=args.timeout_seconds)
                if path.exists():
                    result = json.loads(path.read_text(encoding="utf-8"))
                else:
                    result = risk_result(model_id, backend, file_check)
                    result.update(status="进程异常", error_type="WorkerExitError", error_summary=(completed.stderr or completed.stdout)[-2000:])
            except subprocess.TimeoutExpired as error:
                result = risk_result(model_id, backend, file_check)
                result.update(status="超时", error_type="TimeoutExpired", error_summary=f"超过{args.timeout_seconds}秒")
                path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        result["result_json"] = str(_relative(result_path(model_id),ROOT))
        if not result_path(model_id).exists():
            result_path(model_id).parent.mkdir(parents=True, exist_ok=True)
            result_path(model_id).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        update_report(model_id, result)
        update_ledger(model_id, result)
        statuses[model_id] = str(result["status"])
        if result["status"] != "通过":
            failures += 1
        print(f"[{position}/{len(rows)}] {result['status']} {model_id} {result.get('error_summary', '')}", flush=True)
    print(f"report={REPORT}", flush=True)
    print(f"non_passed_this_run={failures}", flush=True)
    return 0


def main() -> int:
    args = parse_args()
    if args.worker_model_id:
        if not args.worker_result or not args.worker_backend:
            raise SystemExit("worker arguments incomplete")
        preferred = worker_python(args.worker_model_id)
        if preferred.resolve() != Path(sys.executable).resolve():
            os.execv(
                str(preferred),
                [str(preferred), str(Path(__file__).resolve()), *sys.argv[1:]],
            )
        return worker_validate(args.worker_model_id, args.worker_backend, args.worker_result)
    return run_batch(args)


if __name__ == "__main__":
    raise SystemExit(main())
