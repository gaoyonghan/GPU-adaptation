#!/usr/bin/env python3
"""Add matched generic-backbone controls to four remaining diagnostics.

All routes lack same-checkpoint official references.  The controls improve
interpretability but can never change strict official acceptance.
"""

from __future__ import annotations

import gc
import json
import sys
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "evaluation_audit"))
import evaluate_b01 as b01
import evaluate_remaining_11_proxy_diagnostics as proxy

CLIP_BASE = "openai/clip-vit-large-patch14"
CLIP_BASE_REVISION = "32bd64288804d66eefd0ccbe215aa642df71cc41"
SIGLIP_BASE = "google/siglip-so400m-patch14-384"
SIGLIP_BASE_REVISION = "9fdffc58afc957d1a03a25b10dba0329ab15c2a3"
TARGETS = {
    "street": ("geolocal/StreetCLIP", "e3561ba2ad9bf14c9efd6b0092607b8497efbfea"),
    "med": ("google/medsiglip-448", "9cea28a1a1195f665105faa6e8544c112fd960a4"),
    "zh": ("lyua1225/clip-huge-zh-75k-steps-bs4096", "4db247309b7598ab786b09a84bd5a2f57b20a048"),
    "ko": ("Bingsu/clip-vit-large-patch14-ko", "278fc4af1db3edde1f73a501f167600832d0d554"),
}
SOURCE_ROOT = ROOT / "results/proxy_diagnostics/remaining_11"
PREFLIGHT = ROOT / "evaluation_audit/remaining7_control_comparisons_preflight.json"
OUT = ROOT / "results/proxy_diagnostics/enhanced/remaining7_control_comparisons/summary.json"


def now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def receipt_revision(model_id: str) -> str | None:
    receipt = json.loads((b01.model_dir(model_id) / "download_receipt.json").read_text(encoding="utf-8"))
    return receipt.get("revision") or receipt.get("resolved_revision")


def source_result(model_id: str) -> dict:
    path = SOURCE_ROOT.joinpath(*model_id.split("/"), "summary.json")
    return json.loads(path.read_text(encoding="utf-8"))


def preflight() -> tuple[dict, dict[str, dict]]:
    errors = []
    sources = {}
    for key, (model_id, expected_revision) in TARGETS.items():
        if receipt_revision(model_id) != expected_revision:
            errors.append(f"target checkpoint revision mismatch: {model_id}")
        try:
            result = source_result(model_id)
        except FileNotFoundError:
            errors.append(f"missing target diagnostic result: {model_id}")
            continue
        if result.get("status") != "completed_diagnostic_only":
            errors.append(f"target diagnostic incomplete: {model_id}")
        if result.get("model_revision") != expected_revision:
            errors.append(f"target result revision mismatch: {model_id}")
        sources[key] = result
    if receipt_revision(CLIP_BASE) != CLIP_BASE_REVISION:
        errors.append("CLIP-L/14 control revision mismatch")
    if receipt_revision(SIGLIP_BASE) != SIGLIP_BASE_REVISION:
        errors.append("SigLIP-SO400M control revision mismatch")
    try:
        assets = proxy.verify_proxy_assets([
            proxy.MODEL_SPECS[TARGETS["street"][0]],
            proxy.MODEL_SPECS[TARGETS["med"][0]],
            proxy.MODEL_SPECS[TARGETS["zh"][0]],
            proxy.MODEL_SPECS[TARGETS["ko"][0]],
        ])
    except Exception as error:
        assets = {}
        errors.append(f"proxy asset validation failed: {type(error).__name__}: {error}")

    mappings = [
        {
            "target": f"{TARGETS['street'][0]}@{TARGETS['street'][1]}",
            "control": f"{CLIP_BASE}@{CLIP_BASE_REVISION}",
            "dataset_split": "TIGER-Lab/MMEB-eval Country211 test, 1,000 images",
            "protocol_metric": "identical 211-country prompts; top-1/top-5 accuracy",
            "same_checkpoint_official_reference": None,
        },
        {
            "target": f"{TARGETS['zh'][0]}@{TARGETS['zh'][1]}",
            "control": f"{CLIP_BASE}@{CLIP_BASE_REVISION}",
            "dataset_split": "Haon-Chen/XTD-10 Chinese test, 999 image-caption pairs",
            "protocol_metric": "raw captions; text-to-image Recall@1/5/10",
            "same_checkpoint_official_reference": None,
        },
        {
            "target": f"{TARGETS['ko'][0]}@{TARGETS['ko'][1]}",
            "control": f"{CLIP_BASE}@{CLIP_BASE_REVISION}",
            "dataset_split": "Haon-Chen/XTD-10 Korean test, 999 image-caption pairs",
            "protocol_metric": "raw captions; text-to-image Recall@1/5/10",
            "same_checkpoint_official_reference": None,
        },
        {
            "target": f"{TARGETS['med'][0]}@{TARGETS['med'][1]}",
            "control": f"{SIGLIP_BASE}@{SIGLIP_BASE_REVISION}",
            "dataset_split": "PatchCamelyon original test, all 32,768 patches",
            "protocol_metric": "identical frozen pathology prompts; accuracy/balanced accuracy/macro-F1/AUROC",
            "same_checkpoint_official_reference": None,
        },
    ]
    payload = {
        "generated_at": now(),
        "status": "PASS_DIAGNOSTIC_PREFLIGHT" if not errors else "HARD_FAIL",
        "errors": errors,
        "model_to_dataset_to_reference": mappings,
        "strict_official_acceptance": False,
        "user_decision": "use substitute datasets and self-defined matched controls when official evidence is unavailable",
        "assets": assets,
    }
    atomic_json(PREFLIGHT, payload)
    print("REMAINING-7 CONTROL PREFLIGHT MODEL->DATASET->REFERENCE", flush=True)
    print(json.dumps(payload, ensure_ascii=False, indent=2), flush=True)
    if errors:
        raise RuntimeError("; ".join(errors))
    return payload, sources


def delta(target: float, control: float) -> dict:
    return {
        "target": round(target, 6),
        "control": round(control, 6),
        "absolute_delta": round(target - control, 6),
    }


def main() -> None:
    import torch

    preflight_payload, sources = preflight()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")

    clip_adapter = None
    siglip_adapter = None
    try:
        clip_adapter = b01.TransformersAdapter(CLIP_BASE)
        street_control = proxy.evaluate_country211(clip_adapter, 32)
        xtd_control = proxy.evaluate_xtd10(clip_adapter, ["zh", "ko"], 32, 128)
        del clip_adapter
        clip_adapter = None
        gc.collect()
        torch.cuda.empty_cache()

        siglip_adapter = b01.TransformersAdapter(SIGLIP_BASE)
        med_control = proxy.evaluate_pcam(siglip_adapter, 32)
        del siglip_adapter
        siglip_adapter = None
        gc.collect()
        torch.cuda.empty_cache()

        street_target = sources["street"]["result"]
        med_target = sources["med"]["result"]
        zh_target = sources["zh"]["result"]["per_language"]["zh"]
        ko_target = sources["ko"]["result"]["per_language"]["ko"]
        zh_control = xtd_control["per_language"]["zh"]
        ko_control = xtd_control["per_language"]["ko"]

        comparisons = {
            "streetclip_country211": {
                "target_model": TARGETS["street"][0],
                "control_model": CLIP_BASE,
                "target": street_target,
                "control": street_control,
                "deltas": {
                    "top1_accuracy_percent": delta(
                        street_target["top1_accuracy_percent"], street_control["top1_accuracy_percent"]
                    ),
                    "top5_accuracy_percent": delta(
                        street_target["top5_accuracy_percent"], street_control["top5_accuracy_percent"]
                    ),
                },
            },
            "chinese_clip_xtd10_zh": {
                "target_model": TARGETS["zh"][0],
                "control_model": CLIP_BASE,
                "target": zh_target,
                "control": zh_control,
                "deltas": {
                    key: delta(zh_target[key], zh_control[key]) for key in zh_target
                },
            },
            "korean_clip_xtd10_ko": {
                "target_model": TARGETS["ko"][0],
                "control_model": CLIP_BASE,
                "target": ko_target,
                "control": ko_control,
                "deltas": {
                    key: delta(ko_target[key], ko_control[key]) for key in ko_target
                },
            },
            "medsiglip_pcam": {
                "target_model": TARGETS["med"][0],
                "control_model": SIGLIP_BASE,
                "target": med_target,
                "control": med_control,
                "deltas": {
                    key: delta(med_target["metrics"][key], med_control["metrics"][key])
                    for key in (
                        "accuracy_percent", "balanced_accuracy_percent", "macro_f1_percent", "auroc_percent"
                    )
                },
            },
        }
        comparisons["streetclip_country211"]["diagnostic_supports_specialization"] = all(
            row["absolute_delta"] > 0 for row in comparisons["streetclip_country211"]["deltas"].values()
        )
        comparisons["chinese_clip_xtd10_zh"]["diagnostic_supports_specialization"] = all(
            row["absolute_delta"] > 0 for row in comparisons["chinese_clip_xtd10_zh"]["deltas"].values()
        )
        comparisons["korean_clip_xtd10_ko"]["diagnostic_supports_specialization"] = all(
            row["absolute_delta"] > 0 for row in comparisons["korean_clip_xtd10_ko"]["deltas"].values()
        )
        comparisons["medsiglip_pcam"]["diagnostic_supports_specialization"] = (
            comparisons["medsiglip_pcam"]["deltas"]["auroc_percent"]["absolute_delta"] > 0
        )

        result = {
            "status": "completed_diagnostic_only",
            "completed_at": now(),
            "strict_official_acceptance": False,
            "changes_strict_acceptance_count": False,
            "user_relaxed_acceptance": None,
            "reason_not_strict": "matched generic controls are not same-checkpoint official references",
            "preflight": preflight_payload,
            "comparisons": comparisons,
        }
        atomic_json(OUT, result)
        print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    finally:
        del clip_adapter, siglip_adapter
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
