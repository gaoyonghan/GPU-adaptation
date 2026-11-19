#!/usr/bin/env python3
"""Rebuild the 31-result, source-backed comparison without changing any evaluation result."""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path

GITHUB = "https://github.com/"
PAPER = "https://arxiv.org/html/"
# References were checked on 2026-09-06. None of the tolerances below are author acceptance criteria.
# sequence: (comparison metric, released-checkpoint reference, release source, paper reference, paper URL, caveat)
REFS = {
    1: ("AP", 42.0, GITHUB+"facebookresearch/detr", 42.0, PAPER+"2005.12872", "主 AP 接近；论文 AP_small=20.5，本次21.342，不能声称所有分项完全一致。"),
    2: ("AP", 97.0, GITHUB+"microsoft/table-transformer", 96.6, PAPER+"2110.00061", "发布仓库97.0与论文表3的96.6不同；HF预处理与作者PIL缩放存在实现差异，不把任一数字当作精确等价。"),
    4: ("AP", 37.3, "https://huggingface.co/Ultralytics/YOLOv8", None, None, "当前predict逐图+COCOeval，不是模型卡指定的yolo val；差值不能直接归咎权重。"),
    5: ("AP", 28.7, GITHUB+"hustvl/YOLOS", 28.7, PAPER+"2106.00666", "Ti，512短边；不是其他分辨率/蒸馏变体。"),
    7: ("AP", 39.5, "https://huggingface.co/Ultralytics/YOLO11", None, None, "使用nano权重；当前predict+COCOeval与官方val不同；不借用其他论文的YOLO11对照行38.6。"),
    10: ("AP", None, GITHUB+"valentinafevu/fine_tunning_YOLOS_for_fashion", None, None, "模型卡及作者训练示例未找到对应已发布权重的全验证集AP；不是YOLOS原论文COCO结果，也不是Fashionpedia属性联合指标。"),
    11: ("AP", 43.5, GITHUB+"facebookresearch/detr", 43.5, PAPER+"2005.12872", "R101非DC5，COCO val2017。"),
    13: ("AP", 40.1, "https://huggingface.co/Ultralytics/YOLO26", None, None, "已检查本地权重end2end=True；对应40.1，不是标准分支40.9。主数值接近但尚未沿用官方val全链路。"),
    14: ("AP", 36.1, GITHUB+"hustvl/YOLOS", 36.1, PAPER+"2106.00666", "Small，800短边；不是dWr或蒸馏37.2行。"),
    16: ("AP", 67.35, "https://huggingface.co/tech4humans/yolov8s-signature-detector", None, None, "模型卡最终Results及metadata为67.35；调参表另有66.26。本次ONNX test419张，AP50也低于94.499；需原版validator定位。"),
    18: ("AP50", 98.82982754936463, "https://huggingface.co/keremberke/yolov5m-license-plate", None, None, "模型卡公布AP50，不是AP50-95；validation；本次YOLOv5推理API+COCOeval，非原训练validator。"),
    20: ("AP", 38.5, GITHUB+"THU-MIG/yolov10", 38.5, PAPER+"2405.14458", "kadirnar仓库实际选yolov10n.pt；与m49是两个模型清单条目，仍分别保留结果。非官方batch256的val链路。"),
    21: ("AP", 44.9, "https://huggingface.co/ultralyticsplus/yolov8s", None, None, "YOLOv8s，非n；predict+COCOeval与官方val不同。"),
    24: ("AP50", 95.194, "https://huggingface.co/keremberke/yolov8m-table-extraction", None, None, "模型卡仅AP50；validation70张，两类；不能用本次AP88.55直接比较95.194。"),
    25: ("oracle_target_center_coverage", None, "https://huggingface.co/racineai/UI-DETR-1", None, None, "本次89.38%是候选框覆盖率；官方70.8%是含Gemini2.5Pro选择步骤的点击准确率，任务阶段及指标不同。"),
    29: ("AP", 54.4, GITHUB+"THU-MIG/yolov10", 54.4, PAPER+"2405.14458", "X，COCO val2017；需官方val核对逐图predict/缩放/计分差异。"),
    33: ("AP", None, "https://huggingface.co/morsetechlab/yolov11-license-plate-detection", None, None, "实际v1n；模型卡72.60/AP50=98.13标题是YOLOv11x，不能套用。作者披露train/test污染；同页README也未提供n行。"),
    36: ("AP", 53.1, GITHUB+"lyuwenyu/RT-DETR", 53.1, PAPER+"2304.08069", "R50vd，640，COCO训练；不是Objects365预训练行。"),
    40: ("AP", None, "https://huggingface.co/unity/inference-engine-yolo", None, None, "该ONNX发布页未给独立AP；上游YOLOv8n37.3只作间接背景，不作为转换权重的已证实官方基准。"),
    43: ("AP", 53.4, GITHUB+"lyuwenyu/RT-DETR", 53.4, PAPER+"2407.17140", "RT-DETRv2-L对应R50vd；不是v2-M。"),
    44: ("AP", 42.0, GITHUB+"hustvl/YOLOS", 42.0, PAPER+"2106.00666", "YOLOS-B，800短边。"),
    45: ("AP50", 97.83431294995892, "https://huggingface.co/keremberke/yolov5n-license-plate", None, None, "模型卡仅AP50；不是m模型98.8298；原训练validator尚未复现。"),
    49: ("AP", 38.5, GITHUB+"THU-MIG/yolov10", 38.5, PAPER+"2405.14458", "官方HF nano权重；论文表17，非带增强训练标记的39.5。"),
    50: ("AP", 44.5, GITHUB+"fundamentalvision/Deformable-DETR", 43.8, PAPER+"2010.04159", "本地4层特征、300query、无box-refine、非two-stage；对齐发布仓库44.5。论文表1为43.8，仓库说明实现平台差异；不是46.2/46.9变体。"),
    52: ("AP50", 98.376, "https://huggingface.co/keremberke/yolov8s-table-extraction", None, None, "模型卡仅AP50；validation70张；当前COCOeval与原Ultralytics训练validator不同。"),
    56: ("AP", 44.9, GITHUB+"facebookresearch/detr", 44.9, PAPER+"2005.12872", "DETR R101-DC5。"),
    57: ("AP", 62.9, GITHUB+"jozhang97/DETA", None, PAPER+"2212.06137", "协议不一致：官方deta_swin_ft.sh --bigger为短边1200/长边2000，本地HF配置800/1333；论文63.5是test-dev，不能与val5000比较。"),
    59: ("AP", 41.6, GITHUB+"iMoonLab/yolov13", 41.6, PAPER+"2506.17733", "nano；官方使用其仓库validator，本次predict+COCOeval且SDPA代替FlashAttention。尚未隔离各因素，不能声称差值全由计分脚本造成。"),
    60: ("AP50", 36.53557610428755, "https://huggingface.co/keremberke/yolov5n-construction-safety", None, None, "AP50需与34.391比较，而非本次AP20.130；validation57张；原validator需复核。"),
    61: ("AP50", 27.342, "https://huggingface.co/keremberke/yolov8m-protective-equipment-detection", None, None, "作者本身公布的AP50较低；本次应比26.683与27.342，不是用AP12.55判定性能腰斩。"),
    62: ("AP", 56.2, GITHUB+"lyuwenyu/RT-DETR", 56.2, PAPER+"2304.08069", "R101vd，Objects365预训练再COCO微调；不能比较普通COCO训练54.3。"),
}
# Additional paper columns, in AP,AP50,AP75,APs,APm,APl order (all percentage points).
PAPER_COLUMNS = {
    1: [42.0,62.4,44.2,20.5,45.8,61.1], 11:[43.5,63.8,46.4,21.9,48.0,61.8],
    56:[44.9,64.7,47.7,23.7,49.5,62.3], 20:[38.5,53.8,41.7,18.9,42.4,54.6],
    49:[38.5,53.8,41.7,18.9,42.4,54.6], 29:[54.4,71.3,59.3,37.0,59.8,70.9],
    50:[43.8,62.6,47.7,26.4,47.1,58.0], 62:[56.2,74.6,61.3,38.3,60.5,73.5],
}

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-root", type=Path, default=Path("/data2/gaoyonghan/gpu_model_standalone_20261110/release/model/检测/foduucom--stockmarket-pattern-detection-yolov8/resources/runtime/assets/workspace/results/object_detection_official_v2"))
    args = parser.parse_args()
    root = args.result_root
    manifest = json.loads((root / "_source_snapshot/models.json").read_text())
    rows = []
    for spec in manifest["models"]:
        seq = spec["sequence"]
        if seq not in REFS:
            continue
        directory = root / spec["model_key"]
        status = json.loads((directory / "status.json").read_text())
        document = json.loads((directory / "metrics.json").read_text())
        assert status["state"] == "complete" and status["completed_images"] == status["total_images"], directory
        metrics = document.get("metrics_percent") or {k:v*100 for k,v in document["metrics"].items()}
        assert all(math.isfinite(v) for v in metrics.values()), directory
        key, reference, source, paper, paper_url, note = REFS[seq]
        current = metrics[key]
        delta = None if reference is None else current-reference
        label = "无直接可比基准" if delta is None else ("主指标数值接近" if abs(delta)<=0.1 else "主指标存在差异")
        comparisons = {}
        for metric, value in zip(["AP","AP50","AP75","AP_small","AP_medium","AP_large"], PAPER_COLUMNS.get(seq, [])):
            comparisons[metric] = {"local": metrics[metric], "paper": value, "delta_pp": metrics[metric]-value}
        rows.append({"sequence":seq,"model_key":spec["model_key"],"model_id":spec["model_id"],
                     "weight_file":spec["model"]["weight_file"],"dataset":spec["dataset"],
                     "images":status["total_images"],"local_metrics_percent":metrics,"comparison_metric":key,
                     "local_value":current,"official_release_reference":reference,"release_source":source,
                     "paper_reference_same_split":paper,"paper_source":paper_url,"delta_pp":delta,
                     "numeric_classification":label,"protocol_note":note,"strict_official_reproduction_verified":False,
                     "additional_paper_comparisons":comparisons,"result_file":str(directory/"metrics.json"),
                     "metrics_sha256":hashlib.sha256((directory/"metrics.json").read_bytes()).hexdigest(),
                     "run_fingerprint":status.get("run_fingerprint"),"dataset_snapshot":status.get("dataset_snapshot")})
    assert len(rows)==31
    counts = dict(Counter(row["numeric_classification"] for row in rows))
    output = root / "official_reference_audit_20260906"
    payload = {"checked_on":"2026-09-06","scope":"31 previously completed green detection entries, not the separate remaining-25 suite",
               "counts":counts,"close_threshold_pp":0.1,"threshold_is_author_criterion":False,
               "policy":"Completed and numerically close do not establish strict official reproduction. AP50 is not AP50-95. Missing/other-split references remain null.","models":rows}
    output.with_suffix(".json").write_text(json.dumps(payload,ensure_ascii=False,indent=2)+"\n")
    lines = ["# 31 个已完成检测模型：官方数值核查（2026-09-06）", "",
             "结论：不能把31个全量完成统一标为已复现官方。"+"；".join(f"{k}：{v}个" for k,v in counts.items())+"。", "",
             "‘主指标数值接近’仅表示差≤0.1个百分点，是本报告描述用分组，不是作者认可的误差范围，也不意味着所有分项一致。原始指标及运行记录均未修改。", "",
             "发布参考与论文分列：有些公开checkpoint结果与论文不同；无本模型独立论文或论文无同split对应行时留空。未开展全部checkpoint逐张与官方原生输出对照，因此本报告不授予‘严格复现’认证。", "",
             "| 序号 / 模型（实际权重） | 数据集 / split / 图数 | 本次同指标 | 发布参考 | 论文同split | 差值pp | 结论与核查说明 |",
             "|---|---|---:|---:|---:|---:|---|"]
    for r in rows:
        release = "—" if r["official_release_reference"] is None else f'[{r["official_release_reference"]:.4f}]({r["release_source"]})'
        paper = "—" if r["paper_reference_same_split"] is None else f'[{r["paper_reference_same_split"]:.2f}]({r["paper_source"]})'
        delta = "—" if r["delta_pp"] is None else f'{r["delta_pp"]:+.4f}'
        lines.append(f'| {r["sequence"]:02d} [{r["model_id"]}](https://huggingface.co/{r["model_id"]})<br>{r["weight_file"]} | {r["dataset"]["name"]}<br>{r["dataset"]["split"]} / {r["images"]} | {r["comparison_metric"]} {r["local_value"]:.4f} | {release} | {paper} | {delta} | {r["numeric_classification"]}；{r["protocol_note"]} |')
    lines += ["", "## 必须保留的限定", "",
              "- 旧runner对多种YOLO调用predict后用pycocotools计算指标；作者通常用val/训练validator。两者的输入letterbox、矩形batch、multi-label后处理及AP积分实现需要逐项核对。尚未做控制实验，不能断言差异只由一个环节造成。",
              "- 数据集名称和split依据本地下载清单、模型卡与结果快照核对；部分Roboflow/HF转存数据未与作者训练时逐文件hash校验，不能把名称一致夸大成已验证逐字节一致。每条本地annotation hash保留在JSON中。",
              "- 原始COCO小目标分项为-1（百分数文件-100）表示该分组无有效样本，不是负的模型性能。",
              "- AP50/AP75/尺寸分项的已核对论文列及逐列差值见同名JSON的additional_paper_comparisons；不是只看主AP就认可全部指标。",
              "- DETA发布配置证据：[deta_swin_ft.sh --bigger](https://github.com/jozhang97/DETA/blob/master/configs/deta_swin_ft.sh)；[数据变换将800乘1.5、最大边设2000](https://github.com/jozhang97/DETA/blob/master/datasets/coco.py)。本地800/1333配置不一致。",
              "- UI-DETR缺少Gemini选择阶段；morsetechlab本次n权重没有可直接使用的x基准，且作者声明测试集污染。这两项不得写成已复现论文。", "",
              "## 下一步", "",
              "本轮仅修复并补跑另外6个失败条目，不重写这31份结果。后续优先用原版validator复核签名模型、表格提取m、施工安全和YOLOv13，再处理其他YOLO差异及DETA更大输入协议。保留当前适配器结果，另开官方协议结果目录，禁止为了贴近数字调参。", ""]
    output.with_suffix(".md").write_text("\n".join(lines))
    print(json.dumps({"counts":counts,"report":str(output.with_suffix('.md'))},ensure_ascii=False))

if __name__ == "__main__":
    main()
