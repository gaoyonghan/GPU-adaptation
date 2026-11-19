"""Recover the 25 excluded records without changing the running green batch."""
import copy
import hashlib
import json
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
SOURCE = HERE.parent / 'official_object_detection_eval/models.json'
GROUPS = {
    '原协议/划分不明，现有数据可做参考评测': [9, 12, 17, 19, 28, 46, 47, 58],
    '数据已有，缺任务适配或多权重覆盖': [22, 26, 30, 37, 41, 42, 54, 55],
    '旧替代集类别不全，需核对任务覆盖': [3, 34, 38, 39],
    '私有数据且缺任务匹配的公开金标': [8, 31],
    '仅下载了说明/发布链接，实际权重缺失': [32, 35],
    '数据下载未完成，当前目录为空': [48],
}
TASKS = {
    8: ['face_bbox_mask', 'manface_bbox_mask', 'womanface_bbox_mask', 'eye_bbox_mask',
        'headhair_bbox_mask', 'breast_bbox_mask', 'breast_size_detection', 'drone_detection',
        'anime_score_classification', 'text_bbox_mask', 'manga_panel_bbox_mask'],
    32: ['palm_detection', 'hand_landmarks_21'],
    35: ['face_detection', 'face_detector_keypoints_6', 'face_landmarks_468'],
    38: ['layout_detection_classification', 'reading_order'],
    39: ['layout_detection_classification', 'polygon_segmentation', 'reading_order'],
    41: ['paper_layout', 'report_layout', 'publaynet_layout', 'jiaocai_layout', 'general6_layout'],
}
NEXT = {
    3: '按新要求运行检测任务，单列5/6类覆盖；StockLine无金标，不能计0分。',
    8: '按11个任务分别选择检测框、真实掩码或分类金标；仅检测AP不能代替mask/分类指标。',
    9: '冻结build_chart test 249图，修正旧清单关于无框/时间切分的过时描述，运行替代检测AP。',
    12: '运行作者validation 70图；切分未披露，指标相近也不能证明原协议身份。',
    17: '复用cTDaR图像和XML，运行全部439图，报告明确阈值下F1和COCO AP参考基线。',
    19: '官方DocLayNet val/test分别运行；保留切分不明，不按更高分挑选官方结果。',
    22: '接入本地NVIDIA原生YOLOX；DocLayNet与InfoDet按来源/类别分别评分，未标类别不得作为负类。',
    26: '接入WIDER FACE三档真实难度标签及官方评估流程。',
    28: '使用现有350图近似替代，明确license-plates名称映射与历史380图不一致。',
    30: 'Fashionpedia七类显式合并，冻结映射并完成检测AP。',
    31: '保留历史人工跳过证据；查找独立人工金标，不用模型伪标签冒充真值。',
    32: '从release_assets固定URL补ONNX；分别核查palm框与21点/handedness任务和标注定义。',
    34: 'PlantDoc同名物种/病害检测标签显式映射；按放宽条件报告共有类指标和覆盖缺口。',
    35: '补官方发布权重；核对发布包是否还含468点landmark子模型，匹配全部输出任务金标。',
    37: 'PubTables结构XML加table words构造真实cell/row/column金标，使用NVIDIA后处理。',
    38: 'OmniDocBench检测框与order可覆盖检测/阅读顺序；需原生Paddle推理与匹配后排序指标。',
    39: 'OmniDocBench多边形和order覆盖mask/排序；使用支持pp_doclayout_v3的Transformers。',
    41: '分别评估5种领域权重；PubLayNet用于publaynet，其余用OmniDocBench等替代并记录类别范围。',
    42: '完整DocLayNet test为同源参考集，核对本模型标签空间后跑AP，不冒充500页官方F1。',
    46: '使用ISFire-4 test合并四类为fire/smoke；验证YOLOv10加载兼容性。',
    47: 'FLIR thermal validation六类映射后运行检测AP，记录热成像模态。',
    48: '重新获取PMC ChartInfo，只有核对560图/11520框/10类后才能认定官方同版。',
    54: '接入SKU110K官方test CSV；保留小框，明确COCO maxDets与密集货架原协议差异。',
    55: 'COCO-WholeBody同时保留person框和有效face框，分别报告AP。',
    58: '运行官方来源validation，切分未知只影响官方可比性，不阻止参考指标。',
}


def main():
    if (HERE / "models.json").exists():
        raise SystemExit("Initial audit already exists; refusing to overwrite enriched models.json")
    original = json.loads(SOURCE.read_text())
    records = []
    for m in original['models']:
        if m['green_dataset_confirmed']:
            continue
        s = m['sequence']
        root = Path(m['model']['local_path'])
        weights = [{'file': str(p.relative_to(root)), 'bytes': p.stat().st_size}
                   for p in sorted(root.rglob('*')) if p.is_file()
                   and p.suffix in {'.pt', '.pth', '.bin', '.onnx', '.safetensors', '.pdiparams', '.tflite'}]
        records.append({
            'sequence': s, 'excel_row': s + 1, 'model_id': m['model_id'],
            'primary_blocker': next(k for k, ids in GROUPS.items() if s in ids),
            'historical_reason': m['official_audit_note'],
            'required_tasks': TASKS.get(s, ['bbox_detection_classification']),
            'task_inventory_status': 'requires_release_and_checkpoint_audit' if s in {32, 35} else 'model_card_and_prior_audit',
            'weight_files_present': weights, 'next_action': NEXT[s],
            'source_model_card': m['official_sources']['model_card'],
            'local_model_card': str(root / 'README.md'),
            'technical_complete': False, 'metrics': None,
        })
    assert len(records) == len({r['model_id'] for r in records}) == 25
    counts = Counter(r['primary_blocker'] for r in records)
    report = {'audited_on': '2026-09-05', 'total_detection_models': 62,
              'existing_green_batch': 37, 'remaining_models': 25,
              'scope_note': '37是已选入队列数，不是全量完成数。25按唯一model_id统计；任务/权重子记录不重复计数。',
              'source_manifest': str(SOURCE), 'source_sha256': hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
              'primary_blocker_counts': dict(counts), 'models': records}
    (HERE / 'blocker_audit.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    manifest = copy.deepcopy(original)
    manifest['models'] = [m for m in manifest['models'] if not m['green_dataset_confirmed']]
    manifest['suite'] = 'remaining25-task-coverage-reference-evaluation'
    manifest['selection'] = {'sequences': [r['sequence'] for r in records], 'count': 25}
    manifest.pop('summary', None)
    manifest['coverage_policy'] = '全部模型卡任务必须有对应金标和指标；类别子集可评测但必须报告覆盖，缺失类记null；不比较官方排名。'
    for m in manifest['models']:
        s = m['sequence']
        m['execution_state'] = 'pending_adapter'
        m['next_action'] = NEXT[s]
        m['required_tasks'] = TASKS.get(s, ['bbox_detection_classification'])
        m['tasks'] = []
        if s in {3, 9, 12, 28, 46, 47, 58}:
            m['execution_state'] = 'ready'
            m['tasks'] = [{'key': 'bbox', 'dataset_id': m['dataset']['id'],
                           'runner': m['protocol']['runner'], 'weight_file': m['model']['weight_file'],
                           'model_label_aliases': {'license-plates': 'license plate'} if s == 28 else {}}]
        m['fidelity'] = 'reference_baseline'
        m['protocol']['official_reference'] = None
    (HERE / 'models.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n')
    lines = ['# 剩余25个检测模型阻塞台账', '', '核查：2026-09-05。62 = 37个既有队列 + 本轮25个；37不代表已全量完成。', '',
             '历史报告的19个冒烟通过、14个失败、29个阻塞属于9月4日旧批次，不能当成本轮25个的分类。', '',
             '本次按用户放宽要求接受公开替代与类别子集；检测、分割、关键点、分类和阅读顺序等全部实际任务仍须覆盖。缺失类别记缺失，不能记0分。指标相近不能证明数据划分相同。', '',
             '| 主阻塞（互斥） | 数量 | 模型序号 |', '|---|---:|---|']
    lines += [f'| {k} | {len(v)} | {", ".join(map(str, v))} |' for k, v in GROUPS.items()]
    lines += ['', '模型序号与Excel行号相差1；旧文档“第23行”对应模型序号22，避免错位。', '',
              '| 序号 | 模型 | 主阻塞 | 本轮处理 |', '|---:|---|---|---|']
    lines += [f"| {r['sequence']} | {r['model_id']} | {r['primary_blocker']} | {r['next_action']} |" for r in records]
    lines += ['', '逐模型旧结论、权重实存文件、模型卡来源保存在 `blocker_audit.json`。本文件是阻塞基线，运行结果以同目录后续进展记录和结果目录为准。']
    (HERE / '阻塞统计.md').write_text('\n'.join(lines) + '\n')


if __name__ == '__main__':
    main()
