"""Collect only completed, independently checked supplemental evaluations."""
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
RESULTS = Path('__MODEL_ROOT__/resources/runtime/assets/workspace/results/remaining25_supplemental_20260905')
RUNS = [
    (19, 'm19-doclaynet-competition498', '原比赛集，整体COCO评分', 498, 11, 'model19/independent_verification.json'),
    (30, 'm30-author-test5544', '作者测试集，batch1', 5544, 7, 'model30/independent_verification.json'),
    (30, 'm30-author-test5544-author-batch8', '作者测试集，作者batch8流程', 5544, 7, 'model30/independent_verification_batch8.json'),
    (48, 'm48-cached-val560-native10', '原val，官方示例默认', 560, 10, 'model48/independent_verification.json'),
    (48, 'm48-cached-val560-native10-raw-wrapper', '同一val，仅取消额外0.1过滤', 560, 10, 'model48/independent_verification_raw.json'),
]


def main():
    records = []
    for seq, name, label, images, classes, check in RUNS:
        root = RESULTS / name
        status = json.loads((root / 'status.json').read_text())
        scores = json.loads((root / 'metrics.json').read_text())
        verification = json.loads((HERE / check).read_text())
        assert status['state'] == 'complete' and status['completed_images'] == images
        assert verification['verified'] and verification['images'] == images
        assert len(verification['ground_truth_per_class']) == classes
        assert scores['protocol_fingerprint'] == status['protocol_fingerprint']
        if seq == 19:
            ap, ap50, ar = (scores['pooled'][k] for k in ['AP', 'AP50', 'AR100'])
        elif seq == 30:
            ap, ap50, ar = (scores[k] for k in ['map', 'map_50', 'mar_100'])
        else:
            ap, ap50, ar = (scores['metrics'][k] for k in ['AP', 'AP50', 'AR100'])
        records.append({'sequence': seq, 'label': label, 'state': 'complete',
                        'result_directory': str(root), 'images': images, 'classes': classes,
                        'gt_instances': verification['annotations'], 'predictions': verification['predictions'],
                        'AP': ap, 'AP50': ap50, 'AR100': ar, 'unit': 'fraction_0_to_1',
                        'protocol_fingerprint': status['protocol_fingerprint'],
                        'weight_sha256': status['protocol']['weight_sha256'],
                        'independent_verification': check,
                        'metrics_sha256': hashlib.sha256((root / 'metrics.json').read_bytes()).hexdigest()})
    summary = {'completed_at': datetime.now(timezone.utc).isoformat(), 'unique_models': 3,
               'dataset_image_count_sum': 6602, 'completed_full_runs': len(records),
               'inference_image_records': sum(r['images'] for r in records),
               'original_results_overwritten': False, 'manual_accepted_sequences_unchanged': [12, 26, 58],
               'runs': records, 'model19_official_four_domain_score': None,
               'model19_four_domain_note': 'Official ZIPs omit usable domain membership. Pooled metrics are reported, not official four-domain ranking scores.'}
    (HERE / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n')
    lines = ['# 补充数据集与推理结果', '',
             '已补齐19、30、48号的3套数据，共6602张／页；完成5组全量推理（12706图次），全部通过独立结果核验。旧结果保留，新结果均有独立目录和协议指纹。', '',
             '| 模型 | 固定协议 | 图像数 | 类别数 | AP% | AP50% | AR100% |',
             '|---:|---|---:|---:|---:|---:|---:|']
    for r in records:
        lines.append(f"| {r['sequence']} | {r['label']} | {r['images']} | {r['classes']} | {100*r['AP']:.4f} | {100*r['AP50']:.4f} | {100*r['AR100']:.4f} |")
    lines += ['', '19号使用比赛原498页、9158框、11类。整体AP和AR舍入后均与[Aryn论文§4](https://arxiv.org/html/2409.00847v2#S4)的0.602／0.743一致；论文未明确汇总实现。比赛官网另要求四文档域平均，实际公开包缺少可用域映射，因此官方四域排名分保留为空。模型卡57.1的具体来源仍未确认。', '',
              '30号使用[作者公开测试集](https://huggingface.co/datasets/yainage90/fashion-object-detection)全部5544张、23629框：Fashionpedia1155＋ModaNet4389。类别直接沿用作者七类；一次原image_id碰撞通过唯一行序ID处理，未丢图。batch1与作者batch8均完整保存；后续优先对照batch8明确流程，样本顺序、padding、移除pixel_mask及坐标归一化也是协议组成部分。模型卡AP为75.42%；作者batch8本轮80.6298%，高5.2098个百分点，差距仍未解释。batch1与8仅相差0.3191个百分点，已核对的处理器版本路径也没有本轮可触发的语义变化，不能据此给分差归因。不通过筛样本或扫描阈值贴合原数字。', '',
              '48号使用CACHED原val560张、11520框、10个原生类别；逐类图数和框数均匹配[NVIDIA模型卡](https://huggingface.co/nvidia/nemotron-graphic-elements-v1)。模型卡10类AP的算术均值为74.37%（本次计算，并非卡中单列总值）。7类默认AP与卡相差不到2个百分点，主要差距在other、legend_title、mark_label；取消额外0.1过滤只提升总AP0.186个百分点，不是主要差距来源。', '',
              '本次依靠固定来源的数据和真实权重重新推理。19、48沿用已有环境；30为对齐作者评分在自身目录隔离安装依赖，未修改共享环境。所有新结果都保留完整逐图预测、实际GT、指标、版本和哈希；30还保留batch/padding坐标约定，48保留过滤前wrapper输出。共享GPU上的耗时仅作运行记录。', '',
              '详细结果：[19号](model19/结果报告.md)、[30号](model30/补评结果.md)、[48号](model48/结果报告.md)。机器记录：[summary.json](summary.json)。数据根目录：__DATASET_ROOT__/supplemental_20260905/；结果根目录：__MODEL_ROOT__/resources/runtime/assets/workspace/results/remaining25_supplemental_20260905/。', '',
              '12、26、58此前由用户接受的基线记录保持不变。本次完成情况与“用户逐项接受”分开记录；42此前已有预测补评分，未找到作者500页名单，因此未另造替代清单重复跑；17未找到可直接绑定原权重的论文评测方案，本次未改其439图基线。']
    (HERE / '补跑结果汇总.md').write_text('\n'.join(lines) + '\n')
    print(f"Collected {len(records)} verified runs, {summary['inference_image_records']} inference-image records.")


if __name__ == '__main__':
    main()
