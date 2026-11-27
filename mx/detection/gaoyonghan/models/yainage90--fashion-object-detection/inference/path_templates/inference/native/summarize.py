"""Reconcile unique model IDs, verified metrics and outstanding task coverage."""
import json
from collections import Counter, defaultdict
from pathlib import Path
from datetime import datetime, timezone

HERE=Path(__file__).resolve().parent
RESULTS=Path('__MODEL_ROOT__/resources/runtime/assets/workspace/results')
ROOTS=['remaining25_v2','remaining25_v3','remaining25_v4','remaining25_cmarkea_v1','remaining25_pp_v1',
       'remaining25_wider_v1','remaining25_wider_v2','remaining25_nemotron_v1','remaining25_nemotron_v2','remaining25_v5','remaining25_pp_v2','remaining25_pp_v3','remaining25_qualcomm_v1','remaining25_qualcomm_v2','remaining25_anzhc_v2','remaining25_anzhc_v3','remaining25_qualcomm_v3']


def main():
    manifest=json.loads((HERE/'models.json').read_text())
    baseline=json.loads((HERE/'blocker_audit.json').read_text())
    by_model=defaultdict(list)
    for name in ROOTS:
        for path in (RESULTS/name).rglob('status.json'):
            record=json.loads(path.read_text())
            seq=record.get('sequence') or record.get('protocol',{}).get('sequence')
            if seq is None:continue
            metric_path=path.with_name('metrics.json')
            metric=json.loads(metric_path.read_text()) if metric_path.exists() else None
            valid=record['state']=='complete' and metric is not None and not record.get('invalid_results')
            if seq==42 and name=='remaining25_v3':valid=False
            if seq in {32,35} and name in {'remaining25_qualcomm_v1','remaining25_qualcomm_v2'}:valid=False
            by_model[seq].append({'status':record['state'],'task':record.get('task',name),
                'valid_complete':valid,'images':record.get('completed_images'),'result_directory':str(path.parent),
                'metrics':metric,'coverage':record.get('coverage'),'error':record.get('error')})
    models=[]
    for m in manifest['models']:
        seq=m['sequence'];rows=by_model[seq];done=[r for r in rows if r['valid_complete']]
        expected={t['key'] for t in m['tasks']}
        complete=bool(expected) and expected<={r['task'] for r in done}
        if seq in {38,39,42,26,37,48}:
            complete=bool(done) if not expected else complete
        if seq==42:
            complete=any('semantic_F1' in (r['metrics'] or {}) and 'bbox' in (r['metrics'] or {}) for r in done)
        if seq in {38,39}:
            complete=any('reading_order' in (r['metrics'] or {}) and (seq==38 or 'polygon_segm_AP' in (r['metrics'] or {})) for r in done)
        if seq in {8,31,32,35}:complete=False
        state='complete' if complete else ('running' if any(r['status']=='running' for r in rows) else ('partial_complete' if done else ('blocked' if seq in {8,31,32,35} else 'pending')))
        models.append({'sequence':seq,'model_id':m['model_id'],'state':state,'all_tasks_complete':complete,
                       'required_tasks':m['required_tasks'],'results':rows,'next_action':m['next_action']})
    counts=Counter(m['state'] for m in models)
    output={'generated_at':datetime.now(timezone.utc).isoformat(),'total_detection_models':62,'existing_green_queue':37,'current_scope_unique_models':25,
            'initial_primary_blocker_counts':baseline['primary_blocker_counts'],'current_state_counts':dict(counts),
            'counting_rule':'Unique model IDs; all required tasks must have complete frozen benchmark metrics. PubTables means the model-blind 1000-image benchmark, not the original 93834-image test. Invalid cmarkea and all superseded Qualcomm v1/v2 results excluded.','fidelity_counts':{'reference_baseline':25},'comparison_counts':{'not_officially_comparable':25},'case_type_counts':{'special':25},
            'models':models}
    (HERE/'progress.json').write_text(json.dumps(output,ensure_ascii=False,indent=2)+'\n')
    (HERE/'audit_summary.json').write_text(json.dumps(output,ensure_ascii=False,indent=2)+'\n')
    lines=['# 剩余25模型处理进展','',f'本轮25个唯一模型：已完成全部已配置任务 {counts["complete"]} 个，运行中 {counts["running"]} 个，已完成部分任务 {counts["partial_complete"]} 个，阻塞 {counts["blocked"]} 个，待处理 {counts["pending"]} 个。既有37个模型队列另计。',
           '', '指标均为当前冻结数据和适配协议下的结果；公开替代、类别子集和分类合并不与作者官方数字排名比较。',
           '', '| 序号 | 模型 | 状态 | 已完成数据与指标 | 尚需处理 |','|---:|---|---|---|---|']
    for m in models:
        summaries=[]
        for r in m['results']:
            if not r['valid_complete']:continue
            metric=r['metrics'];score=(metric.get('metrics') or metric.get('bbox') or metric)
            ap=score.get('AP')
            text=f"{r['task']}：{r['images']}图"
            if ap is not None:text+=f"，AP {ap*100:.2f}%"
            if 'AP50' in score and score['AP50'] is not None:text+=f"，AP50 {score['AP50']*100:.2f}%"
            if r['coverage']:
                coverage=r['coverage']; valid_groups=sum(v is not None for v in score.get('per_class_AP',{}).values())
                text+=f"，{valid_groups}个有正样本金标评分组（原生{coverage['total_model_classes']}类）"
            if 'easy_AP' in metric:text+=f"，Easy/Medium/Hard AP {metric['easy_AP']*100:.2f}/{metric['medium_AP']*100:.2f}/{metric['hard_AP']*100:.2f}%"
            if 'reading_order' in metric:text+=f"，排序配对准确率 {metric['reading_order']['pair_accuracy']*100:.2f}%，排序元素召回 {metric['reading_order']['matched_element_recall']*100:.2f}%"
            if 'polygon_segm_AP' in metric:text+=f"，多边形AP {metric['polygon_segm_AP']*100:.2f}%"
            if 'landmarks_2d' in metric:
                lm=metric['landmarks_2d']
                if lm.get('PCK_0_1_conditional') is not None:text+=f"，已匹配点PCK@0.1 {lm['PCK_0_1_conditional']*100:.2f}%，端到端 {lm['PCK_0_1_end_to_end']*100:.2f}%"
            if 'segm_AP' in metric:text+=f"，mask AP {metric['segm_AP']*100:.2f}%"
            if 'semantic_macro_foreground_F1' in metric:text+=f"，分割macro F1 {metric['semantic_macro_foreground_F1']*100:.2f}%"
            summaries.append(text)
        lines.append(f"| {m['sequence']} | {m['model_id']} | {m['state']} | {'；'.join(summaries) or '尚无有效全量指标'} | {'已记录协议差异' if m['all_tasks_complete'] else m['next_action']} |")
    lines+=['','## 已发现并保留的异常','',
            '- cmarkea 原先按检测模型自动加载会随机初始化大量参数。该次结果无效；正确模型是 `DetrForSegmentation`，必须同时评估检测与分割。',
            '- 股票涨跌在build_chart上的AP很低；抽查显示金标覆盖较大趋势区间，模型较高分预测集中在少数K线附近。类别相同不保证框粒度一致，此替代结果需保留这一限制。',
            '- PlantDoc压缩包同时包含训练和测试集的同名图像，读取必须限定TEST目录；不能使用跨整个ZIP的basename索引。',
            '- 原始清单仅用于恢复历史原因；Qualcomm实际ONNX已补齐，PMC包已重新下载，当前进展不能沿用旧“未下载”状态。',
            '- PP V3空掩码与COCO RLE格式错误已修复；Qualcomm人脸旧ROI旋转错误结果无效；Anzhc文本原图掩码OOM已改用原生默认掩码后处理。',
            '- Qualcomm少量旋转关键点取整后重合，原后处理产生NaN。v3显式保留检测并记录姿态失败，两项5000图全部重跑，旧v1/v2排除。',
            '- 表结构完整评估范围是预先冻结的1000图子集；图表的10个原生标签合并为8个评分组。只有忽略区域、没有正例的类别不算覆盖。',
            '- 首批7个模型的代码快照保存在 `remaining25_v2/_source_snapshot`。新协议不覆盖旧预测。',
            '', '详细指标、数据覆盖、异常、结果路径见 `progress.json`；最初阻塞分类见 `阻塞统计.md`。']
    (HERE/'处理进展.md').write_text('\n'.join(lines)+'\n')
    print(dict(counts))


if __name__=='__main__':main()
