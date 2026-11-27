"""Score saved predictions under explicit, qualified label mappings."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def calculate(pairs):
    if not pairs:
        raise ValueError('No eligible samples')
    confusion = Counter(pairs)
    targets = sorted({t for t, _ in pairs})
    stats = {}
    for label in targets:
        tp = confusion[label, label]
        support = sum(v for (t,p),v in confusion.items() if t == label)
        predicted = sum(v for (t,p),v in confusion.items() if p == label)
        precision = tp / predicted if predicted else 0
        recall = tp / support
        f1 = 2 * tp / (support + predicted) if support + predicted else 0
        stats[str(label)] = dict(support=support, correct=tp, precision=precision, recall=recall, f1=f1)
    correct = sum(t == p for t,p in pairs)
    return dict(images=len(pairs), correct=correct, accuracy=correct/len(pairs),
                balanced_accuracy=sum(s['recall'] for s in stats.values())/len(stats),
                macro_f1_over_retained_mapped_classes=sum(s['f1'] for s in stats.values())/len(stats),
                per_class=stats, confusion=[dict(target=t,prediction=p,count=n) for (t,p),n in sorted(confusion.items())])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model',type=int,choices=[81,189,190],required=True)
    parser.add_argument('--predictions',type=Path,required=True)
    parser.add_argument('--index',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    predictions=rows(args.predictions)
    excluded=Counter();pairs=[]
    if args.model==81:
        mapping=[0,1,2,3,6,4,5]
        ids=[r['image_id'] for r in predictions]
        assert len(ids)==len(set(ids))==3589
        for row in predictions:
            raw=row['raw_prediction_model_id']
            assert 0<=raw<7 and len(row['logits'])==7
            assert raw==max(range(7),key=lambda k:row['logits'][k])
            assert mapping[raw]==row['provisional_prediction_fer_id']
            pairs.append((row['target_fer_id'],mapping[raw]))
        note='固定历史候选顺序anger/disgust/fear/happy/neutral/sad/surprise；配置仍为LABEL_0..6，缺少作者编号映射证明。禁止用测试集优化类别排列。'
        qualification='provisional_label_mapping'
    else:
        if args.index is None:raise ValueError('--index required')
        index_rows=rows(args.index);index={r['id']:r for r in index_rows}
        pred={r['id']:r for r in predictions}
        assert len(index)==len(index_rows) and len(pred)==len(predictions) and set(index)==set(pred)
        if args.model==189:
            mapping={'Queen Anne architecture':3,'Tudor Revival architecture':4}
            excluded_labels={'American craftsman style'}
            expected=1175
            note='沿用原表子集规则：排除American craftsman style；Queen Anne→3，Tudor Revival→4。后者为较宽风格到Tudor cottage的代理映射，非已证实分类体系等价。'
        else:
            mapping={'0-2':0,'20-29':2,'30-39':3,'40-49':4,'50-59':5,'60-69':6,'more than 70':7}
            excluded_labels={'3-9','10-19'}
            expected=8417
            note='沿用原表子集规则：排除3-9、10-19；其余源标签映射到相应十岁段，预测类别7/8/9合并为70+。年龄端点定义未由作者确认，因此是有假设的代理评分。'
        for key,row in index.items():
            label=row['source_label']
            if label in excluded_labels:excluded[label]+=1;continue
            if label not in mapping:raise ValueError(f'Unexpected source label: {label}')
            prediction=pred[key]['prediction']
            assert 0<=prediction<(5 if args.model==189 else 10)
            if args.model==190:prediction=min(prediction,7)
            pairs.append((mapping[label],prediction))
        assert len(pairs)==expected
        qualification='conditional_filtered_proxy'
    data=calculate(pairs)
    data.update(sequence=args.model,inference_images=len(predictions),excluded_images=sum(excluded.values()),
                excluded_by_source_label=dict(excluded),scoring_status=qualification,
                accepted_as_unqualified_formal_score=False,official_reference_comparable=False,
                mapping=mapping,mapping_note=note,new_inference=False,
                predictions_source=str(args.predictions),predictions_sha256=digest(args.predictions),
                scoring_script_sha256=digest(Path(__file__)))
    if args.index:data.update(index_source=str(args.index),index_sha256=digest(args.index))
    if args.model==189:
        data['exact_name_subset_queen_anne_only']=calculate([p for p in pairs if p[0]==3])
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:data[k] for k in ['sequence','images','correct','accuracy','excluded_images','scoring_status']},ensure_ascii=False))


if __name__=='__main__':main()
