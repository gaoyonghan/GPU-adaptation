"""CPU-only annotation inventory; does not convert labels or run inference."""
import argparse
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image


DATA = Path('__DATASET_ROOT__/omnidocbench_eval')
MODEL = Path('__MODEL_ROOT__/weights/PaddlePaddle/PP-DocLayoutV2/config.json')
# A: explicit annotation counterpart; B: coarse/ambiguous; C: no independent positive label.
# These are audit judgments, NOT an executable ground-truth conversion map.
ROWS = [
    ('abstract', '摘要', 'C', (), 'text_block未区分摘要，不按正文内容猜标签。'),
    ('algorithm', '算法', 'C', (), 'algorithm_mask是忽略区域；code_txt不能直接当算法。'),
    ('aside_text', '侧栏文本', 'C', (), '没有侧栏独立标签，不能按页面位置自动推断。'),
    ('chart', '图表', 'C', (), 'chart_mask不是正样本，figure也没有图表子类。'),
    ('content', '目录', 'C', (), '没有目录独立标签；不能把所有text_block当目录。'),
    ('display_formula', '独立公式', 'A', ('equation_isolated',), '明确对应；旧PRED_MAP误用formula，漏掉本模型display_formula。'),
    ('doc_title', '文档标题', 'B', ('title',), 'title不区分文档标题和段落标题；与paragraph_title共享候选，不能重复计分。'),
    ('figure_title', '图表标题', 'B', ('figure_caption', 'table_caption'), '候选标签有标题/描述混用；需核实与vision_footnote的边界。'),
    ('footer', '页脚', 'A', ('footer',), '有对应标签；仍需抽查是否包含页脚图像，处理越界框。'),
    ('footer_image', '页脚图像', 'C', (), '没有页脚图像独立标签。'),
    ('footnote', '脚注', 'A', ('page_footnote',), '页脚注释正文；不包含span层footnote_mark上标。'),
    ('formula_number', '公式编号', 'A', ('equation_caption',), '官方定义为公式编号；旧流程未使用。'),
    ('header', '页眉', 'A', ('header',), '有对应标签；仍需抽查是否包含页眉图像，处理越界框。'),
    ('header_image', '页眉图像', 'C', (), '没有页眉图像独立标签。'),
    ('image', '图像', 'B', ('figure',), 'Omni的figure比本模型image可能更宽，未区分chart等细类。'),
    ('inline_formula', '行内公式', 'A', ('equation_inline',), '仅span层；有标注不代表全1651页标注完整，需先确定可评测范围。'),
    ('number', '页码', 'A', ('page_number',), '明确对应页码，不是任意数字。'),
    ('paragraph_title', '段落标题', 'B', ('title',), '与doc_title共享title，原始attribute未提供标题层级。'),
    ('reference', '参考文献', 'B', ('reference',), '同名不证明同义；本模型另有reference_content，不能照名字直接映射。'),
    ('reference_content', '参考文献内容', 'B', ('reference',), '原标注包含具体文献条目，但两种原生reference类别的边界未核实。'),
    ('seal', '印章', 'C', (), '没有独立印章标注；水印属性不等于印章框。'),
    ('table', '表格', 'A', ('table',), '明确对应表格主体；table_mask不计正样本。'),
    ('text', '正文', 'B', ('text_block',), '原标注没有摘要/侧栏/目录/竖排等完整细分，不能认为都是原生text。'),
    ('vertical_text', '竖排文字', 'C', (), '没有独立标签；旋转属性和竖排文字不是同一概念。'),
    ('vision_footnote', '图注', 'B', ('figure_caption', 'figure_footnote', 'table_footnote'), '标题、说明和注释需按原生定义核对；不能简单沿用旧footnote合并。'),
]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def annotations(region, level='block', inherited_ignore=False):
    category = region['category_type']
    ignored = (inherited_ignore or bool(region.get('ignore', False))
               or category.endswith('_mask') or category in {'abandon', 'equation_ignore'})
    yield level, region, ignored
    for child in region.get('merge_list', []):
        yield from annotations(child, 'fragment', ignored)
    for child in region.get('line_with_spans', []):
        yield from annotations(child, 'span', ignored)


def geometry(poly, width, height):
    if (not isinstance(poly, list) or len(poly) < 6 or len(poly) % 2
            or not all(isinstance(v, (int, float)) and math.isfinite(v) for v in poly)):
        return False, False
    xs, ys = poly[::2], poly[1::2]
    valid = max(xs) > min(xs) and max(ys) > min(ys)
    outside = min(xs) < 0 or min(ys) < 0 or max(xs) > width or max(ys) > height
    return valid, outside


def audit(data):
    pages = json.loads((data / 'OmniDocBench.json').read_text())
    stats, page_sets = defaultdict(Counter), defaultdict(set)
    issues, fragments, sources, rotations = [], Counter(), Counter(), Counter()
    order, span_pages, basenames = Counter(), set(), set()
    formats = Counter()
    nonempty_attributes = 0
    for index, page in enumerate(pages):
        info = page['page_info']
        name = Path(info['image_path']).name
        if name in basenames:
            issues.append(f'重复图像文件名：{name}')
        basenames.add(name)
        sources[info.get('page_attribute', {}).get('data_source', '未提供')] += 1
        image = data / 'images' / name
        try:
            with Image.open(image) as im:
                formats[im.format] += 1
                if im.size != (info['width'], info['height']):
                    issues.append(f'图像尺寸与标注不同：{name}')
        except OSError as exc:
            issues.append(f'图像读取失败：{name}：{exc}')
        orders = []
        for root in page['layout_dets']:
            for level, region, ignored in annotations(root):
                attrs = region.get('attribute', {})
                if level == 'block':
                    nonempty_attributes += bool(attrs)
                if 'text_rotate' in attrs:
                    rotations[str(attrs['text_rotate'])] += 1
                if level == 'fragment':
                    fragments[region['category_type']] += 1
                    continue
                key = (level, region['category_type'])
                count = stats[key]
                count['raw'] += 1
                count['ignored'] += ignored
                valid, outside = geometry(region.get('poly'), info['width'], info['height'])
                count['bad_geometry'] += not valid
                count['outside'] += outside
                if valid and not ignored:
                    count['candidate'] += 1
                    page_sets[key].add(index)
                if level == 'span':
                    span_pages.add(index)
                if level == 'block' and not ignored and valid:
                    value = region.get('order')
                    if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value):
                        order['with_order'] += 1
                        orders.append(value)
                    else:
                        order['without_order'] += 1
        order['pages_with_order'] += bool(orders)
        order['pages_with_comparable_pairs'] += len(set(orders)) >= 2
        order['pages_with_tied_order'] += len(orders) != len(set(orders))
    return pages, stats, page_sets, issues, fragments, sources, rotations, order, span_pages, nonempty_attributes, formats


def report(data, model):
    labels = json.loads(model.read_text())['label_list']
    if labels != [row[0] for row in ROWS]:
        raise ValueError('模型标签或顺序已变更，请重新核对人工分类表。')
    pages, stats, page_sets, issues, fragments, sources, rotations, order, span_pages, attrs, formats = audit(data)
    lines = [
        '# 38号 PP-DocLayoutV2：OmniDocBench覆盖审计', '',
        '审计口径：原始标注库存检查，不是推理结果，不代表已完成新评测。只针对本地快照，未下载或升级数据。', '',
        f'- 标注：`{data / "OmniDocBench.json"}`',
        f'- 标注SHA256：`{sha(data / "OmniDocBench.json")}`',
        f'- 模型配置：`{model}`；SHA256：`{sha(model)}`',
        f'- 审计脚本SHA256：`{sha(Path(__file__))}`',
        f'- 页数：{len(pages)}；图像存在/文件头尺寸异常：{len(issues)}。真实编码：`{dict(formats)}`。'
        '部分.png文件实际为JPEG，按文件内容识别，不因此判损坏；未做完整图像解码或像素哈希。', '',
        '## 结论', '',
        '按25个原生类别保守划分：A类8项有明确对应标注（其中行内公式仅部分页面有span标注）；'
        'B类8项只有上位标签或语义边界待核实；C类9项没有独立正样本标签。',
        '这是类别对应关系盘点，不是32%的数据完整率，也不能写成已完整覆盖16/25类。'
        'A类仍需按标注范围、忽略规则和越界坐标确定最终评分协议。', '',
        '## 25类逐项对应', '',
        'A=明确对应标注；B=上位/歧义候选，不能独立评分；C=无独立正样本标注。'
        '候选框不含ignore与无效几何；越界框只统计，未裁剪。B类行之间有重复候选，不可求和。', '',
        '| 原生类别 | 含义 | 状态 | Omni候选标签 | 候选框数 | 涉及页数 | 说明 |',
        '|---|---|---|---|---:|---:|---|',
    ]
    for label, chinese, state, cats, note in ROWS:
        keys = [('span' if cat == 'equation_inline' else 'block', cat) for cat in cats]
        count = sum(stats[key]['candidate'] for key in keys)
        involved = set().union(*(page_sets[key] for key in keys))
        lines.append(f'| {label} | {chinese} | {state} | {", ".join(cats) or "—"} | {count} | {len(involved)} | {note} |')
    lines += ['', '## 原始层级库存', '',
              '块级仅计layout_dets；merge_list仅作为组成片段，不再加入块级正样本。'
              '递归提取其中line_with_spans并继承祖先ignore。不能将所有层级合计为独立框数。', '',
              '| 层级 | 标签 | 原始条目 | 忽略条目 | 无效几何 | 越界条目 | 候选框 | 候选页 |',
              '|---|---|---:|---:|---:|---:|---:|---:|']
    for key, count in sorted(stats.items()):
        values = [count[k] for k in ('raw', 'ignored', 'bad_geometry', 'outside', 'candidate')]
        lines.append('| ' + ' | '.join(map(str, [*key, *values, len(page_sets[key])])) + ' |')
    lines += ['', f'未重复计入的merge_list组成片段：{sum(fragments.values())}条；分布：`{dict(fragments)}`。',
              f'顶层块非空attribute：{attrs}；片段旋转属性：`{dict(rotations)}`。没有可直接拆分标题层级的结构化属性。', '',
              '## 标注范围和阅读顺序', '',
              f'- 含任意span条目的页面：{len(span_pages)}/{len(pages)}。行内公式正样本页数见上表。'
              '其余页面不能自动认作行内公式负样本；有span的页面也不自动证明标注穷尽。',
              '- 不能只取行内公式正样本页报告全量AP；需要核实完整标注页面及负样本范围。',
              f'- 非忽略、几何有效块：有数值order {order["with_order"]}条，缺order {order["without_order"]}条；'
              f'有order的页面{order["pages_with_order"]}，至少两个不同order的页面{order["pages_with_comparable_pairs"]}，'
              f'有重复order的页面{order["pages_with_tied_order"]}。范围为全部非忽略块，不是旧11组子集。',
              '- 阅读顺序可作为单独子任务，但必须报告参与元素范围、匹配召回、并列/缺失order处理。'
              '旧项目的匹配后顺序正确率不是全页面端到端正确率，也不是原作者官方指标。',
              f'- 页面来源分布：`{dict(sorted(sources.items()))}`。各来源页保留原有标注，不重划训练/测试集。', '',
              '## 已确认旧流程问题与下一步边界', '',
              '1. run_pp_layout.py的PRED_MAP写formula，本模型输出display_formula，导致独立公式预测被丢弃。旧公式0分不能据此判断模型能力。',
              '2. 旧流程未纳入公式编号及span行内公式；table_footnote也未纳入。',
              '3. 旧title/text/reference等合并组不能冒充原生细分类别；图表标题与图注的原生边界还需核实。',
              '4. *_mask、abandon和equation_ignore只作忽略区域，不作为正样本。无标签不等于负样本。',
              '5. 下一轮先修正明确映射并保存全部25类原始预测；对B类抽样核实，对行内公式先确定完整标注范围。'
              '不能凭关键词或模型预测给GT补类别；不能为凑齐任务合并不等义标签。',
              '6. 本轮没有修改推理脚本、GT、历史metrics或模型验收状态；没有生成转换数据或执行GPU推理。'
              '暂不下载新数据，先依据本表决定补标或补集。', '',
              '## 依据与复查', '',
              '- [OmniDocBench官方标注定义与层级说明](https://github.com/opendatalab/OmniDocBench#evaluation-set-introduction)：用于标签及ignore解释；本地快照SHA是本次统计依据，不声称等于最新公开版本。',
              '- [PP-DocLayoutV2模型卡](https://huggingface.co/PaddlePaddle/PP-DocLayoutV2)：模型身份；25类顺序以本地官方config.json为准。',
              '- [PaddleX类别释义](https://paddlepaddle.github.io/PaddleX/latest/module_usage/tutorials/ocr_modules/layout_analysis.html)：当前页面为V3，公共同名类别释义仅作辅助，不据此宣称V2评估协议已对齐。', '',
              '复查命令（Python标准库及已有Pillow，CPU统计；默认更新本审计报告，不改任何数据或推理结果）：', '',
              '```bash', 'cd __MODEL_ROOT__/resources/runtime/assets/workspace/remaining_detection_eval',
              'python3 audit_m38_omni.py', '```', '']
    if issues:
        lines += ['## 文件异常', '', *[f'- {issue}' for issue in issues], '']
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=DATA)
    parser.add_argument('--model-config', type=Path, default=MODEL)
    parser.add_argument('--output', type=Path, default=Path(__file__).with_name('m38_omni_coverage.md'))
    args = parser.parse_args()
    result = report(args.data, args.model_config)
    args.output.write_text(result, encoding='utf-8')
    print(f'CPU覆盖审计完成：{args.output}')


if __name__ == '__main__':
    main()
