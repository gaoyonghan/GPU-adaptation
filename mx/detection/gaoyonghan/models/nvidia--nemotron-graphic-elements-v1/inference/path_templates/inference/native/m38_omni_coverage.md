# 38号 PP-DocLayoutV2：OmniDocBench覆盖审计

审计口径：原始标注库存检查，不是推理结果，不代表已完成新评测。只针对本地快照，未下载或升级数据。

- 标注：`__DATASET_ROOT__/omnidocbench_eval/OmniDocBench.json`
- 标注SHA256：`a45cd84b04ad8b793e775089640e6b681209abea33ead54c1828ddca35fae496`
- 模型配置：`__MODEL_ROOT__/weights/PaddlePaddle/PP-DocLayoutV2/config.json`；SHA256：`e1a25a0cbdc59c3668bc96467ecca6b9747eca29ff26d90282f452adc89df21f`
- 审计脚本SHA256：`87cd8e34100df18500392bf850f1b8592ee48f6a69b3e8c36b17b0f38e914ed9`
- 页数：1651；图像存在/文件头尺寸异常：0。真实编码：`{'JPEG': 1015, 'PNG': 636}`。部分.png文件实际为JPEG，按文件内容识别，不因此判损坏；未做完整图像解码或像素哈希。

## 结论

按25个原生类别保守划分：A类8项有明确对应标注（其中行内公式仅部分页面有span标注）；B类8项只有上位标签或语义边界待核实；C类9项没有独立正样本标签。
这是类别对应关系盘点，不是32%的数据完整率，也不能写成已完整覆盖16/25类。A类仍需按标注范围、忽略规则和越界坐标确定最终评分协议。

## 25类逐项对应

A=明确对应标注；B=上位/歧义候选，不能独立评分；C=无独立正样本标注。候选框不含ignore与无效几何；越界框只统计，未裁剪。B类行之间有重复候选，不可求和。

| 原生类别 | 含义 | 状态 | Omni候选标签 | 候选框数 | 涉及页数 | 说明 |
|---|---|---|---|---:|---:|---|
| abstract | 摘要 | C | — | 0 | 0 | text_block未区分摘要，不按正文内容猜标签。 |
| algorithm | 算法 | C | — | 0 | 0 | algorithm_mask是忽略区域；code_txt不能直接当算法。 |
| aside_text | 侧栏文本 | C | — | 0 | 0 | 没有侧栏独立标签，不能按页面位置自动推断。 |
| chart | 图表 | C | — | 0 | 0 | chart_mask不是正样本，figure也没有图表子类。 |
| content | 目录 | C | — | 0 | 0 | 没有目录独立标签；不能把所有text_block当目录。 |
| display_formula | 独立公式 | A | equation_isolated | 2066 | 313 | 明确对应；旧PRED_MAP误用formula，漏掉本模型display_formula。 |
| doc_title | 文档标题 | B | title | 4441 | 1179 | title不区分文档标题和段落标题；与paragraph_title共享候选，不能重复计分。 |
| figure_title | 图表标题 | B | figure_caption, table_caption | 1208 | 614 | 候选标签有标题/描述混用；需核实与vision_footnote的边界。 |
| footer | 页脚 | A | footer | 741 | 529 | 有对应标签；仍需抽查是否包含页脚图像，处理越界框。 |
| footer_image | 页脚图像 | C | — | 0 | 0 | 没有页脚图像独立标签。 |
| footnote | 脚注 | A | page_footnote | 126 | 90 | 页脚注释正文；不包含span层footnote_mark上标。 |
| formula_number | 公式编号 | A | equation_caption | 281 | 55 | 官方定义为公式编号；旧流程未使用。 |
| header | 页眉 | A | header | 1920 | 1076 | 有对应标签；仍需抽查是否包含页眉图像，处理越界框。 |
| header_image | 页眉图像 | C | — | 0 | 0 | 没有页眉图像独立标签。 |
| image | 图像 | B | figure | 1385 | 613 | Omni的figure比本模型image可能更宽，未区分chart等细类。 |
| inline_formula | 行内公式 | A | equation_inline | 1462 | 96 | 仅span层；有标注不代表全1651页标注完整，需先确定可评测范围。 |
| number | 页码 | A | page_number | 1188 | 1158 | 明确对应页码，不是任意数字。 |
| paragraph_title | 段落标题 | B | title | 4441 | 1179 | 与doc_title共享title，原始attribute未提供标题层级。 |
| reference | 参考文献 | B | reference | 275 | 17 | 同名不证明同义；本模型另有reference_content，不能照名字直接映射。 |
| reference_content | 参考文献内容 | B | reference | 275 | 17 | 原标注包含具体文献条目，但两种原生reference类别的边界未核实。 |
| seal | 印章 | C | — | 0 | 0 | 没有独立印章标注；水印属性不等于印章框。 |
| table | 表格 | A | table | 665 | 458 | 明确对应表格主体；table_mask不计正样本。 |
| text | 正文 | B | text_block | 16520 | 1492 | 原标注没有摘要/侧栏/目录/竖排等完整细分，不能认为都是原生text。 |
| vertical_text | 竖排文字 | C | — | 0 | 0 | 没有独立标签；旋转属性和竖排文字不是同一概念。 |
| vision_footnote | 图注 | B | figure_caption, figure_footnote, table_footnote | 1091 | 488 | 标题、说明和注释需按原生定义核对；不能简单沿用旧footnote合并。 |

## 原始层级库存

块级仅计layout_dets；merge_list仅作为组成片段，不再加入块级正样本。递归提取其中line_with_spans并继承祖先ignore。不能将所有层级合计为独立框数。

| 层级 | 标签 | 原始条目 | 忽略条目 | 无效几何 | 越界条目 | 候选框 | 候选页 |
|---|---|---:|---:|---:|---:|---:|---:|
| block | abandon | 760 | 760 | 0 | 0 | 0 | 0 |
| block | algorithm_mask | 1 | 1 | 0 | 0 | 0 | 0 |
| block | chart_mask | 26 | 26 | 0 | 0 | 0 | 0 |
| block | code_txt | 40 | 0 | 0 | 0 | 40 | 17 |
| block | code_txt_caption | 3 | 0 | 0 | 0 | 3 | 1 |
| block | equation_caption | 281 | 0 | 0 | 0 | 281 | 55 |
| block | equation_explanation | 16 | 0 | 0 | 0 | 16 | 11 |
| block | equation_isolated | 2066 | 0 | 0 | 1 | 2066 | 313 |
| block | equation_semantic | 160 | 0 | 0 | 0 | 160 | 70 |
| block | figure | 1385 | 0 | 0 | 8 | 1385 | 613 |
| block | figure_caption | 722 | 0 | 0 | 0 | 722 | 356 |
| block | figure_footnote | 175 | 0 | 0 | 0 | 175 | 103 |
| block | footer | 741 | 0 | 0 | 1 | 741 | 529 |
| block | header | 1920 | 0 | 0 | 3 | 1920 | 1076 |
| block | list_group | 79 | 0 | 0 | 0 | 79 | 51 |
| block | need_mask | 6 | 6 | 0 | 0 | 0 | 0 |
| block | organic_chemical_formula_mask | 15 | 15 | 0 | 0 | 0 | 0 |
| block | page_footnote | 126 | 0 | 0 | 0 | 126 | 90 |
| block | page_number | 1188 | 0 | 0 | 0 | 1188 | 1158 |
| block | reference | 275 | 0 | 0 | 0 | 275 | 17 |
| block | table | 665 | 0 | 0 | 0 | 665 | 458 |
| block | table_caption | 486 | 0 | 0 | 0 | 486 | 318 |
| block | table_footnote | 194 | 0 | 0 | 0 | 194 | 140 |
| block | table_mask | 25 | 25 | 0 | 0 | 0 | 0 |
| block | text_block | 16520 | 0 | 0 | 41 | 16520 | 1492 |
| block | text_mask | 465 | 465 | 0 | 0 | 0 | 0 |
| block | title | 4441 | 0 | 0 | 0 | 4441 | 1179 |
| block | unknown_mask | 17 | 17 | 0 | 0 | 0 | 0 |
| span | equation_ignore | 645 | 645 | 0 | 0 | 0 | 0 |
| span | equation_inline | 1462 | 0 | 0 | 0 | 1462 | 96 |
| span | footnote_mark | 26 | 0 | 0 | 0 | 26 | 7 |
| span | text_span | 9160 | 0 | 0 | 0 | 9160 | 342 |

未重复计入的merge_list组成片段：7259条；分布：`{'text_block': 7086, 'figure_caption': 114, 'title': 16, 'table_caption': 8, 'page_footnote': 17, 'code_txt': 11, 'figure_footnote': 2, 'table_footnote': 5}`。
顶层块非空attribute：0；片段旋转属性：`{'normal': 6008, 'rotate180 ': 1}`。没有可直接拆分标题层级的结构化属性。

## 标注范围和阅读顺序

- 含任意span条目的页面：343/1651。行内公式正样本页数见上表。其余页面不能自动认作行内公式负样本；有span的页面也不自动证明标注穷尽。
- 不能只取行内公式正样本页报告全量AP；需要核实完整标注页面及负样本范围。
- 非忽略、几何有效块：有数值order 28116条，缺order 3367条；有order的页面1650，至少两个不同order的页面1602，有重复order的页面0。范围为全部非忽略块，不是旧11组子集。
- 阅读顺序可作为单独子任务，但必须报告参与元素范围、匹配召回、并列/缺失order处理。旧项目的匹配后顺序正确率不是全页面端到端正确率，也不是原作者官方指标。
- 页面来源分布：`{'PPT2PDF': 253, 'academic_literature': 215, 'book': 276, 'colorful_textbook': 159, 'exam_paper': 193, 'historical_document': 5, 'magazine': 149, 'newspaper': 151, 'note': 118, 'research_report': 132}`。各来源页保留原有标注，不重划训练/测试集。

## 已确认旧流程问题与下一步边界

1. run_pp_layout.py的PRED_MAP写formula，本模型输出display_formula，导致独立公式预测被丢弃。旧公式0分不能据此判断模型能力。
2. 旧流程未纳入公式编号及span行内公式；table_footnote也未纳入。
3. 旧title/text/reference等合并组不能冒充原生细分类别；图表标题与图注的原生边界还需核实。
4. *_mask、abandon和equation_ignore只作忽略区域，不作为正样本。无标签不等于负样本。
5. 下一轮先修正明确映射并保存全部25类原始预测；对B类抽样核实，对行内公式先确定完整标注范围。不能凭关键词或模型预测给GT补类别；不能为凑齐任务合并不等义标签。
6. 本轮没有修改推理脚本、GT、历史metrics或模型验收状态；没有生成转换数据或执行GPU推理。暂不下载新数据，先依据本表决定补标或补集。

## 依据与复查

- [OmniDocBench官方标注定义与层级说明](https://github.com/opendatalab/OmniDocBench#evaluation-set-introduction)：用于标签及ignore解释；本地快照SHA是本次统计依据，不声称等于最新公开版本。
- [PP-DocLayoutV2模型卡](https://huggingface.co/PaddlePaddle/PP-DocLayoutV2)：模型身份；25类顺序以本地官方config.json为准。
- [PaddleX类别释义](https://paddlepaddle.github.io/PaddleX/latest/module_usage/tutorials/ocr_modules/layout_analysis.html)：当前页面为V3，公共同名类别释义仅作辅助，不据此宣称V2评估协议已对齐。

复查命令（Python标准库及已有Pillow，CPU统计；默认更新本审计报告，不改任何数据或推理结果）：

```bash
cd __MODEL_ROOT__/resources/runtime/assets/workspace/remaining_detection_eval
python3 audit_m38_omni.py
```
