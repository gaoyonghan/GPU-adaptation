# 公共评测协议兼容性

现有公共 evaluate.py 只接受 single_label_classification，且使用 Top-1/Top-5。以下原任务不能无损映射，未修改公共规则。建议维护者另行定义检索、偏好与 IQA 协议；未获决定前不可计为公共分类交付通过。

- BAAI/AltCLIP：retrieval
- Bingsu/clip-vit-large-patch14-ko：retrieval
- HuggingFaceM4/siglip-so400m-14-980-flash-attn2-navit：retrieval
- Marqo/marqo-fashionCLIP：retrieval
- Marqo/marqo-fashionSigLIP：retrieval
- google/siglip-base-patch16-256-multilingual：retrieval
- google/siglip-so400m-patch14-224：retrieval
- google/siglip-so400m-patch16-256-i18n：retrieval
- kakaobrain/align-base：retrieval
- lyua1225/clip-huge-zh-75k-steps-bs4096：retrieval
- patrickjohncyh/fashion-clip：retrieval
- q-future/one-align：image_quality
- yuvalkirstain/PickScore_v1：pairwise_preference

时序、检索候选集等小样本诊断与原全量指标不直接比较。Fashion 的诊断范围只是文档编码器，不等同完整检索流程验证。
