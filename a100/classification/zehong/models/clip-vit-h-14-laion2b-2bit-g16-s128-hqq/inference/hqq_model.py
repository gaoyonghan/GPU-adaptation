"""官方HQQ视觉塔＋OpenCLIP文本分类器；保留七模板与FP16计算。"""
import gc
import os
from types import SimpleNamespace
import torch
import open_clip
from hqq.engine.timm import HQQtimm
from huggingface_hub import snapshot_download

MODEL_ID = "mobiuslabsgmbh/CLIP-ViT-H-14-laion2B-2bit_g16_s128-HQQ"

MODEL_REVISION = "b0729107678a03c203010c4aaf6634141e4ea77c"

OPEN_CLIP_ARCH = "ViT-H-14"

OPEN_CLIP_WEIGHTS = "laion2B-s32B-b79K"

TEMPLATES = (
    "itap of a {}.",
    "a origami {}.",
    "a bad photo of the {}.",
    "a photo of the large {}.",
    "a photo of the small {}.",
    "a {} in a video game.",
    "art of the {}.",
)

@torch.inference_mode()
def build_classifier(text_model, tokenizer, classnames, batch_size=256):
    """按模型卡先平均每类的七个文本特征，再归一化。"""
    prompts = [template.format(classname) for classname in classnames for template in TEMPLATES]
    chunks = []
    for start in range(0, len(prompts), batch_size):
        tokens = tokenizer(prompts[start:start + batch_size]).to("cuda:0")
        chunks.append(text_model.encode_text(tokens, normalize=False).half())
    features = torch.cat(chunks).reshape(len(classnames), len(TEMPLATES), -1).mean(1)
    features = features / features.norm(dim=1, keepdim=True)
    return features.T.contiguous()

class HQQZeroShotClassifier(torch.nn.Module):
    """把量化视觉塔与固定零样本文本分类器组合成标准 Logits 接口。"""
    def __init__(self, visual, classifier):
        super().__init__(); self.visual = visual
        self.register_buffer("classifier", classifier)
        self.templates = TEMPLATES
        self.metadata = {
            "model_id": MODEL_ID, "revision": MODEL_REVISION,
            "visual": "vit_huge_patch14_clip_224.laion2b HQQ 2-bit g16 s128",
            "text_model": f"{OPEN_CLIP_ARCH}/{OPEN_CLIP_WEIGHTS}",
            "num_labels": int(classifier.shape[1]), "device": "cuda:0",
            "compute_dtype": "torch.float16",
        }

    def forward(self, images):
        features = self.visual(images)
        features = features / features.norm(dim=-1, keepdim=True)
        return features @ self.classifier

class Processor:
    def __init__(self,transform): self.transform=transform
    def __call__(self,images,return_tensors="pt"):
        return {"pixel_values":self.transform(images).half().unsqueeze(0)}
    def __str__(self): return str(self.transform)+"; float16"

def load_model(source,offline,text_weights):
    from pathlib import Path
    snapshot=source if Path(source).is_dir() else snapshot_download(source,revision=MODEL_REVISION,local_files_only=offline)
    visual=HQQtimm.from_quantized(snapshot,compute_dtype=torch.float16,device="cuda:0").eval()
    if offline and not Path(text_weights).is_file():
        raise FileNotFoundError(f"离线模式需OpenCLIP文本权重: {text_weights}")
    pretrained=str(text_weights) if Path(text_weights).is_file() else OPEN_CLIP_WEIGHTS
    text,_,transform=open_clip.create_model_and_transforms(OPEN_CLIP_ARCH,pretrained=pretrained,cache_dir=os.environ.get("HF_HUB_CACHE"))
    text=text.half().to("cuda:0").eval()
    classifier=build_classifier(text,open_clip.get_tokenizer(OPEN_CLIP_ARCH),open_clip.IMAGENET_CLASSNAMES)
    model=HQQZeroShotClassifier(visual,classifier).eval()
    model.config=SimpleNamespace(id2label=dict(enumerate(open_clip.IMAGENET_CLASSNAMES)))
    del text
    gc.collect(); torch.cuda.empty_cache()
    return Processor(transform),model

def forward(model,inputs): return model(inputs["pixel_values"])
