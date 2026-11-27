"""Local Hugging Face CLIP checkpoint adapter for the official benchmark runner."""

import torch
from transformers import AutoImageProcessor, AutoModel, AutoTokenizer


class HFImageTransform:
    def __init__(self, checkpoint):
        self.processor = AutoImageProcessor.from_pretrained(
            checkpoint, local_files_only=True, use_fast=False
        )

    def __call__(self, image):
        return self.processor(images=image, return_tensors="pt")["pixel_values"][0]


class HFTokenizer:
    def __init__(self, checkpoint):
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint, local_files_only=True)

    def __call__(self, texts):
        return self.tokenizer(
            texts, padding="max_length", truncation=True,
            max_length=self.tokenizer.model_max_length, return_tensors="pt",
        )["input_ids"]


class HFCLIP(torch.nn.Module):
    def __init__(self, checkpoint):
        super().__init__()
        self.model = AutoModel.from_pretrained(checkpoint, local_files_only=True)
        self.pad_token_id = self.model.config.text_config.pad_token_id

    def encode_text(self, input_ids):
        attention_mask = input_ids.ne(self.pad_token_id).long()
        return self.model.get_text_features(
            input_ids=input_ids, attention_mask=attention_mask
        )

    def encode_image(self, pixel_values):
        return self.model.get_image_features(pixel_values=pixel_values)


def load_hf_clip(model_name, pretrained, cache_dir=None, device="cpu"):
    del model_name, cache_dir
    model = HFCLIP(pretrained).to(device)
    return model, HFImageTransform(pretrained), HFTokenizer(pretrained)
