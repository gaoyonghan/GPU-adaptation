import json
import os

import open_clip


def load_open_clip(model_name: str = "ViT-B-32-quickgelu", pretrained: str = "laion400m_e32", cache_dir: str = None, device="cpu"):
    preprocess_kwargs = {}
    # Local checkpoint paths do not carry OpenCLIP's named-pretrained metadata.
    # Recover the checkpoint-native preprocessing shipped beside the weights so
    # local evaluation matches the official named-checkpoint protocol.
    if os.path.isfile(pretrained):
        config_path = os.path.join(os.path.dirname(pretrained), "open_clip_config.json")
        if os.path.isfile(config_path):
            with open(config_path, encoding="utf-8") as stream:
                local_config = json.load(stream)
            preprocess_cfg = local_config.get("preprocess_cfg", {})
            argument_names = {
                "mean": "image_mean",
                "std": "image_std",
                "interpolation": "image_interpolation",
                "resize_mode": "image_resize_mode",
            }
            for key, argument_name in argument_names.items():
                if key in preprocess_cfg:
                    preprocess_kwargs[argument_name] = preprocess_cfg[key]
    model, _, transform = open_clip.create_model_and_transforms(
        model_name,
        pretrained=pretrained,
        cache_dir=cache_dir,
        **preprocess_kwargs,
    )
    model = model.to(device)
    tokenizer = open_clip.get_tokenizer(model_name)
    return model, transform, tokenizer
