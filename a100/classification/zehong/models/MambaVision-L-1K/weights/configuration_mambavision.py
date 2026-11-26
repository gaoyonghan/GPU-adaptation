from transformers import PretrainedConfig

class MambaVisionConfig(PretrainedConfig):
    model_type = "mambavision"

    def __init__(
        self,
        depths=[3, 3, 10, 5],
        num_heads=[4, 8, 16, 32],
        window_size=[8, 8, 14, 7],
        dim=196,
        in_dim=64,
        mlp_ratio=4,
        drop_path_rate=0.3,
        layer_scale=1e-5,
        layer_scale_conv=None,
        **kwargs,
    ):
        self.depths = depths
        self.num_heads = num_heads
        self.window_size = window_size
        self.dim = dim
        self.in_dim = in_dim
        self.mlp_ratio = mlp_ratio
        self.drop_path_rate = drop_path_rate
        self.layer_scale=layer_scale
        self.layer_scale_conv=layer_scale_conv
        super().__init__(**kwargs)