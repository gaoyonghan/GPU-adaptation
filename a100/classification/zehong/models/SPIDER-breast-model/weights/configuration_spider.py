from transformers import PretrainedConfig, AutoConfig
from typing import List, Optional

class SPIDERPatchConfig(PretrainedConfig):
    model_type = "SPIDERPatchClassifier"
    def __init__(
        self,
        vocab_size: int = 1,
        hidden_size: int = 1024,
        num_hidden_layers: int = 1,
        num_attention_heads: int = 8,
        intermediate_size: int = 1024,
        max_position_embeddings: int = 25,
        is_decoder: bool = False,
        hidden_dropout_prob: float = 0.5,
        attention_probs_dropout_prob: float = 0.5,
        head_dropout_prob: float = 0.5,
        num_labels: int = 2,
        image_size: int = 224,
        class_names: Optional[List[str]] = None,
        **kwargs,
    ):
        """
        Custom configuration for SPIDERPatchClassifier.
        
        Args:
            vocab_size (int): Dummy vocab size since token embeddings are unused.
            hidden_size (int): Hidden size for the transformer encoder.
            num_hidden_layers (int): Number of transformer layers.
            num_attention_heads (int): Number of attention heads.
            intermediate_size (int): Size of the feed-forward layer.
            max_position_embeddings (int): Maximum number of position embeddings.
            is_decoder (bool): Whether the model is used as a decoder.
            hidden_dropout_prob (float): Dropout probability for hidden layers.
            attention_probs_dropout_prob (float): Dropout probability for attention.
            head_dropout_prob (float): Dropout probability before the classification head.
            num_labels (int): Number of classification labels.
            image_size (int): Patch size for the input images.
            class_names (Optional[List[str]]): List of class names mapping label indices to strings.
        """
        super().__init__(
            **kwargs
            )
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.max_position_embeddings = max_position_embeddings
        self.is_decoder = is_decoder
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.head_dropout_prob = head_dropout_prob
        self.num_labels = num_labels
        self.image_size = image_size
        self.class_names = class_names

    def to_dict(self):
        """Ensure num_labels is included when saving."""
        output = super().to_dict()
        output["num_labels"] = self.num_labels  # Forces inclusion in config.json
        return output