import torch
import torch.nn as nn
from transformers import (
    PreTrainedModel,
    BertModel,
    BertConfig,
    AutoModel,
)
from transformers.modeling_outputs import ModelOutput
from dataclasses import dataclass
from typing import Optional, Tuple, List

from .configuration_spider import SPIDERPatchConfig

@dataclass
class SPIDERPatchClassifierOutput(ModelOutput):
    """
    Custom output class for SPIDERPatchClassifier.
    
    Attributes:
        logits (torch.FloatTensor): Classification logits.
        hidden_states (Optional[Tuple[torch.FloatTensor]]): Hidden states from the transformer encoder.
        attentions (Optional[Tuple[torch.FloatTensor]]): Attention maps from the transformer encoder.
        label (Optional[torch.LongTensor]): Predicted label indices.
        predicted_class_names (Optional[List[str]]): Human-readable class names corresponding to the predictions.
    """
    logits: torch.FloatTensor
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    label: Optional[torch.LongTensor] = None
    predicted_class_names: Optional[List[str]] = None

class SPIDERPatchClassifier(PreTrainedModel):
    config_class = SPIDERPatchConfig    
    def __init__(self, config):
        """
        Initializes the SPIDERPatchClassifier.
        
        Args:
            config (SPIDERPatchConfig): Custom configuration instance.
        """
        
        super().__init__(config)
        self.feature_extractor = AutoModel.from_pretrained(
            "histai/hibou-L",
            trust_remote_code=True,
        )
        
        bert_config = BertConfig(
            vocab_size=config.vocab_size,
            hidden_size=config.hidden_size,
            num_hidden_layers=config.num_hidden_layers,
            num_attention_heads=config.num_attention_heads,
            intermediate_size=config.intermediate_size,
            max_position_embeddings=config.max_position_embeddings,
            is_decoder=config.is_decoder,
            hidden_dropout_prob=config.hidden_dropout_prob,
            attention_probs_dropout_prob=config.attention_probs_dropout_prob,
        )
        self.classifier = BertModel(bert_config, add_pooling_layer=False)
        
        self.head = nn.Linear(config.hidden_size, config.num_labels)
        self.head_dropout = nn.Dropout(config.head_dropout_prob)
        
        # Optional embedder.
        if config.hidden_size != 1024:
            self.embedder = nn.Sequential(
                nn.Linear(1024, config.hidden_size),
                nn.GELU(),
                nn.LayerNorm(config.hidden_size),
            )
        else:
            self.embedder = nn.Identity()

    def label_to_class(self, labels: torch.Tensor) -> List[str]:
        """
        Converts label indices to class names.
        
        Args:
            labels (torch.Tensor): Tensor of label indices.
            
        Returns:
            List[str]: List of class names corresponding to the labels.
        """
        if self.config.class_names is None:
            return [str(label.item()) for label in labels]
        return [self.config.class_names[label.item()] for label in labels]

    def forward(self, pixel_values: torch.Tensor) -> SPIDERPatchClassifierOutput:
        """
        Forward pass for SPIDERPatchClassifier.
        
        Args:
            pixel_values (torch.Tensor): Input tensor of shape (batch_size, 3, H, W).
        
        Returns:
            SPIDERPatchClassifierOutput: Output object containing logits, label, and predicted class names.
        """
        step = 224

        # Validate dimensions.
        assert pixel_values.shape[2] % step == 0, f"Input width {pixel_values.shape[2]} not divisible by {step}"
        assert pixel_values.shape[3] % step == 0, f"Input height {pixel_values.shape[3]} not divisible by {step}"
        assert (pixel_values.shape[2] // step) % 2 == 1, "Input width must yield an odd number of patches"
        assert (pixel_values.shape[3] // step) % 2 == 1, "Input height must yield an odd number of patches"

        # Unfold image into patches.
        patches = pixel_values.unfold(2, step, step).unfold(3, step, step)
        patches = patches.permute(0, 2, 3, 1, 4, 5).contiguous().view(-1, 3, step, step)

        # Extract patch features.
        features = self.feature_extractor(patches).pooler_output
        features = features.view(pixel_values.shape[0], -1, features.shape[1])
        
        # Separate central patch from context.
        central_idx = features.shape[1] // 2
        central_features = features[:, central_idx:central_idx + 1, :]
        context_features = torch.cat([features[:, :central_idx, :], features[:, central_idx + 1:, :]], dim=1)
        input_features = torch.cat([central_features, context_features], dim=1)
        input_features = self.embedder(input_features)

        attention_mask = torch.ones(input_features.shape[:2], dtype=torch.float, device=input_features.device)
        bert_output = self.classifier(inputs_embeds=input_features, attention_mask=attention_mask)
        logits = self.head(self.head_dropout(bert_output.last_hidden_state[:, 0, :]))

        predicted_label = logits.argmax(dim=-1)
        predicted_class_names = self.label_to_class(predicted_label)

        hidden_states = getattr(bert_output, "hidden_states", None)
        attentions = getattr(bert_output, "attentions", None)

        return SPIDERPatchClassifierOutput(
            logits=logits,
            hidden_states=hidden_states,
            attentions=attentions,
            label=predicted_label,
            predicted_class_names=predicted_class_names,
        )

if __name__ == "__main__":
    example_class_names = [
        "Actinic keratosis",
        "Apocrine glands",
        "Basal cell carcinoma",
        "Carcinoma in situ",
        "Collagen",
        "Epidermis",
        "Fat",
        "Follicle",
        "Inflammation",
        "Invasive melanoma",
        "Kaposi's sarcoma",
        "Keratin",
        "Melanoma in situ",
        "Mercel cell carcinoma",
        "Muscle",
        "Necrosis",
        "Nerves",
        "Sebaceous gland",
        "Seborrheic keratosis",
        "Solar elastosis",
        "Squamous cell carcinoma",
        "Vessels",
        "Wart",
    ]
    config = SPIDERPatchConfig(
        num_labels=2,
        image_size=224,
        class_names=example_class_names,
    )
    model = SPIDERPatchClassifier(config=config)
    dummy_input = torch.rand((2, 3, 1120, 1120))
    output = model(dummy_input)
    print("Logits shape:", output.logits.shape)
    print("Predicted labels:", output.label)
    print("Predicted class names:", output.predicted_class_names)