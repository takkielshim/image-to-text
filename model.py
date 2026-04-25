"""EfficientNet-B0 image encoder and DistilBERT text encoder.
Both project into a shared 256-dimensional L2-normalised embedding space.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from transformers import DistilBertModel


class ImageEncoder(nn.Module):
    def __init__(self, embed_dim: int = 256):
        super().__init__()
        backbone = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        self.features = backbone.features
        self.avgpool = backbone.avgpool
        self.projection = nn.Sequential(
            nn.Flatten(),
            nn.Linear(1280, embed_dim),
            nn.LayerNorm(embed_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        return self.projection(x)


class TextEncoder(nn.Module):
    def __init__(self, embed_dim: int = 256):
        super().__init__()
        self.bert = DistilBertModel.from_pretrained('distilbert-base-uncased')
        self.projection = nn.Sequential(
            nn.Linear(768, embed_dim),
            nn.LayerNorm(embed_dim),
        )

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls = out.last_hidden_state[:, 0, :]  # [CLS] token representation
        return self.projection(cls)


class ImageTextModel(nn.Module):
    def __init__(self, embed_dim: int = 256):
        super().__init__()
        self.image_encoder = ImageEncoder(embed_dim)
        self.text_encoder = TextEncoder(embed_dim)
        # Learnable temperature parameter, initialised to 0.07
        self.logit_scale = nn.Parameter(torch.ones([]) * 0.07)

    def encode_image(self, image: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.image_encoder(image), dim=-1)

    def encode_text(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.text_encoder(input_ids, attention_mask), dim=-1)

    def forward(
        self,
        image: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> tuple:
        img_emb = self.encode_image(image)
        txt_emb = self.encode_text(input_ids, attention_mask)
        return img_emb, txt_emb, self.logit_scale.exp()
