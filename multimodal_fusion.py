"""
src/fusion/multimodal_fusion.py
Attention-based multimodal fusion of image and text embeddings.

Architecture:
  ┌──────────────────────────────────────────────────────────────────────────┐
  │  image_emb [B, D]   text_emb [B, D]   rad_emb [B, D]                   │
  │        │                  │                  │                           │
  │        └──────────────────┴──────────────────┘                          │
  │                     stack → [B, 3, D]                                   │
  │                           │                                              │
  │            Cross-Modal Transformer Encoder                               │
  │                  (num_heads=8, num_layers=2)                             │
  │                           │                                              │
  │               CLS aggregation → [B, D]                                  │
  │                           │                                              │
  │                    MLP Prediction Head                                   │
  │                           │                                              │
  │         disease_logits [B, C]   risk_score [B, 1]                       │
  └──────────────────────────────────────────────────────────────────────────┘
"""

import logging
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from configs.config import APP_CONFIG

logger = logging.getLogger(__name__)

_model_cfg = APP_CONFIG.model


class CrossModalAttentionFusion(nn.Module):
    """
    Fuses three modality tokens {image_clip, image_rad, text_bio} via
    a lightweight Transformer Encoder, then produces disease logits and
    a calibrated risk score.
    """

    NUM_MODALITIES = 3          # BiomedCLIP, RadVLP, BioBERT

    def __init__(
        self,
        embed_dim: int = _model_cfg.fusion_hidden_dim,       # 512
        num_heads: int = _model_cfg.fusion_num_heads,         # 8
        num_layers: int = 2,
        num_classes: int = _model_cfg.num_disease_classes,    # 14
        dropout: float = _model_cfg.fusion_dropout,           # 0.1
    ):
        super().__init__()
        self.embed_dim = embed_dim

        # ── Learnable modality-type embeddings (like positional for BERT) ──
        self.modality_embeddings = nn.Embedding(self.NUM_MODALITIES, embed_dim)

        # ── CLS token for pooling ──────────────────────────────────────────
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        # ── Cross-modal Transformer encoder ───────────────────────────────
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,                   # Pre-norm (more stable)
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(embed_dim)

        # ── Prediction head: disease classification ────────────────────────
        self.disease_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim // 2, num_classes),
        )

        # ── Prediction head: risk regression ──────────────────────────────
        self.risk_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim // 4, 1),
        )

        # ── Uncertainty estimation (Monte Carlo Dropout) ───────────────────
        self.mc_dropout = nn.Dropout(p=0.15)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(
        self,
        clip_emb: torch.Tensor,       # [B, D] – BiomedCLIP image
        rad_emb: torch.Tensor,        # [B, D] – RadVLP image
        text_emb: torch.Tensor,       # [B, D] – BioBERT text
        return_attention: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Returns
        -------
        disease_logits : [B, num_classes]
        risk_score     : [B, 1]  – sigmoid output in [0, 1]
        attn_weights   : [B, num_heads, L, L] or None
        """
        B = clip_emb.shape[0]
        modality_ids = torch.arange(self.NUM_MODALITIES, device=clip_emb.device)
        mod_emb = self.modality_embeddings(modality_ids)   # [3, D]

        # Stack modality tokens: [B, 3, D]
        tokens = torch.stack([clip_emb, rad_emb, text_emb], dim=1) + mod_emb.unsqueeze(0)

        # Prepend CLS token: [B, 1+3, D]
        cls = self.cls_token.expand(B, -1, -1)
        tokens = torch.cat([cls, tokens], dim=1)

        # Apply MC dropout for uncertainty
        tokens = self.mc_dropout(tokens)

        # Cross-modal attention
        fused = self.transformer(tokens)                   # [B, 4, D]
        fused = self.norm(fused)

        cls_out = fused[:, 0, :]                           # [B, D]

        disease_logits = self.disease_head(cls_out)        # [B, C]
        risk_logit = self.risk_head(cls_out)               # [B, 1]
        risk_score = torch.sigmoid(risk_logit)             # [B, 1]

        return disease_logits, risk_score, None

    @torch.no_grad()
    def predict(
        self,
        clip_emb: torch.Tensor,
        rad_emb: torch.Tensor,
        text_emb: torch.Tensor,
        num_mc_samples: int = 10,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Monte Carlo Dropout inference for uncertainty estimation.

        Returns
        -------
        mean_logits : [B, C]
        mean_risk   : [B, 1]
        risk_std    : [B, 1]  – aleatoric uncertainty
        """
        self.train()              # Enable dropout
        logits_list, risk_list = [], []

        for _ in range(num_mc_samples):
            logits, risk, _ = self(clip_emb, rad_emb, text_emb)
            logits_list.append(logits.unsqueeze(0))
            risk_list.append(risk.unsqueeze(0))

        self.eval()

        all_logits = torch.cat(logits_list, dim=0)     # [MC, B, C]
        all_risk = torch.cat(risk_list, dim=0)         # [MC, B, 1]

        mean_logits = all_logits.mean(0)               # [B, C]
        mean_risk = all_risk.mean(0)                   # [B, 1]
        risk_std = all_risk.std(0)                     # [B, 1]

        return mean_logits, mean_risk, risk_std


class ConcatFusion(nn.Module):
    """
    Simple concatenation baseline (alternative to attention fusion).
    concat([clip, rad, text]) → MLP → predictions
    """

    def __init__(
        self,
        embed_dim: int = _model_cfg.fusion_hidden_dim,
        num_classes: int = _model_cfg.num_disease_classes,
        dropout: float = _model_cfg.fusion_dropout,
    ):
        super().__init__()
        in_dim = embed_dim * 3
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
        )
        self.disease_head = nn.Linear(embed_dim // 2, num_classes)
        self.risk_head = nn.Linear(embed_dim // 2, 1)

    def forward(
        self,
        clip_emb: torch.Tensor,
        rad_emb: torch.Tensor,
        text_emb: torch.Tensor,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor, None]:
        x = torch.cat([clip_emb, rad_emb, text_emb], dim=-1)
        h = self.mlp(x)
        return self.disease_head(h), torch.sigmoid(self.risk_head(h)), None


def build_fusion_module(fusion_type: str = "attention") -> nn.Module:
    if fusion_type == "attention":
        logger.info("Using CrossModalAttentionFusion.")
        return CrossModalAttentionFusion()
    elif fusion_type == "concat":
        logger.info("Using ConcatFusion.")
        return ConcatFusion()
    else:
        raise ValueError(f"Unknown fusion_type: {fusion_type}. Choose 'attention' or 'concat'.")
