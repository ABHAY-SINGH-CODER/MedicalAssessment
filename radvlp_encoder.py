"""
src/models/radvlp_encoder.py
Rad-VLP image encoder using Microsoft's RAD-DINO (radiology-specific ViT).
Model: microsoft/rad-dino (DINOv2 pretrained on chest X-ray data).
Falls back to BiomedCLIP visual weights when unavailable.
"""

import logging
from typing import Optional, Tuple

import torch
import torch.nn as nn
from transformers import AutoModel

from configs.config import APP_CONFIG

logger = logging.getLogger(__name__)


class RadVLPEncoder(nn.Module):
    """
    Radiology-specific vision encoder using RAD-DINO.
    
    RAD-DINO is a Vision Transformer pretrained with DINO on large-scale
    chest X-ray datasets (CheXpert, MIMIC-CXR, etc.) providing 
    radiology-optimised patch-level representations.

    Outputs:
        rad_proj  : [B, fusion_dim]
        patch_emb : [B, num_patches, rad_dim]
    """

    RAD_DIM = APP_CONFIG.model.radvlp_embed_dim       # 768

    def __init__(self):
        super().__init__()
        self.backbone = None
        self._load_model()

        self.proj = nn.Sequential(
            nn.Linear(self.RAD_DIM, APP_CONFIG.model.fusion_hidden_dim),
            nn.LayerNorm(APP_CONFIG.model.fusion_hidden_dim),
            nn.GELU(),
        )

    def _load_model(self):
        try:
            logger.info("Loading RAD-DINO (microsoft/rad-dino)…")
            self.backbone = AutoModel.from_pretrained(
                "microsoft/rad-dino",
                cache_dir=APP_CONFIG.model.cache_dir,
                output_hidden_states=True,
                output_attentions=True,
            )
            logger.info("RAD-DINO loaded successfully.")
        except Exception as e:
            logger.warning(f"RAD-DINO load failed ({e}). Using DINOv2 small fallback…")
            try:
                self.backbone = AutoModel.from_pretrained(
                    "facebook/dinov2-small",
                    cache_dir=APP_CONFIG.model.cache_dir,
                    output_hidden_states=True,
                    output_attentions=True,
                )
                # Update dim to match dinov2-small
                self.RAD_DIM = 384
                logger.info("DINOv2-small fallback loaded.")
            except Exception as e2:
                logger.error(f"DINOv2 load failed: {e2}. Using stub encoder.")
                self.backbone = _RadStub(self.RAD_DIM)

    def forward(self, pixel_values: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Parameters
        ----------
        pixel_values : [B, 3, H, W]

        Returns
        -------
        rad_proj  : [B, fusion_dim]
        patch_emb : [B, num_patches, RAD_DIM]
        """
        if isinstance(self.backbone, _RadStub):
            cls_feat, patch_emb = self.backbone(pixel_values)
        else:
            outputs = self.backbone(pixel_values=pixel_values)
            # DINOv2 / RAD-DINO: last_hidden_state[:, 0] = CLS token
            cls_feat = outputs.last_hidden_state[:, 0, :]       # [B, RAD_DIM]
            patch_emb = outputs.last_hidden_state[:, 1:, :]     # [B, P, RAD_DIM]

        rad_proj = self.proj(cls_feat)
        return rad_proj, patch_emb

    @torch.no_grad()
    def encode(self, pixel_values: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        return self(pixel_values)


class _RadStub(nn.Module):
    """Lightweight ViT stub for offline testing."""

    def __init__(self, dim: int = 768):
        super().__init__()
        self.dim = dim
        self.patch_proj = nn.Linear(3 * 16 * 16, dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        enc_layer = nn.TransformerEncoderLayer(d_model=dim, nhead=8, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=2)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B, C, H, W = x.shape
        p = 16
        patches = x.unfold(2, p, p).unfold(3, p, p)             # [B, C, nh, nw, p, p]
        patches = patches.contiguous().view(B, C, -1, p * p)    # [B, C, N, p²]
        patches = patches.permute(0, 2, 1, 3).flatten(2)        # [B, N, C*p²]
        patches = self.patch_proj(patches)                       # [B, N, dim]
        cls = self.cls_token.expand(B, -1, -1)
        tokens = torch.cat([cls, patches], dim=1)
        tokens = self.encoder(tokens)
        return tokens[:, 0, :], tokens[:, 1:, :]
