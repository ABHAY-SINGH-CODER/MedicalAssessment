"""
src/models/biomedclip_encoder.py
BiomedCLIP image encoder – extracts medical vision embeddings.
Model: microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224 via open_clip.
Falls back to ViT-B/16 OpenAI weights when HF hub is unavailable.
"""

import logging
from typing import Optional, Tuple

import torch
import torch.nn as nn

from configs.config import APP_CONFIG

logger = logging.getLogger(__name__)


class BiomedCLIPEncoder(nn.Module):
    """
    Wraps microsoft/BiomedCLIP via open_clip.

    Outputs:
        img_proj  : [B, fusion_dim]
        patch_emb : [B, num_patches, clip_dim]  – for patch-level attention maps
    """

    CLIP_DIM = APP_CONFIG.model.biomedclip_embed_dim  # 512

    def __init__(self):
        super().__init__()
        self.model = None
        self._load_model()

        # Projection to shared fusion dim
        self.proj = nn.Sequential(
            nn.Linear(self.CLIP_DIM, APP_CONFIG.model.fusion_hidden_dim),
            nn.LayerNorm(APP_CONFIG.model.fusion_hidden_dim),
            nn.GELU(),
        )

    def _load_model(self):
        try:
            import open_clip
            logger.info("Loading BiomedCLIP via open_clip…")
            model, _, _ = open_clip.create_model_and_transforms(
                "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224",
                cache_dir=APP_CONFIG.model.cache_dir,
            )
            self.model = model.visual
            logger.info("BiomedCLIP loaded successfully.")
        except Exception as e:
            logger.warning(f"BiomedCLIP load failed ({e}). Trying ViT-B/16-224 OpenAI fallback…")
            try:
                import open_clip
                model, _, _ = open_clip.create_model_and_transforms(
                    "ViT-B-16",
                    pretrained="openai",
                    cache_dir=APP_CONFIG.model.cache_dir,
                )
                self.model = model.visual
                logger.info("ViT-B/16 fallback loaded.")
            except Exception as e2:
                logger.error(f"All CLIP loads failed: {e2}. Using random-init ViT stub.")
                self.model = _StubVisionEncoder(self.CLIP_DIM)

    def forward(self, pixel_values: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Parameters
        ----------
        pixel_values : [B, 3, H, W]

        Returns
        -------
        img_proj  : [B, fusion_dim]
        patch_emb : [B, num_patches, CLIP_DIM]  (None for stub)
        """
        if isinstance(self.model, _StubVisionEncoder):
            img_feat, patch_emb = self.model(pixel_values)
        else:
            # open_clip visual returns [B, CLIP_DIM] when called directly
            img_feat = self.model(pixel_values)          # [B, CLIP_DIM]
            patch_emb = None

        img_proj = self.proj(img_feat)                   # [B, fusion_dim]
        return img_proj, patch_emb

    @torch.no_grad()
    def encode(self, pixel_values: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        return self(pixel_values)


class _StubVisionEncoder(nn.Module):
    """Deterministic ViT-like stub for offline / test environments."""

    def __init__(self, dim: int = 512):
        super().__init__()
        self.dim = dim
        self.patch_embed = nn.Conv2d(3, dim, kernel_size=16, stride=16)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        encoder_layer = nn.TransformerEncoderLayer(d_model=dim, nhead=8, batch_first=True, norm_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=4)
        self.norm = nn.LayerNorm(dim)
        self.pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B = x.shape[0]
        patches = self.patch_embed(x)                    # [B, dim, h, w]
        patches = patches.flatten(2).transpose(1, 2)     # [B, num_patches, dim]
        cls = self.cls_token.expand(B, -1, -1)           # [B, 1, dim]
        tokens = torch.cat([cls, patches], dim=1)        # [B, 1+P, dim]
        tokens = self.transformer(tokens)
        tokens = self.norm(tokens)
        cls_out = tokens[:, 0, :]                        # [B, dim]
        return cls_out, tokens[:, 1:, :]                 # img_feat, patch_emb
