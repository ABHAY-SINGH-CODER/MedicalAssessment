"""
src/models/biobert_encoder.py
BioBERT text encoder – extracts contextualized embeddings from symptom text.
Model: dmis-lab/biobert-base-cased-v1.2
"""

import logging
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
from transformers import AutoModel

from configs.config import APP_CONFIG

logger = logging.getLogger(__name__)


class BioBERTEncoder(nn.Module):
    """
    Wraps dmis-lab/biobert-base-cased-v1.2.

    Outputs:
        cls_emb  : [B, 768]  – [CLS] token embedding (sentence-level)
        token_emb: [B, L, 768] – full sequence embeddings (for attention explainability)
    """

    MODEL_NAME = APP_CONFIG.model.biobert_model_name
    HIDDEN_SIZE = APP_CONFIG.model.biobert_hidden_size

    def __init__(self, freeze_layers: int = 10):
        super().__init__()
        logger.info(f"Loading BioBERT: {self.MODEL_NAME}")
        try:
            self.bert = AutoModel.from_pretrained(
                self.MODEL_NAME,
                cache_dir=APP_CONFIG.model.cache_dir,
                output_attentions=True,
                output_hidden_states=True,
            )
        except Exception as e:
            logger.warning(f"BioBERT load failed ({e}); falling back to bert-base-cased.")
            self.bert = AutoModel.from_pretrained(
                "bert-base-cased",
                cache_dir=APP_CONFIG.model.cache_dir,
                output_attentions=True,
                output_hidden_states=True,
            )

        # Freeze bottom N transformer layers to save memory
        self._freeze_layers(freeze_layers)

        # Projection head → shared fusion dim
        self.proj = nn.Sequential(
            nn.Linear(self.HIDDEN_SIZE, APP_CONFIG.model.fusion_hidden_dim),
            nn.LayerNorm(APP_CONFIG.model.fusion_hidden_dim),
            nn.GELU(),
        )

    def _freeze_layers(self, n: int) -> None:
        modules = [self.bert.embeddings, *self.bert.encoder.layer[:n]]
        for m in modules:
            for p in m.parameters():
                p.requires_grad = False
        logger.info(f"BioBERT: froze {n} encoder layers.")

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, tuple]:
        """
        Returns
        -------
        cls_proj    : [B, fusion_dim]
        token_emb   : [B, L, hidden_size]
        attentions  : tuple of attention tensors per layer
        """
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        cls_emb = outputs.last_hidden_state[:, 0, :]          # [B, H]
        token_emb = outputs.last_hidden_state                  # [B, L, H]
        attentions = outputs.attentions                         # tuple[Tensor] or None

        cls_proj = self.proj(cls_emb)                          # [B, fusion_dim]
        return cls_proj, token_emb, attentions

    @torch.no_grad()
    def encode(self, encoding: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, tuple]:
        """Convenience wrapper for inference (no grad)."""
        return self(
            input_ids=encoding["input_ids"],
            attention_mask=encoding["attention_mask"],
            token_type_ids=encoding.get("token_type_ids"),
        )
