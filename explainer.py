"""
src/explainability/explainer.py
Interpretability layer for the Medical Risk Assessment System.

Provides:
  1. Top contributing symptoms (token-level attention weights)
  2. Image region importance (patch attention rollout)
  3. GARD-based risk factor narrative
  4. Plain-language explanation generation
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from src.utils.helpers import sigmoid, softmax, clamp

logger = logging.getLogger(__name__)


# ─────────────────────────── Symptom attribution ─────────────────────────────

def compute_symptom_importance(
    attentions: Optional[tuple],
    token_list: List[str],
    method: str = "attention_rollout",
) -> Dict[str, float]:
    """
    Compute per-symptom importance from BioBERT attention weights.

    Parameters
    ----------
    attentions  : tuple of [B, heads, L, L] tensors from BioBERT layers
    token_list  : list of string tokens (from tokeniser)
    method      : 'attention_rollout' | 'last_layer_mean'

    Returns
    -------
    dict: {token: importance_score}
    """
    if attentions is None or not token_list:
        return {t: 1.0 / len(token_list) for t in token_list} if token_list else {}

    try:
        if method == "attention_rollout":
            # Attention Rollout (Abnar & Zuidema 2020)
            rollout = _attention_rollout(attentions)        # [L, L]
            # CLS token attention to each position
            cls_attn = rollout[0, 1: len(token_list) + 1]  # skip CLS itself
        else:
            # Mean last-layer attention to CLS
            last = attentions[-1][0]                         # [heads, L, L]
            cls_attn = last.mean(0)[0, 1: len(token_list) + 1]

        cls_attn = cls_attn.cpu().float().numpy()
        cls_attn = cls_attn[:len(token_list)]

        # Normalise to [0, 1]
        if cls_attn.max() > 0:
            cls_attn = cls_attn / cls_attn.max()

        return {tok: round(float(w), 4) for tok, w in zip(token_list, cls_attn)}

    except Exception as e:
        logger.warning(f"Symptom attribution failed: {e}")
        return {t: round(1.0 / len(token_list), 4) for t in token_list}


def _attention_rollout(attentions: tuple) -> torch.Tensor:
    """Recursively multiply attention matrices with residual connection."""
    eye = None
    rollout = None
    for layer_attn in attentions:
        attn = layer_attn[0].mean(0)                         # [L, L] mean over heads
        attn = attn + torch.eye(attn.shape[0], device=attn.device)
        attn = attn / attn.sum(-1, keepdim=True)
        if rollout is None:
            rollout = attn
        else:
            rollout = torch.matmul(attn, rollout)
    return rollout


# ─────────────────────────── Image patch attention ────────────────────────────

def compute_patch_attention(
    patch_emb: Optional[torch.Tensor],
    image_size: int = 224,
    patch_size: int = 16,
) -> Optional[np.ndarray]:
    """
    Convert patch embeddings → spatial attention map [H, W] via self-similarity.
    This is a lightweight proxy for Grad-CAM in the absence of task-specific labels.

    Returns
    -------
    heatmap : np.ndarray [H_patches, W_patches] or None
    """
    if patch_emb is None:
        return None
    try:
        # Self-similarity of patches to mean patch (proxy for "interesting" regions)
        mean_patch = patch_emb[0].mean(0, keepdim=True)      # [1, D]
        sim = F.cosine_similarity(patch_emb[0], mean_patch.expand_as(patch_emb[0]), dim=-1)
        n_patches = sim.shape[0]
        grid = int(n_patches ** 0.5)
        heatmap = sim.view(grid, grid).cpu().float().numpy()
        # Normalise
        heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
        return heatmap
    except Exception as e:
        logger.warning(f"Patch attention failed: {e}")
        return None


# ─────────────────────────── Explanation narrative ───────────────────────────

def build_explanation(
    disease_label: str,
    risk_score: float,
    risk_std: float,
    symptom_importance: Dict[str, float],
    gard_result: Dict,
    disease_probs: Dict[str, float],
) -> Dict:
    """
    Compose a structured, human-readable explanation.

    Returns
    -------
    dict with:
      summary       : short plain-language summary
      top_symptoms  : list of {symptom, importance}
      risk_narrative: 2-3 sentence clinical narrative
      confidence    : confidence level string
      differential  : top-3 differential diagnoses with probabilities
    """
    # Sort symptoms by importance
    sorted_symptoms = sorted(symptom_importance.items(), key=lambda x: x[1], reverse=True)
    top_symptoms = [
        {"symptom": sym, "importance": round(imp, 3)}
        for sym, imp in sorted_symptoms[:5]
    ]

    # Risk label
    risk_pct = round(risk_score * 100, 1)
    if risk_score < 0.25:
        risk_label = "Low"
        risk_color = "green"
    elif risk_score < 0.50:
        risk_label = "Moderate"
        risk_color = "yellow"
    elif risk_score < 0.75:
        risk_label = "High"
        risk_color = "orange"
    else:
        risk_label = "Critical"
        risk_color = "red"

    # Confidence from uncertainty
    confidence_pct = round(clamp(1 - risk_std * 4) * 100, 1)
    if confidence_pct > 80:
        confidence_label = "High"
    elif confidence_pct > 60:
        confidence_label = "Moderate"
    else:
        confidence_label = "Low"

    # Rare disease narrative
    rare_narrative = ""
    if gard_result.get("rare_candidates"):
        top_rare = gard_result["rare_candidates"][0]
        rare_narrative = (
            f" GARD analysis identified '{top_rare['name']}' as a rare disease candidate "
            f"(match score: {top_rare['match_score']:.2f})."
        )

    # Differential diagnoses
    sorted_diseases = sorted(disease_probs.items(), key=lambda x: x[1], reverse=True)
    differential = [
        {"disease": d, "probability": round(p * 100, 1)}
        for d, p in sorted_diseases[:3]
    ]

    # Top symptoms text
    sym_text = ", ".join([s["symptom"] for s in top_symptoms[:3]]) if top_symptoms else "unspecified"

    summary = (
        f"Primary assessment: {disease_label} with {risk_label} risk ({risk_pct}%). "
        f"Key contributing factors: {sym_text}.{rare_narrative}"
    )

    risk_narrative = (
        f"The multimodal analysis yields a risk score of {risk_pct}% ({risk_label} risk) "
        f"with {confidence_label.lower()} model confidence ({confidence_pct}%). "
        f"The image features and symptom pattern most strongly suggest {disease_label}. "
        f"{'GARD enrichment indicates potential rare disease overlap. ' if rare_narrative else ''}"
        f"Clinical correlation and specialist review are recommended."
    )

    return {
        "summary": summary,
        "risk_label": risk_label,
        "risk_color": risk_color,
        "risk_percentage": risk_pct,
        "confidence_label": confidence_label,
        "confidence_percentage": confidence_pct,
        "top_symptoms": top_symptoms,
        "risk_narrative": risk_narrative,
        "differential_diagnoses": differential,
        "gard_rare_candidates": gard_result.get("rare_candidates", []),
        "gard_premium": gard_result.get("gard_premium", 0.0),
        "symptom_risk_weights": gard_result.get("symptom_weights", {}),
    }


# ─────────────────────────── Heatmap visualisation ───────────────────────────

def overlay_heatmap_on_image(
    pil_image,
    heatmap: Optional[np.ndarray],
    alpha: float = 0.4,
) -> Optional[object]:
    """
    Overlay a heatmap on the original image using matplotlib colormap.
    Returns a PIL image or None.
    """
    if heatmap is None:
        return None
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.cm as cm
        from PIL import Image
        import io

        img_arr = np.array(pil_image.resize((224, 224)))
        heatmap_resized = np.array(
            Image.fromarray((heatmap * 255).astype(np.uint8)).resize((224, 224))
        ) / 255.0

        colormap = cm.jet(heatmap_resized)[:, :, :3]      # RGB
        overlay = (1 - alpha) * img_arr / 255.0 + alpha * colormap
        overlay = np.clip(overlay * 255, 0, 255).astype(np.uint8)

        return Image.fromarray(overlay)
    except Exception as e:
        logger.warning(f"Heatmap overlay failed: {e}")
        return None
