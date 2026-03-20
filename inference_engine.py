"""
src/inference_engine.py
Central inference orchestrator for the Multimodal Medical Risk Assessment System.

Pipeline:
  Input → Preprocessing → [BioBERT | BiomedCLIP | RadVLP] → Fusion → GARD → Explainability → Output
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from configs.config import APP_CONFIG
from src.explainability.explainer import (
    build_explanation,
    compute_patch_attention,
    compute_symptom_importance,
    overlay_heatmap_on_image,
)
from src.fusion.multimodal_fusion import build_fusion_module
from src.models.biobert_encoder import BioBERTEncoder
from src.models.biomedclip_encoder import BiomedCLIPEncoder
from src.models.gard_enricher import GARDEnricher
from src.models.radvlp_encoder import RadVLPEncoder
from src.utils.helpers import clamp, decode_base64_image, setup_logging
from src.utils.preprocessing import ImagePreprocessor, TextPreprocessor

logger = logging.getLogger(__name__)


@dataclass
class AssessmentResult:
    """Structured output from the inference engine."""
    # Core predictions
    primary_disease: str
    disease_probabilities: Dict[str, float]
    risk_score: float
    risk_uncertainty: float
    risk_label: str

    # Explainability
    explanation: Dict
    top_symptoms: List[Dict]
    gard_enrichment: Dict

    # Metadata
    inference_time_ms: float
    model_versions: Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return {
            "primary_disease": self.primary_disease,
            "disease_probabilities": self.disease_probabilities,
            "risk_score": round(self.risk_score, 4),
            "risk_uncertainty": round(self.risk_uncertainty, 4),
            "risk_label": self.risk_label,
            "explanation": self.explanation,
            "top_symptoms": self.top_symptoms,
            "gard_enrichment": self.gard_enrichment,
            "metadata": {
                "inference_time_ms": round(self.inference_time_ms, 1),
                "model_versions": self.model_versions,
            },
        }


class MedicalRiskEngine:
    """
    Singleton inference engine.

    Usage
    -----
    engine = MedicalRiskEngine()
    result = engine.predict(image_b64="...", symptoms="fever, cough")
    """

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        setup_logging(APP_CONFIG.log_level)
        logger.info("Initialising MedicalRiskEngine…")
        t0 = time.perf_counter()

        self.device = torch.device(APP_CONFIG.model.device)
        self.cfg = APP_CONFIG.model
        self.labels = self.cfg.disease_labels

        # ── Load models ───────────────────────────────────────────────────
        self.img_preprocessor = ImagePreprocessor()
        self.txt_preprocessor = TextPreprocessor()

        self.biobert = BioBERTEncoder().to(self.device).eval()
        self.biomedclip = BiomedCLIPEncoder().to(self.device).eval()
        self.radvlp = RadVLPEncoder().to(self.device).eval()
        self.fusion = build_fusion_module(self.cfg.fusion_type).to(self.device).eval()
        self.gard = GARDEnricher()

        elapsed = (time.perf_counter() - t0) * 1000
        logger.info(f"MedicalRiskEngine ready in {elapsed:.0f}ms.")
        self._initialized = True

    # ─────────────────────────────── Public API ───────────────────────────────

    def predict(
        self,
        image_b64: Optional[str] = None,
        image_pil: Optional[Image.Image] = None,
        symptoms: str = "",
        num_mc_samples: int = 8,
    ) -> AssessmentResult:
        """
        Run full multimodal inference.

        Parameters
        ----------
        image_b64    : base64-encoded image string (priority if both provided)
        image_pil    : PIL Image (used when image_b64 is None)
        symptoms     : comma-separated symptom string
        num_mc_samples: Monte Carlo samples for uncertainty estimation

        Returns
        -------
        AssessmentResult
        """
        t_start = time.perf_counter()

        # ── 1. Preprocessing ──────────────────────────────────────────────
        if image_b64:
            img_tensor, pil_image = self.img_preprocessor.from_base64(image_b64)
        elif image_pil:
            img_tensor = self.img_preprocessor.from_pil(image_pil)
            pil_image = image_pil
        else:
            raise ValueError("Provide either image_b64 or image_pil.")

        img_tensor = img_tensor.to(self.device)
        text_encoding = {k: v.to(self.device) for k, v in self.txt_preprocessor(symptoms).items()}
        token_list = self.txt_preprocessor.get_tokens(symptoms)

        # ── 2. Feature extraction ─────────────────────────────────────────
        with torch.no_grad():
            clip_proj, clip_patches = self.biomedclip.encode(img_tensor)
            rad_proj, rad_patches = self.radvlp.encode(img_tensor)
            bio_proj, token_emb, attentions = self.biobert.encode(text_encoding)

        # ── 3. Fusion + MC Dropout inference ─────────────────────────────
        mean_logits, mean_risk, risk_std = self.fusion.predict(
            clip_proj, rad_proj, bio_proj,
            num_mc_samples=num_mc_samples,
        )

        # ── 4. Disease probabilities ──────────────────────────────────────
        disease_probs_tensor = F.softmax(mean_logits[0], dim=-1).cpu().numpy()
        disease_probs = {label: round(float(p), 4) for label, p in zip(self.labels, disease_probs_tensor)}
        primary_disease = max(disease_probs, key=disease_probs.get)
        base_risk = float(mean_risk[0, 0].cpu())
        uncertainty = float(risk_std[0, 0].cpu())

        # ── 5. GARD enrichment ────────────────────────────────────────────
        gard_result = self.gard.enrich(symptoms, base_risk)
        final_risk = gard_result["enriched_risk"]
        risk_label = self.gard.get_risk_label(final_risk)

        # ── 6. Explainability ─────────────────────────────────────────────
        symptom_importance = compute_symptom_importance(attentions, token_list)
        patch_attn = compute_patch_attention(clip_patches or rad_patches)

        explanation = build_explanation(
            disease_label=primary_disease,
            risk_score=final_risk,
            risk_std=uncertainty,
            symptom_importance=symptom_importance,
            gard_result=gard_result,
            disease_probs=disease_probs,
        )

        elapsed_ms = (time.perf_counter() - t_start) * 1000

        return AssessmentResult(
            primary_disease=primary_disease,
            disease_probabilities=disease_probs,
            risk_score=final_risk,
            risk_uncertainty=uncertainty,
            risk_label=risk_label,
            explanation=explanation,
            top_symptoms=explanation.get("top_symptoms", []),
            gard_enrichment={
                "rare_candidates": gard_result.get("rare_candidates", []),
                "gard_premium": gard_result.get("gard_premium", 0.0),
                "symptom_risk_weights": gard_result.get("symptom_weights", {}),
            },
            inference_time_ms=elapsed_ms,
            model_versions={
                "biobert": self.cfg.biobert_model_name,
                "biomedclip": self.cfg.biomedclip_model_name,
                "radvlp": self.cfg.radvlp_model_name,
                "fusion": self.cfg.fusion_type,
            },
        )

    def predict_batch(self, requests: List[Dict]) -> List[AssessmentResult]:
        """Process a list of {image_b64, symptoms} dicts."""
        return [
            self.predict(
                image_b64=r.get("image_base64"),
                symptoms=r.get("symptoms", ""),
            )
            for r in requests
        ]
