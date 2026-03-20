"""
src/models/gard_enricher.py
GARD (Genetic and Rare Disease) Risk Enrichment Layer.

Uses the NIH GARD disease database to:
  1. Match symptoms → rare disease candidates
  2. Compute a rare-disease risk premium
  3. Enrich the final risk score with rare-disease context
  4. Return matched rare-disease names for explainability
"""

import json
import logging
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from configs.config import APP_CONFIG
from src.utils.helpers import clean_symptom_text, clamp

logger = logging.getLogger(__name__)


class GARDEnricher:
    """
    Lightweight, dictionary-based GARD enrichment (no neural model required).
    
    For production, this can be extended to query the NIH GARD REST API:
      https://rarediseases.info.nih.gov/api/diseases
    """

    def __init__(self, gard_file: Optional[str] = None):
        gard_file = gard_file or APP_CONFIG.model.gard_disease_file
        self._load_database(gard_file)

    def _load_database(self, path: str) -> None:
        try:
            gard_path = Path(path)
            with gard_path.open() as f:
                data = json.load(f)
            self.rare_diseases = data.get("rare_diseases", [])
            self.symptom_risk_map = data.get("symptom_risk_map", {})
            logger.info(f"GARD: loaded {len(self.rare_diseases)} rare diseases.")
        except Exception as e:
            logger.warning(f"GARD database not found ({e}). Using empty database.")
            self.rare_diseases = []
            self.symptom_risk_map = {}

    def enrich(
        self,
        symptom_text: str,
        base_risk: float,
        top_k: int = 3,
    ) -> Dict:
        """
        Parameters
        ----------
        symptom_text : raw symptom string
        base_risk    : float in [0, 1] from the fusion prediction head
        top_k        : max rare disease candidates to return

        Returns
        -------
        dict with:
          enriched_risk     : float – adjusted risk score
          rare_candidates   : list of {name, id, score}
          symptom_weights   : dict mapping symptom → risk weight
          gard_premium      : additive risk contribution from GARD
        """
        cleaned = clean_symptom_text(symptom_text)
        symptom_tokens = [s.strip() for s in re.split(r"[,;]|\band\b", cleaned) if s.strip()]

        # ── Symptom-level risk weights ────────────────────────────────────────
        symptom_weights = {}
        for token in symptom_tokens:
            for key, weight in self.symptom_risk_map.items():
                if key in token or token in key:
                    symptom_weights[token] = max(symptom_weights.get(token, 0), weight)
                    break

        # ── Rare disease matching ─────────────────────────────────────────────
        candidates = []
        for disease in self.rare_diseases:
            score = 0.0
            matched_kw = 0
            for kw in disease["keywords"]:
                if kw in cleaned:
                    score += disease["risk_weight"]
                    matched_kw += 1
            if matched_kw > 0:
                score = score * (matched_kw / len(disease["keywords"]))
                candidates.append({
                    "name": disease["name"],
                    "id": disease["id"],
                    "match_score": round(score, 4),
                    "matched_keywords": matched_kw,
                })

        candidates.sort(key=lambda x: x["match_score"], reverse=True)
        top_candidates = candidates[:top_k]

        # ── Risk premium ──────────────────────────────────────────────────────
        # Weighted average of symptom risks + rare disease top-1 boost
        sym_risk = (
            np.mean(list(symptom_weights.values())) if symptom_weights else 0.0
        )
        rare_risk = top_candidates[0]["match_score"] if top_candidates else 0.0
        gard_premium = clamp(0.4 * sym_risk + 0.6 * rare_risk, 0.0, 0.3)

        # Blend: keep 70% model prediction, 30% GARD context
        enriched_risk = clamp(0.7 * base_risk + 0.3 * gard_premium + gard_premium * 0.1)

        return {
            "enriched_risk": round(enriched_risk, 4),
            "gard_premium": round(float(gard_premium), 4),
            "rare_candidates": top_candidates,
            "symptom_weights": symptom_weights,
        }

    def get_risk_label(self, risk_score: float) -> str:
        if risk_score < 0.25:
            return "Low"
        elif risk_score < 0.50:
            return "Moderate"
        elif risk_score < 0.75:
            return "High"
        else:
            return "Critical"
