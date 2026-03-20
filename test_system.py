"""
tests/test_system.py
Comprehensive test suite for the Medical Risk Assessment System.

Run: pytest tests/test_system.py -v
"""

import base64
import io
import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
from PIL import Image

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from configs.config import APP_CONFIG
from src.utils.helpers import (
    clean_symptom_text,
    decode_base64_image,
    encode_image_to_base64,
    extract_symptom_tokens,
    resize_and_pad,
    sigmoid,
    validate_image,
)


# ─────────────────────────── Fixtures ────────────────────────────────────────

def _make_dummy_image(w: int = 256, h: int = 256) -> Image.Image:
    arr = np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)
    return Image.fromarray(arr)


def _make_dummy_b64(w: int = 256, h: int = 256) -> str:
    img = _make_dummy_image(w, h)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


@pytest.fixture(scope="session")
def dummy_image():
    return _make_dummy_image()


@pytest.fixture(scope="session")
def dummy_b64():
    return _make_dummy_b64()


# ─────────────────────────── Utility tests ───────────────────────────────────

class TestHelpers:
    def test_decode_base64_image(self, dummy_b64):
        img = decode_base64_image(dummy_b64)
        assert isinstance(img, Image.Image)
        assert img.mode == "RGB"

    def test_decode_base64_raw(self, dummy_b64):
        raw = dummy_b64.split(",", 1)[1]
        img = decode_base64_image(raw)
        assert isinstance(img, Image.Image)

    def test_encode_decode_roundtrip(self, dummy_image):
        b64 = encode_image_to_base64(dummy_image)
        recovered = decode_base64_image(b64)
        assert recovered.size == dummy_image.size

    def test_resize_and_pad_square(self, dummy_image):
        padded = resize_and_pad(dummy_image, 224)
        assert padded.size == (224, 224)

    def test_resize_and_pad_non_square(self):
        img = _make_dummy_image(640, 480)
        padded = resize_and_pad(img, 224)
        assert padded.size == (224, 224)

    def test_validate_image_ok(self, dummy_image):
        valid, msg = validate_image(dummy_image)
        assert valid, msg

    def test_validate_image_too_small(self):
        tiny = _make_dummy_image(10, 10)
        valid, _ = validate_image(tiny)
        assert not valid

    def test_sigmoid_range(self):
        assert 0 < sigmoid(0) < 1
        assert sigmoid(100) > 0.99
        assert sigmoid(-100) < 0.01

    def test_clean_symptom_text_basic(self):
        raw = "FEVER, COUGH, Chest Pain!!!"
        cleaned = clean_symptom_text(raw)
        assert "fever" in cleaned
        assert "cough" in cleaned
        assert "chest pain" in cleaned

    def test_clean_symptom_text_abbreviations(self):
        cleaned = clean_symptom_text("SOB and CP")
        assert "shortness of breath" in cleaned
        assert "chest pain" in cleaned

    def test_clean_symptom_text_empty(self):
        assert clean_symptom_text("") == ""
        assert clean_symptom_text("   ") == ""

    def test_extract_symptom_tokens(self):
        tokens = extract_symptom_tokens("fever, cough; chest pain and dyspnea")
        assert len(tokens) >= 3
        assert any("fever" in t for t in tokens)


# ─────────────────────────── Preprocessing tests ─────────────────────────────

class TestPreprocessing:
    def test_image_preprocessor_from_pil(self, dummy_image):
        from src.utils.preprocessing import ImagePreprocessor
        prep = ImagePreprocessor()
        tensor = prep.from_pil(dummy_image)
        assert tensor.shape == (1, 3, 224, 224)
        assert tensor.dtype == torch.float32

    def test_image_preprocessor_from_b64(self, dummy_b64):
        from src.utils.preprocessing import ImagePreprocessor
        prep = ImagePreprocessor()
        tensor, pil = prep.from_base64(dummy_b64)
        assert tensor.shape == (1, 3, 224, 224)
        assert isinstance(pil, Image.Image)

    def test_text_preprocessor_output_shape(self):
        from src.utils.preprocessing import TextPreprocessor
        prep = TextPreprocessor()
        enc = prep("fever, cough, chest pain")
        assert "input_ids" in enc
        assert enc["input_ids"].shape[0] == 1

    def test_text_preprocessor_empty(self):
        from src.utils.preprocessing import TextPreprocessor
        prep = TextPreprocessor()
        enc = prep("")
        assert "input_ids" in enc

    def test_text_preprocessor_tokens(self):
        from src.utils.preprocessing import TextPreprocessor
        prep = TextPreprocessor()
        tokens = prep.get_tokens("fever and cough")
        assert isinstance(tokens, list)


# ─────────────────────────── Model tests ─────────────────────────────────────

class TestBioBERTEncoder:
    def test_forward_shape(self):
        from src.models.biobert_encoder import BioBERTEncoder
        from src.utils.preprocessing import TextPreprocessor
        encoder = BioBERTEncoder().eval()
        prep = TextPreprocessor()
        enc = prep("fever, cough")
        proj, token_emb, attns = encoder.encode(enc)
        assert proj.shape == (1, APP_CONFIG.model.fusion_hidden_dim)
        assert token_emb.shape[0] == 1
        assert token_emb.shape[2] == APP_CONFIG.model.biobert_hidden_size

    def test_deterministic_eval(self):
        from src.models.biobert_encoder import BioBERTEncoder
        from src.utils.preprocessing import TextPreprocessor
        encoder = BioBERTEncoder().eval()
        prep = TextPreprocessor()
        enc = prep("shortness of breath")
        with torch.no_grad():
            p1, _, _ = encoder.encode(enc)
            p2, _, _ = encoder.encode(enc)
        assert torch.allclose(p1, p2, atol=1e-5)


class TestBiomedCLIPEncoder:
    def test_forward_shape(self, dummy_image):
        from src.models.biomedclip_encoder import BiomedCLIPEncoder
        from src.utils.preprocessing import ImagePreprocessor
        encoder = BiomedCLIPEncoder().eval()
        prep = ImagePreprocessor()
        tensor = prep.from_pil(dummy_image)
        proj, patches = encoder.encode(tensor)
        assert proj.shape == (1, APP_CONFIG.model.fusion_hidden_dim)


class TestRadVLPEncoder:
    def test_forward_shape(self, dummy_image):
        from src.models.radvlp_encoder import RadVLPEncoder
        from src.utils.preprocessing import ImagePreprocessor
        encoder = RadVLPEncoder().eval()
        prep = ImagePreprocessor()
        tensor = prep.from_pil(dummy_image)
        proj, patches = encoder.encode(tensor)
        assert proj.shape == (1, APP_CONFIG.model.fusion_hidden_dim)


# ─────────────────────────── Fusion tests ────────────────────────────────────

class TestFusion:
    def _make_embeds(self, B: int = 1):
        D = APP_CONFIG.model.fusion_hidden_dim
        return (
            torch.randn(B, D),
            torch.randn(B, D),
            torch.randn(B, D),
        )

    def test_attention_fusion_forward(self):
        from src.fusion.multimodal_fusion import CrossModalAttentionFusion
        model = CrossModalAttentionFusion().eval()
        clip, rad, text = self._make_embeds()
        with torch.no_grad():
            logits, risk, _ = model(clip, rad, text)
        assert logits.shape == (1, APP_CONFIG.model.num_disease_classes)
        assert risk.shape == (1, 1)
        assert 0.0 <= risk.item() <= 1.0

    def test_concat_fusion_forward(self):
        from src.fusion.multimodal_fusion import ConcatFusion
        model = ConcatFusion().eval()
        clip, rad, text = self._make_embeds()
        with torch.no_grad():
            logits, risk, _ = model(clip, rad, text)
        assert logits.shape == (1, APP_CONFIG.model.num_disease_classes)
        assert 0.0 <= risk.item() <= 1.0

    def test_mc_dropout_returns_uncertainty(self):
        from src.fusion.multimodal_fusion import CrossModalAttentionFusion
        model = CrossModalAttentionFusion()
        clip, rad, text = self._make_embeds()
        mean_logits, mean_risk, risk_std = model.predict(clip, rad, text, num_mc_samples=5)
        assert risk_std.shape == (1, 1)
        assert risk_std.item() >= 0.0

    def test_build_fusion_module(self):
        from src.fusion.multimodal_fusion import build_fusion_module
        attn = build_fusion_module("attention")
        cat = build_fusion_module("concat")
        assert attn is not None
        assert cat is not None

        with pytest.raises(ValueError):
            build_fusion_module("unknown")


# ─────────────────────────── GARD tests ──────────────────────────────────────

class TestGARDEnricher:
    def test_enrich_known_symptoms(self):
        from src.models.gard_enricher import GARDEnricher
        enricher = GARDEnricher()
        result = enricher.enrich("cough lung fibrosis breathlessness crackles", 0.4)
        assert "enriched_risk" in result
        assert 0.0 <= result["enriched_risk"] <= 1.0
        assert "rare_candidates" in result
        assert "gard_premium" in result

    def test_enrich_empty_symptoms(self):
        from src.models.gard_enricher import GARDEnricher
        enricher = GARDEnricher()
        result = enricher.enrich("", 0.5)
        assert 0.0 <= result["enriched_risk"] <= 1.0

    def test_risk_label(self):
        from src.models.gard_enricher import GARDEnricher
        enricher = GARDEnricher()
        assert enricher.get_risk_label(0.1) == "Low"
        assert enricher.get_risk_label(0.35) == "Moderate"
        assert enricher.get_risk_label(0.65) == "High"
        assert enricher.get_risk_label(0.9) == "Critical"

    def test_gard_premium_bounded(self):
        from src.models.gard_enricher import GARDEnricher
        enricher = GARDEnricher()
        for sym in ["hemoptysis renal vasculitis granuloma", "fever", ""]:
            result = enricher.enrich(sym, 0.3)
            assert 0.0 <= result["gard_premium"] <= 0.3


# ─────────────────────────── Explainability tests ────────────────────────────

class TestExplainability:
    def test_symptom_importance_no_attention(self):
        from src.explainability.explainer import compute_symptom_importance
        tokens = ["fever", "cough", "pain"]
        result = compute_symptom_importance(None, tokens)
        assert set(result.keys()) == set(tokens)
        assert all(v >= 0 for v in result.values())

    def test_patch_attention_none(self):
        from src.explainability.explainer import compute_patch_attention
        assert compute_patch_attention(None) is None

    def test_patch_attention_from_tensor(self):
        from src.explainability.explainer import compute_patch_attention
        patches = torch.randn(1, 196, 512)
        heatmap = compute_patch_attention(patches)
        assert heatmap is not None
        assert heatmap.shape == (14, 14)
        assert heatmap.min() >= 0.0
        assert heatmap.max() <= 1.0

    def test_build_explanation_structure(self):
        from src.explainability.explainer import build_explanation
        exp = build_explanation(
            disease_label="Pneumonia",
            risk_score=0.72,
            risk_std=0.05,
            symptom_importance={"fever": 0.8, "cough": 0.6},
            gard_result={"rare_candidates": [], "gard_premium": 0.1, "symptom_weights": {}},
            disease_probs={"Pneumonia": 0.6, "Consolidation": 0.2, "Edema": 0.1},
        )
        assert "summary" in exp
        assert "risk_label" in exp
        assert "risk_narrative" in exp
        assert "differential_diagnoses" in exp
        assert "top_symptoms" in exp


# ─────────────────────────── End-to-end integration test ─────────────────────

class TestEndToEnd:
    """Integration test: full pipeline with dummy data."""

    def test_full_pipeline(self, dummy_b64):
        from src.inference_engine import MedicalRiskEngine
        engine = MedicalRiskEngine()
        result = engine.predict(
            image_b64=dummy_b64,
            symptoms="fever, cough, chest pain",
        )
        assert result.primary_disease in APP_CONFIG.model.disease_labels
        assert 0.0 <= result.risk_score <= 1.0
        assert result.risk_label in ("Low", "Moderate", "High", "Critical")
        assert result.inference_time_ms > 0
        assert isinstance(result.gard_enrichment, dict)

    def test_full_pipeline_no_symptoms(self, dummy_b64):
        from src.inference_engine import MedicalRiskEngine
        engine = MedicalRiskEngine()
        result = engine.predict(image_b64=dummy_b64, symptoms="")
        assert result.risk_score >= 0.0

    def test_result_serialisation(self, dummy_b64):
        from src.inference_engine import MedicalRiskEngine
        engine = MedicalRiskEngine()
        result = engine.predict(image_b64=dummy_b64, symptoms="dyspnea")
        d = result.to_dict()
        json_str = json.dumps(d)        # must be JSON-serialisable
        parsed = json.loads(json_str)
        assert parsed["primary_disease"] == result.primary_disease

    def test_batch_predict(self, dummy_b64):
        from src.inference_engine import MedicalRiskEngine
        engine = MedicalRiskEngine()
        batch = [
            {"image_base64": dummy_b64, "symptoms": "cough"},
            {"image_base64": dummy_b64, "symptoms": "fever, chest pain"},
        ]
        results = engine.predict_batch(batch)
        assert len(results) == 2
        for r in results:
            assert 0.0 <= r.risk_score <= 1.0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
