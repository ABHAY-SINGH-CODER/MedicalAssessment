"""
client/medical_risk_client.py
Python client SDK for the Multimodal Medical Risk Assessment API.

Usage
-----
from client.medical_risk_client import MedicalRiskClient

client = MedicalRiskClient("https://your-space.hf.space")
result = client.assess_from_file("chest_xray.png", symptoms="fever, cough")
print(result.risk_label, result.risk_score)
"""

import base64
import io
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Union

import requests
from PIL import Image


# ─────────────────────── Response dataclasses ────────────────────────────────

@dataclass
class SymptomImportance:
    symptom: str
    importance: float


@dataclass
class DifferentialDiagnosis:
    disease: str
    probability: float


@dataclass
class RareCandidate:
    name: str
    id: str
    match_score: float


@dataclass
class AssessmentResponse:
    primary_disease: str
    disease_probabilities: Dict[str, float]
    risk_score: float
    risk_uncertainty: float
    risk_label: str
    summary: str
    risk_narrative: str
    confidence_label: str
    confidence_percentage: float
    top_symptoms: List[SymptomImportance]
    differential_diagnoses: List[DifferentialDiagnosis]
    gard_rare_candidates: List[RareCandidate]
    gard_premium: float
    inference_time_ms: float
    raw: Dict = field(default_factory=dict)

    @classmethod
    def from_dict(cls, d: Dict) -> "AssessmentResponse":
        exp = d.get("explanation", {})
        gard = d.get("gard_enrichment", {})
        meta = d.get("metadata", {})
        return cls(
            primary_disease=d.get("primary_disease", ""),
            disease_probabilities=d.get("disease_probabilities", {}),
            risk_score=d.get("risk_score", 0.0),
            risk_uncertainty=d.get("risk_uncertainty", 0.0),
            risk_label=d.get("risk_label", ""),
            summary=exp.get("summary", ""),
            risk_narrative=exp.get("risk_narrative", ""),
            confidence_label=exp.get("confidence_label", ""),
            confidence_percentage=exp.get("confidence_percentage", 0.0),
            top_symptoms=[
                SymptomImportance(**s) for s in exp.get("top_symptoms", [])
            ],
            differential_diagnoses=[
                DifferentialDiagnosis(**dd) for dd in exp.get("differential_diagnoses", [])
            ],
            gard_rare_candidates=[
                RareCandidate(
                    name=r["name"], id=r["id"], match_score=r["match_score"]
                )
                for r in gard.get("rare_candidates", [])
            ],
            gard_premium=gard.get("gard_premium", 0.0),
            inference_time_ms=meta.get("inference_time_ms", 0.0),
            raw=d,
        )

    def __repr__(self) -> str:
        return (
            f"AssessmentResponse(\n"
            f"  primary_disease={self.primary_disease!r},\n"
            f"  risk_score={self.risk_score:.4f},\n"
            f"  risk_label={self.risk_label!r},\n"
            f"  confidence={self.confidence_label} ({self.confidence_percentage:.1f}%),\n"
            f"  inference_time={self.inference_time_ms:.0f}ms\n"
            f")"
        )


# ─────────────────────── Client ──────────────────────────────────────────────

class MedicalRiskClient:
    """
    Thin Python wrapper around the Medical Risk Assessment REST API.

    Parameters
    ----------
    base_url  : e.g. "https://your-space.hf.space" or "http://localhost:8000"
    api_key   : optional bearer token (for gated deployments)
    timeout   : request timeout in seconds
    """

    def __init__(
        self,
        base_url: str = "http://localhost:8000",
        api_key: Optional[str] = None,
        timeout: int = 60,
    ):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update({"Content-Type": "application/json"})
        if api_key:
            self.session.headers["Authorization"] = f"Bearer {api_key}"

    # ── Helpers ───────────────────────────────────────────────────────────

    @staticmethod
    def _encode_image(source: Union[str, Path, Image.Image, bytes]) -> str:
        """Convert any image source to base64 string."""
        if isinstance(source, str) and source.startswith("data:"):
            return source                                           # already data-URI
        if isinstance(source, (str, Path)):
            with open(source, "rb") as f:
                raw = f.read()
        elif isinstance(source, Image.Image):
            buf = io.BytesIO()
            source.save(buf, format="PNG")
            raw = buf.getvalue()
        elif isinstance(source, bytes):
            raw = source
        else:
            raise TypeError(f"Unsupported image type: {type(source)}")
        return "data:image/png;base64," + base64.b64encode(raw).decode()

    def _post(self, endpoint: str, payload: Dict) -> Dict:
        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        resp = self.session.post(url, json=payload, timeout=self.timeout)
        resp.raise_for_status()
        return resp.json()

    # ── Public API ────────────────────────────────────────────────────────

    def health(self) -> Dict:
        """Check API health."""
        resp = self.session.get(f"{self.base_url}/api/v1/health", timeout=10)
        resp.raise_for_status()
        return resp.json()

    def model_info(self) -> Dict:
        """Get deployed model versions."""
        resp = self.session.get(f"{self.base_url}/api/v1/models", timeout=10)
        resp.raise_for_status()
        return resp.json()

    def assess(
        self,
        image: Union[str, Path, Image.Image, bytes],
        symptoms: str = "",
    ) -> AssessmentResponse:
        """
        Run a single medical risk assessment.

        Parameters
        ----------
        image    : file path, PIL Image, bytes, or base64/data-URI string
        symptoms : comma-separated symptom string

        Returns
        -------
        AssessmentResponse
        """
        b64 = self._encode_image(image)
        data = self._post(
            "api/v1/assess",
            {"image_base64": b64, "symptoms": symptoms},
        )
        return AssessmentResponse.from_dict(data)

    # Convenience aliases
    def assess_from_file(self, path: Union[str, Path], symptoms: str = "") -> AssessmentResponse:
        return self.assess(path, symptoms)

    def assess_from_pil(self, image: Image.Image, symptoms: str = "") -> AssessmentResponse:
        return self.assess(image, symptoms)

    def assess_batch(
        self,
        items: List[Dict],             # list of {"image": ..., "symptoms": ...}
    ) -> List[AssessmentResponse]:
        """
        Batch assessment (max 16 items).

        Parameters
        ----------
        items : list of dicts with keys "image" and optional "symptoms"

        Returns
        -------
        list of AssessmentResponse
        """
        requests_payload = [
            {
                "image_base64": self._encode_image(item["image"]),
                "symptoms": item.get("symptoms", ""),
            }
            for item in items
        ]
        data = self._post("api/v1/assess/batch", {"requests": requests_payload})
        return [AssessmentResponse.from_dict(r) for r in data.get("results", [])]


# ─────────────────────── CLI quick test ──────────────────────────────────────

if __name__ == "__main__":
    import sys

    url = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8000"
    client = MedicalRiskClient(url)

    print("Health:", client.health())
    print("Models:", client.model_info())

    # Create a dummy white image for quick smoke test
    dummy = Image.new("RGB", (224, 224), color=(200, 200, 200))
    result = client.assess_from_pil(dummy, symptoms="fever, cough, chest pain")
    print("\nResult:")
    print(result)
    print("\nTop symptoms:")
    for s in result.top_symptoms:
        print(f"  {s.symptom}: {s.importance:.3f}")
    print("\nDifferential:")
    for d in result.differential_diagnoses:
        print(f"  {d.disease}: {d.probability:.1f}%")
