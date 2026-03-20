"""
api/main.py
Production FastAPI application for the Multimodal Medical Risk Assessment System.

Endpoints:
  POST /api/v1/assess        – Single assessment (image + symptoms)
  POST /api/v1/assess/batch  – Batch assessments
  GET  /api/v1/health        – Health check
  GET  /api/v1/models        – Model version info
  GET  /docs                 – Swagger UI (auto-generated)
"""

import logging
import time
from contextlib import asynccontextmanager
from typing import Dict, List, Optional

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator

from configs.config import APP_CONFIG
from src.inference_engine import MedicalRiskEngine

logger = logging.getLogger(__name__)

# ─────────────────────────── Pydantic Models ─────────────────────────────────

class AssessmentRequest(BaseModel):
    image_base64: str = Field(
        ...,
        description="Base64-encoded medical image (JPEG/PNG). May include data-URI prefix.",
        example="data:image/png;base64,iVBORw0KGgo...",
    )
    symptoms: Optional[str] = Field(
        default="",
        description="Comma-separated symptom list, e.g. 'fever, cough, chest pain'.",
        example="fever, cough, chest pain",
        max_length=2048,
    )

    @validator("image_base64")
    def validate_b64(cls, v):
        if not v or len(v.strip()) < 20:
            raise ValueError("image_base64 appears to be empty or invalid.")
        return v.strip()


class BatchAssessmentRequest(BaseModel):
    requests: List[AssessmentRequest] = Field(..., max_items=16)


class HealthResponse(BaseModel):
    status: str
    engine_ready: bool
    timestamp: float
    version: str = "1.0.0"


# ─────────────────────────── Lifespan (startup / shutdown) ───────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Pre-load the inference engine at startup."""
    logger.info("Starting up: loading MedicalRiskEngine…")
    app.state.engine = MedicalRiskEngine()
    logger.info("Engine ready.")
    yield
    logger.info("Shutting down.")


# ─────────────────────────── App factory ─────────────────────────────────────

def create_app() -> FastAPI:
    cfg = APP_CONFIG.api
    app = FastAPI(
        title=cfg.title,
        description=cfg.description,
        version="1.0.0",
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
    )

    # CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=cfg.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ── Request timing middleware ─────────────────────────────────────────
    @app.middleware("http")
    async def add_process_time_header(request: Request, call_next):
        t0 = time.perf_counter()
        response = await call_next(request)
        response.headers["X-Process-Time-Ms"] = str(round((time.perf_counter() - t0) * 1000, 1))
        return response

    # ── Exception handler ─────────────────────────────────────────────────
    @app.exception_handler(Exception)
    async def generic_exception_handler(request: Request, exc: Exception):
        logger.error(f"Unhandled exception: {exc}", exc_info=True)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"error": "Internal server error", "detail": str(exc)},
        )

    # ──────────────────────── Routes ──────────────────────────────────────

    @app.get("/api/v1/health", response_model=HealthResponse, tags=["System"])
    async def health_check(request: Request):
        """Returns API health status and engine readiness."""
        engine_ready = hasattr(request.app.state, "engine") and request.app.state.engine._initialized
        return HealthResponse(
            status="ok" if engine_ready else "degraded",
            engine_ready=engine_ready,
            timestamp=time.time(),
        )

    @app.get("/api/v1/models", tags=["System"])
    async def model_info(request: Request):
        """Returns the model versions used in the current deployment."""
        cfg = APP_CONFIG.model
        return {
            "models": {
                "text_encoder": {"name": "BioBERT", "checkpoint": cfg.biobert_model_name},
                "image_encoder_clip": {"name": "BiomedCLIP", "checkpoint": cfg.biomedclip_model_name},
                "image_encoder_rad": {"name": "Rad-VLP / RAD-DINO", "checkpoint": cfg.radvlp_model_name},
                "risk_enrichment": {"name": "GARD Enricher", "type": "lexical+neural"},
                "fusion": {"type": cfg.fusion_type, "hidden_dim": cfg.fusion_hidden_dim},
            },
            "disease_classes": cfg.disease_labels,
        }

    @app.post(
        "/api/v1/assess",
        tags=["Assessment"],
        summary="Single Multimodal Medical Assessment",
        response_description="Disease prediction, risk score, and full explanation.",
    )
    async def assess(payload: AssessmentRequest, request: Request):
        """
        Run a full multimodal medical risk assessment.

        - **image_base64**: Base64-encoded chest X-ray or medical image
        - **symptoms**: Comma-separated clinical symptoms

        Returns:
        - `primary_disease`: Top predicted disease class
        - `disease_probabilities`: Full probability distribution over 14 classes
        - `risk_score`: Calibrated risk score [0–1]
        - `risk_uncertainty`: Monte Carlo uncertainty estimate
        - `risk_label`: Low / Moderate / High / Critical
        - `explanation`: Interpretable breakdown (top symptoms, narrative, differential)
        - `gard_enrichment`: Rare disease context from GARD database
        """
        engine: MedicalRiskEngine = request.app.state.engine
        try:
            result = engine.predict(
                image_b64=payload.image_base64,
                symptoms=payload.symptoms or "",
            )
            return result.to_dict()
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        except Exception as e:
            logger.error(f"Assessment failed: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail="Assessment failed. See logs.")

    @app.post(
        "/api/v1/assess/batch",
        tags=["Assessment"],
        summary="Batch Multimodal Assessment (max 16 items)",
    )
    async def assess_batch(payload: BatchAssessmentRequest, request: Request):
        """
        Process multiple assessment requests in one call (max 16).
        Each item follows the same schema as the single `/assess` endpoint.
        """
        engine: MedicalRiskEngine = request.app.state.engine
        try:
            results = engine.predict_batch([r.dict() for r in payload.requests])
            return {"results": [r.to_dict() for r in results], "count": len(results)}
        except Exception as e:
            logger.error(f"Batch assessment failed: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail="Batch assessment failed.")

    @app.get("/", include_in_schema=False)
    async def root():
        return {
            "message": "Multimodal Medical Risk Assessment API",
            "docs": "/docs",
            "health": "/api/v1/health",
        }

    return app


app = create_app()


# ─────────────────────────── Entry-point ─────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "api.main:app",
        host=APP_CONFIG.api.host,
        port=APP_CONFIG.api.port,
        reload=False,
        log_level=APP_CONFIG.log_level.lower(),
    )
