---
title: Multimodal Medical Risk Assessment
emoji: 🏥
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: "4.7.1"
app_file: app.py
pinned: false
license: apache-2.0
tags:
  - medical
  - radiology
  - multimodal
  - biobert
  - biomedclip
  - rad-vlp
  - gard
  - chest-xray
  - risk-assessment
---

# 🏥 Multimodal Medical Risk Assessment System

A production-ready AI system for medical risk assessment using state-of-the-art biomedical models.

## Architecture

```
Input (Image + Symptoms)
        │
        ├── Image Branch
        │     ├── BiomedCLIP (microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224)
        │     └── Rad-VLP    (microsoft/rad-dino — radiology-specific ViT)
        │
        └── Text Branch
              └── BioBERT    (dmis-lab/biobert-base-cased-v1.2)
                         │
              Cross-Modal Attention Fusion
                         │
              GARD Risk Enrichment Layer
                         │
              ┌──────────┴──────────┐
         Disease Label         Risk Score [0–1]
                         │
              Explainability Module
              (symptom importance, differential, narrative)
```

## Models Used

| Model | Purpose | Source |
|-------|---------|--------|
| BioBERT | Symptom text understanding | `dmis-lab/biobert-base-cased-v1.2` |
| BiomedCLIP | Medical vision-language embeddings | `microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224` |
| Rad-VLP / RAD-DINO | Radiology image encoder | `microsoft/rad-dino` |
| GARD Enricher | Rare disease risk context | NIH GARD database |

## API Usage

The system exposes a REST API alongside the Gradio UI:

### Single Assessment

```bash
curl -X POST "https://your-space.hf.space/api/v1/assess" \
  -H "Content-Type: application/json" \
  -d '{
    "image_base64": "data:image/png;base64,...",
    "symptoms": "fever, cough, chest pain, shortness of breath"
  }'
```

### Response Schema

```json
{
  "primary_disease": "Pneumonia",
  "disease_probabilities": {
    "Pneumonia": 0.6241,
    "Consolidation": 0.1832,
    "Edema": 0.0891,
    "..."
  },
  "risk_score": 0.7823,
  "risk_uncertainty": 0.0412,
  "risk_label": "High",
  "explanation": {
    "summary": "Primary assessment: Pneumonia with High risk (78.2%)...",
    "risk_label": "High",
    "risk_color": "orange",
    "risk_percentage": 78.2,
    "confidence_label": "Moderate",
    "confidence_percentage": 83.5,
    "top_symptoms": [
      {"symptom": "chest pain", "importance": 0.842},
      {"symptom": "shortness of breath", "importance": 0.756}
    ],
    "risk_narrative": "...",
    "differential_diagnoses": [
      {"disease": "Pneumonia", "probability": 62.4},
      {"disease": "Consolidation", "probability": 18.3}
    ],
    "gard_rare_candidates": []
  },
  "gard_enrichment": {
    "rare_candidates": [],
    "gard_premium": 0.0312,
    "symptom_risk_weights": {
      "chest pain": 0.7,
      "shortness of breath": 0.75
    }
  },
  "metadata": {
    "inference_time_ms": 284.3,
    "model_versions": {
      "biobert": "dmis-lab/biobert-base-cased-v1.2",
      "biomedclip": "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224",
      "radvlp": "microsoft/rad-dino",
      "fusion": "attention"
    }
  }
}
```

### Batch Assessment

```bash
curl -X POST "https://your-space.hf.space/api/v1/assess/batch" \
  -H "Content-Type: application/json" \
  -d '{
    "requests": [
      {"image_base64": "...", "symptoms": "fever, cough"},
      {"image_base64": "...", "symptoms": "chest pain, edema"}
    ]
  }'
```

### Health Check

```bash
curl "https://your-space.hf.space/api/v1/health"
```

## Quick Start (Local)

```bash
# Clone
git clone https://huggingface.co/spaces/your-username/medical-risk-assessment
cd medical-risk-assessment

# Install
pip install -r requirements.txt

# Run Gradio UI
python app.py

# Run FastAPI (separate terminal)
python -m uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload

# Run tests
pytest tests/ -v
```

## Disease Classes (CheXpert 14)

```
No Finding · Enlarged Cardiomediastinum · Cardiomegaly · Lung Opacity
Lung Lesion · Edema · Consolidation · Pneumonia · Atelectasis
Pneumothorax · Pleural Effusion · Pleural Other · Fracture · Support Devices
```

## Disclaimer

> This system is intended for **research and educational purposes only**.  
> It is **not** a substitute for professional medical diagnosis, advice, or treatment.  
> Always consult a qualified healthcare provider for medical decisions.

## License

Apache 2.0 — See [LICENSE](LICENSE) for details.
