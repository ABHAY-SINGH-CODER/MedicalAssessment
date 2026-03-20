import os
import base64
import requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional

# ======================
# CONFIG
# ======================
HF_TOKEN = os.getenv("HF_TOKEN")

HEADERS = {
    "Authorization": f"Bearer {HF_TOKEN}"
}

# Models (ALL FREE HF API)
NER_MODEL = "d4data/biomedical-ner-all"
IMAGE_MODEL = "Salesforce/blip-image-captioning-base"
LLM_MODEL = "mistralai/Mistral-7B-Instruct-v0.2"

app = FastAPI(title="Multimodal Medical AI")

# ======================
# INPUT MODEL
# ======================
class InputData(BaseModel):
    symptoms: Optional[str] = None
    image_base64: Optional[str] = None


# ======================
# HF API CALL
# ======================
def query_hf(model, payload, is_image=False):
    url = f"https://api-inference.huggingface.co/models/{model}"

    try:
        if is_image:
            res = requests.post(url, headers=HEADERS, data=payload, timeout=60)
        else:
            res = requests.post(url, headers=HEADERS, json=payload, timeout=60)

        if res.status_code != 200:
            return {"error": res.text}

        return res.json()

    except Exception as e:
        return {"error": str(e)}


# ======================
# TEXT → SYMPTOMS (NER)
# ======================
def extract_symptoms(text):
    result = query_hf(NER_MODEL, {"inputs": text})

    if isinstance(result, dict) and "error" in result:
        return []

    symptoms = list(set([ent['word'] for ent in result if ent['entity_group'] == 'Disease']))

    return symptoms


# ======================
# IMAGE → DESCRIPTION
# ======================
def describe_image(base64_str):
    try:
        image_bytes = base64.b64decode(base64_str)
        result = query_hf(IMAGE_MODEL, image_bytes, is_image=True)

        if isinstance(result, list) and len(result) > 0:
            return result[0].get("generated_text", "")

        return ""

    except Exception:
        return ""


# ======================
# LLM REASONING
# ======================
def medical_reasoning(symptoms, image_desc):
    prompt = f"""
You are a medical assistant AI (non-diagnostic).

Input:
Symptoms: {symptoms}
Image Description: {image_desc}

Tasks:
1. Predict possible diseases
2. Assign risk level (Low/Medium/High)
3. Suggest ONLY general remedies (no medicines)

Return JSON:
{{
  "diseases": [],
  "risk": "",
  "remedies": []
}}
"""

    result = query_hf(LLM_MODEL, {"inputs": prompt})

    if isinstance(result, dict) and "error" in result:
        return {"error": result["error"]}

    try:
        return result[0]["generated_text"]
    except:
        return {"raw": result}


# ======================
# MAIN ENDPOINT
# ======================
@app.post("/analyze")
def analyze(data: InputData):

    if not data.symptoms and not data.image_base64:
        raise HTTPException(status_code=400, detail="Provide text or image or both")

    symptoms_extracted = []
    image_description = ""

    if data.symptoms:
        symptoms_extracted = extract_symptoms(data.symptoms)

    if data.image_base64:
        image_description = describe_image(data.image_base64)

    reasoning = medical_reasoning(symptoms_extracted, image_description)

    return {
        "status": "success",
        "extracted_symptoms": symptoms_extracted,
        "image_description": image_description,
        "final_analysis": reasoning
    }