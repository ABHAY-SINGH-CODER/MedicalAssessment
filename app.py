import os
import time
import json
import base64
import requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, List, Dict

# ======================
# CONFIG & CLIENT
# ======================
HF_TOKEN = os.getenv("HF_TOKEN")
HEADERS = {"Authorization": f"Bearer {HF_TOKEN}"}

# Models
NER_MODEL = "d4data/biomedical-ner-all"
IMAGE_MODEL = "Salesforce/blip-image-captioning-base"
LLM_MODEL = "mistralai/Mistral-7B-Instruct-v0.2"

app = FastAPI(title="Refined Medical AI")

class InputData(BaseModel):
    symptoms: Optional[str] = None
    image_base64: Optional[str] = None

# ======================
# HELPER: ROBUST HF CALL
# ======================
def query_hf(model: str, payload: dict, is_binary: bool = False):
    url = f"https://api-inference.huggingface.co/models/{model}"
    
    # Retry logic for model "Loading" states (503 errors)
    for _ in range(3):
        try:
            if is_binary:
                response = requests.post(url, headers=HEADERS, data=payload, timeout=30)
            else:
                response = requests.post(url, headers=HEADERS, json=payload, timeout=30)
            
            result = response.json()
            
            # Handle model warming up
            if response.status_code == 503:
                time.sleep(5)
                continue
                
            return result
        except Exception as e:
            return {"error": str(e)}
    return {"error": "Model failed to load after retries"}

# ======================
# LOGIC LAYERS
# ======================

def extract_entities(text: str) -> List[str]:
    """Extracts diseases/symptoms and cleans up the word fragments."""
    results = query_hf(NER_MODEL, {"inputs": text, "parameters": {"aggregation_strategy": "simple"}})
    
    if not isinstance(results, list):
        return []
        
    # 'simple' aggregation merges ##fragments into whole words automatically
    entities = [ent['word'] for ent in results if ent['entity_group'] in ['Disease', 'Sign_symptom']]
    return list(set(entities))

def describe_image(base64_str: str) -> str:
    """Decodes image and gets caption."""
    try:
        image_bytes = base64.b64decode(base64_str)
        result = query_hf(IMAGE_MODEL, image_bytes, is_binary=True)
        return result[0].get("generated_text", "No visual features identified.") if isinstance(result, list) else ""
    except:
        return "Image processing failed."

def get_medical_reasoning(symptoms: List[str], image_desc: str) -> Dict:
    """Uses Mistral with a strict instruction format for JSON output."""
    symptoms_str = ", ".join(symptoms) if symptoms else "None reported"
    
    # Mistral uses [INST] tags for instruction following
    prompt = f"<s>[INST] You are a medical assistant. Based on these symptoms: {symptoms_str} and image description: {image_desc}, provide a risk assessment. Return ONLY a JSON object with keys: 'diseases' (list), 'risk' (Low/Medium/High), and 'remedies' (list of non-medicine advice). [/INST]</s>"

    result = query_hf(LLM_MODEL, {
        "inputs": prompt,
        "parameters": {"max_new_tokens": 200, "return_full_text": False}
    })

    try:
        # Mistral output is often a list of dicts
        raw_text = result[0]['generated_text'] if isinstance(result, list) else str(result)
        # Attempt to find the JSON part if the model added conversational filler
        start_idx = raw_text.find('{')
        end_idx = raw_text.rfind('}') + 1
        return json.loads(raw_text[start_idx:end_idx])
    except:
        return {"error": "Could not parse AI response", "raw": result}

# ======================
# ENDPOINT
# ======================
@app.post("/analyze")
async def analyze(data: InputData):
    if not data.symptoms and not data.image_base64:
        raise HTTPException(status_code=400, detail="Missing input data.")

    # 1. Process Text
    found_symptoms = extract_entities(data.symptoms) if data.symptoms else []
    
    # 2. Process Image
    img_caption = describe_image(data.image_base64) if data.image_base64 else ""

    # 3. Get LLM Reasoning
    analysis = get_medical_reasoning(found_symptoms, img_caption)

    return {
        "symptoms_identified": found_symptoms,
        "image_analysis": img_caption,
        "medical_insight": analysis
    }