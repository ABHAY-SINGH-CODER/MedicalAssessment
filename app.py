import os
import time
import httpx  # Better for async than 'requests'
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

HF_TOKEN = os.getenv("HF_TOKEN")
API_URL = "https://api-inference.huggingface.co/models/google/medgemma-1.1-7b-it"
headers = {"Authorization": f"Bearer {HF_TOKEN}"}

class AssessmentRequest(BaseModel):
    image_base64: str = None
    symptoms: str

async def call_huggingface(payload, retries=3, delay=20):
    """
    Logic to handle 503 (Loading) errors by waiting and retrying.
    """
    async with httpx.AsyncClient(timeout=60.0) as client:
        for i in range(retries):
            response = await client.post(API_URL, headers=headers, json=payload)
            
            if response.status_code == 200:
                return response.json()
            
            # If model is loading, wait and try again
            if response.status_code == 503:
                print(f"Model loading... retry {i+1}/{retries}")
                time.sleep(delay)
                continue
            
            # If any other error occurs, raise it
            raise HTTPException(status_code=response.status_code, detail=response.text)
        
        raise HTTPException(status_code=504, detail="Model took too long to load. Please try again in a minute.")

@app.post("/assess")
async def risk_assessment(data: AssessmentRequest):
    # 1. Input Validation: Basic check for empty symptoms
    if not data.symptoms.strip():
        raise HTTPException(status_code=400, detail="Symptoms text is required.")

    # 2. Build Payload
    prompt = f"System: Medical Assistant. Analyze risks.\nSymptoms: {data.symptoms}"
    
    if data.image_base64 and len(data.image_base64) > 100:
        # Multimodal request
        payload = {
            "inputs": {
                "image": data.image_base64,
                "text": f"{prompt}\nAnalyze the image provided."
            }
        }
    else:
        # Text-only request
        payload = {
            "inputs": prompt,
            "parameters": {"max_new_tokens": 500, "return_full_text": False}
        }

    # 3. Call HF with retry logic
    return await call_huggingface(payload)

@app.get("/health")
def health():
    return {"status": "ok"}