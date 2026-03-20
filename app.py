import os
import time
import httpx
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# Enable CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration
HF_TOKEN = os.getenv("HF_TOKEN")
# Using the Hugging Face Router for better reliability
API_URL = "https://router.huggingface.co/google/medgemma-1.1-7b-it"
headers = {"Authorization": f"Bearer {HF_TOKEN}"}

class AssessmentRequest(BaseModel):
    # Fixed: Added Optional and default None to prevent Pydantic string errors
    image_base64: Optional[str] = None
    symptoms: str

async def call_huggingface(payload, retries=3, delay=15):
    """
    Handles the request to Hugging Face Router with a retry loop
    for 503 (Loading) and 504 (Gateway) errors.
    """
    async with httpx.AsyncClient(timeout=90.0) as client:
        for i in range(retries):
            try:
                response = await client.post(API_URL, headers=headers, json=payload)
                
                # Success
                if response.status_code == 200:
                    return response.json()
                
                # Model is still loading on the HF side
                if response.status_code in [503, 504]:
                    print(f"Model busy or loading... attempt {i+1} of {retries}")
                    time.sleep(delay)
                    continue
                
                # Other API errors
                raise HTTPException(status_code=response.status_code, detail=response.text)
            
            except httpx.ReadTimeout:
                print("Request timed out, retrying...")
                continue
        
        raise HTTPException(status_code=504, detail="The AI model is currently unavailable or taking too long to respond.")
    

@app.get("/")
async def root():
    return {"message": "Medical Assessment API is running. Use /assess for POST requests."}

@app.post("/assess")
async def risk_assessment(data: AssessmentRequest):
    # Validation
    if not data.symptoms or len(data.symptoms.strip()) < 3:
        raise HTTPException(status_code=400, detail="Please provide more detailed symptoms.")

    # Construct Prompt
    base_prompt = f"System: You are a medical diagnostic assistant. Analyze the symptoms and image (if provided) to assess health risks.\nSymptoms: {data.symptoms}"

    # Multimodal vs Text-only Logic
    if data.image_base64 and len(data.image_base64) > 100:
        # Clean the base64 string if it contains the data:image prefix
        clean_base64 = data.image_base64.split(",")[-1] if "," in data.image_base64 else data.image_base64
        
        payload = {
            "inputs": {
                "image": clean_base64,
                "text": f"{base_prompt}\nPlease identify visible abnormalities in the image related to these symptoms."
            }
        }
    else:
        # Fallback to Text-only
        payload = {
            "inputs": f"{base_prompt}\nProvide a risk assessment based solely on these symptoms.",
            "parameters": {"max_new_tokens": 500}
        }

    return await call_huggingface(payload)

@app.get("/health")
def health_check():
    return {"status": "online", "endpoint": "Hugging Face Router"}