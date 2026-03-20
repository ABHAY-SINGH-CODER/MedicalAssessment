import os
import io
import httpx
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="Medical AI API")

HF_TOKEN = os.getenv("HF_TOKEN")
if not HF_TOKEN:
    raise ValueError("HF_TOKEN not set in .env file")

HEADERS = {"Authorization": f"Bearer {HF_TOKEN}"}

# Model Endpoints
BIOBERT_API = "https://api-inference.huggingface.co/models/dmis-lab/biobert-base-cased-v1.1"
BIOMEDCLIP_API = "https://api-inference.huggingface.co/models/microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"

async def query_hf_api(url, data=None, is_image=False):
    async with httpx.AsyncClient() as client:
        try:
            if is_image:
                # Images are sent as raw binary
                response = await client.post(url, headers=HEADERS, content=data, timeout=30)
            else:
                # Text is sent as JSON
                response = await client.post(url, headers=HEADERS, json={"inputs": data}, timeout=20)
            
            if response.status_code != 200:
                return None
            return response.json()
        except Exception as e:
            print(f"API Error: {e}")
            return None

@app.post("/assess")
async def assess(
    symptoms: str = Form(None),
    image: UploadFile = File(None)
):
    if not symptoms and not image:
        raise HTTPException(status_code=400, detail="Provide symptoms or an image.")

    results = {"text_res": None, "img_res": None}
    
    # Process Text
    if symptoms:
        results["text_res"] = await query_hf_api(BIOBERT_API, data=symptoms)

    # Process Image
    if image:
        img_bytes = await image.read()
        results["img_res"] = await query_hf_api(BIOMEDCLIP_API, data=img_bytes, is_image=True)

    # Simplified Fusion Logic
    # Note: BioBERT returns NER tags or embeddings; for a real 'diagnosis' 
    # you'd typically use a classification head model.
    prediction = "Inconclusive"
    confidence = 0.0

    if results["text_res"]:
        prediction = "Text Analysis Complete"
        confidence = 0.7  # Placeholder logic
    
    if results["img_res"]:
        prediction = "Visual Analysis Complete"
        confidence = 0.8

    return {
        "status": "success",
        "analysis": prediction,
        "confidence": confidence,
        "raw_responses": results
    }

@app.get("/")
def root():
    return {"message": "Medical AI API is active"}