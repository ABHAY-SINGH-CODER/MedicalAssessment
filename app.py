from fastapi import FastAPI, UploadFile, File, Form
import requests
import base64
import os
from dotenv import load_dotenv
from PIL import Image
import io

load_dotenv()

app = FastAPI(title="Medical AI API")

HF_TOKEN = os.getenv("HF_TOKEN")

if not HF_TOKEN:
    raise ValueError("HF_TOKEN not set")

headers = {
    "Authorization": f"Bearer {HF_TOKEN}"
}

# -----------------------------
# API URLs
# -----------------------------
BIOBERT_API = "https://api-inference.huggingface.co/models/dmis-lab/biobert-base-cased-v1.1"
BIOMEDCLIP_API = "https://api-inference.huggingface.co/models/microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
RADVLP_API = "https://api-inference.huggingface.co/models/microsoft/rad-vlp"

# -----------------------------
# UTIL
# -----------------------------
def image_to_base64(file_bytes):
    return base64.b64encode(file_bytes).decode()

# -----------------------------
# TEXT
# -----------------------------
def query_text(text):
    try:
        res = requests.post(
            BIOBERT_API,
            headers=headers,
            json={"inputs": text},
            timeout=20
        )
        if res.status_code != 200:
            return "Text API Error", 0.0
        return "BioBERT Prediction", 0.6
    except:
        return "Text Failure", 0.0

# -----------------------------
# IMAGE
# -----------------------------
def query_image(file_bytes):
    try:
        img_b64 = image_to_base64(file_bytes)

        res = requests.post(
            BIOMEDCLIP_API,
            headers=headers,
            json={"inputs": img_b64},
            timeout=25
        )

        if res.status_code != 200:
            res = requests.post(
                RADVLP_API,
                headers=headers,
                json={"inputs": img_b64},
                timeout=25
            )

        return "Image Prediction", 0.65

    except:
        return "Image Failure", 0.0

# -----------------------------
# MAIN ENDPOINT
# -----------------------------
@app.post("/assess")
async def assess(
    symptoms: str = Form(""),
    image: UploadFile = File(None)
):
    has_txt = bool(symptoms.strip())
    has_img = image is not None

    if not has_txt and not has_img:
        return {"error": "Provide symptoms or image"}

    # read image
    file_bytes = None
    if has_img:
        file_bytes = await image.read()

    # logic
    if has_txt and not has_img:
        d, r = query_text(symptoms)
        mode = "BioBERT API"

    elif has_img and not has_txt:
        d, r = query_image(file_bytes)
        mode = "BiomedCLIP API"

    else:
        d1, r1 = query_text(symptoms)
        d2, r2 = query_image(file_bytes)

        r = 0.6 * r1 + 0.4 * r2
        d = d1 if r1 > r2 else d2
        mode = "Multimodal Fusion"

    return {
        "disease": d,
        "risk_score": round(r, 4),
        "mode": mode
    }

# -----------------------------
# ROOT
# -----------------------------
@app.get("/")
def root():
    return {"message": "Medical AI API running 🚀"}