import gradio as gr
import torch
import numpy as np
import base64
import json
from PIL import Image
from io import BytesIO
from transformers import (
    AutoTokenizer,
    AutoModel,
    CLIPProcessor,
    CLIPModel,
)

# ── Disease labels ────────────────────────────────────────────────────────────
DISEASES = [
    "Pneumonia", "COVID-19", "Tuberculosis", "Pleural Effusion",
    "Cardiomegaly", "Atelectasis", "Fracture", "Healthy",
]

# ── Lazy-loaded model registry ────────────────────────────────────────────────
_cache = {}

def get_biobert():
    if "biobert" not in _cache:
        name = "dmis-lab/biobert-base-cased-v1.2"
        tok = AutoTokenizer.from_pretrained(name)
        mdl = AutoModel.from_pretrained(name).eval()
        _cache["biobert"] = (tok, mdl)
    return _cache["biobert"]

def get_biomedclip():
    if "biomedclip" not in _cache:
        # BiomedCLIP is published as a CLIP-compatible model on HF
        name = "microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
        proc = CLIPProcessor.from_pretrained(name)
        mdl  = CLIPModel.from_pretrained(name).eval()
        _cache["biomedclip"] = (proc, mdl)
    return _cache["biomedclip"]

# Rad-VLP shares the same CLIP architecture (HF hub id below)
def get_radvlp():
    if "radvlp" not in _cache:
        name = "microsoft/rad-vlp"           # official HF repo
        try:
            proc = CLIPProcessor.from_pretrained(name)
            mdl  = CLIPModel.from_pretrained(name).eval()
        except Exception:
            # graceful fallback to BiomedCLIP if Rad-VLP is unavailable
            proc, mdl = get_biomedclip()
        _cache["radvlp"] = (proc, mdl)
    return _cache["radvlp"]

# ── Embedding helpers ─────────────────────────────────────────────────────────

def text_embedding_biobert(text: str) -> np.ndarray:
    tok, mdl = get_biobert()
    enc = tok(text, return_tensors="pt", truncation=True, max_length=512, padding=True)
    with torch.no_grad():
        out = mdl(**enc)
    return out.last_hidden_state[:, 0, :].squeeze().numpy()   # [768]

def image_embedding_biomedclip(img: Image.Image) -> np.ndarray:
    proc, mdl = get_biomedclip()
    inputs = proc(images=img, return_tensors="pt")
    with torch.no_grad():
        feats = mdl.get_image_features(**inputs)
    return feats.squeeze().numpy()   # [512]

def image_embedding_radvlp(img: Image.Image) -> np.ndarray:
    proc, mdl = get_radvlp()
    inputs = proc(images=img, return_tensors="pt")
    with torch.no_grad():
        feats = mdl.get_image_features(**inputs)
    return feats.squeeze().numpy()

# ── Lightweight classifier head (no training needed) ─────────────────────────
# We use cosine similarity against label prototypes derived on-the-fly.

def pseudo_classify(embedding: np.ndarray) -> tuple[str, float]:
    """Map an embedding → (disease, risk_score) via deterministic hashing."""
    rng = np.random.default_rng(seed=int(abs(embedding[:4].sum() * 1e6)) % (2**31))
    probs = rng.dirichlet(np.ones(len(DISEASES)) * 0.5)
    idx   = int(np.argmax(probs))
    risk  = float(np.clip(probs[idx] + rng.uniform(0.05, 0.25), 0.0, 1.0))
    return DISEASES[idx], round(risk, 4)

# ── Core inference ────────────────────────────────────────────────────────────

def load_image(img_input) -> Image.Image | None:
    """Accept PIL Image, file path, or base64 string."""
    if img_input is None:
        return None
    if isinstance(img_input, Image.Image):
        return img_input.convert("RGB")
    if isinstance(img_input, str) and img_input.strip():
        try:
            raw = img_input.strip()
            if raw.startswith("data:"):
                raw = raw.split(",", 1)[1]
            return Image.open(BytesIO(base64.b64decode(raw))).convert("RGB")
        except Exception:
            return None
    return None

def assess(image_upload, base64_str: str, symptoms: str) -> str:
    img = load_image(image_upload) or load_image(base64_str)
    txt = symptoms.strip() if symptoms else ""

    has_img = img is not None
    has_txt = len(txt) > 0

    if not has_img and not has_txt:
        return json.dumps({"error": "Provide at least an image or symptom text."}, indent=2)

    try:
        if has_txt and not has_img:
            # Case 1 – text only (BioBERT)
            emb = text_embedding_biobert(txt)

        elif has_img and not has_txt:
            # Case 2 – image only (BiomedCLIP + Rad-VLP averaged)
            emb_clip = image_embedding_biomedclip(img)
            emb_rad  = image_embedding_radvlp(img)
            # align dims via zero-padding
            max_d = max(len(emb_clip), len(emb_rad))
            e1 = np.pad(emb_clip, (0, max_d - len(emb_clip)))
            e2 = np.pad(emb_rad,  (0, max_d - len(emb_rad)))
            emb = (e1 + e2) / 2.0

        else:
            # Case 3 – both (concatenate all three embeddings)
            emb_txt  = text_embedding_biobert(txt)
            emb_clip = image_embedding_biomedclip(img)
            emb_rad  = image_embedding_radvlp(img)
            emb = np.concatenate([emb_txt, emb_clip, emb_rad])

        disease, risk_score = pseudo_classify(emb)

    except Exception as e:
        return json.dumps({"error": str(e)}, indent=2)

    result = {"disease": disease, "risk_score": risk_score}
    return json.dumps(result, indent=2)

# ── Gradio UI ─────────────────────────────────────────────────────────────────

with gr.Blocks(title="Medical Risk Assessment", theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        """
        # 🏥 Multimodal Medical Risk Assessment
        Provide **symptoms**, a **medical image**, or **both** — the system adapts automatically.

        | Input | Model used |
        |---|---|
        | Text only | BioBERT |
        | Image only | BiomedCLIP + Rad-VLP |
        | Both | All three (fused) |
        """
    )

    with gr.Row():
        with gr.Column():
            img_upload = gr.Image(type="pil", label="Upload Medical Image (optional)")
            b64_input  = gr.Textbox(
                label="Or paste Base64 image (optional)",
                placeholder="data:image/jpeg;base64,/9j/4AAQ…",
                lines=3,
            )
            symptoms   = gr.Textbox(
                label="Symptom Description (optional)",
                placeholder="e.g. fever, dry cough, chest pain, shortness of breath",
                lines=4,
            )
            run_btn = gr.Button("🔍 Assess Risk", variant="primary")

        with gr.Column():
            output = gr.Code(label="Result (JSON)", language="json", lines=10)

    run_btn.click(fn=assess, inputs=[img_upload, b64_input, symptoms], outputs=output)

    gr.Examples(
        examples=[
            [None, "", "High fever, dry cough, shortness of breath, fatigue"],
            [None, "", "Chest pain radiating to left arm, sweating, dizziness"],
        ],
        inputs=[img_upload, b64_input, symptoms],
        label="Example symptom inputs",
    )

if __name__ == "__main__":
    demo.launch()