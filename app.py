"""
app.py  (Hugging Face Spaces entry-point)
Gradio UI for the Multimodal Medical Risk Assessment System.

Launch:  python app.py
HF Space: set SDK=gradio in README.md, entry-point = app.py
"""

import base64
import io
import json
import logging
from typing import Optional, Tuple

import gradio as gr
import numpy as np
from PIL import Image

from configs.config import APP_CONFIG
from src.inference_engine import MedicalRiskEngine
from src.utils.helpers import setup_logging

setup_logging("INFO")
logger = logging.getLogger(__name__)

# ─────────────────────── Load engine (once) ───────────────────────────────────
engine = MedicalRiskEngine()


# ─────────────────────── Inference wrapper ────────────────────────────────────

def run_assessment(
    image_input: Optional[np.ndarray],
    image_b64_input: str,
    symptoms: str,
) -> Tuple:
    """
    Called by Gradio on submit.
    Accepts either:
      - Uploaded image (numpy array from gr.Image)
      - Pasted base64 string

    Returns 6 outputs for the UI panels.
    """
    pil_image = None
    b64_str = None

    if image_input is not None:
        pil_image = Image.fromarray(image_input.astype("uint8"))
    elif image_b64_input and image_b64_input.strip():
        b64_str = image_b64_input.strip()
    else:
        return (
            "⚠️ Please upload an image or paste a base64 string.",
            "", "", "", "{}", ""
        )

    try:
        result = engine.predict(
            image_b64=b64_str,
            image_pil=pil_image,
            symptoms=symptoms or "",
        )
        d = result.to_dict()
        exp = d["explanation"]

        # ── Panel 1: Primary result ───────────────────────────────────────
        risk_emoji = {"Low": "🟢", "Moderate": "🟡", "High": "🟠", "Critical": "🔴"}.get(
            d["risk_label"], "⚪"
        )
        primary_md = (
            f"## {risk_emoji} {d['risk_label']} Risk\n\n"
            f"**Primary Disease:** {d['primary_disease']}\n\n"
            f"**Risk Score:** `{d['risk_score']:.3f}` / 1.0  "
            f"*(uncertainty ±{d['risk_uncertainty']:.3f})*\n\n"
            f"**Confidence:** {exp['confidence_label']} ({exp['confidence_percentage']}%)\n\n"
            f"---\n{exp['risk_narrative']}"
        )

        # ── Panel 2: Differential diagnoses ──────────────────────────────
        diff_rows = "\n".join([
            f"| {i+1} | {dd['disease']} | {dd['probability']}% |"
            for i, dd in enumerate(exp.get("differential_diagnoses", []))
        ])
        diff_md = (
            "## Differential Diagnoses\n\n"
            "| # | Disease | Probability |\n"
            "|---|---------|-------------|\n"
            + diff_rows
        )

        # ── Panel 3: Symptom importance ───────────────────────────────────
        sym_rows = "\n".join([
            f"| {s['symptom']} | {'█' * int(s['importance'] * 10)} {s['importance']:.3f} |"
            for s in exp.get("top_symptoms", [])
        ])
        sym_md = (
            "## Symptom Importance\n\n"
            "| Symptom | Importance |\n"
            "|---------|------------|\n"
            + (sym_rows or "| — | No symptom tokens detected |")
        )

        # ── Panel 4: GARD rare disease enrichment ─────────────────────────
        gard = d.get("gard_enrichment", {})
        rare = gard.get("rare_candidates", [])
        gard_rows = "\n".join([
            f"| {r['name']} | {r['id']} | {r['match_score']:.3f} |"
            for r in rare
        ]) or "| No matches | — | — |"
        gard_md = (
            f"## GARD Rare Disease Enrichment\n\n"
            f"**GARD Risk Premium:** `{gard.get('gard_premium', 0):.4f}`\n\n"
            "| Disease | GARD ID | Match Score |\n"
            "|---------|---------|-------------|\n"
            + gard_rows
        )

        # ── Panel 5: Full JSON ────────────────────────────────────────────
        json_out = json.dumps(d, indent=2)

        # ── Panel 6: Inference metadata ───────────────────────────────────
        meta = d.get("metadata", {})
        meta_md = (
            f"**Inference time:** {meta.get('inference_time_ms', 0):.0f} ms\n\n"
            + "\n".join([f"- **{k}:** `{v}`" for k, v in meta.get("model_versions", {}).items()])
        )

        return primary_md, diff_md, sym_md, gard_md, json_out, meta_md

    except ValueError as e:
        return f"❌ Input error: {e}", "", "", "", "{}", ""
    except Exception as e:
        logger.error(f"Assessment error: {e}", exc_info=True)
        return f"❌ System error: {e}", "", "", "", "{}", ""


# ─────────────────────── Build Gradio UI ─────────────────────────────────────

DESCRIPTION = """
# 🏥 Multimodal Medical Risk Assessment System

**Models:** BioBERT · BiomedCLIP · Rad-VLP (RAD-DINO) · GARD Enrichment  
**Fusion:** Cross-Modal Attention Transformer  
**Output:** Disease prediction · Risk score [0–1] · Interpretable explanation

> ⚠️ **Disclaimer:** For research and educational use only. Not a substitute for professional medical diagnosis.
"""

EXAMPLES = [
    [None, "", "fever, cough, shortness of breath, chest pain"],
    [None, "", "night sweats, weight loss, hemoptysis"],
    [None, "", "edema, dyspnea, fatigue"],
]

with gr.Blocks(
    theme=gr.themes.Soft(
        primary_hue="blue",
        secondary_hue="slate",
        neutral_hue="slate",
    ),
    title="Medical Risk Assessment",
    css="""
    .risk-card { border-radius: 12px; padding: 16px; }
    .output-panel { font-family: 'JetBrains Mono', monospace; font-size: 13px; }
    footer { display: none !important; }
    """,
) as demo:
    gr.Markdown(DESCRIPTION)

    with gr.Row():
        # ── Left column: Inputs ───────────────────────────────────────────
        with gr.Column(scale=1):
            gr.Markdown("### 📥 Input")

            with gr.Tab("Upload Image"):
                image_upload = gr.Image(
                    label="Medical Image (Chest X-Ray, CT slice, etc.)",
                    type="numpy",
                    height=280,
                )

            with gr.Tab("Paste Base64"):
                image_b64 = gr.Textbox(
                    label="Base64 Image String",
                    placeholder="data:image/png;base64,iVBORw0KGgo... or raw base64",
                    lines=6,
                )

            symptoms_input = gr.Textbox(
                label="🩺 Symptoms (comma-separated)",
                placeholder="e.g. fever, cough, chest pain, shortness of breath",
                lines=3,
            )

            submit_btn = gr.Button("🔍 Run Assessment", variant="primary", size="lg")
            clear_btn = gr.ClearButton(
                components=[image_upload, image_b64, symptoms_input],
                value="🗑️ Clear",
            )

            gr.Markdown(
                "**Quick Examples:**",
            )
            gr.Examples(
                examples=EXAMPLES,
                inputs=[image_upload, image_b64, symptoms_input],
                label="",
            )

        # ── Right column: Outputs ─────────────────────────────────────────
        with gr.Column(scale=2):
            gr.Markdown("### 📊 Assessment Results")

            with gr.Tab("🎯 Primary Result"):
                primary_output = gr.Markdown(elem_classes=["risk-card"])

            with gr.Tab("📋 Differential"):
                diff_output = gr.Markdown()

            with gr.Tab("💊 Symptoms"):
                sym_output = gr.Markdown()

            with gr.Tab("🧬 GARD Enrichment"):
                gard_output = gr.Markdown()

            with gr.Tab("📦 Full JSON"):
                json_output = gr.Code(language="json", elem_classes=["output-panel"])

            with gr.Tab("ℹ️ Model Info"):
                meta_output = gr.Markdown()

    submit_btn.click(
        fn=run_assessment,
        inputs=[image_upload, image_b64, symptoms_input],
        outputs=[primary_output, diff_output, sym_output, gard_output, json_output, meta_output],
        api_name="assess",
    )

    gr.Markdown(
        """
        ---
        **Architecture:** Input → BioBERT (text) + BiomedCLIP + Rad-VLP (image) 
        → Cross-Modal Attention Fusion → GARD Risk Enrichment → Explainability  
        **Repo:** [GitHub](https://github.com/your-org/medical-risk-system) · 
        **API Docs:** [/docs](/docs)
        """
    )


# ─────────────────────── Launch ──────────────────────────────────────────────

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
    )
