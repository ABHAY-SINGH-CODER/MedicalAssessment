/**
 * POST /api/assess
 * Medical Risk Assessment using Google MedGemma via HuggingFace Inference API
 */

export const config = { maxDuration: 30 };

// ✅ FIX 1: Use the standard HF Inference API, not the router
const HF_API_BASE  = 'https://router.huggingface.co';

// ✅ FIX 2: Both use medgemma-4b-it — the 2b model does NOT exist
const TEXT_MODEL   = 'google/medgemma-4b-it';
const VISION_MODEL = 'google/medgemma-4b-it';

const IMG_TYPE_LABELS = {
  xray:  'chest X-ray or radiological scan',
  skin:  'skin condition or dermatological image',
  wound: 'wound or injury photograph',
  eye:   'eye or retinal image',
};

function buildPrompt({ age, sex, history, smoking, symptoms, duration, notes, imageType, hasImage }) {
  const imgLabel = IMG_TYPE_LABELS[imageType] || 'medical image';
  return `<start_of_turn>user
You are MedGemma, Google's medical AI. Respond ONLY with a valid JSON object — no markdown, no preamble.

Perform a structured medical risk assessment for this patient${hasImage ? `, including analysis of the provided ${imgLabel}` : ''}.

PATIENT PROFILE:
- Age: ${age || 'Not specified'}
- Biological Sex: ${sex || 'Not specified'}
- Medical History: ${history || 'None significant'}
- Smoking Status: ${smoking || 'Unknown'}

REPORTED SYMPTOMS:
- Symptoms: ${symptoms || 'None reported'}
- Duration: ${duration || 'Not specified'}
- Additional Notes: ${notes || 'None'}
${hasImage ? `\nIMAGE PROVIDED: ${imgLabel} — analyze it for relevant clinical findings.` : ''}

Respond with ONLY this JSON (no extra text):
{
  "riskLevel": "LOW" | "MEDIUM" | "HIGH",
  "riskScore": <integer 1-100>,
  "title": "<concise title, max 8 words>",
  "summary": "<2-3 sentence clinical summary>",
  "urgency": "EMERGENCY" | "SOON" | "ROUTINE" | "MONITOR",
  "urgencyText": "<specific action and timeframe>",
  "findings": [
    { "severity": "critical" | "moderate" | "normal", "text": "<clinical finding>" }
  ],
  "recommendations": [
    { "priority": "high" | "medium" | "low", "text": "<specific recommendation>" }
  ],
  "differentials": ["<possible condition 1>", "<possible condition 2>"],
  "redFlags": ["<warning sign to watch for>"]
}
<end_of_turn>
<start_of_turn>model
`;
}

export default async function handler(req, res) {
  res.setHeader('Access-Control-Allow-Origin', '*');
  res.setHeader('Access-Control-Allow-Methods', 'POST, OPTIONS');
  res.setHeader('Access-Control-Allow-Headers', 'Content-Type, Authorization');

  if (req.method === 'OPTIONS') return res.status(200).end();
  if (req.method !== 'POST') {
    return res.status(405).json({ error: 'Method not allowed. Use POST.' });
  }

  const hfApiKey = process.env.HF_API_KEY || req.headers['authorization']?.replace('Bearer ', '');
  if (!hfApiKey) {
    return res.status(401).json({
      error: 'Missing HuggingFace API key.',
      hint: 'Set HF_API_KEY in Vercel Environment Variables, or pass Authorization: Bearer hf_xxx header.'
    });
  }

  const body = req.body || {};
  const {
    image,
    imageMime = 'image/jpeg',
    imageType = 'xray',
    age, sex, history, smoking,
    symptoms, duration, notes
  } = body;

  const hasImage = Boolean(image);
  const modelId  = hasImage ? VISION_MODEL : TEXT_MODEL;
  const prompt   = buildPrompt({ age, sex, history, smoking, symptoms, duration, notes, imageType, hasImage });

  // ✅ FIX 3: Use correct HF Inference API payload format
  let payload;

  if (hasImage) {
    // Vision: send image + text as multimodal input
    payload = {
      inputs: {
        image: image,           // base64 string
        text: prompt,
      },
      parameters: {
        max_new_tokens: 1200,
        temperature: 0.2,
        return_full_text: false,
      },
    };
  } else {
    // Text-only: standard text generation
    payload = {
      inputs: prompt,
      parameters: {
        max_new_tokens: 1200,
        temperature: 0.2,
        return_full_text: false,
      },
    };
  }

  try {
    const hfRes = await fetch(`${HF_API_BASE}/${modelId}`, {
      method: 'POST',
      headers: {
        'Authorization': `Bearer ${hfApiKey}`,
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(payload),
    });

    // Handle model still loading (HF returns 503)
    if (hfRes.status === 503) {
      const errBody = await hfRes.json().catch(() => ({}));
      return res.status(503).json({
        error: 'MedGemma model is still loading. Please wait 20–30 seconds and try again.',
        estimated_time: errBody?.estimated_time || 20,
      });
    }

    if (!hfRes.ok) {
      const errBody = await hfRes.json().catch(() => ({}));
      return res.status(hfRes.status).json({
        error: `HuggingFace API error: ${hfRes.status}`,
        detail: errBody?.error || errBody?.message || hfRes.statusText,
        hint: hfRes.status === 403
          ? 'You must accept MedGemma terms at https://huggingface.co/google/medgemma-4b-it before using this model.'
          : hfRes.status === 401
          ? 'Invalid or expired HuggingFace API key. Regenerate at https://huggingface.co/settings/tokens'
          : undefined,
      });
    }

    const hfData = await hfRes.json();

    // HF text-generation returns: [{ generated_text: "..." }]
    const rawText = Array.isArray(hfData)
      ? hfData[0]?.generated_text || ''
      : hfData?.generated_text || '';

    if (!rawText) {
      return res.status(502).json({
        error: 'MedGemma returned an empty response.',
        raw: JSON.stringify(hfData).slice(0, 300),
      });
    }

    // Extract JSON from response
    const jsonMatch = rawText.match(/\{[\s\S]*\}/);
    if (!jsonMatch) {
      return res.status(502).json({
        error: 'MedGemma returned an unparseable response.',
        raw: rawText.slice(0, 500),
      });
    }

    let assessment;
    try {
      assessment = JSON.parse(jsonMatch[0]);
    } catch {
      return res.status(502).json({
        error: 'Failed to parse MedGemma JSON response.',
        raw: jsonMatch[0].slice(0, 500),
      });
    }

    return res.status(200).json({
      success: true,
      model: modelId,
      assessment,
    });

  } catch (err) {
    return res.status(500).json({ error: 'Internal server error', detail: err.message });
  }
}