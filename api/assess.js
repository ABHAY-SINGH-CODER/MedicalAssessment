/**
 * POST /api/assess
 *
 * Medical Risk Assessment endpoint using Google MedGemma via HuggingFace.
 *
 * Body (multipart/form-data OR application/json):
 *   - image        (optional) base64 string of medical image
 *   - imageMime    (optional) e.g. "image/jpeg"
 *   - imageType    (optional) "xray" | "skin" | "wound" | "eye"
 *   - age          (optional) number
 *   - sex          (optional) string
 *   - history      (optional) string
 *   - smoking      (optional) string
 *   - symptoms     (optional) comma-separated string
 *   - duration     (optional) string
 *   - notes        (optional) string
 *
 * Returns:
 *   JSON { riskLevel, riskScore, title, summary, urgency, urgencyText, findings[], recommendations[] }
 */

export const config = { maxDuration: 30 };

const HF_API_BASE = 'https://router.huggingface.co/hf-inference/models';
const TEXT_MODEL   = 'google/medgemma-2b-it';
const VISION_MODEL = 'google/medgemma-4b-it';

const IMG_TYPE_LABELS = {
  xray:  'chest X-ray or radiological scan',
  skin:  'skin condition or dermatological image',
  wound: 'wound or injury photograph',
  eye:   'eye or retinal image',
};

function buildSystemPrompt() {
  return `You are MedGemma, Google's medical AI model specialized in clinical risk assessment. 
You analyze patient information, symptoms, and medical images to produce structured risk assessments. 
Always respond ONLY with a valid JSON object — no markdown, no explanation, no preamble.`;
}

function buildUserPrompt({ age, sex, history, smoking, symptoms, duration, notes, imageType, hasImage }) {
  const imgLabel = IMG_TYPE_LABELS[imageType] || 'medical image';
  return `Perform a structured medical risk assessment for the following patient${hasImage ? `, including analysis of the provided ${imgLabel}` : ''}.

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

Respond with ONLY this JSON structure (no extra text):
{
  "riskLevel": "LOW" | "MEDIUM" | "HIGH",
  "riskScore": <integer 1-100>,
  "title": "<concise assessment title, max 8 words>",
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
}`;
}

export default async function handler(req, res) {
  // CORS headers
  res.setHeader('Access-Control-Allow-Origin', '*');
  res.setHeader('Access-Control-Allow-Methods', 'POST, OPTIONS');
  res.setHeader('Access-Control-Allow-Headers', 'Content-Type, Authorization');

  if (req.method === 'OPTIONS') {
    return res.status(200).end();
  }

  if (req.method !== 'POST') {
    return res.status(405).json({ error: 'Method not allowed. Use POST.' });
  }

  // Get HuggingFace API key — from env var (preferred) or Authorization header
  const hfApiKey = process.env.HF_API_KEY || req.headers['authorization']?.replace('Bearer ', '');
  if (!hfApiKey) {
    return res.status(401).json({
      error: 'Missing HuggingFace API key. Set HF_API_KEY environment variable in Vercel, or pass Authorization: Bearer hf_xxx header.'
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

  const userPrompt = buildUserPrompt({ age, sex, history, smoking, symptoms, duration, notes, imageType, hasImage });

  // Build HuggingFace messages payload
  const messages = [
    { role: 'system', content: buildSystemPrompt() },
    {
      role: 'user',
      content: hasImage
        ? [
            { type: 'image_url', image_url: { url: `data:${imageMime};base64,${image}` } },
            { type: 'text', text: userPrompt }
          ]
        : userPrompt
    }
  ];

  try {
    const hfRes = await fetch(`${HF_API_BASE}/${modelId}/v1/chat/completions`, {
      method: 'POST',
      headers: {
        'Authorization': `Bearer ${hfApiKey}`,
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        model: modelId,
        messages,
        max_tokens: 1200,
        temperature: 0.2,
        stream: false,
      }),
    });

    if (!hfRes.ok) {
      const errBody = await hfRes.json().catch(() => ({}));
      return res.status(hfRes.status).json({
        error: `HuggingFace API error: ${hfRes.status}`,
        detail: errBody?.error || errBody?.message || hfRes.statusText,
        hint: hfRes.status === 403
          ? 'Accept the MedGemma model terms at https://huggingface.co/google/medgemma-4b-it'
          : hfRes.status === 401
          ? 'Invalid or expired HuggingFace API key.'
          : undefined
      });
    }

    const hfData = await hfRes.json();
    const rawText = hfData.choices?.[0]?.message?.content || '';

    // Extract JSON from response
    const jsonMatch = rawText.match(/\{[\s\S]*\}/);
    if (!jsonMatch) {
      return res.status(502).json({
        error: 'MedGemma returned an unparseable response.',
        raw: rawText.slice(0, 500),
      });
    }

    const assessment = JSON.parse(jsonMatch[0]);

    return res.status(200).json({
      success: true,
      model: modelId,
      assessment,
    });

  } catch (err) {
    return res.status(500).json({ error: 'Internal server error', detail: err.message });
  }
}