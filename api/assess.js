/**
 * POST /api/assess
 * Medical Risk Assessment using Google Gemini 1.5 Flash
 *
 * Body (JSON):
 *   image       {string}  base64-encoded image (optional)
 *   imageMime   {string}  e.g. "image/jpeg" (default: image/jpeg)
 *   imageType   {string}  xray | skin | wound | eye
 *   age         {string|number}
 *   sex         {string}
 *   history     {string}
 *   symptoms    {string}  comma-separated
 *   duration    {string}
 *   notes       {string}
 *
 * Auth:
 *   Set GEMINI_API_KEY in Vercel env vars  OR
 *   Pass  Authorization: Bearer AIza...  header
 */

export const config = { maxDuration: 30 };

const MODEL = 'gemini-2.0-flash';
const URL   = `https://generativelanguage.googleapis.com/v1beta/models/${MODEL}:generateContent`;

const IMG_LABELS = {
  xray:  'chest X-ray or radiological scan',
  skin:  'skin condition or dermatological image',
  wound: 'wound or injury photograph',
  eye:   'eye or retinal image',
};

function buildPrompt({ age, sex, history, smoking, symptoms, duration, notes, imageType, hasImage }) {
  const imgLabel = IMG_LABELS[imageType] || 'medical image';
  return `You are a medical AI assistant. Respond ONLY with a valid JSON object — no markdown, no backticks, no explanation.

Perform a clinical risk assessment for this patient${hasImage ? `, including analysis of the provided ${imgLabel}` : ''}.

PATIENT:
- Age: ${age || 'Not specified'}
- Sex: ${sex || 'Not specified'}
- History: ${history || 'None'}
- Symptoms: ${symptoms || 'None reported'}
- Duration: ${duration || 'Not specified'}
- Notes: ${notes || 'None'}
${hasImage ? `\nAnalyze the attached ${imgLabel} for clinical findings.` : ''}

Return ONLY this JSON:
{
  "riskLevel": "LOW" | "MEDIUM" | "HIGH",
  "riskScore": <integer 1-100>,
  "title": "<max 8 words>",
  "summary": "<2-3 sentence clinical summary>",
  "urgency": "EMERGENCY" | "SOON" | "ROUTINE" | "MONITOR",
  "urgencyText": "<specific action and timeframe>",
  "findings": [{ "severity": "critical"|"moderate"|"normal", "text": "<finding>" }],
  "recommendations": [{ "priority": "high"|"medium"|"low", "text": "<recommendation>" }],
  "differentials": ["<condition>"],
  "redFlags": ["<warning sign>"]
}`;
}

export default async function handler(req, res) {
  res.setHeader('Access-Control-Allow-Origin', '*');
  res.setHeader('Access-Control-Allow-Methods', 'POST, OPTIONS');
  res.setHeader('Access-Control-Allow-Headers', 'Content-Type, Authorization');
  if (req.method === 'OPTIONS') return res.status(200).end();
  if (req.method !== 'POST') return res.status(405).json({ error: 'Use POST.' });

  const apiKey = process.env.GEMINI_API_KEY
    || req.headers['authorization']?.replace('Bearer ', '').trim();

  if (!apiKey) {
    return res.status(401).json({
      error: 'Missing Gemini API key.',
      hint: 'Set GEMINI_API_KEY in Vercel env vars or pass Authorization: Bearer AIza... header.',
      keyUrl: 'https://aistudio.google.com/app/apikey',
    });
  }

  const {
    image, imageMime = 'image/jpeg', imageType = 'xray',
    age, sex, history, smoking, symptoms, duration, notes,
  } = req.body || {};

  const hasImage = Boolean(image);
  const prompt   = buildPrompt({ age, sex, history, smoking, symptoms, duration, notes, imageType, hasImage });

  // Build Gemini parts — image first if present
  const parts = [];
  if (hasImage) {
    parts.push({ inline_data: { mime_type: imageMime, data: image } });
  }
  parts.push({ text: prompt });

  try {
    const gRes = await fetch(`${URL}?key=${apiKey}`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        contents: [{ role: 'user', parts }],
        generationConfig: { temperature: 0.2, maxOutputTokens: 1200 },
      }),
    });

    const gData = await gRes.json();

    if (!gRes.ok) {
      return res.status(gRes.status).json({
        error: `Gemini API error ${gRes.status}`,
        detail: gData?.error?.message || gRes.statusText,
        hint: gRes.status === 400 ? 'Check your request payload.'
            : gRes.status === 403 ? 'Invalid API key — check https://aistudio.google.com/app/apikey'
            : undefined,
      });
    }

    const raw = gData?.candidates?.[0]?.content?.parts?.[0]?.text || '';
    if (!raw) {
      return res.status(502).json({ error: 'Empty response from Gemini.', raw: JSON.stringify(gData).slice(0, 300) });
    }

    // Strip markdown fences and extract JSON object
    const cleaned = raw.replace(/```json|```/g, '').trim();
    const match   = cleaned.match(/\{[\s\S]*\}/);
    if (!match) {
      return res.status(502).json({ error: 'Could not extract JSON from response.', raw: raw.slice(0, 500) });
    }

    let assessment;
    try { assessment = JSON.parse(match[0]); }
    catch { return res.status(502).json({ error: 'JSON parse failed.', raw: match[0].slice(0, 500) }); }

    return res.status(200).json({ success: true, model: `Gemini ${MODEL}`, assessment });

  } catch (err) {
    return res.status(500).json({ error: 'Internal server error', detail: err.message });
  }
}