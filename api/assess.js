/**
 * POST /api/assess
 * Medical Risk Assessment using Anthropic Claude API
 *
 * Set ANTHROPIC_API_KEY in Vercel env vars
 * Get key: https://console.anthropic.com/settings/keys
 */

export const config = { maxDuration: 30 };

const MODEL = 'claude-haiku-4-5-20251001'; // fast + cheap
const URL   = 'https://api.anthropic.com/v1/messages';

const IMG_LABELS = {
  xray:  'chest X-ray or radiological scan',
  skin:  'skin condition or dermatological image',
  wound: 'wound or injury photograph',
  eye:   'eye or retinal image',
};

function buildPrompt({ age, sex, history, symptoms, duration, notes, imageType, hasImage }) {
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

  const apiKey = process.env.ANTHROPIC_API_KEY
    || req.headers['authorization']?.replace('Bearer ', '').trim();

  if (!apiKey) {
    return res.status(401).json({
      error: 'Missing Anthropic API key.',
      hint: 'Set ANTHROPIC_API_KEY in Vercel env vars.',
      keyUrl: 'https://console.anthropic.com/settings/keys',
    });
  }

  const {
    image, imageMime = 'image/jpeg', imageType = 'xray',
    age, sex, history, symptoms, duration, notes,
  } = req.body || {};

  const hasImage = Boolean(image);
  const prompt = buildPrompt({ age, sex, history, symptoms, duration, notes, imageType, hasImage });

  // Build content array — Claude supports vision natively
  const content = [];
  if (hasImage) {
    content.push({
      type: 'image',
      source: { type: 'base64', media_type: imageMime, data: image },
    });
  }
  content.push({ type: 'text', text: prompt });

  try {
    const aRes = await fetch(URL, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'x-api-key': apiKey,
        'anthropic-version': '2023-06-01',
      },
      body: JSON.stringify({
        model: MODEL,
        max_tokens: 1200,
        messages: [{ role: 'user', content }],
      }),
    });

    const aData = await aRes.json();

    if (!aRes.ok) {
      return res.status(aRes.status).json({
        error: `Anthropic API error ${aRes.status}`,
        detail: aData?.error?.message || aRes.statusText,
        hint: aRes.status === 401
          ? 'Invalid API key — check https://console.anthropic.com/settings/keys'
          : aRes.status === 429
          ? 'Rate limit hit — wait a moment and retry.'
          : undefined,
      });
    }

    const raw = aData?.content?.[0]?.text || '';
    if (!raw) {
      return res.status(502).json({ error: 'Empty response from Claude.', raw: JSON.stringify(aData).slice(0, 300) });
    }

    const cleaned = raw.replace(/```json|```/g, '').trim();
    const match = cleaned.match(/\{[\s\S]*\}/);
    if (!match) {
      return res.status(502).json({ error: 'Could not extract JSON.', raw: raw.slice(0, 400) });
    }

    let assessment;
    try { assessment = JSON.parse(match[0]); }
    catch { return res.status(502).json({ error: 'JSON parse failed.', raw: match[0].slice(0, 400) }); }

    return res.status(200).json({ success: true, model: `Claude ${MODEL}`, assessment });

  } catch (err) {
    return res.status(500).json({ error: 'Internal server error', detail: err.message });
  }
}