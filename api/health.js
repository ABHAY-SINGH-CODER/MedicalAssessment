/**
 * GET /api/health
 * Health check — returns API status and available models.
 */

export default function handler(req, res) {
  res.setHeader('Access-Control-Allow-Origin', '*');

  if (req.method !== 'GET') {
    return res.status(405).json({ error: 'Method not allowed' });
  }

  const hasKey = Boolean(process.env.HF_API_KEY);

  return res.status(200).json({
    status: 'ok',
    service: 'MedGemma Risk Assessment API',
    version: '1.0.0',
    timestamp: new Date().toISOString(),
    apiKeyConfigured: hasKey,
    models: {
      textOnly:   'google/medgemma-2b-it',
      multimodal: 'google/medgemma-4b-it',
    },
    endpoints: {
      'POST /api/assess':  'Run medical risk assessment (image + symptoms)',
      'GET  /api/health':  'Health check',
      'GET  /api/models':  'List available models and their capabilities',
    },
    disclaimer: 'For informational purposes only. Not a substitute for professional medical advice.',
  });
}