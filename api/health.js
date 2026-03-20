/** GET /api/health */
export default function handler(req, res) {
  res.setHeader('Access-Control-Allow-Origin', '*');
  res.status(200).json({
    status: 'ok',
    service: 'MedAssess API',
    model: 'claude-haiku-4-5-20251001',
    provider: 'Anthropic',
    apiKeyConfigured: Boolean(process.env.ANTHROPIC_API_KEY),
    endpoints: ['/api/assess', '/api/health', '/api/models'],
    timestamp: new Date().toISOString(),
  });
}