/** GET /api/health */
export default function handler(req, res) {
  res.setHeader('Access-Control-Allow-Origin', '*');
  res.status(200).json({
    status: 'ok',
    service: 'MedAssess API',
    model: 'gemini-2.0-flash',
    apiKeyConfigured: Boolean(process.env.GEMINI_API_KEY),
    endpoints: ['/api/assess', '/api/health', '/api/models'],
    timestamp: new Date().toISOString(),
  });
}