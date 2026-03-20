/** GET /api/models */
export default function handler(req, res) {
  res.setHeader('Access-Control-Allow-Origin', '*');
  res.status(200).json({
    models: [
      {
        id: 'claude-haiku-4-5-20251001',
        provider: 'Anthropic',
        type: 'vision + text',
        supportsImage: true,
        imageTypes: ['xray', 'skin', 'wound', 'eye'],
        selected: true,
      },
    ],
  });
}
 