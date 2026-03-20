/** GET /api/models */
export default function handler(req, res) {
  res.setHeader('Access-Control-Allow-Origin', '*');
  res.status(200).json({
    models: [
      {
        id: 'gemini-2.0-flash',
        provider: 'Google',
        type: 'vision + text',
        contextWindow: 1000000,
        supportsImage: true,
        imageTypes: ['xray', 'skin', 'wound', 'eye'],
        free: true,
        selected: true,
      },
    ],
    selection: 'gemini-1.5-flash is used for all requests (text and vision)',
  });
}
 