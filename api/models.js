/**
 * GET /api/models
 * Returns available MedGemma models and their capabilities.
 */

export default function handler(req, res) {
  res.setHeader('Access-Control-Allow-Origin', '*');

  if (req.method !== 'GET') {
    return res.status(405).json({ error: 'Method not allowed' });
  }

  return res.status(200).json({
    models: [
      {
        id: 'google/medgemma-2b-it',
        name: 'MedGemma 2B Instruct',
        type: 'text-only',
        contextWindow: 8192,
        usedFor: 'Symptom-only assessment (no image)',
        speed: 'fast',
        free: true,
      },
      {
        id: 'google/medgemma-4b-it',
        name: 'MedGemma 4B Instruct (Multimodal)',
        type: 'vision + text',
        contextWindow: 131072,
        usedFor: 'Image + symptom assessment (X-rays, skin, wounds, eyes)',
        speed: 'moderate',
        free: true,
        supportedImageTypes: ['xray', 'skin', 'wound', 'eye'],
        supportedMimeTypes: ['image/jpeg', 'image/png', 'image/webp'],
      }
    ],
    selectionLogic: 'If image is provided in /api/assess, medgemma-4b-it is used automatically. Otherwise medgemma-2b-it is used.',
    huggingfaceUrl: 'https://huggingface.co/google',
  });
}