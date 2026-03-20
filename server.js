const express = require('express');
const axios = require('axios');
const cors = require('cors');
require('dotenv').config();

const app = express();
const PORT = process.env.PORT || 3000;

// Middleware
app.use(cors());
app.use(express.json({ limit: '10mb' })); // Increased limit for base64 images

// Configuration
const HF_TOKEN = process.env.HF_TOKEN;
const API_URL = "https://router.huggingface.co/google/medgemma-1.1-7b-it";

// Helper function for retries (handles model loading)
const callHuggingFace = async (payload, retries = 3, delay = 10000) => {
    for (let i = 0; i < retries; i++) {
        try {
            const response = await axios.post(API_URL, payload, {
                headers: { 
                    'Authorization': `Bearer ${HF_TOKEN}`,
                    'Content-Type': 'application/json'
                },
                timeout: 90000 // 90 seconds
            });
            return response.data;
        } catch (error) {
            const status = error.response?.status;
            
            // If model is loading (503) or gateway timeout (504), wait and retry
            if ((status === 503 || status === 504) && i < retries - 1) {
                console.log(`Model busy, retrying attempt ${i + 1}...`);
                await new Promise(resolve => setTimeout(resolve, delay));
                continue;
            }
            
            throw error;
        }
    }
};

// Routes
app.get('/', (req, res) => {
    res.json({ message: "Medical Assessment API (Node.js) is running." });
});

app.post('/assess', async (req, res) => {
    const { symptoms, image_base64 } = req.body;

    if (!symptoms || symptoms.length < 5) {
        return res.status(400).json({ error: "Please provide detailed symptoms." });
    }

    const systemPrompt = "System: You are a medical diagnostic assistant. Analyze the symptoms and image to assess health risks.";
    
    let payload;
    if (image_base64 && image_base64.length > 100) {
        // Remove data:image/png;base64, prefix if present
        const cleanBase64 = image_base64.includes(',') ? image_base64.split(',')[1] : image_base64;
        
        payload = {
            inputs: {
                image: cleanBase64,
                text: `${systemPrompt}\nSymptoms: ${symptoms}\nQuestion: What abnormalities are visible in this clinical image?`
            }
        };
    } else {
        payload = {
            inputs: `${systemPrompt}\nSymptoms: ${symptoms}\nAssessment:`,
            parameters: { max_new_tokens: 500, temperature: 0.7 }
        };
    }

    try {
        const result = await callHuggingFace(payload);
        res.json(result);
    } catch (error) {
        console.error("HF Error:", error.response?.data || error.message);
        res.status(error.response?.status || 500).json({
            error: "Failed to communicate with the AI model.",
            details: error.response?.data || error.message
        });
    }
});

app.get('/health', (req, res) => {
    res.json({ status: "online", node_version: process.version });
});

app.listen(PORT, () => {
    console.log(`Server is running on port ${PORT}`);
});