require('dotenv').config();
const express = require('express');
const cors = require('cors');
const fs = require('fs');
const path = require('path');

const app = express();
const PORT = process.env.PORT || 3001;
const HF_API_KEY = process.env.HF_API_KEY;

const HF_MODEL_URL = 'https://router.huggingface.co/hf-inference/models/MIT/ast-finetuned-audioset-10-10-0.4593';

if (!HF_API_KEY) {
  console.error('ERROR: HF_API_KEY is not set. Create a .env file with HF_API_KEY=your_key');
  process.exit(1);
}

app.use(cors());

app.use('/api/cry-detect', express.raw({ type: '*/*', limit: '15mb' }));

app.post('/api/cry-detect', async (req, res) => {
  try {
    const audioBuffer = req.body;
    if (!audioBuffer || audioBuffer.length === 0) {
      return res.status(400).json({ error: 'No audio data received.' });
    }
    const hfResponse = await fetch(HF_MODEL_URL, {
      method: 'POST',
      headers: {
        Authorization: `Bearer ${HF_API_KEY}`,
        'Content-Type': req.headers['content-type'] || 'audio/wav',
      },
      body: audioBuffer,
    });
    if (!hfResponse.ok) {
      const errText = await hfResponse.text().catch(() => '');
      console.error('Hugging Face API error:', hfResponse.status, errText);
      if (hfResponse.status === 503) {
        return res.status(503).json({ error: 'Model is loading, please retry shortly.' });
      }
      if (hfResponse.status === 404) {
        return res.status(404).json({ error: 'Model not found on the current Hugging Face router.' });
      }
      return res.status(hfResponse.status).json({ error: 'Hugging Face request failed.' });
    }
    const data = await hfResponse.json();
    return res.json(data);
  } catch (err) {
    console.error('Cry detection proxy error:', err);
    return res.status(500).json({ error: 'Internal server error.' });
  }
});

app.use('/api/classify-cause', express.raw({ type: '*/*', limit: '15mb' }));

app.post('/api/classify-cause', async (req, res) => {
  try {
    const audioBuffer = req.body;
    if (!audioBuffer || audioBuffer.length === 0) {
      return res.status(400).json({ error: 'No audio data received.' });
    }
    const pyResponse = await fetch('http://127.0.0.1:5001/classify-cause', {
      method: 'POST',
      headers: { 'Content-Type': req.headers['content-type'] || 'audio/wav' },
      body: audioBuffer,
    });
    if (!pyResponse.ok) {
      const errText = await pyResponse.text().catch(() => '');
      console.error('Cause classifier error:', pyResponse.status, errText);
      return res.status(pyResponse.status).json({ error: 'Cause classification failed.' });
    }
    const data = await pyResponse.json();
    return res.json(data);
  } catch (err) {
    console.error('Cause classifier proxy error:', err);
    return res.status(503).json({
      error: 'Cause classification service unavailable. Is predict_server.py running on port 5001?',
    });
  }
});

const FEEDBACK_DIR = path.join(__dirname, 'feedback_data');
const FEEDBACK_AUDIO_DIR = path.join(FEEDBACK_DIR, 'audio');
if (!fs.existsSync(FEEDBACK_AUDIO_DIR)) fs.mkdirSync(FEEDBACK_AUDIO_DIR, { recursive: true });

app.use('/api/feedback', express.raw({ type: '*/*', limit: '15mb' }));

app.post('/api/feedback', (req, res) => {
  try {
    const audioBuffer = req.body;
    if (!audioBuffer || audioBuffer.length === 0) {
      return res.status(400).json({ error: 'No audio data received.' });
    }
    const { predicted, confidence, correct, wasCorrect } = req.query;
    const timestamp = Date.now();
    const filename = `feedback_${timestamp}.wav`;
    fs.writeFileSync(path.join(FEEDBACK_AUDIO_DIR, filename), audioBuffer);
    const logEntry = {
      timestamp: new Date(timestamp).toISOString(),
      audio_file: filename,
      predicted_cause: predicted || null,
      predicted_confidence: confidence ? parseFloat(confidence) : null,
      was_correct: wasCorrect === 'true',
      correct_cause: correct || null,
    };
    fs.appendFileSync(
      path.join(FEEDBACK_DIR, 'feedback_log.jsonl'),
      JSON.stringify(logEntry) + '\n'
    );
    return res.json({ status: 'ok' });
  } catch (err) {
    console.error('Feedback save error:', err);
    return res.status(500).json({ error: 'Failed to save feedback.' });
  }
});

app.use(express.json());

app.post('/api/tts', async (req, res) => {
  try {
    const { text } = req.body;
    if (!text) {
      return res.status(400).json({ error: 'No text provided.' });
    }
    const ttsResponse = await fetch('http://127.0.0.1:5001/synthesize-urdu', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ text }),
    });
    if (!ttsResponse.ok) {
      const errText = await ttsResponse.text().catch(() => '');
      console.error('Urdu TTS error:', ttsResponse.status, errText);
      return res.status(ttsResponse.status).json({ error: 'TTS synthesis failed.' });
    }
    const audioBuffer = Buffer.from(await ttsResponse.arrayBuffer());
    res.set('Content-Type', 'audio/wav');
    return res.send(audioBuffer);
  } catch (err) {
    console.error('TTS proxy error:', err);
    return res.status(503).json({ error: 'TTS service unavailable. Is predict_server.py running on port 5001?' });
  }
});

app.get('/health', (req, res) => res.json({ status: 'ok' }));

app.listen(PORT, () => {
  console.log(`Cry detection proxy running on http://localhost:${PORT}`);
});
