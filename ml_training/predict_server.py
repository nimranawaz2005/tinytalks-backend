import io
import numpy as np
import librosa
import torch
import joblib
import soundfile as sf
from flask import Flask, request, jsonify, Response
from flask_cors import CORS
from transformers import ASTFeatureExtractor, ASTModel, VitsModel, AutoTokenizer

SAMPLE_RATE = 16000
CLASS_NAMES = ["hungry", "tired", "pain_discomfort"]
LOW_CONFIDENCE_CLASSES = {"tired"}

print("Loading AST embedding model...")
MODEL_NAME = "MIT/ast-finetuned-audioset-10-10-0.4593"
feature_extractor = ASTFeatureExtractor.from_pretrained(MODEL_NAME)
embedding_model = ASTModel.from_pretrained(MODEL_NAME)
embedding_model.eval()

print("Loading trained classifier + scaler...")
classifier = joblib.load("saved_model/cry_cause_classifier.joblib")
scaler = joblib.load("saved_model/scaler.joblib")

print("Loading Urdu TTS model (MMS)...")
tts_model = VitsModel.from_pretrained("facebook/mms-tts-urd-script_arabic")
tts_tokenizer = AutoTokenizer.from_pretrained("facebook/mms-tts-urd-script_arabic")

# In-memory cache: exact text -> generated WAV bytes. Repeated phrases
# (like fixed test/prompt text) are synthesized once, then served
# instantly from memory on every later request. Cleared on server restart.
tts_cache = {}

app = Flask(__name__)
CORS(app)


def get_embedding(y, sr):
    inputs = feature_extractor(y, sampling_rate=sr, return_tensors="pt")
    with torch.no_grad():
        outputs = embedding_model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze().numpy()


@app.route("/classify-cause", methods=["POST"])
def classify_cause():
    audio_bytes = request.data
    if not audio_bytes:
        return jsonify({"error": "No audio data received."}), 400

    try:
        y, sr = librosa.load(io.BytesIO(audio_bytes), sr=SAMPLE_RATE)
    except Exception as e:
        return jsonify({"error": f"Could not decode audio: {e}"}), 400

    try:
        embedding = get_embedding(y, sr)
        embedding_scaled = scaler.transform([embedding])
        probs = classifier.predict_proba(embedding_scaled)[0]
        pred_idx = int(np.argmax(probs))
        pred_class = CLASS_NAMES[pred_idx]

        result = {
            "predicted_cause": pred_class,
            "confidence": float(probs[pred_idx]),
            "low_confidence_class": pred_class in LOW_CONFIDENCE_CLASSES,
            "all_probabilities": {
                CLASS_NAMES[i]: float(p) for i, p in enumerate(probs)
            },
        }
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": f"Prediction failed: {e}"}), 500


@app.route("/synthesize-urdu", methods=["POST"])
def synthesize_urdu():
    data = request.get_json()
    text = data.get("text", "") if data else ""
    if not text:
        return jsonify({"error": "No text provided."}), 400

    if text in tts_cache:
        return Response(tts_cache[text], mimetype="audio/wav")

    try:
        inputs = tts_tokenizer(text, return_tensors="pt")
        with torch.no_grad():
            output = tts_model(**inputs).waveform

        waveform = output.squeeze().numpy()
        buffer = io.BytesIO()
        sf.write(buffer, waveform, tts_model.config.sampling_rate, format="WAV")
        buffer.seek(0)
        audio_bytes = buffer.read()

        tts_cache[text] = audio_bytes
        return Response(audio_bytes, mimetype="audio/wav")
    except Exception as e:
        return jsonify({"error": f"TTS synthesis failed: {e}"}), 500


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001)
