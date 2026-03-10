from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os
import requests as http_requests
from aura_engine import analyse

app = Flask(__name__)
CORS(app)
BASE = os.path.dirname(os.path.abspath(__file__))

def generate_aura_description(scores):
    """Call Cerebras API directly via HTTP."""
    api_key = os.environ.get("CEREBRAS_API_KEY", "")
    if not api_key:
        return None

    symmetry = scores["Face Symmetry"]
    glow     = scores["Skin Glow"]
    eyes     = scores["Eye Intensity"]
    jaw      = scores["Jawline"]

    prompt = f"""You are a brutally honest but kind face analyst. Analyse these real facial measurement scores and give a 2-3 sentence honest reading about what these scores mean about this person's actual face features. Then give 2 very specific tips to improve the score. Talk about the FACE not about lighting or cameras.

Face Symmetry: {symmetry}/100 (how balanced left and right side of face is)
Skin Glow: {glow}/100 (skin clarity and brightness)
Eye Intensity: {eyes}/100 (eye openness and contrast)
Jawline Definition: {jaw}/100 (sharpness of jaw and chin)

Respond EXACTLY in this format:
READING: [2-3 sentences about their actual face based on scores]
TIPS: [Tip 1]. [Tip 2]."""

    try:
        response = http_requests.post(
            "https://api.cerebras.ai/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            },
            json={
                "model": "llama3.1-8b",
                "max_tokens": 250,
                "messages": [{"role": "user", "content": prompt}]
            },
            timeout=15
        )
        data = response.json()
        return data["choices"][0]["message"]["content"]
    except Exception as e:
        print(f"Cerebras error: {e}")
        return None

def parse_ai_response(text):
    """Parse READING and TIPS from AI response."""
    reading = ""
    tips    = ""
    if not text:
        return reading, tips
    for line in text.strip().split("\n"):
        line = line.strip()
        if line.startswith("READING:"):
            reading = line.replace("READING:", "").strip()
        elif line.startswith("TIPS:"):
            tips = line.replace("TIPS:", "").strip()
    # fallback if format slightly off
    if not reading and text:
        reading = text[:300]
    return reading, tips

@app.route('/')
def index():
    return send_from_directory(BASE, 'index.html')

@app.route('/result')
def result():
    return send_from_directory(BASE, 'result.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    if 'image' not in request.files:
        return jsonify({"error": "No image uploaded."}), 400
    file = request.files['image']
    if file.filename == '':
        return jsonify({"error": "Empty filename."}), 400

    result_data = analyse(file.read())
    if "error" in result_data:
        return jsonify(result_data), 422

    # Generate AI description
    ai_text = generate_aura_description(result_data["metrics"])
    reading, tips = parse_ai_response(ai_text)

    result_data["reading"] = reading
    result_data["tips"]    = tips

    return jsonify(result_data)

@app.route('/health')
def health():
    return jsonify({"status": "ok"})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
