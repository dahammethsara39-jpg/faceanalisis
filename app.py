from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os
import requests as req
from aura_engine import analyse

app = Flask(__name__)
CORS(app)
BASE = os.path.dirname(os.path.abspath(__file__))

def generate_aura_description(scores):
    api_key = os.environ.get("CEREBRAS_API_KEY", "")
    
    # Debug: print if key exists
    print(f"[DEBUG] API key found: {'YES' if api_key else 'NO'}")
    
    if not api_key:
        print("[DEBUG] No API key - skipping AI")
        return None

    symmetry = scores["Face Symmetry"]
    glow     = scores["Skin Glow"]
    eyes     = scores["Eye Intensity"]
    jaw      = scores["Jawline"]

    prompt = f"""You are a brutally honest but kind face analyst. Based on these real facial measurement scores, write 2-3 sentences about what the scores reveal about this person's actual face. Then give 2 specific tips to improve. Talk about the FACE FEATURES only.

Face Symmetry: {symmetry}/100
Skin Glow: {glow}/100
Eye Intensity: {eyes}/100
Jawline: {jaw}/100

Reply EXACTLY like this:
READING: [your honest reading here]
TIPS: [tip 1]. [tip 2]."""

    try:
        print("[DEBUG] Calling Cerebras API...")
        response = req.post(
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
            timeout=20
        )
        print(f"[DEBUG] Response status: {response.status_code}")
        print(f"[DEBUG] Response body: {response.text[:300]}")
        
        if response.status_code == 200:
            data = response.json()
            return data["choices"][0]["message"]["content"]
        else:
            print(f"[DEBUG] API error: {response.text}")
            return None
    except Exception as e:
        print(f"[DEBUG] Exception: {str(e)}")
        return None

def parse_ai_response(text):
    reading, tips = "", ""
    if not text:
        return reading, tips
    for line in text.strip().split("\n"):
        line = line.strip()
        if line.startswith("READING:"):
            reading = line.replace("READING:", "").strip()
        elif line.startswith("TIPS:"):
            tips = line.replace("TIPS:", "").strip()
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
