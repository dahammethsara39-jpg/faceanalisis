from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os
from aura_engine import analyse
from cerebras.cloud.sdk import Cerebras

app = Flask(__name__)
CORS(app)
BASE = os.path.dirname(os.path.abspath(__file__))

# Cerebras client
cerebras_client = Cerebras(api_key=os.environ.get("CEREBRAS_API_KEY"))

def generate_aura_description(scores):
    """Ask Cerebras AI to write an honest, human-like aura reading."""
    symmetry = scores["Face Symmetry"]
    glow     = scores["Skin Glow"]
    eyes     = scores["Eye Intensity"]
    jaw      = scores["Jawline"]
    total    = (symmetry + glow + eyes + jaw) / 4

    prompt = f"""You are an honest and direct face analyst. Based on these facial scores, write a SHORT 3-sentence honest description and 2 specific tips to improve the score. Be real, friendly, not harsh. No fluff.

Scores:
- Face Symmetry: {symmetry}/100
- Skin Glow: {glow}/100  
- Eye Intensity: {eyes}/100
- Jawline Definition: {jaw}/100
- Overall Aura: {total:.1f}/100

Format your response EXACTLY like this:
READING: [2-3 honest sentences about their face based on the scores]
TIPS: [2 specific actionable tips to improve their aura score]"""

    try:
        response = cerebras_client.chat.completions.create(
            model="llama3.1-8b",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=200,
        )
        return response.choices[0].message.content
    except Exception as e:
        return "READING: Your aura has been measured across four key dimensions.\nTIPS: Try shooting in natural daylight with a neutral expression for a higher score."

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

    # Parse into reading + tips
    reading = ""
    tips    = ""
    for line in ai_text.split("\n"):
        if line.startswith("READING:"):
            reading = line.replace("READING:", "").strip()
        elif line.startswith("TIPS:"):
            tips = line.replace("TIPS:", "").strip()

    result_data["reading"] = reading
    result_data["tips"]    = tips

    return jsonify(result_data)

@app.route('/health')
def health():
    return jsonify({"status": "ok"})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
