"""
app.py  –  Flask backend for the Aura Checker
──────────────────────────────────────────────
Routes:
  GET  /             → serves index.html
  POST /analyze      → accepts image, returns JSON aura scores
  GET  /result       → serves result.html
"""

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os
from aura_engine import analyse

app = Flask(
    __name__,
    static_folder=os.path.join(os.path.dirname(__file__), '..', 'frontend'),
    static_url_path=''
)
CORS(app)   # allow frontend on any origin to call /analyze

# ── Serve frontend pages ─────────────────────────────────────────────────────
@app.route('/')
def index():
    return send_from_directory(app.static_folder, 'index.html')

@app.route('/result')
def result():
    return send_from_directory(app.static_folder, 'result.html')

# ── Analysis endpoint ────────────────────────────────────────────────────────
@app.route('/analyze', methods=['POST'])
def analyze():
    if 'image' not in request.files:
        return jsonify({"error": "No image file uploaded."}), 400

    file  = request.files['image']
    if file.filename == '':
        return jsonify({"error": "Empty filename."}), 400

    # Read raw bytes and pass to engine
    image_bytes = file.read()
    result_data = analyse(image_bytes)

    if "error" in result_data:
        return jsonify(result_data), 422

    return jsonify(result_data)

# ── Health check (Render uses this) ─────────────────────────────────────────
@app.route('/health')
def health():
    return jsonify({"status": "ok"})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
