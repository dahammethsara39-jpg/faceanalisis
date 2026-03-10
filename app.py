from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os
from aura_engine import analyse

app = Flask(__name__)
CORS(app)

BASE = os.path.dirname(os.path.abspath(__file__))

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
    return jsonify(result_data)

@app.route('/health')
def health():
    return jsonify({"status": "ok"})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
