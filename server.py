"""
GYM PARTNER — Serveur Flask
============================
Lance le serveur :
    python server.py

Puis ouvre : http://localhost:5000
"""

import os
import json
import tempfile
import threading
from flask import Flask, request, jsonify, send_from_directory

# Import du moteur d'analyse
from engine import analyze_video
from airtable import log_analysis, get_history

app = Flask(__name__, static_folder='.')

# ── CORS manuel (pas besoin de flask-cors) ──────────
@app.after_request
def add_cors(response):
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type'
    response.headers['Access-Control-Allow-Methods'] = 'POST, OPTIONS'
    return response

# ── Servir l'interface HTML ─────────────────────────
@app.route('/')
def index():
    return send_from_directory('.', 'index.html')

# ── Endpoint d'analyse vidéo ────────────────────────
@app.route('/analyze', methods=['POST', 'OPTIONS'])
def analyze():
    if request.method == 'OPTIONS':
        return '', 204

    # Vérifier qu'une vidéo est bien envoyée
    if 'video' not in request.files:
        return jsonify({'error': 'Aucune vidéo reçue'}), 400

    video_file = request.files['video']

    if video_file.filename == '':
        return jsonify({'error': 'Fichier vide'}), 400

    exercise = request.form.get('exercise', 'squat')

    # Sauvegarder dans un fichier temporaire
    suffix = os.path.splitext(video_file.filename)[1] or '.mp4'
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        video_file.save(tmp.name)
        tmp_path = tmp.name

    try:
        # Lancer l'analyse
        result = analyze_video(tmp_path, exercise=exercise)

        # Logger dans Airtable (fire-and-forget, ne bloque pas la réponse)
        if "error" not in result:
            threading.Thread(target=log_analysis, args=(result,), daemon=True).start()

        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    finally:
        # Nettoyer le fichier temporaire
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


# ── Historique des analyses ─────────────────────────
@app.route('/history', methods=['GET'])
def history():
    return jsonify(get_history(limit=50))


if __name__ == '__main__':
    print("\n" + "="*45)
    print("  🏋️  GYM PARTNER — Serveur démarré")
    print("="*45)
    print("  → Ouvre : http://localhost:5000")
    print("="*45 + "\n")
    app.run(debug=True, port=8080, host='0.0.0.0')
