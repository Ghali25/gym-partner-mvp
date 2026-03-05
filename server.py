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


# ── Endpoint analyse combinée (profil + face) ───────
@app.route('/analyze-combined', methods=['POST', 'OPTIONS'])
def analyze_combined():
    if request.method == 'OPTIONS':
        return '', 204

    if 'video_profil' not in request.files or 'video_face' not in request.files:
        return jsonify({'error': 'Deux vidéos requises (video_profil et video_face)'}), 400

    profil_file = request.files['video_profil']
    face_file   = request.files['video_face']

    if profil_file.filename == '' or face_file.filename == '':
        return jsonify({'error': 'Fichiers vides'}), 400

    suffix_p = os.path.splitext(profil_file.filename)[1] or '.mp4'
    suffix_f = os.path.splitext(face_file.filename)[1] or '.mp4'
    tmp_profil = tmp_face = None

    try:
        with tempfile.NamedTemporaryFile(suffix=suffix_p, delete=False) as tp:
            profil_file.save(tp.name)
            tmp_profil = tp.name
        with tempfile.NamedTemporaryFile(suffix=suffix_f, delete=False) as tf:
            face_file.save(tf.name)
            tmp_face = tf.name

        result_profil = analyze_video(tmp_profil, exercise='squat')
        result_face   = analyze_video(tmp_face,   exercise='squat_face')

        if 'error' in result_profil:
            return jsonify({'error': f"Vue de profil — {result_profil['error']}"}), 400
        if 'error' in result_face:
            return jsonify({'error': f"Vue de face — {result_face['error']}"}), 400

        # Fusion : score pondéré (profil 55 % + face 45 %)
        score_combined = round(result_profil['score'] * 0.55 + result_face['score'] * 0.45)
        angles_combined = {**result_profil['angles'], **result_face['angles']}
        recos_combined  = sorted(
            result_profil['recommandations'] + result_face['recommandations'],
            key=lambda r: {'critique': 0, 'avertissement': 1, 'conseil': 2}.get(r['niveau'], 3)
        )

        combined = {
            'exercise':        'squat_combined',
            'score':           score_combined,
            'score_profil':    result_profil['score'],
            'score_face':      result_face['score'],
            'angles':          angles_combined,
            'recommandations': recos_combined,
            'frames_analyzed': result_profil['frames_analyzed'] + result_face['frames_analyzed'],
        }

        threading.Thread(target=log_analysis, args=(combined,), daemon=True).start()
        return jsonify(combined)

    except Exception as e:
        return jsonify({'error': str(e)}), 500
    finally:
        for p in [tmp_profil, tmp_face]:
            if p and os.path.exists(p):
                os.remove(p)


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
