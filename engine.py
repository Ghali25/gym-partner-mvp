"""
GYM PARTNER — Moteur d'analyse générique
==========================================
Expose : analyze_video(video_path, exercise='squat') -> dict

Usage CLI :
    python engine.py --video mon_squat.mp4
    python engine.py --video mon_bench.mp4 --exercise bench
"""

import cv2
import numpy as np
import argparse
import json
import os
import urllib.request
from dataclasses import dataclass
from typing import List

import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision


# ─────────────────────────────────────────
# 1. MODÈLE MEDIAPIPE
# ─────────────────────────────────────────

MODEL_URL  = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task"
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pose_landmarker.task")

def ensure_model():
    """Télécharge le modèle MediaPipe si absent (une seule fois)."""
    if os.path.exists(MODEL_PATH):
        return
    print("📥 Téléchargement du modèle MediaPipe (une seule fois)...")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    print("✅ Modèle téléchargé.")


# ─────────────────────────────────────────
# 2. CALCUL DES ANGLES (générique)
# ─────────────────────────────────────────

MIN_VIS = 0.5  # Seuil minimal de visibilité MediaPipe (0–1)

def calculate_angle(a, b, c) -> float:
    """Angle en degrés entre 3 points (b = sommet) via produit scalaire.
    Toujours dans [0°, 180°] — plus robuste que arctan2."""
    a = np.array(a, dtype=float)
    b = np.array(b, dtype=float)
    c = np.array(c, dtype=float)
    ba, bc = a - b, c - b
    denom = np.linalg.norm(ba) * np.linalg.norm(bc)
    if denom < 1e-6:
        return 0.0
    cosine = np.clip(np.dot(ba, bc) / denom, -1.0, 1.0)
    return round(float(np.degrees(np.arccos(cosine))), 1)


def calculate_back_angle(hip, shoulder) -> float:
    """Inclinaison du dos par rapport à la verticale en 3D (world landmarks).
    0° = dos parfaitement droit, augmente avec le penchement vers l'avant.
    En world coords MediaPipe, Y pointe vers le haut."""
    spine = np.array([shoulder[0] - hip[0],
                      shoulder[1] - hip[1],
                      shoulder[2] - hip[2] if len(shoulder) > 2 else 0.0], dtype=float)
    norm = np.linalg.norm(spine)
    if norm < 1e-6:
        return 0.0
    vertical_up = np.array([0.0, 1.0, 0.0])   # Y est vers le haut en world coords
    cosine = np.clip(np.dot(spine / norm, vertical_up), -1.0, 1.0)
    return round(float(np.degrees(np.arccos(cosine))), 1)


# ─────────────────────────────────────────
# 3. STRUCTURE RECOMMANDATION (générique)
# ─────────────────────────────────────────

@dataclass
class Recommendation:
    niveau: str      # 'critique' | 'avertissement' | 'conseil'
    message: str
    partie: str      # zone du corps — varie selon l'exercice
    risque: str = ""
    correction: str = ""


SEVERITY_ORDER = {'critique': 0, 'avertissement': 1, 'conseil': 2}


# ─────────────────────────────────────────
# 4. PIPELINE MEDIAPIPE (générique)
# ─────────────────────────────────────────

def _build_landmarker_options():
    return mp_vision.PoseLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=mp_vision.RunningMode.IMAGE,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )


def _extract_raw_frames(video_path: str, stride: int = 3) -> list:
    """Lit la vidéo et échantillonne 1 frame sur `stride`. Retourne None si erreur."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
    raw = []
    count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        count += 1
        if count % stride == 0:
            raw.append(frame)
    cap.release()
    return raw


# ─────────────────────────────────────────
# 5. POINT D'ENTRÉE PUBLIC
# ─────────────────────────────────────────

EXERCISES = ('squat', 'bench', 'squat_face')

def analyze_video(video_path: str, exercise: str = 'squat') -> dict:
    """
    Analyse une vidéo pour l'exercice donné.
    Retourne un dict JSON-sérialisable.
    """
    if exercise not in EXERCISES:
        return {"error": f"Exercice inconnu : '{exercise}'. Exercices disponibles : {', '.join(EXERCISES)}"}

    ensure_model()

    raw_frames = _extract_raw_frames(video_path)
    if raw_frames is None:
        return {"error": f"Impossible d'ouvrir la vidéo : {video_path}"}
    if not raw_frames:
        return {"error": "Vidéo vide ou illisible."}

    # Import dynamique du module exercice
    if exercise == 'squat':
        from exercises.squat import build_frame_data, find_bottom, evaluate, validate
    elif exercise == 'bench':
        from exercises.bench import build_frame_data, find_bottom, evaluate, validate
    elif exercise == 'squat_face':
        from exercises.squat_face import build_frame_data, find_bottom, evaluate, validate

    options = _build_landmarker_options()
    frames = []
    analyzed = 0

    with mp_vision.PoseLandmarker.create_from_options(options) as landmarker:
        for raw in raw_frames:
            h, w = raw.shape[:2]
            rgb = cv2.cvtColor(raw, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = landmarker.detect(mp_image)

            if not result.pose_landmarks:
                continue

            analyzed += 1

            # ── Landmarks 3D (world coords en mètres, origine = centre des hanches)
            # Préférés aux 2D car insensibles à l'angle et à la distance de caméra.
            # Fallback sur les 2D normalisés si indisponibles (cas rare).
            if result.pose_world_landmarks and len(result.pose_world_landmarks) > 0:
                lm = result.pose_world_landmarks[0]
                frame = build_frame_data(lm)
            else:
                lm = result.pose_landmarks[0]
                frame = build_frame_data(lm)
            if frame is not None:
                frames.append(frame)

    if not frames:
        if analyzed == 0:
            return {"error": "Aucune pose détectée. Assure-toi que le corps est entièrement visible et bien éclairé."}
        return {"error": "Les angles calculés semblent incorrects. Filme-toi de profil, corps entier visible, avec une bonne luminosité."}

    # Vérifier que la vidéo correspond à l'exercice sélectionné
    mismatch = validate(frames)
    if mismatch:
        error_type, message = mismatch
        return {"error": message, "error_type": error_type, "exercise": exercise}

    bottom_idx = find_bottom(frames)
    score, recs = evaluate(frames, bottom_idx)

    return {
        "exercise": exercise,
        "frames_analyzed": analyzed,
        "angles": frames[bottom_idx].to_angles_dict(),
        "score": score,
        "recommandations": [
            {"niveau": r.niveau, "message": r.message, "partie": r.partie,
             "risque": r.risque, "correction": r.correction}
            for r in recs
        ],
    }


# ─────────────────────────────────────────
# 6. POINT D'ENTRÉE CLI
# ─────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GYM PARTNER — Moteur d'analyse")
    parser.add_argument("--video",    required=True, help="Chemin vers la vidéo MP4")
    parser.add_argument("--exercise", default="squat", choices=list(EXERCISES), help="Exercice à analyser")
    parser.add_argument("--json",     action="store_true", help="Sortie JSON brute")
    args = parser.parse_args()

    result = analyze_video(args.video, exercise=args.exercise)

    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print(json.dumps(result, ensure_ascii=False, indent=2))
