"""
GYM PARTNER — Analyseur de Squat (wrapper de compatibilité)
=============================================================
Ce fichier est conservé pour la compatibilité CLI.
La logique a été déplacée dans engine.py + exercises/squat.py.

Usage :
    python analyze_squat.py --video mon_squat.mp4
    python engine.py --video mon_squat.mp4 --exercise squat
"""
# Réexport pour compatibilité
from engine import analyze_video  # noqa: F401

import cv2
import numpy as np
import argparse
import json
import os
import urllib.request
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision


# ─────────────────────────────────────────
# 1. TÉLÉCHARGEMENT DU MODÈLE
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
# 2. CALCUL DES ANGLES
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
    """Inclinaison du dos par rapport à la verticale.
    0° = dos droit, augmente avec le penchement vers l'avant."""
    spine = np.array([shoulder[0] - hip[0], shoulder[1] - hip[1]], dtype=float)
    norm = np.linalg.norm(spine)
    if norm < 1e-6:
        return 0.0
    vertical_up = np.array([0.0, -1.0])
    cosine = np.clip(np.dot(spine / norm, vertical_up), -1.0, 1.0)
    return round(float(np.degrees(np.arccos(cosine))), 1)


# ─────────────────────────────────────────
# 3. STRUCTURES DE DONNÉES
# ─────────────────────────────────────────

@dataclass
class FrameData:
    """Angles mesurés sur une frame."""
    knee_angle: float
    hip_angle:  float
    back_angle: float


@dataclass
class Recommendation:
    niveau: str      # 'critique' | 'avertissement' | 'conseil'
    message: str
    partie: str      # 'genou' | 'hanche' | 'dos' | 'global'
    risque: str = ""
    correction: str = ""


SEVERITY_ORDER = {'critique': 0, 'avertissement': 1, 'conseil': 2}


# ─────────────────────────────────────────
# 4. RÈGLES BIOMÉCANIQUES DU SQUAT
# ─────────────────────────────────────────

def evaluate_squat(frames: List[FrameData], bottom_idx: int) -> Tuple[int, List[Recommendation]]:
    """
    Analyse biomécanique complète : point bas + dynamique du mouvement.
    Retourne (score/100, recommandations triées par sévérité).
    """
    recs: List[Recommendation] = []
    score = 100

    bottom  = frames[bottom_idx]
    descent = frames[:bottom_idx]
    ascent  = frames[bottom_idx + 1:]

    # ── A. POINT BAS ──────────────────────────────────────────────

    # Profondeur (angle genou)
    if bottom.knee_angle > 110:
        recs.append(Recommendation('critique',
            "❌ Squat pas assez profond",
            'genou',
            risque="Sous-solicitation des quadriceps et fessiers, report de charge sur les lombaires.",
            correction="Descends jusqu'à ce que les cuisses soient parallèles au sol. Travaille ta mobilité de cheville si tu bloques."))
        score -= 30
    elif bottom.knee_angle < 60:
        recs.append(Recommendation('avertissement',
            "⚠️ Squat trop profond",
            'genou',
            risque="Contrainte excessive sur les ligaments du genou et cisaillement rotulien à hautes charges.",
            correction="Remonte légèrement pour viser un angle de genou entre 70° et 100°."))
        score -= 10
    else:
        recs.append(Recommendation('conseil', "✅ Bonne profondeur de squat", 'genou'))

    # Dos au point bas
    if bottom.back_angle > 50:
        recs.append(Recommendation('critique',
            "❌ Dos trop penché en avant",
            'dos',
            risque="Risque de hernie discale et de douleurs lombaires chroniques, surtout sous charge lourde.",
            correction="Garde la poitrine haute, regarde un point fixe devant toi. Renforce le gainage du tronc."))
        score -= 25
    elif bottom.back_angle > 35:
        recs.append(Recommendation('avertissement',
            "⚠️ Légère inclinaison du dos",
            'dos',
            risque="Fatigue lombaire prématurée qui peut s'accentuer à la fatigue ou avec plus de charge.",
            correction="Pense à 'sortir la poitrine' en descendant. Inspire profondément avant chaque répétition."))
        score -= 10
    else:
        recs.append(Recommendation('conseil', "✅ Bonne position du dos", 'dos'))

    # Hanches au point bas + détection butt wink
    butt_wink = bottom.knee_angle < 85 and bottom.hip_angle < 55
    if butt_wink:
        recs.append(Recommendation('avertissement',
            "⚠️ Rétroversion du bassin en fond de squat (butt wink)",
            'hanche',
            risque="Compression discale lombaire en position de flexion maximale, surtout problématique sous charge.",
            correction="Travaille la mobilité de cheville (élévation sur planche) et de hanche. Essaie un talon légèrement surélevé."))
        score -= 15
    elif bottom.hip_angle < 55:
        recs.append(Recommendation('avertissement',
            "⚠️ Hanches trop fermées au point bas",
            'hanche',
            risque="Compensation par une antéversion du bassin, tension excessive sur les adducteurs.",
            correction="Écarte légèrement les pieds et oriente les orteils vers l'extérieur (30–45°). Travaille la mobilité de hanche."))
        score -= 15
    elif bottom.hip_angle > 110:
        recs.append(Recommendation('avertissement',
            "⚠️ Hanches trop ouvertes au point bas",
            'hanche',
            risque="Instabilité du bassin et tension excessive sur les abducteurs.",
            correction="Resserre légèrement l'écart des pieds pour retrouver un alignement optimal."))
        score -= 10

    # ── B. ANALYSE TEMPORELLE ──────────────────────────────────────

    # B1. Hip shoot-up à la remontée (hanches qui montent avant le torse)
    if len(ascent) >= 3:
        early = ascent[:max(1, len(ascent) // 3)]
        avg_back_early = float(np.mean([f.back_angle for f in early]))
        if avg_back_early - bottom.back_angle > 15:
            recs.append(Recommendation('critique',
                "❌ Les hanches montent avant le torse à la remontée",
                'dos',
                risque="Cisaillement lombaire brutal sous charge — l'un des patterns les plus dangereux du squat.",
                correction="Pense 'genoux et hanches remontent en même temps'. Travaille des pause squats pour renforcer le point bas."))
            score -= 20

    # B2. Effondrement progressif du torse pendant la descente
    if len(descent) >= 4:
        quarter = max(1, len(descent) // 4)
        first_back = float(np.mean([f.back_angle for f in descent[:quarter]]))
        last_back  = float(np.mean([f.back_angle for f in descent[-quarter:]]))
        if last_back - first_back > 20:
            recs.append(Recommendation('avertissement',
                "⚠️ Le torse s'incline progressivement pendant la descente",
                'dos',
                risque="Surcharge des érecteurs du rachis et perte de tension qui fragilise le bas du dos.",
                correction="Inspire et engage les abdominaux avant de commencer la descente (technique Valsalva)."))
            score -= 10

    # Trier critique → avertissement → conseil
    recs.sort(key=lambda r: SEVERITY_ORDER[r.niveau])

    return max(0, score), recs


# ─────────────────────────────────────────
# 5. ANALYSE DE LA VIDÉO
# ─────────────────────────────────────────

def analyze_video(video_path: str) -> dict:
    ensure_model()

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return {"error": f"Impossible d'ouvrir la vidéo : {video_path}"}

    raw_frames = []
    count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        count += 1
        if count % 3 == 0:
            raw_frames.append(frame)
    cap.release()

    if not raw_frames:
        return {"error": "Vidéo vide ou illisible."}

    options = mp_vision.PoseLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=mp_vision.RunningMode.IMAGE,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    # Indices MediaPipe Tasks API (gauche / droite)
    # shoulder=11/12  hip=23/24  knee=25/26  ankle=27/28
    SIDES = {
        'left':  [11, 23, 25, 27],
        'right': [12, 24, 26, 28],
    }

    def best_side(lm, w, h):
        """Retourne (shoulder, hip, knee, ankle, min_visibility) du côté le plus visible."""
        vis = {side: min(lm[i].visibility for i in idxs)
               for side, idxs in SIDES.items()}
        chosen = max(vis, key=vis.get)
        idxs = SIDES[chosen]
        pts = [[lm[i].x * w, lm[i].y * h] for i in idxs]
        return pts[0], pts[1], pts[2], pts[3], vis[chosen]

    frames: List[FrameData] = []
    analyzed = 0

    with mp_vision.PoseLandmarker.create_from_options(options) as landmarker:
        for raw in raw_frames:
            h, w = raw.shape[:2]
            rgb = cv2.cvtColor(raw, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = landmarker.detect(mp_image)

            if not result.pose_landmarks:
                continue

            lm = result.pose_landmarks[0]
            shoulder, hip, knee, ankle, vis = best_side(lm, w, h)

            if vis < MIN_VIS:
                continue

            analyzed += 1

            knee_angle = calculate_angle(hip, knee, ankle)
            hip_angle  = calculate_angle(shoulder, hip, knee)
            back_angle = calculate_back_angle(hip, shoulder)

            # Validation anatomique
            if not (20.0 <= knee_angle <= 180.0):
                continue
            if not (20.0 <= hip_angle <= 180.0):
                continue
            if not (0.0 <= back_angle <= 85.0):
                continue

            frames.append(FrameData(knee_angle, hip_angle, back_angle))

    if not frames:
        if analyzed == 0:
            return {"error": "Aucune pose détectée. Assure-toi que le corps est entièrement visible et bien éclairé."}
        return {"error": "Les angles calculés semblent incorrects. Filme-toi de profil, corps entier visible, avec une bonne luminosité."}

    # Trouver le point bas (angle genou minimal)
    bottom_idx = min(range(len(frames)), key=lambda i: frames[i].knee_angle)
    bottom = frames[bottom_idx]

    score, recs = evaluate_squat(frames, bottom_idx)

    return {
        "frames_analyzed": analyzed,
        "angles": {
            "genou":  bottom.knee_angle,
            "hanche": bottom.hip_angle,
            "dos":    bottom.back_angle,
        },
        "score": score,
        "recommandations": [
            {"niveau": r.niveau, "message": r.message, "partie": r.partie,
             "risque": r.risque, "correction": r.correction}
            for r in recs
        ],
    }


# ─────────────────────────────────────────
# 6. AFFICHAGE TERMINAL
# ─────────────────────────────────────────

def print_report(result: dict):
    print("\n" + "=" * 50)
    print("       🏋️  GYM PARTNER — Analyse Squat")
    print("=" * 50)

    if "error" in result:
        print(f"\n❌ Erreur : {result['error']}")
        return

    print(f"\n📊 Frames analysées : {result['frames_analyzed']}")
    print(f"\n📐 Angles au point bas :")
    print(f"   • Genou  : {result['angles']['genou']}°  (idéal : 70°–100°)")
    print(f"   • Hanche : {result['angles']['hanche']}°  (idéal : 55°–110°)")
    print(f"   • Dos    : {result['angles']['dos']}°   (idéal : < 35°)")
    print(f"\n🎯 Score global : {result['score']} / 100")
    print("\n💬 Recommandations :")
    for rec in result['recommandations']:
        niveau = rec['niveau'].upper()
        print(f"   [{niveau}] {rec['message']}")
    print("\n" + "=" * 50 + "\n")


# ─────────────────────────────────────────
# 7. POINT D'ENTRÉE
# ─────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GYM PARTNER — Analyseur de squat")
    parser.add_argument("--video", required=True, help="Chemin vers la vidéo MP4")
    parser.add_argument("--json",  action="store_true", help="Sortie JSON")
    args = parser.parse_args()

    result = analyze_video(args.video)

    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print_report(result)
