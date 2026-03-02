"""
exercises/squat.py — Analyse biomécanique du squat
====================================================
Interface attendue par engine.py :
    build_frame_data(lm, w, h) -> SquatFrameData | None
    find_bottom(frames)        -> int
    evaluate(frames, bottom_idx) -> (score, List[Recommendation])
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple
import numpy as np

from engine import calculate_angle, calculate_back_angle, Recommendation, SEVERITY_ORDER, MIN_VIS


# ─────────────────────────────────────────
# LANDMARKS squat (profil)
# shoulder=11/12  hip=23/24  knee=25/26  ankle=27/28
# ─────────────────────────────────────────

SIDES = {
    'left':  [11, 23, 25, 27],
    'right': [12, 24, 26, 28],
}


@dataclass
class SquatFrameData:
    knee_angle: float
    hip_angle:  float
    back_angle: float

    def to_angles_dict(self) -> dict:
        return {
            "genou":  self.knee_angle,
            "hanche": self.hip_angle,
            "dos":    self.back_angle,
        }


def _best_side(lm, w, h):
    """Retourne (shoulder, hip, knee, ankle, min_visibility) du côté le plus visible."""
    vis = {side: min(lm[i].visibility for i in idxs)
           for side, idxs in SIDES.items()}
    chosen = max(vis, key=vis.get)
    idxs = SIDES[chosen]
    pts = [[lm[i].x * w, lm[i].y * h] for i in idxs]
    return pts[0], pts[1], pts[2], pts[3], vis[chosen]


def build_frame_data(lm, w: int, h: int) -> Optional[SquatFrameData]:
    shoulder, hip, knee, ankle, vis = _best_side(lm, w, h)
    if vis < MIN_VIS:
        return None

    knee_angle = calculate_angle(hip, knee, ankle)
    hip_angle  = calculate_angle(shoulder, hip, knee)
    back_angle = calculate_back_angle(hip, shoulder)

    if not (20.0 <= knee_angle <= 180.0):
        return None
    if not (20.0 <= hip_angle <= 180.0):
        return None
    if not (0.0 <= back_angle <= 85.0):
        return None

    return SquatFrameData(knee_angle, hip_angle, back_angle)


def find_bottom(frames: List[SquatFrameData]) -> int:
    """Point bas = frame avec l'angle de genou minimal (position la plus profonde)."""
    return min(range(len(frames)), key=lambda i: frames[i].knee_angle)


def validate(frames: List[SquatFrameData]):
    """
    Vérifie que la vidéo correspond bien à un squat debout.
    Retourne (error_type, message) si problème, None sinon.
    """
    avg_back = float(np.mean([f.back_angle for f in frames]))
    min_knee = min(f.knee_angle for f in frames)

    # Personne allongée (bench press filmé en mode squat) → dos proche de l'horizontale
    if avg_back > 60:
        return ("mismatch",
                "Position horizontale détectée — cette vidéo ne ressemble pas à un squat debout. "
                "Sélectionne l'exercice 'Bench' ou filme-toi debout de profil pour analyser un squat.")

    # Aucune flexion de genou → pas de squat
    if min_knee > 155:
        return ("mismatch",
                "Aucune flexion significative du genou détectée. "
                "Assure-toi que la descente complète du squat est visible dans la vidéo (au moins 3-5 répétitions).")

    return None


def evaluate(frames: List[SquatFrameData], bottom_idx: int) -> Tuple[int, List[Recommendation]]:
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

    recs.sort(key=lambda r: SEVERITY_ORDER[r.niveau])
    return max(0, score), recs
