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
            "❌ Tu ne descends pas assez bas",
            'genou',
            risque="En restant trop haut, les cuisses et les fesses ne travaillent presque pas.",
            correction="Descends jusqu'à ce que tes cuisses soient parallèles au sol (comme si tu t'asseyais sur une chaise basse). Si tu bloques, essaie d'écarter un peu les pieds."))
        score -= 30
    elif bottom.knee_angle < 60:
        recs.append(Recommendation('avertissement',
            "⚠️ Tu descends trop bas",
            'genou',
            risque="Aller trop bas peut mettre trop de pression sur les genoux, surtout avec du poids.",
            correction="Arrête-toi quand tes cuisses sont parallèles au sol. Ce n'est pas nécessaire d'aller plus bas."))
        score -= 10
    else:
        recs.append(Recommendation('conseil', "✅ Bonne profondeur de squat", 'genou'))

    # Dos au point bas
    if bottom.back_angle > 50:
        recs.append(Recommendation('critique',
            "❌ Ton dos est trop penché en avant",
            'dos',
            risque="Pencher trop le dos en avant met une forte pression sur le bas du dos, surtout avec du poids.",
            correction="Garde la poitrine haute et le regard droit devant toi. Imagine que tu as un verre d'eau posé sur la poitrine que tu ne veux pas renverser."))
        score -= 25
    elif bottom.back_angle > 35:
        recs.append(Recommendation('avertissement',
            "⚠️ Ton dos s'incline un peu trop vers l'avant",
            'dos',
            risque="Avec plus de poids ou à la fatigue, cette position peut fatiguer le bas du dos.",
            correction="Avant de descendre, prends une grande inspiration, gonfle le ventre et garde cette tension. Ça va naturellement redresser le dos."))
        score -= 10
    else:
        recs.append(Recommendation('conseil', "✅ Belle position du dos", 'dos'))

    # Hanches au point bas + détection butt wink
    butt_wink = bottom.knee_angle < 85 and bottom.hip_angle < 55
    if butt_wink:
        recs.append(Recommendation('avertissement',
            "⚠️ Le bas de ton dos s'arrondit tout en bas du squat",
            'hanche',
            risque="Le bas du dos qui s'arrondit sous charge peut provoquer des douleurs lombaires.",
            correction="Essaie de surélever légèrement tes talons (avec des petites plaques ou des chaussures à talon). Ça aide beaucoup si tu as peu de souplesse des chevilles."))
        score -= 15
    elif bottom.hip_angle < 55:
        recs.append(Recommendation('avertissement',
            "⚠️ Tes hanches ne s'ouvrent pas assez",
            'hanche',
            risque="Les hanches trop fermées limitent la profondeur et peuvent causer des douleurs à l'aine.",
            correction="Écarte un peu plus les pieds et tourne les orteils vers l'extérieur (environ 30°). Teste différentes largeurs pour trouver ta position confortable."))
        score -= 15
    elif bottom.hip_angle > 110:
        recs.append(Recommendation('avertissement',
            "⚠️ Tes pieds sont trop écartés",
            'hanche',
            risque="Un écart trop grand peut rendre le mouvement instable.",
            correction="Rapproche un peu les pieds pour trouver ta largeur naturelle — généralement un peu plus large que les épaules."))
        score -= 10

    # ── B. ANALYSE TEMPORELLE ──────────────────────────────────────

    # B1. Hip shoot-up à la remontée (hanches qui montent avant le torse)
    if len(ascent) >= 3:
        early = ascent[:max(1, len(ascent) // 3)]
        avg_back_early = float(np.mean([f.back_angle for f in early]))
        if avg_back_early - bottom.back_angle > 15:
            recs.append(Recommendation('critique',
                "❌ Tes hanches remontent avant ton dos à la montée",
                'dos',
                risque="C'est l'une des erreurs les plus dangereuses du squat — ça crée une forte pression sur le bas du dos.",
                correction="Pense à pousser avec les deux jambes en même temps et à garder le dos droit tout au long de la montée. Imagine que tu veux toucher le plafond avec le haut du crâne."))
            score -= 20

    # B2. Effondrement progressif du torse pendant la descente
    if len(descent) >= 4:
        quarter = max(1, len(descent) // 4)
        first_back = float(np.mean([f.back_angle for f in descent[:quarter]]))
        last_back  = float(np.mean([f.back_angle for f in descent[-quarter:]]))
        if last_back - first_back > 20:
            recs.append(Recommendation('avertissement',
                "⚠️ Ton dos se penche de plus en plus pendant la descente",
                'dos',
                risque="Les muscles du dos se fatiguent en cours de descente, ce qui fragilise le bas du dos.",
                correction="Avant de descendre, prends une grande inspiration et gonfle le ventre comme un ballon. Garde cette tension tout au long de la répétition."))
            score -= 10

    recs.sort(key=lambda r: SEVERITY_ORDER[r.niveau])
    return max(0, score), recs
