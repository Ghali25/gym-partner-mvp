"""
exercises/squat_face.py — Analyse squat vue de face
====================================================
Détecte : knee valgus gauche/droit, asymétrie, largeur de stance.

Interface attendue par engine.py :
    build_frame_data(lm, w, h)   -> FaceSquatFrameData | None
    find_bottom(frames)          -> int
    validate(frames)             -> (error_type, message) | None
    evaluate(frames, bottom_idx) -> (score, List[Recommendation])
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple
import numpy as np

from engine import calculate_angle, Recommendation, SEVERITY_ORDER, MIN_VIS


# ─────────────────────────────────────────
# LANDMARKS vue de face
#   left:  hip=23  knee=25  ankle=27  shoulder=11
#   right: hip=24  knee=26  ankle=28  shoulder=12
# ─────────────────────────────────────────

LEFT_IDS  = [23, 25, 27]
RIGHT_IDS = [24, 26, 28]


@dataclass
class FaceSquatFrameData:
    valgus_g:  float   # angle hip-knee-ankle genou gauche (° — plus petit = valgus)
    valgus_d:  float   # angle hip-knee-ankle genou droit
    asymetrie: float   # |valgus_g - valgus_d|
    largeur:   float   # ratio distance-chevilles / distance-hanches

    def to_angles_dict(self) -> dict:
        return {
            "valgus_g":  self.valgus_g,
            "valgus_d":  self.valgus_d,
            "asymetrie": self.asymetrie,
        }


def build_frame_data(lm, w: int, h: int) -> Optional[FaceSquatFrameData]:
    # Les deux côtés doivent être visibles
    vis_l = min(lm[i].visibility for i in LEFT_IDS)
    vis_r = min(lm[i].visibility for i in RIGHT_IDS)
    if vis_l < MIN_VIS or vis_r < MIN_VIS:
        return None

    # Points en pixels
    left_hip    = [lm[23].x * w, lm[23].y * h]
    left_knee   = [lm[25].x * w, lm[25].y * h]
    left_ankle  = [lm[27].x * w, lm[27].y * h]
    right_hip   = [lm[24].x * w, lm[24].y * h]
    right_knee  = [lm[26].x * w, lm[26].y * h]
    right_ankle = [lm[28].x * w, lm[28].y * h]

    # Angle valgus : hip → knee → ankle dans le plan frontal
    valgus_g = calculate_angle(left_hip,  left_knee,  left_ankle)
    valgus_d = calculate_angle(right_hip, right_knee, right_ankle)

    if not (100.0 <= valgus_g <= 180.0):
        return None
    if not (100.0 <= valgus_d <= 180.0):
        return None

    asymetrie = round(abs(valgus_g - valgus_d), 1)

    # Largeur de stance : ratio chevilles / hanches (coords normalisées)
    hip_width   = abs(lm[24].x - lm[23].x)
    ankle_width = abs(lm[28].x - lm[27].x)
    largeur = round(ankle_width / hip_width, 2) if hip_width > 0.01 else 1.0

    return FaceSquatFrameData(valgus_g, valgus_d, asymetrie, largeur)


def find_bottom(frames: List[FaceSquatFrameData]) -> int:
    """Point bas = frame où l'angle valgus moyen est minimal (genoux les plus fléchis)."""
    return min(range(len(frames)), key=lambda i: (frames[i].valgus_g + frames[i].valgus_d) / 2)


def validate(frames: List[FaceSquatFrameData]):
    """
    Vérifie que la vidéo est bien un squat vu de face.
    - Flexion de genou détectée (angle valgus moyen < 170°)
    - Asymétrie raisonnable (si > 40° c'est probablement une vue de profil)
    """
    min_avg = min((f.valgus_g + f.valgus_d) / 2 for f in frames)
    if min_avg > 168:
        return ("mismatch",
                "Aucune flexion de genou détectée en vue de face. "
                "Assure-toi que le squat est complet et que tu es bien face à la caméra.")

    avg_asym = float(np.mean([f.asymetrie for f in frames]))
    if avg_asym > 40:
        return ("mismatch",
                "Vue de profil détectée. Pour l'analyse de face, place-toi face à la caméra "
                "— les deux jambes doivent être visibles.")

    return None


def evaluate(frames: List[FaceSquatFrameData], bottom_idx: int) -> Tuple[int, List[Recommendation]]:
    """Analyse biomécanique vue de face : valgus, asymétrie, stance, évolution dynamique."""
    recs: List[Recommendation] = []
    score = 100
    bottom = frames[bottom_idx]

    # ── A. VALGUS GENOU GAUCHE ────────────────────────────────────
    if bottom.valgus_g < 155:
        recs.append(Recommendation('critique',
            "❌ Effondrement du genou gauche vers l'intérieur (valgus)",
            'genou',
            risque="Contrainte excessive sur le ligament croisé antérieur et le ménisque interne gauche.",
            correction="Active les fessiers et pousse activement le genou gauche vers l'extérieur "
                       "dans la descente. Un mini-band autour des genoux à l'entraînement aide à corriger ce pattern."))
        score -= 25
    elif bottom.valgus_g < 168:
        recs.append(Recommendation('avertissement',
            "⚠️ Léger valgus du genou gauche",
            'genou',
            risque="Fatigue articulaire progressive du genou gauche, risque accru sous charge.",
            correction="Concentre-toi sur l'alignement du genou gauche au-dessus du 2e orteil. "
                       "Renforce les abducteurs de hanche (clamshells, bandes élastiques)."))
        score -= 12

    # ── B. VALGUS GENOU DROIT ─────────────────────────────────────
    if bottom.valgus_d < 155:
        recs.append(Recommendation('critique',
            "❌ Effondrement du genou droit vers l'intérieur (valgus)",
            'genou',
            risque="Contrainte excessive sur le ligament croisé antérieur et le ménisque interne droit.",
            correction="Pousse activement le genou droit vers l'extérieur. "
                       "Renforce les abducteurs et les fessiers du côté droit."))
        score -= 25
    elif bottom.valgus_d < 168:
        recs.append(Recommendation('avertissement',
            "⚠️ Léger valgus du genou droit",
            'genou',
            risque="Fatigue articulaire progressive du genou droit, risque accru sous charge.",
            correction="Aligne le genou droit au-dessus du 2e orteil. "
                       "Travaille les abducteurs de hanche."))
        score -= 12

    if bottom.valgus_g >= 168 and bottom.valgus_d >= 168:
        recs.append(Recommendation('conseil',
            "✅ Bon alignement des genoux — pas de valgus détecté", 'genou'))

    # ── C. ASYMÉTRIE GAUCHE / DROITE ──────────────────────────────
    if bottom.asymetrie > 20:
        recs.append(Recommendation('critique',
            f"❌ Forte asymétrie gauche/droite ({bottom.asymetrie:.0f}°)",
            'global',
            risque="Surcharge unilatérale chronique — risque de blessure du côté le plus sollicité.",
            correction="Travaille chaque jambe séparément (fentes, Bulgarian split squat) "
                       "pour identifier et corriger le côté faible."))
        score -= 20
    elif bottom.asymetrie > 10:
        recs.append(Recommendation('avertissement',
            f"⚠️ Légère asymétrie gauche/droite ({bottom.asymetrie:.0f}°)",
            'global',
            risque="Compensation musculaire qui s'aggrave sous charge ou à la fatigue.",
            correction="Intègre des exercices unilatéraux à ton programme "
                       "pour équilibrer les deux côtés."))
        score -= 10
    else:
        recs.append(Recommendation('conseil',
            "✅ Bonne symétrie gauche/droite", 'global'))

    # ── D. LARGEUR DE STANCE ──────────────────────────────────────
    if bottom.largeur < 0.7:
        recs.append(Recommendation('avertissement',
            "⚠️ Écartement des pieds trop étroit",
            'hanche',
            risque="Limite la profondeur et augmente la contrainte sur les chevilles et genoux.",
            correction="Écarte les pieds à la largeur des épaules ou légèrement plus, "
                       "orteils légèrement tournés vers l'extérieur (20-30°)."))
        score -= 10
    elif bottom.largeur > 1.8:
        recs.append(Recommendation('avertissement',
            "⚠️ Écartement des pieds trop large",
            'hanche',
            risque="Tension excessive sur les adducteurs et instabilité du bassin.",
            correction="Rapproche légèrement les pieds pour trouver ta position naturelle."))
        score -= 8
    else:
        recs.append(Recommendation('conseil',
            "✅ Bon écartement des pieds", 'hanche'))

    # ── E. VALGUS DYNAMIQUE (effondrement progressif) ─────────────
    if len(frames) >= 5:
        init_avg = float(np.mean([(f.valgus_g + f.valgus_d) / 2 for f in frames[:3]]))
        if (bottom.valgus_g < init_avg - 20) or (bottom.valgus_d < init_avg - 20):
            recs.append(Recommendation('avertissement',
                "⚠️ Les genoux s'effondrent progressivement pendant la descente",
                'genou',
                risque="Les stabilisateurs cèdent à la fatigue — dangereux en fin de série sous charge.",
                correction="Réduis le poids le temps de corriger le pattern. "
                           "Renforce les abducteurs (monster walks, clamshells avec bande)."))
            score -= 10

    recs.sort(key=lambda r: SEVERITY_ORDER[r.niveau])
    return max(0, score), recs
