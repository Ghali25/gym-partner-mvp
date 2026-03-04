"""
exercises/bench.py — Analyse biomécanique du développé couché (bench press)
=============================================================================
Vue de profil recommandée — côté dominant face à la caméra.

Interface attendue par engine.py :
    build_frame_data(lm, w, h) -> BenchFrameData | None
    find_bottom(frames)        -> int
    evaluate(frames, bottom_idx) -> (score, List[Recommendation])
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple
import numpy as np

from engine import calculate_angle, calculate_back_angle, Recommendation, SEVERITY_ORDER, MIN_VIS


# ─────────────────────────────────────────
# LANDMARKS bench (profil)
# shoulder=12/11  elbow=14/13  wrist=16/15  hip=24/23
# ─────────────────────────────────────────

SIDES = {
    'right': [12, 14, 16, 24],   # shoulder, elbow, wrist, hip
    'left':  [11, 13, 15, 23],
}

# Seuil de visibilité légèrement assoupli pour position allongée
MIN_VIS_BENCH = 0.4


@dataclass
class BenchFrameData:
    coude_angle:  float   # angle(shoulder, elbow, wrist)
    epaule_angle: float   # angle(hip, shoulder, elbow)
    dos_angle:    float   # calculate_back_angle(hip, shoulder)

    def to_angles_dict(self) -> dict:
        return {
            "coude":  self.coude_angle,
            "epaule": self.epaule_angle,
            "dos":    self.dos_angle,
        }


def _best_side(lm, w, h):
    """Retourne (shoulder, elbow, wrist, hip, min_visibility) du côté le plus visible."""
    vis = {side: min(lm[i].visibility for i in idxs)
           for side, idxs in SIDES.items()}
    chosen = max(vis, key=vis.get)
    idxs = SIDES[chosen]
    pts = [[lm[i].x * w, lm[i].y * h] for i in idxs]
    # pts[0]=shoulder, pts[1]=elbow, pts[2]=wrist, pts[3]=hip
    return pts[0], pts[1], pts[2], pts[3], vis[chosen]


def build_frame_data(lm, w: int, h: int) -> Optional[BenchFrameData]:
    shoulder, elbow, wrist, hip, vis = _best_side(lm, w, h)
    if vis < MIN_VIS_BENCH:
        return None

    coude_angle  = calculate_angle(shoulder, elbow, wrist)
    epaule_angle = calculate_angle(hip, shoulder, elbow)
    dos_angle    = calculate_back_angle(hip, shoulder)

    # Validation anatomique
    if not (20.0 <= coude_angle <= 180.0):
        return None
    if not (0.0 <= epaule_angle <= 180.0):
        return None

    return BenchFrameData(coude_angle, epaule_angle, dos_angle)


def find_bottom(frames: List[BenchFrameData]) -> int:
    """Point bas du bench = coude le plus fléchi (barre au plus proche de la poitrine)."""
    return min(range(len(frames)), key=lambda i: frames[i].coude_angle)


def validate(frames: List[BenchFrameData]):
    """
    Vérifie que la vidéo correspond bien à un développé couché.
    Retourne (error_type, message) si problème, None sinon.
    """
    avg_dos = float(np.mean([f.dos_angle for f in frames]))
    coude_range = max(f.coude_angle for f in frames) - min(f.coude_angle for f in frames)

    # Personne debout (squat filmé en mode bench) → dos proche de la verticale
    if avg_dos < 50:
        return ("mismatch",
                "Position debout détectée — cette vidéo ne ressemble pas à un développé couché. "
                "Sélectionne l'exercice 'Squat' ou allonge-toi sur le banc de profil pour analyser un bench press.")

    # Aucun mouvement de poussée détecté
    if coude_range < 20:
        return ("mismatch",
                "Aucun mouvement de poussée détecté. "
                "Assure-toi que la descente et la remontée complètes sont visibles dans la vidéo.")

    return None


def evaluate(frames: List[BenchFrameData], bottom_idx: int) -> Tuple[int, List[Recommendation]]:
    """
    Analyse biomécanique du bench press.
    Retourne (score/100, recommandations triées par sévérité).
    """
    recs: List[Recommendation] = []
    score = 100

    bottom  = frames[bottom_idx]
    descent = frames[:bottom_idx]
    ascent  = frames[bottom_idx + 1:]

    # ── A. POINT BAS ──────────────────────────────────────────────

    # Règle 1 — Profondeur (coude_angle au point bas)
    if bottom.coude_angle > 100:
        recs.append(Recommendation(
            'avertissement',
            "⚠️ La barre ne descend pas assez bas",
            'coude',
            risque="En ne descendant pas jusqu'à la poitrine, tes pectoraux ne travaillent que sur la moitié du mouvement.",
            correction="Descends la barre jusqu'à ce qu'elle effleure le bas de ta poitrine avant de repousser. Ton coude doit former un angle d'environ 90°."))
        score -= 15

    # Règle 2 — Flare excessif des coudes (epaule_angle trop grand)
    if bottom.epaule_angle > 80:
        recs.append(Recommendation(
            'critique',
            "❌ Tes coudes s'écartent trop sur les côtés",
            'epaule',
            risque="Quand les coudes partent trop sur le côté, l'épaule encaisse toute la charge. C'est la blessure la plus courante au bench press.",
            correction="Rentre légèrement les coudes vers ton ventre — vise environ 45° par rapport à ton corps. Imagine que tu veux visser tes mains vers l'intérieur sans bouger la barre."))
        score -= 30

    # Règle 3 — Coudes trop serrés
    elif bottom.epaule_angle < 30:
        recs.append(Recommendation(
            'conseil',
            "💡 Tes coudes sont trop collés au corps",
            'epaule',
            risque="Les pectoraux ne s'activent presque pas, c'est uniquement les triceps (arrière du bras) qui poussent.",
            correction="Écarte un peu les coudes — vise environ 45° par rapport au corps. Tu vas sentir beaucoup plus les pectoraux travailler."))
        score -= 5

    # ── B. ANALYSE TEMPORELLE ─────────────────────────────────────

    # Règle 4 — Lockout incomplet (extension des coudes en haut)
    if len(ascent) >= 2:
        max_elbow_ascent = max(f.coude_angle for f in ascent)
        if max_elbow_ascent < 155:
            recs.append(Recommendation(
                'conseil',
                "💡 Tu n'étends pas complètement les bras en haut",
                'coude',
                risque="Ne pas étendre complètement les bras garde les muscles sous tension constante — bonne pour la pump mais fatigue vite.",
                correction="Pousse jusqu'à ce que les bras soient presque tendus en haut. Ça te permet de souffler entre chaque répétition."))
            score -= 10

    # Règle 5 — Descente non contrôlée
    if len(descent) >= 4:
        quarter = max(1, len(descent) // 4)
        early_coude = float(np.mean([f.coude_angle for f in descent[:quarter]]))
        late_coude  = float(np.mean([f.coude_angle for f in descent[-quarter:]]))
        if early_coude - late_coude > 40:
            recs.append(Recommendation(
                'avertissement',
                "⚠️ Tu laisses tomber la barre trop vite",
                'coude',
                risque="Lâcher la barre sans contrôle peut blesser les épaules ou les poignets par le choc à l'impact.",
                correction="Prends 2 à 3 secondes pour descendre la barre lentement. La descente contrôlée est aussi efficace que la montée."))
            score -= 10

    # Règle 6 — Arc lombaire instable pendant la remontée
    if len(ascent) >= 3:
        dos_std = float(np.std([f.dos_angle for f in ascent]))
        if dos_std > 12:
            recs.append(Recommendation(
                'conseil',
                "💡 Ton dos bouge trop pendant la poussée",
                'dos',
                risque="Un dos qui se tortille réduit la force que tu peux pousser et peut causer des douleurs.",
                correction="Garde les omoplates bien serrées l'une contre l'autre et les fesses collées au banc du début à la fin. Contracte le ventre comme si tu allais recevoir un coup."))
            score -= 5

    recs.sort(key=lambda r: SEVERITY_ORDER[r.niveau])
    return max(0, score), recs
