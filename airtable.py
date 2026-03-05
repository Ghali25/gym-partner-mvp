"""
GYM PARTNER — Module Airtable
===============================
Logue chaque analyse réussie dans Airtable et récupère l'historique.

Configuration requise dans .env :
    AIRTABLE_API_KEY=your_personal_access_token
    AIRTABLE_BASE_ID=appXXXXXXXXXXXXXX

Table Airtable attendue : "Analyses"
Champs : Exercice, Score, Date, Critiques, Avertissements, Conseils,
         Recommandations, Angles, Utilisateur
"""

import os
import json
import requests
from datetime import datetime, timezone
from dotenv import load_dotenv

load_dotenv()

_API_KEY = os.getenv("AIRTABLE_API_KEY", "")
_BASE_ID = os.getenv("AIRTABLE_BASE_ID", "")
_TABLE   = "Analyses"
_BASE_URL = f"https://api.airtable.com/v0/{_BASE_ID}/{_TABLE}"


def _headers():
    return {
        "Authorization": f"Bearer {_API_KEY}",
        "Content-Type": "application/json",
    }


def _is_configured() -> bool:
    return bool(_API_KEY and _BASE_ID
                and not _API_KEY.startswith("your_")
                and not _BASE_ID.startswith("app" + "X"))


def log_analysis(result: dict, user: str = '') -> None:
    """
    Enregistre une analyse réussie dans Airtable.
    Silencieux si les credentials ne sont pas configurés ou si Airtable est hors ligne.
    """
    if not _is_configured():
        return

    recs = result.get("recommandations", [])
    critiques      = sum(1 for r in recs if r.get("niveau") == "critique")
    avertissements = sum(1 for r in recs if r.get("niveau") == "avertissement")
    conseils       = sum(1 for r in recs if r.get("niveau") == "conseil")

    payload = {
        "fields": {
            "Exercice":        result.get("exercise", "squat").capitalize(),
            "Score":           result.get("score", 0),
            "Date":            datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.000Z"),
            "Critiques":       critiques,
            "Avertissements":  avertissements,
            "Conseils":        conseils,
            "Utilisateur":     user or "Anonyme",
            # Stockés en JSON pour reconstituer les résultats au clic
            "Recommandations": json.dumps(recs, ensure_ascii=False),
            "Angles":          json.dumps(result.get("angles", {}), ensure_ascii=False),
        }
    }

    try:
        requests.post(_BASE_URL, json=payload, headers=_headers(), timeout=5)
    except Exception:
        pass  # Ne jamais bloquer le flux principal


def get_history(limit: int = 50, user: str = '') -> list:
    """
    Retourne les `limit` dernières analyses triées par date décroissante.
    Si `user` est fourni, filtre par utilisateur (insensible à la casse).
    Retourne [] si non configuré ou en cas d'erreur.
    """
    if not _is_configured():
        return []

    try:
        params = {
            "sort[0][field]":     "Date",
            "sort[0][direction]": "desc",
            "maxRecords":         limit,
        }
        if user:
            safe = user.replace("'", "\\'")
            params["filterByFormula"] = f"LOWER({{Utilisateur}})=LOWER('{safe}')"

        resp = requests.get(
            _BASE_URL,
            headers=_headers(),
            params=params,
            timeout=5,
        )
        resp.raise_for_status()
        records = resp.json().get("records", [])
        return [{"id": r["id"], **r.get("fields", {})} for r in records]
    except Exception:
        return []
