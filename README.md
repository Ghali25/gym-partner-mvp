# 🏋️ GYM PARTNER — MVP

Analyse biomécanique de mouvements sportifs par vidéo. MediaPipe détecte les articulations, Flask sert l'API, Airtable stocke l'historique.

---

## Stack

| Composant | Tech |
|---|---|
| Backend | Python 3 · Flask |
| Vision | MediaPipe PoseLandmarker |
| Base de données | Airtable |
| Frontend | HTML/CSS/JS vanilla |

---

## Installation

### 1. Cloner le repo

```bash
git clone git@github.com:Ghali25/gym-partner-mvp.git
cd gym-partner-mvp
```

### 2. Créer et activer l'environnement virtuel

```bash
python3 -m venv gympartner-env
source gympartner-env/bin/activate
```

### 3. Installer les dépendances

```bash
pip install -r requirements.txt
```

### 4. Télécharger le modèle MediaPipe

```bash
curl -O https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task
mv pose_landmarker_heavy.task pose_landmarker.task
```

### 5. Configurer Airtable

Copie le fichier d'exemple et remplis tes credentials :

```bash
cp .env.example .env
```

Édite `.env` :

```
AIRTABLE_API_KEY=patXXXXXXXXXXXXXX...
AIRTABLE_BASE_ID=appXXXXXXXXXXXXXX
```

**Pour obtenir ces valeurs :**
- **API Key** → [airtable.com/create/tokens](https://airtable.com/create/tokens)
  - Scopes requis : `data.records:read` · `data.records:write` · `schema.bases:read` · `schema.bases:write`
  - Access : sélectionne ta base GymPartner
- **Base ID** → dans l'URL de ta base : `airtable.com/appXXXXXXXX/...`

> ⚠️ Sans `.env`, l'app fonctionne normalement mais ne sauvegarde pas l'historique.

---

## Lancer le serveur

```bash
source gympartner-env/bin/activate
python server.py
```

Ouvre **[http://localhost:8080](http://localhost:8080)** dans ton navigateur.

> **Note :** Le port 5000 est réservé par macOS AirPlay, on utilise le **8080**.

### Accès depuis un autre appareil (même WiFi)

Trouve ton IP locale :
```bash
ipconfig getifaddr en0
```

Puis ouvre `http://[TON_IP]:8080` sur ton téléphone ou un autre ordi.

---

## Architecture

```
gym_partner_mvp/
├── server.py          # Flask — endpoints /analyze et /history
├── engine.py          # Pipeline MediaPipe générique
├── exercises/
│   ├── squat.py       # Logique biomécanique squat
│   └── bench.py       # Logique biomécanique bench press
├── airtable.py        # Wrapper Airtable (logging + historique)
├── index.html         # UI complète (dark/light mode)
├── requirements.txt
└── .env               # Credentials Airtable (gitignored)
```

### Ajouter un nouvel exercice

Crée `exercises/mon_exercice.py` avec 4 fonctions :

```python
def build_frame_data(landmarks, w, h): ...  # extrait les angles d'une frame
def find_bottom(frames): ...               # trouve le point clé du mouvement
def validate(frames): ...                  # détecte un mauvais exercice
def evaluate(frames, bottom_idx): ...      # retourne (score, recommandations)
```

C'est tout — `engine.py` route automatiquement.

---

## Endpoints API

| Méthode | Endpoint | Description |
|---|---|---|
| `POST` | `/analyze` | Analyse une vidéo · body: `video` (file) + `exercise` (squat/bench) |
| `GET` | `/history` | Retourne les 10 dernières analyses Airtable |

---

## Lancer les commandes utiles

```bash
# Tuer le port 8080 si occupé
lsof -ti :8080 | xargs kill -9

# Pusher ses modifications
git add .
git commit -m "feat: description"
git push
```
