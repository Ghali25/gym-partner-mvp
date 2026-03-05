FROM python:3.11-slim

# Libs système requises par opencv-python-headless et mediapipe
RUN apt-get update && apt-get install -y \
    libxcb1 \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD gunicorn --timeout 120 --workers 2 --threads 2 --bind 0.0.0.0:$PORT server:app
