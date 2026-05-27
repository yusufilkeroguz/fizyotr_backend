# Sportif Performans Testleri — CSV girisli Flask API.
# Pose cikarimi tarayicida (MediaPipe.js) yapildigi icin imaj sade:
# yalnizca flask + gunicorn + numpy. cv2/mediapipe/model GEREKMEZ.
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Once bagimliliklar -> katman onbellegi requirements degismedikce korunur
COPY requirements.txt .
RUN pip install -r requirements.txt

# Uygulama kodu
COPY . .

# Cloud Run istegi $PORT uzerinden gonderir (varsayilan 8080).
ENV PORT=8080
EXPOSE 8080

# Analizler CPU-yogun ama kisa; numpy buyuk islemlerde GIL'i birakir.
# workers/threads WEB_CONCURRENCY / THREADS env ile override edilebilir.
# --timeout 0: gunicorn worker timeout kapali (Cloud Run kendi timeout'unu uygular).
# exec: gunicorn PID 1 olur, boylece SIGTERM dogru iletilir (temiz scale-to-zero).
CMD exec gunicorn --bind :$PORT \
    --workers ${WEB_CONCURRENCY:-1} \
    --threads ${THREADS:-8} \
    --timeout 0 \
    app:app
