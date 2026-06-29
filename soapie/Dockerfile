# SOAPIE servisi — BAGIMSIZ Cloud Run uygulamasi.
# Build context = soapie/ (deploy: gcloud run deploy fizyotr-soapie --source soapie/).
# Kaynak, `soapie` PAKETI olarak /app/soapie/ altina kopyalanir; boylece
# paket-ici relative import'lar (from . import ...) calisir.
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Once bagimliliklar -> katman onbellegi requirements degismedikce korunur
COPY requirements.txt .
RUN pip install -r requirements.txt

# Paket kaynagini /app/soapie/ altina koy (data/chunks.json dahil)
COPY . /app/soapie/

ENV PORT=8080
EXPOSE 8080

# SOAPIE yaniti (Opus + thinking) ~45 sn surebilir; --timeout 0 ile gunicorn
# worker timeout kapali, Cloud Run kendi --timeout'unu uygular. threads ile
# es zamanli istek; LLM cagrisi I/O-bound oldugu icin thread yeterli.
CMD exec gunicorn --bind :$PORT \
    --workers ${WEB_CONCURRENCY:-1} \
    --threads ${THREADS:-8} \
    --timeout 0 \
    soapie.wsgi:app
