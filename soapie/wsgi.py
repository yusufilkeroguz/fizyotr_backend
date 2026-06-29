"""SOAPIE servisi — bagimsiz Flask uygulamasi (kendi Cloud Run servisi).

Ana analiz servisinden (app.py) AYRIDIR: kendi WSGI giris noktasi, kendi auth'u,
kendi Dockerfile/requirements. Boylece bagimsiz olcekleme/bellek/timeout alir.

Calistirma:
    # lokal:
    ANTHROPIC_API_KEY=... AUTH_DISABLED=1 python -m soapie.wsgi
    # production (Cloud Run):
    gunicorn -b :$PORT soapie.wsgi:app
"""
from __future__ import annotations

import os

from flask import Flask, jsonify

from . import retrieval                 # paket import => __init__ .env yukler
from .auth import require_auth
from .routes import register as register_soapie


def create_app() -> Flask:
    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = 4 * 1024 * 1024   # 4 MB (JSON istek)

    # CORS — frontend'in /soapie/suggest cagirabilmesi icin.
    _origins_env = os.environ.get("ALLOWED_ORIGINS", "*").strip()
    _origins = "*" if _origins_env == "*" else [o.strip() for o in _origins_env.split(",") if o.strip()]
    try:
        from flask_cors import CORS
        CORS(app, origins=_origins, allow_headers=["Content-Type", "Authorization"])
    except ImportError:
        pass   # flask-cors yoksa CORS'suz devam (lokal)

    @app.get("/health")
    def health():
        try:
            ready = retrieval.get_index().meta.get("chunks", 0) > 0
        except Exception:
            ready = False
        return jsonify({"status": "ok", "service": "soapie", "ready": ready})

    # /soapie ve /soapie/suggest rotalarini ekle.
    register_soapie(app, require_auth)
    return app


app = create_app()


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8081"))
    app.run(host="0.0.0.0", port=port)
