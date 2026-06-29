"""SOAPIE endpoint'leri — mevcut Flask app'ine register(app, auth) ile baglanir.

Boylece auth decorator'i (app.py'deki require_auth) disaridan enjekte edilir;
dairesel import olmaz ve tum SOAPIE mantigi bu paket icinde kalir.
"""
from __future__ import annotations

from flask import jsonify, request

from . import agent, retrieval


def register(app, auth):
    """SOAPIE rotalarini verilen Flask app'ine ekler.

    auth: app.py'deki require_auth decorator'i (korumali rota icin).
    """

    @app.get("/soapie")
    def soapie_info():
        """Indeks durumu — auth gerektirmez."""
        try:
            meta = retrieval.get_index().meta
            ok = True
        except Exception as e:  # indeks yok / bozuk
            meta, ok = {"error": str(e)}, False
        return jsonify({
            "service": "soapie-agent",
            "ready": ok,
            "index": meta,
            "usage": "POST /soapie/suggest  body: {\"subjective\": \"...\", \"patient\": {...}}",
            "note": "Terapist S (subjektif) girer; agent O/A icin oneri ve P icin test/uyari uretir.",
        })

    @app.post("/soapie/suggest")
    @auth
    def soapie_suggest():
        data = request.get_json(silent=True) or {}
        subjective = data.get("subjective")
        patient = data.get("patient")
        try:
            k = int(data.get("k", 8))
        except (TypeError, ValueError):
            k = 8
        k = max(3, min(k, 15))

        try:
            result = agent.suggest(subjective, patient=patient, k=k)
        except agent.SoapieError as e:
            return jsonify({"error": str(e)}), e.status
        except Exception as e:
            return jsonify({"error": f"{type(e).__name__}: {e}"}), 500
        return jsonify(result)
