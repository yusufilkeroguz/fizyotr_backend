"""SOAPIE endpoint'leri — mevcut Flask app'ine register(app, auth) ile baglanir.

Boylece auth decorator'i (app.py'deki require_auth) disaridan enjekte edilir;
dairesel import olmaz ve tum SOAPIE mantigi bu paket icinde kalir.
"""
from __future__ import annotations

import json

from flask import Response, jsonify, request, stream_with_context

from . import agent, retrieval


def _sse(obj: dict) -> str:
    return f"data: {json.dumps(obj, ensure_ascii=False)}\n\n"


def _read_params(data: dict):
    subjective = data.get("subjective")
    patient = data.get("patient")
    try:
        k = int(data.get("k", 8))
    except (TypeError, ValueError):
        k = 8
    return subjective, patient, max(3, min(k, 15))


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
        """Tek-parca (non-streaming) SOAP onerisi."""
        subjective, patient, k = _read_params(request.get_json(silent=True) or {})
        try:
            result = agent.suggest(subjective, patient=patient, k=k)
        except agent.SoapieError as e:
            return jsonify({"error": str(e)}), e.status
        except Exception as e:
            return jsonify({"error": f"{type(e).__name__}: {e}"}), 500
        return jsonify(result)

    @app.post("/soapie/suggest/stream")
    @auth
    def soapie_suggest_stream():
        """Streaming (SSE) SOAP onerisi — sonuc parca parca akar.

        Once senkron asama (sorgu+retrieval) calisir; HTTP hatasi burada doner.
        Sonra text/event-stream: {type:meta} -> {type:delta}* -> {type:result|error}.
        """
        subjective, patient, k = _read_params(request.get_json(silent=True) or {})
        try:
            qt, chunks = agent.prepare(subjective, patient=patient, k=k)
        except agent.SoapieError as e:
            return jsonify({"error": str(e)}), e.status
        except Exception as e:
            return jsonify({"error": f"{type(e).__name__}: {e}"}), 500

        def gen():
            yield _sse({
                "type": "meta",
                "labels": agent.panel_labels(qt.language),   # paneller basliklarla erken cikar
                "retrieved_pages": sorted({c["page"] for c in chunks}),
                "query_terms": qt.model_dump(),
                "source": retrieval.get_index().meta.get("source"),
            })
            for ev in agent.stream_synthesis(subjective, chunks, qt, patient=patient):
                yield _sse(ev)

        return Response(stream_with_context(gen()), mimetype="text/event-stream",
                        headers={"Cache-Control": "no-cache",
                                 "X-Accel-Buffering": "no"})
