"""SOAPIE servisi icin kendi Supabase JWT auth'u (bagimsiz uygulama).

Ana analiz servisiyle ayni dogrulama mantigi, ama bu servis kendi basina
deploy edildigi icin kendi kopyasini tasir (app.py'ye bagimli degil).

Env:
  SUPABASE_URL       virgulle ayrilmis bir veya daha fazla proje URL'i
  SUPABASE_JWT_AUD   (varsayilan "authenticated")
  AUTH_DISABLED=1    lokal gelistirmede JWT bypass (PRODUCTION'DA SET ETME)
"""
from __future__ import annotations

import os
from functools import wraps

import jwt
from flask import g, jsonify, request
from jwt import PyJWKClient

SUPABASE_URLS = [u.strip().rstrip("/") for u in
                 os.environ.get("SUPABASE_URL", "").split(",") if u.strip()]
SUPABASE_JWT_AUD = os.environ.get("SUPABASE_JWT_AUD", "authenticated")

_jwks_clients: dict[str, PyJWKClient] = {}


def _jwks_for(base_url: str) -> PyJWKClient:
    c = _jwks_clients.get(base_url)
    if c is None:
        c = PyJWKClient(f"{base_url}/auth/v1/.well-known/jwks.json")
        _jwks_clients[base_url] = c
    return c


def require_auth(fn):
    """Supabase JWT bekler; claims'i g.user'a koyar. AUTH_DISABLED=1 ile bypass."""
    @wraps(fn)
    def wrapper(*args, **kwargs):
        if os.environ.get("AUTH_DISABLED", "").strip().lower() in ("1", "true", "yes"):
            g.user = {"sub": "local-dev", "role": "service"}
            return fn(*args, **kwargs)
        auth = request.headers.get("Authorization", "")
        if not auth.startswith("Bearer "):
            return jsonify({"error": "missing bearer token"}), 401
        token = auth[len("Bearer "):].strip()
        if not SUPABASE_URLS:
            return jsonify({"error": "SUPABASE_URL env var tanimlanmamis"}), 500
        try:
            # 'iss'e bak (imza henuz dogrulanmadi); sadece guvenilen projeyle dogrula.
            iss = jwt.decode(token, options={"verify_signature": False}).get("iss", "")
            base = next((u for u in SUPABASE_URLS if f"{u}/auth/v1" == iss), None)
            if base is None:
                return jsonify({"error": f"untrusted issuer: {iss}"}), 401
            key = _jwks_for(base).get_signing_key_from_jwt(token).key
            g.user = jwt.decode(
                token, key,
                algorithms=["RS256", "ES256"],
                audience=SUPABASE_JWT_AUD,
                issuer=f"{base}/auth/v1",
            )
        except Exception as e:
            return jsonify({"error": f"invalid token: {type(e).__name__}: {e}"}), 401
        return fn(*args, **kwargs)
    return wrapper
