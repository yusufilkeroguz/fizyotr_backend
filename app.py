"""Sportif performans testleri — CSV girisli Flask API.

Pose cikarimi TARAYICIDA (MediaPipe.js, GET /app) yapilir; sunucu yalnizca
pose noktalarinin CSV'sini alir, ilgili analizcinin analyze() fonksiyonunu
calistirir ve sonuc JSON doner. Sunucuda cv2/mediapipe/model GEREKMEZ
(yalnizca flask + numpy).

19 test (sebt haric) POST /analyze/<test_id> ile cagrilir.

Calistirma:
    python app.py                      # http://localhost:8080  (arayuz: /app)
    # veya:  gunicorn -b :8080 app:app   (Cloud Run icin)

Ornek:
    curl -F csv=@pose.csv -F height_m=1.80 http://localhost:8080/analyze/squat-jump
    curl -F csv=@pose.csv -F joint=knee     http://localhost:8080/analyze/rom
"""
from __future__ import annotations

import os
import shutil
import sys
import tempfile
import traceback
from dataclasses import dataclass
from functools import wraps
from pathlib import Path
from typing import Callable

import jwt
from flask import Flask, Response, g, jsonify, request
from flask_cors import CORS
from jwt import PyJWKClient
from werkzeug.utils import secure_filename

ROOT = Path(__file__).resolve().parent
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))   # 'src' paketi import edilebilsin

# ── FMS analizcileri (src/fms paketi) ────────────────────────────────────
from src.fms import (  # noqa: E402
    aslr, deep_squat, hurdle_step, in_line_lunge,
    rotary_stability, shoulder_mobility, trunk_stability_pushup,
)

# ── prod analizcileri (src/prod paketi) ──────────────────────────────────
from src.prod import (  # noqa: E402
    squat_jump as prod_squat_jump,
    broad_jump as prod_broad_jump,
    single_leg_hop as prod_single_leg_hop,
    single_leg_jump as prod_single_leg_jump,
)

# ── tests/ analizcileri (yeni 7 + cmj) ───────────────────────────────────
from src.tests import (  # noqa: E402
    jump_height, less as less_mod,
    bosco_60, rsi_10_5, flamingo,
    functional_reach, ten_m_walk, rom,
)


# ── Istek baglami: form alanlarini tipli okur ────────────────────────────
class ParamError(Exception):
    """Eksik/gecersiz istek parametresi -> HTTP 400."""


class Ctx:
    def __init__(self, form, files: dict):
        self.form = form
        self.files = files            # ek dosya yollari (orn. less 'front')

    @property
    def height(self):
        return self.f("height_m", self.f("height", None))

    def model(self):
        return self.s("model", "heavy")

    def f(self, k, d=None):
        v = self.form.get(k)
        try:
            return float(v) if v not in (None, "") else d
        except (TypeError, ValueError):
            return d

    def i(self, k, d=None):
        v = self.form.get(k)
        try:
            return int(float(v)) if v not in (None, "") else d
        except (TypeError, ValueError):
            return d

    def b(self, k, d=False):
        v = self.form.get(k)
        if v is None:
            return d
        return str(v).strip().lower() in ("1", "true", "yes", "on", "evet", "var")

    def s(self, k, d=None):
        v = self.form.get(k)
        return v if v not in (None, "") else d


def _need_height(c: Ctx) -> float:
    if c.height is None:
        raise ParamError("height_m gerekli (kisinin boyu, metre)")
    return c.height


# ── Test kaydi ───────────────────────────────────────────────────────────
@dataclass
class Test:
    id: str
    category: str
    run: Callable[[str, Ctx], dict]
    extra_files: tuple = ()           # ek pose CSV alanlari (orn. less -> front)
    params: tuple = ()                # dokumantasyon icin


TESTS: dict[str, Test] = {}


def _reg(t: Test):
    TESTS[t.id] = t


# ── Adaptorler (her biri (target, ctx) -> dict) ──────────────────────────
def _fms(mod):
    def run(v, c):
        return mod.analyze(v, _need_height(c), painful=c.b("painful"),
                           model=c.model())
    return run


def _fms_deep_squat(v, c):
    return deep_squat.analyze(v, _need_height(c), board=c.b("board"),
                              painful=c.b("painful"), model=c.model())


def _prod(mod):
    def run(v, c):
        return mod.analyze(v, _need_height(c), model=c.model())
    return run


def _cmj(v, c):
    samples, fps = jump_height.extract_samples(v)
    px = jump_height.compute_pixel_height(samples, c.height)
    phys = jump_height.compute_physics_height(samples, fps)
    return {
        "fps": round(fps, 2),
        "jump_height_cm": (round(phys["jump_m"] * 100, 1) if phys else None),
        "physics_method": phys,
        "pixel_method": px,
    }


def _less(v, c):
    return less_mod.analyze(v, front_path=c.files.get("front"), model=c.model())


def _bosco(v, c):
    return bosco_60.analyze(v, duration_s=c.f("duration", 60.0), model=c.model())


def _rsi(v, c):
    return rsi_10_5.analyze(v, best_n=c.i("best", 5), model=c.model())


def _flamingo(v, c):
    return flamingo.analyze(v, duration_s=c.f("duration", 60.0),
                            height_m=c.height, model=c.model())


def _frt(v, c):
    return functional_reach.analyze(v, height_m=c.height, model=c.model())


def _rom(v, c):
    return rom.analyze(v, joint=c.s("joint", "knee"), side=c.s("side", "auto"),
                       model=c.model())


def _10mwt(v, c):
    return ten_m_walk.analyze(v, distance_m=c.f("distance", 10.0),
                              start_x=c.i("start_x"), finish_x=c.i("finish_x"),
                              model=c.model())


# ── Kayitlar ─────────────────────────────────────────────────────────────
_FMS = "FMS"
_LOWER = "Sicrama/Alt-ekstremite"
_BAL = "Denge/Fonksiyonel"
_GAIT = "Yuruyus"
_ROM = "ROM"
_H = "height_m*"

_reg(Test("fms-deep-squat", _FMS, _fms_deep_squat, params=(_H, "board(bool)", "painful(bool)")))
_reg(Test("fms-hurdle-step", _FMS, _fms(hurdle_step), params=(_H, "painful(bool)")))
_reg(Test("fms-inline-lunge", _FMS, _fms(in_line_lunge), params=(_H, "painful(bool)")))
_reg(Test("fms-shoulder-mobility", _FMS, _fms(shoulder_mobility), params=(_H, "painful(bool)")))
_reg(Test("fms-aslr", _FMS, _fms(aslr), params=(_H, "painful(bool)")))
_reg(Test("fms-trunk-stability-pushup", _FMS, _fms(trunk_stability_pushup), params=(_H, "painful(bool)")))
_reg(Test("fms-rotary-stability", _FMS, _fms(rotary_stability), params=(_H, "painful(bool)")))

_reg(Test("squat-jump", _LOWER, _prod(prod_squat_jump), params=(_H,)))
_reg(Test("broad-jump", _LOWER, _prod(prod_broad_jump), params=(_H,)))
_reg(Test("single-leg-cmj", _LOWER, _prod(prod_single_leg_jump), params=(_H,)))
_reg(Test("single-leg-hop", _LOWER, _prod(prod_single_leg_hop), params=(_H,)))
_reg(Test("cmj", _LOWER, _cmj, params=("height_m(ops, cm icin)",)))
_reg(Test("less", _LOWER, _less, extra_files=("front",), params=("front(pose CSV, ops)",)))
_reg(Test("bosco-60", _LOWER, _bosco, params=("duration(=60)",)))
_reg(Test("rsi-10-5", _LOWER, _rsi, params=("best(=5)",)))

_reg(Test("flamingo", _BAL, _flamingo, params=("duration(=60)", "height_m(ops)")))
_reg(Test("functional-reach", _BAL, _frt, params=("height_m(ops)",)))

_reg(Test("10mwt", _GAIT, _10mwt, params=("distance(=10)", "start_x(ops)", "finish_x(ops)")))

_reg(Test("rom", _ROM, _rom, params=("joint(=knee: knee/hip/elbow/shoulder/ankle/trunk)", "side(=auto)")))


# ── Supabase JWT dogrulamasi ─────────────────────────────────────────────
# Frontend Supabase'den aldigi JWT'yi `Authorization: Bearer <token>` ile yollar.
# Backend JWKS uzerinden public key'le dogrular — secret dagitmaya gerek yok.
SUPABASE_URL = os.environ.get("SUPABASE_URL", "").rstrip("/")
SUPABASE_JWT_AUD = os.environ.get("SUPABASE_JWT_AUD", "authenticated")
_jwks_client: PyJWKClient | None = None


def _jwks() -> PyJWKClient:
    """PyJWKClient kendi icinde key'leri cache'ler (default 5 dk)."""
    global _jwks_client
    if _jwks_client is None:
        if not SUPABASE_URL:
            raise RuntimeError("SUPABASE_URL env var tanimlanmamis")
        _jwks_client = PyJWKClient(f"{SUPABASE_URL}/auth/v1/.well-known/jwks.json")
    return _jwks_client


def require_auth(fn):
    """Supabase JWT bekler; claims'i g.user'a koyar (g.user['sub'] = user_id)."""
    @wraps(fn)
    def wrapper(*args, **kwargs):
        auth = request.headers.get("Authorization", "")
        if not auth.startswith("Bearer "):
            return jsonify({"error": "missing bearer token"}), 401
        token = auth[len("Bearer "):].strip()
        try:
            key = _jwks().get_signing_key_from_jwt(token).key
            g.user = jwt.decode(
                token, key,
                algorithms=["RS256", "ES256"],
                audience=SUPABASE_JWT_AUD,
                issuer=f"{SUPABASE_URL}/auth/v1",
            )
        except Exception as e:
            return jsonify({"error": f"invalid token: {type(e).__name__}: {e}"}), 401
        return fn(*args, **kwargs)
    return wrapper


# ── Flask ────────────────────────────────────────────────────────────────
app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 1024 * 1024 * 1024  # 1 GB

# CORS — frontend'in /analyze cagirabilmesi icin. ALLOWED_ORIGINS env var
# virgulle ayrilmis liste (orn. "https://dash.fizyotr.com,https://app.fizyotr.com").
# Tanimlanmazsa "*" (gelistirme); production'da set et.
_origins_env = os.environ.get("ALLOWED_ORIGINS", "*").strip()
_origins = "*" if _origins_env == "*" else [o.strip() for o in _origins_env.split(",") if o.strip()]
CORS(app, origins=_origins, allow_headers=["Content-Type", "Authorization"])


@app.get("/health")
def health():
    return {"status": "ok", "tests": len(TESTS)}


@app.get("/app")
def webapp():
    """Tarayicida pose CSV cikarip /analyze'a gonderen basit arayuz."""
    html = ROOT / "web" / "index.html"
    if not html.exists():
        return jsonify({"error": "web/index.html bulunamadi"}), 404
    return Response(html.read_text(encoding="utf-8"), mimetype="text/html")


@app.get("/")
def index():
    cats: dict[str, list] = {}
    for t in TESTS.values():
        cats.setdefault(t.category, []).append(
            {"id": t.id, "params": list(t.params)})
    return jsonify({
        "service": "sport-tests",
        "count": len(TESTS),
        "usage": "POST /analyze/<test_id>  (multipart: csv=<pose.csv> + form params)",
        "note": "* = zorunlu. Pose CSV'si tarayicida uretilir: GET /app",
        "categories": cats,
    })


@app.post("/analyze/<test_id>")
@require_auth
def analyze(test_id):
    t = TESTS.get(test_id)
    if t is None:
        return jsonify({"error": f"bilinmeyen test: {test_id}",
                        "available": sorted(TESTS)}), 404

    tmp = Path(tempfile.mkdtemp(prefix="sport_"))
    try:
        ctx_files: dict[str, str] = {}
        up_csv = request.files.get("csv")          # tarayicida cikarilmis pose CSV
        if up_csv is None:
            return jsonify({"error": "'csv' dosyasi gerekli (multipart/form-data). "
                                     "Pose CSV'sini GET /app arayuzu uretir."}), 400
        cpath = tmp / "pose.csv"
        up_csv.save(str(cpath))
        target = str(cpath)
        for ef in t.extra_files:                   # ek pose CSV (orn. less 'front')
            f = request.files.get(ef)
            if f is not None:
                p = tmp / f"{secure_filename(ef)}.csv"
                f.save(str(p))
                ctx_files[ef] = str(p)

        ctx = Ctx(request.form, ctx_files)
        result = t.run(target, ctx)
        return jsonify({"test": test_id, "category": t.category, "result": result})

    except ParamError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        return jsonify({"error": str(e),
                        "trace": traceback.format_exc().splitlines()[-3:]}), 500
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port, threaded=False)
