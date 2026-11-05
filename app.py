"""Sportif performans testleri — CSV girisli Flask API.

Pose cikarimi TARAYICIDA (MediaPipe.js, GET /app) yapilir; sunucu yalnizca
pose noktalarinin CSV'sini alir, ilgili analizcinin analyze() fonksiyonunu
calistirir ve sonuc JSON doner. Sunucuda cv2/mediapipe/model GEREKMEZ
(yalnizca flask + numpy).

25 test POST /analyze/<test_id> ile cagrilir. sebt iki kameralidir
(`csv` alaninda front.csv + side.csv); digerleri tek pose CSV'si alir.

Calistirma:
    python app.py                      # http://localhost:8080  (arayuz: /app)
    # veya:  gunicorn -b :8080 app:app   (Cloud Run icin)

Ornek:
    curl -F csv=@pose.csv -F height_m=1.80 http://localhost:8080/analyze/squat-jump
    curl -F csv=@pose.csv -F joint=knee     http://localhost:8080/analyze/rom
"""
from __future__ import annotations

import gzip
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
    sprint_30m, sprint_10_20_30, sprint_5_15, shuttle_5_0_5, shuttle_5_10_5,
    sebt, ada, fugl_meyer,
    gait_lateral, gait_frontal,
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
    multi_csv: bool = False           # `csv` alaninda 1-2 dosya (sebt: front/side)


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
    # Yan (ana) kamera: sicrama yuksekligi (fizik + piksel).
    samples, fps = jump_height.extract_samples(v)
    px = jump_height.compute_pixel_height(samples, c.height)
    phys = jump_height.compute_physics_height(samples, fps)
    result = {
        "fps": round(fps, 2),
        "jump_height_cm": (round(phys["jump_m"] * 100, 1) if phys else None),
        "physics_method": phys,
        "pixel_method": px,
    }
    # Ek ON kamera (opsiyonel): inis aninda diz valgus (FPPA) + sag-sol asimetri.
    # LESS'in frontal kalemini yeniden kullaniyoruz (ayni protokol: inis IC'sinde
    # hip-knee-ankle acisi). asimetri = |sol FPPA - sag FPPA|.
    front = c.files.get("front")
    if front:
        fm = less_mod._frontal_metrics(front, c.model(), 0.3)
        if fm:
            ic = fm["ic"]
            fl, fr = ic["fppa_l"], ic["fppa_r"]
            fmin = min(fl, fr)
            asym = abs(fl - fr)
            result["frontal"] = {
                "ic_frame": fm["ic_frame"],
                "fppa_left_deg": round(fl, 1),
                "fppa_right_deg": round(fr, 1),
                "fppa_min_deg": round(fmin, 1),
                "knee_valgus_error": bool(fmin < 170.0),
                "valgus_asymmetry_deg": round(asym, 1),
                # >8 deg sol-sag fark klinik olarak anlamli asimetri kabul edilir
                "asymmetry_flag": bool(asym > 8.0),
            }
        else:
            result["frontal"] = None  # on videoda inis/ucus tespit edilemedi
    return result


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


def _gait_lateral(v, c):
    # Yan kamera: zaman-mesafe parametreleri (stride/step uzunluk, hiz, stance/swing).
    return gait_lateral.analyze(v, height_m=c.height, model=c.model())


def _gait_frontal(v, c):
    # On kamera: step width, kadans, temporal simetri, diz valgus/varus, FPA.
    return gait_frontal.analyze(v, height_m=c.height,
                                shoulder_width_m=c.f("shoulder_width_m", 0.40),
                                model=c.model())


def _need_int(c: Ctx, name: str) -> int:
    v = c.i(name)
    if v is None:
        raise ParamError(f"{name} gerekli (piksel x koordinati, tam sayi)")
    return v


def _sprint_30m(v, c):
    return sprint_30m.analyze(v, distance_m=c.f("distance", 30.0),
                              start_x=c.i("start_x"), finish_x=c.i("finish_x"),
                              model=c.model())


def _sprint_10_20_30(v, c):
    return sprint_10_20_30.analyze(
        v,
        start_x=_need_int(c, "start_x"),
        split_x_10=_need_int(c, "split_x_10"),
        split_x_20=_need_int(c, "split_x_20"),
        finish_x=_need_int(c, "finish_x"),
        model=c.model(),
    )


def _sprint_5_15(v, c):
    return sprint_5_15.analyze(
        v,
        start_x=_need_int(c, "start_x"),
        split_x_5=_need_int(c, "split_x_5"),
        finish_x=_need_int(c, "finish_x"),
        model=c.model(),
    )


def _shuttle_5_10_5(v, c):
    return shuttle_5_10_5.analyze(
        v,
        left_x=_need_int(c, "left_x"),
        center_x=_need_int(c, "center_x"),
        right_x=_need_int(c, "right_x"),
        model=c.model(),
    )


def _shuttle_5_0_5(v, c):
    return shuttle_5_0_5.analyze(
        v,
        line_x=_need_int(c, "line_x"),
        turn_x=_need_int(c, "turn_x"),
        model=c.model(),
    )


def _ada(v, c):
    return ada.analyze(v, start_x=c.i("start_x"), finish_x=c.i("finish_x"),
                       cal_distance_m=c.f("cal_distance_m", 20.0),
                       height_m=c.height, model=c.model())


def _sebt(v, c):
    # SEBT: standart tek pose CSV'si yerine front/side dosyalari (multi_csv).
    # 'v' kullanilmaz; dosyalar c.files['front'] / c.files['side'].
    return sebt.analyze(
        front_path=c.files.get("front"),
        side_path=c.files.get("side"),
        height_m=_need_height(c),
        directions=c.s("directions"),
        legs=c.s("legs"),
        limb_length_cm=c.f("limb_length_cm"),
    )


def _fugl(test_id: str):
    """Fugl-Meyer test adaptoru. test_id sabit (rota basina bir test);
    side=auto/left/right, model, ve FM-D icin opsiyonel 'unaffected' CSV."""
    def run(v, c):
        return fugl_meyer.analyze(
            v, test=test_id, side=c.s("side", "auto"),
            model=c.model(), unaffected=c.files.get("unaffected"))
    return run


# ── Kayitlar ─────────────────────────────────────────────────────────────
_FMS = "FMS"
_LOWER = "Sicrama/Alt-ekstremite"
_BAL = "Denge/Fonksiyonel"
_GAIT = "Yuruyus"
_SPRINT = "Sprint/Suratlilik"
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
_reg(Test("cmj", _LOWER, _cmj, extra_files=("front",),
          params=("height_m(ops, cm icin)", "front(ON kamera pose CSV, ops: valgus+asimetri)")))
_reg(Test("less", _LOWER, _less, extra_files=("front",), params=("front(pose CSV, ops)",)))
_reg(Test("bosco-60", _LOWER, _bosco, params=("duration(=60)",)))
_reg(Test("rsi-10-5", _LOWER, _rsi, params=("best(=5)",)))

_reg(Test("flamingo", _BAL, _flamingo, params=("duration(=60)", "height_m(ops)")))
_reg(Test("functional-reach", _BAL, _frt, params=("height_m(ops)",)))
_reg(Test("sebt", _BAL, _sebt, multi_csv=True,
          params=(_H, "directions*", "legs*", "limb_length_cm(ops)",
                  "csv=front.csv/side.csv (1-2 dosya)")))

_reg(Test("10mwt", _GAIT, _10mwt, params=("distance(=10)", "start_x(ops)", "finish_x(ops)")))
_reg(Test("gait-lateral", _GAIT, _gait_lateral, params=("height_m(ops — uzunluk/hiz icin)",)))
_reg(Test("gait-frontal", _GAIT, _gait_frontal,
          params=("height_m(ops)", "shoulder_width_m(=0.40, boy yoksa olcek)")))

_reg(Test("30m", _SPRINT, _sprint_30m,
          params=("start_x(ops)", "finish_x(ops)", "distance(=30)")))
_reg(Test("10-20-30m", _SPRINT, _sprint_10_20_30,
          params=("start_x*", "split_x_10*", "split_x_20*", "finish_x*")))
_reg(Test("5-15", _SPRINT, _sprint_5_15,
          params=("start_x*", "split_x_5*", "finish_x*")))
_reg(Test("5-10-5", _SPRINT, _shuttle_5_10_5,
          params=("left_x*", "center_x*", "right_x*")))
_reg(Test("5-0-5", _SPRINT, _shuttle_5_0_5,
          params=("line_x*", "turn_x*")))
_reg(Test("ada", _SPRINT, _ada,
          params=("start_x(ops)", "finish_x(ops)", "cal_distance_m(=20)",
                  "height_m(ops)")))

_reg(Test("rom", _ROM, _rom, params=("joint(=knee: knee/hip/elbow/shoulder/ankle/trunk)", "side(=auto)")))

# ── Fugl-Meyer Ust Ekstremite (FMA-UE) ───────────────────────────────────
# Pose CSV'si el landmark'lari da icermeli (lhx*/rhx* sutunlari); aksi halde
# onkol rotasyon / bilek / FM-D alt-skorlari olculemez. side=auto/left/right.
_FUGL = "Fugl-Meyer (FMA-UE)"
_FP = ("side(=auto: left/right)",)
_reg(Test("fugl-flexor", _FUGL, _fugl("flexor"), params=_FP))
_reg(Test("fugl-extensor", _FUGL, _fugl("extensor"), params=_FP))
_reg(Test("fugl-synergy", _FUGL, _fugl("all"), params=_FP))   # Bolum II ozeti
_reg(Test("fugl-fm6", _FUGL, _fugl("FM-6"), params=_FP))
_reg(Test("fugl-fm7", _FUGL, _fugl("FM-7"), params=_FP))
_reg(Test("fugl-fm8", _FUGL, _fugl("FM-8"), params=_FP))
_reg(Test("fugl-fm9", _FUGL, _fugl("FM-9"), params=_FP))
_reg(Test("fugl-fm10", _FUGL, _fugl("FM-10"), params=_FP))
_reg(Test("fugl-fm11", _FUGL, _fugl("FM-11"), params=_FP))
_reg(Test("fugl-fm12", _FUGL, _fugl("FM-12"), params=_FP))
_reg(Test("fugl-fm13", _FUGL, _fugl("FM-13"), params=_FP))
_reg(Test("fugl-fmd", _FUGL, _fugl("FM-D"), extra_files=("unaffected",),
          params=("side(=auto)", "unaffected(pose CSV, ops — paired skorlama)")))


# ── Supabase JWT dogrulamasi ─────────────────────────────────────────────
# Frontend Supabase'den aldigi JWT'yi `Authorization: Bearer <token>` ile yollar.
# Backend JWKS uzerinden public key'le dogrular — secret dagitmaya gerek yok.
#
# Coklu proje: SUPABASE_URL virgulle ayrilmis liste olabilir (orn.
# "https://aaa.supabase.co,https://bbb.supabase.co"). Token'in 'iss' claim'ine
# gore dogru proje secilir; sadece listedeki issuer'lara guvenilir.
SUPABASE_URLS = [u.strip().rstrip("/") for u in
                 os.environ.get("SUPABASE_URL", "").split(",") if u.strip()]
SUPABASE_JWT_AUD = os.environ.get("SUPABASE_JWT_AUD", "authenticated")
_jwks_clients: dict[str, PyJWKClient] = {}


def _jwks_for(base_url: str) -> PyJWKClient:
    """base_url projesi icin PyJWKClient (her client key'leri ~5 dk cache'ler)."""
    c = _jwks_clients.get(base_url)
    if c is None:
        c = PyJWKClient(f"{base_url}/auth/v1/.well-known/jwks.json")
        _jwks_clients[base_url] = c
    return c


def require_auth(fn):
    """Supabase JWT bekler; claims'i g.user'a koyar (g.user['sub'] = user_id).

    Lokal gelistirme icin AUTH_DISABLED=1 ile bypass edilebilir; bu durumda
    g.user mock degerle doldurulur. Production deploy'da bu env var SET
    EDILMEMELIDIR — varsayilan JWT dogrulamasi devrede kalir.
    """
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
            # Token'in 'iss'ine bak (imza HENUZ dogrulanmadi), sadece guvenilen
            # projelerden biriyle eslesirse o projenin JWKS'iyle dogrula.
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


def _save_upload(f, path: Path) -> None:
    """Yuklenen CSV'yi diske yaz; gzip'liyse (1f 8b magic) akista ac.

    Uzun videolarin pose CSV'si 40+ MB'a cikip Cloud Run'in HTTP/1 istek
    limitine (32 MiB) / yukleme suresine takilir. Istemci CSV'yi gzip'leyip
    (CompressionStream("gzip")) gonderebilir: ~2.5-3x kuculur. Dosya adi
    veya Content-Encoding'e degil icerige bakilir; duz CSV aynen kaydedilir.
    """
    head = f.stream.read(2)
    f.stream.seek(0)
    if head == b"\x1f\x8b":
        with gzip.GzipFile(fileobj=f.stream) as src, open(path, "wb") as dst:
            shutil.copyfileobj(src, dst, 1024 * 1024)
    else:
        f.save(str(path))


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
        if t.multi_csv:
            # 'csv' alaninda 1-2 dosya gelir; filename'e gore front/side ayrilir
            # (sebt). Standart tek-CSV akisi (target=pose.csv) gecerli degil.
            uploads = request.files.getlist("csv")
            for f in uploads:
                if f is None or not f.filename:
                    continue
                stem = secure_filename(f.filename).lower()
                key = "front" if "front" in stem else "side" if "side" in stem else None
                if key is None or key in ctx_files:
                    continue
                p = tmp / f"{key}.csv"
                _save_upload(f, p)
                ctx_files[key] = str(p)
            if not ctx_files:
                return jsonify({"error": "'csv' dosyasi gerekli: front.csv ve/veya "
                                         "side.csv (multipart/form-data)."}), 400
            target = None
        else:
            up_csv = request.files.get("csv")      # tarayicida cikarilmis pose CSV
            if up_csv is None:
                return jsonify({"error": "'csv' dosyasi gerekli (multipart/form-data). "
                                         "Pose CSV'sini GET /app arayuzu uretir."}), 400
            cpath = tmp / "pose.csv"
            _save_upload(up_csv, cpath)
            target = str(cpath)
            for ef in t.extra_files:               # ek pose CSV (orn. less 'front')
                f = request.files.get(ef)
                if f is not None:
                    p = tmp / f"{secure_filename(ef)}.csv"
                    _save_upload(f, p)
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
