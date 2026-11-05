"""Frontal (on kamera) yuruyus analizi — lateral'in tamamlayicisi.

Protokol: kisi kameraya DOGRU (veya uzagina) yurur. Onden gorunum, yandan
olculemeyen/guvenilmez parametreleri verir:
  * Step Width (adim genisligi)  — yandan derinlik belirsizligi yuzunden sacmalar
  * Kadans + adim/stride sureleri, stance/swing (zamanlama, aciya bagimsiz)
  * Zamansal simetri / Limp Index (stance orani)
  * Dinamik diz Valgus/Varus (frontal duzlem) — SADECE onden olculur
  * Foot Progression Angle (ice/disa basma)
Adim/stride UZUNLUGU ve YURUME HIZI onden olculemez (ilerleme derinlik ekseninde)
-> bunlar gait_lateral'den alinir.

Olcek (px/m): onden cekimde kisi yaklastikca olcek SUREKLI DEGISIR; sabit tek
kalibrasyon kullanilamaz. Her karede olcek yeniden hesaplanir:
  - height_m verilirse: burun-ayak bilegi dikey piksel / (boy * NOSE_ANKLE_RATIO)
  - verilmezse: omuz genisligi piksel / shoulder_width_m (varsayilan 0.40 m)

Olay tespiti: onden kisi yaklastigi icin ayak bilegi y sinyali guclu bir
'yaklasma trendi' icerir; adim salinimi bunun uzerine biner. Trend hareketli
ortalamayla cikarilir (detrend), heel-strike = belirgin yerel tepe.

Girdi tarayici pose CSV'si VEYA video olabilir (pose_common.iter_pose devreder).

CLI:
    python -m src.tests.gait_frontal walk_front.csv --height 1.80
    python -m src.tests.gait_frontal pinar_22_frontal.mp4 --shoulder-width 0.40
"""
from __future__ import annotations

import argparse
import json
import sys
from typing import Optional

import numpy as np

from ..pose_common import (
    iter_pose, pt2d, smooth_median, NOSE_ANKLE_RATIO,
    NOSE, LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_HIP, RIGHT_HIP,
    LEFT_KNEE, RIGHT_KNEE, LEFT_ANKLE, RIGHT_ANKLE,
    LEFT_HEEL, RIGHT_HEEL, LEFT_FOOT, RIGHT_FOOT,
)


# ── Frontal duzlem acilari ───────────────────────────────────────────────
def knee_frontal_deviation(hip, knee, ankle) -> Optional[float]:
    """Frontal duzlemde diz sapmasi (derece). Kalca-ayak bilegi cizgisine gore
    dizin yanal (x) sapmasi, uyluk uzunluguna normalize.
    + : diz mediale kayik (VALGUS egilimi)   - : laterale kayik (VARUS)."""
    hx, hy = hip; kx, ky = knee; ax, ay = ankle
    if abs(ay - hy) < 1e-6:
        return None
    t = (ky - hy) / (ay - hy)
    expected_x = hx + t * (ax - hx)
    thigh = float(np.hypot(kx - hx, ky - hy)) + 1e-6
    return float(np.degrees(np.arctan2(kx - expected_x, thigh)))


def foot_progression_angle(heel, toe) -> float:
    """Ayak progresyon acisi (derece), onden gorunumde: ayak (topuk->parmak)
    vektorunun dusey eksenden yanal sapmasi. 0 = kameraya dik."""
    hx, hy = heel; tx, ty = toe
    return float(np.degrees(np.arctan2(tx - hx, abs(ty - hy) + 1e-9)))


# ── Trend-arindirilmis olay tespiti ──────────────────────────────────────
def _moving_average(x: np.ndarray, win: int) -> np.ndarray:
    if win < 3:
        return np.asarray(x, dtype=float)
    win = int(win) | 1
    pad = win // 2
    xp = np.pad(np.asarray(x, dtype=float), pad, mode="edge")
    return np.convolve(xp, np.ones(win) / win, mode="valid")


def detect_foot_events(ankle_y, times, fps, *, min_dt: float = 0.30,
                       prom_frac: float = 0.15, stance_lo: float = 0.40,
                       stance_hi: float = 0.80) -> dict:
    """Trend-arindirilmis ayak bilegi y izinden HS (yere temas) ve TO (kalkis).

    HS = ayak en altta (detr yerel MAKS) — guvenilir. TO dogrulamasi lateral
    ile ayni: her STRIDE penceresinde (ardisik HS arasi) TO = detr MINIMUMU
    (ayak en yukarida). Foot-off [stance_lo, stance_hi] disi ise stance/swing
    guvenilmez kabul edilir; ortalamalara/Limp Index'e katilmaz."""
    y = np.asarray(ankle_y, dtype=float)
    n = len(y)
    valid = ~np.isnan(y)
    empty = {"strike_frames": [], "strike_times": [], "toe_off_frames": [],
             "toe_off_times": [], "stride_times": [], "stance_times": [],
             "swing_times": [], "foot_off_pcts": [],
             "strides_total": 0, "strides_reliable": 0}
    if valid.sum() < 5:
        return empty
    idx = np.arange(n)
    y_filled = np.interp(idx, idx[valid], y[valid])
    trend = _moving_average(y_filled, max(5, int(round(fps * 0.8))))
    detr = _moving_average(y_filled - trend, max(3, int(round(fps * 0.12))))

    prom = prom_frac * (float(np.nanmax(detr) - np.nanmin(detr)) + 1e-9)
    min_gap = max(1, int(round(min_dt * fps)))

    cand = [i for i in range(1, n - 1)
            if detr[i] >= detr[i - 1] and detr[i] > detr[i + 1]]
    cand.sort(key=lambda i: -detr[i])
    strikes: list[int] = []
    for p in cand:
        if detr[p] < prom * 0.5:
            break
        if all(abs(p - c) >= min_gap for c in strikes):
            strikes.append(p)
    strikes.sort()                       # ayak en altta (y en buyuk) = HS

    strike_t = [times[i] for i in strikes]
    stride_times, toe_offs, to_t = [], [], []
    stance_times, swing_times, foot_off_pcts = [], [], []
    reliable = 0
    for a, b in zip(strikes[:-1], strikes[1:]):
        stride = times[b] - times[a]
        if stride <= 0.1 or (b - a) < 3:
            continue
        stride_times.append(stride)
        seg = detr[a:b + 1]
        if not np.isfinite(seg).any():
            continue
        k = a + int(np.nanargmin(seg))   # stride ici ayak en yukarida = TO
        toe_offs.append(k); to_t.append(times[k])
        fo = (times[k] - times[a]) / stride
        if stance_lo <= fo <= stance_hi:
            stance_times.append(times[k] - times[a])
            swing_times.append(times[b] - times[k])
            foot_off_pcts.append(fo * 100.0)
            reliable += 1

    return {
        "strike_frames": strikes, "strike_times": strike_t,
        "toe_off_frames": toe_offs, "toe_off_times": to_t,
        "stride_times": [d for d in stride_times if 0.1 < d < 2.5],
        "stance_times": stance_times, "swing_times": swing_times,
        "foot_off_pcts": foot_off_pcts,
        "strides_total": len(stride_times), "strides_reliable": reliable,
    }


def _mean(vals) -> Optional[float]:
    vals = [v for v in vals if v is not None and not (isinstance(v, float) and np.isnan(v))]
    return float(np.mean(vals)) if vals else None


def _r(v, nd=3):
    return None if v is None else round(float(v), nd)


# ── Ana analiz ───────────────────────────────────────────────────────────
def analyze(video_path: str, height_m: Optional[float] = None, *,
            shoulder_width_m: float = 0.40, model: str = "heavy",
            conf: float = 0.2, min_vis: float = 0.4) -> dict:
    frames: list = []           # her kare: gerekli noktalar dict'i veya None
    scale_samples: list = []
    fps = 30.0

    for fi, t, l2, l3, W, H, fps in iter_pose(video_path, model=model, conf=conf):
        if l2 is None:
            frames.append(None)
            continue
        P = lambda i: pt2d(l2, i, W, H)
        v = lambda i: l2[i].visibility
        # anlik olcek
        px_per_m = None
        if height_m and v(NOSE) > min_vis:
            na = abs(max(P(LEFT_ANKLE)[1], P(RIGHT_ANKLE)[1]) - P(NOSE)[1])
            if na > 5:
                px_per_m = na / (height_m * NOSE_ANKLE_RATIO)
        if px_per_m is None and v(LEFT_SHOULDER) > min_vis and v(RIGHT_SHOULDER) > min_vis:
            sw = abs(P(LEFT_SHOULDER)[0] - P(RIGHT_SHOULDER)[0])
            if sw > 5:
                px_per_m = sw / shoulder_width_m
        if px_per_m:
            scale_samples.append(px_per_m)
        frames.append({
            "px_per_m": px_per_m,
            "r_ank": P(RIGHT_ANKLE), "l_ank": P(LEFT_ANKLE),
            "r_heel": P(RIGHT_HEEL), "l_heel": P(LEFT_HEEL),
            "r_knee": P(RIGHT_KNEE), "l_knee": P(LEFT_KNEE),
            "r_hip": P(RIGHT_HIP), "l_hip": P(LEFT_HIP),
            "r_toe": P(RIGHT_FOOT), "l_toe": P(LEFT_FOOT),
            "vis": {"rk": v(RIGHT_KNEE), "lk": v(LEFT_KNEE),
                    "rh": v(RIGHT_HEEL), "lh": v(LEFT_HEEL),
                    "rt": v(RIGHT_FOOT), "lt": v(LEFT_FOOT)},
        })

    n = len(frames)
    if n < 5 or sum(1 for f in frames if f) < 5:
        raise RuntimeError("Yeterli pose verisi yok.")
    times = [i / fps for i in range(n)]
    median_scale = float(np.median(scale_samples)) if scale_samples else None

    def scale_at(i):
        s = frames[i]["px_per_m"] if frames[i] else None
        return s if s else median_scale

    # olay tespiti (iki ayak, trend-arindirilmis)
    r_ank_y = [f["r_ank"][1] if f else np.nan for f in frames]
    l_ank_y = [f["l_ank"][1] if f else np.nan for f in frames]
    # kucuk gurultuyu azalt
    r_ank_y = smooth_median(np.array(r_ank_y), 3)
    l_ank_y = smooth_median(np.array(l_ank_y), 3)
    r_ev = detect_foot_events(r_ank_y, times, fps)
    l_ev = detect_foot_events(l_ank_y, times, fps)
    r_hs, l_hs = set(r_ev["strike_frames"]), set(l_ev["strike_frames"])

    # ── Adim genisligi (medio-lateral) ─────────────────────────────────────
    # Ardisik karsi-ayak temaslari arasindaki CIFT-DESTEK penceresinde iki AYAK
    # BILEGININ yanal (x) ayriminin TEPESI adim genisligi olarak alinir.
    #  * Neden bilek (27/28), topuk (29/30) degil: topuk landmark'i frontal 2D'de
    #    yanal olarak gurultulu ve mediale cekik cikip genisligi ~%40 kucuk verir.
    #  * Neden tepe, anlik HS degil: Vicon adim genisligini iki tam basmis ayak
    #    izi arasinda olcer; HS aninda salinan ayak henuz inis noktasina ulasmaz,
    #    tam yanal ayrim cift-destek icinde olusur.
    #  * STEP_WIDTH_GAIN: yukaridan sonra kalan sistematik 2D kucuk-tahmini duzelten
    #    kalibrasyon (5-kisi Vicon dogrulamasindan; ayrinti icin proje notu).
    STEP_WIDTH_GAIN = 1.13
    all_hs_frames = sorted(r_hs | l_hs)
    step_widths_m = []
    for a, b in zip(all_hs_frames[:-1], all_hs_frames[1:]):
        sc = scale_at(a)
        if not sc:
            continue
        seg = [abs(frames[i]["r_ank"][0] - frames[i]["l_ank"][0])
               for i in range(a, b + 1) if frames[i]]
        if seg:
            step_widths_m.append(max(seg) / sc * STEP_WIDTH_GAIN)

    r_valgus, l_valgus, r_fpa, l_fpa = [], [], [], []
    for i, f in enumerate(frames):
        if not f:
            continue
        rv = knee_frontal_deviation(f["r_hip"], f["r_knee"], f["r_ank"])
        lv = knee_frontal_deviation(f["l_hip"], f["l_knee"], f["l_ank"])
        if rv is not None and f["vis"]["rk"] > min_vis:
            r_valgus.append(rv)
        if lv is not None and f["vis"]["lk"] > min_vis:
            l_valgus.append(lv)
        if f["vis"]["rt"] > min_vis and f["vis"]["rh"] > min_vis:
            r_fpa.append(foot_progression_angle(f["r_heel"], f["r_toe"]))
        if f["vis"]["lt"] > min_vis and f["vis"]["lh"] > min_vis:
            l_fpa.append(foot_progression_angle(f["l_heel"], f["l_toe"]))

    all_hs = sorted(r_ev["strike_times"] + l_ev["strike_times"])
    cadence = None
    if len(all_hs) >= 2 and (all_hs[-1] - all_hs[0]) > 0:
        cadence = round((len(all_hs) - 1) / (all_hs[-1] - all_hs[0]) * 60, 1)

    rs, ls = _mean(r_ev["stance_times"]), _mean(l_ev["stance_times"])
    limp_index = round(max(rs, ls) / min(rs, ls), 3) if (rs and ls and min(rs, ls) > 0) else None
    asym = (round(abs(rs - ls) / ((rs + ls) / 2) * 100, 1)
            if (rs and ls and (rs + ls) > 0) else None)

    sw_mean = _mean(step_widths_m)
    return {
        "fps": round(fps, 2),
        "frames": n,
        "scale_source": ("height (burun-ayak bilegi)" if height_m else
                         f"omuz genisligi {shoulder_width_m} m"),
        "px_per_m_median": _r(median_scale, 1),
        "cadence_steps_min": cadence,
        "step_width_mean_m": _r(sw_mean),
        "step_width_std_m": _r(np.std(step_widths_m) if step_widths_m else None),
        "limp_index": limp_index,
        "stance_asymmetry_pct": asym,
        "right": {
            "strikes": len(r_ev["strike_frames"]),
            "strides_reliable": r_ev["strides_reliable"],
            "stride_time_s": _r(_mean(r_ev["stride_times"])),
            "stance_time_s": _r(rs),
            "swing_time_s": _r(_mean(r_ev["swing_times"])),
            "foot_off_pct": _r(_mean(r_ev["foot_off_pcts"]), 1),
            "knee_valgus_varus_deg": _r(_mean(r_valgus), 2),
            "foot_progression_deg": _r(_mean(r_fpa), 2),
        },
        "left": {
            "strikes": len(l_ev["strike_frames"]),
            "strides_reliable": l_ev["strides_reliable"],
            "stride_time_s": _r(_mean(l_ev["stride_times"])),
            "stance_time_s": _r(ls),
            "swing_time_s": _r(_mean(l_ev["swing_times"])),
            "foot_off_pct": _r(_mean(l_ev["foot_off_pcts"]), 1),
            "knee_valgus_varus_deg": _r(_mean(l_valgus), 2),
            "foot_progression_deg": _r(_mean(l_fpa), 2),
        },
        "note": "Uzunluk/hiz onden olculemez; bunlar icin lateral analiz kullanin.",
    }


def main():
    ap = argparse.ArgumentParser(description="Frontal (on) yuruyus analizi")
    ap.add_argument("video", help="pose CSV veya video yolu")
    ap.add_argument("--height", type=float, default=None, dest="height_m",
                    help="Kisinin boyu (m) — olcek icin (onerilen)")
    ap.add_argument("--shoulder-width", type=float, default=0.40,
                    dest="shoulder_width_m",
                    help="Omuz genisligi (m); boy verilmezse olcek referansi")
    ap.add_argument("--model", default="heavy", choices=["heavy", "lite"])
    ap.add_argument("--conf", type=float, default=0.2)
    args = ap.parse_args()
    r = analyze(args.video, height_m=args.height_m,
                shoulder_width_m=args.shoulder_width_m,
                model=args.model, conf=args.conf)
    print(f"\nFrontal yuruyus  ->  kadans {r['cadence_steps_min']} adim/dk   "
          f"step width {r['step_width_mean_m']} m   Limp Index {r['limp_index']}")
    for side in ("left", "right"):
        s = r[side]
        print(f"  {side:5s}: stance {s['stance_time_s']} s  "
              f"valgus/varus {s['knee_valgus_varus_deg']}deg  "
              f"FPA {s['foot_progression_deg']}deg  (HS x{s['strikes']})")
    print()
    json.dump(r, sys.stdout, indent=2, ensure_ascii=False)
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
