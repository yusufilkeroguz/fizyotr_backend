"""Lateral (yan kamera) yuruyus analizi — zaman-mesafe parametreleri.

Protokol: kisi kameranin ONUNDE, goruntu duzlemine dik bir hatta yatay (X)
yonunde yurur. Yan kameradan sagittal duzlem gorunur; adim/stride UZUNLUKLARI,
YURUME HIZI ve zamansal parametreler (stance/swing/support) guvenilir olcuulur.
Adim GENISLIGI (medio-lateral) yandan olculemez -> onun icin gait_frontal.

Olay tespiti (Zeni ve ark. 2008): topugun PELVIS merkezine gore yatay konumu
kullanilir. Ayak en ONDE iken (goreli x tepe) topuk vurusu (HS), en ARKADA iken
(goreli x dip) parmak kalkisi (TO). Bu yontem yurume yonunden ve olcekten
bagimsizdir; zeminin adaptif takibine gore cok daha dayaniklidir.

Girdi tarayici pose CSV'si VEYA video olabilir (pose_common.iter_pose devreder).
px/m kalibrasyonu icin kisinin boyu (height_m) verilirse uzunluk/hiz metre
cinsinden; verilmezse piksel cinsinden (length/speed None) raporlanir.

CLI:
    python -m src.tests.gait_lateral walk.csv --height 1.80
    python -m src.tests.gait_lateral "baran 13 lateral.mp4" --height 1.81
"""
from __future__ import annotations

import argparse
import json
import sys
from typing import Optional

import numpy as np

from ..pose_common import (
    iter_pose, pt2d, smooth_median, body_px_per_m,
    NOSE, LEFT_HIP, RIGHT_HIP, LEFT_HEEL, RIGHT_HEEL,
    LEFT_ANKLE, RIGHT_ANKLE,
)


# ── Olay tespiti (tek ayak) ──────────────────────────────────────────────
def _local_extrema(sig: np.ndarray, want_max: bool, min_gap: int,
                   prom: float) -> list[int]:
    """sig icinde belirgin (prom) ve min_gap kareden seyrek yerel uc noktalar.
    want_max=True -> tepe (maks), False -> dip (min). Onem sirasina gore secilir,
    boylece yakin kucuk dalgalanmalar elenir."""
    s = sig if want_max else -sig
    n = len(s)
    med = float(np.nanmedian(s))
    cand = []
    for i in range(1, n - 1):
        if np.isnan(s[i]):
            continue
        if s[i] >= s[i - 1] and s[i] > s[i + 1] and (s[i] - med) >= prom * 0.5:
            cand.append(i)
    cand.sort(key=lambda i: -s[i])
    chosen: list[int] = []
    for p in cand:
        if all(abs(p - c) >= min_gap for c in chosen):
            chosen.append(p)
    return sorted(chosen)


def detect_gait_events(heel_x_rel, heel_x_img, times, fps, *,
                       min_dt: float = 0.30, prom_frac: float = 0.15,
                       stance_lo: float = 0.40, stance_hi: float = 0.80) -> dict:
    """Zeni yontemiyle bir ayagin HS/TO olaylarini bulur (dogrulamali TO).

    heel_x_rel : topuk_x - pelvis_x, ILERI yon POZITIF olacak sekilde
                 isaretlenmis (yani yurume yonuyle carpilmis).
    heel_x_img : topugun goruntudeki ham x'i (stride/step uzunlugu icin).

    HS = topuk en ONDE (goreli x yerel MAKS) — guvenilir.
    TO dogrulamasi: bagimsiz min-arama sahte/erken TO uretebilir. Onun yerine
    her STRIDE penceresinde (ardisik ayni-ayak HS arasi) TO = goreli x'in
    MINIMUMU. Boylece her stride'a TAM bir TO duser (siralama garanti) ve
    foot-off orani dogal olarak (0,1) icinde kalir. Fizyolojik kapi: foot-off
    [stance_lo, stance_hi] disindaki stride'in stance/swing'i GUVENILMEZ kabul
    edilir ve ortalamalara (dolayisiyla Limp Index'e) katilmaz.
    """
    empty = {"strike_frames": [], "strike_times": [], "strike_x": [],
             "toe_off_frames": [], "toe_off_times": [], "stride_times": [],
             "stance_times": [], "swing_times": [], "foot_off_pcts": [],
             "strides_total": 0, "strides_reliable": 0}
    rel = np.asarray(heel_x_rel, dtype=float)
    valid = ~np.isnan(rel)
    if valid.sum() < 5:
        return empty
    span = float(np.nanmax(rel) - np.nanmin(rel)) + 1e-9
    prom = prom_frac * span
    min_gap = max(1, int(round(min_dt * fps)))

    strikes = _local_extrema(rel, True, min_gap, prom)     # ileri tepe = HS
    strike_t = [times[i] for i in strikes]
    strike_x = [heel_x_img[i] for i in strikes]

    stride_times, toe_offs, to_t = [], [], []
    stance_times, swing_times, foot_off_pcts = [], [], []
    reliable = 0
    for a, b in zip(strikes[:-1], strikes[1:]):
        stride = times[b] - times[a]
        if stride <= 0.1 or (b - a) < 3:
            continue
        stride_times.append(stride)
        seg = rel[a:b + 1]
        if not np.isfinite(seg).any():
            continue
        k = a + int(np.nanargmin(seg))          # stride ici geri-dip = TO
        toe_offs.append(k); to_t.append(times[k])
        fo = (times[k] - times[a]) / stride      # foot-off orani (0..1)
        if stance_lo <= fo <= stance_hi:         # fizyolojik kapi
            stance_times.append(times[k] - times[a])
            swing_times.append(times[b] - times[k])
            foot_off_pcts.append(fo * 100.0)
            reliable += 1

    return {
        "strike_frames": strikes, "strike_times": strike_t, "strike_x": strike_x,
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
            model: str = "heavy", conf: float = 0.2) -> dict:
    hip_x = []
    r_heel_x, r_heel_y, l_heel_x, l_heel_y = [], [], [], []
    nose_ankle = []
    fps = 30.0

    for fi, t, l2, l3, W, H, fps in iter_pose(video_path, model=model, conf=conf):
        if l2 is None:
            for L in (hip_x, r_heel_x, r_heel_y, l_heel_x, l_heel_y, nose_ankle):
                L.append(np.nan)
            continue
        lh, rh = pt2d(l2, LEFT_HIP, W, H), pt2d(l2, RIGHT_HIP, W, H)
        hip_x.append(float((lh[0] + rh[0]) / 2))
        rhl, lhl = pt2d(l2, RIGHT_HEEL, W, H), pt2d(l2, LEFT_HEEL, W, H)
        r_heel_x.append(float(rhl[0])); r_heel_y.append(float(rhl[1]))
        l_heel_x.append(float(lhl[0])); l_heel_y.append(float(lhl[1]))
        nose_y = pt2d(l2, NOSE, W, H)[1]
        ank_y = max(pt2d(l2, LEFT_ANKLE, W, H)[1], pt2d(l2, RIGHT_ANKLE, W, H)[1])
        nose_ankle.append(abs(ank_y - nose_y))

    n = len(hip_x)
    if n < 5:
        raise RuntimeError("Yeterli pose verisi yok.")
    times = [i / fps for i in range(n)]

    hipx = smooth_median(np.array(hip_x), 5)
    rhx = smooth_median(np.array(r_heel_x), 5)
    lhx = smooth_median(np.array(l_heel_x), 5)
    valid = np.where(~np.isnan(hipx))[0]
    if valid.size < 5:
        raise RuntimeError("Yeterli gecerli kare yok.")

    # yurume yonu (ilk->son kalca yer degistirmesi)
    direction = 1 if hipx[valid[-1]] >= hipx[valid[0]] else -1

    px_per_m = body_px_per_m(np.array(nose_ankle), height_m) if height_m else None

    # Zeni: topuk_x - pelvis_x, ileri POZITIF (yon ile carp)
    r_rel = (rhx - hipx) * direction
    l_rel = (lhx - hipx) * direction

    r_ev = detect_gait_events(r_rel, rhx, times, fps)
    l_ev = detect_gait_events(l_rel, lhx, times, fps)

    def stride_lengths(ev):
        xs = ev["strike_x"]
        return [abs(xs[i] - xs[i - 1]) for i in range(1, len(xs))]

    def to_m(px):
        return (px / px_per_m) if (px_per_m and px_per_m > 0) else None

    # per-ayak zaman-mesafe
    def foot_summary(ev, other_ev):
        strides_px = stride_lengths(ev)
        stride_m = [to_m(p) for p in strides_px]
        stride_t = ev["stride_times"]
        speed = [(m / t) if (m and t and t > 0) else None
                 for m, t in zip(stride_m, stride_t)]
        stance = ev["stance_times"]
        swing = ev["swing_times"]
        # step length/time: bu ayagin HS'i ile bir SONRAKI karsi-ayak HS'i arasi
        step_len_m, step_t = [], []
        oth = other_ev["strike_times"]
        oth_x = other_ev["strike_x"]
        for hs, hx in zip(ev["strike_times"], ev["strike_x"]):
            nxt = [(t, x) for t, x in zip(oth, oth_x) if t > hs]
            if nxt:
                t2, x2 = nxt[0]
                step_t.append(t2 - hs)
                step_len_m.append(to_m(abs(x2 - hx)))
        # foot off % dogrudan dogrulanmis (kapiden gecen) stride'lardan
        foot_off = _mean(ev["foot_off_pcts"])
        return {
            "strikes": len(ev["strike_frames"]),
            "strides_reliable": ev["strides_reliable"],
            "strides_total": ev["strides_total"],
            "stride_length_m": _r(_mean(stride_m)),
            "stride_time_s": _r(_mean(stride_t)),
            "walking_speed_m_s": _r(_mean(speed)),
            "step_length_m": _r(_mean(step_len_m)),
            "step_time_s": _r(_mean(step_t)),
            "stance_time_s": _r(_mean(stance)),
            "swing_time_s": _r(_mean(swing)),
            "foot_off_pct": _r(foot_off, 1),
            "stride_lengths_m": [_r(v) for v in stride_m],
            "walking_speeds_m_s": [_r(v) for v in speed],
        }

    right = foot_summary(r_ev, l_ev)
    left = foot_summary(l_ev, r_ev)

    # kadans: iki ayagin tum HS'leri
    all_hs = sorted(r_ev["strike_times"] + l_ev["strike_times"])
    cadence = None
    if len(all_hs) >= 2 and (all_hs[-1] - all_hs[0]) > 0:
        cadence = round((len(all_hs) - 1) / (all_hs[-1] - all_hs[0]) * 60, 1)

    # Limp Index = stance sureleri orani (>=1). Vicon 'Limp Index' ile ayni tanim.
    rs, ls = _mean(r_ev["stance_times"]), _mean(l_ev["stance_times"])
    limp_index = None
    if rs and ls and min(rs, ls) > 0:
        limp_index = round(max(rs, ls) / min(rs, ls), 3)

    # simetri (asimetri %) — stance sureleri uzerinden
    asym = None
    if rs and ls and (rs + ls) > 0:
        asym = round(abs(rs - ls) / ((rs + ls) / 2) * 100, 1)

    return {
        "fps": round(fps, 2),
        "frames": n,
        "direction": "right" if direction > 0 else "left",
        "px_per_m": _r(px_per_m, 1),
        "height_m": height_m,
        "calibrated": bool(px_per_m),
        "cadence_steps_min": cadence,
        "limp_index": limp_index,
        "stance_asymmetry_pct": asym,
        "right": right,
        "left": left,
        "note": (None if px_per_m else
                 "height_m verilmedi: uzunluk/hiz metre cinsinden hesaplanamadi "
                 "(sadece zamansal parametreler)."),
    }


def main():
    ap = argparse.ArgumentParser(description="Lateral (yan) yuruyus analizi")
    ap.add_argument("video", help="pose CSV veya video yolu")
    ap.add_argument("--height", type=float, default=None, dest="height_m",
                    help="Kisinin boyu (m) — px/m kalibrasyonu icin")
    ap.add_argument("--model", default="heavy", choices=["heavy", "lite"])
    ap.add_argument("--conf", type=float, default=0.2)
    args = ap.parse_args()
    r = analyze(args.video, height_m=args.height_m, model=args.model, conf=args.conf)
    print(f"\nLateral yuruyus  ->  kadans {r['cadence_steps_min']} adim/dk   "
          f"yon {r['direction']}   Limp Index {r['limp_index']}")
    for side in ("left", "right"):
        s = r[side]
        print(f"  {side:5s}: hiz {s['walking_speed_m_s']} m/s  stride {s['stride_length_m']} m  "
              f"stride-t {s['stride_time_s']} s  foot-off {s['foot_off_pct']}%  "
              f"(HS x{s['strikes']})")
    if r["note"]:
        print(f"  ! {r['note']}")
    print()
    json.dump(r, sys.stdout, indent=2, ensure_ascii=False)
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
