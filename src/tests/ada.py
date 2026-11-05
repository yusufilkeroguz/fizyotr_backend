"""ADA — Hizlanma-Yavaslama (Acceleration-Deceleration Ability) testi.

Protokol: sporcu duz bir hatta maksimum hizlanir, pik hizina ulasir ve
ardindan mumkun oldugunca cabuk YAVASLAYIP durur. Test hem hizlanma
(acceleration) hem de YAVASLAMA (deceleration) yetenegini olcer; yavaslama
kapasitesi yon degistirme performansi ve yaralanma riski (ozellikle hamstring/
diz) acisindan kritiktir.

Kamera: YAN kamera, kurs dik gorunur; hareket yatay (X) yonunde — sprint
testleriyle ayni kurulum.

Pipeline: hip_x(t) [px] -> pozisyon [m] -> hiz v(t) [m/s] -> ivme a(t) [m/s^2].
Pik hiza kadar = hizlanma fazi; pik hizdan duruşa = yavaslama fazi.

Cizgiler: start_x = baslangic, finish_x = bitis cizgisi. Sporcu start'tan
hizlanir, finish_x'i yuksek hizda gecer ve ardindan FRENLEYIP durur. Bitis
cizgisi sonrasi frenleme (mesafe/sure/yavaslama) ayrica raporlanir.

Kalibrasyon (px -> m), su oncelikle:
  1) Cizgi modu: start_x, finish_x ayni zamanda kalibrasyon cizgileridir;
     aralarindaki gercek mesafe cal_distance_m (varsayilan 20 m) -> px_per_m =
     |finish_x-start_x| / cal_distance_m.
  2) Boy modu: height_m verilirse burun-ayak bilegi dikey pikselinden
     (body_px_per_m). Cizgi modu daha hassastir.

CLI:
    python -m src.tests.ada ada.csv --start-x 200 --finish-x 1700 --cal-distance 20
    python -m src.tests.ada ada.csv --height 1.80
"""
from __future__ import annotations

import argparse
import json
import sys
from typing import Optional

import numpy as np

from ..pose_common import (
    iter_pose, pt2d, smooth_median, body_px_per_m, motion_onset, crossing_frame,
    NOSE, LEFT_HIP, RIGHT_HIP, LEFT_ANKLE, RIGHT_ANKLE,
)


def _interp_nan(arr: np.ndarray) -> np.ndarray:
    """Gecerli ornekler arasinda lineer interpolasyon (turev stabilitesi icin)."""
    a = np.asarray(arr, dtype=float)
    idx = np.where(~np.isnan(a))[0]
    if idx.size < 2:
        return a
    out = a.copy()
    out[idx[0]:idx[-1] + 1] = np.interp(
        np.arange(idx[0], idx[-1] + 1), idx, a[idx])
    return out


def _moving_avg(arr: np.ndarray, win: int) -> np.ndarray:
    if win < 2:
        return arr
    kernel = np.ones(win) / win
    return np.convolve(arr, kernel, mode="same")


def analyze(video_path: str, *, start_x: Optional[int] = None,
            finish_x: Optional[int] = None, cal_distance_m: Optional[float] = 20.0,
            height_m: Optional[float] = None, model: str = "heavy",
            conf: float = 0.2) -> dict:
    hip_x: list[float] = []
    nose_ank: list[float] = []
    fps = 30.0
    for fi, t, l2, l3, W, H, fps in iter_pose(video_path, model=model, conf=conf):
        if l2 is None:
            hip_x.append(np.nan)
            nose_ank.append(np.nan)
            continue
        lh, rh = pt2d(l2, LEFT_HIP, W, H), pt2d(l2, RIGHT_HIP, W, H)
        hip_x.append(float((lh[0] + rh[0]) / 2))
        nose = pt2d(l2, NOSE, W, H)
        ank_y = max(pt2d(l2, LEFT_ANKLE, W, H)[1], pt2d(l2, RIGHT_ANKLE, W, H)[1])
        nose_ank.append(abs(ank_y - nose[1]))

    hip_raw = np.array(hip_x)
    raw_valid = np.where(~np.isnan(hip_raw))[0]   # smoothing ONCESI ham tespitler
    hipx = smooth_median(hip_raw, 5)
    valid = np.where(~np.isnan(hipx))[0]
    if valid.size < 8:
        raise RuntimeError("Yeterli pose verisi yok.")

    # Pose kapsami: aktif aralikta (ilk-son ham tespit) pose bulunan kare orani.
    # RAW uzerinden olculur; cunku smooth_median ~2 karelik bosluklari doldurur
    # ve smoothing sonrasi sayim kapsami olduğundan yuksek gosterir.
    # Cok dusukse (kisi karede kucuk/uzak ya da kareye girip cikiyor) trayektori
    # parcali olur, hiz/ivme guvenilmez cikar; yaniltici "v_max" hatasi yerine
    # gercek sebebi (zayif takip) net bildir.
    span = int(raw_valid[-1] - raw_valid[0]) + 1 if raw_valid.size >= 2 else 0
    coverage = raw_valid.size / span if span > 0 else 0.0
    if coverage < 0.5:
        raise RuntimeError(
            f"Pose kapsami cok dusuk (%{round(coverage * 100)}). Kisi karede "
            "kucuk/uzak ya da kareye girip cikiyor; trayektori parcali. "
            "Kamerayi yaklastir (vucut karenin en az yarisi) ve sporcunun tum "
            "kosu boyunca karede kalmasini sagla.")

    # ── Kalibrasyon px -> m ───────────────────────────────────────────────
    # Cizgi modu: start_x ve finish_x ayni zamanda kalibrasyon cizgileridir;
    # aralarindaki gercek mesafe cal_distance_m -> px_per_m.
    px_per_m: Optional[float] = None
    cal_mode = None
    if start_x is not None and finish_x is not None and cal_distance_m:
        seg = abs(finish_x - start_x)
        if seg < 50:
            raise RuntimeError("start_x ile finish_x cok yakin.")
        px_per_m = seg / cal_distance_m
        cal_mode = "line"
    elif height_m:
        px_per_m = body_px_per_m(np.array(nose_ank), height_m)
        cal_mode = "height"
    if not px_per_m or px_per_m <= 0:
        raise RuntimeError("Kalibrasyon gerekli: (start_x + finish_x + "
                           "cal_distance_m) veya height_m.")

    # ── Pozisyon (m), yon ─────────────────────────────────────────────────
    first_x = float(hipx[valid[0]])
    last_x = float(hipx[valid[-1]])
    direction = 1 if last_x >= first_x else -1

    onset = motion_onset(hipx, int(valid[0]),
                         threshold_px=max(30.0, px_per_m * 0.3))
    onset_f = int(round(onset)) if onset is not None else int(valid[0])
    onset_f = max(onset_f, int(valid[0]))
    last_f = int(valid[-1])
    if last_f - onset_f < 4:
        raise RuntimeError("Hareket araligi cok kisa.")

    pos_px = (_interp_nan(hipx) - first_x) * direction
    pos_m = pos_px / px_per_m

    # ── Hiz / ivme ────────────────────────────────────────────────────────
    dt = 1.0 / fps
    win = max(3, int(round(0.15 * fps)))
    vel = np.gradient(pos_m, dt)
    vel = _moving_avg(vel, win)
    acc = np.gradient(vel, dt)
    acc = _moving_avg(acc, win)

    seg = slice(onset_f, last_f + 1)
    v_seg = vel[seg]
    if v_seg.size < 4 or not np.isfinite(v_seg).any():
        raise RuntimeError("Hiz profili hesaplanamadi.")
    peak_rel = int(np.nanargmax(v_seg))
    peak_f = onset_f + peak_rel
    v_max = float(vel[peak_f])
    if v_max < 1.0:
        raise RuntimeError("Anlamli hiz tespit edilemedi (v_max < 1 m/s).")

    # ── Durus tespiti: pik sonrasi hiz esigin altina iner ─────────────────
    stop_thr = max(0.5, 0.1 * v_max)
    stop_f = None
    for i in range(peak_f + 1, last_f + 1):
        if np.isfinite(vel[i]) and vel[i] <= stop_thr:
            stop_f = i
            break
    stopped = stop_f is not None
    if stop_f is None:
        stop_f = last_f               # kare icinde tam durmadi

    # ── Faz metrikleri ────────────────────────────────────────────────────
    accel_time = (peak_f - onset_f) / fps
    accel_dist = float(pos_m[peak_f] - pos_m[onset_f])
    avg_accel = v_max / accel_time if accel_time > 0 else None
    peak_accel = float(np.nanmax(acc[onset_f:peak_f + 1])) if peak_f > onset_f else None

    decel_time = (stop_f - peak_f) / fps
    decel_dist = float(pos_m[stop_f] - pos_m[peak_f])
    avg_decel = v_max / decel_time if decel_time > 0 else None
    peak_decel = (abs(float(np.nanmin(acc[peak_f:stop_f + 1])))
                  if stop_f > peak_f else None)

    # Yavaslama / hizlanma orani: >1 ise frene cikistan daha hizli.
    decel_accel_ratio = (round(avg_decel / avg_accel, 3)
                         if (avg_decel and avg_accel) else None)

    # ── Bitis cizgisi sonrasi frenleme (kullanici-tanimli finish_x) ───────
    # Yukaridaki "deceleration" pik-hizdan duruşa otomatik faz; bu blok ise
    # FINISH cizgisini gectikten sonra duruşa kadarki frenlemeyi olcer
    # (sporcu cizgiyi yuksek hizda gecip frenler). Sadece finish_x verilince.
    braking_after_finish = None
    if finish_x is not None:
        fin = crossing_frame(hipx, float(finish_x), direction, onset_f + 1)
        if fin is not None and fin < stop_f:
            fin_f = int(round(fin))
            v_finish = float(vel[fin_f])
            brake_time = (stop_f - fin) / fps
            brake_dist = float(pos_m[stop_f] - pos_m[fin_f])
            brake_avg = v_finish / brake_time if brake_time > 0 else None
            brake_peak = (abs(float(np.nanmin(acc[fin_f:stop_f + 1])))
                          if stop_f > fin_f else None)
            braking_after_finish = {
                "finish_frame": fin_f,
                "v_at_finish_m_s": round(v_finish, 3),
                "distance_m": round(brake_dist, 2),
                "time_s": round(brake_time, 3),
                "avg_m_s2": round(brake_avg, 3) if brake_avg else None,
                "peak_m_s2": round(brake_peak, 3) if brake_peak else None,
                "stopped_in_view": stopped,
            }

    return {
        "fps": round(fps, 2),
        "calibration": cal_mode,
        "px_per_m": round(px_per_m, 1),
        "direction": "right" if direction > 0 else "left",
        "v_max_m_s": round(v_max, 3),
        "total_distance_m": round(float(pos_m[stop_f] - pos_m[onset_f]), 2),
        "total_time_s": round((stop_f - onset_f) / fps, 3),
        "frames": {"onset": onset_f, "peak_velocity": peak_f, "stop": stop_f},
        "acceleration": {
            "time_s": round(accel_time, 3),
            "distance_m": round(accel_dist, 2),
            "avg_m_s2": round(avg_accel, 3) if avg_accel else None,
            "peak_m_s2": round(peak_accel, 3) if peak_accel else None,
        },
        "deceleration": {
            "time_s": round(decel_time, 3),
            "distance_m": round(decel_dist, 2),
            "avg_m_s2": round(avg_decel, 3) if avg_decel else None,
            "peak_m_s2": round(peak_decel, 3) if peak_decel else None,
            "stopped_in_view": stopped,
        },
        "braking_after_finish": braking_after_finish,
        "decel_accel_ratio": decel_accel_ratio,
    }


def main():
    ap = argparse.ArgumentParser(
        description="ADA — Hizlanma-Yavaslama (Acceleration-Deceleration) testi")
    ap.add_argument("video", help="YAN kamera pose CSV'si veya video")
    ap.add_argument("--start-x", type=int, default=None, dest="start_x")
    ap.add_argument("--finish-x", type=int, default=None, dest="finish_x")
    ap.add_argument("--cal-distance", type=float, default=20.0, dest="cal_distance_m",
                    help="start_x-finish_x arasi gercek mesafe (m, varsayilan 20)")
    ap.add_argument("--height", type=float, default=None, dest="height_m",
                    help="Boy (m) — cizgi yoksa kalibrasyon")
    ap.add_argument("--model", default="heavy", choices=["heavy", "lite"])
    ap.add_argument("--conf", type=float, default=0.2)
    args = ap.parse_args()
    r = analyze(args.video, start_x=args.start_x, finish_x=args.finish_x,
                cal_distance_m=args.cal_distance_m, height_m=args.height_m,
                model=args.model, conf=args.conf)
    a, d = r["acceleration"], r["deceleration"]
    print(f"\nADA  ->  v_max {r['v_max_m_s']} m/s | "
          f"accel {a['avg_m_s2']} m/s²  decel {d['avg_m_s2']} m/s²"
          f"  (oran {r['decel_accel_ratio']})")
    bf = r.get("braking_after_finish")
    if bf:
        print(f"Bitis sonrasi frenleme: {bf['distance_m']} m / {bf['time_s']} s "
              f"(giris {bf['v_at_finish_m_s']} m/s, durdu={bf['stopped_in_view']})")
    print()
    json.dump(r, sys.stdout, indent=2, ensure_ascii=False)
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
