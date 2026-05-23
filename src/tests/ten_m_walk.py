"""10MWT — 10 Metre Yuruyus Testi (gait speed) analizi.

Protokol: kisi duz bir hatta normal (veya hizli) tempoda yurur. Yuruyus hizi
(m/s) ve istenirse kadans / adim uzunlugu hesaplanir.

Iki mod:
  1) Cizgi modu (onerilen): kullanici start_x ve finish_x piksel cizgilerini
     ve aralarindaki gercek mesafeyi (--distance, ön 10 m) verir. Sure, kalca
     merkezinin bu iki cizgiyi gectigi anlardan hesaplanir.
  2) Otomatik mod: cizgi verilmezse, kisinin goruntudeki ilk->son izlenen
     yatay konumu `distance` metre kabul edilir (daha az hassas).

Kamera: YAN kamera, kursu dik gorur; kisi yatay (X) yonunde hareket eder.

CLI:
    python ten_m_walk.py walk.mp4 --start-x 200 --finish-x 1700 --distance 10
    python ten_m_walk.py walk.mp4 --distance 10        # otomatik mod
"""
from __future__ import annotations

import argparse
import json
import sys
from typing import Optional

import numpy as np

from ..pose_common import (
    iter_pose, pt2d, smooth_median, crossing_frame,
    LEFT_HIP, RIGHT_HIP, LEFT_ANKLE, RIGHT_ANKLE,
)


def analyze(video_path: str, *, distance_m: float = 10.0,
            start_x: Optional[int] = None, finish_x: Optional[int] = None,
            model: str = "heavy", conf: float = 0.2) -> dict:
    hip_x, lank_x, rank_x = [], [], []
    fps = 30.0
    for fi, t, l2, l3, W, H, fps in iter_pose(video_path, model=model, conf=conf):
        if l2 is None:
            for L in (hip_x, lank_x, rank_x):
                L.append(np.nan)
            continue
        lh, rh = pt2d(l2, LEFT_HIP, W, H), pt2d(l2, RIGHT_HIP, W, H)
        hip_x.append(float((lh[0] + rh[0]) / 2))
        lank_x.append(pt2d(l2, LEFT_ANKLE, W, H)[0])
        rank_x.append(pt2d(l2, RIGHT_ANKLE, W, H)[0])
    hipx = smooth_median(np.array(hip_x), 5)
    valid = np.where(~np.isnan(hipx))[0]
    if valid.size < 5:
        raise RuntimeError("Yeterli pose verisi yok.")

    first_x = float(hipx[valid[0]])
    last_x = float(hipx[valid[-1]])
    direction = 1 if last_x >= first_x else -1

    if start_x is not None and finish_x is not None:
        track_px = float(abs(finish_x - start_x))
        if track_px < 50:
            raise RuntimeError("start_x ile finish_x cok yakin.")
        px_per_m = track_px / distance_m
        direction = 1 if finish_x > start_x else -1
        sc = crossing_frame(hipx, float(start_x), direction, valid[0])
        fc = crossing_frame(hipx, float(finish_x), direction,
                            int(sc) + 1 if sc else valid[0])
        if sc is None or fc is None:
            raise RuntimeError("Start/finish cizgisi gecisi bulunamadi.")
        start_f, finish_f = sc, fc
        mode = "line"
    else:
        # Otomatik: ilk->son izlenen kare arasi = distance_m
        track_px = abs(last_x - first_x)
        if track_px < 50:
            raise RuntimeError("Yatay hareket cok az (otomatik mod basarisiz).")
        px_per_m = track_px / distance_m
        start_f, finish_f = float(valid[0]), float(valid[-1])
        mode = "auto"

    time_s = (finish_f - start_f) / fps
    if time_s <= 0:
        raise RuntimeError("Gecersiz sure.")
    speed = distance_m / time_s

    # Kadans (best-effort): on-arka ayak degisimi = adim. Sol/sag ayak bilegi
    # x farkinin isaret degistirmesi bir adimdir. Sifir civari titresimi
    # elemek icin deadband (ayak acikligi) + refrakter sure uygulanir.
    la = smooth_median(np.array(lank_x), 5)
    ra = smooth_median(np.array(rank_x), 5)
    lo, hi = int(round(start_f)), int(round(finish_f))
    diff = (la - ra)[lo:hi + 1]
    deadband = 0.15 * np.nanmax(np.abs(diff)) if np.isfinite(diff).any() else 0.0
    refractory = max(2, int(0.25 * fps))   # iki adim arasi min kare
    steps = 0
    last_sign = 0
    last_step_i = -refractory
    for i, d in enumerate(diff):
        if np.isnan(d) or abs(d) < deadband:
            continue
        s = 1 if d > 0 else -1
        if s != last_sign and last_sign != 0 and (i - last_step_i) >= refractory:
            steps += 1
            last_step_i = i
        last_sign = s
    cadence = round(steps / time_s * 60, 1) if steps else None
    step_len = round(distance_m / steps, 3) if steps else None

    return {
        "fps": round(fps, 2),
        "mode": mode,
        "distance_m": distance_m,
        "direction": "right" if direction > 0 else "left",
        "px_per_m": round(px_per_m, 1),
        "start_frame": round(start_f, 1),
        "finish_frame": round(finish_f, 1),
        "time_s": round(time_s, 3),
        "gait_speed_m_s": round(speed, 3),
        "steps": steps,
        "cadence_steps_min": cadence,
        "step_length_m": step_len,
    }


def main():
    ap = argparse.ArgumentParser(description="10 m yuruyus testi (10MWT)")
    ap.add_argument("video")
    ap.add_argument("--distance", type=float, default=10.0)
    ap.add_argument("--start-x", type=int, default=None, dest="start_x")
    ap.add_argument("--finish-x", type=int, default=None, dest="finish_x")
    ap.add_argument("--model", default="heavy", choices=["heavy", "lite"])
    ap.add_argument("--conf", type=float, default=0.2)
    args = ap.parse_args()
    r = analyze(args.video, distance_m=args.distance,
                start_x=args.start_x, finish_x=args.finish_x,
                model=args.model, conf=args.conf)
    print(f"\n10MWT ({r['mode']} mod)  ->  {r['gait_speed_m_s']} m/s")
    print(f"Mesafe {r['distance_m']:.0f} m   sure {r['time_s']}s   "
          f"yon {r['direction']}")
    if r["cadence_steps_min"]:
        print(f"Kadans ~{r['cadence_steps_min']} adim/dk   "
              f"adim uzunlugu ~{r['step_length_m']} m  (tahmini)")
    print()
    json.dump(r, sys.stdout, indent=2, ensure_ascii=False)
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
