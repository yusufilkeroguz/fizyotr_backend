"""30 m Sprint — yatay yer degistirme zaman / hiz analizi.

Protokol: kisi 30 m'lik duz hatta maksimum hizla kosar. Cikti: toplam sure,
ortalama hiz, pik anlik hiz.

Iki mod:
  1) Cizgi modu (onerilen): start_x ve finish_x piksel cizgilerini ver.
     Kalibre px_per_m = |finish_x - start_x| / distance_m.
  2) Otomatik mod: ilk -> son izlenen yatay konum = distance_m kabul edilir
     (daha az hassas).

Kamera: YAN kamera, kurs dik gorunur; hareket yatay (X) yonunde.

CLI:
    python sprint_30m.py run.mp4 --start-x 200 --finish-x 1700 --distance 30
"""
from __future__ import annotations

import argparse
import json
import sys
from typing import Optional

import numpy as np

from ..pose_common import (
    iter_pose, pt2d, smooth_median, crossing_frame, motion_onset,
    LEFT_HIP, RIGHT_HIP,
)


def analyze(video_path: str, *, distance_m: float = 30.0,
            start_x: Optional[int] = None, finish_x: Optional[int] = None,
            model: str = "heavy", conf: float = 0.2) -> dict:
    hip_x: list[float] = []
    fps = 30.0
    for fi, t, l2, l3, W, H, fps in iter_pose(video_path, model=model, conf=conf):
        if l2 is None:
            hip_x.append(np.nan)
            continue
        lh, rh = pt2d(l2, LEFT_HIP, W, H), pt2d(l2, RIGHT_HIP, W, H)
        hip_x.append(float((lh[0] + rh[0]) / 2))
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
        if sc is None:
            # Sporcu start_x cizgisinde durup koşmaya basladi: crossing yok ->
            # motion_onset ile rest pozisyonundan ilk anlamli sapma.
            onset = motion_onset(hipx, int(valid[0]),
                                 threshold_px=max(30.0, px_per_m * 0.3))
            sc = onset if onset is not None else float(valid[0])
        fc = crossing_frame(hipx, float(finish_x), direction,
                            int(sc) + 1)
        if fc is None:
            raise RuntimeError("Finish cizgisi gecisi bulunamadi.")
        start_f, finish_f = sc, fc
        mode = "line"
    else:
        track_px = abs(last_x - first_x)
        if track_px < 50:
            raise RuntimeError("Yatay hareket cok az (otomatik mod basarisiz).")
        px_per_m = track_px / distance_m
        start_f, finish_f = float(valid[0]), float(valid[-1])
        mode = "auto"

    time_s = (finish_f - start_f) / fps
    if time_s <= 0:
        raise RuntimeError("Gecersiz sure.")
    avg_speed = distance_m / time_s

    # Pik hiz: hipx'in pencerelenmis turevini al; en yuksek anlik hiz.
    peak_speed = None
    if px_per_m > 0:
        win = max(3, int(0.2 * fps))
        lo, hi = int(round(start_f)), int(round(finish_f))
        seg = hipx[max(0, lo):min(len(hipx), hi + 1)].copy()
        if seg.size >= win + 1:
            # NaN'lari ileri-doldur, basit hareketli ortalama
            mask = np.isnan(seg)
            if mask.any():
                idx = np.where(~mask)[0]
                if idx.size:
                    seg = np.interp(np.arange(len(seg)), idx, seg[idx])
            kernel = np.ones(win) / win
            sm = np.convolve(seg, kernel, mode="valid")
            d = np.abs(np.diff(sm))
            if d.size:
                peak_speed = round(float(np.max(d)) * fps / px_per_m, 3)

    return {
        "fps": round(fps, 2),
        "mode": mode,
        "distance_m": distance_m,
        "direction": "right" if direction > 0 else "left",
        "px_per_m": round(px_per_m, 1),
        "start_frame": round(start_f, 1),
        "finish_frame": round(finish_f, 1),
        "time_s": round(time_s, 3),
        "avg_speed_m_s": round(avg_speed, 3),
        "peak_speed_m_s": peak_speed,
    }


def main():
    ap = argparse.ArgumentParser(description="30 m sprint")
    ap.add_argument("video")
    ap.add_argument("--distance", type=float, default=30.0)
    ap.add_argument("--start-x", type=int, default=None, dest="start_x")
    ap.add_argument("--finish-x", type=int, default=None, dest="finish_x")
    ap.add_argument("--model", default="heavy", choices=["heavy", "lite"])
    ap.add_argument("--conf", type=float, default=0.2)
    args = ap.parse_args()
    r = analyze(args.video, distance_m=args.distance,
                start_x=args.start_x, finish_x=args.finish_x,
                model=args.model, conf=args.conf)
    print(f"\n30 m sprint ({r['mode']} mod)  ->  {r['time_s']}s  "
          f"({r['avg_speed_m_s']} m/s ort.)")
    if r["peak_speed_m_s"]:
        print(f"Pik hiz ~{r['peak_speed_m_s']} m/s")
    print()
    json.dump(r, sys.stdout, indent=2, ensure_ascii=False)
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
