"""Flamingo denge testi analizi.

Protokol: kisi tek bacak uzerinde (serbest bacak diz/kalca fleksiyonda)
mumkun oldugunca uzun (klasik 60 sn) dengede durur. Her denge kaybi
(serbest ayagin yere temasi / destek arayisi) sayilir. Az hata = iyi denge.

Olculenler:
  - destek bacagi
  - denge kaybi sayisi (serbest ayak yere temas olaylari) ve zamanlari
  - postural salinim: kalca merkezinin yatay (ML) genligi ve std'si (cm)

Kamera: ON kamera, tum govde gorunur. Boy verilirse salinim cm cinsinden.

CLI:
    python flamingo.py flamingo.mp4 --duration 60 --height 1.75
"""
from __future__ import annotations

import argparse
import json
import sys
from typing import Optional

import numpy as np

from ..pose_common import (
    iter_pose, pt2d, smooth_median, body_px_per_m,
    NOSE, LEFT_HIP, RIGHT_HIP, LEFT_ANKLE, RIGHT_ANKLE,
    LEFT_HEEL, RIGHT_HEEL, LEFT_FOOT, RIGHT_FOOT,
)


def _foot_low_y(l2, side: str, W: int, H: int) -> float:
    """Bir ayagin en alt (yere en yakin) noktasinin y'si."""
    idxs = ((LEFT_ANKLE, LEFT_HEEL, LEFT_FOOT) if side == "left"
            else (RIGHT_ANKLE, RIGHT_HEEL, RIGHT_FOOT))
    return max(pt2d(l2, i, W, H)[1] for i in idxs)


def analyze(video_path: str, *, duration_s: float = 60.0,
            height_m: Optional[float] = None,
            model: str = "heavy", conf: float = 0.3) -> dict:
    lfoot, rfoot, hip_x, nose_ank = [], [], [], []
    fps = 30.0
    for fi, t, l2, l3, W, H, fps in iter_pose(video_path, model=model, conf=conf):
        if l2 is None:
            for L in (lfoot, rfoot, hip_x, nose_ank):
                L.append(np.nan)
            continue
        lfoot.append(_foot_low_y(l2, "left", W, H))
        rfoot.append(_foot_low_y(l2, "right", W, H))
        lh, rh = pt2d(l2, LEFT_HIP, W, H), pt2d(l2, RIGHT_HIP, W, H)
        hip_x.append(float((lh[0] + rh[0]) / 2))
        nose = pt2d(l2, NOSE, W, H)
        ank_y = max(pt2d(l2, LEFT_ANKLE, W, H)[1], pt2d(l2, RIGHT_ANKLE, W, H)[1])
        nose_ank.append(abs(ank_y - nose[1]))
    lf = np.array(lfoot)
    rf = np.array(rfoot)
    if np.isfinite(lf).sum() < 5:
        raise RuntimeError("Yeterli pose verisi yok.")

    n = len(lf)
    end = min(n, int(duration_s * fps))
    lf, rf = lf[:end], rf[:end]
    hipx = smooth_median(np.array(hip_x[:end]), 5)

    # Destek bacagi: ortalama olarak daha alttaki (yere yakin, buyuk y) ayak
    support = "left" if np.nanmean(lf) > np.nanmean(rf) else "right"
    free = rf if support == "left" else lf
    sup = lf if support == "left" else rf

    # Yer seviyesi destek ayagindan; serbest ayak buna yaklasinca = temas
    ground_y = float(np.nanpercentile(sup, 80))
    free_s = smooth_median(free, 3)
    margin = 0.06 * np.nanstd(sup) if np.nanstd(sup) > 0 else 8.0
    margin = max(margin, 12.0)
    touchdown = free_s >= (ground_y - margin)

    # Yukaridan-asagiya gecisleri (denge kaybi olaylari) say
    losses = 0
    loss_frames = []
    prev = False
    for i in range(end):
        cur = bool(touchdown[i]) if not np.isnan(free_s[i]) else prev
        if cur and not prev:
            losses += 1
            loss_frames.append(i)
        prev = cur
    loss_times = [round(f / fps, 2) for f in loss_frames]

    # Salinim (ML): kalca x genligi
    hv = hipx[~np.isnan(hipx)]
    sway_px_range = float(hv.max() - hv.min()) if hv.size else float("nan")
    sway_px_std = float(np.std(hv)) if hv.size else float("nan")
    ppm = body_px_per_m(np.array(nose_ank[:end]), height_m) if height_m else None
    sway_cm_range = round(sway_px_range / ppm * 100, 1) if ppm else None
    sway_cm_std = round(sway_px_std / ppm * 100, 1) if ppm else None

    return {
        "fps": round(fps, 2),
        "duration_analyzed_s": round(end / fps, 2),
        "support_leg": support,
        "balance_losses": losses,
        "loss_times_s": loss_times,
        "sway_ml_range_cm": sway_cm_range,
        "sway_ml_std_cm": sway_cm_std,
        "sway_ml_range_px": round(sway_px_range, 1),
        "px_per_m": round(ppm, 1) if ppm else None,
    }


def main():
    ap = argparse.ArgumentParser(description="Flamingo denge testi")
    ap.add_argument("video", help="ON kamera videosu")
    ap.add_argument("--duration", type=float, default=60.0)
    ap.add_argument("--height", type=float, default=None,
                    help="Boy (m) — salinimi cm'e cevirmek icin")
    ap.add_argument("--model", default="heavy", choices=["heavy", "lite"])
    ap.add_argument("--conf", type=float, default=0.3)
    args = ap.parse_args()
    r = analyze(args.video, duration_s=args.duration, height_m=args.height,
                model=args.model, conf=args.conf)
    print(f"\nFlamingo  ->  destek bacagi: {r['support_leg']}")
    print(f"Denge kaybi: {r['balance_losses']} kez "
          f"({r['duration_analyzed_s']}s analiz edildi)")
    if r["loss_times_s"]:
        print(f"  zamanlar: {r['loss_times_s']}")
    if r["sway_ml_range_cm"] is not None:
        print(f"Salinim (ML): aralik {r['sway_ml_range_cm']} cm, "
              f"std {r['sway_ml_std_cm']} cm")
    else:
        print(f"Salinim (ML): {r['sway_ml_range_px']} px (boy verilirse cm)")
    print()
    json.dump(r, sys.stdout, indent=2, ensure_ascii=False)
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
