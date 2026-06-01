"""5-15 m Sprint — 5 m split + 15 m bitis tek kosulu sprint.

Protokol: kisi tek yonde 15 m kosu yapar. Ara split 5 m'de alinir. Cikti:
t_5, t_15, segment hizlari (0-5 hizlanma, 5-15 ust hiz) ve ortalama hiz.

Kullanim: 3 piksel cizgisi:
    start_x, split_x_5, finish_x   (=split_x_15)
Gercek mesafeler 5 m ve 15 m kabul edilir.

CLI:
    python sprint_5_15.py run.mp4 --start-x 200 --split-x-5 500 --finish-x 1400
"""
from __future__ import annotations

import argparse
import json
import sys

import numpy as np

from ..pose_common import (
    iter_pose, pt2d, smooth_median, crossing_frame, motion_onset,
    LEFT_HIP, RIGHT_HIP,
)


def analyze(video_path: str, *, start_x: int, split_x_5: int, finish_x: int,
            model: str = "heavy", conf: float = 0.2) -> dict:
    for nm, vv in (("start_x", start_x), ("split_x_5", split_x_5),
                   ("finish_x", finish_x)):
        if vv is None:
            raise RuntimeError(f"{nm} gerekli.")

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

    direction = 1 if finish_x > start_x else -1
    track_px = float(abs(finish_x - start_x))
    if track_px < 50:
        raise RuntimeError("start_x ile finish_x cok yakin.")
    px_per_m = track_px / 15.0

    sf = crossing_frame(hipx, float(start_x), direction, valid[0])
    if sf is None:
        # Sporcu start_x cizgisinde durup koşmaya basladi -> motion_onset fallback.
        onset = motion_onset(hipx, int(valid[0]),
                             threshold_px=max(30.0, px_per_m * 0.3))
        if onset is None:
            raise RuntimeError("Start cizgisi gecisi/hareket baslangici bulunamadi.")
        sf = onset
    f5 = crossing_frame(hipx, float(split_x_5), direction, int(sf) + 1)
    if f5 is None:
        raise RuntimeError("5 m cizgisi gecisi bulunamadi.")
    ff = crossing_frame(hipx, float(finish_x), direction, int(f5) + 1)
    if ff is None:
        raise RuntimeError("15 m (finish) cizgisi gecisi bulunamadi.")

    t5 = (f5 - sf) / fps
    t15 = (ff - sf) / fps
    seg_0_5 = t5
    seg_5_15 = t15 - t5

    speeds = {
        "0_5_m_s": round(5.0 / seg_0_5, 3) if seg_0_5 > 0 else None,
        "5_15_m_s": round(10.0 / seg_5_15, 3) if seg_5_15 > 0 else None,
        "avg_m_s": round(15.0 / t15, 3) if t15 > 0 else None,
    }

    return {
        "fps": round(fps, 2),
        "mode": "line",
        "direction": "right" if direction > 0 else "left",
        "px_per_m": round(px_per_m, 1),
        "frames": {"start": round(sf, 1), "f5": round(f5, 1),
                   "finish": round(ff, 1)},
        "splits_s": {"t_5": round(t5, 3), "t_15": round(t15, 3)},
        "segments_s": {"0_5": round(seg_0_5, 3), "5_15": round(seg_5_15, 3)},
        "speeds": speeds,
    }


def main():
    ap = argparse.ArgumentParser(description="5-15 m sprint")
    ap.add_argument("video")
    ap.add_argument("--start-x", type=int, required=True, dest="start_x")
    ap.add_argument("--split-x-5", type=int, required=True, dest="split_x_5")
    ap.add_argument("--finish-x", type=int, required=True, dest="finish_x")
    ap.add_argument("--model", default="heavy", choices=["heavy", "lite"])
    ap.add_argument("--conf", type=float, default=0.2)
    args = ap.parse_args()
    r = analyze(args.video, start_x=args.start_x, split_x_5=args.split_x_5,
                finish_x=args.finish_x, model=args.model, conf=args.conf)
    sp = r["splits_s"]
    print(f"\n5-15 m sprint  ->  5m {sp['t_5']}s   15m {sp['t_15']}s")
    print()
    json.dump(r, sys.stdout, indent=2, ensure_ascii=False)
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
