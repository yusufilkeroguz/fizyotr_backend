"""10-20-30 m Sprint — split zamanli sprint analizi.

Protokol: 30 m duz hatta maksimum sprint; 10 m, 20 m ve 30 m noktalarinda
split zamanlari olculur. Hizlanma profilini (0-10, 10-20, 20-30 segment
hizlari) gosterir; en hizli segmenti rapor eder.

Kullanim: 4 piksel cizgisi:
    start_x, split_x_10, split_x_20, finish_x   (=split_x_30)
Aralarindaki gercek mesafeler sirasiyla 10, 20, 30 m kabul edilir.

CLI:
    python sprint_10_20_30.py run.mp4 \
        --start-x 200 --split-x-10 600 --split-x-20 1100 --finish-x 1700
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


def analyze(video_path: str, *, start_x: int, split_x_10: int,
            split_x_20: int, finish_x: int,
            model: str = "heavy", conf: float = 0.2) -> dict:
    for nm, vv in (("start_x", start_x), ("split_x_10", split_x_10),
                   ("split_x_20", split_x_20), ("finish_x", finish_x)):
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
    px_per_m = track_px / 30.0

    sf = crossing_frame(hipx, float(start_x), direction, valid[0])
    if sf is None:
        # Sporcu start_x cizgisinde durup koşmaya basladi -> motion_onset fallback.
        onset = motion_onset(hipx, int(valid[0]),
                             threshold_px=max(30.0, px_per_m * 0.3))
        if onset is None:
            raise RuntimeError("Start cizgisi gecisi/hareket baslangici bulunamadi.")
        sf = onset
    f10 = crossing_frame(hipx, float(split_x_10), direction, int(sf) + 1)
    if f10 is None:
        raise RuntimeError("10 m cizgisi gecisi bulunamadi.")
    f20 = crossing_frame(hipx, float(split_x_20), direction, int(f10) + 1)
    if f20 is None:
        raise RuntimeError("20 m cizgisi gecisi bulunamadi.")
    ff = crossing_frame(hipx, float(finish_x), direction, int(f20) + 1)
    if ff is None:
        raise RuntimeError("30 m (finish) cizgisi gecisi bulunamadi.")

    t10 = (f10 - sf) / fps
    t20 = (f20 - sf) / fps
    t30 = (ff - sf) / fps
    seg_0_10 = t10
    seg_10_20 = t20 - t10
    seg_20_30 = t30 - t20

    speeds = {
        "0_10_m_s": round(10.0 / seg_0_10, 3) if seg_0_10 > 0 else None,
        "10_20_m_s": round(10.0 / seg_10_20, 3) if seg_10_20 > 0 else None,
        "20_30_m_s": round(10.0 / seg_20_30, 3) if seg_20_30 > 0 else None,
        "avg_m_s": round(30.0 / t30, 3) if t30 > 0 else None,
    }
    seg_speeds = [(k, v) for k, v in speeds.items()
                  if k != "avg_m_s" and v is not None]
    fastest = max(seg_speeds, key=lambda kv: kv[1], default=(None, None))[0]

    return {
        "fps": round(fps, 2),
        "mode": "line",
        "direction": "right" if direction > 0 else "left",
        "px_per_m": round(px_per_m, 1),
        "frames": {"start": round(sf, 1), "f10": round(f10, 1),
                   "f20": round(f20, 1), "finish": round(ff, 1)},
        "splits_s": {"t_10": round(t10, 3), "t_20": round(t20, 3),
                     "t_30": round(t30, 3)},
        "segments_s": {"0_10": round(seg_0_10, 3),
                       "10_20": round(seg_10_20, 3),
                       "20_30": round(seg_20_30, 3)},
        "speeds": speeds,
        "fastest_segment": fastest,
    }


def main():
    ap = argparse.ArgumentParser(description="10-20-30 m sprint")
    ap.add_argument("video")
    ap.add_argument("--start-x", type=int, required=True, dest="start_x")
    ap.add_argument("--split-x-10", type=int, required=True, dest="split_x_10")
    ap.add_argument("--split-x-20", type=int, required=True, dest="split_x_20")
    ap.add_argument("--finish-x", type=int, required=True, dest="finish_x")
    ap.add_argument("--model", default="heavy", choices=["heavy", "lite"])
    ap.add_argument("--conf", type=float, default=0.2)
    args = ap.parse_args()
    r = analyze(args.video, start_x=args.start_x, split_x_10=args.split_x_10,
                split_x_20=args.split_x_20, finish_x=args.finish_x,
                model=args.model, conf=args.conf)
    sp = r["splits_s"]
    print(f"\n10-20-30 m sprint  ->  10m {sp['t_10']}s  20m {sp['t_20']}s  "
          f"30m {sp['t_30']}s")
    print(f"En hizli segment: {r['fastest_segment']}")
    print()
    json.dump(r, sys.stdout, indent=2, ensure_ascii=False)
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
