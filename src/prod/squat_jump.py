"""Squat jump (cift ayak dikey zıplama) — production analizi.

Kisi cift ayakla olduğu yerde dikey olarak ziplar. Ucus suresinden fizik
yontemiyle yükseklik hesaplanır; kalca yukselisi piksel bazli alternatif.

analyze(video_path, height_m) -> dict
CLI: python squat_jump.py <video> <height_m>
"""
import argparse
import json
import sys
from typing import Optional, TypedDict

import numpy as np

try:
    from ..pose_common import iter_pose
except ImportError:  # standalone: python squat_jump.py
    import os as _os
    sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
    from pose_common import iter_pose

NOSE, LEFT_HIP, RIGHT_HIP = 0, 23, 24
LEFT_ANKLE, RIGHT_ANKLE = 27, 28


class SquatJumpResult(TypedDict):
    jump_height_cm: float  # fizik (ucus suresi)
    hip_rise_cm: Optional[float]  # piksel (kalca)
    flight_time_ms: float
    takeoff_frame: int
    landing_frame: int
    peak_hip_frame: int
    fps: float
    px_per_m: Optional[float]


def analyze(video_path: str, height_m: float,
            model: str = "heavy", conf: float = 0.2) -> SquatJumpResult:
    hip_ys, ankle_ys, body_pxs = [], [], []
    fps, H = 30.0, 0
    for fi, t, lm, l3, W, H, fps in iter_pose(video_path, model=model, conf=conf):
        if lm is not None:
            lhip, rhip = lm[LEFT_HIP], lm[RIGHT_HIP]
            la, ra = lm[LEFT_ANKLE], lm[RIGHT_ANKLE]
            nose = lm[NOSE]
            hips = [p for p in (lhip, rhip) if p.visibility > 0.5] or [lhip, rhip]
            ankles = [p for p in (la, ra) if p.visibility > 0.5] or [la, ra]
            hip_ys.append(float(np.mean([p.y for p in hips]) * H))
            ankle_ys.append(float(np.mean([p.y for p in ankles]) * H))
            ankle_y_abs = max(p.y for p in ankles) * H
            body_pxs.append(abs(ankle_y_abs - nose.y * H))
        else:
            for lst in (hip_ys, ankle_ys, body_pxs):
                lst.append(np.nan)

    n = len(hip_ys)
    hip_arr = np.array(hip_ys)
    ankle_arr = np.array(ankle_ys)
    body_arr = np.array(body_pxs)
    valid = np.where(~np.isnan(hip_arr) & ~np.isnan(ankle_arr))[0]
    if valid.size < 10:
        raise RuntimeError("Yeterli pose verisi yok.")

    first_n = max(3, int(valid.size * 0.3))
    gidx = valid[:first_n]
    ground_hip_y = float(np.median(hip_arr[gidx]))
    ground_ankle_y = float(np.median(ankle_arr[gidx]))

    takeoff_thr = ground_ankle_y - 30
    airborne = (~np.isnan(ankle_arr)) & (ankle_arr < takeoff_thr)
    if not airborne.any():
        raise RuntimeError("Havada kare yok.")
    idx = np.where(airborne)[0]
    splits = np.split(idx, np.where(np.diff(idx) > 1)[0] + 1)
    longest = max(splits, key=len)
    takeoff_frame = int(longest[0])
    landing_frame = int(longest[-1])

    # Sub-frame interpolasyon (yere 5px'e kadar)
    thr_fine = ground_ankle_y - 5.0

    def interp(k_in, k_out):
        if not (0 <= k_out < n) or np.isnan(ankle_arr[k_out]) or \
                np.isnan(ankle_arr[k_in]):
            return float(k_in)
        y_in, y_out = ankle_arr[k_in], ankle_arr[k_out]
        if y_in == y_out:
            return float(k_in)
        frac = (thr_fine - y_in) / (y_out - y_in)
        frac = max(0.0, min(1.0, frac))
        return k_in + frac * (k_out - k_in)

    takeoff_sub = interp(takeoff_frame, takeoff_frame - 1)
    landing_sub = interp(landing_frame, landing_frame + 1)
    flight_s = (landing_sub - takeoff_sub) / fps

    g = 9.81
    jump_cm = g * flight_s * flight_s / 8.0 * 100

    peak_hip_frame = int(np.nanargmin(hip_arr))
    peak_hip_y = float(hip_arr[peak_hip_frame])

    px_per_m = None
    hip_rise_cm = None
    body_valid = body_arr[gidx]
    body_valid = body_valid[~np.isnan(body_valid)]
    if body_valid.size > 0:
        px_per_m = float(np.median(body_valid)) / (height_m * 0.87)
        hip_rise_cm = round((ground_hip_y - peak_hip_y) / px_per_m * 100, 1)

    return SquatJumpResult(
        jump_height_cm=round(jump_cm, 1),
        hip_rise_cm=hip_rise_cm,
        flight_time_ms=round(flight_s * 1000, 1),
        takeoff_frame=takeoff_frame,
        landing_frame=landing_frame,
        peak_hip_frame=peak_hip_frame,
        fps=round(fps, 2),
        px_per_m=round(px_per_m, 1) if px_per_m else None,
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Squat jump analizi")
    ap.add_argument("video")
    ap.add_argument("height_m", type=float)
    args = ap.parse_args()
    result = analyze(args.video, args.height_m)
    json.dump(result, sys.stdout, indent=2)
    sys.stdout.write("\n")
