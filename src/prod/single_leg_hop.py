"""Tek bacak sicrama (single leg hop) — production analizi.

analyze(video_path, height_m) -> dict
CLI: python single_leg_hop.py <video> <height_m>
"""
import argparse
import json
import sys
from typing import TypedDict

import numpy as np

try:
    from ..pose_common import iter_pose
except ImportError:  # standalone: python single_leg_hop.py
    import os as _os
    sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
    from pose_common import iter_pose

NOSE, LEFT_HIP, RIGHT_HIP = 0, 23, 24
LEFT_ANKLE, RIGHT_ANKLE = 27, 28
LEFT_HEEL, RIGHT_HEEL = 29, 30
LEFT_FOOT, RIGHT_FOOT = 31, 32


class SingleLegHopResult(TypedDict):
    distance_m: float
    distance_px: float
    flight_time_ms: float
    takeoff_side: str
    landing_side: str
    takeoff_frame: int
    landing_frame: int
    fps: float
    direction: str
    px_per_m: float


def analyze(video_path: str, height_m: float,
            model: str = "heavy", conf: float = 0.2,
            airborne_thr: float = 0.25) -> SingleLegHopResult:
    hip_xs, hip_ys = [], []
    l_foot_xs, r_foot_xs = [], []
    l_heel_xs, r_heel_xs = [], []
    l_heel_ys, r_heel_ys = [], []
    foot_top, foot_ground, body_pxs = [], [], []
    fps, W, H = 30.0, 0, 0
    for fi, t, lm, l3, W, H, fps in iter_pose(video_path, model=model, conf=conf):
        if lm is not None:
            lhip, rhip = lm[LEFT_HIP], lm[RIGHT_HIP]
            lf, rf = lm[LEFT_FOOT], lm[RIGHT_FOOT]
            lhl, rhl = lm[LEFT_HEEL], lm[RIGHT_HEEL]
            nose = lm[NOSE]
            hips = [p for p in (lhip, rhip) if p.visibility > 0.5] or [lhip, rhip]
            hip_xs.append(float(np.mean([p.x for p in hips]) * W))
            hip_ys.append(float(np.mean([p.y for p in hips]) * H))
            l_foot_xs.append(lf.x * W); r_foot_xs.append(rf.x * W)
            l_heel_xs.append(lhl.x * W); r_heel_xs.append(rhl.x * W)
            l_heel_ys.append(lhl.y * H); r_heel_ys.append(rhl.y * H)
            foot_top.append(min(lf.y, rf.y, lhl.y, rhl.y) * H)
            foot_ground.append(max(lf.y, rf.y, lhl.y, rhl.y) * H)
            foot_y_abs = max(lf.y, rf.y, lhl.y, rhl.y) * H
            body_pxs.append(abs(foot_y_abs - nose.y * H))
        else:
            for lst in (hip_xs, hip_ys, l_foot_xs, r_foot_xs,
                        l_heel_xs, r_heel_xs, l_heel_ys, r_heel_ys,
                        foot_top, foot_ground, body_pxs):
                lst.append(np.nan)

    n = len(hip_xs)
    hip_x = np.array(hip_xs)
    foot_top_arr = np.array(foot_top)
    foot_ground_arr = np.array(foot_ground)
    body_arr = np.array(body_pxs)

    for i in range(n):
        if np.isnan(hip_x[i]):
            continue
        nbr = [hip_x[j] for j in range(max(0, i - 3), min(n, i + 4))
               if j != i and not np.isnan(hip_x[j])]
        if len(nbr) < 2 or abs(hip_x[i] - np.median(nbr)) > 150:
            hip_x[i] = foot_top_arr[i] = foot_ground_arr[i] = np.nan

    body_valid = body_arr[~np.isnan(body_arr)]
    if body_valid.size == 0:
        raise RuntimeError("Kalibrasyon yok.")
    px_per_m = float(np.percentile(body_valid, 95)) / (height_m * 0.87)

    ground_y = float(np.nanpercentile(foot_ground_arr, 90))
    peak_y = float(np.nanpercentile(foot_ground_arr, 2))
    amplitude = ground_y - peak_y
    if amplitude < 20:
        raise RuntimeError(f"Sicrama tespit edilemedi (amp={amplitude:.0f}px)")

    # Tek bacak: airborne tespiti foot_ground (max y) uzerinden
    airborne = foot_ground_arr < (ground_y - amplitude * airborne_thr)
    if not airborne.any():
        raise RuntimeError("Havada kare yok.")
    idx = np.where(airborne)[0]
    splits = np.split(idx, np.where(np.diff(idx) > 1)[0] + 1)
    longest = max(splits, key=len)
    takeoff_frame = int(longest[0])
    landing_frame = int(longest[-1])

    def find_valid_frame(start, step):
        i = start
        while 0 <= i < n:
            if (not np.isnan(l_foot_xs[i]) and not np.isnan(r_foot_xs[i])
                    and not np.isnan(l_heel_xs[i]) and not np.isnan(r_heel_xs[i])):
                return i
            i += step
        return start
    pre_to = find_valid_frame(max(0, takeoff_frame - 1), -1)
    post_ld = find_valid_frame(min(n - 1, landing_frame + 1), 1)

    flight_s = (landing_frame - takeoff_frame + 1) / fps

    hip_to = hip_x[pre_to] if not np.isnan(hip_x[pre_to]) else \
        float(np.nanmean(hip_x[max(0, takeoff_frame-3):takeoff_frame+1]))
    hip_ld = hip_x[post_ld] if not np.isnan(hip_x[post_ld]) else \
        float(np.nanmean(hip_x[landing_frame:min(n, landing_frame+4)]))
    direction = 1 if hip_ld > hip_to else -1

    # Destek bacak: daha yuksek heel_y (yere yakin) olan ayak
    side_to = "L" if l_heel_ys[pre_to] > r_heel_ys[pre_to] else "R"
    side_ld = "L" if l_heel_ys[post_ld] > r_heel_ys[post_ld] else "R"

    takeoff_x = l_foot_xs[pre_to] if side_to == "L" else r_foot_xs[pre_to]
    landing_x = l_heel_xs[post_ld] if side_ld == "L" else r_heel_xs[post_ld]

    jump_px = float(abs(landing_x - takeoff_x))
    jump_m = jump_px / px_per_m

    return SingleLegHopResult(
        distance_m=round(jump_m, 3),
        distance_px=round(jump_px, 1),
        flight_time_ms=round(flight_s * 1000, 1),
        takeoff_side=side_to,
        landing_side=side_ld,
        takeoff_frame=takeoff_frame,
        landing_frame=landing_frame,
        fps=round(fps, 2),
        direction="right" if direction > 0 else "left",
        px_per_m=round(px_per_m, 1),
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Single leg hop analizi")
    ap.add_argument("video")
    ap.add_argument("height_m", type=float)
    args = ap.parse_args()
    result = analyze(args.video, args.height_m)
    json.dump(result, sys.stdout, indent=2)
    sys.stdout.write("\n")
