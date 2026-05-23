"""Durarak uzun atlama (broad jump) — production analizi.

Tek fonksiyon: analyze(video_path, height_m) -> dict
CLI: python broad_jump.py <video> <height_m>
"""
import argparse
import json
import sys
from typing import Optional, TypedDict

import numpy as np

try:
    from ..pose_common import iter_pose
except ImportError:  # standalone: python broad_jump.py
    import os as _os
    sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
    from pose_common import iter_pose

NOSE, LEFT_HIP, RIGHT_HIP = 0, 23, 24
LEFT_ANKLE, RIGHT_ANKLE = 27, 28
LEFT_HEEL, RIGHT_HEEL = 29, 30
LEFT_FOOT, RIGHT_FOOT = 31, 32


class BroadJumpResult(TypedDict):
    distance_m: float
    distance_px: float
    flight_time_ms: float
    horizontal_velocity_m_s: float
    apex_height_cm: Optional[float]
    takeoff_angle_deg: Optional[float]
    takeoff_frame: int
    landing_frame: int
    fps: float
    direction: str
    px_per_m: float


def analyze(video_path: str, height_m: float,
            model: str = "heavy", conf: float = 0.2,
            airborne_thr: float = 0.25) -> BroadJumpResult:
    hip_xs, foot_top, foot_ground = [], [], []
    l_foot_xs, r_foot_xs = [], []
    l_heel_xs, r_heel_xs = [], []
    body_pxs = []
    fps, W, H = 30.0, 0, 0
    for fi, t, lm, l3, W, H, fps in iter_pose(video_path, model=model, conf=conf):
        if lm is not None:
            lhip, rhip = lm[LEFT_HIP], lm[RIGHT_HIP]
            lf, rf = lm[LEFT_FOOT], lm[RIGHT_FOOT]
            lhl, rhl = lm[LEFT_HEEL], lm[RIGHT_HEEL]
            nose = lm[NOSE]
            hips = [p for p in (lhip, rhip) if p.visibility > 0.5] or [lhip, rhip]
            hip_xs.append(float(np.mean([p.x for p in hips]) * W))
            l_foot_xs.append(lf.x * W); r_foot_xs.append(rf.x * W)
            l_heel_xs.append(lhl.x * W); r_heel_xs.append(rhl.x * W)
            foot_top.append(min(lf.y, rf.y, lhl.y, rhl.y) * H)
            foot_ground.append(max(lf.y, rf.y, lhl.y, rhl.y) * H)
            foot_y_abs = max(lf.y, rf.y, lhl.y, rhl.y) * H
            body_pxs.append(abs(foot_y_abs - nose.y * H))
        else:
            for lst in (hip_xs, foot_top, foot_ground, l_foot_xs, r_foot_xs,
                        l_heel_xs, r_heel_xs, body_pxs):
                lst.append(np.nan)

    n = len(hip_xs)
    hip_x = np.array(hip_xs)
    foot_top_arr = np.array(foot_top)
    foot_ground_arr = np.array(foot_ground)
    body_arr = np.array(body_pxs)

    # Outlier filtre (yalitik pose hallusinasyonlari)
    for i in range(n):
        if np.isnan(hip_x[i]):
            continue
        nbr = [hip_x[j] for j in range(max(0, i - 3), min(n, i + 4))
               if j != i and not np.isnan(hip_x[j])]
        if len(nbr) < 2 or abs(hip_x[i] - np.median(nbr)) > 150:
            hip_x[i] = foot_top_arr[i] = foot_ground_arr[i] = np.nan

    body_valid = body_arr[~np.isnan(body_arr)]
    if body_valid.size == 0:
        raise RuntimeError("Kalibrasyon icin gecerli pose yok.")
    nose_ankle_px = float(np.percentile(body_valid, 95))
    px_per_m = nose_ankle_px / (height_m * 0.87)

    ground_y = float(np.nanpercentile(foot_ground_arr, 90))
    peak_y = float(np.nanpercentile(foot_top_arr, 2))
    amplitude = ground_y - peak_y
    if amplitude < 20:
        raise RuntimeError(f"Ziplama tespit edilemedi (amp={amplitude:.0f}px)")

    airborne = foot_top_arr < (ground_y - amplitude * airborne_thr)
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

    hip_to = hip_x[pre_to]
    hip_ld = hip_x[post_ld]
    if np.isnan(hip_to) or np.isnan(hip_ld):
        hip_to = float(np.nanmean(hip_x[max(0, takeoff_frame-3):takeoff_frame+1]))
        hip_ld = float(np.nanmean(hip_x[landing_frame:min(n, landing_frame+4)]))
    direction = 1 if hip_ld > hip_to else -1

    lf_x_to, rf_x_to = l_foot_xs[pre_to], r_foot_xs[pre_to]
    takeoff_x = max(lf_x_to, rf_x_to) if direction > 0 else min(lf_x_to, rf_x_to)
    lh_x_ld, rh_x_ld = l_heel_xs[post_ld], r_heel_xs[post_ld]
    landing_x = min(lh_x_ld, rh_x_ld) if direction > 0 else max(lh_x_ld, rh_x_ld)

    jump_px = float(abs(landing_x - takeoff_x))
    jump_m = jump_px / px_per_m

    dt = 1.0 / fps
    if takeoff_frame + 2 < n and not (np.isnan(hip_x[takeoff_frame]) or
                                      np.isnan(hip_x[takeoff_frame + 2])):
        vx_px = (hip_x[takeoff_frame + 2] - hip_x[takeoff_frame]) / (2 * dt)
        vx = float(abs(vx_px) / px_per_m)
    else:
        vx = jump_m / flight_s if flight_s > 0 else 0.0

    g = 9.81
    apex_cm = g * flight_s * flight_s / 8.0 * 100
    if vx > 0.1:
        vy = g * flight_s / 2.0
        angle_deg = float(np.degrees(np.arctan2(vy, vx)))
    else:
        angle_deg = None

    return BroadJumpResult(
        distance_m=round(jump_m, 3),
        distance_px=round(jump_px, 1),
        flight_time_ms=round(flight_s * 1000, 1),
        horizontal_velocity_m_s=round(vx, 3),
        apex_height_cm=round(apex_cm, 1),
        takeoff_angle_deg=round(angle_deg, 1) if angle_deg is not None else None,
        takeoff_frame=takeoff_frame,
        landing_frame=landing_frame,
        fps=round(fps, 2),
        direction="right" if direction > 0 else "left",
        px_per_m=round(px_per_m, 1),
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Broad jump analizi")
    ap.add_argument("video")
    ap.add_argument("height_m", type=float)
    args = ap.parse_args()
    result = analyze(args.video, args.height_m)
    json.dump(result, sys.stdout, indent=2)
    sys.stdout.write("\n")
