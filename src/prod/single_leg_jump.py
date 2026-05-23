"""Tek bacak dikey zıplama (single leg vertical jump) — production analizi.

analyze(video_path, height_m) -> dict
CLI: python single_leg_jump.py <video> <height_m>
"""
import argparse
import json
import sys
from typing import Optional, TypedDict

import numpy as np

try:
    from ..pose_common import iter_pose
except ImportError:  # standalone: python single_leg_jump.py
    import os as _os
    sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
    from pose_common import iter_pose

NOSE, LEFT_HIP, RIGHT_HIP = 0, 23, 24
LEFT_ANKLE, RIGHT_ANKLE = 27, 28
LEFT_HEEL, RIGHT_HEEL = 29, 30


class SingleLegJumpResult(TypedDict):
    jump_height_cm: float  # fizik yontemiyle (uçuş suresi)
    flight_time_ms: float
    takeoff_frame: int
    landing_frame: int
    support_side: str
    hip_rise_cm: Optional[float]
    fps: float
    px_per_m: float


def analyze(video_path: str, height_m: float,
            model: str = "heavy", conf: float = 0.2) -> SingleLegJumpResult:
    hip_ys, body_pxs = [], []
    l_heel_ys, r_heel_ys = [], []
    foot_ground = []
    fps, H = 30.0, 0
    for fi, t, lm, l3, W, H, fps in iter_pose(video_path, model=model, conf=conf):
        if lm is not None:
            lhip, rhip = lm[LEFT_HIP], lm[RIGHT_HIP]
            la, ra = lm[LEFT_ANKLE], lm[RIGHT_ANKLE]
            lhl, rhl = lm[LEFT_HEEL], lm[RIGHT_HEEL]
            nose = lm[NOSE]
            hips = [p for p in (lhip, rhip) if p.visibility > 0.3] or [lhip, rhip]
            hip_ys.append(float(np.mean([p.y for p in hips]) * H))
            l_heel_ys.append(lhl.y * H); r_heel_ys.append(rhl.y * H)
            foot_ground.append(max(la.y, ra.y, lhl.y, rhl.y) * H)
            ankle_y_abs = max(la.y, ra.y) * H
            if nose.visibility > 0.5:
                body_pxs.append(abs(ankle_y_abs - nose.y * H) / 0.87)
            else:
                hip_y_abs = float(np.mean([p.y for p in hips])) * H
                body_pxs.append(abs(ankle_y_abs - hip_y_abs) / 0.52)
        else:
            for lst in (hip_ys, l_heel_ys, r_heel_ys, foot_ground, body_pxs):
                lst.append(np.nan)

    n = len(hip_ys)
    hip_arr = np.array(hip_ys)
    lh_arr = np.array(l_heel_ys)
    rh_arr = np.array(r_heel_ys)
    foot_ground_arr = np.array(foot_ground)
    body_arr = np.array(body_pxs)

    valid_idx = np.where(~np.isnan(foot_ground_arr))[0]
    if valid_idx.size < 10:
        raise RuntimeError("Yeterli pose verisi yok.")

    first_n = max(3, int(valid_idx.size * 0.3))
    gidx = valid_idx[:first_n]
    ground_foot_y = float(np.median(foot_ground_arr[gidx]))
    ground_hip_y = float(np.median(hip_arr[gidx]))

    takeoff_coarse = foot_ground_arr < (ground_foot_y - 30)
    if not takeoff_coarse.any():
        raise RuntimeError("Havada kare bulunamadi.")
    idx = np.where(takeoff_coarse)[0]
    splits = np.split(idx, np.where(np.diff(idx) > 1)[0] + 1)
    longest = max(splits, key=len)
    takeoff_frame = int(longest[0])
    landing_frame = int(longest[-1])

    # Sub-frame: yere 5px'e kadar yaklasmis kare icin interpolasyon
    thr_fine = ground_foot_y - 5.0

    def interp_crossing(k_in, k_out):
        if not (0 <= k_out < n) or np.isnan(foot_ground_arr[k_out]) or \
                np.isnan(foot_ground_arr[k_in]):
            return float(k_in)
        y_in, y_out = foot_ground_arr[k_in], foot_ground_arr[k_out]
        if y_in == y_out:
            return float(k_in)
        frac = (thr_fine - y_in) / (y_out - y_in)
        frac = max(0.0, min(1.0, frac))
        return k_in + frac * (k_out - k_in)

    takeoff_sub = interp_crossing(takeoff_frame, takeoff_frame - 1)
    landing_sub = interp_crossing(landing_frame, landing_frame + 1)
    flight_s = (landing_sub - takeoff_sub) / fps

    g = 9.81
    jump_cm = g * flight_s * flight_s / 8.0 * 100 if flight_s > 0 else 0.0

    pre = max(0, takeoff_frame - 1)
    support = "L" if lh_arr[pre] > rh_arr[pre] else "R"

    px_per_m = None
    hip_rise_cm = None
    body_valid = body_arr[gidx]
    body_valid = body_valid[~np.isnan(body_valid)]
    if body_valid.size > 0:
        body_px = float(np.median(body_valid))
        px_per_m = body_px / height_m
        peak_hip_y = float(np.nanmin(hip_arr[takeoff_frame:landing_frame+1]))
        hip_rise_cm = round((ground_hip_y - peak_hip_y) / px_per_m * 100, 1)

    return SingleLegJumpResult(
        jump_height_cm=round(jump_cm, 1),
        flight_time_ms=round(flight_s * 1000, 1),
        takeoff_frame=takeoff_frame,
        landing_frame=landing_frame,
        support_side=support,
        hip_rise_cm=hip_rise_cm,
        fps=round(fps, 2),
        px_per_m=round(px_per_m, 1) if px_per_m else None,
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Single leg vertical jump analizi")
    ap.add_argument("video")
    ap.add_argument("height_m", type=float)
    args = ap.parse_args()
    result = analyze(args.video, args.height_m)
    json.dump(result, sys.stdout, indent=2)
    sys.stdout.write("\n")
