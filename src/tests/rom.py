"""ROM — Eklem Hareket Acikligi (Range of Motion) analizi.

Bir video boyunca secilen eklemin acisini kare kare olcer ve min / maks / ROM
(aralik) degerlerini raporlar. Genel amacli: diz, kalca, dirsek, omuz, ayak
bilegi fleksiyon/ekstansiyonu veya govde fleksiyonu.

Eklem -> kullanilan landmark uclusu (B = tepe noktasi):
    knee     : kalca - DIZ - ayak bilegi
    hip      : omuz - KALCA - diz
    elbow    : omuz - DIRSEK - bilek
    shoulder : kalca - OMUZ - dirsek
    ankle    : diz - AYAK BILEGI - ayak ucu
    trunk    : govde cizgisinin (kalca->omuz) dikeyden sapmasi

Kamera: YAN kamera, olculen eklem net gorunur olmali.

CLI:
    python rom.py video.mp4 --joint knee --side auto --debug rom_dbg
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

import numpy as np

from ..pose_common import (
    iter_pose, pt2d, angle_deg, vertical_tilt_deg, smooth_median, pick_visible_side,
    LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_ELBOW, RIGHT_ELBOW,
    LEFT_WRIST, RIGHT_WRIST, LEFT_HIP, RIGHT_HIP, LEFT_KNEE, RIGHT_KNEE,
    LEFT_ANKLE, RIGHT_ANKLE, LEFT_FOOT, RIGHT_FOOT,
)

# joint -> (A, B, C) landmark indeksleri (sol taraf; sag icin +1 ofsetli es)
_JOINT_TRIPLETS = {
    "knee":     (LEFT_HIP, LEFT_KNEE, LEFT_ANKLE),
    "hip":      (LEFT_SHOULDER, LEFT_HIP, LEFT_KNEE),
    "elbow":    (LEFT_SHOULDER, LEFT_ELBOW, LEFT_WRIST),
    "shoulder": (LEFT_HIP, LEFT_SHOULDER, LEFT_ELBOW),
    "ankle":    (LEFT_KNEE, LEFT_ANKLE, LEFT_FOOT),
}
# sol -> sag landmark eslestirmesi
_LR = {
    LEFT_SHOULDER: RIGHT_SHOULDER, LEFT_ELBOW: RIGHT_ELBOW,
    LEFT_WRIST: RIGHT_WRIST, LEFT_HIP: RIGHT_HIP, LEFT_KNEE: RIGHT_KNEE,
    LEFT_ANKLE: RIGHT_ANKLE, LEFT_FOOT: RIGHT_FOOT,
}
_SIDE_PICK = {  # auto secimi icin temsil eden landmark cifti
    "knee": (LEFT_KNEE, RIGHT_KNEE), "hip": (LEFT_HIP, RIGHT_HIP),
    "elbow": (LEFT_ELBOW, RIGHT_ELBOW), "shoulder": (LEFT_SHOULDER, RIGHT_SHOULDER),
    "ankle": (LEFT_ANKLE, RIGHT_ANKLE),
}


def analyze(video_path: str, *, joint: str = "knee", side: str = "auto",
            model: str = "heavy", conf: float = 0.3,
            debug_dir: Optional[str] = None) -> dict:
    joint = joint.lower()
    if joint != "trunk" and joint not in _JOINT_TRIPLETS:
        raise ValueError(f"Bilinmeyen joint: {joint}")

    angles = []
    fps = 30.0
    sides_used = []
    for fi, t, l2, l3, W, H, fps in iter_pose(video_path, model=model, conf=conf):
        if l2 is None:
            angles.append(np.nan)
            sides_used.append(None)
            continue
        if joint == "trunk":
            hip = (pt2d(l2, LEFT_HIP, W, H) + pt2d(l2, RIGHT_HIP, W, H)) / 2
            sh = (pt2d(l2, LEFT_SHOULDER, W, H) + pt2d(l2, RIGHT_SHOULDER, W, H)) / 2
            angles.append(vertical_tilt_deg(hip, sh))
            sides_used.append("center")
            continue
        # taraf sec
        if side == "auto":
            lp, rp = _SIDE_PICK[joint]
            cur_side = pick_visible_side(l2, lp, rp)
        else:
            cur_side = side
        a, b, c = _JOINT_TRIPLETS[joint]
        if cur_side == "right":
            a, b, c = _LR[a], _LR[b], _LR[c]
        ang = angle_deg(pt2d(l2, a, W, H), pt2d(l2, b, W, H), pt2d(l2, c, W, H))
        angles.append(ang)
        sides_used.append(cur_side)

    arr = smooth_median(np.array(angles), 5)
    valid = arr[~np.isnan(arr)]
    if valid.size < 3:
        raise RuntimeError("Eklem acisi takip edilemedi.")
    min_i = int(np.nanargmin(arr))
    max_i = int(np.nanargmax(arr))
    min_a = float(arr[min_i])
    max_a = float(arr[max_i])

    # baskin taraf
    side_counts = {}
    for s in sides_used:
        if s:
            side_counts[s] = side_counts.get(s, 0) + 1
    dom_side = max(side_counts, key=side_counts.get) if side_counts else side

    result = {
        "fps": round(fps, 2),
        "joint": joint,
        "side": dom_side,
        "min_angle_deg": round(min_a, 1),
        "max_angle_deg": round(max_a, 1),
        "rom_deg": round(max_a - min_a, 1),
        "mean_angle_deg": round(float(np.nanmean(arr)), 1),
        "min_frame": min_i,
        "min_time_s": round(min_i / fps, 3),
        "max_frame": max_i,
        "max_time_s": round(max_i / fps, 3),
        "n_valid_frames": int(valid.size),
    }
    if debug_dir:
        _draw_debug(video_path, result, Path(debug_dir))
    return result


def _draw_debug(video, result, out_dir: Path):
    import cv2
    out_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(video)
    for tag, fr, ang in (("min", result["min_frame"], result["min_angle_deg"]),
                         ("max", result["max_frame"], result["max_angle_deg"])):
        cap.set(cv2.CAP_PROP_POS_FRAMES, fr)
        ok, img = cap.read()
        if not ok:
            continue
        label = (f"{result['joint']} ({result['side']}) {tag}={ang} deg  "
                 f"ROM={result['rom_deg']} deg")
        cv2.putText(img, label, (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                    (0, 0, 0), 6, cv2.LINE_AA)
        cv2.putText(img, label, (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                    (0, 255, 255), 2, cv2.LINE_AA)
        cv2.imwrite(str(out_dir / f"rom_{result['joint']}_{tag}.jpg"), img)
    cap.release()


def main():
    ap = argparse.ArgumentParser(description="ROM (eklem hareket acikligi)")
    ap.add_argument("video")
    ap.add_argument("--joint", default="knee",
                    choices=["knee", "hip", "elbow", "shoulder", "ankle", "trunk"])
    ap.add_argument("--side", default="auto", choices=["auto", "left", "right"])
    ap.add_argument("--model", default="heavy", choices=["heavy", "lite"])
    ap.add_argument("--conf", type=float, default=0.3)
    ap.add_argument("--debug", default=None)
    args = ap.parse_args()
    r = analyze(args.video, joint=args.joint, side=args.side,
                model=args.model, conf=args.conf, debug_dir=args.debug)
    print(f"\nROM {r['joint']} ({r['side']})  ->  "
          f"min {r['min_angle_deg']} deg / max {r['max_angle_deg']} deg / "
          f"ROM {r['rom_deg']} deg")
    print(f"min @ t={r['min_time_s']}s (#{r['min_frame']}), "
          f"max @ t={r['max_time_s']}s (#{r['max_frame']})")
    print()
    json.dump(r, sys.stdout, indent=2, ensure_ascii=False)
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
