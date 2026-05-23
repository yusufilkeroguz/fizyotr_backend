"""FMS - In-Line Lunge analizi.

Protokol: tandem durus (on-ayak-topuk arka-ayak-parmak ucu bitisik);
sirtta dowel dik (kafa-torakal-sakrum uc noktadan temas). Lunge yapilir,
arka diz yere degene kadar. Iki tarafa da yapilir.

Skorlama:
    3 = dowel ~dik, gövde dik, arka diz yere degdi, denge korundu
    2 = hafif hizalanma kaybi ama hareket tamamlandi
    1 = tamamlanamaz / denge kaybi
    0 = agri

Yan kamera zorunlu.

CLI:
    python -m fms.in_line_lunge <video> <height_m> [--painful]
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

try:
    import cv2  # opsiyonel: yalnizca video/render icin (CSV-only'da gerekmez)
except ImportError:
    cv2 = None
import numpy as np

from .common import (
    LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_HIP, RIGHT_HIP,
    LEFT_KNEE, RIGHT_KNEE, LEFT_ANKLE, RIGHT_ANKLE,
    LEFT_HEEL, RIGHT_HEEL, LEFT_FOOT, RIGHT_FOOT,
    LEFT_WRIST, RIGHT_WRIST, NOSE,
    iter_pose, pt2d, line_angle_deg, angle_deg,
    smooth_median, px_per_m_from_body,
)


@dataclass
class InLineLungeMetrics:
    bottom_frame: int
    bottom_time_s: float
    front_leg: str                     # "left" veya "right"
    rear_knee_to_ground_cm: float      # arka diz y -> zemin (en yakin heel y)
    rear_knee_touches: bool            # < 5 cm
    torso_tilt_deg: float              # dikeyden sapma
    dowel_tilt_deg: float              # iki bilek cizgisinin dikeyden sapmasi
    torso_upright: bool
    dowel_vertical: bool
    front_knee_angle_deg: float        # kalca-diz-ayakbilegi ici acisi
    fps: float
    px_per_m: float
    score: int
    score_reason: str


def analyze(video_path: str, height_m: float,
            painful: bool = False, model: str = "heavy",
            conf: float = 0.3,
            debug_path: Optional[str] = None) -> dict:
    if painful:
        return {"score": 0, "score_reason": "agri"}

    frames = []
    fps = 30.0; W = H = 0
    for fi, t, l2, l3, W, H, fps in iter_pose(video_path, model=model, conf=conf):
        frames.append((fi, t, l2, l3, W, H))
    n = len(frames)
    if n == 0:
        raise RuntimeError("Video okunamadi.")

    # Bottom kare: kalcanin y'si maksimum (en asagida) = lunge dip
    hip_y = np.full(n, np.nan)
    for k, (_fi, _t, l2, _l3, _W, _H) in enumerate(frames):
        if l2 is None: continue
        hip_y[k] = 0.5 * (l2[LEFT_HIP].y + l2[RIGHT_HIP].y) * H
    hip_y_s = smooth_median(hip_y, 7)
    if not np.isfinite(hip_y_s).any():
        raise RuntimeError("Kalca bulunamadi.")
    bottom_k = int(np.nanargmax(hip_y_s))
    b_fi, b_t, b_l2, _b_l3, _bW, _bH = frames[bottom_k]

    px_per_m = px_per_m_from_body(b_l2, W, H, height_m)

    # On/arka bacak: bottom karede x (yatay) ekseninde daha onde olan = on
    # Yan kamerada on = kameraya hangi yana bakiyorsa (sag/sol)
    la = pt2d(b_l2, LEFT_ANKLE, W, H); ra = pt2d(b_l2, RIGHT_ANKLE, W, H)
    # On bacak = kalcanin X'ine gore daha az mesafede olani? Hayir,
    # lunge'da on-ayak one, arka-ayak geride. Ekrandaki X'e bakalim:
    # kisiyi kameraya gore sagsag veya sollsol duran; yuzu yana donuk.
    # On bacak kalcadan yataya dogru UZAYAN; arka bacak kalcadan
    # geriye UZAYAN. Tibia diziliminden bakalim: on bacakta tibia
    # dikeye yakin (flex dizlik), arka bacakta tibia yataya dogru.
    lkn = pt2d(b_l2, LEFT_KNEE, W, H); rkn = pt2d(b_l2, RIGHT_KNEE, W, H)
    l_tibia_tilt = abs(90 - line_angle_deg(lkn, la))
    l_tibia_tilt = min(l_tibia_tilt, 180 - l_tibia_tilt)
    r_tibia_tilt = abs(90 - line_angle_deg(rkn, ra))
    r_tibia_tilt = min(r_tibia_tilt, 180 - r_tibia_tilt)
    # On bacak = daha dik tibia
    if l_tibia_tilt < r_tibia_tilt:
        front = "left"; FK, FA, FH = LEFT_KNEE, LEFT_ANKLE, LEFT_HEEL
        RK, RA, RH = RIGHT_KNEE, RIGHT_ANKLE, RIGHT_HEEL
    else:
        front = "right"; FK, FA, FH = RIGHT_KNEE, RIGHT_ANKLE, RIGHT_HEEL
        RK, RA, RH = LEFT_KNEE, LEFT_ANKLE, LEFT_HEEL
    FHIP = LEFT_HIP if front == "left" else RIGHT_HIP

    def P(i): return pt2d(b_l2, i, W, H)

    # Arka diz yere mesafesi: arka diz y -> iki heel y maksimum (zemin)
    rk_pt = P(RK); fh_pt = P(FH); rh_pt = P(RH)
    ground_y = float(max(fh_pt[1], rh_pt[1]))
    rear_knee_dist_px = float(ground_y - rk_pt[1])
    rear_knee_dist_cm = 100.0 * rear_knee_dist_px / px_per_m
    rear_knee_touches = bool(rear_knee_dist_cm < 5.0)

    # Torso (dikey)
    ls_pt = P(LEFT_SHOULDER); rs_pt = P(RIGHT_SHOULDER)
    lh_pt = P(LEFT_HIP); rh_pt2 = P(RIGHT_HIP)
    shm = (ls_pt + rs_pt) / 2
    hpm = (lh_pt + rh_pt2) / 2
    torso_line = line_angle_deg(hpm, shm)
    torso_tilt = abs(90.0 - torso_line)
    torso_tilt = min(torso_tilt, 180 - torso_tilt)
    torso_upright = bool(torso_tilt < 15.0)

    # Dowel dikey: sirtta dik dowel tutulur; iki bilek yakin, omuz-uzerinden
    # kafa arkasina dogru. Burada yaklasim: omuz merkezi -> bilek ortalamasi
    # cizgisi dikey mi. 3-nokta temasi degil, kabaca dowel pozisyonu.
    lw_pt = P(LEFT_WRIST); rw_pt = P(RIGHT_WRIST)
    wm = (lw_pt + rw_pt) / 2
    dowel_line = line_angle_deg(shm, wm)
    dowel_tilt = abs(90.0 - dowel_line)
    dowel_tilt = min(dowel_tilt, 180 - dowel_tilt)
    dowel_vertical = bool(dowel_tilt < 20.0)

    # Front diz ici acisi
    fh = P(FHIP); fk = P(FK); fa = P(FA)
    front_knee_angle = angle_deg(fh, fk, fa)

    # Skorlama
    if rear_knee_touches and torso_upright and dowel_vertical:
        score, reason = 3, "tum kriterler saglandi"
    elif rear_knee_touches:
        fails = []
        if not torso_upright: fails.append("govde egik")
        if not dowel_vertical: fails.append("dowel egik")
        score, reason = 2, "lunge tamamlandi; eksik: " + ", ".join(fails)
    else:
        score, reason = 1, f"arka diz yere ulasmadi ({rear_knee_dist_cm:.1f} cm)"

    result = InLineLungeMetrics(
        bottom_frame=b_fi, bottom_time_s=round(b_t, 3),
        front_leg=front,
        rear_knee_to_ground_cm=round(rear_knee_dist_cm, 1),
        rear_knee_touches=rear_knee_touches,
        torso_tilt_deg=round(torso_tilt, 1),
        dowel_tilt_deg=round(dowel_tilt, 1),
        torso_upright=torso_upright,
        dowel_vertical=dowel_vertical,
        front_knee_angle_deg=round(front_knee_angle, 1),
        fps=round(fps, 2),
        px_per_m=round(px_per_m, 1),
        score=score, score_reason=reason,
    )

    if debug_path:
        _draw_debug(video_path, bottom_k, result, frames, Path(debug_path))
    return asdict(result)


def _draw_debug(video_path, bottom_k, m, frames, out_path):
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frames[bottom_k][0])
    ok, img = cap.read(); cap.release()
    if not ok: return
    H, W = img.shape[:2]
    _fi, _t, l2, _l3, _W, _H = frames[bottom_k]
    if l2 is None:
        cv2.imwrite(str(out_path), img); return

    front = m.front_leg
    FK = LEFT_KNEE if front == "left" else RIGHT_KNEE
    FA = LEFT_ANKLE if front == "left" else RIGHT_ANKLE
    FH_ = LEFT_HEEL if front == "left" else RIGHT_HEEL
    FHIP = LEFT_HIP if front == "left" else RIGHT_HIP
    RK = RIGHT_KNEE if front == "left" else LEFT_KNEE
    RA = RIGHT_ANKLE if front == "left" else LEFT_ANKLE
    RH_ = RIGHT_HEEL if front == "left" else LEFT_HEEL

    def P(i): return tuple(pt2d(l2, i, W, H).astype(int))
    fk, fa, fhl = P(FK), P(FA), P(FH_)
    rk, ra, rhl = P(RK), P(RA), P(RH_)
    ls, rs = P(LEFT_SHOULDER), P(RIGHT_SHOULDER)
    lh, rh = P(LEFT_HIP), P(RIGHT_HIP)
    lw, rw = P(LEFT_WRIST), P(RIGHT_WRIST)
    fhip_p = P(FHIP)

    ground_y = max(fhl[1], rhl[1])
    cv2.line(img, (0, ground_y), (W, ground_y), (200, 200, 100), 2, cv2.LINE_AA)

    # Arka diz - zemin cizgisi
    cv2.line(img, rk, (rk[0], ground_y), (180, 80, 255), 4, cv2.LINE_AA)
    cv2.putText(img, f"{m.rear_knee_to_ground_cm} cm", (rk[0]+10, rk[1]+20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (180, 80, 255), 2, cv2.LINE_AA)

    # Torso + dowel + front tibia
    shm = ((ls[0]+rs[0])//2, (ls[1]+rs[1])//2)
    hpm = ((lh[0]+rh[0])//2, (lh[1]+rh[1])//2)
    wm = ((lw[0]+rw[0])//2, (lw[1]+rw[1])//2)
    cv2.line(img, hpm, shm, (0, 200, 255), 5, cv2.LINE_AA)         # torso
    cv2.line(img, shm, wm, (255, 200, 0), 5, cv2.LINE_AA)          # dowel
    cv2.line(img, fk, fa, (0, 150, 255), 5, cv2.LINE_AA)           # front tibia
    cv2.line(img, fhip_p, fk, (0, 255, 100), 5, cv2.LINE_AA)

    def txt(s, y, color=(255, 255, 255)):
        cv2.putText(img, s, (30, y), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                    (0, 0, 0), 5, cv2.LINE_AA)
        cv2.putText(img, s, (30, y), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                    color, 2, cv2.LINE_AA)
    status = lambda ok: (60, 220, 60) if ok else (60, 60, 220)
    txt(f"IN-LINE LUNGE  skor = {m.score}  (on bacak={m.front_leg})",
        50, (0, 255, 255))
    txt(f"arka diz yerde ({m.rear_knee_to_ground_cm} cm): {m.rear_knee_touches}",
        95, status(m.rear_knee_touches))
    txt(f"govde dik (tilt={m.torso_tilt_deg}°): {m.torso_upright}",
        135, status(m.torso_upright))
    txt(f"dowel dik (tilt={m.dowel_tilt_deg}°): {m.dowel_vertical}",
        175, status(m.dowel_vertical))
    cv2.imwrite(str(out_path), img)


def main():
    ap = argparse.ArgumentParser(description="FMS In-Line Lunge analizi")
    ap.add_argument("video")
    ap.add_argument("height_m", type=float)
    ap.add_argument("--painful", action="store_true")
    ap.add_argument("--model", default="heavy", choices=["heavy", "lite"])
    ap.add_argument("--debug", default=None)
    args = ap.parse_args()
    r = analyze(args.video, args.height_m, painful=args.painful,
                model=args.model, debug_path=args.debug)
    print(json.dumps(r, indent=2, ensure_ascii=False,
                     default=lambda o: float(o)))


if __name__ == "__main__":
    main()
