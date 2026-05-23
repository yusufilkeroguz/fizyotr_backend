"""FMS - Hurdle Step (engel adimi) analizi.

Protokol: denek dowel'i omuzlar uzerinde YATAY tutar; tibial-tuberosity
hizasinda kurulmus bir engelin karsisinda durup bir bacagini kaldirir,
engelin ustunden gecirir, karsi tarafta topugu yere degdirir (ya da
degdirmeden geri doner, FMS tarifine baglidir). Kamera yan.

Bu analizci yan-gorus (sagittal duzlem) varsayar. Pose visibility'den
hangi yanin gorundugu otomatik secilir.

Skorlama:
    3 = hizalanma korunur, dowel ~yatay, gövde ~dik, diz engeli gecer
    2 = hizalanma kaybolur (dowel egilir, gövde egilir) ama engel gecilir
    1 = engeli gecemez ya da ayak dokunarak geciyor (tibia egrisi yok)
    0 = agri (manuel)

CLI:
    python -m fms.hurdle_step <video> <height_m> [--painful]
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
    LEFT_WRIST, RIGHT_WRIST,
    iter_pose, pt2d, line_angle_deg,
    smooth_median, px_per_m_from_body,
)


@dataclass
class HurdleStepMetrics:
    peak_frame: int
    peak_time_s: float
    moving_leg: str                    # "left" / "right"
    stance_leg: str
    # Pik-karede olculen metrikler
    knee_lift_cm: float                # moving knee'nin stance ankle'a gore yuksekligi
    hurdle_height_cm: float            # engel hizasi = stance leg tibia uzunlugu
    stance_tibia_tilt_deg: float       # dikeyden sapma (0=dik)
    torso_tilt_deg: float              # dikeyden sapma
    dowel_tilt_deg: float              # yataydan sapma (iki bilek cizgisi)
    # Kriterler
    clears_hurdle: bool                # diz >= engel hizasi
    torso_upright: bool                # govde tilt < 10
    stance_stable: bool                # stance tibia tilt < 10
    dowel_level: bool                  # dowel tilt < 10
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

    # Zaman serileri: diz/ayak bilegi yukseklikleri
    lkn_y = np.full(n, np.nan); rkn_y = np.full(n, np.nan)
    lan_y = np.full(n, np.nan); ran_y = np.full(n, np.nan)
    for k, (_fi, _t, l2, _l3, _W, _H) in enumerate(frames):
        if l2 is None: continue
        lkn_y[k] = l2[LEFT_KNEE].y * H
        rkn_y[k] = l2[RIGHT_KNEE].y * H
        lan_y[k] = l2[LEFT_ANKLE].y * H
        ran_y[k] = l2[RIGHT_ANKLE].y * H

    # Moving leg: video boyunca diz'in (ankle'a gore) en cok yukari cikani
    # Pik yukseklik = ankle_y - knee_y (piksel). Daha buyuk = daha yukari.
    # Iki bacaktan hangisinin pik lifti buyukse o moving.
    lift_L = np.nanmax(lan_y - lkn_y) if np.isfinite(lan_y - lkn_y).any() else 0.0
    lift_R = np.nanmax(ran_y - rkn_y) if np.isfinite(ran_y - rkn_y).any() else 0.0
    if lift_L >= lift_R:
        moving = "left"
        kn_y = lkn_y
        MK, MA = LEFT_KNEE, LEFT_ANKLE
        SK, SA = RIGHT_KNEE, RIGHT_ANKLE
    else:
        moving = "right"
        kn_y = rkn_y
        MK, MA = RIGHT_KNEE, RIGHT_ANKLE
        SK, SA = LEFT_KNEE, LEFT_ANKLE
    stance = "right" if moving == "left" else "left"

    # Peak frame: stance ayak bileginden en fazla yukarida olan moving knee
    # Stance ayak bilegi zamanla sabit (yerde); peak = max(stance_ank_y - kn_y)
    st_an_y = ran_y if moving == "left" else lan_y
    lift = st_an_y - kn_y                 # + = knee stance ayak uzerinde
    lift_sm = smooth_median(lift, 7)
    if not np.isfinite(lift_sm).any():
        raise RuntimeError("Diz yuksekligi hesaplanamadi.")
    peak_k = int(np.nanargmax(lift_sm))
    b_fi, b_t, b_l2, _b_l3, _bW, _bH = frames[peak_k]

    px_per_m = px_per_m_from_body(b_l2, W, H, height_m)

    # Peak kare noktalari (piksel)
    def P(i): return pt2d(b_l2, i, W, H)
    mk_p = P(MK)
    sk_p, sa_p = P(SK), P(SA)
    lh_p = P(LEFT_HIP); rh_p = P(RIGHT_HIP)
    ls_p = P(LEFT_SHOULDER); rs_p = P(RIGHT_SHOULDER)
    lw_p = P(LEFT_WRIST); rw_p = P(RIGHT_WRIST)

    # Moving knee'nin stance-ankle'a goreli yuksekligi
    knee_lift_px = float(sa_p[1] - mk_p[1])  # + = knee stance ankle ustunde
    knee_lift_cm = 100.0 * knee_lift_px / px_per_m

    # Engel hizasi = stance leg tibia uzunlugu (stance diz - stance ayak bilegi)
    hurdle_px = float(sa_p[1] - sk_p[1])
    hurdle_cm = 100.0 * hurdle_px / px_per_m

    # Stance tibia egimi (dikeyden sapma)
    stance_tibia_line = line_angle_deg(sk_p, sa_p)   # 0..180 yatay=0, dikey=90
    stance_tibia_tilt = abs(90.0 - stance_tibia_line)
    stance_tibia_tilt = min(stance_tibia_tilt, 180 - stance_tibia_tilt)

    # Gövde egimi: omuz ortasi - kalca ortasi
    shm = (ls_p + rs_p) / 2
    hpm = (lh_p + rh_p) / 2
    torso_line = line_angle_deg(hpm, shm)
    torso_tilt = abs(90.0 - torso_line)
    torso_tilt = min(torso_tilt, 180 - torso_tilt)

    # Dowel egimi: iki bilek cizgisi yataya gore
    dowel_line = line_angle_deg(lw_p, rw_p)
    dowel_tilt = min(dowel_line, 180 - dowel_line)

    # Kriterler
    clears_hurdle = bool(knee_lift_px > 0.9 * hurdle_px)   # diz >= 0.9 * engel
    torso_upright = bool(torso_tilt < 10.0)
    stance_stable = bool(stance_tibia_tilt < 10.0)
    dowel_level = bool(dowel_tilt < 10.0)

    fails = []
    if not clears_hurdle: fails.append("diz engeli gecmiyor")
    if not torso_upright: fails.append("govde egik")
    if not stance_stable: fails.append("stance tibia egik")
    if not dowel_level: fails.append("dowel egik")

    if clears_hurdle and torso_upright and stance_stable and dowel_level:
        score, reason = 3, "tum kriterler saglandi"
    elif clears_hurdle:
        score, reason = 2, "engel gecti; eksik: " + ", ".join(fails)
    else:
        score, reason = 1, "engel gecilemedi / hizalanma yok: " + ", ".join(fails)

    result = HurdleStepMetrics(
        peak_frame=b_fi, peak_time_s=round(b_t, 3),
        moving_leg=moving, stance_leg=stance,
        knee_lift_cm=round(knee_lift_cm, 1),
        hurdle_height_cm=round(hurdle_cm, 1),
        stance_tibia_tilt_deg=round(stance_tibia_tilt, 1),
        torso_tilt_deg=round(torso_tilt, 1),
        dowel_tilt_deg=round(dowel_tilt, 1),
        clears_hurdle=clears_hurdle,
        torso_upright=torso_upright,
        stance_stable=stance_stable,
        dowel_level=dowel_level,
        fps=round(fps, 2),
        px_per_m=round(px_per_m, 1),
        score=score, score_reason=reason,
    )

    if debug_path:
        _draw_debug(video_path, peak_k, result, frames, Path(debug_path))
    return asdict(result)


def _draw_debug(video_path, peak_k, m, frames, out_path):
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frames[peak_k][0])
    ok, img = cap.read(); cap.release()
    if not ok: return
    H, W = img.shape[:2]
    _fi, _t, l2, _l3, _W, _H = frames[peak_k]
    if l2 is None:
        cv2.imwrite(str(out_path), img); return

    moving = m.moving_leg
    MK = LEFT_KNEE if moving == "left" else RIGHT_KNEE
    MA = LEFT_ANKLE if moving == "left" else RIGHT_ANKLE
    SK = RIGHT_KNEE if moving == "left" else LEFT_KNEE
    SA = RIGHT_ANKLE if moving == "left" else LEFT_ANKLE

    def P(i): return tuple(pt2d(l2, i, W, H).astype(int))
    mk, ma = P(MK), P(MA)
    sk, sa = P(SK), P(SA)
    ls, rs = P(LEFT_SHOULDER), P(RIGHT_SHOULDER)
    lh, rh = P(LEFT_HIP), P(RIGHT_HIP)
    lw, rw = P(LEFT_WRIST), P(RIGHT_WRIST)

    # Engel hizasi cizgisi (stance knee y seviyesi yok, stance tibia uzunlugu
    # yerine stance knee y'sinin altina cizelim)
    hurdle_y = sk[1]
    cv2.line(img, (0, hurdle_y), (W, hurdle_y), (60, 180, 220), 3, cv2.LINE_AA)
    cv2.putText(img, f"engel hizasi ({m.hurdle_height_cm} cm)",
                (20, hurdle_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                (60, 180, 220), 2, cv2.LINE_AA)

    # Moving diz yuksekligi
    cv2.line(img, (0, mk[1]), (W, mk[1]), (80, 220, 80), 2, cv2.LINE_AA)
    cv2.putText(img, f"diz ({m.knee_lift_cm} cm)",
                (W - 260, mk[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                (80, 220, 80), 2, cv2.LINE_AA)

    # Govde ve dowel
    shm = ((ls[0]+rs[0])//2, (ls[1]+rs[1])//2)
    hpm = ((lh[0]+rh[0])//2, (lh[1]+rh[1])//2)
    cv2.line(img, hpm, shm, (0, 200, 255), 5, cv2.LINE_AA)
    cv2.line(img, lw, rw, (255, 200, 0), 5, cv2.LINE_AA)
    # Stance tibia
    cv2.line(img, sk, sa, (0, 150, 255), 5, cv2.LINE_AA)
    # Moving leg
    cv2.line(img, mk, ma, (180, 80, 255), 5, cv2.LINE_AA)

    def txt(s, y, color=(255, 255, 255)):
        cv2.putText(img, s, (30, y), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                    (0, 0, 0), 5, cv2.LINE_AA)
        cv2.putText(img, s, (30, y), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                    color, 2, cv2.LINE_AA)
    status = lambda ok: (60, 220, 60) if ok else (60, 60, 220)
    txt(f"HURDLE STEP  skor = {m.score}  ({m.moving_leg} bacak hareket)",
        50, (0, 255, 255))
    txt(f"engel gecti: {m.clears_hurdle}", 95, status(m.clears_hurdle))
    txt(f"govde dik (tilt={m.torso_tilt_deg}°): {m.torso_upright}",
        135, status(m.torso_upright))
    txt(f"stance tibia dik (tilt={m.stance_tibia_tilt_deg}°): {m.stance_stable}",
        175, status(m.stance_stable))
    txt(f"dowel yatay (tilt={m.dowel_tilt_deg}°): {m.dowel_level}",
        215, status(m.dowel_level))
    cv2.imwrite(str(out_path), img)


def main():
    ap = argparse.ArgumentParser(description="FMS Hurdle Step analizi")
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
