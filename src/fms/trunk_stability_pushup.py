"""FMS - Trunk Stability Push-Up analizi.

Protokol: kisi yerde prone, basparmaklar uygun hizada (erkek=omuz,
kadin=cene). Gövde tek parca halinde push-up; omurgada sag-sol
egrilik olmamali, lomber ekstansiyon senkron kalkmali.

Skorlama:
    3 = tam hizalanma ile push-up tamamlandi (erkek omuz hizasi, kadin
        cene hizasindan)
    2 = ayni alignment bir-alt-seviyeden (erkek cene, kadin klavikula)
        tamamlandi
    1 = tamamlayamaz
    0 = spinal ekstansiyon clearing pozitif (manuel)

Bu analizci yan kamera ile:
    - Kalk fazini (max vucut yukselis) bulur
    - Omuz-kalca-diz cizgisinin DUZ (cokme veya cukur yok) olup olmadigini
      kontrol eder
    - Tam push-up tamamlandi mi (dirsek ~duz, omuz yerden yuksek)
    - Thumb-pozisyonu otomatik bilinmez; flag ile verilebilir veya
      varsayim=3 (en ust).

CLI:
    python -m fms.trunk_stability_pushup <video> <height_m>
        [--sex m|f] [--level top|lower|cant] [--painful]
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
    NOSE, LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_HIP, RIGHT_HIP,
    LEFT_KNEE, RIGHT_KNEE, LEFT_ANKLE, RIGHT_ANKLE,
    iter_pose, pt2d, ptw, angle_deg, smooth_median,
)


@dataclass
class TSPushUpMetrics:
    top_frame: int
    top_time_s: float
    side_used: str
    body_line_angle_diff_deg: float    # omuz-kalca vs kalca-diz acilarinin farki
    body_aligned: bool                 # tek parca dik hat
    hip_sag_cm: float                  # kalca y - (omuz+diz)/2 y (pozitif = cukur)
    pushup_completed: bool             # kalkis gerceklesmis mi (omuz yerden yuksek)
    fps: float
    px_per_m: float
    score: int
    score_reason: str


def analyze(video_path: str, height_m: float,
            sex: str = "m",            # "m" / "f" - thumb level default
            level: str = "lower",      # "top" / "lower" / "cant"
            painful: bool = False,
            model: str = "heavy", conf: float = 0.3,
            debug_path: Optional[str] = None) -> dict:
    if painful:
        return {"score": 0, "score_reason": "agri / clearing"}

    frames = []
    fps = 30.0; W = H = 0
    for fi, t, l2, l3, W, H, fps in iter_pose(video_path, model=model, conf=conf):
        frames.append((fi, t, l2, l3, W, H))
    n = len(frames)
    if n == 0:
        raise RuntimeError("Video okunamadi.")

    # Top kare: omuz y minimum (imajda en yukarida) = push-up uste
    sh_y = np.full(n, np.nan)
    for k, (_fi, _t, l2, _l3, _W, _H) in enumerate(frames):
        if l2 is None: continue
        sh_y[k] = 0.5 * (l2[LEFT_SHOULDER].y + l2[RIGHT_SHOULDER].y) * H
    sh_y_s = smooth_median(sh_y, 7)
    if not np.isfinite(sh_y_s).any():
        raise RuntimeError("Omuz bulunamadi.")
    top_k = int(np.nanargmin(sh_y_s))
    b_fi, b_t, b_l2, b_l3, _bW, _bH = frames[top_k]
    if b_l3 is None:
        raise RuntimeError("3D pose bulunamadi.")

    # 3D metrikler (hip-merkezli, metre). Kamera acisina duyarsiz.
    shm3 = 0.5 * (ptw(b_l3, LEFT_SHOULDER) + ptw(b_l3, RIGHT_SHOULDER))
    hpm3 = 0.5 * (ptw(b_l3, LEFT_HIP) + ptw(b_l3, RIGHT_HIP))
    knm3 = 0.5 * (ptw(b_l3, LEFT_KNEE) + ptw(b_l3, RIGHT_KNEE))
    anm3 = 0.5 * (ptw(b_l3, LEFT_ANKLE) + ptw(b_l3, RIGHT_ANKLE))

    # Body line: omuz-kalca-diz ici acisi (180° = duz vucut).
    bend = angle_deg(shm3, hpm3, knm3)
    diff = abs(180.0 - bend)
    body_aligned = bool(diff < 20.0)

    # Kalca sarkmasi: omuz-diz dogrusuna gore kalca uzakligi (metre)
    seg = knm3 - shm3
    seg_n = float(np.linalg.norm(seg))
    if seg_n < 1e-6:
        hip_sag_m = 0.0
    else:
        # kalca noktasi ile [shm3,knm3] dogrusu arasi dik mesafe
        t = float(np.dot(hpm3 - shm3, seg) / (seg_n**2))
        foot = shm3 + t * seg
        hip_sag_m = float(np.linalg.norm(hpm3 - foot))
    hip_sag_cm = 100.0 * hip_sag_m

    # Push-up completed: omuz<->ankle 3D mesafe rest'e gore yeterince buyuk
    # (vucut duzlemde uzaniyor). Burada kolay kontrol: omuz-ankle 3D mesafe
    # > 0.8 * height.
    sh_an_m = float(np.linalg.norm(shm3 - anm3))
    pushup_completed = bool(sh_an_m > 0.6 * height_m)

    # Hangi yan yalnizca debug icin
    lvis = b_l2[LEFT_HIP].visibility; rvis = b_l2[RIGHT_HIP].visibility
    side = "left" if lvis >= rvis else "right"

    # Gorsel debug icin 2D kalibrasyon (yatay vucut) - metriklere etki etmez
    nose2 = pt2d(b_l2, NOSE, W, H)
    la2 = pt2d(b_l2, LEFT_ANKLE, W, H); ra2 = pt2d(b_l2, RIGHT_ANKLE, W, H)
    an2 = la2 if b_l2[LEFT_ANKLE].visibility >= b_l2[RIGHT_ANKLE].visibility else ra2
    px_per_m = float(np.linalg.norm(nose2 - an2)) / (height_m * 0.87)

    # Skorlama
    if not pushup_completed:
        score = 1; reason = "push-up tamamlanamadi"
    elif level == "cant":
        score = 1; reason = "kullanici: tamamlanamadi"
    elif not body_aligned or hip_sag_cm > 10.0:
        score = 1; reason = (
            f"gövde tek parca degil (aci fark={diff:.1f}°, "
            f"kalca sark={hip_sag_cm:.1f} cm)"
        )
    elif level == "top":
        score = 3; reason = f"{sex}: ust seviyeden tek parca"
    else:   # level == "lower" veya varsayim
        score = 2; reason = f"{sex}: alt seviyeden tek parca"

    result = TSPushUpMetrics(
        top_frame=b_fi, top_time_s=round(b_t, 3),
        side_used=side,
        body_line_angle_diff_deg=round(diff, 1),
        body_aligned=body_aligned,
        hip_sag_cm=round(hip_sag_cm, 1),
        pushup_completed=pushup_completed,
        fps=round(fps, 2), px_per_m=round(px_per_m, 1),
        score=score, score_reason=reason,
    )

    if debug_path:
        _draw_debug(video_path, top_k, result, frames, Path(debug_path))
    return asdict(result)


def _draw_debug(video_path, top_k, m, frames, out_path):
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frames[top_k][0])
    ok, img = cap.read(); cap.release()
    if not ok: return
    H, W = img.shape[:2]
    _fi, _t, l2, _l3, _W, _H = frames[top_k]
    if l2 is None:
        cv2.imwrite(str(out_path), img); return

    side = m.side_used
    SH = LEFT_SHOULDER if side == "left" else RIGHT_SHOULDER
    HP = LEFT_HIP if side == "left" else RIGHT_HIP
    KN = LEFT_KNEE if side == "left" else RIGHT_KNEE
    AN = LEFT_ANKLE if side == "left" else RIGHT_ANKLE
    def P(i): return tuple(pt2d(l2, i, W, H).astype(int))
    sh, hp, kn, an = P(SH), P(HP), P(KN), P(AN)

    # Vücut cizgisi (omuz-kalca-diz-ayak)
    cv2.line(img, sh, hp, (0, 200, 255), 5, cv2.LINE_AA)
    cv2.line(img, hp, kn, (0, 255, 100), 5, cv2.LINE_AA)
    cv2.line(img, kn, an, (180, 80, 255), 5, cv2.LINE_AA)
    # Referans duz cizgi: omuzdan dize dogru duz
    cv2.line(img, sh, kn, (180, 180, 180), 2, cv2.LINE_AA)

    def txt(s, y, color=(255, 255, 255)):
        cv2.putText(img, s, (30, y), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                    (0, 0, 0), 5, cv2.LINE_AA)
        cv2.putText(img, s, (30, y), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                    color, 2, cv2.LINE_AA)
    status = lambda ok: (60, 220, 60) if ok else (60, 60, 220)
    txt(f"TRUNK STABILITY PUSH-UP  skor = {m.score}",
        50, (0, 255, 255))
    txt(f"govde tek parca (fark={m.body_line_angle_diff_deg}°): "
        f"{m.body_aligned}", 95, status(m.body_aligned))
    txt(f"kalca sarkmasi = {m.hip_sag_cm} cm",
        135, status(m.hip_sag_cm < 5.0))
    txt(f"push-up tamamlandi: {m.pushup_completed}",
        175, status(m.pushup_completed))
    cv2.imwrite(str(out_path), img)


def main():
    ap = argparse.ArgumentParser(description="FMS Trunk Stability Push-Up")
    ap.add_argument("video")
    ap.add_argument("height_m", type=float)
    ap.add_argument("--sex", default="m", choices=["m", "f"])
    ap.add_argument("--level", default="lower",
                    choices=["top", "lower", "cant"],
                    help="top=en ust el pozisyonu, lower=bir alt seviye")
    ap.add_argument("--painful", action="store_true")
    ap.add_argument("--model", default="heavy", choices=["heavy", "lite"])
    ap.add_argument("--debug", default=None)
    args = ap.parse_args()
    r = analyze(args.video, args.height_m, sex=args.sex, level=args.level,
                painful=args.painful, model=args.model,
                debug_path=args.debug)
    print(json.dumps(r, indent=2, ensure_ascii=False,
                     default=lambda o: float(o)))


if __name__ == "__main__":
    main()
