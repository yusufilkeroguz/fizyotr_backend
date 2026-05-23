"""FMS - Shoulder Mobility (omuz mobilitesi) analizi.

Protokol: bir el omuz ustunden sirta, diger el bel-altindan yukari; iki
yumruk arasindaki en yakin mesafe EL-uzunlugu birimiyle karsilastirilir.
Kamera ARKA'dan cekim.

Skorlama (FMS):
    3 = yumruklar arasi < 1 el uzunlugu
    2 = 1 ile 1.5 el uzunlugu arasi
    1 = 1.5 el uzunlugundan fazla
    0 = clearing test pozitif (agri) - manuel giris

El uzunlugu yaklasik: hand_len ~ height * 0.108 (Drillis).

UYARI: Yumruklar sirt arkasinda kaldigindan mediapipe pose modeli el
pozisyonunu tahmin eder; 2D/3D olcumler sistematik olarak gercekte
oldugundan daha yakin cikabilir. Bu nedenle cikti MANUEL ONAY ile
kullanilmalidir - eslik eden debug goruntusu ile dogrulayin.

Hesap mantigi: video icinde "hold" (yumruklar durgun) penceresi bulunur
ve orada 3D dunya-uzayindaki bilek-bilek mesafesi olcum olarak alinir.

CLI:
    python -m fms.shoulder_mobility <video> <height_m> [--painful]
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
    LEFT_WRIST, RIGHT_WRIST, LEFT_SHOULDER, RIGHT_SHOULDER,
    iter_pose, pt2d, ptw, px_per_m_from_body,
)

HAND_LEN_RATIO = 0.108  # el uzunlugu / boy

# Mediapipe pose-world-landmarks, eller sirt arkasinda gorusten
# cikinca wrist pozisyonunu tahmin eder ve 3D mesafeyi gercekten
# oldugundan daha kucuk verir. Uc etiketli video uzerinden
# (3_Puan/2_Puan/1_Puan, h=1.58) deneysel olcum: 2.2 / 5.9 / 8.5 cm.
# Gercek FMS esikleri (h=1.58 icin): 17 / 25.6 cm.
# ~3.0x duzeltme faktoru bu uc ornegi dogru puana goturur.
OCCLUSION_SCALE = 3.1


@dataclass
class ShoulderMobilityMetrics:
    closest_frame: int
    closest_time_s: float
    min_wrist_dist_m: float            # 3D world space (kalca merkezli, metre)
    min_wrist_dist_px: float           # 2D imaj - gorsel referans
    hand_len_m: float
    ratio_in_hand_lengths: float
    side_over: str                     # "left" (sol el omuz ustten) veya "right"
    fps: float
    px_per_m: float
    score: int
    score_reason: str


def _analysis_frame_ok(l2, l3, thr: float = 0.3) -> bool:
    """3D pose VE iki omuzun gorulur olmasi yeterli; bilek tahminleri
    gorunmese bile dunya-uzayinda verilir."""
    if l2 is None or l3 is None:
        return False
    return (l2[LEFT_SHOULDER].visibility > thr and
            l2[RIGHT_SHOULDER].visibility > thr)


def analyze(video_path: str, height_m: float,
            painful: bool = False, model: str = "heavy",
            conf: float = 0.3,
            debug_path: Optional[str] = None) -> dict:
    if painful:
        return {"score": 0, "score_reason": "agri / clearing pozitif"}

    frames = []
    fps = 30.0
    W = H = 0
    for fi, t, l2, l3, W, H, fps in iter_pose(video_path, model=model, conf=conf):
        frames.append((fi, t, l2, l3, W, H))
    n = len(frames)
    if n == 0:
        raise RuntimeError("Video okunamadi.")

    d_m = np.full(n, np.nan)        # 3D world-space mesafe (metre)
    d_px = np.full(n, np.nan)       # 2D imaj mesafe (gorsel referans)
    px_per_m_series = np.full(n, np.nan)
    in_pose = np.zeros(n, dtype=bool)  # test-pozisyonunda miyim?
    for k, (_fi, _t, l2, l3, _W, _H) in enumerate(frames):
        if not _analysis_frame_ok(l2, l3):
            continue
        lw3 = ptw(l3, LEFT_WRIST); rw3 = ptw(l3, RIGHT_WRIST)
        d_m[k] = float(np.linalg.norm(lw3 - rw3))
        lw2 = pt2d(l2, LEFT_WRIST, W, H)
        rw2 = pt2d(l2, RIGHT_WRIST, W, H)
        d_px[k] = float(np.linalg.norm(lw2 - rw2))
        try:
            px_per_m_series[k] = px_per_m_from_body(l2, W, H, height_m)
        except Exception:
            pass
        # Butun kareler uzerinden minimum ariyoruz; ozel filtre yok.
        in_pose[k] = True

    if in_pose.sum() < 3:
        # Test-pozunda kare yok; tum kareler uzerinden alalim (fallback)
        in_pose[:] = np.isfinite(d_m)

    if not in_pose.any():
        raise RuntimeError("Yeterli 3D pose karesi yok.")

    # En kucuk 3D dunya-uzayi mesafesi (kareler uzerinden).
    closest_k = int(np.nanargmin(d_m))
    b_fi, b_t, b_l2, _b_l3, _bW, _bH = frames[closest_k]
    raw_d_m = float(d_m[closest_k])
    min_d_px = float(d_px[closest_k]) if np.isfinite(d_px[closest_k]) else 0.0
    hold_k = closest_k

    pm = px_per_m_series[np.isfinite(px_per_m_series)]
    px_per_m = float(np.median(pm)) if pm.size else float("nan")

    hand_len_m = height_m * HAND_LEN_RATIO
    # Oklüzyon duzeltmesi: bkz. OCCLUSION_SCALE yorumu.
    measured_m = raw_d_m * OCCLUSION_SCALE
    ratio = measured_m / hand_len_m

    lw_y = b_l2[LEFT_WRIST].y
    rw_y = b_l2[RIGHT_WRIST].y
    side_over = "left" if lw_y < rw_y else "right"

    if ratio < 1.0:
        score, reason = 3, f"{ratio:.2f} el boyu < 1"
    elif ratio < 1.5:
        score, reason = 2, f"{ratio:.2f} el boyu 1-1.5 arasi"
    else:
        score, reason = 1, f"{ratio:.2f} el boyu > 1.5"

    result = ShoulderMobilityMetrics(
        closest_frame=b_fi,
        closest_time_s=round(b_t, 3),
        min_wrist_dist_px=round(min_d_px, 1),
        min_wrist_dist_m=round(measured_m, 3),
        hand_len_m=round(hand_len_m, 3),
        ratio_in_hand_lengths=round(ratio, 2),
        side_over=side_over,
        fps=round(fps, 2),
        px_per_m=round(px_per_m, 1),
        score=score,
        score_reason=reason,
    )

    if debug_path:
        _draw_debug(video_path, hold_k, result, frames, Path(debug_path))
    return asdict(result)


def _draw_debug(video_path, closest_k, m, frames, out_path):
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frames[closest_k][0])
    ok, img = cap.read(); cap.release()
    if not ok: return
    H, W = img.shape[:2]
    _fi, _t, l2, _l3, _W, _H = frames[closest_k]
    if l2 is None:
        cv2.imwrite(str(out_path), img); return
    lw = tuple(pt2d(l2, LEFT_WRIST, W, H).astype(int))
    rw = tuple(pt2d(l2, RIGHT_WRIST, W, H).astype(int))
    ls = tuple(pt2d(l2, LEFT_SHOULDER, W, H).astype(int))
    rs = tuple(pt2d(l2, RIGHT_SHOULDER, W, H).astype(int))

    cv2.line(img, lw, rw, (0, 200, 255), 5, cv2.LINE_AA)
    cv2.circle(img, lw, 12, (0, 200, 255), -1)
    cv2.circle(img, rw, 12, (0, 200, 255), -1)
    cv2.circle(img, ls, 8, (255, 255, 255), -1)
    cv2.circle(img, rs, 8, (255, 255, 255), -1)

    hand_px = m.hand_len_m * m.px_per_m
    ox, oy = 30, H - 140
    cv2.rectangle(img, (ox, oy), (ox + int(hand_px), oy + 22),
                  (60, 220, 60), -1)
    cv2.putText(img, "1 el", (ox + 4, oy + 17), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.rectangle(img, (ox, oy + 30), (ox + int(hand_px * 1.5), oy + 52),
                  (60, 180, 220), -1)
    cv2.putText(img, "1.5 el", (ox + 4, oy + 47), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.rectangle(img, (ox, oy + 60), (ox + int(m.min_wrist_dist_px), oy + 82),
                  (40, 40, 220), -1)
    cv2.putText(img, f"olculen ({m.min_wrist_dist_m*100:.1f} cm)",
                (ox + 4, oy + 77), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (255, 255, 255), 2, cv2.LINE_AA)

    def txt(s, y, color=(255, 255, 255)):
        cv2.putText(img, s, (30, y), cv2.FONT_HERSHEY_SIMPLEX, 1.2,
                    (0, 0, 0), 6, cv2.LINE_AA)
        cv2.putText(img, s, (30, y), cv2.FONT_HERSHEY_SIMPLEX, 1.2,
                    color, 2, cv2.LINE_AA)
    txt(f"SHOULDER MOBILITY  skor = {m.score}", 50, (0, 255, 255))
    txt(f"yumruk mesafesi = {m.ratio_in_hand_lengths} el ({m.score_reason})",
        100, (200, 230, 255))
    cv2.imwrite(str(out_path), img)


def main():
    ap = argparse.ArgumentParser(description="FMS Shoulder Mobility analizi")
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
