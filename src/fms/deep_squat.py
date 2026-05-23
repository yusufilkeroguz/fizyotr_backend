"""FMS - Deep Squat (derin cokelme) analizi.

Protokol: denek ayaklari omuz genisliginde, dowel (cubuk) iki kolu dik
yukarida tutarken maks derinlige cokelir.

Skorlama (FMS):
    3 = (hepsi): dowel bas-uzeri alignli (kollar dik), gövde neredeyse
        tibia ile paralel, femur horizontalin altinda (kalca diz cizgisinin
        altinda), dizler ayak ucunun onunde, topuklar yerde.
    2 = ayni kriterler ama topuklarin altina 2x6 board konularak tamamlandi
        (bu otomatik tespit edilemez; kullanici flag ile belirtebilir).
    1 = skor 3 kriterlerini board ile bile saglayamaz.
    0 = agri (manuel giris).

Bu analyzer YAN KAMERA varsayar (saga veya sola bakar; pose visibility'den
otomatik secer). Iki kameradan gelen cekimler de desteklenebilir ileride.

CLI:
    python -m fms.deep_squat <video> <height_m> [--board] [--painful]
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

try:
    import cv2  # opsiyonel: yalnizca video/render icin (CSV-only'da gerekmez)
except ImportError:
    cv2 = None
import numpy as np

from .common import (
    LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_ELBOW, RIGHT_ELBOW,
    LEFT_WRIST, RIGHT_WRIST,
    LEFT_HIP, RIGHT_HIP, LEFT_KNEE, RIGHT_KNEE,
    LEFT_ANKLE, RIGHT_ANKLE, LEFT_HEEL, RIGHT_HEEL,
    LEFT_FOOT, RIGHT_FOOT, NOSE,
    iter_pose, pt2d, angle_deg, line_angle_deg,
    pick_visible_side, smooth_median, px_per_m_from_body,
)


@dataclass
class DeepSquatMetrics:
    # Maks derinlikteki kare
    bottom_frame: int
    bottom_time_s: float
    # Kriterler (True/False)
    hip_below_knee: bool            # femur horizontalin altinda
    torso_parallel_tibia: bool      # gövde-tibia aci farki < 15°
    knees_not_past_toes: bool       # diz ayak ucunu gecmemis
    heels_on_ground: bool           # topuk pozisyonu sabit
    dowel_overhead: bool            # bilek omuzun onunde/ustunde, dirsek ~dik
    # Olcumler
    knee_angle_deg: float           # bottom'da diz ici acisi
    hip_knee_dy_px: float           # kalca y - diz y (pozitif = kalca daha asagida)
    torso_angle_deg: float          # govde (omuz-kalca) yatayla aci
    tibia_angle_deg: float          # tibia (diz-ayak bilegi) yatayla aci
    torso_vs_tibia_deg: float
    knee_toe_dx_px: float           # diz x - ayak_index x (+ = diz one dogru kacmis)
    heel_drift_px: float            # topuk pozisyonunun toplam degisimi
    dowel_tilt_deg: float           # omuz->bilek cizgisi dikey ile aci
    # Yardimci
    side_used: str                  # "left" / "right"
    fps: float
    px_per_m: float
    # Skor
    score: int
    score_reason: str


def _criterion(hip_below_knee: bool, torso_tibia: bool, knees_toes: bool,
               heels_ok: bool, dowel_ok: bool) -> tuple[bool, list[str]]:
    """Tum kriterler gecti mi + eksik olan(lar)in listesi."""
    fails = []
    if not hip_below_knee: fails.append("kalca diz altinda degil")
    if not torso_tibia: fails.append("govde-tibia paralel degil")
    if not knees_toes: fails.append("diz ayak ucunu geciyor")
    if not heels_ok: fails.append("topuk kalkmis")
    if not dowel_ok: fails.append("dowel bas uzeri alignsiz")
    return len(fails) == 0, fails


def analyze(video_path: str, height_m: float,
            board: bool = False, painful: bool = False,
            model: str = "heavy", conf: float = 0.3,
            debug_path: Optional[str] = None) -> dict:
    """Returns dict with metrics + FMS score."""
    if painful:
        return {"score": 0, "score_reason": "agri (manuel giris)"}

    frames = []  # (fi, t, lm2d, lm3d, W, H)
    fps = 30.0
    W = H = 0
    for fi, t, l2, l3, W, H, fps in iter_pose(video_path, model=model, conf=conf):
        frames.append((fi, t, l2, l3, W, H))
    n = len(frames)
    if n == 0:
        raise RuntimeError("Video okunamadi.")

    # Ayak bilegi y (piksel) zaman serisi (max iki ayakbileginden)
    ank_y = np.full(n, np.nan)
    hip_y = np.full(n, np.nan)
    knee_y = np.full(n, np.nan)
    for k, (_fi, _t, l2, _l3, _W, _H) in enumerate(frames):
        if l2 is None:
            continue
        la = pt2d(l2, LEFT_ANKLE, W, H)[1]
        ra = pt2d(l2, RIGHT_ANKLE, W, H)[1]
        lh = pt2d(l2, LEFT_HIP, W, H)[1]
        rh = pt2d(l2, RIGHT_HIP, W, H)[1]
        lk = pt2d(l2, LEFT_KNEE, W, H)[1]
        rk = pt2d(l2, RIGHT_KNEE, W, H)[1]
        ank_y[k] = max(la, ra)
        hip_y[k] = (lh + rh) / 2
        knee_y[k] = (lk + rk) / 2
    hip_y_s = smooth_median(hip_y, 5)

    # Bottom kare: kalcanin piksel-y'si maksimum (asagida)
    if np.all(np.isnan(hip_y_s)):
        raise RuntimeError("Kalca bulunamadi.")
    bottom_k = int(np.nanargmax(hip_y_s))
    b_fi, b_t, b_l2, b_l3, bW, bH = frames[bottom_k]

    # Hangi yan gorunur (yan-kamera)
    side = pick_visible_side(b_l2, LEFT_HIP, RIGHT_HIP)
    HIP = LEFT_HIP if side == "left" else RIGHT_HIP
    KNEE = LEFT_KNEE if side == "left" else RIGHT_KNEE
    ANK = LEFT_ANKLE if side == "left" else RIGHT_ANKLE
    HEEL = LEFT_HEEL if side == "left" else RIGHT_HEEL
    FOOT = LEFT_FOOT if side == "left" else RIGHT_FOOT
    SH = LEFT_SHOULDER if side == "left" else RIGHT_SHOULDER
    WR = LEFT_WRIST if side == "left" else RIGHT_WRIST
    ELB = LEFT_ELBOW if side == "left" else RIGHT_ELBOW

    # Bottom'daki noktalar (piksel)
    hip_p = pt2d(b_l2, HIP, W, H)
    knee_p = pt2d(b_l2, KNEE, W, H)
    ank_p = pt2d(b_l2, ANK, W, H)
    heel_p = pt2d(b_l2, HEEL, W, H)
    foot_p = pt2d(b_l2, FOOT, W, H)
    sh_p = pt2d(b_l2, SH, W, H)
    wr_p = pt2d(b_l2, WR, W, H)

    px_per_m = px_per_m_from_body(b_l2, W, H, height_m)

    # --- Kriterler ---
    # 1) Kalca diz altinda (imajda hip.y > knee.y, y=0 ust)
    hip_knee_dy = float(hip_p[1] - knee_p[1])  # + = hip daha asagi
    hip_below_knee = bool(hip_knee_dy > 0.02 * px_per_m)  # en az 2 cm altinda

    # 2) Govde (omuz->kalca) acisi yatayla; tibia (diz->ayak bilegi) acisi yatayla
    torso_ang = line_angle_deg(sh_p, hip_p)      # 0..180, dikey=90
    tibia_ang = line_angle_deg(knee_p, ank_p)
    diff = abs(torso_ang - tibia_ang)
    if diff > 90:
        diff = 180 - diff
    torso_tibia_ok = bool(diff < 15.0)

    # 3) Diz ayak ucunu gecmedi (imaj yatay ekseninde):
    #    yan goruste foot_index one bakan dogrultuda. |knee.x - foot.x| kucuk olmali
    knee_toe_dx = float(knee_p[0] - foot_p[0])
    knees_toes_ok = bool(abs(knee_toe_dx) < 0.10 * px_per_m)  # <10 cm kayma

    # 4) Topuk yerde: sadece squat penceresinde (bottom ± 0.5s) heel_y degisimi.
    half_win = int(0.5 * fps)
    lo_i = max(0, bottom_k - half_win)
    hi_i = min(n, bottom_k + half_win + 1)
    heel_ys = []
    for (_fi, _t, l2, _l3, _W, _H) in frames[lo_i:hi_i]:
        if l2 is None:
            continue
        p = l2[HEEL]
        if p.visibility < 0.5:
            continue
        heel_ys.append(p.y * H)
    heel_drift = float(np.max(heel_ys) - np.min(heel_ys)) if len(heel_ys) >= 3 else float("nan")
    heels_ok = bool(heel_drift < 0.04 * px_per_m) if not np.isnan(heel_drift) else True

    # 5) Dowel bas uzeri: bilek omuzun tam ustunde/onunde,
    #    omuz->bilek cizgisi dikey ile <20°
    dowel_tilt = abs(90.0 - line_angle_deg(sh_p, wr_p))
    # line_angle: 0=yatay, 90=dikey; dikeyden sapma = |line - 90|
    # duzeltme: line_angle_deg 0..180, dikey=90
    dowel_tilt = min(dowel_tilt, 180 - dowel_tilt)
    dowel_ok = bool(dowel_tilt < 20.0)

    # Knee ici aci (ek bilgi)
    knee_angle = angle_deg(hip_p, knee_p, ank_p)

    # Gecti / bosl
    passed_all, fails = _criterion(hip_below_knee, torso_tibia_ok,
                                   knees_toes_ok, heels_ok, dowel_ok)

    if passed_all:
        score = 3
        reason = "tum kriterler saglandi"
    elif board:
        # Board altinda topuk kalkma affedilir; kalan kriterler lazim
        passed_no_heel, fails2 = _criterion(hip_below_knee, torso_tibia_ok,
                                            knees_toes_ok, True, dowel_ok)
        if passed_no_heel:
            score = 2
            reason = "board ile saglandi (topuk haric)"
        else:
            score = 1
            reason = "board ile bile eksik: " + ", ".join(fails2)
    else:
        # Board kullanilmadi; yalniz topuk kaldigi halde digerleri okse skor = 2?
        # FMS'de skor 2 SADECE board ile verilir. Aksi halde skor 1.
        score = 1
        reason = "eksik: " + ", ".join(fails)

    result = DeepSquatMetrics(
        bottom_frame=b_fi,
        bottom_time_s=round(b_t, 3),
        hip_below_knee=hip_below_knee,
        torso_parallel_tibia=torso_tibia_ok,
        knees_not_past_toes=knees_toes_ok,
        heels_on_ground=heels_ok,
        dowel_overhead=dowel_ok,
        knee_angle_deg=round(knee_angle, 1),
        hip_knee_dy_px=round(hip_knee_dy, 1),
        torso_angle_deg=round(torso_ang, 1),
        tibia_angle_deg=round(tibia_ang, 1),
        torso_vs_tibia_deg=round(diff, 1),
        knee_toe_dx_px=round(knee_toe_dx, 1),
        heel_drift_px=round(heel_drift, 1),
        dowel_tilt_deg=round(dowel_tilt, 1),
        side_used=side,
        fps=round(fps, 2),
        px_per_m=round(px_per_m, 1),
        score=score,
        score_reason=reason,
    )

    if debug_path:
        _draw_debug(video_path, b_fi, result, frames, Path(debug_path))

    return asdict(result)


def _draw_debug(video_path: str, bottom_frame: int, m: DeepSquatMetrics,
                frames: list, out_path: Path) -> None:
    """Maks derinlik karesine analiz cizimi kaydeder."""
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, bottom_frame)
    ok, img = cap.read()
    cap.release()
    if not ok:
        return
    H, W = img.shape[:2]
    # Bottom kare lm2d
    _fi, _t, l2, _l3, _W, _H = frames[bottom_frame]
    if l2 is None:
        cv2.imwrite(str(out_path), img); return
    side = m.side_used
    HIP = LEFT_HIP if side == "left" else RIGHT_HIP
    KNEE = LEFT_KNEE if side == "left" else RIGHT_KNEE
    ANK = LEFT_ANKLE if side == "left" else RIGHT_ANKLE
    HEEL = LEFT_HEEL if side == "left" else RIGHT_HEEL
    FOOT = LEFT_FOOT if side == "left" else RIGHT_FOOT
    SH = LEFT_SHOULDER if side == "left" else RIGHT_SHOULDER
    WR = LEFT_WRIST if side == "left" else RIGHT_WRIST
    def p(i): return tuple(pt2d(l2, i, W, H).astype(int))

    # Iskelet kollar, govde, bacak
    for a, b, c in [(SH, WR, (255, 220, 0)),    # dowel kolu
                    (SH, HIP, (0, 200, 255)),   # govde
                    (HIP, KNEE, (0, 255, 100)), # uyluk
                    (KNEE, ANK, (0, 100, 255)), # tibia
                    (HEEL, FOOT, (200, 0, 200))]:  # ayak
        cv2.line(img, p(a), p(b), c, 6, cv2.LINE_AA)

    # Kalca-diz yatay referans cizgi
    hip_pt = p(HIP); knee_pt = p(KNEE)
    cv2.line(img, (0, knee_pt[1]), (W, knee_pt[1]), (120, 120, 120), 2, cv2.LINE_AA)
    cv2.circle(img, hip_pt, 10, (0, 0, 255), -1)
    cv2.circle(img, knee_pt, 10, (0, 255, 0), -1)

    def txt(s, y, color=(255, 255, 255)):
        cv2.putText(img, s, (30, y), cv2.FONT_HERSHEY_SIMPLEX, 1.2,
                    (0, 0, 0), 6, cv2.LINE_AA)
        cv2.putText(img, s, (30, y), cv2.FONT_HERSHEY_SIMPLEX, 1.2,
                    color, 2, cv2.LINE_AA)

    status = lambda ok: (60, 220, 60) if ok else (60, 60, 220)
    txt(f"DEEP SQUAT  skor = {m.score}  ({m.score_reason})", 50, (0, 255, 255))
    txt(f"kalca<diz: {m.hip_below_knee}  (dy={m.hip_knee_dy_px:.0f}px)",
        110, status(m.hip_below_knee))
    txt(f"govde|tibia: {m.torso_parallel_tibia}  (fark={m.torso_vs_tibia_deg:.1f}°)",
        160, status(m.torso_parallel_tibia))
    txt(f"diz<=parmak: {m.knees_not_past_toes}  (dx={m.knee_toe_dx_px:.0f}px)",
        210, status(m.knees_not_past_toes))
    txt(f"topuk yerde: {m.heels_on_ground}  (drift={m.heel_drift_px:.0f}px)",
        260, status(m.heels_on_ground))
    txt(f"dowel dik: {m.dowel_overhead}  (egim={m.dowel_tilt_deg:.1f}°)",
        310, status(m.dowel_overhead))
    cv2.imwrite(str(out_path), img)


def main():
    ap = argparse.ArgumentParser(description="FMS Deep Squat analizi")
    ap.add_argument("video")
    ap.add_argument("height_m", type=float)
    ap.add_argument("--board", action="store_true",
                    help="topuk altinda 2x6 board kullanildi (skor 2 icin)")
    ap.add_argument("--painful", action="store_true", help="agri var (skor 0)")
    ap.add_argument("--model", default="heavy", choices=["heavy", "lite"])
    ap.add_argument("--debug", default=None, help="maks derinlik karesi cikti")
    args = ap.parse_args()
    r = analyze(args.video, args.height_m, board=args.board,
                painful=args.painful, model=args.model, debug_path=args.debug)
    print(json.dumps(r, indent=2, ensure_ascii=False, default=lambda o: float(o)))


if __name__ == "__main__":
    main()
