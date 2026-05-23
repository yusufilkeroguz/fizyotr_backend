"""FMS - Active Straight-Leg Raise (ASLR) analizi.

Protokol: kisi sirt ustu yatar, kollar yan tarafta, bir bacagi diz duz
halde yukari kaldirir. Kaldirilan bacagin MALLEOLUSU (ayak bilegi)
altta kalan bacagin ASIS-patella hatti uzerinde hangi noktada oldugu
uzerinden skorlanir. Kamera yan.

Yaklasim: yan goruste, yatay bir zeminde yatan kisi icin imajda
dikey eksen = vucut boyuna yatay; yani yukari-kaldirma yatay-eksen
boyunca hareket gibi projekte olur. Referans cizgiler:
    - mid-thigh (orta-uyluk) seviyesi: down-hip ile down-knee arasi
    - patella (diz kapagi) seviyesi: down-knee

Skorlama:
    3 = kaldirilan malleolus down-leg mid-thigh seviyesinden OTE gecer
    2 = mid-thigh ile patella arasinda kalir
    1 = patella seviyesini asmaz
    0 = agri

CLI:
    python -m fms.aslr <video> <height_m> [--painful]
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
    LEFT_HIP, RIGHT_HIP, LEFT_KNEE, RIGHT_KNEE,
    LEFT_ANKLE, RIGHT_ANKLE,
    iter_pose, pt2d, angle_deg,
    smooth_median, px_per_m_from_body,
)


@dataclass
class ASLRMetrics:
    peak_frame: int
    peak_time_s: float
    raised_leg: str                    # "left" / "right"
    down_leg: str
    # Video duzleminde, "yukari" hareket yonu hesaplanir
    raised_ankle_progress: float       # 0 at down-ankle, 1 at down-hip
    # Referans seviyeler (aynı koordinatta):
    mid_thigh_progress: float          # =0.5 varsayim
    patella_progress: float            # =0.0 varsayim (down knee)
    # Kriterler
    passes_mid_thigh: bool
    passes_patella: bool
    raised_leg_straight: bool          # hip-knee-ankle ~170°
    raised_knee_angle_deg: float
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

    # Videoda her karede HER iki ankle'in y'si takip edilir; FARK en buyuk
    # oldugu kare = bir bacagin yukari kalktigi an (yatan kisi, bacak
    # yukari kaldirildiginda imajda ankle y azalir).
    lank_y = np.full(n, np.nan); rank_y = np.full(n, np.nan)
    for k, (_fi, _t, l2, _l3, _W, _H) in enumerate(frames):
        if l2 is None: continue
        lank_y[k] = l2[LEFT_ANKLE].y * H
        rank_y[k] = l2[RIGHT_ANKLE].y * H
    diff = lank_y - rank_y
    diff_s = smooth_median(diff, 7)
    if not np.isfinite(diff_s).any():
        raise RuntimeError("Ayak bilekleri takip edilemedi.")
    k_left = int(np.nanargmin(diff_s))   # sol ayak imajda daha yukarida
    k_right = int(np.nanargmax(diff_s))  # sag ayak daha yukarida
    # Mutlak pik = hangisinde fark daha net
    if abs(diff_s[k_left]) >= abs(diff_s[k_right]):
        peak_k = k_left; raised = "left"
    else:
        peak_k = k_right; raised = "right"
    down = "right" if raised == "left" else "left"

    b_fi, b_t, b_l2, _b_l3, _bW, _bH = frames[peak_k]
    px_per_m = px_per_m_from_body(b_l2, W, H, height_m)

    RH = LEFT_HIP if raised == "left" else RIGHT_HIP
    RK = LEFT_KNEE if raised == "left" else RIGHT_KNEE
    RA = LEFT_ANKLE if raised == "left" else RIGHT_ANKLE
    DH = LEFT_HIP if down == "left" else RIGHT_HIP
    DK = LEFT_KNEE if down == "left" else RIGHT_KNEE
    DA = LEFT_ANKLE if down == "left" else RIGHT_ANKLE

    def P(i): return pt2d(b_l2, i, W, H)
    rhp, rkp, rap = P(RH), P(RK), P(RA)
    dhp, dkp, dap = P(DH), P(DK), P(DA)

    # Yan goruste kisi yerde yatiyor. Yukari-yon = ankle->hip dogrultusu.
    # Down leg "yatay" uzanir; yukari yon dikey-benzeri oluyor.
    # Progress hesabi: raised ankle'in down-ankle (0) ile down-hip (1)
    # arasinda yaptigi projeksiyon.
    axis = dhp - dap
    axis_n = float(np.linalg.norm(axis))
    if axis_n < 1e-6:
        raise RuntimeError("Down-leg sifir boylu.")
    axis_u = axis / axis_n
    # Raised ankle'in yukari-yondeki projeksiyonu (down-ankle orijin)
    rel = rap - dap
    proj = float(np.dot(rel, axis_u)) / axis_n   # 0..1 (ve asabilir)
    # Ayni metric icin patella (down-knee) ve mid-thigh
    mid_thigh_pt = (dhp + dkp) / 2
    patella_pt = dkp
    mt_proj = float(np.dot(mid_thigh_pt - dap, axis_u)) / axis_n
    pat_proj = float(np.dot(patella_pt - dap, axis_u)) / axis_n

    passes_mid = bool(proj > mt_proj)
    passes_pat = bool(proj > pat_proj)

    # Raised leg dizlik: ~170+° demek duz
    raised_knee_angle = angle_deg(rhp, rkp, rap)
    raised_straight = bool(raised_knee_angle > 160.0)

    if passes_mid and raised_straight:
        score, reason = 3, "malleolus mid-thigh uzerinde, diz duz"
    elif passes_pat:
        score, reason = 2, "malleolus mid-thigh ile patella arasinda"
    else:
        score, reason = 1, "malleolus patella altinda"

    result = ASLRMetrics(
        peak_frame=b_fi, peak_time_s=round(b_t, 3),
        raised_leg=raised, down_leg=down,
        raised_ankle_progress=round(proj, 3),
        mid_thigh_progress=round(mt_proj, 3),
        patella_progress=round(pat_proj, 3),
        passes_mid_thigh=passes_mid,
        passes_patella=passes_pat,
        raised_leg_straight=raised_straight,
        raised_knee_angle_deg=round(raised_knee_angle, 1),
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

    raised = m.raised_leg
    RH_ = LEFT_HIP if raised == "left" else RIGHT_HIP
    RK_ = LEFT_KNEE if raised == "left" else RIGHT_KNEE
    RA_ = LEFT_ANKLE if raised == "left" else RIGHT_ANKLE
    DH_ = RIGHT_HIP if raised == "left" else LEFT_HIP
    DK_ = RIGHT_KNEE if raised == "left" else LEFT_KNEE
    DA_ = RIGHT_ANKLE if raised == "left" else LEFT_ANKLE

    def P(i): return tuple(pt2d(l2, i, W, H).astype(int))
    rhp, rkp, rap = P(RH_), P(RK_), P(RA_)
    dhp, dkp, dap = P(DH_), P(DK_), P(DA_)

    # Kaldirilan bacak
    cv2.line(img, rhp, rkp, (0, 255, 100), 5, cv2.LINE_AA)
    cv2.line(img, rkp, rap, (0, 200, 255), 5, cv2.LINE_AA)
    # Yatan bacak
    cv2.line(img, dhp, dkp, (140, 140, 140), 3, cv2.LINE_AA)
    cv2.line(img, dkp, dap, (140, 140, 140), 3, cv2.LINE_AA)

    # Referans izleri: patella ve mid-thigh
    mt = ((dhp[0]+dkp[0])//2, (dhp[1]+dkp[1])//2)
    # Raised ankle'i referans izlerine bag-cizgisi
    cv2.circle(img, rap, 14, (0, 220, 255), -1)
    cv2.circle(img, dkp, 10, (60, 180, 220), -1)         # patella
    cv2.circle(img, mt, 10, (60, 220, 60), -1)            # mid-thigh
    cv2.line(img, rap, dkp, (60, 180, 220), 2, cv2.LINE_AA)
    cv2.line(img, rap, mt, (60, 220, 60), 2, cv2.LINE_AA)

    def txt(s, y, color=(255, 255, 255)):
        cv2.putText(img, s, (30, y), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                    (0, 0, 0), 5, cv2.LINE_AA)
        cv2.putText(img, s, (30, y), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                    color, 2, cv2.LINE_AA)
    status = lambda ok: (60, 220, 60) if ok else (60, 60, 220)
    txt(f"ASLR  skor = {m.score}  (kaldirilan={m.raised_leg})",
        50, (0, 255, 255))
    txt(f"mid-thigh gecildi: {m.passes_mid_thigh}",
        95, status(m.passes_mid_thigh))
    txt(f"patella gecildi: {m.passes_patella}",
        135, status(m.passes_patella))
    txt(f"diz duz ({m.raised_knee_angle_deg}°): {m.raised_leg_straight}",
        175, status(m.raised_leg_straight))
    cv2.imwrite(str(out_path), img)


def main():
    ap = argparse.ArgumentParser(description="FMS ASLR analizi")
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
