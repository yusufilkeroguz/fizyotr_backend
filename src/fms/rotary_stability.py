"""FMS - Rotary Stability analizi.

Protokol: quadruped (eller omuz, dizler kalca altinda). Bir ayni-taraf
(ipsilateral) kol ile bacagi uzatir, dirsek ve dizi govde altinda temas
ettirir. Skorlama:
    3 = ipsilateral (ayni taraf) tam tamamlanir, dirsek-diz temas
    2 = kontralateral (capraz) tamamlanir
    1 = capraz ile bile tamamlanamaz
    0 = flex/ext clearing pozitif (agri)

Bu analizci yan-kameradan:
    - Uzatma fazini (kol ve bacak yataya en yakin) ve temas fazini
      (dirsek-diz en yakin) tespit eder
    - Kaldirilan kol ve bacagin ayni taraf mi (ipsilateral) yoksa capraz mi
      oldugunu belirler
    - Dirsek-diz temas mesafesini olcer (metre)

CLI:
    python -m fms.rotary_stability <video> <height_m> [--painful]
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
    LEFT_ELBOW, RIGHT_ELBOW, LEFT_KNEE, RIGHT_KNEE,
    LEFT_ANKLE, RIGHT_ANKLE, LEFT_WRIST, RIGHT_WRIST,
    iter_pose, pt2d, ptw, smooth_median,
)


@dataclass
class RotaryStabilityMetrics:
    contact_frame: int
    contact_time_s: float
    lifted_arm: str            # "left" / "right"
    lifted_leg: str            # "left" / "right"
    pattern: str               # "ipsilateral" / "contralateral"
    elbow_knee_dist_m: float   # temas fazinda 3D
    touches: bool              # < ~7cm
    fps: float
    score: int
    score_reason: str


def _vec(a, b): return b - a


def analyze(video_path: str, height_m: float,
            painful: bool = False, model: str = "heavy",
            conf: float = 0.3,
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

    # Her karede iki (ipsi, kontra) kombinasyonu icin dirsek-diz 3D mesafe
    d_LL = np.full(n, np.nan)   # sol dirsek - sol diz (ipsilateral sol)
    d_RR = np.full(n, np.nan)
    d_LR = np.full(n, np.nan)   # sol dirsek - sag diz (contralateral)
    d_RL = np.full(n, np.nan)
    for k, (_fi, _t, _l2, l3, _W, _H) in enumerate(frames):
        if l3 is None: continue
        le = ptw(l3, LEFT_ELBOW); re = ptw(l3, RIGHT_ELBOW)
        lk = ptw(l3, LEFT_KNEE); rk = ptw(l3, RIGHT_KNEE)
        d_LL[k] = float(np.linalg.norm(le - lk))
        d_RR[k] = float(np.linalg.norm(re - rk))
        d_LR[k] = float(np.linalg.norm(le - rk))
        d_RL[k] = float(np.linalg.norm(re - lk))

    finite_any = any(np.isfinite(d).any() for d in [d_LL, d_RR, d_LR, d_RL])
    if not finite_any:
        raise RuntimeError("3D pose takip edilemedi.")

    # Her kombinasyon icin minimum mesafe (en yakin temas)
    mins = {"ipsilateral_L": np.nanmin(d_LL),
            "ipsilateral_R": np.nanmin(d_RR),
            "contralateral_LR": np.nanmin(d_LR),
            "contralateral_RL": np.nanmin(d_RL)}
    best_key = min(mins, key=lambda k: mins[k])
    min_d = float(mins[best_key])

    if best_key.startswith("ipsi"):
        pattern = "ipsilateral"
        side = best_key.split("_")[1]  # L veya R
        arm = leg = "left" if side == "L" else "right"
        series = d_LL if side == "L" else d_RR
    else:
        pattern = "contralateral"
        arm_side = best_key.split("_")[1][0]   # L veya R
        leg_side = best_key.split("_")[1][1]
        arm = "left" if arm_side == "L" else "right"
        leg = "left" if leg_side == "L" else "right"
        series = d_LR if (arm_side == "L") else d_RL

    contact_k = int(np.nanargmin(series))
    b_fi, b_t, _b_l2, _b_l3, _bW, _bH = frames[contact_k]
    touches = bool(min_d < 0.07)   # 7 cm

    if pattern == "ipsilateral" and touches:
        score = 3; reason = f"ipsilateral ({arm}) temas ({min_d*100:.1f} cm)"
    elif pattern == "contralateral" and touches:
        score = 2; reason = f"kontralateral ({arm} kol / {leg} bacak) temas"
    else:
        score = 1; reason = f"temas yok (en yakin {min_d*100:.1f} cm)"

    result = RotaryStabilityMetrics(
        contact_frame=b_fi,
        contact_time_s=round(b_t, 3),
        lifted_arm=arm, lifted_leg=leg,
        pattern=pattern,
        elbow_knee_dist_m=round(min_d, 3),
        touches=touches,
        fps=round(fps, 2),
        score=score, score_reason=reason,
    )

    if debug_path:
        _draw_debug(video_path, contact_k, result, frames, Path(debug_path))
    return asdict(result)


def _draw_debug(video_path, contact_k, m, frames, out_path):
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frames[contact_k][0])
    ok, img = cap.read(); cap.release()
    if not ok: return
    H, W = img.shape[:2]
    _fi, _t, l2, _l3, _W, _H = frames[contact_k]
    if l2 is None:
        cv2.imwrite(str(out_path), img); return

    E = LEFT_ELBOW if m.lifted_arm == "left" else RIGHT_ELBOW
    K = LEFT_KNEE if m.lifted_leg == "left" else RIGHT_KNEE
    W_ = LEFT_WRIST if m.lifted_arm == "left" else RIGHT_WRIST
    A = LEFT_ANKLE if m.lifted_leg == "left" else RIGHT_ANKLE
    S = LEFT_SHOULDER if m.lifted_arm == "left" else RIGHT_SHOULDER
    HIP = LEFT_HIP if m.lifted_leg == "left" else RIGHT_HIP

    def P(i): return tuple(pt2d(l2, i, W, H).astype(int))
    e, k, w_, a = P(E), P(K), P(W_), P(A)
    s, hip = P(S), P(HIP)

    cv2.line(img, s, e, (0, 255, 100), 5, cv2.LINE_AA)
    cv2.line(img, e, w_, (0, 200, 255), 5, cv2.LINE_AA)
    cv2.line(img, hip, k, (180, 80, 255), 5, cv2.LINE_AA)
    cv2.line(img, k, a, (255, 200, 0), 5, cv2.LINE_AA)
    cv2.line(img, e, k, (0, 230, 230), 3, cv2.LINE_AA)  # temas cizgisi
    cv2.circle(img, e, 14, (0, 230, 230), -1)
    cv2.circle(img, k, 14, (0, 230, 230), -1)

    def txt(s, y, color=(255, 255, 255)):
        cv2.putText(img, s, (30, y), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                    (0, 0, 0), 5, cv2.LINE_AA)
        cv2.putText(img, s, (30, y), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                    color, 2, cv2.LINE_AA)
    status = lambda ok: (60, 220, 60) if ok else (60, 60, 220)
    txt(f"ROTARY STABILITY  skor = {m.score}", 50, (0, 255, 255))
    txt(f"pattern = {m.pattern}  (kol={m.lifted_arm}, bacak={m.lifted_leg})",
        95, (220, 220, 220))
    txt(f"dirsek-diz temas: {m.touches} ({m.elbow_knee_dist_m*100:.1f} cm)",
        135, status(m.touches))
    cv2.imwrite(str(out_path), img)


def main():
    ap = argparse.ArgumentParser(description="FMS Rotary Stability analizi")
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
