"""5-0-5 Pro Agility Shuttle — yon degistirmeli surat (cevikilik) testi.

Protokol: orta cizgide basla, 5 m bir tarafa kos ve DUBAYA dokun, 10 m
karsiya kos ve DUBAYA dokun, 5 m geri orta cizgide bitir. Toplam 20 m.

Sporcu dubalari (cizgileri) GENELDE GECMEZ — egilir, dokunur, doner.
Bu yuzden hip_x trayektorisi right_x/left_x cizgilerine ulasmaz, biraz
geride zirve/dip yapar. Dönüsleri "cizgi gecisi" ile aramak yerine hip_x'in
lokal extremumu (argmax / argmin) olarak buluyoruz.

Kullanim: 3 piksel cizgisi:
    left_x, center_x, right_x      (= duba/cizgi konumlari)
center_x mutlaka left_x ile right_x arasinda olmali. Kalibre:
    px_per_m = |right_x - left_x| / 10      (dubalar arasi gercek mesafe 10 m)

CLI:
    python shuttle_5_10_5.py run.mp4 --left-x 200 --center-x 950 --right-x 1700
"""
from __future__ import annotations

import argparse
import json
import sys

import numpy as np

from ..pose_common import (
    iter_pose, pt2d, smooth_median, crossing_frame, motion_onset,
    LEFT_HIP, RIGHT_HIP,
)


def analyze(video_path: str, *, left_x: int, center_x: int, right_x: int,
            model: str = "heavy", conf: float = 0.2) -> dict:
    for nm, vv in (("left_x", left_x), ("center_x", center_x),
                   ("right_x", right_x)):
        if vv is None:
            raise RuntimeError(f"{nm} gerekli.")
    if not (left_x < center_x < right_x or right_x < center_x < left_x):
        raise RuntimeError("center_x, left_x ile right_x arasinda olmali.")

    hip_x: list[float] = []
    fps = 30.0
    for fi, t, l2, l3, W, H, fps in iter_pose(video_path, model=model, conf=conf):
        if l2 is None:
            hip_x.append(np.nan)
            continue
        lh, rh = pt2d(l2, LEFT_HIP, W, H), pt2d(l2, RIGHT_HIP, W, H)
        hip_x.append(float((lh[0] + rh[0]) / 2))
    hipx = smooth_median(np.array(hip_x), 5)
    valid = np.where(~np.isnan(hipx))[0]
    if valid.size < 10:
        raise RuntimeError("Yeterli pose verisi yok.")

    track_px = float(abs(right_x - left_x))
    if track_px < 50:
        raise RuntimeError("left_x ile right_x cok yakin.")
    px_per_m = track_px / 10.0

    # Sag tarafin piksel ekseninde isareti: right_x > center_x ise +1, aksi -1.
    right_is_pos = 1 if right_x > center_x else -1

    # 1) Start: SPORCU CENTER'DA DURUR ve kosmaya baslar -> crossing_frame
    # buradaki rest pozisyonunu (center civari) "gecis" sayamaz; sf ya None doner
    # ya da geç bir geçişi (genelde finish'i) bulur. Bu yuzden motion_onset
    # kullaniyoruz: rest pozisyonundan ilk anlamli sapma. Esik = 0.3 m
    # (px_per_m * 0.3) ama en az 30 px. Sub-frame interpolasyon dahil.
    onset_thr = max(30.0, px_per_m * 0.3)
    onset = motion_onset(hipx, int(valid[0]), threshold_px=onset_thr)
    sf = onset if onset is not None else float(valid[0])

    # Ilk hareketin yonu: onset'ten sonraki birkac karede net delta.
    look = max(3, int(0.3 * fps))
    i0 = int(round(sf))
    i1 = int(min(valid[-1], i0 + look))
    delta = float(hipx[i1] - hipx[i0]) if i1 > i0 else 0.0
    if not np.isfinite(delta) or abs(delta) < 5:
        # Fallback: trayektorinin extremumlarina bak.
        mx, mn = float(np.nanmax(hipx)), float(np.nanmin(hipx))
        delta = (mx - hipx[i0]) - (hipx[i0] - mn)
    first_dir = 1 if delta > 0 else -1
    first_side = "right" if first_dir == right_is_pos else "left"

    # first_far_x ve other_far_x kalibrasyon ve raporlama icin tutulur
    # (turn detection peak-based; cizgi ulasimi gerektirmez).
    first_far_x = right_x if first_side == "right" else left_x
    other_far_x = left_x if first_side == "right" else right_x
    end_idx = int(valid[-1])
    # 2) turn1: sf'den sonra hip_x'in first_dir yonundeki zirvesi.
    # Sporcu dubaya dokunup donerken hip lokal extremum yapar.
    seg1 = hipx[int(round(sf)):end_idx + 1]
    if not np.isfinite(seg1).any():
        raise RuntimeError("Birinci dönüs araliginda pose verisi yok.")
    t1_off = int(np.nanargmax(seg1) if first_dir > 0 else np.nanargmin(seg1))
    t1 = float(int(round(sf)) + t1_off)
    if t1 <= sf:
        raise RuntimeError("Birinci dönüs zirvesi bulunamadi.")

    # 3) center_pass: turn1'den sonra center_x'i ters yonde gec.
    cm = crossing_frame(hipx, float(center_x), -first_dir, int(t1) + 1)
    if cm is None:
        raise RuntimeError("Orta cizgi geri gecisi bulunamadi.")

    # 4) turn2: cm'den sonra hip_x'in ters yondeki dibi (other duba).
    seg2 = hipx[int(round(cm)):end_idx + 1]
    if not np.isfinite(seg2).any():
        raise RuntimeError("Ikinci dönüs araliginda pose verisi yok.")
    t2_off = int(np.nanargmin(seg2) if first_dir > 0 else np.nanargmax(seg2))
    t2 = float(int(round(cm)) + t2_off)
    if t2 <= cm:
        raise RuntimeError("Ikinci dönüs dibi bulunamadi.")

    # 5) Finish: t2'den sonra center_x'i first_dir yonunde gec.
    ff = crossing_frame(hipx, float(center_x), first_dir, int(t2) + 1)
    if ff is None:
        raise RuntimeError("Bitis cizgisi gecisi bulunamadi.")

    t_first = (t1 - sf) / fps
    t_middle = (t2 - t1) / fps
    t_last = (ff - t2) / fps
    t_total = (ff - sf) / fps
    if t_total <= 0:
        raise RuntimeError("Gecersiz toplam sure.")

    speeds = {
        "first_5m_m_s": round(5.0 / t_first, 3) if t_first > 0 else None,
        "middle_10m_m_s": round(10.0 / t_middle, 3) if t_middle > 0 else None,
        "last_5m_m_s": round(5.0 / t_last, 3) if t_last > 0 else None,
        "avg_m_s": round(20.0 / t_total, 3) if t_total > 0 else None,
    }

    return {
        "fps": round(fps, 2),
        "distance_m": 20.0,
        "first_side": first_side,
        "px_per_m": round(px_per_m, 1),
        "frames": {
            "start": round(sf, 1), "turn1": round(t1, 1),
            "center_pass": round(cm, 1), "turn2": round(t2, 1),
            "finish": round(ff, 1),
        },
        "splits_s": {
            "first_5m": round(t_first, 3),
            "middle_10m": round(t_middle, 3),
            "last_5m": round(t_last, 3),
        },
        "total_time_s": round(t_total, 3),
        "speeds": speeds,
    }


def main():
    ap = argparse.ArgumentParser(description="5-10-5 pro agility shuttle")
    ap.add_argument("video")
    ap.add_argument("--left-x", type=int, required=True, dest="left_x")
    ap.add_argument("--center-x", type=int, required=True, dest="center_x")
    ap.add_argument("--right-x", type=int, required=True, dest="right_x")
    ap.add_argument("--model", default="heavy", choices=["heavy", "lite"])
    ap.add_argument("--conf", type=float, default=0.2)
    args = ap.parse_args()
    r = analyze(args.video, left_x=args.left_x, center_x=args.center_x,
                right_x=args.right_x, model=args.model, conf=args.conf)
    print(f"\n5-10-5 shuttle  ->  toplam {r['total_time_s']}s "
          f"(ilk yon: {r['first_side']})")
    sp = r["splits_s"]
    print(f"5m {sp['first_5m']}s | 10m {sp['middle_10m']}s | 5m {sp['last_5m']}s")
    print()
    json.dump(r, sys.stdout, indent=2, ensure_ascii=False)
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
