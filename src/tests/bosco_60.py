"""Bosco surekli sicrama testi (15 / 30 / 60 sn) analizi.

Protokol: sporcu eller belde, belirlenen sure boyunca araliksiz, her seferinde
maksimum dikey CMJ yapar. Yere temas suresi minimum, ucus maksimum hedeflenir.

Hesaplananlar (Bosco ve ark.):
  - sicrama sayisi (n), her sicramanin ucus suresi ve yuksekligi
  - toplam ucus suresi Tf, test suresi Tt
  - ortalama mekanik guc:  P = (g^2 * Tf * Tt) / (4 * n * (Tt - Tf))   [W/kg]
  - yorgunluk indeksi: ilk %25 vs son %25 ortalama yukseklik dususu

Kamera: YAN kamera, tum govde + ayaklar gorunur olmali.

CLI:
    python bosco_60.py bosco.mp4 --duration 60
"""
from __future__ import annotations

import argparse
import json
import sys

import numpy as np

from ..pose_common import (
    iter_pose, pt2d, detect_flight_blocks, flight_time_to_height_cm, GRAVITY,
    LEFT_ANKLE, RIGHT_ANKLE, LEFT_HEEL, RIGHT_HEEL, LEFT_FOOT, RIGHT_FOOT,
)


def analyze(video_path: str, *, duration_s: float = 60.0,
            model: str = "heavy", conf: float = 0.2) -> dict:
    foot_g = []
    fps = 30.0
    for fi, t, l2, l3, W, H, fps in iter_pose(video_path, model=model, conf=conf):
        if l2 is None:
            foot_g.append(np.nan)
            continue
        ys = [pt2d(l2, i, W, H)[1] for i in (LEFT_ANKLE, RIGHT_ANKLE,
                                             LEFT_HEEL, RIGHT_HEEL,
                                             LEFT_FOOT, RIGHT_FOOT)]
        foot_g.append(max(ys))
    foot = np.array(foot_g)
    blocks, meta = detect_flight_blocks(foot, fps, thr_ratio=0.2,
                                        min_air_frames=2, max_flight_s=1.2)
    if not blocks:
        raise RuntimeError("Hic sicrama tespit edilemedi.")

    # Test penceresi: ilk kalkistan itibaren duration_s
    first_to = blocks[0]["takeoff_sub"] / fps
    window = [b for b in blocks
              if (b["takeoff_sub"] / fps - first_to) <= duration_s]
    if not window:
        window = blocks

    jumps = []
    for i, b in enumerate(window):
        ft = b["flight_time_s"]
        h = flight_time_to_height_cm(ft)
        # temas suresi: bu sicramanin kalkisi - onceki sicramanin inisi
        if i > 0:
            tc = (b["takeoff_sub"] - window[i - 1]["landing_sub"]) / fps
            tc = round(tc, 4) if tc > 0 else None
        else:
            tc = None
        jumps.append({
            "index": i + 1,
            "flight_time_ms": round(ft * 1000, 1),
            "height_cm": round(h, 1),
            "contact_time_ms": round(tc * 1000, 1) if tc else None,
            "takeoff_time_s": round(b["takeoff_sub"] / fps, 3),
        })

    n = len(jumps)
    heights = np.array([j["height_cm"] for j in jumps])
    flights = np.array([j["flight_time_ms"] for j in jumps]) / 1000.0
    Tf = float(flights.sum())
    Tt = (window[-1]["landing_sub"] - window[0]["takeoff_sub"]) / fps
    Tt = max(Tt, Tf + 1e-3)

    # Bosco mekanik guc (W/kg)
    if n > 0 and (Tt - Tf) > 1e-6:
        power = (GRAVITY ** 2 * Tf * Tt) / (4.0 * n * (Tt - Tf))
    else:
        power = float("nan")

    # Yorgunluk: ilk %25 vs son %25 (en az 1'er sicrama)
    q = max(1, n // 4)
    first_q = float(np.mean(heights[:q]))
    last_q = float(np.mean(heights[-q:]))
    fatigue_pct = (first_q - last_q) / first_q * 100.0 if first_q > 0 else float("nan")

    contacts = [j["contact_time_ms"] for j in jumps if j["contact_time_ms"]]

    return {
        "fps": round(fps, 2),
        "duration_target_s": duration_s,
        "test_time_s": round(Tt, 2),
        "n_jumps": n,
        "total_flight_time_s": round(Tf, 3),
        "avg_height_cm": round(float(heights.mean()), 1),
        "best_height_cm": round(float(heights.max()), 1),
        "avg_flight_time_ms": round(float(flights.mean()) * 1000, 1),
        "avg_contact_time_ms": round(float(np.mean(contacts)), 1) if contacts else None,
        "mean_power_w_per_kg": round(power, 1) if np.isfinite(power) else None,
        "fatigue_index_pct": round(fatigue_pct, 1) if np.isfinite(fatigue_pct) else None,
        "first_quarter_height_cm": round(first_q, 1),
        "last_quarter_height_cm": round(last_q, 1),
        "jumps": jumps,
    }


def main():
    ap = argparse.ArgumentParser(description="Bosco surekli sicrama testi")
    ap.add_argument("video")
    ap.add_argument("--duration", type=float, default=60.0,
                    help="test suresi (sn): 15/30/60. Default 60.")
    ap.add_argument("--model", default="heavy", choices=["heavy", "lite"])
    ap.add_argument("--conf", type=float, default=0.2)
    args = ap.parse_args()
    r = analyze(args.video, duration_s=args.duration, model=args.model, conf=args.conf)
    print(f"\nBosco {r['duration_target_s']:.0f}s  ->  "
          f"{r['n_jumps']} sicrama, test suresi {r['test_time_s']}s")
    print(f"Ort. yukseklik: {r['avg_height_cm']} cm   "
          f"en iyi: {r['best_height_cm']} cm")
    print(f"Toplam ucus: {r['total_flight_time_s']}s   "
          f"ort. temas: {r['avg_contact_time_ms']} ms")
    print(f"Mekanik guc: {r['mean_power_w_per_kg']} W/kg   "
          f"yorgunluk: {r['fatigue_index_pct']}%")
    print()
    json.dump(r, sys.stdout, indent=2, ensure_ascii=False)
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
