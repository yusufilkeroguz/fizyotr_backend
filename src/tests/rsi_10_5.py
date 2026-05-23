"""RSI 10/5 — Reactive Strength Index (tekrarli sicrama) analizi.

Protokol: sporcu eller belde, dizleri mumkun oldugunca duz tutarak ("pogo")
ardisik tekrarli sicramalar yapar. Amac: yere temas suresini en aza indirip
ucus suresini en yuksek tutmak. 10/5 protokolunde ~10 sicrama yapilir ve en
iyi 5'i degerlendirilir.

    RSI = ucus suresi (Tf) / yere temas suresi (Tc)

Kamera: YAN kamera, tum govde + ayaklar gorunur olmali.

CLI:
    python rsi_10_5.py rsi.mp4 --best 5
"""
from __future__ import annotations

import argparse
import json
import sys

import numpy as np

from ..pose_common import (
    iter_pose, pt2d, detect_flight_blocks, flight_time_to_height_cm,
    LEFT_ANKLE, RIGHT_ANKLE, LEFT_HEEL, RIGHT_HEEL, LEFT_FOOT, RIGHT_FOOT,
)


def analyze(video_path: str, *, best_n: int = 5,
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
    blocks, meta = detect_flight_blocks(foot, fps, thr_ratio=0.25,
                                        min_air_frames=2, max_flight_s=1.2)
    if len(blocks) < 2:
        raise RuntimeError("Yeterli tekrarli sicrama tespit edilemedi "
                           f"({len(blocks)} ucus blogu).")

    hops = []
    for i, b in enumerate(blocks):
        ft = b["flight_time_s"]
        # temas suresi: bu sicramanin kalkisindan ONCEKI temas
        if i > 0:
            tc = (b["takeoff_sub"] - blocks[i - 1]["landing_sub"]) / fps
        else:
            tc = None
        rsi = (ft / tc) if (tc and tc > 0) else None
        hops.append({
            "index": i + 1,
            "flight_time_ms": round(ft * 1000, 1),
            "contact_time_ms": round(tc * 1000, 1) if tc else None,
            "height_cm": round(flight_time_to_height_cm(ft), 1),
            "rsi": round(rsi, 2) if rsi is not None else None,
        })

    valid_rsi = [h["rsi"] for h in hops if h["rsi"] is not None]
    if not valid_rsi:
        raise RuntimeError("RSI hesaplanamadi (temas suresi yakalanamadi).")
    valid_rsi_sorted = sorted(valid_rsi, reverse=True)
    top = valid_rsi_sorted[:best_n]
    contacts = [h["contact_time_ms"] for h in hops if h["contact_time_ms"]]
    flights = [h["flight_time_ms"] for h in hops]

    return {
        "fps": round(fps, 2),
        "n_hops": len(hops),
        "n_valid_rsi": len(valid_rsi),
        "best_rsi": round(max(valid_rsi), 2),
        "best_n": best_n,
        "mean_best_rsi": round(float(np.mean(top)), 2),
        "mean_rsi": round(float(np.mean(valid_rsi)), 2),
        "mean_contact_time_ms": round(float(np.mean(contacts)), 1) if contacts else None,
        "mean_flight_time_ms": round(float(np.mean(flights)), 1),
        "hops": hops,
    }


def main():
    ap = argparse.ArgumentParser(description="RSI 10/5 tekrarli sicrama testi")
    ap.add_argument("video")
    ap.add_argument("--best", type=int, default=5,
                    help="ortalamasi alinacak en iyi sicrama sayisi (default 5)")
    ap.add_argument("--model", default="heavy", choices=["heavy", "lite"])
    ap.add_argument("--conf", type=float, default=0.2)
    args = ap.parse_args()
    r = analyze(args.video, best_n=args.best, model=args.model, conf=args.conf)
    print(f"\nRSI 10/5  ->  {r['n_hops']} sicrama "
          f"({r['n_valid_rsi']} gecerli RSI)")
    print(f"En iyi RSI: {r['best_rsi']}   "
          f"en iyi {r['best_n']} ort.: {r['mean_best_rsi']}")
    print(f"Ort. temas: {r['mean_contact_time_ms']} ms   "
          f"ort. ucus: {r['mean_flight_time_ms']} ms")
    print(f"\n{'#':>3} {'ucus(ms)':>9} {'temas(ms)':>10} {'yuk(cm)':>8} {'RSI':>6}")
    print("-" * 40)
    for h in r["hops"]:
        print(f"{h['index']:>3} {h['flight_time_ms']:>9} "
              f"{str(h['contact_time_ms']):>10} {h['height_cm']:>8} "
              f"{str(h['rsi']):>6}")
    print()
    json.dump(r, sys.stdout, indent=2, ensure_ascii=False)
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
