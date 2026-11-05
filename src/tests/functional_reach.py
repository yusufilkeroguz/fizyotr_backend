"""Functional Reach Test (FRT) — ileri uzanma testi analizi.

Protokol: kisi yana dik durur, kamera tarafindaki kolunu ~90 deg one (yatay)
kaldirir, ardindan adim atmadan govdesini one egerek mumkun oldugunca ileri
uzanir. Olculen: parmak ucunun (burada bilek) baslangic ile maksimum uzanma
arasindaki YATAY yer degisimi (cm).

Kamera: YAN kamera, uzanan kol kameraya yakin tarafta, tum govde gorunur.
Boy (--height) cm kalibrasyonu icin gerekir.

CLI:
    python functional_reach.py frt.mp4 --height 1.75 --debug frt_dbg
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

import numpy as np

from ..pose_common import (
    iter_pose, pt2d, smooth_median, body_px_per_m,
    NOSE, LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_WRIST, RIGHT_WRIST,
    LEFT_HIP, RIGHT_HIP, LEFT_ANKLE, RIGHT_ANKLE,
)


def analyze(video_path: str, *, height_m: Optional[float] = None,
            model: str = "heavy", conf: float = 0.3,
            debug_dir: Optional[str] = None) -> dict:
    wl_x, wl_y, wr_x, wr_y = [], [], [], []
    sh_x, sh_y, hip_x, nose_ank = [], [], [], []
    ank_x = []
    wl_vis, wr_vis = [], []
    fps = 30.0
    for fi, t, l2, l3, W, H, fps in iter_pose(video_path, model=model, conf=conf):
        if l2 is None:
            for L in (wl_x, wl_y, wr_x, wr_y, sh_x, sh_y, hip_x, ank_x,
                      nose_ank, wl_vis, wr_vis):
                L.append(np.nan)
            continue
        lw, rw = pt2d(l2, LEFT_WRIST, W, H), pt2d(l2, RIGHT_WRIST, W, H)
        ls, rs = pt2d(l2, LEFT_SHOULDER, W, H), pt2d(l2, RIGHT_SHOULDER, W, H)
        lh, rh = pt2d(l2, LEFT_HIP, W, H), pt2d(l2, RIGHT_HIP, W, H)
        la, ra = pt2d(l2, LEFT_ANKLE, W, H), pt2d(l2, RIGHT_ANKLE, W, H)
        wl_x.append(lw[0]); wl_y.append(lw[1]); wl_vis.append(l2[LEFT_WRIST].visibility)
        wr_x.append(rw[0]); wr_y.append(rw[1]); wr_vis.append(l2[RIGHT_WRIST].visibility)
        sh_x.append(float((ls[0] + rs[0]) / 2))
        sh_y.append(float((ls[1] + rs[1]) / 2))
        hip_x.append(float((lh[0] + rh[0]) / 2))
        ank_x.append(float((la[0] + ra[0]) / 2))   # sabit referans (ayaklar planted)
        nose = pt2d(l2, NOSE, W, H)
        ank_y = max(la[1], ra[1])
        nose_ank.append(abs(ank_y - nose[1]))

    sx = np.array(sh_x)
    sy = np.array(sh_y)
    body_h = np.nanpercentile(np.array(nose_ank), 90)
    y_tol = 0.18 * body_h if np.isfinite(body_h) else 60.0

    # Uzanan kol: omuz hizasinda (yatay kaldirilmis) en cok kalan bilek.
    # Gorunurluk degil, cunku FRT'de iki kol da gorunur; ayirt eden, kolun
    # gercekten omuz seviyesine kaldirilmis olmasidir.
    def raised_count(wy_arr):
        return int(np.nansum(np.abs(np.array(wy_arr) - sy) < y_tol))
    nl, nr = raised_count(wl_y), raised_count(wr_y)
    if max(nl, nr) >= 3:
        arm = "left" if nl >= nr else "right"
    else:  # ikisi de kaldirilmamis; gorunurluge geri don
        arm = "left" if np.nanmean(wl_vis) >= np.nanmean(wr_vis) else "right"
    wx = np.array(wl_x if arm == "left" else wr_x)
    wy = np.array(wl_y if arm == "left" else wr_y)
    if np.isfinite(wx).sum() < 5:
        raise RuntimeError("Bilek takip edilemedi.")

    wxs = smooth_median(wx, 5)
    # Kol-yatay (kaldirilmis) kareler: bilek omuz hizasinda (|wy - sy| kucuk)
    raised = np.abs(wy - sy) < y_tol
    raised &= ~np.isnan(wxs)
    if raised.sum() < 3:
        raise RuntimeError("Kol-yatay (kaldirilmis) faz tespit edilemedi.")
    raised_idx = np.where(raised)[0]

    # Reach = bilegin YATAY yer degisimi, SABIT referansa (ayak bilegi) gore.
    # Omuza gore degil: FRT'de kisi govdesini one egerek uzanir, bu sirada omuz
    # da one gider; bilek-omuz farki neredeyse sabit kalip uzanmayi olcemez.
    # Ayaklar planted oldugundan ayak bilegi x'i sabit referanstir; ona gore
    # olcmek hem gercek uzanmayi yakalar hem olasi kamera kaymasini siler.
    axs = smooth_median(np.array(ank_x), 5)
    rel = wxs - axs

    # Noter konum: kol kaldirilmis ama govde dik (uzanma faznin BASI). Ilk birkac
    # raised karenin medyani. Reach = bunun uzerinden en buyuk YATAY sapma.
    # Yonu ilk/son kareyi kiyaslayarak DEGIL (kisi sonda kolunu indirip noter'in
    # gerisine gecebilir, bu yon tespitini ters cevirir), en buyuk |sapma| ile
    # buluruz; boylece kamera yonunden (ileri = +x veya -x) bagimsiz.
    base = float(np.nanmedian(rel[raised_idx[:5]]))
    dev = rel - base
    start_frame = int(raised_idx[0])
    dev_masked = np.where(raised, np.abs(dev), np.nan)
    peak_frame = int(np.nanargmax(dev_masked))
    reach_px = float(abs(dev[peak_frame]))

    ppm = body_px_per_m(np.array(nose_ank), height_m) if height_m else None
    reach_cm = round(reach_px / ppm * 100, 1) if ppm else None

    result = {
        "fps": round(fps, 2),
        "arm_side": arm,
        "reach_cm": reach_cm,
        "reach_px": round(float(reach_px), 1),
        "start_frame": start_frame,
        "start_time_s": round(start_frame / fps, 3),
        "peak_frame": peak_frame,
        "peak_time_s": round(peak_frame / fps, 3),
        "px_per_m": round(ppm, 1) if ppm else None,
    }
    if debug_dir:
        _draw_debug(video_path, result, arm, Path(debug_dir))
    return result


def _draw_debug(video, result, arm, out_dir: Path):
    import cv2
    out_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(video)
    for tag, fr in (("start", result["start_frame"]), ("peak", result["peak_frame"])):
        cap.set(cv2.CAP_PROP_POS_FRAMES, fr)
        ok, img = cap.read()
        if not ok:
            continue
        label = (f"FRT reach={result['reach_cm']} cm  [{tag}]" if result["reach_cm"]
                 else f"FRT reach={result['reach_px']} px  [{tag}]")
        cv2.putText(img, label, (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.1,
                    (0, 0, 0), 6, cv2.LINE_AA)
        cv2.putText(img, label, (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.1,
                    (0, 255, 255), 2, cv2.LINE_AA)
        cv2.imwrite(str(out_dir / f"frt_{tag}.jpg"), img)
    cap.release()


def main():
    ap = argparse.ArgumentParser(description="Functional Reach Test")
    ap.add_argument("video", help="YAN kamera videosu")
    ap.add_argument("--height", type=float, default=None,
                    help="Boy (m) — cm kalibrasyonu icin")
    ap.add_argument("--model", default="heavy", choices=["heavy", "lite"])
    ap.add_argument("--conf", type=float, default=0.3)
    ap.add_argument("--debug", default=None)
    args = ap.parse_args()
    r = analyze(args.video, height_m=args.height, model=args.model,
                conf=args.conf, debug_dir=args.debug)
    val = f"{r['reach_cm']} cm" if r["reach_cm"] is not None else f"{r['reach_px']} px (boy verin)"
    print(f"\nFunctional Reach  ->  {val}   (kol: {r['arm_side']})")
    print(f"Baslangic kare #{r['start_frame']} (t={r['start_time_s']}s), "
          f"pik #{r['peak_frame']} (t={r['peak_time_s']}s)")
    print()
    json.dump(r, sys.stdout, indent=2, ensure_ascii=False)
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
