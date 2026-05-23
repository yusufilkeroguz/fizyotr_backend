"""LESS — Landing Error Scoring System analizi.

Protokol: sporcu ~30 cm yukseklikteki kutudan one atlar (drop), iki ayak
uzerine iner (initial contact = IC) ve hemen maksimum dikey sicrama yapar.
Inis mekanigi, IC aninda ve maksimum diz fleksiyonu (pik) aninda degerlendirilir.

Tam LESS 17 kalemdir ve klasik olarak ON + YAN iki kamera ister. Bu analizci:
  - YAN (sagittal) kamera ile pose-tabanli olceulebilen kalemleri hesaplar
    (diz/kalca/govde fleksiyonu, ayak temas tarzi, fleksiyon yer degisimi).
  - --front verilirse ON kamera videosundan diz valgus + stance genisligi
    kalemlerini ekler.
Degerlendirilemeyen kalemler "coverage" altinda raporlanir; less_score
degerlendirilen kalemlerdeki HATA sayisidir (dusuk = iyi inis).

Kamera: YAN kamera atlama yonune dik, tum govde gorunur olmali.

CLI:
    python less.py drop_yan.mp4
    python less.py drop_yan.mp4 --front drop_on.mp4 --debug less_dbg
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

import numpy as np

from ..pose_common import (
    iter_pose, pt2d, angle_deg, vertical_tilt_deg, pick_visible_side,
    detect_flight_blocks,
    NOSE, LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_HIP, RIGHT_HIP,
    LEFT_KNEE, RIGHT_KNEE, LEFT_ANKLE, RIGHT_ANKLE,
    LEFT_HEEL, RIGHT_HEEL, LEFT_FOOT, RIGHT_FOOT,
)


def _collect_sagittal(video: str, model: str, conf: float) -> tuple[dict, float, list]:
    foot_g, knee, hip, trunk, heel_y, toe_y = [], [], [], [], [], []
    frames_lm = []
    fps = 30.0
    for fi, t, l2, l3, W, H, fps in iter_pose(video, model=model, conf=conf):
        frames_lm.append((l2, W, H))
        if l2 is None:
            for L in (foot_g, knee, hip, trunk, heel_y, toe_y):
                L.append(np.nan)
            continue
        side = pick_visible_side(l2, LEFT_KNEE, RIGHT_KNEE)
        SH = LEFT_SHOULDER if side == "left" else RIGHT_SHOULDER
        HP = LEFT_HIP if side == "left" else RIGHT_HIP
        KN = LEFT_KNEE if side == "left" else RIGHT_KNEE
        AN = LEFT_ANKLE if side == "left" else RIGHT_ANKLE
        HE = LEFT_HEEL if side == "left" else RIGHT_HEEL
        FT = LEFT_FOOT if side == "left" else RIGHT_FOOT

        def P(i):
            return pt2d(l2, i, W, H)

        sh, hp, kn, an, he, ft = P(SH), P(HP), P(KN), P(AN), P(HE), P(FT)
        knee.append(angle_deg(hp, kn, an))
        hip.append(angle_deg(sh, hp, kn))
        trunk.append(vertical_tilt_deg(hp, sh))  # 0 dik, one egilince artar
        heel_y.append(he[1])
        toe_y.append(ft[1])
        ys = [P(i)[1] for i in (LEFT_ANKLE, RIGHT_ANKLE, LEFT_HEEL,
                                RIGHT_HEEL, LEFT_FOOT, RIGHT_FOOT)]
        foot_g.append(max(ys))
    series = dict(
        foot_g=np.array(foot_g), knee=np.array(knee), hip=np.array(hip),
        trunk=np.array(trunk), heel_y=np.array(heel_y), toe_y=np.array(toe_y),
    )
    return series, fps, frames_lm


def _frontal_items(video: str, model: str, conf: float) -> Optional[dict]:
    """ON kameradan IC aninda diz valgus (FPPA) + stance genisligi."""
    foot_g = []
    snaps = []  # (l2, W, H)
    for fi, t, l2, l3, W, H, fps in iter_pose(video, model=model, conf=conf):
        snaps.append((l2, W, H))
        if l2 is None:
            foot_g.append(np.nan)
            continue
        ys = [pt2d(l2, i, W, H)[1] for i in (LEFT_ANKLE, RIGHT_ANKLE,
                                             LEFT_HEEL, RIGHT_HEEL)]
        foot_g.append(max(ys))
    blocks, _ = detect_flight_blocks(np.array(foot_g), 30.0, thr_ratio=0.25)
    if not blocks:
        return None
    ic = blocks[0]["landing"]
    l2, W, H = snaps[ic]
    if l2 is None:
        return None

    def P(i):
        return pt2d(l2, i, W, H)

    fppa_l = angle_deg(P(LEFT_HIP), P(LEFT_KNEE), P(LEFT_ANKLE))
    fppa_r = angle_deg(P(RIGHT_HIP), P(RIGHT_KNEE), P(RIGHT_ANKLE))
    fppa_min = float(np.nanmin([fppa_l, fppa_r]))
    ankle_w = abs(P(LEFT_ANKLE)[0] - P(RIGHT_ANKLE)[0])
    shoulder_w = abs(P(LEFT_SHOULDER)[0] - P(RIGHT_SHOULDER)[0])
    width_ratio = float(ankle_w / shoulder_w) if shoulder_w > 1e-6 else float("nan")
    return {
        "ic_frame": int(ic),
        "fppa_left_deg": round(float(fppa_l), 1),
        "fppa_right_deg": round(float(fppa_r), 1),
        "fppa_min_deg": round(fppa_min, 1),
        "knee_valgus_error": bool(fppa_min < 170.0),
        "stance_width_ratio": round(width_ratio, 2),
        "stance_width_error": bool(width_ratio < 0.8 or width_ratio > 1.6),
    }


def analyze(video_path: str, *, front_path: Optional[str] = None,
            model: str = "heavy", conf: float = 0.3,
            debug_dir: Optional[str] = None) -> dict:
    series, fps, frames_lm = _collect_sagittal(video_path, model, conf)
    foot = series["foot_g"]
    blocks, meta = detect_flight_blocks(foot, fps, thr_ratio=0.25)
    if not blocks:
        raise RuntimeError("Inis (ucus blogu) tespit edilemedi. Drop atlamasi "
                           "ve tam govde goruntusu gerekli.")
    ic = blocks[0]["landing"]            # ilk dususun yere temas ani = IC
    next_to = blocks[1]["takeoff"] if len(blocks) >= 2 else len(foot) - 1
    knee = series["knee"]
    seg = knee[ic:next_to + 1]
    if not np.isfinite(seg).any():
        raise RuntimeError("Diz acisi takip edilemedi.")
    peak = ic + int(np.nanargmin(seg))   # en kucuk diz acisi = en derin fleksiyon

    knee_ic, knee_peak = float(knee[ic]), float(knee[peak])
    hip_ic, hip_peak = float(series["hip"][ic]), float(series["hip"][peak])
    trunk_ic, trunk_peak = float(series["trunk"][ic]), float(series["trunk"][peak])
    flex_ic = 180.0 - knee_ic
    knee_disp = knee_ic - knee_peak          # diz fleksiyon yer degisimi
    hip_disp = hip_ic - hip_peak
    trunk_disp = trunk_peak - trunk_ic
    toe_landing = bool(series["toe_y"][ic] > series["heel_y"][ic])
    total_disp = knee_disp + hip_disp + trunk_disp

    items: dict[str, dict] = {
        "knee_flexion_ic": {
            "value_deg": round(flex_ic, 1),
            "error": bool(flex_ic < 30.0),
            "note": "IC'de diz fleksiyonu < 30 deg (sert inis)"},
        "hip_flexion_ic": {
            "value_deg": round(180.0 - hip_ic, 1),
            "error": bool((180.0 - hip_ic) < 30.0),
            "note": "IC'de kalca fleksiyonu yetersiz"},
        "trunk_flexion_ic": {
            "value_deg": round(trunk_ic, 1),
            "error": bool(trunk_ic < 10.0),
            "note": "IC'de govde one egik degil"},
        "ankle_toe_landing": {
            "value": toe_landing,
            "error": bool(not toe_landing),
            "note": "Topuk/duz inis (parmak-once degil)"},
        "knee_flexion_displacement": {
            "value_deg": round(knee_disp, 1),
            "error": bool(knee_disp < 45.0),
            "note": "IC->pik diz fleksiyon artisi < 45 deg"},
        "trunk_flexion_displacement": {
            "value_deg": round(trunk_disp, 1),
            "error": bool(trunk_disp < 5.0),
            "note": "IC->pik govde fleksiyon artisi yetersiz"},
        "joint_displacement": {
            "value_deg": round(total_disp, 1),
            "error": bool(total_disp < 60.0),
            "note": "Toplam eklem yer degisimi az (sert/yumusak-degil inis)"},
    }

    coverage = ["sagittal: 7 kalem"]
    frontal = None
    if front_path:
        frontal = _frontal_items(front_path, model, conf)
        if frontal:
            items["knee_valgus"] = {
                "value_deg": frontal["fppa_min_deg"],
                "error": frontal["knee_valgus_error"],
                "note": "ON kamera: diz valgus (FPPA < 170 deg)"}
            items["stance_width"] = {
                "value_ratio": frontal["stance_width_ratio"],
                "error": frontal["stance_width_error"],
                "note": "ON kamera: stance genisligi/omuz orani anormal"}
            coverage.append("frontal: 2 kalem (valgus, stance)")
        else:
            coverage.append("frontal: ON videoda inis tespit edilemedi")
    not_covered = ["lateral govde fleksiyonu", "ayak rotasyonu",
                   "asimetrik ayak temasi", "genel izlenim"]

    less_score = int(sum(1 for it in items.values() if it["error"]))
    n_items = len(items)
    if less_score <= 2:
        grade = "iyi"
    elif less_score <= 4:
        grade = "orta"
    else:
        grade = "zayif"

    result = {
        "fps": round(fps, 2),
        "ic_frame": int(ic),
        "ic_time_s": round(ic / fps, 3),
        "peak_flexion_frame": int(peak),
        "peak_flexion_time_s": round(peak / fps, 3),
        "knee_angle_ic_deg": round(knee_ic, 1),
        "knee_angle_peak_deg": round(knee_peak, 1),
        "items": items,
        "items_evaluated": n_items,
        "less_score": less_score,
        "grade": grade,
        "coverage": coverage,
        "not_evaluated": not_covered,
        "frontal": frontal,
    }

    if debug_dir:
        _draw_debug(video_path, ic, peak, result, frames_lm, Path(debug_dir))
    return result


def _draw_debug(video, ic, peak, result, frames_lm, out_dir: Path):
    import cv2
    out_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(video)
    for tag, fr in (("ic", ic), ("peak", peak)):
        cap.set(cv2.CAP_PROP_POS_FRAMES, fr)
        ok, img = cap.read()
        if not ok:
            continue
        H, W = img.shape[:2]
        l2, _w, _h = frames_lm[fr] if fr < len(frames_lm) else (None, W, H)
        if l2 is not None:
            for i in (LEFT_HIP, LEFT_KNEE, LEFT_ANKLE, RIGHT_HIP,
                      RIGHT_KNEE, RIGHT_ANKLE):
                p = pt2d(l2, i, W, H).astype(int)
                cv2.circle(img, tuple(p), 8, (0, 220, 255), -1)
        txt = f"LESS skor={result['less_score']} ({result['grade']})  [{tag}]"
        cv2.putText(img, txt, (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.2,
                    (0, 0, 0), 6, cv2.LINE_AA)
        cv2.putText(img, txt, (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.2,
                    (0, 255, 255), 2, cv2.LINE_AA)
        cv2.imwrite(str(out_dir / f"less_{tag}.jpg"), img)
    cap.release()


def main():
    ap = argparse.ArgumentParser(description="LESS (Landing Error Scoring System)")
    ap.add_argument("video", help="YAN kamera drop-jump videosu")
    ap.add_argument("--front", default=None, help="ON kamera videosu (valgus/stance)")
    ap.add_argument("--model", default="heavy", choices=["heavy", "lite"])
    ap.add_argument("--conf", type=float, default=0.3)
    ap.add_argument("--debug", default=None, help="IC/pik kareleri icin cikti klasoru")
    args = ap.parse_args()
    result = analyze(args.video, front_path=args.front, model=args.model,
                     conf=args.conf, debug_dir=args.debug)
    print(f"\nLESS skor: {result['less_score']} / {result['items_evaluated']} "
          f"kalem  ->  {result['grade']}")
    print(f"IC kare #{result['ic_frame']} (t={result['ic_time_s']}s), "
          f"pik fleksiyon #{result['peak_flexion_frame']} "
          f"(t={result['peak_flexion_time_s']}s)")
    print("\nKalem                          deger      hata")
    print("-" * 52)
    for name, it in result["items"].items():
        val = it.get("value_deg", it.get("value_ratio", it.get("value")))
        print(f"{name:<28} {str(val):>8}   {'HATA' if it['error'] else 'ok'}")
    print(f"\nKapsam: {', '.join(result['coverage'])}")
    print()
    json.dump(result, sys.stdout, indent=2, ensure_ascii=False)
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
