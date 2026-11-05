"""LESS — Landing Error Scoring System (resmi 17 kalem / max 19 puan).

Protokol: sporcu ~30 cm kutudan one atlar (drop), iki ayak uzerine iner
(initial contact = IC) ve hemen maksimum dikey sicrama yapar. Inis mekanigi
IC aninda ve maksimum diz fleksiyonu (pik) aninda degerlendirilir.

Kamera ayrimi (resmi tabloya gore):
  YAN (sagittal): 1 diz fleks IC, 2 kalca fleks IC, 3 govde fleks IC,
    4 ayak temasi (parmak-once), 12 diz fleks degisimi, 13 kalca fleks degisimi,
    14 govde fleks degisimi, 16 eklem hareketi (yumusak/orta/sert, 0-2 puan)
  ON (frontal): 5 diz valgus IC, 6 lateral govde fleks, 7 durus genis,
    8 durus dar, 9 parmak iceride, 10 parmak disarida, 11 ayak simetrisi,
    15 valgus degisimi
  HER IKISI: 17 genel izlenim (0-2 puan)

Kalem 1-15 her biri 0/1 puan (HATA=1), 16 ve 17 ise 0/1/2 puan. Toplam 0..19;
dusuk = iyi inis. ON kamera (--front) yoksa frontal kalemler "degerlendirilemedi"
sayilir ve less_score kismi olur.

NOT: Bazi frontal kalemler tek 2D on kameradan yaklasik olceulur (ozellikle ayak
rotasyonu 9/10 transvers duzlemde, on kameradan guvenilirligi DUSUK; confidence
alaninda belirtilir).

Kamera: YAN kamera atlama yonune dik; ON kamera sporcuya bakacak sekilde.

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

# ── Esikler (resmi rubrik proxy'leri) ────────────────────────────────────
KNEE_FLEX_IC_MIN = 30.0      # 1: IC diz fleksiyonu < 30 deg -> hata
HIP_FLEX_IC_MIN = 20.0       # 2: IC kalca fleksiyonu yoksa (~uyluk-govde ayni hat)
TRUNK_FLEX_IC_MIN = 10.0     # 3: IC govde one egik degil
KNEE_DISP_MIN = 45.0         # 12: diz fleksiyon artisi < 45 deg
HIP_DISP_MIN = 5.0           # 13: kalca pikte daha fazla fleks DEGIL
TRUNK_DISP_MIN = 5.0         # 14: govde pikte daha fazla fleks DEGIL
VALGUS_FPPA_MIN = 170.0      # 5/15: FPPA < 170 -> valgus
VALGUS_DISP_MARGIN = 5.0     # 15: IC->pik FPPA bu kadar duserse valgus artmis
LAT_TRUNK_MAX = 10.0         # 6: lateral govde egimi > 10 deg
STANCE_WIDE = 1.20           # 7: ayak/omuz orani > 1.20 -> genis
STANCE_NARROW = 0.80         # 8: ayak/omuz orani < 0.80 -> dar
TOE_ROT_RATIO = 0.45         # 9/10: ayak yatay/dikey orani (yaklasik ~25-30 deg)
FOOT_SYM_MAX = 0.04          # 11: IC'de iki ayak yuksekligi farki / govde > %4
# 16 eklem hareketi 3-seviye (toplam sagittal yer degisimi):
JOINT_SOFT = 75.0            # >=75 yumusak (0), 45-75 orta (1), <45 sert (2)
JOINT_STIFF = 45.0


def _collect_sagittal(video: str, model: str, conf: float):
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


def _frontal_metrics(video: str, model: str, conf: float) -> Optional[dict]:
    """ON kameradan IC ve pik (en derin cokme) anlarinda frontal olcumler:
    valgus (FPPA L/R), lateral govde fleks, durus genisligi, ayak rotasyonu,
    ayak temas simetrisi, valgus degisimi."""
    foot_g, hipmid_y = [], []
    snaps = []
    for fi, t, l2, l3, W, H, fps in iter_pose(video, model=model, conf=conf):
        snaps.append((l2, W, H))
        if l2 is None:
            foot_g.append(np.nan); hipmid_y.append(np.nan); continue
        ys = [pt2d(l2, i, W, H)[1] for i in (LEFT_ANKLE, RIGHT_ANKLE,
                                             LEFT_HEEL, RIGHT_HEEL)]
        foot_g.append(max(ys))
        hipmid_y.append((pt2d(l2, LEFT_HIP, W, H)[1] +
                         pt2d(l2, RIGHT_HIP, W, H)[1]) / 2)
    foot_g = np.array(foot_g); hipmid_y = np.array(hipmid_y)
    blocks, _ = detect_flight_blocks(foot_g, 30.0, thr_ratio=0.25)
    if not blocks:
        return None
    ic = blocks[0]["landing"]
    # pik = IC sonrasi en derin cokme (en buyuk hip y) ~1 sn pencere
    win_end = min(len(hipmid_y) - 1, ic + 30)
    seg = hipmid_y[ic:win_end + 1]
    peak = ic + int(np.nanargmax(seg)) if np.isfinite(seg).any() else ic

    def at(frame):
        l2, W, H = snaps[frame]
        if l2 is None:
            return None

        def P(i):
            return pt2d(l2, i, W, H)
        fppa_l = angle_deg(P(LEFT_HIP), P(LEFT_KNEE), P(LEFT_ANKLE))
        fppa_r = angle_deg(P(RIGHT_HIP), P(RIGHT_KNEE), P(RIGHT_ANKLE))
        sh_mid = ((P(LEFT_SHOULDER)[0] + P(RIGHT_SHOULDER)[0]) / 2,
                  (P(LEFT_SHOULDER)[1] + P(RIGHT_SHOULDER)[1]) / 2)
        hp_mid = ((P(LEFT_HIP)[0] + P(RIGHT_HIP)[0]) / 2,
                  (P(LEFT_HIP)[1] + P(RIGHT_HIP)[1]) / 2)
        lat_tilt = vertical_tilt_deg(np.array(hp_mid), np.array(sh_mid))
        ankle_w = abs(P(LEFT_ANKLE)[0] - P(RIGHT_ANKLE)[0])
        sh_w = abs(P(LEFT_SHOULDER)[0] - P(RIGHT_SHOULDER)[0])
        ratio = float(ankle_w / sh_w) if sh_w > 1e-6 else float("nan")
        # ayak rotasyonu (yaklasik): heel->toe vektorunun yatay/dikey orani,
        # govde orta hattina gore ic/dis yonu.
        midx = hp_mid[0]
        body_span = abs(sh_mid[1] - max(P(LEFT_ANKLE)[1], P(RIGHT_ANKLE)[1])) + 1e-6
        rot = {}
        for nm, HE, FT, AN in (("L", LEFT_HEEL, LEFT_FOOT, LEFT_ANKLE),
                               ("R", RIGHT_HEEL, RIGHT_FOOT, RIGHT_ANKLE)):
            he, ft = P(HE), P(FT)
            dy = abs(ft[1] - he[1]) + 1e-6
            ratio_h = (ft[0] - he[0]) / dy           # +: toe saga
            inward = (midx - he[0])                  # ayak orta hatta gore taraf
            # toe orta hatta dogru mu (ic) yoksa disa mi
            toward_mid = ratio_h * np.sign(inward)   # +: toe ice
            rot[nm] = float(toward_mid)
        # IC'de ayak yuksekligi simetrisi
        lf = max(P(LEFT_ANKLE)[1], P(LEFT_HEEL)[1], P(LEFT_FOOT)[1])
        rf = max(P(RIGHT_ANKLE)[1], P(RIGHT_HEEL)[1], P(RIGHT_FOOT)[1])
        foot_sym = abs(lf - rf) / body_span
        return dict(fppa_l=float(fppa_l), fppa_r=float(fppa_r),
                    lat_tilt=float(lat_tilt), stance_ratio=ratio,
                    rot=rot, foot_sym=float(foot_sym))

    a_ic, a_pk = at(ic), at(peak)
    if a_ic is None:
        return None
    return {"ic_frame": int(ic), "peak_frame": int(peak),
            "ic": a_ic, "peak": a_pk}


def _item(no, name, camera, value, points, maxp, note, confidence="yuksek"):
    d = {"no": no, "name": name, "camera": camera, "value": value,
         "points": int(points), "max": maxp, "note": note}
    if maxp == 1:
        d["error"] = bool(points >= 1)
    if confidence != "yuksek":
        d["confidence"] = confidence
    return d


def analyze(video_path: str, *, front_path: Optional[str] = None,
            model: str = "heavy", conf: float = 0.3,
            debug_dir: Optional[str] = None) -> dict:
    series, fps, frames_lm = _collect_sagittal(video_path, model, conf)
    foot = series["foot_g"]
    blocks, meta = detect_flight_blocks(foot, fps, thr_ratio=0.25)
    if not blocks:
        raise RuntimeError("Inis (ucus blogu) tespit edilemedi. Drop atlamasi "
                           "ve tam govde goruntusu gerekli.")
    ic = blocks[0]["landing"]
    next_to = blocks[1]["takeoff"] if len(blocks) >= 2 else len(foot) - 1
    knee = series["knee"]
    seg = knee[ic:next_to + 1]
    if not np.isfinite(seg).any():
        raise RuntimeError("Diz acisi takip edilemedi.")
    peak = ic + int(np.nanargmin(seg))

    knee_ic, knee_peak = float(knee[ic]), float(knee[peak])
    hip_ic, hip_peak = float(series["hip"][ic]), float(series["hip"][peak])
    trunk_ic, trunk_peak = float(series["trunk"][ic]), float(series["trunk"][peak])
    flex_ic = 180.0 - knee_ic
    hip_flex_ic = 180.0 - hip_ic
    knee_disp = knee_ic - knee_peak
    hip_disp = hip_ic - hip_peak
    trunk_disp = trunk_peak - trunk_ic
    toe_landing = bool(series["toe_y"][ic] > series["heel_y"][ic])
    total_disp = max(0.0, knee_disp) + max(0.0, hip_disp) + max(0.0, trunk_disp)

    items: list[dict] = []
    # ── YAN (sagittal) — IC kalemleri ───────────────────────────────────
    items.append(_item(1, "Diz fleksiyon acisi (IC)", "yan", round(flex_ic, 1),
                       1 if flex_ic < KNEE_FLEX_IC_MIN else 0, 1,
                       "IC'de diz fleksiyonu < 30 deg -> sert inis (hata)"))
    items.append(_item(2, "Kalca fleksiyon acisi (IC)", "yan", round(hip_flex_ic, 1),
                       1 if hip_flex_ic < HIP_FLEX_IC_MIN else 0, 1,
                       "IC'de kalca fleksiyonu yok (uyluk govde ile ayni hat)"))
    items.append(_item(3, "Govde fleksiyon acisi (IC)", "yan", round(trunk_ic, 1),
                       1 if trunk_ic < TRUNK_FLEX_IC_MIN else 0, 1,
                       "IC'de govde one egik degil (dik/ekstansiyon)"))
    items.append(_item(4, "Ayak bilegi plantar fleks. / temas (IC)", "yan",
                       toe_landing, 0 if toe_landing else 1, 1,
                       "Parmak-once temas iyi; topuk/duz temas hata"))
    # ── YAN — degisim (IC -> pik) kalemleri ─────────────────────────────
    items.append(_item(12, "Diz fleksiyonundaki degisim", "yan", round(knee_disp, 1),
                       1 if knee_disp < KNEE_DISP_MIN else 0, 1,
                       "IC->pik diz fleksiyon artisi < 45 deg (hata)"))
    items.append(_item(13, "Kalca fleksiyonu degisimi (max diz fleks.)", "yan",
                       round(hip_disp, 1), 1 if hip_disp < HIP_DISP_MIN else 0, 1,
                       "Pikte kalca IC'ye gore daha fazla fleks DEGILse hata"))
    items.append(_item(14, "Govde fleksiyonu degisimi (max diz fleks.)", "yan",
                       round(trunk_disp, 1), 1 if trunk_disp < TRUNK_DISP_MIN else 0, 1,
                       "Pikte govde IC'ye gore daha fazla fleks DEGILse hata"))
    # ── YAN — 16 eklem hareketi (0-2) ───────────────────────────────────
    if total_disp >= JOINT_SOFT:
        j_pts, j_lbl = 0, "yumusak"
    elif total_disp >= JOINT_STIFF:
        j_pts, j_lbl = 1, "orta"
    else:
        j_pts, j_lbl = 2, "sert"
    items.append(_item(16, "Eklem hareketi degisimi (sagittal)", "yan",
                       {"total_disp_deg": round(total_disp, 1), "level": j_lbl},
                       j_pts, 2, "Toplam sagittal yer degisimi: yumusak=0/orta=1/sert=2"))

    coverage = {"yan": True, "on": False}
    frontal = None
    if front_path:
        frontal = _frontal_metrics(front_path, model, conf)

    frontal_error_present = False
    if frontal:
        coverage["on"] = True
        fi, fp = frontal["ic"], frontal.get("peak") or frontal["ic"]
        fppa_min_ic = min(fi["fppa_l"], fi["fppa_r"])
        # 5: diz valgus (IC)
        v5 = 1 if fppa_min_ic < VALGUS_FPPA_MIN else 0
        items.append(_item(5, "Diz valgus acisi (IC)", "on",
                           {"fppa_left": round(fi["fppa_l"], 1),
                            "fppa_right": round(fi["fppa_r"], 1),
                            "fppa_min": round(fppa_min_ic, 1)},
                           v5, 1, "ON: FPPA < 170 deg -> valgus (hata)"))
        # 6: lateral govde fleksiyonu (IC)
        v6 = 1 if fi["lat_tilt"] > LAT_TRUNK_MAX else 0
        items.append(_item(6, "Lateral govde fleksiyonu (IC)", "on",
                           round(fi["lat_tilt"], 1), v6, 1,
                           "ON: govde yana egik (> 10 deg) -> hata"))
        # 7: durus genis
        v7 = 1 if (np.isfinite(fi["stance_ratio"]) and fi["stance_ratio"] > STANCE_WIDE) else 0
        items.append(_item(7, "Durus genisligi (genis)", "on",
                           round(fi["stance_ratio"], 2), v7, 1,
                           "ON: ayak/omuz orani > 1.20 -> genis (hata)"))
        # 8: durus dar
        v8 = 1 if (np.isfinite(fi["stance_ratio"]) and fi["stance_ratio"] < STANCE_NARROW) else 0
        items.append(_item(8, "Durus genisligi (dar)", "on",
                           round(fi["stance_ratio"], 2), v8, 1,
                           "ON: ayak/omuz orani < 0.80 -> dar (hata)"))
        # 9: parmak iceride / 10: parmak disarida (DUSUK guven)
        toe_in = max(fi["rot"]["L"], fi["rot"]["R"])     # +: ice
        toe_out = max(-fi["rot"]["L"], -fi["rot"]["R"])  # +: disa
        v9 = 1 if toe_in > TOE_ROT_RATIO else 0
        items.append(_item(9, "Ayak pozisyonu: parmak ucu iceride", "on",
                           round(toe_in, 2), v9, 1,
                           "ON: ayak ic rotasyonu (yaklasik)", confidence="dusuk"))
        v10 = 1 if toe_out > TOE_ROT_RATIO else 0
        items.append(_item(10, "Ayak pozisyonu: parmak ucu disarida", "on",
                           round(toe_out, 2), v10, 1,
                           "ON: ayak dis rotasyonu (yaklasik)", confidence="dusuk"))
        # 11: ayak temas simetrisi (IC)
        v11 = 1 if fi["foot_sym"] > FOOT_SYM_MAX else 0
        items.append(_item(11, "Ayak temas simetrisi (IC)", "on",
                           round(fi["foot_sym"], 3), v11, 1,
                           "ON: iki ayak ayni anda temas etmiyor -> asimetrik (hata)"))
        # 15: valgus degisimi (IC -> pik)
        fppa_min_pk = min(fp["fppa_l"], fp["fppa_r"]) if fp else fppa_min_ic
        valgus_inc = (fppa_min_ic - fppa_min_pk)  # + ise pikte daha valgus
        v15 = 1 if valgus_inc > VALGUS_DISP_MARGIN else 0
        items.append(_item(15, "Dizde valgus degisimi", "on",
                           round(valgus_inc, 1), v15, 1,
                           "ON: IC->pik valgus artiyorsa hata"))
        frontal_error_present = any(it["points"] >= 1 for it in items
                                    if it["no"] in (5, 6, 15))

    # ── 17 genel izlenim (her iki kamera) ───────────────────────────────
    # yumusak inis + frontal-duzlem hareketi yok -> mukemmel(0)
    # sert inis + frontal hata -> kotu(2); digerleri orta(1)
    if coverage["on"]:
        if j_lbl == "yumusak" and not frontal_error_present:
            o_pts, o_lbl = 0, "mukemmel"
        elif j_lbl == "sert" and frontal_error_present:
            o_pts, o_lbl = 2, "kotu"
        else:
            o_pts, o_lbl = 1, "orta"
        items.append(_item(17, "Genel izlenim", "yan+on", o_lbl, o_pts, 2,
                           "16 + frontal duzlem hareketine gore: mukemmel/orta/kotu"))

    items.sort(key=lambda d: d["no"])
    less_score = int(sum(it["points"] for it in items))
    max_possible = int(sum(it["max"] for it in items))
    n_eval = len(items)
    errors = int(sum(1 for it in items if it["points"] >= 1))

    # Klasik LESS yorumu (tam 17 kalem uzerinden): <=4 mukemmel, 5 iyi,
    # 6 orta, >6 zayif. Kismi (on yoksa) ham skoru raporlariz.
    if not coverage["on"]:
        grade = "kismi (ON kamera yok)"
    elif less_score <= 4:
        grade = "mukemmel"
    elif less_score <= 5:
        grade = "iyi"
    elif less_score <= 6:
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
        "items_evaluated": n_eval,
        "errors": errors,
        "less_score": less_score,
        "max_possible": max_possible,
        "grade": grade,
        "cameras": {"side": True, "front": coverage["on"]},
        "frontal_meta": ({"ic_frame": frontal["ic_frame"],
                          "peak_frame": frontal["peak_frame"]}
                         if frontal else None),
    }
    if not coverage["on"]:
        result["note"] = ("ON kamera (front) verilmedi: 5,6,7,8,9,10,11,15,17 "
                          "kalemleri degerlendirilemedi; less_score kismi.")

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
        txt = f"LESS={result['less_score']}/{result['max_possible']} ({result['grade']})  [{tag}]"
        cv2.putText(img, txt, (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.1,
                    (0, 0, 0), 6, cv2.LINE_AA)
        cv2.putText(img, txt, (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.1,
                    (0, 255, 255), 2, cv2.LINE_AA)
        cv2.imwrite(str(out_dir / f"less_{tag}.jpg"), img)
    cap.release()


def main():
    ap = argparse.ArgumentParser(description="LESS (Landing Error Scoring System)")
    ap.add_argument("video", help="YAN kamera drop-jump videosu")
    ap.add_argument("--front", default=None, help="ON kamera videosu (frontal kalemler)")
    ap.add_argument("--model", default="heavy", choices=["heavy", "lite"])
    ap.add_argument("--conf", type=float, default=0.3)
    ap.add_argument("--debug", default=None, help="IC/pik kareleri icin cikti klasoru")
    args = ap.parse_args()
    result = analyze(args.video, front_path=args.front, model=args.model,
                     conf=args.conf, debug_dir=args.debug)
    print(f"\nLESS skor: {result['less_score']} / {result['max_possible']} "
          f"({result['items_evaluated']} kalem)  ->  {result['grade']}")
    print(f"IC kare #{result['ic_frame']} (t={result['ic_time_s']}s), "
          f"pik fleksiyon #{result['peak_flexion_frame']} "
          f"(t={result['peak_flexion_time_s']}s)")
    print("\n#   Kalem                                   kamera  puan  deger")
    print("-" * 72)
    for it in result["items"]:
        val = it["value"]
        vs = (json.dumps(val, ensure_ascii=False) if isinstance(val, dict)
              else str(val))
        conf = "  (dusuk guven)" if it.get("confidence") == "dusuk" else ""
        print(f"{it['no']:<3} {it['name']:<38} {it['camera']:<6} "
              f"{it['points']}/{it['max']}   {vs}{conf}")
    if not result["cameras"]["front"]:
        print(f"\n{result.get('note','')}")
    print()
    json.dump(result, sys.stdout, indent=2, ensure_ascii=False)
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
