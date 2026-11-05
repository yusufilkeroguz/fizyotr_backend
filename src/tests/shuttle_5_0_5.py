"""5-0-5 testi — izole yon degistirme (change of direction, COD) surati.

Protokol: sporcu ~10-15 m hizlanma kosusuyla ZAMANLAMA cizgisinden (line_x)
ucarak gecer, 5 m ileride DONUS cizgisine (turn_x) gelir, 180 derece doner
ve 5 m geri kosup zamanlama cizgisinden tekrar gecer. Olculen bolum yalnizca
5 m gidis + 5 m donus = 10 m'dir (hizlanma kosusu sureye dahil degildir).

Donus cizgisinde sporcu cizgiyi GENELDE GECMEZ — yavaslayip doner. Bu yuzden
donus ani, hip_x'in donus yonundeki global extremumu (argmax/argmin) olarak
bulunur (bkz. shuttle_5_10_5 ile ayni yaklasim). Finish'ten sonraki asiri
kosu (follow-through) ters yonde oldugu icin bu extremumu etkilemez.

Ucan baslangic standarttir; sporcu videoda zamanlama cizgisinin gerisinden
kosarak girer ve start = line_x'in donus yonunde ilk gecisi. Sporcu videoya
cizginin ustunde durarak basliyorsa (ayakta-start varyanti) gecis bulunamaz;
o zaman motion_onset ile rest pozisyonundan ilk sapma start sayilir.

Kullanim: 2 piksel cizgisi:
    line_x  (zamanlama / start-finish cizgisi)
    turn_x  (donus cizgisi, 5 m otede)
Kalibre:
    px_per_m = |turn_x - line_x| / 5      (cizgiler arasi gercek mesafe 5 m)

CLI:
    python shuttle_5_0_5.py run.mp4 --line-x 300 --turn-x 1500
"""
from __future__ import annotations

import argparse
import json
import sys
from typing import Optional

import numpy as np

from ..pose_common import (
    iter_pose, pt2d, smooth_median, crossing_frame, motion_onset,
    LEFT_HIP, RIGHT_HIP,
)


ONSET_SPEED_M_S = 0.3     # durağan/hareket esigi
ONSET_REST_S = 0.3        # onset oncesi en az bu kadar durağan kalmali
PRESTART_GAP_S = 0.5      # durağanlik bitisi -> kosu basi arasi izinli ara
SWAY_MAX_M_S = 1.0        # bu aradaki salinimin ust hiz siniri


def run_onset(hipx: np.ndarray, end_f: float, dir_in: int, px_per_m: float,
              fps: float, observed: Optional[np.ndarray] = None, *, speed_thr: float = ONSET_SPEED_M_S,
              rest_s: float = ONSET_REST_S) -> tuple[Optional[float], Optional[str]]:
    """Kosu baslangici (onset): end_f'teki (zamanlama cizgisinden 1. gecis)
    kosuya yol acan kesintisiz hareketin basladigi alt-kare.

    end_f'ten GERIYE yurunur:
      1) dir_in yonundeki hiz > speed_thr olan kesintisiz kosu blogu bulunur
         (blogun basi = ilk esik-ustu kare).
      2) Bloktan hemen once en az rest_s saniye |hiz| < speed_thr (durağan)
         olmalidir.
    Pose'un olmadigi (NaN) kareler blogu keser; aradaki boslugu tahmin etmeyiz.

    observed: kare bazinda gercek tespit maskesi. Canlida girdi tarayicidaki
    YOLO'dan gelen CSV'dir; dashboard tespit olmayan kareleri komsu tespitle
    doldurur (basta/sonda sabit kopya, arada dogrusal ara deger) ve bu karelere
    visibility=0 yazar. Bu kareler durağanlik ve kosu bolumune ALINMAZ (sabit
    kopya sahte "durağanlik" uretir). Onset -> 1. gecis arasinda tek bir
    doldurulmus kare bile varsa sonuc None ("baslangic_kadrajda_degil").
    Onset sub-frame: hizin esigi gectigi nokta lineer interpolasyonla.

    Doner: (onset_frame | None, None | neden_kodu)
    """
    n = len(hipx)
    end_i = min(int(np.floor(end_f)), n - 1)
    ok = ~np.isnan(hipx)
    if observed is not None:
        ok &= np.asarray(observed, dtype=bool)
    if end_i < 2 or not ok[:end_i + 1].any():
        return None, "baslangic_kadrajda_degil"

    # Hiz (m/s, dir_in yonunde isaretli). Gradyan icin NaN'lar gecici olarak
    # doldurulur ama NaN kareler asagida hiz=NaN yapilir (tahmin yok).
    idx = np.arange(n)
    filled = np.interp(idx, idx[ok], hipx[ok])
    v = np.gradient(filled) * fps / px_per_m * dir_in
    win = max(3, int(round(0.1 * fps)))
    v = np.convolve(v, np.ones(win) / win, mode="same")
    # Kenar etkisi: gozlenmeyen kare komsulugundaki hiz guvenilmez (gradyan,
    # hareketli ortalama ve hipx'in 5'lik medyani komsu kareyi karistirir).
    h = max(win // 2, 2)
    bad = np.convolve((~ok).astype(float), np.ones(2 * h + 1), mode="same") > 0
    v[bad] = np.nan

    # 1) Kosu blogu: end_i'den geriye v > esik
    i = end_i
    if not (np.isfinite(v[i]) and v[i] > speed_thr):
        return None, "kosu_blogu_bulunamadi"
    while i - 1 >= 0 and np.isfinite(v[i - 1]) and v[i - 1] > speed_thr:
        i -= 1
    run_start = i
    if run_start == 0 or not np.isfinite(v[run_start - 1]):
        # Video basinda / pose ilk goruldugunde zaten kosuyor.
        return None, "baslangic_kadrajda_degil"

    # 2) Bloktan once durağanlik: en az rest_s boyunca |v| < esik. Sporcu
    # cikistan hemen once kucuk bir agirlik aktarimi / salinim yapabilir; bu
    # yuzden durağan bolumun bitisi ile kosu basi arasinda en fazla
    # PRESTART_GAP_S'lik, yalnizca kucuk hizli (|v| < SWAY_MAX) bir ara kabul
    # edilir. Gercek kosu (ters yon dahil) bu araliga sigmaz.
    rest_n = int(np.ceil(rest_s * fps))
    gap_max = int(round(PRESTART_GAP_S * fps))
    k = run_start - 1
    streak = 0
    found = False
    while k >= 0 and np.isfinite(v[k]) and abs(v[k]) < SWAY_MAX_M_S:
        if abs(v[k]) < speed_thr:
            streak += 1
            if streak >= rest_n:
                found = True
                break
        else:
            streak = 0
        # streak'in en gec karesi ile kosu basi arasindaki ara
        gap = (run_start - 1) - (k + streak - 1) if streak else run_start - k
        if gap > gap_max:
            break
        k -= 1
    if not found:
        # Durağan bekleme yok: ters yonde hareket, kadraja kosarak giris vb.
        return None, "baslangic_kadrajda_degil"

    a, b = v[run_start - 1], v[run_start]
    frac = (speed_thr - a) / (b - a) if b != a else 1.0
    onset = float(run_start - 1 + max(0.0, min(1.0, frac)))

    # Onset -> 1. gecis arasi tamamen gercek tespit olmali.
    span = slice(int(np.floor(onset)), min(int(np.ceil(end_f)), n - 1) + 1)
    if not ok[span].all():
        return None, "baslangic_kadrajda_degil"
    return onset, None


def analyze(video_path: str, *, line_x: int, turn_x: int,
            model: str = "heavy", conf: float = 0.2) -> dict:
    for nm, vv in (("line_x", line_x), ("turn_x", turn_x)):
        if vv is None:
            raise RuntimeError(f"{nm} gerekli.")

    track_px = float(abs(turn_x - line_x))
    if track_px < 50:
        raise RuntimeError("line_x ile turn_x cok yakin.")
    px_per_m = track_px / 5.0

    hip_x: list[float] = []
    observed: list[bool] = []     # visibility>0: gercek tespit (yalniz onset)
    fps = 30.0
    for fi, t, l2, l3, W, H, fps in iter_pose(video_path, model=model, conf=conf):
        if l2 is None:
            hip_x.append(np.nan)
            observed.append(False)
            continue
        lh, rh = pt2d(l2, LEFT_HIP, W, H), pt2d(l2, RIGHT_HIP, W, H)
        hip_x.append(float((lh[0] + rh[0]) / 2))
        observed.append(max(l2[LEFT_HIP].visibility,
                            l2[RIGHT_HIP].visibility) > 0)
    hipx = smooth_median(np.array(hip_x), 5)
    valid = np.where(~np.isnan(hipx))[0]
    if valid.size < 10:
        raise RuntimeError("Yeterli pose verisi yok.")

    # Donus yonu piksel ekseninde: sporcu line_x'ten turn_x'e dogru kosar.
    dir_in = 1 if turn_x > line_x else -1
    turn_side = "right" if dir_in > 0 else "left"

    # 1) Donus: dir_in yonundeki GLOBAL extremum. Finish sonrasi asiri kosu
    # ters yonde oldugu icin bu capayi bozamaz.
    seg = hipx[int(valid[0]):int(valid[-1]) + 1]
    if not np.isfinite(seg).any():
        raise RuntimeError("Donus araliginda pose verisi yok.")
    t_off = int(np.nanargmax(seg) if dir_in > 0 else np.nanargmin(seg))
    tt = float(valid[0] + t_off)

    # 2) Start: donusten ONCE line_x'in dir_in yonundeki SON gecisi (ucus
    # baslangicinda cizgi civari pose gurultusu erken sahte gecis uretebilir;
    # son gecis gercek girisi verir).
    sf = None
    idx = int(valid[0])
    while True:
        c = crossing_frame(hipx, float(line_x), dir_in, idx)
        if c is None or c >= tt:
            break
        sf = c
        idx = int(c) + 1
    standing_start = sf is None
    if sf is None:
        # Ayakta-start varyanti: sporcu cizginin ustunde durur, gecis olusmaz.
        onset_thr = max(30.0, px_per_m * 0.3)
        onset = motion_onset(hipx, int(valid[0]), threshold_px=onset_thr)
        if onset is None or onset >= tt:
            raise RuntimeError("Zamanlama cizgisi gecisi bulunamadi.")
        sf = onset
    if not (sf < tt):
        raise RuntimeError("Donus noktasi bulunamadi.")

    # 3) Finish: donusten sonra line_x'i ters yonde gec (ilk gecis; sonraki
    # asiri kosu follow-through, dikkate alinmaz).
    ff = crossing_frame(hipx, float(line_x), -dir_in, int(tt) + 1)
    if ff is None:
        raise RuntimeError("Bitis cizgisi gecisi bulunamadi.")

    t_in = (tt - sf) / fps
    t_out = (ff - tt) / fps
    t_total = (ff - sf) / fps
    if t_total <= 0:
        raise RuntimeError("Gecersiz toplam sure.")

    speeds = {
        "in_5m_m_s": round(5.0 / t_in, 3) if t_in > 0 else None,
        "out_5m_m_s": round(5.0 / t_out, 3) if t_out > 0 else None,
        "avg_m_s": round(10.0 / t_total, 3) if t_total > 0 else None,
    }

    # Toplam (kronometre) sure: kosu baslangici (onset) -> 2. gecis.
    # total_time_s'e dokunulmaz; full = approach + total.
    frames = {
        "start": round(sf, 1), "turn": round(tt, 1),
        "finish": round(ff, 1),
    }
    full_time_s = approach_time_s = None
    full_time_note = None
    if standing_start:
        # Gecis yok, start zaten hareket baslangici: ayri bir hizlanma bolumu yok.
        full_time_note = "zamanlama_cizgisi_gecisi_yok"
    else:
        onset_f, full_time_note = run_onset(hipx, sf, dir_in, px_per_m, fps,
                                           np.array(observed))
        if onset_f is not None and onset_f < sf:
            approach_time_s = round((sf - onset_f) / fps, 3)
            full_time_s = round(approach_time_s + round(t_total, 3), 3)
            frames["onset"] = round(onset_f, 1)
        elif full_time_note is None:
            full_time_note = "baslangic_kadrajda_degil"

    return {
        "fps": round(fps, 2),
        "distance_m": 10.0,
        "turn_side": turn_side,
        "px_per_m": round(px_per_m, 1),
        "frames": frames,
        "full_time_s": full_time_s,
        "approach_time_s": approach_time_s,
        "full_time_note": full_time_note,
        "splits_s": {
            "in_5m": round(t_in, 3),
            "out_5m": round(t_out, 3),
        },
        "total_time_s": round(t_total, 3),
        "speeds": speeds,
    }


def main():
    ap = argparse.ArgumentParser(description="5-0-5 COD testi")
    ap.add_argument("video")
    ap.add_argument("--line-x", type=int, required=True, dest="line_x")
    ap.add_argument("--turn-x", type=int, required=True, dest="turn_x")
    ap.add_argument("--model", default="heavy", choices=["heavy", "lite"])
    ap.add_argument("--conf", type=float, default=0.2)
    args = ap.parse_args()
    r = analyze(args.video, line_x=args.line_x, turn_x=args.turn_x,
                model=args.model, conf=args.conf)
    print(f"\n5-0-5  ->  toplam {r['total_time_s']}s "
          f"(donus tarafi: {r['turn_side']})")
    sp = r["splits_s"]
    print(f"gidis 5m {sp['in_5m']}s | donus 5m {sp['out_5m']}s")
    if r["full_time_s"] is not None:
        print(f"toplam (onset->finish) {r['full_time_s']}s "
              f"(hizlanma {r['approach_time_s']}s)")
    else:
        print(f"toplam sure yok: {r['full_time_note']}")
    print()
    json.dump(r, sys.stdout, indent=2, ensure_ascii=False)
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
