"""
Video'daki zıplamanın yüksekliğini hesaplar.

İki yöntem kullanır:
  1) Piksel yöntemi: MediaPipe Pose ile KALÇA merkezi (hip center) takip
     edilir — kütle merkezine yakın olduğu için kişi bacaklarını karnına
     çekse bile gerçek yükselişi ölçer. Kişinin boyu verilirse metreye çevrilir.
  2) Fizik yöntemi: Ayak bileği ile kalkış/iniş anları bulunur, uçuş süresi
     t ölçülür, h = g * t^2 / 8 ile yükseklik hesaplanır. Kalibrasyon gerektirmez.

Kullanım:
    python jump_height.py IMG_0279.MOV --height 1.80
    python jump_height.py IMG_0280.MOV
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass

import numpy as np

try:
    from ..pose_common import iter_pose
except ImportError:  # standalone: python jump_height.py
    import os as _os
    sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
    from pose_common import iter_pose

# PoseLandmark indeksleri (MediaPipe BlazePose, 33 nokta)
NOSE = 0
LEFT_HIP, RIGHT_HIP = 23, 24
LEFT_ANKLE, RIGHT_ANKLE = 27, 28


@dataclass
class FrameSample:
    t: float            # saniye
    hip_y: float        # kalça merkezi y (piksel, aşağı = büyük) — yükseklik ölçümü
    ankle_y: float      # ayak bileği y (piksel) — kalkış/iniş tespiti
    body_px: float      # burun-ayak bileği piksel mesafesi (kalibrasyon için)
    visible: bool


def extract_samples(video_path: str) -> tuple[list[FrameSample], float]:
    samples: list[FrameSample] = []
    fps = 30.0
    # conf=0.5: MediaPipe varsayilan esikleriyle ayni (onceki davranisi korur).
    for fi, t, lm, l3, w, h, fps in iter_pose(video_path, conf=0.5):
        if lm is not None:
            la, ra = lm[LEFT_ANKLE], lm[RIGHT_ANKLE]
            lh, rh = lm[LEFT_HIP], lm[RIGHT_HIP]
            nose = lm[NOSE]

            # Yan görünüşte arka taraf landmark'larının visibility'si düşük olur.
            # Her segmentten en az bir taraf görünürse yeterli sayalım.
            ankle_ok = max(la.visibility, ra.visibility) > 0.5
            hip_ok = max(lh.visibility, rh.visibility) > 0.5
            vis = ankle_ok and hip_ok

            # Görünür olan tarafları tercih ederek ortala.
            ankles = [p for p in (la, ra) if p.visibility > 0.5] or [la, ra]
            hips = [p for p in (lh, rh) if p.visibility > 0.5] or [lh, rh]
            hip_y = np.mean([p.y for p in hips]) * h
            ankle_y = np.mean([p.y for p in ankles]) * h
            foot_y_abs = max(p.y for p in ankles) * h
            body_px = abs(foot_y_abs - nose.y * h)

            samples.append(FrameSample(t, hip_y, ankle_y, body_px, vis))
        else:
            samples.append(FrameSample(t, np.nan, np.nan, np.nan, False))

    return samples, fps


def compute_pixel_height(samples: list[FrameSample], user_height_m: float | None):
    hips = np.array([s.hip_y if s.visible else np.nan for s in samples], dtype=float)
    ankles = np.array([s.ankle_y if s.visible else np.nan for s in samples], dtype=float)
    valid = ~np.isnan(hips) & ~np.isnan(ankles)
    if valid.sum() < 5:
        return None

    # Dik duruş referansı: önce ayaklar yerde olan kareleri seç (ankle_y büyük),
    # sonra bunların içinden bacakların en açık olduğu kareleri al. Çömelme
    # (kısa bacak) elenir, kalan kareler "ayakta dik duruyor" demektir.
    leg_len = ankles - hips
    ankle_ground_thr = np.nanpercentile(ankles[valid], 70)
    ground_mask = valid & (ankles >= ankle_ground_thr)
    if ground_mask.sum() < 3:
        return None
    leg_thr = np.nanpercentile(leg_len[ground_mask], 60)
    standing_mask = ground_mask & (leg_len >= leg_thr)
    if standing_mask.sum() < 2:
        standing_mask = ground_mask

    standing_hip_y = np.median(hips[standing_mask])
    # Zirve: kalçanın en yukarı çıktığı an (gürültüye karşı 1. persentil).
    peak_hip_y = np.nanpercentile(hips[valid], 1)
    jump_px = standing_hip_y - peak_hip_y
    if jump_px < 0:
        jump_px = 0

    if user_height_m is None:
        return {"jump_px": jump_px, "jump_m": None}

    # Kalibrasyon: ayakta dururken (yerde, düşük hareket) burun-ayak piksel mesafesi.
    # Nose-ankle ≈ boyun %87'si (CDC antropometri ortalaması).
    body_px_values = np.array(
        [s.body_px for s in samples if s.visible and not np.isnan(s.body_px)]
    )
    if body_px_values.size == 0:
        return {"jump_px": jump_px, "jump_m": None}

    nose_ankle_px = np.median(body_px_values)
    px_per_m = nose_ankle_px / (user_height_m * 0.87)
    return {"jump_px": jump_px, "jump_m": jump_px / px_per_m}


def compute_physics_height(samples: list[FrameSample], fps: float):
    ys = np.array([s.ankle_y if s.visible else np.nan for s in samples], dtype=float)
    valid_idx = np.where(~np.isnan(ys))[0]
    if len(valid_idx) < 10:
        return None

    ground_y = np.nanpercentile(ys[valid_idx], 90)
    peak_y = np.nanpercentile(ys[valid_idx], 2)
    amplitude = ground_y - peak_y
    if amplitude < 10:
        return None

    # En büyük zıplama bloğunu kaba eşikle bul (tepe noktasının %30'u),
    # itme/iniş hazırlık fazını ekleme.
    coarse_thr = ground_y - amplitude * 0.3
    airborne = (ys < coarse_thr)
    if not airborne.any():
        return None
    idx = np.where(airborne)[0]
    splits = np.split(idx, np.where(np.diff(idx) > 1)[0] + 1)
    longest = max(splits, key=len)
    if len(longest) < 2:
        return None

    # Doğrusal interpolasyonla kalkış/iniş anlarını yer seviyesi yakınında bul.
    # Bloğun BİR adım dışındaki kare ile içindeki kare arasında kesişim ara.
    crossing_y = ground_y - amplitude * 0.1

    def interp(inside: int, outside: int) -> float:
        if not (0 <= outside < len(ys)) or np.isnan(ys[outside]) or np.isnan(ys[inside]):
            return float(inside)
        y_in, y_out = ys[inside], ys[outside]
        if y_in == y_out:
            return float(inside)
        frac = (crossing_y - y_in) / (y_out - y_in)
        frac = max(0.0, min(1.0, frac))
        return inside + frac * (outside - inside)

    takeoff = interp(longest[0], longest[0] - 1)
    landing = interp(longest[-1], longest[-1] + 1)
    flight_time = (landing - takeoff) / fps
    if flight_time <= 0:
        return None

    g = 9.81
    jump_m = g * flight_time * flight_time / 8.0
    return {
        "flight_time": flight_time,
        "jump_m": jump_m,
        "start_t": takeoff / fps,
        "amplitude_px": amplitude,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--height", type=float, default=None,
                    help="Kişinin boyu (metre), piksel yöntemi kalibrasyonu için")
    args = ap.parse_args()

    print(f"İşleniyor: {args.video}")
    samples, fps = extract_samples(args.video)
    print(f"  FPS: {fps:.2f}, kare sayısı: {len(samples)}")

    px = compute_pixel_height(samples, args.height)
    phys = compute_physics_height(samples, fps)

    print("\n--- Sonuçlar ---")
    if px:
        print(f"Piksel yöntemi: {px['jump_px']:.1f} px", end="")
        if px["jump_m"] is not None:
            print(f"  ≈  {px['jump_m'] * 100:.1f} cm")
        else:
            print("  (metre için --height verin)")
    else:
        print("Piksel yöntemi: yeterli veri yok")

    if phys:
        print(f"Fizik yöntemi:  uçuş süresi {phys['flight_time']*1000:.0f} ms"
              f"  →  {phys['jump_m'] * 100:.1f} cm"
              f"  (başlangıç t={phys['start_t']:.2f}s, ayak amp={phys['amplitude_px']:.0f}px)")
    else:
        print("Fizik yöntemi: zıplama tespit edilemedi")


if __name__ == "__main__":
    main()
