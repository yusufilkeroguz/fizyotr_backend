"""Performans testi analizcileri icin ortak pose yardimcilari.

Bu modul, kok dizindeki yeni test analizcilerinin (less, bosco_60, rsi_10_5,
flamingo, functional_reach, ten_m_walk, rom) paylastigi pose-cikarim ve
geometri yardimcilarini barindirir. fms/common.py ile ayni ruhta ama fms
paketinden bagimsizdir (ileride Flask/Cloud Run servisi tek yerden import
edebilsin diye).

Tipik kullanim:
    from pose_common import iter_pose, angle_deg, smooth_median
    for fi, t, lm2d, lm3d, W, H, fps in iter_pose(video):
        ...
"""
from __future__ import annotations

import csv as _csv
from collections import namedtuple
from pathlib import Path
from typing import Iterator, Optional

import numpy as np

# Agir bagimliliklar (cv2 / mediapipe) sadece pose cikarirken gerekir; modul
# import edildiginde patlamamasi icin iter_pose icinde lazy import edilir.

# ── MediaPipe BlazePose 33-nokta landmark indisleri ──────────────────────
NOSE = 0
LEFT_SHOULDER, RIGHT_SHOULDER = 11, 12
LEFT_ELBOW, RIGHT_ELBOW = 13, 14
LEFT_WRIST, RIGHT_WRIST = 15, 16
LEFT_HIP, RIGHT_HIP = 23, 24
LEFT_KNEE, RIGHT_KNEE = 25, 26
LEFT_ANKLE, RIGHT_ANKLE = 27, 28
LEFT_HEEL, RIGHT_HEEL = 29, 30
LEFT_FOOT, RIGHT_FOOT = 31, 32  # foot_index

SKELETON_CONNECTIONS = [
    (LEFT_SHOULDER, RIGHT_SHOULDER),
    (LEFT_SHOULDER, LEFT_ELBOW), (LEFT_ELBOW, LEFT_WRIST),
    (RIGHT_SHOULDER, RIGHT_ELBOW), (RIGHT_ELBOW, RIGHT_WRIST),
    (LEFT_SHOULDER, LEFT_HIP), (RIGHT_SHOULDER, RIGHT_HIP),
    (LEFT_HIP, RIGHT_HIP),
    (LEFT_HIP, LEFT_KNEE), (LEFT_KNEE, LEFT_ANKLE),
    (RIGHT_HIP, RIGHT_KNEE), (RIGHT_KNEE, RIGHT_ANKLE),
    (LEFT_ANKLE, LEFT_HEEL), (LEFT_HEEL, LEFT_FOOT), (LEFT_FOOT, LEFT_ANKLE),
    (RIGHT_ANKLE, RIGHT_HEEL), (RIGHT_HEEL, RIGHT_FOOT), (RIGHT_FOOT, RIGHT_ANKLE),
]

# Antropometri: burun-ayak bilegi mesafesi ~ boy * 0.87 (CDC ortalamasi).
NOSE_ANKLE_RATIO = 0.87


def model_path(model: str = "heavy") -> str:
    """pose_landmarker_{model}.task dosyasini bul.

    Su sirayla arar: cwd, cwd/models, ve __file__'in ust dizinleri ile
    onlarin models/ alt klasorleri. Boylece dosya src/ icinde derinde olsa da
    repo kokundeki models/ klasorunu bulur."""
    name = f"pose_landmarker_{model}.task"
    cands = [Path.cwd() / name, Path.cwd() / "models" / name]
    for p in list(Path(__file__).resolve().parents)[:5]:
        cands.append(p / name)
        cands.append(p / "models" / name)
    for c in cands:
        if c.exists():
            return str(c)
    return name


def iter_pose(video_path: str, *, model: str = "heavy",
              conf: float = 0.3) -> Iterator[tuple[int, float,
                                                   Optional[list],
                                                   Optional[list], int, int, float]]:
    """Videoyu kare kare cozer: her karede
    (frame_idx, time_s, lm2d, lm3d, W, H, fps).

    lm2d  : NormalizedLandmark listesi (x,y 0..1, visibility) veya None
    lm3d  : WorldLandmark listesi (metre, kalca merkezli) veya None

    video_path bir .csv ise (tarayicida MediaPipe.js ile cikarilmis pose),
    cv2/mediapipe yerine iter_pose_csv kullanilir; ayni demet uretilir.
    """
    if str(video_path).lower().endswith(".csv"):
        yield from iter_pose_csv(video_path)
        return

    import cv2
    from mediapipe import Image, ImageFormat
    from mediapipe.tasks.python import BaseOptions
    from mediapipe.tasks.python.vision import (
        PoseLandmarker, PoseLandmarkerOptions, RunningMode,
    )

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Video acilamadi: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    opts = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_path(model)),
        running_mode=RunningMode.VIDEO, num_poses=1,
        min_pose_detection_confidence=conf,
        min_pose_presence_confidence=conf,
        min_tracking_confidence=conf,
    )
    try:
        with PoseLandmarker.create_from_options(opts) as lm:
            fi = 0
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                r = lm.detect_for_video(
                    Image(image_format=ImageFormat.SRGB, data=rgb),
                    int(fi * 1000 / fps))
                l2 = r.pose_landmarks[0] if r.pose_landmarks else None
                l3 = r.pose_world_landmarks[0] if r.pose_world_landmarks else None
                yield fi, fi / fps, l2, l3, W, H, fps
                fi += 1
    finally:
        cap.release()


# ── CSV pose girisi (tarayici MediaPipe.js ciktisi) ──────────────────────
# pt2d/.x, ptw/.x.y.z, pick_visible_side/.visibility ile uyumlu hafif landmark.
_LM = namedtuple("_LM", ["x", "y", "z", "visibility"])


def iter_pose_csv(csv_path: str) -> Iterator[tuple[int, float,
                                                   Optional[list],
                                                   Optional[list], int, int, float]]:
    """Pose CSV'sini iter_pose ile ayni demet olarak uretir:
    (frame, time_s, lm2d, lm3d, W, H, fps).

    Beklenen sutunlar (header satiri zorunlu):
        frame, time_s, width, height, fps,
        x0,y0,z0,v0 ... x32,y32,z32,v32     (normalize 2D; x,y 0..1, v=visibility)
        wx0,wy0,wz0 ... wx32,wy32,wz32       (world 3D, metre; opsiyonel)
    Pose bulunmayan karede landmark hucreleri bos birakilir -> lm2d=lm3d=None.
    """
    n_pts = 33
    with open(csv_path, newline="") as fh:
        reader = _csv.reader(fh)
        header = next(reader, None)
        if not header:
            return
        col = {name.strip(): i for i, name in enumerate(header)}
        has_world = "wx0" in col

        def cell(row, name):
            j = col.get(name)
            if j is None or j >= len(row):
                return ""
            return row[j].strip()

        W, H, fps = 0, 0, 30.0
        for k, row in enumerate(reader):
            if not row:
                continue

            def fnum(name, default):
                v = cell(row, name)
                if v == "":
                    return default
                try:
                    return float(v)
                except ValueError:
                    return default

            W = int(fnum("width", W))
            H = int(fnum("height", H))
            fps = fnum("fps", fps) or 30.0
            frame_idx = int(fnum("frame", k))
            t = fnum("time_s", frame_idx / fps)

            if cell(row, "x0") == "":          # bu karede pose yok
                yield frame_idx, t, None, None, W, H, fps
                continue

            l2: list = []
            l3: Optional[list] = [] if has_world else None
            ok = True
            for i in range(n_pts):
                xs, ys = cell(row, f"x{i}"), cell(row, f"y{i}")
                if xs == "" or ys == "":
                    ok = False
                    break
                zs, vs = cell(row, f"z{i}"), cell(row, f"v{i}")
                vis = float(vs) if vs != "" else 1.0
                l2.append(_LM(float(xs), float(ys),
                              float(zs) if zs != "" else 0.0, vis))
                if has_world:
                    wx, wy, wz = (cell(row, f"wx{i}"), cell(row, f"wy{i}"),
                                  cell(row, f"wz{i}"))
                    l3.append(_LM(float(wx) if wx != "" else 0.0,
                                  float(wy) if wy != "" else 0.0,
                                  float(wz) if wz != "" else 0.0, vis))
            if ok:
                yield frame_idx, t, l2, l3, W, H, fps
            else:
                yield frame_idx, t, None, None, W, H, fps


# ── Geometri ─────────────────────────────────────────────────────────────
def pt2d(lm, i: int, W: int, H: int) -> np.ndarray:
    p = lm[i]
    return np.array([p.x * W, p.y * H], dtype=np.float32)


def ptw(lm3d, i: int) -> np.ndarray:
    p = lm3d[i]
    return np.array([p.x, p.y, p.z], dtype=np.float32)


def angle_deg(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """a-b-c noktalariyla olusan B tepesindeki ici aci (derece)."""
    v1, v2 = a - b, c - b
    n1, n2 = float(np.linalg.norm(v1)), float(np.linalg.norm(v2))
    if n1 < 1e-6 or n2 < 1e-6:
        return float("nan")
    cosv = float(np.dot(v1, v2) / (n1 * n2))
    return float(np.degrees(np.arccos(np.clip(cosv, -1.0, 1.0))))


def line_angle_deg(a: np.ndarray, b: np.ndarray) -> float:
    """a->b cizgisinin YATAY ile yaptigi aci (0..180). Dikey = 90."""
    d = b - a
    return float(np.degrees(np.arctan2(-d[1], d[0]))) % 180.0


def vertical_tilt_deg(a: np.ndarray, b: np.ndarray) -> float:
    """a->b cizgisinin DIKEY eksenden sapmasi (0 = tam dik, 90 = yatay).
    Govde/uyluk fleksiyonu gibi 'ne kadar one egik' olculeri icin."""
    d = b - a
    return float(abs(np.degrees(np.arctan2(d[0], -d[1]))))


def pick_visible_side(lm, left_idx: int, right_idx: int,
                      thr: float = 0.5) -> str:
    """Tek-yan goruste hangi taraf daha gorunur (left/right)."""
    lv = lm[left_idx].visibility
    rv = lm[right_idx].visibility
    if lv > thr and rv <= thr:
        return "left"
    if rv > thr and lv <= thr:
        return "right"
    return "left" if lv >= rv else "right"


def smooth_median(arr: np.ndarray, win: int = 5) -> np.ndarray:
    """NaN-dayanikli hareketli median."""
    arr = np.asarray(arr, dtype=float)
    n = len(arr)
    out = np.full(n, np.nan)
    half = win // 2
    for i in range(n):
        lo, hi = max(0, i - half), min(n, i + half + 1)
        seg = arr[lo:hi]
        seg = seg[~np.isnan(seg)]
        if seg.size:
            out[i] = float(np.median(seg))
    return out


def body_px_per_m(nose_ankle_px: np.ndarray, height_m: float,
                  pct: float = 95.0) -> Optional[float]:
    """Burun-ayak bilegi piksel dizisinden px/m kalibrasyonu.
    nose_ankle_px: her karede |ankle_y - nose_y| (piksel)."""
    if height_m is None:
        return None
    arr = np.asarray(nose_ankle_px, dtype=float)
    arr = arr[~np.isnan(arr)]
    if arr.size == 0:
        return None
    return float(np.percentile(arr, pct)) / (height_m * NOSE_ANKLE_RATIO)


# ── Sicrama / ucus tespiti (cok-sicramali testler icin) ──────────────────
def _interp_cross(arr: np.ndarray, k_in: int, k_out: int,
                  cross_y: float) -> float:
    """k_in (blok ici) ile k_out (blok disi) kareleri arasinda cross_y'nin
    asildigi alt-kare konumunu lineer interpolasyonla bul."""
    n = len(arr)
    if not (0 <= k_out < n) or np.isnan(arr[k_out]) or np.isnan(arr[k_in]):
        return float(k_in)
    y_in, y_out = arr[k_in], arr[k_out]
    if y_in == y_out:
        return float(k_in)
    frac = (cross_y - y_in) / (y_out - y_in)
    frac = max(0.0, min(1.0, frac))
    return k_in + frac * (k_out - k_in)


def detect_flight_blocks(foot_ground_y, fps: float, *,
                         ground_pct: float = 90.0, peak_pct: float = 5.0,
                         thr_ratio: float = 0.25, min_air_frames: int = 2,
                         min_amplitude_px: float = 15.0,
                         max_flight_s: Optional[float] = None) -> tuple[list[dict], dict]:
    """Ayak (en alt nokta) y dizisinden ardisik ucus (havada) bloklarini bul.

    foot_ground_y: her karede max(L/R foot/heel y) -> en alt ayak noktasi
                   (yere yakin = buyuk y). Havadayken bu deger yukari (kucuk).

    Donen blok: {
        takeoff, landing            : tamsayi kare indisleri (havadaki ilk/son)
        takeoff_sub, landing_sub    : alt-kare gecis konumlari
        flight_time_s, peak_frame
    }
    Ikinci donus: {ground_y, peak_y, amplitude, threshold_y} meta bilgisi.
    """
    arr = np.asarray(foot_ground_y, dtype=float)
    n = len(arr)
    finite = arr[~np.isnan(arr)]
    meta = {"ground_y": float("nan"), "peak_y": float("nan"),
            "amplitude": 0.0, "threshold_y": float("nan")}
    if finite.size < 5:
        return [], meta
    ground_y = float(np.nanpercentile(arr, ground_pct))
    peak_y = float(np.nanpercentile(arr, peak_pct))
    amp = ground_y - peak_y
    meta.update(ground_y=ground_y, peak_y=peak_y, amplitude=amp)
    if amp < min_amplitude_px:
        return [], meta
    thr_y = ground_y - amp * thr_ratio
    fine_y = ground_y - amp * 0.1
    meta["threshold_y"] = thr_y

    airborne = np.zeros(n, dtype=bool)
    valid = ~np.isnan(arr)
    airborne[valid] = arr[valid] < thr_y
    idx = np.where(airborne)[0]
    if idx.size == 0:
        return [], meta
    splits = np.split(idx, np.where(np.diff(idx) > 1)[0] + 1)

    blocks: list[dict] = []
    for s in splits:
        if len(s) < min_air_frames:
            continue
        t0, t1 = int(s[0]), int(s[-1])
        to_sub = _interp_cross(arr, t0, t0 - 1, fine_y)
        ld_sub = _interp_cross(arr, t1, t1 + 1, fine_y)
        ft = (ld_sub - to_sub) / fps
        if ft <= 0:
            continue
        # Insan sicramasinda ucus suresi fiziksel olarak ~1 sn'i gecmez;
        # daha uzun bloklar pose-kaybi/artefakttir -> ele.
        if max_flight_s is not None and ft > max_flight_s:
            continue
        seg = arr[t0:t1 + 1]
        peak = t0 + (int(np.nanargmin(seg)) if np.isfinite(seg).any() else 0)
        blocks.append({
            "takeoff": t0, "landing": t1,
            "takeoff_sub": round(to_sub, 3), "landing_sub": round(ld_sub, 3),
            "flight_time_s": round(ft, 4), "peak_frame": peak,
        })
    return blocks, meta


def crossing_frame(xs, target_x: float, direction: int,
                   start_idx: int = 0) -> Optional[float]:
    """xs[start_idx:] icinde target_x'i ilk asan alt-kare konumu (float).
    direction: +1 saga, -1 sola hareket."""
    xs = np.asarray(xs, dtype=float)
    n = len(xs)
    prev = np.nan
    for i in range(start_idx, n):
        x = xs[i]
        if np.isnan(x):
            continue
        if not np.isnan(prev):
            crossed = ((direction > 0 and prev < target_x <= x)
                       or (direction < 0 and prev > target_x >= x))
            if crossed:
                t = (target_x - prev) / (x - prev)
                return (i - 1) + t
        prev = x
    return None


GRAVITY = 9.81


def flight_time_to_height_cm(flight_time_s: float) -> float:
    """Ucus suresinden sicrama yuksekligi (cm): h = g * t^2 / 8."""
    return GRAVITY * flight_time_s * flight_time_s / 8.0 * 100.0
