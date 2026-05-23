"""FMS analizcileri icin ortak yardimcilar."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterator, Optional

import numpy as np

# cv2 / mediapipe sadece VIDEO yolu (iter_pose) ve render icin gerekir; CSV-only
# sunucuda kurulu olmasalar da modul import edilebilsin diye lazy/optional tutulur.
try:
    import cv2  # opsiyonel: yalnizca video/render icin
except ImportError:
    cv2 = None

# Pose landmark indisleri
NOSE = 0
LEFT_SHOULDER, RIGHT_SHOULDER = 11, 12
LEFT_ELBOW, RIGHT_ELBOW = 13, 14
LEFT_WRIST, RIGHT_WRIST = 15, 16
LEFT_HIP, RIGHT_HIP = 23, 24
LEFT_KNEE, RIGHT_KNEE = 25, 26
LEFT_ANKLE, RIGHT_ANKLE = 27, 28
LEFT_HEEL, RIGHT_HEEL = 29, 30
LEFT_FOOT, RIGHT_FOOT = 31, 32


def model_path(model: str = "heavy") -> str:
    name = f"pose_landmarker_{model}.task"
    cands = [Path.cwd() / name, Path.cwd() / "models" / name]
    for p in list(Path(__file__).resolve().parents)[:5]:
        cands += [p / name, p / "models" / name]
    for cand in cands:
        if cand.exists():
            return str(cand)
    return name


@dataclass
class FrameLM:
    """Bir karedeki 2D pose + 3D world + meta bilgiler."""
    frame: int
    time_s: float
    lm2d: list  # NormalizedLandmark, 0..1 normalize
    lm3d: list  # WorldLandmark, metre (kalca merkezli)
    W: int
    H: int


def iter_pose(video_path: str, *, model: str = "heavy",
              conf: float = 0.3) -> Iterator[tuple[int, float,
                                                   Optional[list],
                                                   Optional[list], int, int, float]]:
    """Videoyu kare kare cozer, her karede (frame, t, lm2d, lm3d, W, H, fps).

    video_path bir .csv ise tarayicida cikarilmis pose kullanilir."""
    if str(video_path).lower().endswith(".csv"):
        from ..pose_common import iter_pose_csv
        yield from iter_pose_csv(video_path)
        return
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
    cap.release()


def pt2d(lm, i, W, H) -> np.ndarray:
    p = lm[i]
    return np.array([p.x * W, p.y * H], dtype=np.float32)


def ptw(lm3d, i) -> np.ndarray:
    p = lm3d[i]
    return np.array([p.x, p.y, p.z], dtype=np.float32)


def angle_deg(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """a-b-c noktalariyla olusan B tepesindeki ici aci (derece)."""
    v1, v2 = a - b, c - b
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 < 1e-6 or n2 < 1e-6:
        return float("nan")
    cosv = float(np.dot(v1, v2) / (n1 * n2))
    return float(np.degrees(np.arccos(np.clip(cosv, -1.0, 1.0))))


def line_angle_deg(a: np.ndarray, b: np.ndarray) -> float:
    """a->b cizgisinin YATAY ile yaptigi aci (derece, 0..180). Dikey=90."""
    d = b - a
    return float(np.degrees(np.arctan2(-d[1], d[0]))) % 180.0


def pick_visible_side(lm, left_idx: int, right_idx: int,
                      thr: float = 0.5) -> str:
    """Tek-yan gorusteki testler icin hangi yan gorunur?"""
    lv = lm[left_idx].visibility
    rv = lm[right_idx].visibility
    if lv > thr and rv <= thr:
        return "left"
    if rv > thr and lv <= thr:
        return "right"
    # Her ikisi gorunurse daha gorunene oy ver
    return "left" if lv >= rv else "right"


def smooth_median(arr: np.ndarray, win: int = 5) -> np.ndarray:
    """NaN-dayanikli hareketli median."""
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


def px_per_m_from_body(lm2d, W, H, height_m: float) -> float:
    """Nose-to-ankle piksel ile boy orani: px_per_m = px / (height * 0.87)."""
    nose = pt2d(lm2d, NOSE, W, H)
    la = pt2d(lm2d, LEFT_ANKLE, W, H)
    ra = pt2d(lm2d, RIGHT_ANKLE, W, H)
    ank_y = max(la[1], ra[1])
    return abs(ank_y - nose[1]) / (height_m * 0.87)
