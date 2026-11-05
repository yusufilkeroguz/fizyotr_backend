"""videos/ altindaki videolar icin MediaPipe pose overlay (landmark + baglanti)
videolari uretir. Cikti: videos/overlay/<isim>_overlay.mp4

Calistirma (cv2+mediapipe+model olan yorumlayici ile):
    python tools/pose_overlay.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
from mediapipe import Image, ImageFormat
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import (
    PoseLandmarker, PoseLandmarkerOptions, RunningMode,
)

# MediaPipe Pose (33 nokta) standart baglanti listesi. Bu mediapipe kurulumu
# Tasks-only oldugu icin mediapipe.solutions.pose yok; o yuzden gomulu.
POSE_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5), (5, 6), (6, 8), (9, 10),
    (11, 12), (11, 13), (13, 15), (15, 17), (15, 19), (15, 21), (17, 19),
    (12, 14), (14, 16), (16, 18), (16, 20), (16, 22), (18, 20),
    (11, 23), (12, 24), (23, 24),
    (23, 25), (25, 27), (27, 29), (29, 31), (27, 31),
    (24, 26), (26, 28), (28, 30), (30, 32), (28, 32),
]

# Proje kokunu path'e ekle (model_path icin)
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.pose_common import model_path  # noqa: E402

VID_DIR = ROOT / "videos"
OUT_DIR = VID_DIR / "overlay"
CONF = 0.3
VIS_THR = 0.5


def draw(frame, lms, W, H):
    pts = [(int(lm.x * W), int(lm.y * H), lm.visibility) for lm in lms]
    # baglantilar (yesil)
    for a, b in POSE_CONNECTIONS:
        if pts[a][2] < VIS_THR or pts[b][2] < VIS_THR:
            continue
        cv2.line(frame, pts[a][:2], pts[b][:2], (0, 255, 0), 2, cv2.LINE_AA)
    # noktalar (kirmizi)
    for x, y, v in pts:
        if v < VIS_THR:
            continue
        cv2.circle(frame, (x, y), 4, (0, 0, 255), -1, cv2.LINE_AA)
    return frame


def process(video: Path, opts: PoseLandmarkerOptions) -> None:
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        print(f"  ATLANDI (acilamadi): {video.name}")
        return
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f"{video.stem}_overlay.mp4"
    writer = cv2.VideoWriter(str(out_path),
                             cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
    fi = 0
    drawn = 0
    # Her video icin TAZE landmarker: VIDEO modunda timestamp tum landmarker
    # omru boyunca monotonik artmali; videolar arasi sifirlanirsa hata verir.
    with PoseLandmarker.create_from_options(opts) as lm:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            ts = int(fi * 1000.0 / fps)
            res = lm.detect_for_video(
                Image(image_format=ImageFormat.SRGB, data=rgb), ts)
            if res.pose_landmarks:
                draw(frame, res.pose_landmarks[0], W, H)
                drawn += 1
            writer.write(frame)
            fi += 1
    cap.release()
    writer.release()
    print(f"  -> {out_path.name}  ({fi} kare, {drawn} poz cizildi)")


def main() -> None:
    vids = sorted([p for p in VID_DIR.iterdir()
                   if p.suffix.lower() in (".mp4", ".mov") and p.is_file()])
    if not vids:
        print("videos/ altinda video yok.")
        return
    opts = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_path("heavy")),
        running_mode=RunningMode.VIDEO, num_poses=1,
        min_pose_detection_confidence=CONF,
        min_pose_presence_confidence=CONF,
        min_tracking_confidence=CONF,
    )
    print(f"{len(vids)} video isleniyor -> {OUT_DIR}")
    for v in vids:
        print(f"[{v.name}]")
        process(v, opts)
    print("Bitti.")


if __name__ == "__main__":
    main()
