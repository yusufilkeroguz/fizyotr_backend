"""videos/ altindaki videolar icin YOLO person-detection overlay videolari uretir.
Sadece 'person' sinifi (class 0) kutulanir. Cikti: videos/yolo/<isim>_yolo.mp4

Calistirma (ultralytics + torch olan yorumlayici ile):
    python tools/yolo_overlay.py
"""
from __future__ import annotations

from pathlib import Path

import cv2
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
VID_DIR = ROOT / "videos"
OUT_DIR = VID_DIR / "yolo"
WEIGHTS = "/Users/alimusa/Desktop/vitpose/yolov8n.pt"
CONF = 0.3
PERSON = 0  # COCO 'person' sinif id


def process(video: Path, model: YOLO) -> None:
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        print(f"  ATLANDI (acilamadi): {video.name}")
        return
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f"{video.stem}_yolo.mp4"
    writer = cv2.VideoWriter(str(out_path),
                             cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
    fi = 0
    detected = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        res = model.predict(frame, classes=[PERSON], conf=CONF, verbose=False)
        r = res[0]
        if r.boxes is not None and len(r.boxes) > 0:
            detected += 1
        writer.write(r.plot())   # person kutulari cizili BGR kare
        fi += 1
    cap.release()
    writer.release()
    print(f"  -> {out_path.name}  ({fi} kare, {detected} karede person)")


def main() -> None:
    vids = sorted([p for p in VID_DIR.iterdir()
                   if p.suffix.lower() in (".mp4", ".mov") and p.is_file()])
    if not vids:
        print("videos/ altinda video yok.")
        return
    model = YOLO(WEIGHTS)
    print(f"{len(vids)} video isleniyor (YOLO person) -> {OUT_DIR}")
    for v in vids:
        print(f"[{v.name}]")
        process(v, model)
    print("Bitti.")


if __name__ == "__main__":
    main()
