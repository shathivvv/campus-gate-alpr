"""Step 1 of the accuracy check: run plate detection over a recorded video, once.

Saves every detected plate crop (exactly what OCR would receive) plus a detections.csv.
Detection on long 4K footage is slow, so this runs once; evaluate.py then re-reads the
saved crops in a few minutes every time the OCR or voting logic changes.

    python scan_video.py videos/gate.mp4 --sample-fps 10
    python evaluate.py runs/scan_gate --truth eval/ground_truth_gate.csv
"""
import argparse
import csv
import time
from pathlib import Path

import cv2
from ultralytics import YOLO

import config
from plate_reader import crop_plate


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("video")
    parser.add_argument("--sample-fps", type=float, default=10.0)
    parser.add_argument("--out", help="output folder (default runs/scan_<video name>)")
    args = parser.parse_args()

    out = Path(args.out or f"runs/scan_{Path(args.video).stem}")
    (out / "crops").mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(args.video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = max(1, round(fps / args.sample_fps))

    yolo_model = YOLO(config.MODEL_PATH)

    rows = []
    started = time.time()
    for frame_no in range(total):
        if frame_no % step:
            if not cap.grab():
                break
            continue
        ok, frame = cap.read()
        if not ok:
            break

        boxes = yolo_model.predict(frame, conf=config.DET_CONF, imgsz=config.DET_IMGSZ, verbose=False)[0].boxes
        if len(boxes):
            # Same choice as the live app: only the most confident plate in the frame.
            best = int(boxes.conf.argmax())
            box = boxes.xyxy[best].tolist()
            plate_crop = crop_plate(frame, box)
            if plate_crop is not None:
                crop_name = f"{frame_no:06d}.png"
                cv2.imwrite(str(out / "crops" / crop_name), plate_crop)
                rows.append([frame_no, f"{frame_no / fps:.3f}", f"{float(boxes.conf[best]):.3f}",
                             *(round(v) for v in box), crop_name])

        if frame_no % (step * 100) == 0:
            print(f"{frame_no / total:6.1%}  {frame_no / fps:6.1f}s  detections: {len(rows)}  "
                  f"elapsed {time.time() - started:.0f}s", flush=True)

    with open(out / "detections.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["frame", "time_s", "det_conf", "x1", "y1", "x2", "y2", "crop"])
        writer.writerows(rows)
    print(f"\nDone in {time.time() - started:.0f}s: {len(rows)} plate detections -> {out}")


if __name__ == "__main__":
    main()
