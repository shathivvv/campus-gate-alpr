"""Offline accuracy check: runs the ALPR pipeline over a recorded video, frame by frame.

Unlike app.py it doesn't play in real time; it processes SAMPLE_FPS frames per second of
video, groups detections into one event per passing vehicle, and saves for each event the
clearest plate crop plus what the pipeline read. Compare those against the true plates.

    python scan_video.py videos/gate.mp4 --sample-fps 10
"""
import argparse
import csv
import time
from collections import Counter
from pathlib import Path

import cv2
from ultralytics import YOLO

import config
from plate_reader import create_ocr, crop_plate, read_plate
from voter import PlateVoter

EVENT_GAP = 1.0  # seconds without a detection that end a vehicle event


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
    ocr = create_ocr()

    detections = []  # one row per sampled frame with a plate
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
            best = int(boxes.conf.argmax())
            box = boxes.xyxy[best].tolist()
            plate_crop = crop_plate(frame, box)
            reading = read_plate(ocr, plate_crop) if plate_crop is not None else None
            detections.append({
                "time": frame_no / fps,
                "det_conf": float(boxes.conf[best]),
                "box": box,
                "width": box[2] - box[0],
                "text": reading[0] if reading else "",
                "ocr_conf": reading[1] if reading else 0.0,
                "crop": crop_plate(frame, box, pad=0.25),  # wider view, for checking by eye
            })

        if frame_no % (step * 100) == 0:
            print(f"{frame_no / total:6.1%}  {frame_no / fps:6.1f}s  detections so far: {len(detections)}  "
                  f"elapsed {time.time() - started:.0f}s", flush=True)

    events = group_events(detections)
    write_report(out, events)
    print(f"\nDone in {time.time() - started:.0f}s: {len(events)} vehicle events -> {out}")


def group_events(detections):
    events, current = [], []
    for d in detections:
        if current and d["time"] - current[-1]["time"] > EVENT_GAP:
            events.append(current)
            current = []
        current.append(d)
    if current:
        events.append(current)
    return events


def write_report(out, events):
    with open(out / "events.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["event", "start_s", "end_s", "frames_detected", "frames_read",
                         "confirmed", "most_common_read", "all_reads", "crop", "true_plate"])
        for i, event in enumerate(events, 1):
            # Same rule the live app uses: 3 matching reads within the vote window.
            voter = PlateVoter(config.MIN_VOTES, config.VOTE_WINDOW, config.LOG_COOLDOWN)
            confirmed = [c[0] for d in event if d["text"] and (c := voter.add(d["text"], d["ocr_conf"], d["time"]))]
            reads = Counter(d["text"] for d in event if d["text"])

            # The biggest detected plate is usually the closest, sharpest view.
            best = max(event, key=lambda d: d["width"])
            crop_name = f"event_{i:03d}.jpg"
            if best["crop"] is not None:
                cv2.imwrite(str(out / "crops" / crop_name), best["crop"])

            writer.writerow([
                i, f"{event[0]['time']:.1f}", f"{event[-1]['time']:.1f}", len(event), sum(reads.values()),
                " ".join(confirmed), reads.most_common(1)[0][0] if reads else "",
                " ".join(f"{t}x{n}" for t, n in reads.most_common()), crop_name, "",
            ])


if __name__ == "__main__":
    main()
