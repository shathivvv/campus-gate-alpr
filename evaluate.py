"""Step 2 of the accuracy check: score the pipeline against plates checked by eye.

Re-reads the plate crops saved by scan_video.py, runs them through the same OCR and
voting the live app uses (in video time), and compares the logged plates with an
answer key CSV (columns: time_s, true_plate, ...). One row per vehicle in the answer key.

    python evaluate.py runs/scan_gate --truth eval/ground_truth_gate.csv
"""
import argparse
import csv
from pathlib import Path

import cv2

import config
from plate_reader import create_ocr, read_plate
from voter import PlateVoter

MATCH_WINDOW = 10.0  # seconds between the answer-key time and the logged entry


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("scan_dir")
    parser.add_argument("--truth", required=True)
    parser.add_argument("--min-votes", type=int, default=config.MIN_VOTES)
    args = parser.parse_args()

    scan_dir = Path(args.scan_dir)
    with open(scan_dir / "detections.csv", encoding="utf-8") as f:
        detections = list(csv.DictReader(f))
    with open(args.truth, encoding="utf-8") as f:
        truth = [{"time": float(r["time_s"]), "plate": r["true_plate"]} for r in csv.DictReader(f)]

    ocr = create_ocr()
    voter = PlateVoter(args.min_votes, config.VOTE_WINDOW, config.LOG_COOLDOWN)
    logged = []  # (time, plate)
    for i, d in enumerate(detections):
        reading = read_plate(ocr, cv2.imread(str(scan_dir / "crops" / d["crop"])))
        if reading:
            confirmed = voter.add(reading[0], reading[1], float(d["time_s"]))
            if confirmed:
                logged.append((float(d["time_s"]), confirmed[0]))
        if i % 100 == 0:
            print(f"read {i}/{len(detections)} crops", flush=True)

    report(truth, logged, [float(d["time_s"]) for d in detections])


def report(truth, logged, detection_times):
    unmatched = list(logged)
    correct, failures = 0, []
    for vehicle in truth:
        match = next((e for e in unmatched
                      if e[1] == vehicle["plate"] and abs(e[0] - vehicle["time"]) <= MATCH_WINDOW), None)
        if match:
            correct += 1
            unmatched.remove(match)
        else:
            nearby = sorted({p for t, p in unmatched if abs(t - vehicle["time"]) <= MATCH_WINDOW})
            failures.append((vehicle, nearby))

    detected = sum(any(abs(t - v["time"]) <= MATCH_WINDOW for t in detection_times) for v in truth)
    true_plates = {v["plate"] for v in truth}
    wrong = [e for e in unmatched if e[1] not in true_plates]
    repeats = [e for e in unmatched if e[1] in true_plates]

    n = len(truth)
    print(f"\nVehicles in answer key:     {n}")
    print(f"Plate detected:             {detected}/{n} ({detected / n:.0%})")
    print(f"Exact plate logged:         {correct}/{n} ({correct / n:.0%})")
    print(f"Wrong plates logged:        {len(wrong)}  {' '.join(p for _, p in wrong)}")
    print(f"Real plate seen again:      {len(repeats)}  (e.g. the same car leaving)")
    print("\nMissed vehicles:")
    for vehicle, nearby in failures:
        print(f"  {vehicle['time']:6.0f}s  {vehicle['plate']:<9} logged nearby: {', '.join(nearby) or '-'}")


if __name__ == "__main__":
    main()
