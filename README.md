# UMPSA Campus Gate ALPR

Automatic license plate recognition for a campus gate. It finds Malaysian plates in a live
camera or recorded video, reads them, checks them against a registry of staff and student
vehicles, and logs each entry to a live web dashboard.

Final-year project, Universiti Malaysia Pahang Al-Sultan Abdullah (UMPSA).

## How it works

```
video / webcam ──► YOLO plate detector ──► plate crop ──► PaddleOCR ──► format check ──► vote across frames ──► MySQL log
     (thread)          (custom-trained)                    (separate process)            (3 matching reads)        │
                                                                                                                    ▼
                                                        Flask dashboard ◄── MJPEG stream + /api/data ◄──────────────┘
```

- **Detection:** a YOLO model trained on Malaysian plates (`models/plate_yolo.pt`) runs on every frame.
- **OCR in its own process:** OCR is the slow step, so it runs in a separate process. That keeps it
  off the video thread, and Python's GIL can't make the two compete. Frames are only sent while the
  OCR worker is free, so a backlog can never build up.
- **Plate format check:** OCR text is cleaned and matched against the Malaysian plate format
  (e.g. `WA1234B`). Two-line plates are joined before matching.
- **Voting:** one OCR read can be wrong (blur, glare, `0` vs `O`). A plate is logged only after
  3 matching reads within 4 seconds, and the same plate isn't logged again for 15 seconds.
- **Real-time video playback:** recorded footage plays at its real speed. When detection falls
  behind, frames are skipped, just as a live gate camera would drop them.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Create the database (MySQL, e.g. via XAMPP):

```bash
mysql -u root < schema.sql
```

## Running

```bash
# Dashboard on recorded footage (put videos in videos/, which git ignores)
python app.py --source videos/gate_entry.mp4 --loop

# Dashboard on the laptop webcam
python app.py

# Single-image demo (Gradio)
python main.py
```

Then open http://127.0.0.1:5000.

Settings such as the detection threshold, vote count and database login can be overridden
with environment variables. They are all listed in [`config.py`](config.py).

## Tests

```bash
python -m pytest tests
```

## Results

**Accuracy:** 13.5 minutes of 4K daytime footage recorded at the Faculty of Computing,
44 vehicles approaching the camera, each checked against the plate read by eye.

| Measure | Result |
|---|---|
| Plate detected on the vehicle | 44 / 44 (100%) |
| Exact plate logged correctly | 37 / 44 (84%) |
| Wrong plate logged | 1 |
| Extra entries from partial reads | 3 |

Failures: two-line plates (0/2, only one line read), thin characters dropped or confused
(a narrow `1` dropped, `Q` read as `O`), and vehicles that gave fewer than 3 reads
(a fast car, a bus with a small high plate).

**Speed:**

| Step | Speed (laptop CPU) |
|---|---|
| OCR per plate, PaddleOCR default models | ~3.4 s |
| OCR per plate, mobile models (used) | ~0.12 s |
| Dashboard on 1080p video | ~6 frames/s shown, real-time playback |

To re-run the evaluation on a video:

```bash
python scan_video.py videos/gate_entry.mp4 --sample-fps 10
```

This writes `runs/scan_<video>/events.csv` (one row per vehicle) and a plate crop per vehicle.

## Known limitations / next steps

- Only the most confident plate in a frame is read, which fits a single-lane gate.
- Votes are grouped by text, not by vehicle. Object tracking (e.g. ByteTrack) would tie reads to one car.
- Two-line plates aren't read correctly yet (next fix).
- Only tested on daytime footage; night, rain and steep angles are untested.
