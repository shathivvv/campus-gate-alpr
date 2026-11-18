# UMPSA Campus Gate ALPR

[![tests](https://github.com/shathivvv/campus-gate-alpr/actions/workflows/tests.yml/badge.svg)](https://github.com/shathivvv/campus-gate-alpr/actions/workflows/tests.yml)

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

## Dashboard

**Live Monitor** (`/`)
- Welcome banner for each vehicle: *"Welcome, (owner name) / ABC1234 to Fakulti Komputeran"* in green
  for registered vehicles, *"Welcome Visitor"* in amber otherwise. It clears after 8 seconds.
- Today at a glance: total entries, registered, visitors, unique vehicles.
- Live video with plate boxes, the latest entries with owner name and category, and
  an entries-per-hour chart.

**Records** (`/records`)
- Search by plate or owner name, filter by status and date.
- Export the filtered entries to CSV.

The faculty name and banner duration are set in `config.py`
(`ALPR_FACULTY_NAME`, `ALPR_WELCOME_SECONDS`).

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

# Several videos played one after another (folders and * patterns work too);
# the dashboard has a "Next video" button to skip ahead
python app.py --source videos/clip1.mp4 "videos/VID_*.mp4" --loop

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

| Measure | v1 | v2 (current) |
|---|---|---|
| Plate detected on the vehicle | 44 / 44 (100%) | 44 / 44 (100%) |
| Exact plate logged correctly | 37 / 44 (84%) | **39 / 44 (89%)** |
| Wrong plates logged | 4 | **1** |

What changed in v2, and why:
- **Two-line plates** (0/2 -> 2/2): PaddleOCR's text detector found only one of the two lines,
  so near-square plate crops are now read as separate top and bottom halves.
- **Partial-read duplicates** (3 -> 0): after logging `WA1234B` the system also logged `A1234B`.
  A plate that is contained in, or one character away from, a plate logged in the last 15 s
  is now treated as the same car.
- **Tried and rejected:** confirming a plate after 2 matching reads instead of 3. It rescued
  one more vehicle but logged 4 more wrong plates, so it stays at 3.

Remaining misses: a narrow `1` dropped (one wrong plate), `Q` read as `O`, a car that passed
too quickly for 3 reads, and a bus whose plate is small and mounted high.

**Speed:**

| Step | Speed (laptop CPU) |
|---|---|
| OCR per plate, PaddleOCR default models | ~3.4 s |
| OCR per plate, mobile models (used) | ~0.12 s |
| Dashboard on 1080p video | ~6 frames/s shown, real-time playback |

To re-run the evaluation:

```bash
# 1. Detect plates once (slow on long 4K video); saves every plate crop
python scan_video.py videos/gate_entry.mp4 --sample-fps 10

# 2. OCR + voting + scoring against an answer key (a few minutes; re-run after any change)
python evaluate.py runs/scan_gate_entry --truth eval/ground_truth_gate_entry.csv
```

The answer key is a CSV with `time_s` and `true_plate` per vehicle. It holds real people's
plate numbers, so `eval/` is kept out of git.

## Known limitations / next steps

- Only the most confident plate in a frame is read, which fits a single-lane gate.
- Votes are grouped by text, not by vehicle. Object tracking (e.g. ByteTrack) would tie reads to one car.
- Only tested on daytime footage; night, rain and steep angles are untested.
