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

| Step | Speed (CPU) |
|---|---|
| OCR per plate, PaddleOCR default models | ~3.4 s |
| OCR per plate, mobile models (used) | ~0.12 s |

_Accuracy evaluation on labelled campus footage: to do._

## Known limitations / next steps

- Only the most confident plate in a frame is read, which fits a single-lane gate.
- Votes are grouped by text, not by vehicle. Object tracking (e.g. ByteTrack) would tie reads to one car.
- Accuracy hasn't been measured on night, rain or angled footage yet.
