"""Campus gate ALPR: detects plates on a webcam or recorded video and streams to a web dashboard.

    python app.py                                   # laptop webcam
    python app.py --source videos/gate.mp4 --loop   # recorded footage, restarts when it ends
    python app.py --source videos/ --loop           # every video in the folder, one after another
    python app.py --source none                     # start empty; upload videos from the dashboard
"""
import argparse
import csv
import datetime
import glob
import io
import multiprocessing as mp
import threading
import time
from pathlib import Path

import cv2
import numpy as np
from flask import Flask, render_template, Response, jsonify, request
from werkzeug.utils import secure_filename
from ultralytics import YOLO

import config
from db_handler import DatabaseHandler
from plate_reader import crop_plate

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 4 * 1024**3  # largest video upload: 4 GB

# Latest annotated frame, shared by the capture thread and the video stream.
output_frame = None
frame_id = 0
frame_lock = threading.Lock()
source_label = "Live Camera Feed"

# Playlist shared by the capture thread and the upload route: webcam index (int) or video paths.
playlist = []
playlist_lock = threading.Lock()
jump_to = None  # playlist index to play next (set when a video is uploaded)
skip_video = threading.Event()  # set by "Next video" and by uploads to end the current video
VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv"}
UPLOAD_DIR = Path(__file__).resolve().parent / "videos" / "uploads"

# --- BACKGROUND PROCESS: PaddleOCR, voting & database ---
def ocr_db_worker(input_queue, worker_busy, worker_ready):
    from plate_reader import create_ocr, read_plate
    from voter import PlateVoter

    print("[Worker] Loading PaddleOCR and Database...")
    ocr = create_ocr()
    db = DatabaseHandler()
    voter = PlateVoter(config.MIN_VOTES, config.VOTE_WINDOW, config.LOG_COOLDOWN)

    print("[Worker] Ready and waiting for plates...")
    worker_ready.set()
    while True:
        plate_crop = input_queue.get()
        if plate_crop is None:
            break
        try:
            reading = read_plate(ocr, plate_crop)
            if reading is None:
                continue
            text, score = reading
            print(f"[Worker] Read {text} ({score:.2f})")

            confirmed = voter.add(text, score, time.time())
            if confirmed is None:
                continue
            plate, confidence = confirmed
            status = "Registered" if db.authorize_vehicle(plate) else "Visitor"
            db.log_entry(plate, status, confidence)
            print(f"\n[Worker] Logged {plate} as {status} (OCR confidence {confidence:.2f})\n")
        except Exception as e:
            print(f"[Worker Error] {e}")
        finally:
            worker_busy.value = False

# --- CAMERA THREAD: YOLO & Video Capture ---
def open_source(source):
    cap = cv2.VideoCapture(source)
    if isinstance(source, int):
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    return cap if cap.isOpened() else None

def show_message(text):
    """Puts a plain message frame on the video stream (e.g. while waiting for an upload)."""
    global output_frame, frame_id
    frame = np.full((540, 960, 3), (40, 32, 24), dtype=np.uint8)
    cv2.putText(frame, text, (40, 280), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
    with frame_lock:
        output_frame = frame
        frame_id += 1

def detect_and_queue(yolo_model, frame, input_queue, worker_busy):
    """Draws every detected plate and sends the most confident one to OCR if the worker is free."""
    boxes = yolo_model.predict(frame, conf=config.DET_CONF, imgsz=config.DET_IMGSZ, verbose=False)[0].boxes
    if len(boxes) == 0:
        return

    # Crop before drawing, otherwise the green rectangle ends up inside the OCR image.
    plate_crop = None
    if not worker_busy.value:
        best = int(boxes.conf.argmax())
        plate_crop = crop_plate(frame, boxes.xyxy[best].tolist())

    for (x1, y1, x2, y2), conf in zip(boxes.xyxy.int().tolist(), boxes.conf.tolist()):
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
        cv2.putText(frame, f"Plate {conf:.2f}", (x1, max(y1 - 10, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    if plate_crop is not None:
        worker_busy.value = True
        input_queue.put(plate_crop)

def camera_yolo_worker(loop, input_queue, worker_busy, worker_ready):
    """Plays each playlist entry in turn: the webcam, or videos (including uploaded ones)."""
    global source_label, jump_to
    print("Loading YOLO on Main Process...")
    yolo_model = YOLO(config.MODEL_PATH)

    # OCR takes a while to load; starting the video earlier would miss the first cars.
    show_message("Loading models...")
    print("[Camera] Waiting for OCR worker before playing...")
    worker_ready.wait()

    index = 0
    while True:
        with playlist_lock:
            if jump_to is not None:
                index, jump_to = jump_to, None
            elif index >= len(playlist) and loop and playlist:
                index = 0
            source = playlist[index] if index < len(playlist) else None
            total = len(playlist)

        if source is None:
            source_label = "Waiting for a video"
            show_message("Upload a video to start" if total == 0 else "Finished. Upload another video.")
            skip_video.wait(timeout=0.5)  # an upload sets this
            skip_video.clear()
            continue

        cap = open_source(source)
        if cap is None:
            print(f"[Camera] Could not open {source}")
            index += 1
            continue
        if isinstance(source, int):
            source_label = "Live Camera Feed"
        else:
            source_label = f"Recorded Video {index + 1} of {total}: {Path(source).name}"
        print(f"[Camera] Playing {source_label}")
        play(yolo_model, cap, not isinstance(source, int), input_queue, worker_busy)
        cap.release()
        index += 1

def play(yolo_model, cap, is_file, input_queue, worker_busy):
    """Shows one video (or the webcam) until it ends or the dashboard asks for the next video."""
    global output_frame, frame_id
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0  # 0 when the file doesn't say
    start, frame_no = time.perf_counter(), 0
    skip_video.clear()

    while not skip_video.is_set():
        ret, frame = cap.read()
        if not ret:
            if is_file:
                return
            time.sleep(0.05)  # webcam hiccup: retry without spinning the CPU
            continue
        frame_no += 1

        detect_and_queue(yolo_model, frame, input_queue, worker_busy)

        with frame_lock:
            output_frame = frame
            frame_id += 1

        if is_file:
            # Play the video at its real speed, like a live gate camera. If detection is
            # slower than the video, skip frames to catch up; if faster, wait.
            lag = time.perf_counter() - (start + frame_no / fps)
            if lag < 0:
                time.sleep(-lag)
            else:
                for _ in range(int(lag * fps)):
                    if not cap.grab():
                        break
                    frame_no += 1

# --- FLASK STREAMING LOGIC ---
def generate_video_stream():
    last_sent = -1
    while True:
        with frame_lock:
            frame, current_id = output_frame, frame_id
        # Frames are never modified after being published, so encoding outside the lock is safe.
        if frame is None or current_id == last_sent:
            time.sleep(0.01)
            continue
        last_sent = current_id

        ok, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
        if not ok:
            continue
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

# --- FLASK ROUTES ---
@app.route('/')
def index():
    return render_template('index.html', source_label=source_label, faculty=config.FACULTY_NAME)

@app.route('/video_feed')
def video_feed():
    return Response(generate_video_stream(), mimetype='multipart/x-mixed-replace; boundary=frame')

def entry_json(row):
    """One entry-log row in the shape the dashboard uses."""
    return {
        "plate": row["plate_number"],
        "status": row["auth_result"],
        "name": row["full_name"],
        "category": row["category"],
        "confidence": round(row["confidence"] or 0, 2),
        "date": row["time_stamp"].strftime("%Y-%m-%d"),
        "time": row["time_stamp"].strftime("%H:%M:%S"),
    }

@app.route('/api/data')
def get_data():
    """Everything the live monitor shows: welcome banner, recent entries, today's numbers."""
    today = datetime.date.today()
    try:
        db = DatabaseHandler()
        logs = db.recent_entries(10)
        summary = db.day_summary(today)
        hourly = db.hourly_counts(today)
    except Exception as e:
        return jsonify({"error": f"Database unavailable: {e}"}), 503

    # Show the welcome banner for the latest vehicle for a few seconds after it is logged.
    welcome = None
    if logs and logs[0]["seconds_ago"] is not None and logs[0]["seconds_ago"] < config.WELCOME_SECONDS:
        welcome = entry_json(logs[0])

    return jsonify({
        "source_label": source_label,
        "faculty": config.FACULTY_NAME,
        "welcome": welcome,
        "logs": [entry_json(r) for r in logs],
        "summary": summary,
        "hourly": hourly,
    })

def records_filters():
    """Reads the search filters shared by the records table and its CSV export."""
    day = request.args.get("date") or None
    if day:
        day = datetime.date.fromisoformat(day)
    return {"text": request.args.get("q", "").strip(), "status": request.args.get("status", ""), "day": day}

@app.route('/records')
def records():
    return render_template('records.html')

@app.route('/api/records')
def get_records():
    try:
        rows = DatabaseHandler().search_entries(**records_filters())
    except ValueError:
        return jsonify({"error": "Invalid date"}), 400
    return jsonify({"entries": [entry_json(r) for r in rows]})

@app.route('/records/export.csv')
def export_records():
    rows = DatabaseHandler().search_entries(**records_filters(), limit=100000)
    out = io.StringIO()
    writer = csv.writer(out)
    writer.writerow(["Date", "Time", "Plate", "Owner", "Category", "Status", "OCR confidence"])
    for r in map(entry_json, rows):
        writer.writerow([r["date"], r["time"], r["plate"], r["name"] or "", r["category"] or "",
                         r["status"], r["confidence"]])
    filename = f"alpr_entries_{datetime.date.today()}.csv"
    return Response(out.getvalue(), mimetype="text/csv",
                    headers={"Content-Disposition": f"attachment; filename={filename}"})

@app.route('/api/next', methods=['POST'])
def next_video():
    skip_video.set()
    return jsonify({"ok": True})

@app.errorhandler(413)
def upload_too_large(_error):
    return jsonify({"error": "File is too large (max 4 GB)"}), 413

@app.errorhandler(Exception)
def server_error(error):
    # Return JSON so the dashboard can show the reason instead of a generic failure.
    print(f"[Server Error] {error}")
    code = getattr(error, "code", 500)
    return jsonify({"error": str(error)}), code if isinstance(code, int) else 500

@app.route('/api/upload', methods=['POST'])
def upload_video():
    """Saves an uploaded video into videos/uploads/ and plays it straight away."""
    global jump_to
    file = request.files.get("video")
    if file is None or not file.filename:
        return jsonify({"error": "No file selected"}), 400
    if Path(file.filename).suffix.lower() not in VIDEO_EXTENSIONS:
        return jsonify({"error": f"Unsupported file type. Use {', '.join(sorted(VIDEO_EXTENSIONS))}"}), 400

    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    # secure_filename strips path tricks like "../" from the uploaded name.
    path = UPLOAD_DIR / f"{time.strftime('%Y%m%d_%H%M%S')}_{secure_filename(file.filename)}"
    file.save(path)

    cap = cv2.VideoCapture(str(path))
    readable = cap.isOpened() and cap.read()[0]
    cap.release()
    if not readable:
        path.unlink(missing_ok=True)
        return jsonify({"error": "That file couldn't be read as a video"}), 400

    with playlist_lock:
        playlist.append(str(path))
        jump_to = len(playlist) - 1
    skip_video.set()
    print(f"[Upload] Saved {path.name}, playing it now")
    return jsonify({"ok": True, "name": path.name})

# --- MAIN EXECUTION ---
def expand_sources(values):
    """Turns --source values into a webcam index or a list of video files (folders and * patterns allowed)."""
    if len(values) == 1 and values[0].isdigit():
        return [int(values[0])]
    paths = []
    for value in values:
        if Path(value).is_dir():
            paths += sorted(str(f) for f in Path(value).iterdir() if f.suffix.lower() in VIDEO_EXTENSIONS)
        elif any(c in value for c in "*?"):
            paths += sorted(glob.glob(value))
        else:
            paths.append(value)
    return paths

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Campus gate ALPR dashboard")
    parser.add_argument("--source", nargs="+", default=["0"],
                        help="webcam index (e.g. 0), one or more video files / folders / patterns played in order, "
                             "or 'none' to start empty and upload from the dashboard")
    parser.add_argument("--loop", action="store_true", help="replay the video(s) when they end")
    parser.add_argument("--host", default="127.0.0.1", help="use 0.0.0.0 to open the dashboard from other devices")
    parser.add_argument("--port", type=int, default=5000)
    args = parser.parse_args()

    sources = [] if args.source == ["none"] else expand_sources(args.source)
    if args.source != ["none"] and not sources:
        parser.error(f"no videos found for: {' '.join(args.source)}")
    missing = [p for p in sources if isinstance(p, str) and not Path(p).is_file()]
    if missing:
        parser.error(f"video file not found: {', '.join(missing)}")
    playlist.extend(sources)

    mp.set_start_method('spawn', force=True)

    ocr_queue = mp.Queue(maxsize=2)
    worker_busy = mp.Value('b', False)
    worker_ready = mp.Event()

    worker = mp.Process(target=ocr_db_worker, args=(ocr_queue, worker_busy, worker_ready), daemon=True)
    worker.start()

    threading.Thread(target=camera_yolo_worker, args=(args.loop, ocr_queue, worker_busy, worker_ready), daemon=True).start()

    print(f"\n[SYSTEM] Starting Web Server... Open http://127.0.0.1:{args.port} in your browser.\n")
    app.run(host=args.host, port=args.port, debug=False, threaded=True)
