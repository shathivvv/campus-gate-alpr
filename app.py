"""Campus gate ALPR: detects plates on a webcam or recorded video and streams to a web dashboard.

    python app.py                                   # laptop webcam
    python app.py --source videos/gate.mp4 --loop   # recorded footage, restarts when it ends
"""
import argparse
import multiprocessing as mp
import threading
import time
from pathlib import Path

import cv2
from flask import Flask, render_template, Response, jsonify
from ultralytics import YOLO

import config
from db_handler import DatabaseHandler
from plate_reader import crop_plate

app = Flask(__name__)

# Latest annotated frame, shared by the capture thread and the video stream.
output_frame = None
frame_id = 0
frame_lock = threading.Lock()
source_label = "Live Camera Feed"

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
    if not cap.isOpened():
        raise SystemExit(f"Could not open video source: {source}")
    return cap

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

def camera_yolo_worker(cap, is_file, loop, input_queue, worker_busy, worker_ready):
    global output_frame, frame_id
    print("Loading YOLO on Main Process...")
    yolo_model = YOLO(config.MODEL_PATH)

    if is_file:
        # OCR takes a while to load; starting the video earlier would miss the first cars.
        print("[Camera] Waiting for OCR worker before playing the video...")
        worker_ready.wait()

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0  # 0 when the file doesn't say
    start, frame_no = time.perf_counter(), 0

    while True:
        ret, frame = cap.read()
        if not ret:
            if is_file and loop:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                start, frame_no = time.perf_counter(), 0
                continue
            if is_file:
                print("[Camera] End of video. Use --loop to replay it.")
                break
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

    cap.release()

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
    return render_template('index.html', source_label=source_label)

@app.route('/video_feed')
def video_feed():
    return Response(generate_video_stream(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/data')
def get_data():
    try:
        logs = DatabaseHandler().recent_entries(5)
    except Exception as e:
        return jsonify({"error": f"Database unavailable: {e}"}), 503

    formatted_logs = [
        {"plate": r["plate_number"], "status": r["auth_result"], "time": r["time_stamp"].strftime("%H:%M:%S")}
        for r in logs
    ]

    status_text = "Scanning for vehicles..."
    status_color = "gray"

    if logs:
        latest_log = logs[0]
        time_diff = (time.time() - latest_log["time_stamp"].timestamp())
        if time_diff < 8.0:
            if latest_log["auth_result"] == "Registered":
                status_text = f"Registered: {latest_log['plate_number']}"
                status_color = "green"
            else:
                status_text = f"Visitor: {latest_log['plate_number']}"
                status_color = "red"

    return jsonify({
        "status_text": status_text,
        "status_color": status_color,
        "logs": formatted_logs
    })

# --- MAIN EXECUTION ---
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Campus gate ALPR dashboard")
    parser.add_argument("--source", default="0", help="webcam index (e.g. 0) or path to a video file")
    parser.add_argument("--loop", action="store_true", help="replay the video when it ends")
    parser.add_argument("--host", default="127.0.0.1", help="use 0.0.0.0 to open the dashboard from other devices")
    parser.add_argument("--port", type=int, default=5000)
    args = parser.parse_args()

    source = int(args.source) if args.source.isdigit() else args.source
    is_file = isinstance(source, str)
    if is_file and not Path(source).is_file():
        parser.error(f"video file not found: {source}")
    if is_file:
        source_label = f"Recorded Video: {Path(source).name}"

    mp.set_start_method('spawn', force=True)

    ocr_queue = mp.Queue(maxsize=2)
    worker_busy = mp.Value('b', False)
    worker_ready = mp.Event()

    worker = mp.Process(target=ocr_db_worker, args=(ocr_queue, worker_busy, worker_ready), daemon=True)
    worker.start()

    cap = open_source(source)
    threading.Thread(target=camera_yolo_worker, args=(cap, is_file, args.loop, ocr_queue, worker_busy, worker_ready), daemon=True).start()

    print(f"\n[SYSTEM] Starting Web Server... Open http://127.0.0.1:{args.port} in your browser.\n")
    app.run(host=args.host, port=args.port, debug=False, threaded=True)
