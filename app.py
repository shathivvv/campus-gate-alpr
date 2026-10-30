import cv2
import time
import re
import threading
import multiprocessing as mp
from flask import Flask, render_template, Response, jsonify
from ultralytics import YOLO

app = Flask(__name__)

output_frame = None
frame_lock = threading.Lock()

# --- BACKGROUND PROCESS: PaddleOCR & Database ---
def ocr_db_worker(input_queue, worker_busy):
    from paddleocr import PaddleOCR
    from db_handler import DatabaseHandler
    
    print("[Worker] Loading PaddleOCR and Database...")
    # OPTIMIZATION 1: Cap CPU threads to 2 so it doesn't fight YOLO for power
    ocr = PaddleOCR(lang='en', use_angle_cls=False, cpu_threads=2)
    db = DatabaseHandler()
    
    last_plate = ""
    last_log_time = 0
    
    print("[Worker] Ready and waiting for plates...")
    while True:
        try:
            plate_crop = input_queue.get()
            if plate_crop is None:
                break
                
            ocr_generator = ocr.predict(plate_crop)
            ocr_results = list(ocr_generator)
            
            if len(ocr_results) > 0 and len(ocr_results[0].get('rec_texts', [])) > 0:
                clean_text = ocr_results[0]['rec_texts'][0].replace(" ", "").upper()
                
                if re.match(r"^[A-Z]{1,3}\d{1,4}[A-Z]?$", clean_text):
                    current_time = time.time()
                    
                    if clean_text != last_plate or (current_time - last_log_time) > 15.0:
                        print(f"\n[Worker] Processing Plate: {clean_text}")
                        record = db.authorize_vehicle(clean_text)
                        if record:
                            db.log_entry(clean_text, "Registered", 0.99)
                        else:
                            db.log_entry(clean_text, "Visitor", 0.99)
                        
                        last_plate = clean_text
                        last_log_time = current_time
                        
        except Exception as e:
            print(f"[Worker Error] {e}")
        finally:
            worker_busy.value = False

# --- CAMERA THREAD: YOLO & Video Capture ---
def camera_yolo_worker(input_queue, worker_busy):
    global output_frame
    print("Loading YOLO on Main Process...")
    yolo_model = YOLO("best (1).pt") 
    
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    while True:
        ret, frame = cap.read()
        if not ret: continue

        time.sleep(0.01)

        results = yolo_model.predict(frame, conf=0.4, imgsz=320, verbose=False)
        
        if len(results[0].boxes) > 0:
            box = results[0].boxes[0].xyxy[0].cpu().numpy().astype(int)
            x1, y1, x2, y2 = box
            
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
            
            if not worker_busy.value:
                worker_busy.value = True 
                
                # Extract the plate image
                plate_crop = frame[y1:y2, x1:x2].copy()
                
                # OPTIMIZATION 2: Force the image to be tiny (200x50).
                # This makes OCR inference almost instant without losing text legibility.
                try:
                    tiny_crop = cv2.resize(plate_crop, (200, 50))
                    input_queue.put(tiny_crop)
                except Exception as e:
                    # Catch resizing errors if YOLO draws a 0-pixel box on the edge of the frame
                    worker_busy.value = False

        with frame_lock:
            output_frame = frame.copy()

# --- FLASK STREAMING LOGIC ---
def generate_video_stream():
    global output_frame
    while True:
        with frame_lock:
            if output_frame is None:
                continue
            ret, buffer = cv2.imencode('.jpg', output_frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
            frame_bytes = buffer.tobytes()
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        time.sleep(0.02) 

# --- FLASK ROUTES ---
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_video_stream(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/data')
def get_data():
    from db_handler import DatabaseHandler
    temp_db = DatabaseHandler()
    conn = temp_db.connect()
    cursor = conn.cursor(dictionary=True)
    
    cursor.execute("SELECT plate_number, auth_result, time_stamp FROM ENTRY_LOG ORDER BY time_stamp DESC LIMIT 5")
    logs = cursor.fetchall()
    cursor.close()
    conn.close()
    
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
    mp.set_start_method('spawn', force=True)
    
    ocr_queue = mp.Queue(maxsize=2)
    worker_busy = mp.Value('b', False) 
    
    worker = mp.Process(target=ocr_db_worker, args=(ocr_queue, worker_busy), daemon=True)
    worker.start()
    
    threading.Thread(target=camera_yolo_worker, args=(ocr_queue, worker_busy), daemon=True).start()
    
    print("\n[SYSTEM] Starting Web Server... Open http://127.0.0.1:5000 in your browser.\n")
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)