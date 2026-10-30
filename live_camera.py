import cv2
import time
import threading
import queue
import re
from ultralytics import YOLO
from paddleocr import PaddleOCR
from db_handler import DatabaseHandler

print("Loading Models & Database...")
yolo_model = YOLO("best (1).pt") 
ocr = PaddleOCR(lang='en')
db = DatabaseHandler()

# Queue to hold cropped plates for background processing
ocr_queue = queue.Queue()

# Global state variables for UI overlay and spam suppression
last_plate = ""
last_log_time = 0
latest_result_text = "Scanning for vehicles..."
status_color = (255, 255, 0)  # Cyan default

def ocr_db_worker():
    """Background worker thread for OCR and database logging."""
    global last_plate, last_log_time, latest_result_text, status_color
    
    while True:
        plate_crop = ocr_queue.get()
        if plate_crop is None:
            break
            
        ocr_generator = ocr.predict(plate_crop)
        ocr_results = list(ocr_generator)
        
        if len(ocr_results) > 0 and len(ocr_results[0].get('rec_texts', [])) > 0:
            raw_text = ocr_results[0]['rec_texts'][0]
            clean_text = raw_text.replace(" ", "").upper()
            
            # Malaysian standard plate format: 1-3 letters, 1-4 digits, optional trailing letter
            if re.match(r"^[A-Z]{1,3}\d{1,4}[A-Z]?$", clean_text):
                current_time = time.time()
                if clean_text != last_plate or (current_time - last_log_time) > 5.0:
                    print(f"\n[DETECTED VALID PLATE] Plate: {clean_text}")
                    
                    record = db.authorize_vehicle(clean_text)
                    if record:
                        print(f" -> WELCOME BACK: {record['full_name']} ({record['category']})")
                        db.log_entry(clean_text, "Registered", 0.99)
                        latest_result_text = f"Registered: {clean_text} - {record['full_name']}"
                        status_color = (0, 255, 0)  # Green
                    else:
                        print(" -> VISITOR LOGGED")
                        db.log_entry(clean_text, "Visitor", 0.99)
                        latest_result_text = f"Visitor: {clean_text}"
                        status_color = (0, 0, 255)  # Red
                    
                    last_plate = clean_text
                    last_log_time = current_time
        
        ocr_queue.task_done()

# Start background thread
worker_thread = threading.Thread(target=ocr_db_worker, daemon=True)
worker_thread.start()

cap = cv2.VideoCapture(0)
print("\nStarting live camera feed... Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame.")
        break

    # YOLO detection on main thread
    results = yolo_model.predict(frame, conf=0.4, verbose=False)
    
    if len(results[0].boxes) > 0:
        box = results[0].boxes[0].xyxy[0].cpu().numpy().astype(int)
        conf = float(results[0].boxes[0].conf[0].cpu().numpy())
        x1, y1, x2, y2 = box
        
        # Bounding box & confidence overlay
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
        cv2.putText(frame, f"Plate {conf:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        
        # Queue crop for OCR only if worker is idle
        if ocr_queue.empty():
            plate_crop = frame[y1:y2, x1:x2]
            ocr_queue.put(plate_crop)

    # Status header overlay
    cv2.putText(frame, f"Status: {latest_result_text}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)

    cv2.imshow("UMPSA ALPR - Live Feed", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()