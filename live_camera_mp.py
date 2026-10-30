import cv2
import time
import re
import multiprocessing as mp
from ultralytics import YOLO

# Note: PaddleOCR and DatabaseHandler must be initialized INSIDE the worker process.
def ocr_db_worker(input_queue):
    """This runs on a completely separate CPU core, bypassing Python's GIL."""
    # Import heavy libraries locally so they bind to the new process
    from paddleocr import PaddleOCR
    from db_handler import DatabaseHandler
    
    print("[Worker] Loading PaddleOCR and Database in background process...")
    ocr = PaddleOCR(lang='en')
    db = DatabaseHandler()
    
    last_plate = ""
    last_log_time = 0
    
    while True:
        plate_crop = input_queue.get()
        if plate_crop is None: # Exit signal
            break
            
        ocr_generator = ocr.predict(plate_crop)
        ocr_results = list(ocr_generator)
        
        if len(ocr_results) > 0 and len(ocr_results[0].get('rec_texts', [])) > 0:
            clean_text = ocr_results[0]['rec_texts'][0].replace(" ", "").upper()
            
            if re.match(r"^[A-Z]{1,3}\d{1,4}[A-Z]?$", clean_text):
                current_time = time.time()
                if clean_text != last_plate or (current_time - last_log_time) > 5.0:
                    print(f"\n[DETECTED VALID PLATE] Plate: {clean_text}")
                    record = db.authorize_vehicle(clean_text)
                    if record:
                        print(f" -> WELCOME BACK: {record['full_name']} ({record['category']})")
                        db.log_entry(clean_text, "Registered", 0.99)
                    else:
                        print(" -> VISITOR LOGGED")
                        db.log_entry(clean_text, "Visitor", 0.99)
                    
                    last_plate = clean_text
                    last_log_time = current_time

if __name__ == '__main__':
    # Enforce safe process spawning
    mp.set_start_method('spawn', force=True)
    
    # We restrict the queue to 2 items. If PaddleOCR is running slow, 
    # it simply drops older frames instead of stacking them up and crashing.
    ocr_queue = mp.Queue(maxsize=2)  
    
    # Start the PaddleOCR process
    worker = mp.Process(target=ocr_db_worker, args=(ocr_queue,), daemon=True)
    worker.start()

    print("Loading YOLO on Main Process...")
    yolo_model = YOLO("best (1).pt") 
    
    cap = cv2.VideoCapture(0)
    print("\nStarting live camera feed... Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # YOLO detection on main thread
        results = yolo_model.predict(frame, conf=0.4, verbose=False)
        
        if len(results[0].boxes) > 0:
            box = results[0].boxes[0].xyxy[0].cpu().numpy().astype(int)
            conf = float(results[0].boxes[0].conf[0].cpu().numpy())
            x1, y1, x2, y2 = box
            
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
            cv2.putText(frame, f"Plate {conf:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            
            # Only send to the background process if it is not currently full
            if not ocr_queue.full():
                plate_crop = frame[y1:y2, x1:x2].copy()
                ocr_queue.put(plate_crop)

        cv2.imshow("UMPSA ALPR - Live Feed", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Clean shutdown
    ocr_queue.put(None)
    worker.join()
    cap.release()
    cv2.destroyAllWindows()