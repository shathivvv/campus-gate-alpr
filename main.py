import cv2
import time
import gradio as gr
import pandas as pd
from datetime import datetime
from ultralytics import YOLO
import config
from db_handler import DatabaseHandler
from plate_reader import create_ocr, crop_plate, read_plate

# 1. Initialize Models & Database
print("Loading YOLO model...")
yolo_model = YOLO(config.MODEL_PATH)

print("Loading PaddleOCR...")
ocr = create_ocr()

print("Connecting to Database...")
db = DatabaseHandler()

# Keep track of the session log for the dashboard table
session_log = []

def process_pipeline(img):
    global session_log
    start_time = time.time()
    
    if img is None:
        return None, None, "<h2 style='color:red;'>Error: No Image</h2>", pd.DataFrame(session_log)

    # 2. Run YOLO Detection
    results = yolo_model.predict(img, conf=0.4, verbose=False)
    
    if len(results[0].boxes) == 0:
        return img, None, "<h2 style='color:red;'>No License Plate Detected</h2>", pd.DataFrame(session_log)
        
    # Extract bounding box
    box = results[0].boxes[0].xyxy[0].cpu().numpy().astype(int)
    conf = float(results[0].boxes[0].conf[0].cpu().numpy())
    x1, y1, x2, y2 = box
    
    # 3. Crop and Annotate
    plate_crop = crop_plate(img, box)
    annotated_img = img.copy()
    cv2.rectangle(annotated_img, (x1, y1), (x2, y2), (0, 255, 0), 3)
    cv2.putText(annotated_img, f"Plate {conf:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    
    # 4. Run PaddleOCR
    reading = read_plate(ocr, plate_crop) if plate_crop is not None else None
    clean_text = reading[0] if reading else "UNKNOWN"
    
    # 5. Database Authorization & Greeting Logic
    if clean_text == "UNKNOWN":
        greeting_html = f"<div style='background-color:#003366; padding:20px; border-radius:10px; color:white; text-align:center;'><h2>Welcome Visitor</h2><p>Plate Unreadable</p></div>"
        auth_status = "Visitor"
        owner_info = "Unknown"
    else:
        record = db.authorize_vehicle(clean_text)
        if record:
            owner_info = f"{record['full_name']} ({record['category']}, {record['faculty']})"
            greeting_html = f"<div style='background-color:#28a745; padding:20px; border-radius:10px; color:white; text-align:center;'><h2>WELCOME BACK</h2><p>Hi {record['full_name']}</p><p>Plate: {clean_text}</p></div>"
            auth_status = "Registered"
        else:
            owner_info = "Visitor"
            greeting_html = f"<div style='background-color:#003366; padding:20px; border-radius:10px; color:white; text-align:center;'><h2>WELCOME VISITOR</h2><p>Plate: {clean_text}</p></div>"
            auth_status = "Visitor"
            
    # 6. Log to Database
    db.log_entry(clean_text, auth_status, conf)
    
    # Update Session Log for UI
    timestamp = datetime.now().strftime("%H:%M:%S")
    session_log.insert(0, {"Time": timestamp, "Plate": clean_text, "Name/Role": owner_info, "Status": auth_status})
    
    return annotated_img, plate_crop, greeting_html, pd.DataFrame(session_log)

# 7. Build the Gradio Interface
with gr.Blocks() as demo:
    gr.Markdown("# 🚗 Campus Gate Welcome System — Edge-AI ALPR")
    
    with gr.Row():
        with gr.Column(scale=1):
            input_image = gr.Image(label="1 - Choose a vehicle")
            submit_btn = gr.Button("▶ RUN ANALYSIS", variant="primary")
            crop_output = gr.Image(label="AI Visual Verification")
            
        with gr.Column(scale=1):
            greeting_output = gr.HTML(label="2 - Welcome Status")
            annotated_output = gr.Image(label="3 - Detection")
            
    with gr.Row():
        log_output = gr.Dataframe(label="5 - Session Entry Log")

    submit_btn.click(
        fn=process_pipeline,
        inputs=input_image,
        outputs=[annotated_output, crop_output, greeting_output, log_output]
    )

if __name__ == "__main__":
    print("\nStarting Dashboard...")
    demo.launch(share=False)