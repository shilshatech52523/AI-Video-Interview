# screenshot_utils.py
import os, cv2
from datetime import datetime

# Screenshot save
def save_ws_screenshot(image_np, session_id, folder="save_screenshot"):
    os.makedirs(folder, exist_ok=True)
    filename = f"{session_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
    filepath = os.path.join(folder, filename)
    cv2.imwrite(filepath, image_np)
    print(f"📸 Screenshot saved: {filepath}")
    return filepath

# Event log
def log_event(message, log_dir="logs"):
    os.makedirs(log_dir, exist_ok=True)
    logfile = os.path.join(log_dir, f"events_{datetime.now().strftime('%Y%m%d')}.txt")
    with open(logfile, "a") as f:
        f.write(f"[{datetime.now().strftime('%H:%M:%S')}] {message}\n")

# Bounding rectangle utility
def get_face_bounding_rect(landmarks, img_width, img_height):
    x_coords = [int(l.x * img_width) for l in landmarks.landmark]
    y_coords = [int(l.y * img_height) for l in landmarks.landmark]
    return min(x_coords), min(y_coords), max(x_coords), max(y_coords)
