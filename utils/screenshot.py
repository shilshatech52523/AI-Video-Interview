# -------------------------------
# Cheating utilities (screenshots)
# -------------------------------

from datetime import datetime
import os,cv2

def get_face_bounding_rect(landmarks, img_width, img_height):
    x_coords = [int(l.x * img_width) for l in landmarks.landmark]
    y_coords = [int(l.y * img_height) for l in landmarks.landmark]
    return min(x_coords), min(y_coords), max(x_coords), max(y_coords)

def log_event(message):
    with open(f"logs/events_{datetime.now().strftime('%Y%m%d')}.txt", "a") as f:
        f.write(f"[{datetime.now().strftime('%H:%M:%S')}] {message}\n")

reference_encoding = None
cheating_frame_count = 0
device_frame_count = 0
required_consecutive_frames = 3

def save_ws_screenshot(image_np, session_id):
    filename = f"{session_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
    filepath = os.path.join("save_screenshot", filename)
    cv2.imwrite(filepath, image_np)
    print(f"📸 Screenshot saved: {filepath}")

