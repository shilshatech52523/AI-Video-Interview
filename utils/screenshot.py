# -------------------------------
# Cheating utilities (screenshots)
# -------------------------------

from datetime import datetime
import time
import os,cv2
import pyttsx3




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

def run_tts(question_text, session_id):
    # Use WAV for maximum reliability with pyttsx3
    filename = f"tts_{session_id}_{int(time.time())}.wav"
    filepath = os.path.join("audio", filename)
    engine = pyttsx3.init()
    engine.save_to_file(question_text, filepath)
    engine.runAndWait()
    # Return URL the frontend can fetch
    return f"/audio/{filename}"
