import cv2
import threading
from fastapi.responses import StreamingResponse

# Global flag to stop/start camera
stop_camera_flag = threading.Event()

def generate_camera_frames():
    cap = cv2.VideoCapture(0)
    while not stop_camera_flag.is_set():
        ret, frame = cap.read()
        if not ret:
            break
        ret, buffer = cv2.imencode('.jpg', frame)
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
    cap.release()
    stop_camera_flag.clear()

def get_camera_feed():
    return StreamingResponse(generate_camera_frames(), media_type="multipart/x-mixed-replace; boundary=frame")

def stop_camera():
    stop_camera_flag.set()
    return {"message": "Camera stopped"}
