import cv2
from fastapi import FastAPI, File, UploadFile, Form, Query, Request, WebSocket, WebSocketDisconnect
import threading





cheating_flag = {"cheating": False}
stop_camera_flag = threading.Event()
connected_clients: list[WebSocket] = []


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