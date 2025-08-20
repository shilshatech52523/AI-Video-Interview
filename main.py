from fastapi import FastAPI, File, Form, Query, Request, WebSocket
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.templating import Jinja2Templates
from database import SessionLocal, engine
from models import Base
from screenshot_utils import save_ws_screenshot, log_event, get_face_bounding_rect
from questions_utils import questions, sessions_state, extra_person_count
from websocket_routes import websocket_questions
from export_pdf import router as export_pdf_router
from upload_routes import router as upload_router
from camera_utils import generate_camera_frames
from analysis_utils import (
    detect_face_landmarks,
    analyze_eye_contact,
    analyze_posture,
    analyze_smile,
    calculate_blink_rate,
    analyze_confidence
)
import os, threading

import mediapipe as mp
from ultralytics import YOLO

from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base
from sqlalchemy.engine.url import URL

from sentence_transformers import SentenceTransformer

from reportlab.lib.pagesizes import A4


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

os.makedirs("static", exist_ok=True)
os.makedirs("templates", exist_ok=True)
os.makedirs("videos", exist_ok=True)
os.makedirs("audio", exist_ok=True)
os.makedirs("pdf", exist_ok=True)
os.makedirs("save_screenshot", exist_ok=True)
os.makedirs("logs", exist_ok=True)

app.mount("/static", StaticFiles(directory="static"), name="static")
app.mount("/audio", StaticFiles(directory="audio"), name="audio")
app.mount("/videos", StaticFiles(directory="videos"), name="videos")
app.mount("/pdfs", StaticFiles(directory="pdf"), name="pdfs")    
app.mount("/screens", StaticFiles(directory="save_screenshot"), name="screens")
templates = Jinja2Templates(directory="templates")

Base.metadata.create_all(bind=engine)

mp_face_mesh = mp.solutions.face_mesh
face_mesh_analysis = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True)

face_mesh_ws = mp_face_mesh.FaceMesh(max_num_faces=5, refine_landmarks=True)

yolo_model = YOLO('yolov8n.pt') 

sbert_model = SentenceTransformer('all-MiniLM-L6-v2')

app.include_router(upload_router)

cheating_flag = {"cheating": False}
stop_camera_flag = threading.Event()
connected_clients: list[WebSocket] = []

app.include_router(export_pdf_router)
app.add_api_websocket_route("/ws", websocket_questions)


@app.get("/", response_class=HTMLResponse)
async def get_home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/camera_feed")
def camera_feed():
    return StreamingResponse(generate_camera_frames(), media_type="multipart/x-mixed-replace; boundary=frame")

@app.post("/stop_camera")
def stop_camera():
    stop_camera_flag.set()
    return {"message": "Camera stopped"}
