from fastapi import FastAPI, File, UploadFile, Form, Query, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.templating import Jinja2Templates
from fastapi import APIRouter
import os, json
from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base
from ultralytics import YOLO
from sqlalchemy import create_engine, Column, Integer, String, DateTime, Float
from sqlalchemy.orm import declarative_base, sessionmaker
from sentence_transformers import SentenceTransformer, util
from textblob import TextBlob
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas as pdf_canvas

from utils.screenshot import get_face_bounding_rect,log_event,device_frame_count,cheating_frame_count,save_ws_screenshot,required_consecutive_frames,reference_encoding
from utils.pdf_exporter import router as pdf_exporter
from utils.upload_router import router as upload_router
from utils.ws_socket import router as ws_socket_router
from utils.database import SessionLocal, Transcript
from utils.analysis import SentenceTransformer, transcribe_google
from utils.generate_camera import generate_camera_frames, stop_camera_flag
from utils.question import get_questions
from utils.cheating_detection import start_proctoring
from utils.run_tts import run_tts

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
router = APIRouter()

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
app.mount("/pdfs", StaticFiles(directory="pdf"), name="pdfs")     # optional static serving
app.mount("/screens", StaticFiles(directory="save_screenshot"), name="screens")
templates = Jinja2Templates(directory="templates")


app.include_router(upload_router)

@app.get("/", response_class=HTMLResponse)
async def get_home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

app.include_router(ws_socket_router)
app.include_router(pdf_exporter)

@router.get("/start_proctoring")
def run_proctoring():
    start_proctoring()
    return {"status": "started"}


@app.get("/camera_feed")
def camera_feed():
    return StreamingResponse(generate_camera_frames(), media_type="multipart/x-mixed-replace; boundary=frame")

@app.post("/stop_camera")
def stop_camera():
    stop_camera_flag.set()
    return {"message": "Camera stopped"}
