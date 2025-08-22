from fastapi import FastAPI, File, UploadFile, Form, Query, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.templating import Jinja2Templates
from utils.pdf_exporter import router as pdf_exporter
from utils.database import SessionLocal, Transcript
from utils.analysis import extract_frames, analyze_eye_contact, euclidean_dist,calculate_eye_aspect_ratio, analyze_posture, analyze_smile,analyze_confidence,answer_quality_score, SentenceTransformer, normalized_sentiment_score, calculate_blink_rate, intelligent_response_speed_score, transcribe_vosk, detect_face_landmarks


# ////////////////////////////////////////////////////
from fastapi import APIRouter
import json
from utils.cheating_detection import start_proctoring

# ////////////////////////////////////////////////////


import os, shutil, subprocess, json, math, time, random, threading, asyncio, base64
from datetime import datetime
import datetime as dt


import os
from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base
from sqlalchemy.engine.url import URL

# Audio / STT / ML / CV
import speech_recognition as sr
import cv2
import mediapipe as mp
import numpy as np

from ultralytics import YOLO

# DB
from sqlalchemy import create_engine, Column, Integer, String, DateTime, Float
from sqlalchemy.orm import declarative_base, sessionmaker

# NLP
from sentence_transformers import SentenceTransformer, util
from textblob import TextBlob

# PDF
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas as pdf_canvas

# TTS
import pyttsx3

# -------------------------------
# App & CORS
# -------------------------------
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


router = APIRouter()

# -------------------------------
# Static / Templates / Folders
# -------------------------------
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

# -------------------------------
# Database (SQLite)
# -----------------------------
# -------------------------------
# Models & Globals
# -------------------------------
# Mediapipe FaceMesh (analysis)
mp_face_mesh = mp.solutions.face_mesh
face_mesh_analysis = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True)

# Mediapipe FaceMesh (WebSocket up to 5 faces)
face_mesh_ws = mp_face_mesh.FaceMesh(max_num_faces=5, refine_landmarks=True)

# YOLO for device detection
yolo_model = YOLO('yolov8n.pt')  # keep file in project root

# NLP
sbert_model = SentenceTransformer('all-MiniLM-L6-v2')

# Cheating flags / connections
cheating_flag = {"cheating": False}
stop_camera_flag = threading.Event()
connected_clients: list[WebSocket] = []

# Questions
questions = [
    "Q1: What is your name?",
    "Q2: Where do you live?",
    "Q3: What is your favorite programming language?",
    "Q4: Tell me about your hobbies.",
    "Q5: What is your goal for this year?"
]
extra_person_count: dict[str, int] = {}
sessions_state: dict[str, dict] = {}




# -------------------------------
# Upload & PDF
# -------------------------------
@app.post("/upload")
async def upload_video(
    file: UploadFile = File(...),
    question: str = Form(...),
    session_id: str = Form(...),
    expected_answer: str = Form(default=""),
    response_duration: float = Form(default=60.0)
):
    filename = datetime.now().strftime("%Y%m%d_%H%M%S") + ".webm"
    video_path = os.path.join("videos", filename)
    with open(video_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    audio_filename = filename.replace(".webm", ".wav")
    audio_path = os.path.join("audio", audio_filename)
    try:
        subprocess.run([
            "ffmpeg", "-y",
            "-i", video_path,
            "-vn",
            "-acodec", "pcm_s16le",
            "-ar", "16000",
            "-ac", "1",
            audio_path
        ], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except subprocess.CalledProcessError as e:
        return {"status": "error", "message": f"FFmpeg audio extract failed: {e}"}


    try:
        transcript_text = transcribe_vosk(audio_path)
    except Exception as e:
        transcript_text = f"Transcription failed: {e}"


    frames = extract_frames(video_path)
    landmarks_list = detect_face_landmarks(frames)

    frame_width = frames[0].shape[1] if frames else 640
    frame_height = frames[0].shape[0] if frames else 480

    eye_contact_score = analyze_eye_contact(landmarks_list, frame_width, frame_height)
    posture_score = analyze_posture(landmarks_list)
    smile_score = analyze_smile(landmarks_list)
    blink_rate_score = calculate_blink_rate(landmarks_list)
    confidence_score = analyze_confidence(audio_path)
    a_quality_score = answer_quality_score(transcript_text, expected_answer)
    sent_score = normalized_sentiment_score(transcript_text)
    resp_speed_score = intelligent_response_speed_score(response_duration, len(expected_answer.split()))

    final_engagement = (
        0.3 * a_quality_score +
        0.2 * sent_score +
        0.2 * resp_speed_score +
        0.1 * eye_contact_score +
        0.1 * posture_score +
        0.05 * smile_score +
        0.05 * blink_rate_score
    )
    final_engagement = min(max(final_engagement, 0), 1)

    db = SessionLocal()
    new_entry = Transcript(
        session_id=session_id,
        question=question,
        video_path=video_path,
        audio_path=audio_path,
        transcript=transcript_text,
        eye_contact_score=eye_contact_score,
        posture_score=posture_score,
        confidence_score=confidence_score,
        answer_quality_score=a_quality_score,
        sentiment_score=sent_score,
        response_speed_score=resp_speed_score,
        final_engagement_score=final_engagement,
        smile_score=smile_score,
        blink_rate_score=blink_rate_score
    )
    db.add(new_entry)
    db.commit()
    db.refresh(new_entry)
    db.close()

    return {
        "status": "success",
        "message": "Uploaded and evaluated",
        "id": new_entry.id,
        "eye_contact_score": round(eye_contact_score, 2),
        "posture_score": round(posture_score, 2),
        "confidence_score": round(confidence_score, 2),
        "answer_quality_score": round(a_quality_score, 2),
        "sentiment_score": round(sent_score, 2),
        "response_speed_score": round(resp_speed_score, 2),
        "smile_score": round(smile_score, 2),
        "blink_rate_score": round(blink_rate_score, 2),
        "final_engagement_score": round(final_engagement, 2),
        "transcript": transcript_text
    }


# Home
# -------------------------------
@app.get("/", response_class=HTMLResponse)
async def get_home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# -------------------------------
# Cheating utilities (screenshots)
# -------------------------------
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

# -------------------------------
# WebSocket: /ws (JSON protocol)
# -------------------------------
@app.websocket("/ws")
async def websocket_questions(ws: WebSocket):
    await ws.accept()
    session_id = str(int(time.time() * 1000))
    extra_person_count[session_id] = 0
    sessions_state[session_id] = {"question_index": 0, "active_question": ""}

    try:
        # 🔹 Send session_id immediately to client
        await ws.send_text(json.dumps({"type": "session", "session_id": session_id}))

        while True:
            msg = await ws.receive_text()

            # 🔹 Parse incoming data
            try:
                data = json.loads(msg)
            except Exception:
                if msg.startswith("frame:"):
                    data = {"type": "frame", "data": msg.split(":", 1)[1]}
                elif msg == "next_question":
                    data = {"type": "next_question"}
                else:
                    data = {"type": "unknown"}

            # ---------------- FRAME HANDLING ----------------
            if data.get("type") == "frame":
                encoded_data_url = data.get("data", "")
                if "," in encoded_data_url:
                    encoded = encoded_data_url.split(",", 1)[1]
                else:
                    encoded = encoded_data_url

                img_data = base64.b64decode(encoded)
                np_arr = np.frombuffer(img_data, np.uint8)
                frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

                results = face_mesh_ws.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                faces = results.multi_face_landmarks or []

                if len(faces) == 1:   # ✅ bounding box draw
                    h, w, _ = frame.shape
                    x1, y1, x2, y2 = get_face_bounding_rect(faces[0], w, h)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    extra_person_count[session_id] = 0  # reset if single face
                elif len(faces) > 1:
                    extra_person_count[session_id] += 1
                    await ws.send_text(json.dumps({
                        "type": "warning",
                        "message": f"Multiple faces! Count={extra_person_count[session_id]}"
                    }))
                    if extra_person_count[session_id] >= 3:
                        save_ws_screenshot(frame, session_id)
                        await ws.send_text(json.dumps({
                            "type": "stop",
                            "message": "Cheating detected! Session closed."
                        }))
                        break
                else:
                    await ws.send_text(json.dumps({
                        "type": "warning",
                        "message": "No face detected"
                    }))

                # ✅ Send frame back to frontend with box
                _, buffer = cv2.imencode('.jpg', frame)
                encoded_frame = base64.b64encode(buffer).decode("utf-8")
                await ws.send_text(json.dumps({
                    "type": "frame_boxed",
                    "data": f"data:image/jpeg;base64,{encoded_frame}"
                }))

                # 🔹 Random screenshot save
                if np.random.randint(0, 60) == 1:
                    save_ws_screenshot(frame, session_id)

                # 🔹 Device detection (YOLO)
                if np.random.randint(0, 10) == 1:
                    try:
                        yolo_results = yolo_model(frame, verbose=False)
                        device_hit = False
                        for r in yolo_results:
                            for box in r.boxes:
                                cls_id = int(box.cls.cpu())
                                conf = float(box.conf.cpu())
                                cls_name = yolo_model.names.get(cls_id, "").lower()
                                if conf > 0.6 and cls_name in ["cell phone", "cellphone", "phone", "laptop", "tv", "tablet"]:
                                    device_hit = True
                                    break
                            if device_hit:
                                break
                        if device_hit:
                            save_ws_screenshot(frame, session_id)
                            await ws.send_text(json.dumps({
                                "type": "stop",
                                "message": "Device detected! Session closed."
                            }))
                            break
                    except Exception:
                        pass

            # ---------------- NEXT QUESTION ----------------
            elif data.get("type") == "next_question":
                idx = sessions_state[session_id]["question_index"]
                if idx < len(questions):
                    question = questions[idx]
                    sessions_state[session_id]["question_index"] += 1
                    sessions_state[session_id]["active_question"] = question
                    audio_url = run_tts(question, session_id)
                    await ws.send_text(json.dumps({
                        "type": "question",
                        "text": question,
                        "audio": audio_url,
                        "session_id": session_id,
                        "index": idx
                    }))
                else:
                    await ws.send_text(json.dumps({
                        "type": "stop",
                        "message": "Interview completed"
                    }))
                    break

            # ---------------- STOP ----------------
            elif data.get("type") == "stop":
                await ws.send_text(json.dumps({
                    "type": "stop",
                    "message": "Session Completed"
                }))
                break

            else:
                # ignore unknown
                pass

    except WebSocketDisconnect:
        print("Client disconnected from /ws")
    except Exception as e:
        print("WebSocket /ws error:", e)
        try:
            await ws.send_text(json.dumps({"type": "error", "message": str(e)}))
        except:
            pass
# -------------------------------
# Optional camera MJPEG feed (kept, but unused by new UI)
# -------------------------------
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
