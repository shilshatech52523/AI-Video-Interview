from fastapi import FastAPI, File, UploadFile, Form, Query, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.templating import Jinja2Templates

import os, shutil, subprocess, json, math, time, random, threading, asyncio, base64
from datetime import datetime
import datetime as dt

# Audio / STT / ML / CV
import speech_recognition as sr
import cv2
import mediapipe as mp
import numpy as np
import librosa
import face_recognition
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

# -------------------------------
# Static / Templates / Folders
# -------------------------------
os.makedirs("static", exist_ok=True)
os.makedirs("templates", exist_ok=True)
os.makedirs("videos", exist_ok=True)
os.makedirs("audio", exist_ok=True)
os.makedirs("pdf", exist_ok=True)
os.makedirs("screenshots", exist_ok=True)
os.makedirs("save_screenshot", exist_ok=True)
os.makedirs("logs", exist_ok=True)

app.mount("/static", StaticFiles(directory="static"), name="static")
app.mount("/audio", StaticFiles(directory="audio"), name="audio")
app.mount("/videos", StaticFiles(directory="videos"), name="videos")
app.mount("/pdfs", StaticFiles(directory="pdf"), name="pdfs")        # optional static serving
app.mount("/screens", StaticFiles(directory="save_screenshot"), name="screens")
templates = Jinja2Templates(directory="templates")

# -------------------------------
# Database (SQLite)
# -------------------------------
DATABASE_URL = "sqlite:///./interview.db"
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
Base = declarative_base()
SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)

class Transcript(Base):
    __tablename__ = "transcripts"
    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(String, nullable=False)
    question = Column(String, nullable=False)
    video_path = Column(String, nullable=False)
    audio_path = Column(String, nullable=False)
    transcript = Column(String, nullable=False)
    eye_contact_score = Column(Float, default=0.0)
    posture_score = Column(Float, default=0.0)
    confidence_score = Column(Float, default=0.0)
    answer_quality_score = Column(Float, default=0.0)
    sentiment_score = Column(Float, default=0.0)
    response_speed_score = Column(Float, default=0.0)
    final_engagement_score = Column(Float, default=0.0)
    smile_score = Column(Float, default=0.0)
    blink_rate_score = Column(Float, default=0.0)
    created_at = Column(DateTime, default=dt.datetime.utcnow)

Base.metadata.create_all(bind=engine)

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
# Utilities (Analysis pipeline)
# -------------------------------
def extract_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    return frames

def detect_face_landmarks(frames):
    landmarks_list = []
    for frame in frames:
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh_analysis.process(rgb_frame)
        if results.multi_face_landmarks:
            landmarks_list.append(results.multi_face_landmarks[0])
        else:
            landmarks_list.append(None)
    return landmarks_list

def euclidean_dist(a, b):
    return math.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2)

def calculate_eye_aspect_ratio(landmarks, img_width, img_height):
    left_eye_indices = [33, 133, 159, 145, 153, 154]
    right_eye_indices = [362, 263, 386, 374, 380, 381]
    def eye_ratio(eye_pts):
        A = euclidean_dist((eye_pts[1].x*img_width, eye_pts[1].y*img_height), (eye_pts[5].x*img_width, eye_pts[5].y*img_height))
        B = euclidean_dist((eye_pts[2].x*img_width, eye_pts[2].y*img_height), (eye_pts[4].x*img_width, eye_pts[4].y*img_height))
        C = euclidean_dist((eye_pts[0].x*img_width, eye_pts[0].y*img_height), (eye_pts[3].x*img_width, eye_pts[3].y*img_height))
        return (A + B) / (2.0 * C)
    left_eye = [landmarks.landmark[i] for i in left_eye_indices]
    right_eye = [landmarks.landmark[i] for i in right_eye_indices]
    left_ear = eye_ratio(left_eye)
    right_ear = eye_ratio(right_eye)
    return (left_ear + right_ear) / 2.0

def analyze_eye_contact(landmarks_list, frame_width, frame_height):
    ear_values = []
    for landmarks in landmarks_list:
        if landmarks:
            ear = calculate_eye_aspect_ratio(landmarks, frame_width, frame_height)
            ear_values.append(ear)
    if not ear_values:
        return 0.0
    avg_ear = sum(ear_values) / len(ear_values)
    score = min(max((avg_ear - 0.2) * 5, 0), 1)
    return score

def analyze_posture(landmarks_list):
    tilt_angles = []
    for landmarks in landmarks_list:
        if landmarks:
            nose = landmarks.landmark[1]
            chin = landmarks.landmark[152]
            dx = chin.x - nose.x
            dy = chin.y - nose.y
            angle = math.degrees(math.atan2(dy, dx))
            tilt_angles.append(angle)
    if not tilt_angles:
        return 0.0
    avg_tilt = sum(tilt_angles) / len(tilt_angles)
    score = max(0, 1 - abs(avg_tilt - 90) / 45)
    return score

def analyze_smile(landmarks_list):
    scores = []
    for landmarks in landmarks_list:
        if landmarks:
            upper_lip = landmarks.landmark[13]
            lower_lip = landmarks.landmark[14]
            left_corner = landmarks.landmark[61]
            right_corner = landmarks.landmark[291]
            vertical_dist = abs(upper_lip.y - lower_lip.y)
            horizontal_dist = abs(left_corner.x - right_corner.x)
            if horizontal_dist > 0:
                mar = vertical_dist / horizontal_dist
                smile_score = max(0, min(1, 0.3 - mar))
                scores.append(smile_score)
    if not scores:
        return 0.0
    return sum(scores) / len(scores)

def calculate_blink_rate(landmarks_list, fps=30):
    blink_frames = 0
    total_frames = len(landmarks_list)
    EAR_THRESHOLD = 0.2
    for landmarks in landmarks_list:
        if landmarks:
            img_w, img_h = 640, 480
            ear = calculate_eye_aspect_ratio(landmarks, img_w, img_h)
            if ear < EAR_THRESHOLD:
                blink_frames += 1
    blink_rate_per_sec = (blink_frames / total_frames) * fps if total_frames > 0 else 0
    normalized_blink = min(max((blink_rate_per_sec - 0.25) / 0.25, 0), 1)
    blink_score = 1 - abs(normalized_blink - 0.5) * 2
    return max(0, blink_score)

def analyze_confidence(audio_path):
    try:
        y, sr = librosa.load(audio_path, sr=None)
        pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
        pitch_values = pitches[magnitudes > np.median(magnitudes)]
        if len(pitch_values) == 0:
            return 0.0
        pitch_std = np.std(pitch_values)
        duration = librosa.get_duration(y=y, sr=sr)
        words_per_sec = 0  # placeholder (not adding extra logic as requested)
        pitch_score = min(pitch_std / 100, 1.0)
        speech_rate_score = min(words_per_sec / 4, 1.0)
        combined_confidence = 0.7 * pitch_score + 0.3 * speech_rate_score
        return combined_confidence
    except Exception as e:
        print("Confidence analysis error:", e)
        return 0.0

def answer_quality_score(candidate_answer, expected_answer):
    if not candidate_answer or not expected_answer:
        return 0.0
    emb1 = sbert_model.encode(candidate_answer, convert_to_tensor=True)
    emb2 = sbert_model.encode(expected_answer, convert_to_tensor=True)
    sim_score = util.pytorch_cos_sim(emb1, emb2).item()
    return sim_score

def normalized_sentiment_score(text):
    if not text:
        return 0.5
    raw_score = TextBlob(text).sentiment.polarity
    return (raw_score + 1) / 2

def intelligent_response_speed_score(response_duration, expected_answer_length, max_duration=60):
    base_time = max(10, min(expected_answer_length / 2, max_duration))
    score = max(0, 1 - response_duration / base_time)
    return min(score, 1)

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

    recognizer = sr.Recognizer()
    transcript_text = ""
    try:
        with sr.AudioFile(audio_path) as source:
            audio_data = recognizer.record(source)
            transcript_text = recognizer.recognize_google(audio_data, language="en-IN")
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

@app.get("/export_pdf")
async def export_pdf(session_id: str = Query(...)):
    db = SessionLocal()
    rows = db.query(Transcript).filter(Transcript.session_id == session_id).order_by(Transcript.id.asc()).all()
    db.close()

    if not rows:
        return {"status": "error", "message": "No data for this session"}

    file_path = os.path.join("pdf", f"interview_{session_id}.pdf")
    c = pdf_canvas.Canvas(file_path, pagesize=A4)
    width, height = A4
    y = height - 50
    c.setFont("Helvetica-Bold", 14)
    c.drawString(50, y, f"Interview Report - Session: {session_id}")
    y -= 30

    c.setFont("Helvetica", 11)
    for i, row in enumerate(rows, start=1):
        lines = [
            f"{i}. {row.question}",
            f"Answer: {row.transcript}",
            f"Eye Contact: {row.eye_contact_score:.2f} | Posture: {row.posture_score:.2f} | Confidence: {row.confidence_score:.2f}",
            f"Answer Quality: {row.answer_quality_score:.2f} | Sentiment: {row.sentiment_score:.2f} | Speed: {row.response_speed_score:.2f}",
            f"Smile: {row.smile_score:.2f} | Blink: {row.blink_rate_score:.2f} | Final Engagement: {row.final_engagement_score:.2f}",
        ]
        for ln in lines:
            c.drawString(50, y, ln[:110])
            y -= 16
            if y < 60:
                c.showPage(); y = height - 50; c.setFont("Helvetica", 11)
        y -= 8
        if y < 60:
            c.showPage(); y = height - 50; c.setFont("Helvetica", 11)

    c.save()
    return FileResponse(file_path, filename=f"interview_{session_id}.pdf", media_type="application/pdf")

# -------------------------------
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
        # Immediately send session_id to client
        await ws.send_text(json.dumps({"type": "session", "session_id": session_id}))

        while True:
            msg = await ws.receive_text()

            # Expecting JSON: {"type":"frame","data":"data:image/jpeg;base64,..."} or {"type":"next_question"} or {"type":"stop"}
            try:
                data = json.loads(msg)
            except Exception:
                # Backward compat: older clients may send "frame:<base64>" or "next_question"
                if msg.startswith("frame:"):
                    data = {"type": "frame", "data": msg.split(":", 1)[1]}
                elif msg == "next_question":
                    data = {"type": "next_question"}
                else:
                    data = {"type": "unknown"}
            
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

                # Multi-face warnings
                if len(faces) == 0:
                    await ws.send_text(json.dumps({"type": "warning", "message": "No face detected"}))
                elif len(faces) > 1:
                    extra_person_count[session_id] += 1
                    await ws.send_text(json.dumps({"type": "warning", "message": f"Multiple faces! Count={extra_person_count[session_id]}"}))
                    if extra_person_count[session_id] >= 3:
                        save_ws_screenshot(frame, session_id)
                        await ws.send_text(json.dumps({"type": "stop", "message": "Cheating detected! Session closed."}))
                        break
                else:
                    # reset if single face again
                    extra_person_count[session_id] = 0

                # Random screenshot
                if np.random.randint(0, 60) == 1:
                    save_ws_screenshot(frame, session_id)

                # Device detection (simple — runs occasionally for performance)
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
                            await ws.send_text(json.dumps({"type": "stop", "message": "Device detected! Session closed."}))
                            break
                    except Exception:
                        pass

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
                    await ws.send_text(json.dumps({"type": "stop", "message": "Interview completed"}))
                    break

            elif data.get("type") == "stop":
                await ws.send_text(json.dumps({"type": "stop", "message": "Session Completed"}))
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

@app.get("/camera_feed")
def camera_feed():
    return StreamingResponse(generate_camera_frames(), media_type="multipart/x-mixed-replace; boundary=frame")

@app.post("/stop_camera")
def stop_camera():
    stop_camera_flag.set()
    return {"message": "Camera stopped"}
