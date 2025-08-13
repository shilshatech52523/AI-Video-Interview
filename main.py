

from fastapi import FastAPI, File, UploadFile, Form, Query, Request, Body
from fastapi.responses import HTMLResponse, FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.templating import Jinja2Templates
import shutil
import os
from fastapi import WebSocket, WebSocketDisconnect
from datetime import datetime
import subprocess
import speech_recognition as sr
from sqlalchemy import create_engine, Column, Integer, String, DateTime, Float
from sqlalchemy.orm import declarative_base, sessionmaker
import datetime as dt
import cv2
import mediapipe as mp
import math
import numpy as np
import librosa
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
import random
import time
import base64
from pydantic import BaseModel
from datetime import datetime
import face_recognition
# At top of file (global flag)
import asyncio
cheating_flag = {"cheating": False}
from ultralytics import YOLO
yolo_model = YOLO('yolov8n.pt')
import threading

stop_camera_flag = threading.Event()
from fastapi import APIRouter
# NLP imports
from sentence_transformers import SentenceTransformer, util
from textblob import TextBlob

app = FastAPI()

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static and templates
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

os.makedirs("videos", exist_ok=True)
os.makedirs("audio", exist_ok=True)
UPLOAD_FOLDER = "save_screenshot"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)  # <-- Create this folder here  # <-- Create this folder here


# DB setup
DATABASE_URL = "postgresql+psycopg2://postgres:Sp%40495520@localhost:5432/mydb"
engine = create_engine(DATABASE_URL)
Base = declarative_base()
SessionLocal = sessionmaker(bind=engine)

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
    smile_score = Column(Float, default=0.0)  # New
    blink_rate_score = Column(Float, default=0.0)  # New
    created_at = Column(DateTime, default=dt.datetime.utcnow)

Base.metadata.create_all(bind=engine)

# Mediapipe init
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True)

# NLP model
sbert_model = SentenceTransformer('all-MiniLM-L6-v2')

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
        results = face_mesh.process(rgb_frame)
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
    # Use simple mouth aspect ratio or lip distance to detect smile intensity
    # Landmarks: upper lip 13, lower lip 14, mouth corners 61 & 291
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
                mar = vertical_dist / horizontal_dist  # Mouth aspect ratio
                smile_score = max(0, min(1, 0.3 - mar))  # smaller mar ~ smile (approx)
                scores.append(smile_score)
    if not scores:
        return 0.0
    return sum(scores) / len(scores)

def calculate_blink_rate(landmarks_list, fps=30):
    # Count frames where eyes are "closed" based on EAR threshold (<0.2)
    blink_frames = 0
    total_frames = len(landmarks_list)
    EAR_THRESHOLD = 0.2
    for landmarks in landmarks_list:
        if landmarks:
            # Use calculate_eye_aspect_ratio function
            img_w, img_h = 640, 480  # assuming approx frame size, or pass dynamically
            ear = calculate_eye_aspect_ratio(landmarks, img_w, img_h)
            if ear < EAR_THRESHOLD:
                blink_frames += 1
    blink_rate_per_sec = (blink_frames / total_frames) * fps if total_frames > 0 else 0
    # Normalize typical blink rate ~15-30 blinks/min → 0.25-0.5 blinks/sec
    normalized_blink = min(max((blink_rate_per_sec - 0.25) / 0.25, 0), 1)
    # Very low or very high blink rate both might indicate stress; middle is better
    blink_score = 1 - abs(normalized_blink - 0.5) * 2  # peak at 0.5 normalized blink rate
    return max(0, blink_score)

def analyze_confidence(audio_path):
    try:
        y, sr = librosa.load(audio_path, sr=None)
        pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
        pitch_values = pitches[magnitudes > np.median(magnitudes)]
        if len(pitch_values) == 0:
            return 0.0
        pitch_std = np.std(pitch_values)
        # Speech rate estimate: words per second (using silence detection or transcript length / duration)
        duration = librosa.get_duration(y=y, sr=sr)
        # Note: Transcript length can also be used if available
        words_per_sec = 0  # default fallback
        # Confidence heuristic: pitch std + speech rate normalized and combined
        pitch_score = min(pitch_std / 100, 1.0)
        speech_rate_score = min(words_per_sec / 4, 1.0)  # 4 words/sec typical speaking rate
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
        ], check=True)
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


class ScreenshotData(BaseModel):
    image_base64: str
    
@app.post("/save_screenshot")
async def save_screenshot(file: UploadFile = File(...)):
    # File ka naam banate hain
    filename = f"screenshot_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.jpg"
    file_path = os.path.join(UPLOAD_FOLDER, filename)

    # File save karna
    with open(file_path, "wb") as buffer:
        buffer.write(await file.read())

    return {
        "status": "success",
        "message": "Screenshot saved locally",
        "file_path": file_path
    }



@app.websocket("/cheating_status")
async def cheating_status_socket(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            await websocket.send_json({"cheating": cheating_flag["cheating"]})
            await asyncio.sleep(1)
    except WebSocketDisconnect:
        print("WebSocket client disconnected")

@app.get("/export_pdf")
async def export_pdf(session_id: str = Query(...)):
    db = SessionLocal()
    rows = db.query(Transcript).filter(Transcript.session_id == session_id).order_by(Transcript.id.asc()).all()
    db.close()

    if not rows:
        return {"status": "error", "message": "No data for this session"}

    os.makedirs("pdf", exist_ok=True)

    # PDF file ka path ab pdf/ folder mein
    file_path = os.path.join("pdf", f"interview_{session_id}.pdf")
    c = canvas.Canvas(file_path, pagesize=A4)
    width, height = A4
    y = height - 50
    c.setFont("Helvetica-Bold", 14)
    c.drawString(50, y, f"Interview Report - Session: {session_id}")
    y -= 30

    c.setFont("Helvetica", 12)
    for i, row in enumerate(rows, start=1):
        c.drawString(50, y, f"{i}. {row.question}")
        y -= 18
        c.drawString(70, y, f"Answer: {row.transcript}")
        y -= 18
        c.drawString(70, y, f"Eye Contact Score: {row.eye_contact_score:.2f}")
        y -= 18
        c.drawString(70, y, f"Posture Score: {row.posture_score:.2f}")
        y -= 18
        c.drawString(70, y, f"Confidence Score: {row.confidence_score:.2f}")
        y -= 18
        c.drawString(70, y, f"Answer Quality Score: {row.answer_quality_score:.2f}")
        y -= 18
        c.drawString(70, y, f"Sentiment Score: {row.sentiment_score:.2f}")
        y -= 18
        c.drawString(70, y, f"Response Speed Score: {row.response_speed_score:.2f}")
        y -= 18
        c.drawString(70, y, f"Smile Score: {row.smile_score:.2f}")
        y -= 18
        c.drawString(70, y, f"Blink Rate Score: {row.blink_rate_score:.2f}")
        y -= 18
        c.drawString(70, y, f"Final Engagement Score: {row.final_engagement_score:.2f}")
        y -= 30

        if y < 50:
            c.showPage()
            y = height - 50

    c.save()
    return FileResponse(file_path, filename=f"interview_{session_id}.pdf", media_type="application/pdf")

@app.get("/", response_class=HTMLResponse)
async def get_home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


# --------------------------
# New live camera feed endpoint with rectangle detection
# --------------------------
# Load reference face encoding for authentication
def load_reference_encoding():
    try:
        reference_image = face_recognition.load_image_file("static/reference.jpg")
        encodings = face_recognition.face_encodings(reference_image)
        if encodings:
            return encodings[0]
    except Exception:
        pass
    return None
# Helper: Get bounding rectangle from landmarks
def get_face_bounding_rect(landmarks, img_width, img_height):
    x_coords = [int(landmark.x * img_width) for landmark in landmarks.landmark]
    y_coords = [int(landmark.y * img_height) for landmark in landmarks.landmark]
    xmin, xmax = min(x_coords), max(x_coords)
    ymin, ymax = min(y_coords), max(y_coords)
    return xmin, ymin, xmax, ymax
# Camera feed generator with cheating detection and random screenshots
def generate_camera_frames():
    global cheating_flag, stop_camera_flag

    cap = cv2.VideoCapture(0)
    face_mesh_local = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=5, refine_landmarks=True)

    os.makedirs("screenshots", exist_ok=True)

    last_screenshot_time = 0
    next_screenshot_interval = random.randint(5, 10)  # seconds

    show_warning = False
    warning_start_time = 0
    WARNING_DURATION = 3  # seconds
    warning_text = ""

    ref_enc = load_reference_encoding()
    screenshot_taken_at_start = False

    while not stop_camera_flag.is_set():
        ret, frame = cap.read()
        if not ret:
            break

        current_time = time.time()
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh_local.process(img_rgb)

        if not screenshot_taken_at_start:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = f"screenshots/screenshot_{timestamp}_start.jpg"
            cv2.imwrite(filepath, frame)
            warning_text = "📸 Interview Started – Screenshot Taken!"
            show_warning = True
            warning_start_time = current_time
            screenshot_taken_at_start = True

        cheating_detected = False
        cheating_warning_text = ""

        num_faces = len(results.multi_face_landmarks) if results.multi_face_landmarks else 0

        if num_faces > 1:
            cheating_detected = True
            cheating_warning_text = "⚠ Cheating Detected: Multiple persons!"
        elif num_faces == 1:
            if ref_enc is None:
                cheating_warning_text = "ℹ Upload reference image for authentication"
            else:
                small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
                rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
                face_encs = face_recognition.face_encodings(rgb_small_frame)
                if face_encs:
                    match = face_recognition.compare_faces([ref_enc], face_encs[0], tolerance=0.5)
                    if not match[0]:
                        cheating_detected = True
                        cheating_warning_text = "⚠ Cheating Detected: Unauthorized candidate!"
                else:
                    cheating_warning_text = "⚠ Face encoding not found"
        else:
            cheating_warning_text = "⚠ No face detected!"

        device_detected = False
        device_name = ""
        if yolo_model:
            try:
                yolo_results = yolo_model(frame)
                for r in yolo_results:
                    boxes = r.boxes
                    for box in boxes:
                        cls_id = int(box.cls.cpu())
                        cls_name = yolo_model.names.get(cls_id, "").lower()
                        if cls_name in ["cell phone", "cellphone", "phone", "laptop", "tv", "tablet"]:
                            device_detected = True
                            device_name = cls_name
                            cheating_warning_text = f"⚠ Cheating Detected: Electronic device - {cls_name}"
                            break
                    if device_detected:
                        break
            except Exception:
                pass

        take_screenshot = False

        if cheating_detected or device_detected:
            take_screenshot = True
            warning_text = cheating_warning_text
            cheating_flag["cheating"] = True  # 🔴 set flag to True
        else:
            cheating_flag["cheating"] = False  # ✅ no cheating

        if not take_screenshot and (current_time - last_screenshot_time > next_screenshot_interval):
            take_screenshot = True
            warning_text = "📸 Screenshot Taken!"

        if take_screenshot:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = f"screenshots/screenshot_{timestamp}.jpg"
            cv2.imwrite(filepath, frame)
            last_screenshot_time = current_time
            next_screenshot_interval = random.randint(5, 10)
            show_warning = True
            warning_start_time = current_time

        if show_warning and (current_time - warning_start_time > WARNING_DURATION):
            show_warning = False
            warning_text = ""

        if results.multi_face_landmarks:
            h, w, _ = frame.shape
            for face_landmarks in results.multi_face_landmarks:
                xmin, ymin, xmax, ymax = get_face_bounding_rect(face_landmarks, w, h)
                cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)

        if warning_text:
            cv2.rectangle(frame, (0, 0), (frame.shape[1], 50), (0, 0, 0), -1)
            cv2.putText(frame, warning_text, (10, 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2, cv2.LINE_AA)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    cap.release()
    stop_camera_flag.clear()


@app.get("/camera_feed")
def camera_feed():
    return StreamingResponse(generate_camera_frames(),
                             media_type="multipart/x-mixed-replace; boundary=frame")


@app.post("/stop_camera")
def stop_camera():
    stop_camera_flag.set()
    return {"message": "Camera stopped"}