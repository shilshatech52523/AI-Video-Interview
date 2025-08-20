from fastapi import APIRouter, File, UploadFile, Form
from database import SessionLocal
from models import Transcript
import shutil, subprocess, os
from datetime import datetime
import speech_recognition as sr
import cv2, math, librosa, numpy as np
from sentence_transformers import SentenceTransformer, util
from textblob import TextBlob

# -------------------------------
# NLP Model
# -------------------------------
sbert_model = SentenceTransformer('all-MiniLM-L6-v2')

router = APIRouter()

# -------------------------------
# Utility Functions
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

def answer_quality_score(candidate_answer, expected_answer):
    if not candidate_answer or not expected_answer:
        return 0.0
    emb1 = sbert_model.encode(candidate_answer, convert_to_tensor=True)
    emb2 = sbert_model.encode(expected_answer, convert_to_tensor=True)
    return util.pytorch_cos_sim(emb1, emb2).item()

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
# Upload Route
# -------------------------------
@router.post("/upload")
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

    # Extract audio
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
        return {"status": "error", "message": f"FFmpeg failed: {e}"}

    # Speech-to-text
    recognizer = sr.Recognizer()
    transcript_text = ""
    try:
        with sr.AudioFile(audio_path) as source:
            audio_data = recognizer.record(source)
            transcript_text = recognizer.recognize_google(audio_data, language="en-IN")
    except Exception as e:
        transcript_text = f"Transcription failed: {e}"

    # Analysis
    frames = extract_frames(video_path)
    a_quality_score = answer_quality_score(transcript_text, expected_answer)
    sent_score = normalized_sentiment_score(transcript_text)
    resp_speed_score = intelligent_response_speed_score(response_duration, len(expected_answer.split()))

    final_engagement = (
        0.5 * a_quality_score +
        0.25 * sent_score +
        0.25 * resp_speed_score
    )
    final_engagement = min(max(final_engagement, 0), 1)

    # Save in DB
    db = SessionLocal()
    new_entry = Transcript(
        session_id=session_id,
        question=question,
        video_path=video_path,
        audio_path=audio_path,
        transcript=transcript_text,
        answer_quality_score=a_quality_score,
        sentiment_score=sent_score,
        response_speed_score=resp_speed_score,
        final_engagement_score=final_engagement
    )
    db.add(new_entry)
    db.commit()
    db.refresh(new_entry)
    db.close()

    return {
        "status": "success",
        "id": new_entry.id,
        "answer_quality_score": round(a_quality_score, 2),
        "sentiment_score": round(sent_score, 2),
        "response_speed_score": round(resp_speed_score, 2),
        "final_engagement_score": round(final_engagement, 2),
        "transcript": transcript_text
    }
